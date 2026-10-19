"""Transactional, repeatable import of synthetic CSVs into a local SQLite database."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from contextlib import contextmanager
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path
from typing import Iterator

from .schemas import TABLE_KEYS, TABLE_SCHEMAS, validate_headers, validate_manifest, validate_row

TABLES = TABLE_KEYS


class ImportRejected(ValueError):
    """A rejected atomic import, with a JSON-serializable, value-free error report."""

    def __init__(self, report: dict):
        self.report = report
        errors = report['errors']
        super().__init__(f"Import rejected ({len(errors)} errors): {errors[0]['message']}")


class DuplicateImportRejected(ImportRejected, sqlite3.IntegrityError):
    """Preserve the original duplicate-key IntegrityError interface."""


def _error(code, file, message, *, row=None, field=None):
    return {'code': code, 'file': file, 'row': row, 'field': field, 'message': message}


def _reject(report, errors):
    report.update(status='REJECTED', errors=errors)
    if report.get('reconciliation') and any(e['code'] == 'RECONCILIATION_MISMATCH' for e in errors):
        report['reconciliation']['matched'] = False
    exception = DuplicateImportRejected if any(e['code'] == 'DUPLICATE_KEY' for e in errors) else ImportRejected
    raise exception(report)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Nonfinite JSON value')


def _read_source(data_dir, relative):
    """Reject aliases before opening fixture inputs, including manifest and CSVs."""
    candidate = data_dir
    for part in Path(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError('Source symlinks are not accepted')
    if not candidate.resolve().is_relative_to(data_dir):
        raise ValueError('Source path escapes the data directory')
    if not candidate.is_file():
        raise OSError('Expected a regular source file')
    return candidate.read_bytes()


def _read_table(data_dir, table, errors):
    name = f'{table}.csv'
    try:
        raw = _read_source(data_dir, name)
        reader = csv.reader(io.StringIO(raw.decode('utf-8'), newline=''), strict=True)
        headers = next(reader, [])
        header_errors = validate_headers(table, headers)
        errors.extend(header_errors)
        if header_errors:
            return raw, [], []
        rows, positions = [], []
        while True:
            position = reader.line_num + 1
            values = next(reader, None)
            if values is None:
                break
            if len(values) != len(headers):
                errors.append(_error('ROW_WIDTH', name, 'CSV row does not match the header width.',
                                     row=position))
                continue
            row = dict(zip(headers, values))
            row_errors = validate_row(table, row, position)
            errors.extend(row_errors)
            if not row_errors:
                rows.append(row)
                positions.append(position)
        return raw, rows, positions
    except (OSError, UnicodeError):
        errors.append(_error('SOURCE_READ_ERROR', name, 'Cannot read required UTF-8 source file.'))
    except (RuntimeError, ValueError):
        errors.append(_error('SOURCE_PATH_ERROR', name, 'Source paths must be regular files without symlinks.'))
    except csv.Error:
        errors.append(_error('INVALID_CSV', name, 'Malformed CSV syntax.', row=reader.line_num))
    return b'', [], []


def _validate_references(tables, positions, manifest):
    errors = []
    for table, rows in tables.items():
        seen = set()
        invoice_keys = set()
        for row, position in zip(rows, positions[table]):
            key = tuple(row[field] for field in TABLES[table])
            if key in seen:
                errors.append(_error('DUPLICATE_KEY', f'{table}.csv', 'Duplicate source primary key.',
                                     row=position, field=','.join(TABLES[table])))
            seen.add(key)
            if row['tenant_id'] not in manifest['tenants']:
                errors.append(_error('UNKNOWN_TENANT', f'{table}.csv',
                                     'Tenant missing from completeness manifest.',
                                     row=position, field='tenant_id'))
            if table == 'invoice_lines':
                key = (row['tenant_id'], row['invoice_id'], row['line_number'])
                if key in invoice_keys:
                    errors.append(_error('DUPLICATE_KEY', 'invoice_lines.csv',
                        'Duplicate invoice business key.', row=position, field='invoice_id,line_number'))
                invoice_keys.add(key)
    suppliers = {(r['tenant_id'], r['supplier_id']) for r in tables['suppliers']}
    lines = {(r['tenant_id'], r['line_id']) for r in tables['invoice_lines']}
    docs = {(r['tenant_id'], r['doc_id']): r for r in tables['documents']}
    terms = {(r['tenant_id'], r['term_id']) for r in tables['reviewed_terms']}
    for table in ('invoice_lines', 'documents', 'reviewed_terms', 'purchase_orders'):
        for row, position in zip(tables[table], positions[table]):
            if (row['tenant_id'], row['supplier_id']) not in suppliers:
                errors.append(_error('UNKNOWN_SUPPLIER', f'{table}.csv', 'Unknown supplier reference.',
                                     row=position, field='supplier_id'))
    for table in ('service_entries', 'payments', 'adjustments'):
        for row, position in zip(tables[table], positions[table]):
            if (row['tenant_id'], row['line_id']) not in lines:
                errors.append(_error('ORPHAN_LINE', f'{table}.csv', 'Orphan line reference.',
                                     row=position, field='line_id'))
    for row, position in zip(tables['reviewed_terms'], positions['reviewed_terms']):
        if row['supersedes'] and (row['tenant_id'], row['supersedes']) not in terms:
            errors.append(_error('ORPHAN_SUPERSESSION', 'reviewed_terms.csv',
                                 'Unknown superseded term reference.', row=position, field='supersedes'))
        doc = docs.get((row['tenant_id'], row['doc_id']))
        if not doc or any(doc[field] != row[field]
                          for field in ('supplier_id', 'contract_id', 'role', 'location')):
            errors.append(_error('TERM_DOCUMENT_MISMATCH', 'reviewed_terms.csv',
                                 'Term/document identity mismatch.', row=position, field='doc_id'))
        elif 'body' in doc and row['source_quote'] not in doc['body']:
            errors.append(_error('UNSUPPORTED_QUOTE', 'reviewed_terms.csv',
                                 'Unsupported source quote.', row=position, field='source_quote'))
    return errors


def _metrics(tables, manifest):
    """Reconcile exact strings and Decimal totals; currencies/statuses never mix."""
    totals = {}
    with localcontext() as context:
        context.prec = 128
        for table, amount in (('invoice_lines', 'billed_amount'), ('payments', 'amount'),
                              ('adjustments', 'amount')):
            buckets = {}
            for row in tables[table]:
                group = buckets.setdefault(row['tenant_id'], {}).setdefault(row['currency'], {})
                status = row.get('status', 'ALL')
                group[status] = group.get(status, Decimal('0')) + Decimal(row[amount])
            totals[table] = {tenant: {currency: {status: {amount: format(value, 'f')}
                for status, value in statuses.items()} for currency, statuses in currencies.items()}
                for tenant, currencies in buckets.items()}
    digests = {}
    for table, rows in tables.items():
        fields = list(TABLE_SCHEMAS[table]) + (['body'] if table == 'documents' else [])
        records = sorted([[row[field] for field in fields] for row in rows])
        digests[table] = hashlib.sha256(json.dumps(records, ensure_ascii=False).encode()).hexdigest()
    return {'row_counts': {table: len(rows) for table, rows in tables.items()},
            'totals': totals, 'row_digests': digests, 'as_of': manifest['as_of'],
            'extracted_at': manifest.get('extracted_at')}


@contextmanager
def connect(path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=5000')
    try:
        yield conn
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    conn.execute('''CREATE TABLE IF NOT EXISTS review_events (
        event_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
        line_id TEXT NOT NULL, decision TEXT NOT NULL, reviewer TEXT NOT NULL,
        note TEXT NOT NULL, case_digest TEXT NOT NULL, recorded_at TEXT NOT NULL
    )''')


def load_snapshot(db_path: str | Path, data_dir: str | Path) -> dict:
    """Replace source snapshot atomically; preserve the separate append-only review journal.

    expected_results.json is NEVER read by this function or the calculation engine.
    Duplicate IDs, unsafe paths, changed document hashes and broken references fail the import.
    """
    report = {'schema_version': 1, 'status': 'REJECTED', 'synthetic_only': True,
              'dataset_id': None, 'as_of': None, 'extracted_at': None,
              'source_digest': None, 'errors': [], 'reconciliation': None}
    try:
        data_dir = Path(data_dir).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        _reject(report, [_error('SOURCE_PATH_ERROR', 'manifest.json', 'Invalid fixture directory path.')])
    try:
        manifest_raw = _read_source(data_dir, 'manifest.json')
        manifest = json.loads(manifest_raw.decode('utf-8'), object_pairs_hook=_json_object,
                              parse_constant=_invalid_constant)
    except (OSError, UnicodeError, RuntimeError, ValueError):
        _reject(report, [_error('INVALID_MANIFEST', 'manifest.json',
                              'Cannot read a valid UTF-8 JSON manifest with unique fields.')])
    # This validation, including the synthetic-only gate, precedes every CSV/document read.
    errors = validate_manifest(manifest)
    if errors:
        _reject(report, errors)
    report.update(dataset_id=manifest['dataset_id'], as_of=manifest['as_of'],
                  extracted_at=manifest.get('extracted_at'))
    tables, positions = {}, {}
    digest = hashlib.sha256(manifest_raw)
    for table in TABLES:
        raw, tables[table], positions[table] = _read_table(data_dir, table, errors)
        digest.update(table.encode())
        digest.update(raw)
    if errors:
        _reject(report, errors)
    for row, position in zip(tables['documents'], positions['documents']):
        try:
            file = (data_dir / row['path']).resolve()
            relative = file.relative_to(data_dir)
            if (len(relative.parts) != 2 or relative.parts[0] != 'contracts'
                    or relative.suffix != '.md'
                    or any(part.lower().startswith('.env') for part in relative.parts)):
                raise ValueError('Unsafe resolved path')
        except (OSError, RuntimeError, ValueError):
            errors.append(_error('UNSAFE_DOCUMENT_PATH', 'documents.csv',
                                 'Document path escapes the permitted contracts directory.',
                                 row=position, field='path'))
            continue
        try:
            content = _read_source(data_dir, row['path'])
            body = content.decode('utf-8')
        except (RuntimeError, ValueError):
            errors.append(_error('UNSAFE_DOCUMENT_PATH', 'documents.csv',
                                 'Document source paths must not contain symlinks.',
                                 row=position, field='path'))
            continue
        except (OSError, UnicodeError):
            errors.append(_error('DOCUMENT_READ_ERROR', 'documents.csv',
                                 'Cannot read the referenced UTF-8 document.',
                                 row=position, field='path'))
            continue
        if hashlib.sha256(content).hexdigest() != row['sha256']:
            errors.append(_error('DOCUMENT_HASH_MISMATCH', 'documents.csv',
                                 'Document integrity mismatch.', row=position, field='sha256'))
            continue
        row['body'] = body
        digest.update(content)
    errors.extend(_validate_references(tables, positions, manifest))
    if errors:
        _reject(report, errors)
    report.update(source_digest=digest.hexdigest(), lines=len(tables['invoice_lines']))
    source = _metrics(tables, manifest)
    try:
        with connect(db_path) as conn, conn:
            # Include schema creation, FTS, metadata and source replacement in one transaction.
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='lifecycle_migrations'").fetchone():
                _reject(report, [_error('LIFECYCLE_DATA_PROTECTED', 'database',
                    'This database has versioned lifecycle data. Fixture initialization cannot replace it; '
                    'use a separate new database for a new fixture snapshot.')])
            init_schema(conn)
            for table, keys in TABLES.items():
                fields = list(TABLE_SCHEMAS[table]) + (['body'] if table == 'documents' else [])
                definitions = ', '.join(f'"{x}" TEXT NOT NULL' for x in fields)
                key_sql = ', '.join(f'"{x}"' for x in keys)
                conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({definitions}, PRIMARY KEY ({key_sql}))')
                conn.execute(f'DELETE FROM "{table}"')
                names = ','.join(f'"{x}"' for x in fields)
                marks = ','.join('?' for _ in fields)
                conn.executemany(f'INSERT INTO "{table}" ({names}) VALUES ({marks})',
                                 [[row[x] for x in fields] for row in tables[table]])
            conn.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(
                tenant_id UNINDEXED, doc_id UNINDEXED, body
            )''')
            conn.execute('DELETE FROM document_fts')
            conn.executemany('INSERT INTO document_fts VALUES (?, ?, ?)', [
                (r['tenant_id'], r['doc_id'], r['body']) for r in tables['documents']])
            conn.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('manifest', json.dumps(manifest)))
            conn.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', ('source_digest', digest.hexdigest()))
            # Persist the expected successful report before the final read-only checks.
            # A failure rolls it back with the sources, including trigger-induced drift.
            report.update(status='IMPORTED', reconciliation={
                'matched': True, 'source': source, 'imported': source, 'manifest_matched': True,
                'document_fts': {'source_rows': len(tables['documents']),
                                 'imported_rows': len(tables['documents']), 'matched': True}})
            conn.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)',
                         ('import_report', json.dumps(report, sort_keys=True)))
            loaded = {table: [dict(row) for row in conn.execute(f'SELECT * FROM "{table}"')]
                      for table in TABLES}
            report['reconciliation'] = {'matched': False, 'source': source, 'imported': None}
            try:
                stored_manifest = json.loads(metadata(conn, 'manifest'))
                imported = _metrics(loaded, stored_manifest)
            except (ValueError, KeyError, TypeError, DecimalException):
                _reject(report, [_error('RECONCILIATION_MISMATCH', 'database',
                    'Stored source values failed reconciliation; prior snapshot was preserved.')])
            source_fts = sorted((r['tenant_id'], r['doc_id'], r['body']) for r in tables['documents'])
            imported_fts = sorted(tuple(row) for row in conn.execute('SELECT * FROM document_fts'))
            manifest_matched = json.dumps(stored_manifest, sort_keys=True) == json.dumps(manifest, sort_keys=True)
            matched = source == imported and source_fts == imported_fts and manifest_matched
            report['reconciliation'] = {'matched': matched, 'source': source, 'imported': imported,
                'manifest_matched': manifest_matched,
                'document_fts': {'source_rows': len(source_fts), 'imported_rows': len(imported_fts),
                                 'matched': source_fts == imported_fts}}
            if not matched:
                _reject(report, [_error('RECONCILIATION_MISMATCH', 'database',
                    'Source/import reconciliation failed; prior snapshot was preserved.')])
            try:
                stored_digest = metadata(conn, 'source_digest')
            except ValueError:
                stored_digest = None
            if stored_digest != report['source_digest']:
                _reject(report, [_error('RECONCILIATION_MISMATCH', 'metadata',
                    'Source digest reconciliation failed; prior snapshot was preserved.')])
            try:
                stored_report = json.loads(metadata(conn, 'import_report'))
            except (ValueError, TypeError):
                stored_report = None
            if json.dumps(stored_report, sort_keys=True) != json.dumps(report, sort_keys=True):
                _reject(report, [_error('RECONCILIATION_MISMATCH', 'metadata',
                    'Import report persistence failed; prior snapshot was preserved.')])
    except ImportRejected:
        raise
    except (OSError, sqlite3.Error):
        _reject(report, [_error('DATABASE_ERROR', 'database',
                               'Database import failed; prior snapshot was preserved.')])
    return report


def metadata(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
    if row is None:
        raise ValueError('Database not initialized. Run: recover init')
    return row['value']


def one_line(conn: sqlite3.Connection, tenant: str, line_id: str) -> dict:
    row = conn.execute('SELECT * FROM invoice_lines WHERE tenant_id=? AND line_id=?',
                       (tenant, line_id)).fetchone()
    if row is None:
        raise ValueError('Line not found in the selected tenant')
    return dict(row)


def scoped_documents(conn: sqlite3.Connection, line: dict) -> list[dict]:
    return [dict(r) for r in conn.execute('''SELECT * FROM documents
        WHERE tenant_id=? AND supplier_id=? AND contract_id=? AND role=? AND location=?
        AND kind='RATE_AGREEMENT' AND status='SIGNED' ORDER BY doc_id''',
        tuple(line[k] for k in ('tenant_id', 'supplier_id', 'contract_id', 'role', 'location')))]
