"""Immutable, synthetic migration history, separate from operational cases.

MIGRATED_APPROVED describes an invented old-system record. It is never a new
workflow approval, current candidate, approved draft, or realized recovery.
Archive reads use SQL only and never calculate cases or invoke providers.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .storage import connect, metadata

STATUS = 'MIGRATED_APPROVED'
LOCK_ERROR = 'MIGRATED_APPROVED_READ_ONLY: migrated synthetic history is locked'
_YEARS = (2024, 2025)
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}\Z')
_MONEY = re.compile(r'(?:0|[1-9][0-9]{0,14})\.[0-9]{2}\Z')
_FIELDS = {
    'suppliers': ('tenant_id', 'supplier_id', 'name'),
    'contracts': ('tenant_id', 'supplier_id', 'contract_id', 'title', 'origin_year'),
    'cases': ('tenant_id', 'line_id', 'supplier_id', 'contract_id', 'invoice_id',
              'invoice_date', 'origin_year', 'currency', 'historical_approved_amount',
              'source_record_id', 'description'),
}
_TABLES = ('legacy_archive_batches', 'legacy_suppliers', 'legacy_contracts', 'legacy_cases')


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _sha(value):
    return hashlib.sha256(_encode(value).encode('utf-8')).hexdigest()


def _exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def assert_not_archived(conn, tenant, line_id):
    """Fail before provider/checkpoint/journal effects; absent archive is a no-op."""
    if _exists(conn, 'legacy_cases') and conn.execute(
            'SELECT 1 FROM legacy_cases WHERE tenant_id=? AND line_id=?',
            (tenant, line_id)).fetchone():
        raise ValueError(LOCK_ERROR)


def _text(value, field, *, maximum=500):
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f'Invalid archive {field}')


def _identifier(value, field, *, legacy=False):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f'Invalid archive {field}')
    if legacy and not value.startswith('LEGACY-'):
        raise ValueError(f'Archive {field} must use the LEGACY- namespace')


def _validated(payload):
    fields = {'schema_version', 'synthetic', 'archive_id', 'source_system',
              'provenance', *_FIELDS}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError('Archive payload must have the exact versioned schema fields')
    if type(payload['schema_version']) is not int or payload['schema_version'] != 1:
        raise ValueError('Unsupported archive schema version')
    if payload['synthetic'] is not True:
        raise ValueError('SYNTHETIC_ONLY: archive accepts invented migration records only')
    _identifier(payload['archive_id'], 'archive_id', legacy=True)
    _text(payload['source_system'], 'source_system', maximum=200)
    _text(payload['provenance'], 'provenance', maximum=1000)
    result = {key: payload[key] for key in fields - set(_FIELDS)}
    for group, names in _FIELDS.items():
        rows = payload[group]
        if not isinstance(rows, list) or not rows or len(rows) > 100000:
            raise ValueError(f'Archive {group} must be a nonempty bounded list')
        checked = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(names):
                raise ValueError(f'Archive {group} row has missing or unexpected fields')
            for name in names:
                value = row[name]
                if name == 'origin_year':
                    if type(value) is not int or value not in _YEARS:
                        raise ValueError('Archive origin_year must be 2024 or 2025')
                elif name == 'historical_approved_amount':
                    if not isinstance(value, str) or not _MONEY.fullmatch(value):
                        raise ValueError('Archive amounts must be nonnegative, two-place Decimal strings')
                elif name == 'invoice_date':
                    try:
                        parsed = date.fromisoformat(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError('Invalid archive invoice_date') from exc
                    if parsed.isoformat() != value or parsed.year != row['origin_year']:
                        raise ValueError('Archive invoice_date must match origin_year')
                elif name == 'currency':
                    if not isinstance(value, str) or re.fullmatch(r'[A-Z]{3}', value) is None:
                        raise ValueError('Invalid archive currency')
                elif name.endswith('_id'):
                    _identifier(value, name, legacy=name in {'line_id', 'supplier_id', 'contract_id'})
                else:
                    _text(value, name, maximum=1000)
            checked.append(dict(row))
        key = {'suppliers': 'supplier_id', 'contracts': 'contract_id', 'cases': 'line_id'}[group]
        identities = [(row['tenant_id'], row[key]) for row in checked]
        if len(set(identities)) != len(identities):
            raise ValueError(f'Duplicate archive {group} identity')
        result[group] = sorted(checked, key=lambda row: (row['tenant_id'], row[key]))
    suppliers = {(row['tenant_id'], row['supplier_id']) for row in result['suppliers']}
    contracts = {(row['tenant_id'], row['supplier_id'], row['contract_id']): row['origin_year']
                 for row in result['contracts']}
    if any((row['tenant_id'], row['supplier_id']) not in suppliers for row in result['contracts']):
        raise ValueError('Archive contract references an unknown supplier')
    source_ids = set()
    for row in result['cases']:
        if contracts.get((row['tenant_id'], row['supplier_id'], row['contract_id'])) != row['origin_year']:
            raise ValueError('Archive case must match its supplier, contract and origin year')
        source_id = row['tenant_id'], row['source_record_id']
        if source_id in source_ids:
            raise ValueError('Duplicate archive source record identity')
        source_ids.add(source_id)
    return result


def _create_schema(conn):
    statements = [
        '''CREATE TABLE IF NOT EXISTS legacy_archive_batches (
            archive_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL CHECK(schema_version=1),
            synthetic INTEGER NOT NULL CHECK(synthetic=1), source_system TEXT NOT NULL,
            provenance TEXT NOT NULL, payload_sha256 TEXT NOT NULL, imported_at TEXT NOT NULL,
            receipt_json TEXT NOT NULL, UNIQUE(archive_id,source_system))''',
        '''CREATE TABLE IF NOT EXISTS legacy_suppliers (
            tenant_id TEXT NOT NULL,supplier_id TEXT NOT NULL,name TEXT NOT NULL,
            archive_id TEXT NOT NULL REFERENCES legacy_archive_batches(archive_id),
            PRIMARY KEY(tenant_id,supplier_id),UNIQUE(archive_id,tenant_id,supplier_id))''',
        '''CREATE TABLE IF NOT EXISTS legacy_contracts (
            tenant_id TEXT NOT NULL,supplier_id TEXT NOT NULL,contract_id TEXT NOT NULL,
            title TEXT NOT NULL,origin_year INTEGER NOT NULL CHECK(origin_year IN(2024,2025)),
            archive_id TEXT NOT NULL REFERENCES legacy_archive_batches(archive_id),
            PRIMARY KEY(tenant_id,contract_id),
            UNIQUE(archive_id,tenant_id,supplier_id,contract_id,origin_year),
            FOREIGN KEY(archive_id,tenant_id,supplier_id)
                REFERENCES legacy_suppliers(archive_id,tenant_id,supplier_id))''',
        '''CREATE TABLE IF NOT EXISTS legacy_cases (
            tenant_id TEXT NOT NULL,line_id TEXT NOT NULL,supplier_id TEXT NOT NULL,
            contract_id TEXT NOT NULL,invoice_id TEXT NOT NULL,invoice_date TEXT NOT NULL,
            origin_year INTEGER NOT NULL CHECK(origin_year IN(2024,2025)),currency TEXT NOT NULL,
            historical_approved_amount TEXT NOT NULL,source_record_id TEXT NOT NULL,
            description TEXT NOT NULL,archive_id TEXT NOT NULL,source_system TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status='MIGRATED_APPROVED'),locked INTEGER NOT NULL CHECK(locked=1),
            synthetic INTEGER NOT NULL CHECK(synthetic=1),case_sha256 TEXT NOT NULL,
            PRIMARY KEY(tenant_id,line_id),UNIQUE(tenant_id,source_system,source_record_id),
            FOREIGN KEY(archive_id,source_system)
                REFERENCES legacy_archive_batches(archive_id,source_system),
            FOREIGN KEY(archive_id,tenant_id,supplier_id,contract_id,origin_year)
                REFERENCES legacy_contracts(archive_id,tenant_id,supplier_id,contract_id,origin_year))''',
        '''CREATE INDEX IF NOT EXISTS legacy_cases_filter
            ON legacy_cases(tenant_id,origin_year,supplier_id,contract_id,line_id)''',
    ]
    for sql in statements:
        conn.execute(sql)
    for table in _TABLES:
        for action in ('UPDATE', 'DELETE'):
            conn.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_immutable_{action.lower()}
                BEFORE {action} ON {table} BEGIN
                    SELECT RAISE(ABORT,'{LOCK_ERROR}'); END''')
    for table, identity in (
            ('legacy_archive_batches', 'archive_id=NEW.archive_id'),
            ('legacy_suppliers', 'tenant_id=NEW.tenant_id AND supplier_id=NEW.supplier_id'),
            ('legacy_contracts', 'tenant_id=NEW.tenant_id AND contract_id=NEW.contract_id'),
            ('legacy_cases', '(tenant_id=NEW.tenant_id AND line_id=NEW.line_id) OR '
             '(tenant_id=NEW.tenant_id AND source_system=NEW.source_system '
             'AND source_record_id=NEW.source_record_id)')):
        # SQLite REPLACE can bypass DELETE triggers unless recursive triggers
        # are enabled. Reject conflicting inserts before REPLACE removes a row.
        conn.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_replace
            BEFORE INSERT ON {table} WHEN EXISTS (SELECT 1 FROM {table} WHERE {identity})
            BEGIN SELECT RAISE(ABORT,'{LOCK_ERROR}'); END''')
    # Import checks current identities; these symmetric triggers also prevent a
    # later canonical source import from making an archive record actionable.
    for table, key, archived in (('invoice_lines', 'line_id', 'legacy_cases'),
                                  ('suppliers', 'supplier_id', 'legacy_suppliers')):
        for action in ('INSERT', 'UPDATE'):
            conn.execute(f'''CREATE TRIGGER IF NOT EXISTS legacy_lock_{table}_{action.lower()}
                BEFORE {action} ON {table} WHEN EXISTS (
                    SELECT 1 FROM {archived} WHERE tenant_id=NEW.tenant_id AND {key}=NEW.{key})
                BEGIN SELECT RAISE(ABORT,'{LOCK_ERROR}'); END''')


def import_archive(db, payload):
    """Atomically append a validated invented archive; identical imports are no-ops."""
    checked = _validated(payload)
    if not Path(db).is_file():
        raise ValueError('Initialize an existing synthetic snapshot before importing its archive')
    with connect(db) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        manifest = json.loads(metadata(conn, 'manifest'))
        if manifest.get('synthetic') is not True:
            raise ValueError('SYNTHETIC_ONLY: archive requires a synthetic source snapshot')
        tenants = set(manifest['tenants'])
        if any(row['tenant_id'] not in tenants for group in _FIELDS for row in checked[group]):
            raise ValueError('Archive tenant is not in the supplied synthetic snapshot')
        payload_sha = _sha(checked)
        if _exists(conn, 'legacy_archive_batches'):
            prior = conn.execute('SELECT * FROM legacy_archive_batches WHERE archive_id=?',
                                 (checked['archive_id'],)).fetchone()
            if prior:
                if prior['payload_sha256'] != payload_sha:
                    raise ValueError('Archive ID already belongs to different immutable records')
                return {**json.loads(prior['receipt_json']), 'already_imported': True}
        for row in checked['suppliers']:
            if conn.execute('SELECT 1 FROM suppliers WHERE tenant_id=? AND supplier_id=?',
                            (row['tenant_id'], row['supplier_id'])).fetchone():
                raise ValueError('Archive supplier overlaps an operational supplier')
        for row in checked['contracts']:
            if conn.execute('SELECT 1 FROM documents WHERE tenant_id=? AND contract_id=? '
                'UNION ALL SELECT 1 FROM invoice_lines WHERE tenant_id=? AND contract_id=? LIMIT 1',
                (row['tenant_id'], row['contract_id']) * 2).fetchone():
                raise ValueError('Archive contract overlaps an operational contract')
        for row in checked['cases']:
            if conn.execute('SELECT 1 FROM invoice_lines WHERE tenant_id=? AND line_id=?',
                            (row['tenant_id'], row['line_id'])).fetchone():
                raise ValueError('Archive invoice overlaps an operational invoice')
        _create_schema(conn)
        receipt = {'schema_version': 1, 'status': 'IMPORTED', 'synthetic': True,
            'archive_id': checked['archive_id'], 'payload_sha256': payload_sha,
            'row_counts': {group: len(checked[group]) for group in _FIELDS},
            'counts_by_year': {str(year): sum(row['origin_year'] == year for row in checked['cases'])
                               for year in _YEARS},
            'already_imported': False}
        conn.execute('INSERT INTO legacy_archive_batches VALUES (?,?,?,?,?,?,?,?)',
            (checked['archive_id'], 1, 1, checked['source_system'], checked['provenance'],
             payload_sha, datetime.now(timezone.utc).isoformat(), _encode(receipt)))
        for group, table in (('suppliers', 'legacy_suppliers'), ('contracts', 'legacy_contracts')):
            names = (*_FIELDS[group], 'archive_id')
            marks = ','.join('?' for _ in names)
            conn.executemany(f'INSERT INTO {table} ({",".join(names)}) VALUES ({marks})',
                [tuple(row[name] for name in _FIELDS[group]) + (checked['archive_id'],)
                 for row in checked[group]])
        names = (*_FIELDS['cases'], 'archive_id', 'source_system', 'status', 'locked', 'synthetic', 'case_sha256')
        marks = ','.join('?' for _ in names)
        conn.executemany(f'INSERT INTO legacy_cases ({",".join(names)}) VALUES ({marks})',
            [tuple(row[name] for name in _FIELDS['cases']) +
             (checked['archive_id'], checked['source_system'], STATUS, 1, 1, _sha(row))
             for row in checked['cases']])
        return receipt


@contextmanager
def _read(db):
    if not Path(db).is_file():
        yield None
        return
    conn = sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn if _exists(conn, 'legacy_cases') else None
    finally:
        conn.close()


_JOIN = '''FROM legacy_cases c JOIN legacy_suppliers s
    ON c.tenant_id=s.tenant_id AND c.supplier_id=s.supplier_id
    JOIN legacy_contracts k ON c.tenant_id=k.tenant_id AND c.contract_id=k.contract_id
    JOIN legacy_archive_batches b ON c.archive_id=b.archive_id'''


def _filters(tenant, origin_year, supplier_id, contract_id, search):
    _identifier(tenant, 'tenant_id')
    clauses, args = ['c.tenant_id=?'], [tenant]
    if origin_year is not None:
        if type(origin_year) is not int or origin_year not in _YEARS:
            raise ValueError('Archive year must be 2024 or 2025')
        clauses.append('c.origin_year=?')
        args.append(origin_year)
    for field, value in (('supplier_id', supplier_id), ('contract_id', contract_id)):
        if value is not None:
            _identifier(value, field)
            clauses.append(f'c.{field}=?')
            args.append(value)
    if not isinstance(search, str) or len(search) > 200:
        raise ValueError('Archive search must be text of at most 200 characters')
    if search:
        literal = search.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        fields = ('c.line_id', 'c.invoice_id', 'c.description', 'c.supplier_id',
                  's.name', 'c.contract_id', 'k.title')
        clauses.append('(' + ' OR '.join(f"{field} LIKE ? ESCAPE '\\'" for field in fields) + ')')
        args.extend(['%' + literal + '%'] * len(fields))
    return ' WHERE ' + ' AND '.join(clauses), args


def list_archived_cases(db, tenant='DEMO', *, origin_year=None, supplier_id=None,
                        contract_id=None, search='', limit=None, offset=0):
    where, args = _filters(tenant, origin_year, supplier_id, contract_id, search)
    if (type(offset) is not int or offset < 0 or
            (limit is not None and (type(limit) is not int or not 1 <= limit <= 1000))):
        raise ValueError('Archive pages require a positive limit up to 1000 and nonnegative offset')
    sql = ('SELECT c.*,s.name AS supplier_name,k.title AS contract_title,b.provenance '
           + _JOIN + where + ' ORDER BY c.origin_year DESC,c.line_id')
    if limit is not None:
        sql += ' LIMIT ? OFFSET ?'
        args.extend((limit, offset))
    elif offset:
        sql += ' LIMIT -1 OFFSET ?'
        args.append(offset)
    with _read(db) as conn:
        return [{**dict(row), 'locked': True, 'synthetic': True}
                for row in conn.execute(sql, args)] if conn else []


def count_archived_cases(db, tenant='DEMO', *, origin_year=None, supplier_id=None,
                         contract_id=None, search=''):
    where, args = _filters(tenant, origin_year, supplier_id, contract_id, search)
    with _read(db) as conn:
        return conn.execute('SELECT count(*) ' + _JOIN + where, args).fetchone()[0] if conn else 0


def get_archived_case(db, tenant, line_id):
    _identifier(line_id, 'line_id')
    # Exact ID lookup is scoped, rather than a search substring match.
    with _read(db) as conn:
        if not conn:
            return None
        row = conn.execute('SELECT c.*,s.name AS supplier_name,k.title AS contract_title,b.provenance '
            + _JOIN + ' WHERE c.tenant_id=? AND c.line_id=?', (tenant, line_id)).fetchone()
        return {**dict(row), 'locked': True, 'synthetic': True} if row else None


def list_archived_suppliers(db, tenant='DEMO', *, origin_year=None):
    where, args = _filters(tenant, origin_year, None, None, '')
    with _read(db) as conn:
        return [dict(row) for row in conn.execute('SELECT c.supplier_id,s.name AS supplier_name,'
            'count(DISTINCT c.contract_id) AS contract_count,count(*) AS case_count '
            + _JOIN + where + ' GROUP BY c.supplier_id,s.name ORDER BY s.name,c.supplier_id', args)] if conn else []


def list_archived_contracts(db, tenant='DEMO', supplier_id=None, *, origin_year=None):
    where, args = _filters(tenant, origin_year, supplier_id, None, '')
    with _read(db) as conn:
        return [dict(row) for row in conn.execute('SELECT c.supplier_id,c.contract_id,'
            'k.title AS contract_title,c.origin_year,count(*) AS case_count '
            + _JOIN + where + ' GROUP BY c.supplier_id,c.contract_id,k.title,c.origin_year '
            'ORDER BY c.origin_year DESC,c.contract_id', args)] if conn else []


def archive_summary(db, tenant='DEMO'):
    result = {'case_count': 0, 'supplier_count': 0, 'contract_count': 0,
              'counts_by_year': {'2024': 0, '2025': 0}, 'totals_by_currency': {}}
    with _read(db) as conn:
        if not conn:
            return result
        counts = conn.execute('SELECT count(*),count(DISTINCT supplier_id),count(DISTINCT contract_id) '
            'FROM legacy_cases WHERE tenant_id=?', (tenant,)).fetchone()
        result.update(zip(('case_count', 'supplier_count', 'contract_count'), counts, strict=True))
        result['counts_by_year'].update({str(row[0]): row[1] for row in conn.execute(
            'SELECT origin_year,count(*) FROM legacy_cases WHERE tenant_id=? GROUP BY origin_year', (tenant,))})
        with localcontext() as context:
            context.prec = 80
            totals = {}
            for currency, value in conn.execute('SELECT currency,historical_approved_amount '
                                                'FROM legacy_cases WHERE tenant_id=?', (tenant,)):
                totals[currency] = totals.get(currency, Decimal('0')) + Decimal(value)
            result['totals_by_currency'] = {currency: format(value, '.2f') for currency, value in sorted(totals.items())}
    return result
