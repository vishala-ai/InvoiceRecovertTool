"""Explicit private publication of a validated synthetic business workspace.

No account store is accepted as input: Cloud accounts start empty. Publication
copies the selected portfolio and its matching checkpoints under SQLite writer
reservations, then commits all three roles through CloudStore in one transaction.
"""
from __future__ import annotations

from contextlib import closing, ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time

from .engine import audit
from .lifecycle_storage import active, validate_projections, validate_source_projections
from .local_auth import initialize
from .schemas import TABLE_SCHEMAS, validate_manifest, validate_row
from .storage import _metrics, _validate_references, metadata


class CloudSeedError(RuntimeError):
    """Safe operator-facing rejection without credentials or source record bodies."""


_FILES = {'portfolio': 'portfolio.sqlite3',
          'checkpoints': 'portfolio-ui-checkpoints.sqlite3', 'accounts': 'accounts.sqlite3'}
_BUSINESS_TABLES = {*TABLE_SCHEMAS, 'metadata', 'review_events', 'document_fts',
    'document_fts_data', 'document_fts_idx', 'document_fts_content', 'document_fts_docsize',
    'document_fts_config', 'sqlite_sequence', 'lifecycle_migrations', 'scope_state',
    'scope_events', 'source_stages', 'source_versions', 'operation_keys', 'term_proposals',
    'term_review_events', 'approved_term_versions', 'source_dispositions',
    'assessment_revisions', 'evidence_requests', 'evidence_request_events',
    'case_review_assessments', 'review_intents', 'legacy_archive_batches',
    'legacy_suppliers', 'legacy_contracts', 'legacy_cases', 'forecast_assumptions',
    'forecast_all_approved_assumptions', 'historical_recovery_receipts',
    'historical_recovery_items', 'vendor_contacts'}
_DATASETS = {'synthetic-rate-recovery-v1', 'synthetic-million-approved-demo-v1'}
_MAX_FILE_BYTES = 128 * 1024 * 1024
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z', re.ASCII)
_INVALID = 'Synthetic workspace validation failed; no publication was confirmed.'
_UNCERTAIN = ('Cloud publication could not be confirmed. Verify the remote workspace '
              'before retrying; do not assume the save failed.')


def _require(condition, message=_INVALID):
    if not condition:
        raise CloudSeedError(message)


def _path(value, *, required=True):
    path = Path(value).absolute()
    _require(not any(part in {'.', '..', 'local-auth', 'secrets.toml'} or part.lower().startswith('.env')
                     for part in path.parts), 'Choose explicit synthetic business database paths.')
    _require(path.name not in {'accounts.sqlite3', 'auth.sqlite3', 'account.sqlite3'},
             'Account databases are never accepted as seed inputs.')
    _require(not any(item.is_symlink() for item in (path, *path.parents)))
    if required:
        _require(path.is_file() and path.stat().st_nlink == 1,
                 'The synthetic portfolio and matching checkpoint files must already exist.')
    return path


def _read(path):
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    connection.execute('PRAGMA trusted_schema=OFF')
    return closing(connection)


def _tables(connection):
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _integrity(connection):
    return [row[0] for row in connection.execute('PRAGMA integrity_check')] == ['ok']


def _backup(source, target):
    _require(not target.exists() and not target.is_symlink())
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    deadline = time.monotonic() + 20

    def progress(status, remaining, total):
        _require(time.monotonic() <= deadline, 'Synthetic snapshot timed out; no publication was confirmed.')

    with _read(source) as reader, closing(sqlite3.connect(target)) as writer:
        # Reader is separate from the BEGIN IMMEDIATE reservation connection.
        # Backing up a connection holding its own write transaction would hang.
        reader.backup(writer, pages=256, progress=progress, sleep=0.01)
        writer.execute('PRAGMA journal_mode=DELETE')
        _require(writer.execute('PRAGMA page_count').fetchone()[0]
                 * writer.execute('PRAGMA page_size').fetchone()[0] <= _MAX_FILE_BYTES,
                 'Synthetic workspace exceeds the Cloud pilot storage limit.')
        _require(writer.execute('PRAGMA integrity_check').fetchall() == [('ok',)])


def _archives(connection, names, tenants):
    archive_tables = {'legacy_archive_batches', 'legacy_suppliers', 'legacy_contracts', 'legacy_cases'}
    if not names.intersection(archive_tables):
        return 0
    _require(archive_tables <= names)
    from .legacy_archive import _FIELDS, _sha, _validated, LOCK_ERROR, STATUS
    triggers = {row['name']: row['sql'] for row in connection.execute(
        "SELECT name,sql FROM sqlite_master WHERE type='trigger'")}
    for table in archive_tables:
        for suffix in ('immutable_update', 'immutable_delete', 'no_replace'):
            _require(LOCK_ERROR in triggers.get(table + '_' + suffix, ''))
    batches = connection.execute('SELECT * FROM legacy_archive_batches').fetchall()
    for batch in batches:
        _require(batch['synthetic'] == 1)
        payload = {'schema_version': batch['schema_version'], 'synthetic': True,
            'archive_id': batch['archive_id'], 'source_system': batch['source_system'],
            'provenance': batch['provenance']}
        for group, fields in _FIELDS.items():
            payload[group] = [dict(row) for row in connection.execute(
                f'SELECT {",".join(fields)} FROM legacy_{group} WHERE archive_id=?',
                (batch['archive_id'],))]
            _require(all(row['tenant_id'] in tenants for row in payload[group]))
        _require(_sha(_validated(payload)) == batch['payload_sha256'])
        for row in connection.execute('SELECT * FROM legacy_cases WHERE archive_id=?',
                                      (batch['archive_id'],)):
            _require(row['synthetic'] == 1 and row['locked'] == 1 and row['status'] == STATUS
                     and row['case_sha256'] == _sha({field: row[field] for field in _FIELDS['cases']}))
    _require(not connection.execute('''SELECT 1 FROM legacy_cases c JOIN invoice_lines i
        ON i.tenant_id=c.tenant_id AND i.line_id=c.line_id LIMIT 1''').fetchone())
    return connection.execute('SELECT COUNT(*) FROM legacy_cases').fetchone()[0]


def _profile(connection, current_count, archive_count):
    row = connection.execute("SELECT value FROM metadata WHERE key='demo_portfolio'").fetchone()
    if row is None:
        return None
    from .demo_portfolio import AS_OF, PROFILE_ID
    profile = json.loads(row[0])
    _require(profile.get('synthetic') is True and profile.get('profile_id') == PROFILE_ID
             and profile.get('as_of') == AS_OF and profile.get('reconciliation', {}).get('matched') is True
             and profile.get('new_realized_recovery') == '0.00'
             and profile.get('current_cohort') == {'cases': 1000, 'vendors': 35,
                 'approved': 500, 'awaiting': 250, 'evidence': 250}
             and current_count == 1030 and archive_count == 2000)
    # The retained reconciliation covers every appended transaction row; use the
    # generator's canonical serialization without reading any expected results.
    from .demo_portfolio import _encode
    reconciliation = profile['reconciliation']
    for table, count in reconciliation['new_row_counts'].items():
        _require(table in TABLE_SCHEMAS and type(count) is int and count >= 0)
        key = {'suppliers': 'supplier_id', 'documents': 'doc_id', 'reviewed_terms': 'term_id',
               'invoice_lines': 'line_id', 'service_entries': 'entry_id',
               'payments': 'payment_id', 'adjustments': 'adjustment_id', 'purchase_orders': 'po_id'}[table]
        # Appended records use DEMO-SCALE-* for scopes, SCALE-2026-* for invoices
        # and SVC-/PAY-/PO- wrappers for their transaction identities.
        rows = [dict(row) for row in connection.execute('SELECT * FROM ' + table)
                if 'SCALE-' in row[key]]
        _require(len(rows) == count and hashlib.sha256(_encode(sorted(rows, key=_encode)).encode()).hexdigest()
                 == reconciliation['new_row_digests'][table])
    return PROFILE_ID


def _validate_portfolio(path):
    with _read(path) as connection:
        _require(_integrity(connection))
        _require(not connection.execute('PRAGMA foreign_key_check').fetchall())
        names = _tables(connection)
        _require(set(TABLE_SCHEMAS) | {'metadata', 'review_events'} <= names)
        _require(names <= _BUSINESS_TABLES, 'Seed inputs may contain only supported synthetic business tables.')
        manifest = json.loads(metadata(connection, 'manifest'))
        _require(not validate_manifest(manifest) and manifest['dataset_id'] in _DATASETS
                 and 'DEMO' in manifest['tenants'], 'SYNTHETIC_ONLY: use a supported synthetic portfolio.')
        tables = {}
        for table, fields in TABLE_SCHEMAS.items():
            tables[table] = [dict(row) for row in connection.execute('SELECT * FROM ' + table)]
            for position, row in enumerate(tables[table], 1):
                errors = validate_row(table, {field: row[field] for field in fields}, position)
                if table == 'documents' and errors and 'source_versions' in names:
                    intake = connection.execute('''SELECT version_id,sha256 FROM source_versions
                        WHERE tenant_id=? AND doc_id=? AND origin='SYNTHETIC_INTAKE' ''',
                        (row['tenant_id'], row['doc_id'])).fetchone()
                    # Admitted evidence has its own canonical storage path; the
                    # original CSV schema's contracts/*.md rule does not apply.
                    if (intake is not None and row['sha256'] == intake['sha256']
                            and row['path'] == 'evidence/' + intake['version_id'].replace(':', '-') + '.txt'):
                        errors = [error for error in errors if not (
                            error['field'] == 'path' and error['code'] == 'INVALID_PATH')]
                _require(not errors)
                if table == 'documents':
                    _require(isinstance(row['body'], str)
                             and hashlib.sha256(row['body'].encode()).hexdigest() == row['sha256'])
        _require(not _validate_references(tables,
            {table: range(1, len(rows) + 1) for table, rows in tables.items()}, manifest))
        retained = json.loads(metadata(connection, 'import_report'))
        _require(retained.get('synthetic_only') is True and retained.get('status') == 'IMPORTED'
                 and retained.get('source_digest') == metadata(connection, 'source_digest')
                 and retained.get('reconciliation', {}).get('matched') is True)
        if not active(connection):
            _require(_metrics(tables, manifest) == retained['reconciliation']['source'])
        validate_source_projections(connection)
        for line in tables['invoice_lines']:
            validate_projections(connection, line)
        archive_count = _archives(connection, names, manifest['tenants'])
        current_count = sum(row['tenant_id'] == 'DEMO' for row in tables['invoice_lines'])
        profile = _profile(connection, current_count, archive_count)
        report = audit(connection, 'DEMO')
        _require(report['realized_recovery'] == '0.00')
        summary = {'synthetic': True, 'dataset_id': manifest['dataset_id'], 'as_of': manifest['as_of'],
            'profile_id': profile, 'current_cases': current_count, 'locked_historical_cases': archive_count,
            'candidate_total_usd': report['candidate_total'], 'new_realized_recovery': '0.00',
            'existing_review_events': connection.execute('SELECT COUNT(*) FROM review_events').fetchone()[0],
            'accounts_transferred': 0, 'model_calls': 0}
        identities = {(row['tenant_id'], row['line_id']): row for row in tables['invoice_lines']}
        _require(all((row['tenant_id'], row['line_id']) in identities for row in connection.execute(
            'SELECT tenant_id,line_id FROM review_events')))
        retained = {row[0] for row in connection.execute('SELECT case_digest FROM review_events')}
        if 'assessment_revisions' in names:
            retained.update(row[0] for row in connection.execute('SELECT case_digest FROM assessment_revisions'))
        return summary, identities, metadata(connection, 'source_digest'), retained


def _validate_checkpoints(path, identities, source_digest, retained):
    from .workflow import _existing_checkpoint_saver
    with _read(path) as connection:
        _require(_integrity(connection))
        _require(_tables(connection) == {'checkpoints', 'writes'},
                 'The matching LangGraph checkpoint database is required.')
    count = 0
    with _existing_checkpoint_saver(path) as saver:
        _require(saver is not None)
        for saved in saver.list(None):
            state = saved.checkpoint.get('channel_values', {})
            if not state.get('case'):
                # START checkpoints retain node inputs before case assessment.
                state = state.get('__start__', state)
            tenant, line_id = state.get('tenant'), state.get('line_id')
            if tenant is None and line_id is None:
                continue  # Graph-internal channel checkpoint, never an invoice authority.
            line = identities.get((tenant, line_id))
            _require(line is not None, 'Checkpoint records do not match this synthetic portfolio.')
            case = state.get('case')
            if case:
                digest = hashlib.sha256(json.dumps(
                    {key: value for key, value in case.items() if key != 'case_digest'},
                    sort_keys=True).encode()).hexdigest()
                _require(case.get('tenant_id') == tenant and case.get('line_id') == line_id
                    and case.get('supplier_id') == line['supplier_id']
                    and case.get('contract_id') == line['contract_id']
                    and case.get('case_digest') == digest
                    and (case.get('source_digest') == source_digest or digest in retained),
                    'Checkpoint records do not match this synthetic portfolio.')
                count += 1
    return count


def prepare_seed(source, directory):
    """Create a fresh private bundle locally; no network or source writes occur."""
    source = _path(source)
    checkpoints = _path(source.with_name(source.stem + '-ui-checkpoints.sqlite3'))
    _require(all(path.stat().st_size <= _MAX_FILE_BYTES for path in (source, checkpoints)),
             'Synthetic workspace exceeds the Cloud pilot storage limit.')
    directory = _path(directory, required=False)
    _require(not directory.exists(), 'Use a new private staging directory; existing files are never replaced.')
    directory.mkdir(mode=0o700, parents=True)
    files = {role: directory / name for role, name in _FILES.items()}
    try:
        with ExitStack() as stack:
            for path in (source, checkpoints):
                reservation = stack.enter_context(closing(sqlite3.connect(
                    path.as_uri() + '?mode=rw', uri=True, timeout=5, isolation_level=None)))
                reservation.execute('BEGIN IMMEDIATE')
            _backup(source, files['portfolio'])
            _backup(checkpoints, files['checkpoints'])
        summary, identities, source_digest, retained = _validate_portfolio(files['portfolio'])
        summary['checkpoint_case_snapshots'] = _validate_checkpoints(files['checkpoints'], identities, source_digest, retained)
        initialize(files['accounts'])
        return summary
    except BaseException as exc:
        shutil.rmtree(directory)
        if isinstance(exc, Exception) and not isinstance(exc, CloudSeedError):
            raise CloudSeedError(_INVALID) from None
        raise


def check_seed(source):
    """Validate a disposable snapshot without credentials, a provider or a store."""
    try:
        with tempfile.TemporaryDirectory(prefix='recovery-cloud-seed-check-') as temporary:
            return {**prepare_seed(source, Path(temporary).resolve() / 'bundle'), 'publication': 'not_attempted'}
    except CloudSeedError:
        raise
    except Exception:
        raise CloudSeedError(_INVALID) from None


def publish_seed(source, *, workspace_id, dsn, root_cert_pem, store_factory=None):
    """Publish once to an empty locked remote workspace; never overwrite it."""
    _require(isinstance(workspace_id, str) and _ID.fullmatch(workspace_id),
             'Choose a workspace identifier using letters, numbers, _ or -.')
    try:
        with tempfile.TemporaryDirectory(prefix='recovery-cloud-seed-') as temporary:
            base = Path(temporary).resolve()
            staged = base / 'validated'
            summary = prepare_seed(source, staged)
            if store_factory is None:
                from .cloud_store import CloudStore
                store_factory = CloudStore
            store = store_factory(dsn, root_cert_pem=root_cert_pem)
            with store.workspace(workspace_id, base / 'remote') as files:
                _require(set(files) == set(_FILES))
                _require(not any(path.exists() or path.is_symlink() for path in files.values()),
                         'Remote workspace already contains records; seed publication refuses replacement.')
                for role, filename in _FILES.items():
                    destination = Path(files[role])
                    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, 'wb') as output, (staged / filename).open('rb') as original:
                        shutil.copyfileobj(original, output)
            # Only a successful context exit confirms the PostgreSQL commit.
            return {**summary, 'workspace_id': workspace_id, 'publication': 'committed'}
    except CloudSeedError:
        raise
    except Exception:
        raise CloudSeedError(_UNCERTAIN) from None


def main(argv=None):
    """Operator CLI; connection secrets use a private prompt or exported environment."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Existing synthetic portfolio; matching checkpoints are derived by filename')
    parser.add_argument('--check', action='store_true', help='Validate local copies without network access')
    parser.add_argument('--workspace', default='invoice-recovery-demo', help='New empty remote workspace ID')
    parser.add_argument('--ca-cert', type=Path, help='Public PostgreSQL root certificate PEM file')
    parser.add_argument('--prompt-database-url', action='store_true',
                        help='Privately enter the Supabase connection URI without shell history')
    args = parser.parse_args(argv)
    try:
        if args.check:
            result = check_seed(args.source)
        else:
            if args.prompt_database_url:
                from getpass import getpass
                dsn = getpass('Supabase PostgreSQL connection URL (hidden): ')
            else:
                dsn = os.environ.get('RECOVERY_CLOUD_DATABASE_URL')
            _require(bool(dsn),
                     'Set RECOVERY_CLOUD_DATABASE_URL privately in the operator environment.')
            _require(args.ca_cert is not None, 'Supply the public PostgreSQL root certificate with --ca-cert.')
            certificate_path = _path(args.ca_cert)
            _require(certificate_path.suffix.lower() in {'.pem', '.crt', '.cer'},
                     'Choose a public certificate file with a .pem, .crt or .cer extension.')
            _require(certificate_path.stat().st_size <= 65536,
                     'Supply a bounded public PostgreSQL root certificate.')
            with certificate_path.open('rb') as public_file:
                _require(public_file.readline(128).strip() == b'-----BEGIN CERTIFICATE-----',
                         'Supply only a public root certificate.')
            certificate = certificate_path.read_text(encoding='ascii')
            _require(certificate.lstrip().startswith('-----BEGIN CERTIFICATE-----')
                     and 'PRIVATE KEY' not in certificate, 'Supply only a public root certificate.')
            result = publish_seed(args.source, workspace_id=args.workspace,
                dsn=dsn, root_cert_pem=certificate)
        print(json.dumps(result, indent=2))
        return 0
    except CloudSeedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print(_INVALID, file=sys.stderr)
        return 2
