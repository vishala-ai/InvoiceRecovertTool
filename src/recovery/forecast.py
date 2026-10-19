"""Human assumptions for a synthetic cash forecast, separate from recovery facts.

Approved drafts are the forecast basis. They are neither collected cash nor a
calibrated collection probability. Reading never creates tables or assumptions;
only an explicit save appends a reviewed assumption to its own immutable table.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, localcontext
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid

from .storage import connect


POLICY_VERSION = 'synthetic-approved-cash-forecast-v1'
POLICY_ALL_APPROVED = 'synthetic-all-approved-cash-forecast-v1'
LABEL = 'Estimated cash recovery under demo assumptions'
NOTICE = ('Explicit, uncalibrated demo assumption. This is not measured savings, '
          'a historical collection rate, a supplier commitment or realized recovery.')
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_MONEY = re.compile(r'[0-9]{1,24}(?:\.[0-9]{1,2})?\Z')
_PERCENT = re.compile(r'[0-9]{1,3}(?:\.[0-9]{1,6})?\Z')
_SCOPES = {'current': 'forecast_assumptions', 'all_approved': 'forecast_all_approved_assumptions'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _digest(value):
    return hashlib.sha256(_encode(value).encode('utf-8')).hexdigest()


def _text(value, field, maximum=4000):
    _require(isinstance(value, str) and bool(value.strip()) and len(value) <= maximum,
             f'Enter a nonempty {field} (at most {maximum} characters).')
    return value.strip()


def _amount(value):
    _require(isinstance(value, str) and bool(_MONEY.fullmatch(value)),
             'Forecast basis amounts must be nonnegative Decimal strings with at most two decimal places.')
    return Decimal(value)


def _percentage(value):
    _require(isinstance(value, str) and bool(_PERCENT.fullmatch(value)),
             'Recovery percentage must be a Decimal string from 0 to 100, with at most six decimal places.')
    number = Decimal(value)
    _require(Decimal('0') <= number <= Decimal('100'), 'Recovery percentage must be from 0 to 100.')
    # Decimal.normalize() applies the ambient context and could round a reviewed
    # percentage before multiplication. Formatting/trimming preserves its value.
    canonical = format(number, 'f')
    return canonical.rstrip('0').rstrip('.') if '.' in canonical else canonical


def _date(value, field):
    _require(isinstance(value, str) and bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', value)),
             f'{field} must be a calendar date in YYYY-MM-DD format.')
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'{field} must be a valid calendar date.') from exc


def basis_from_summary(summary, *, tenant='DEMO'):
    """Take only current approved USD cases from an existing dashboard summary.

    This pure helper cannot establish that a supplied summary is fresh. Save
    independently reloads the canonical worklist, under database/checkpoint locks.
    """
    tenant = _text(tenant, 'tenant', 200)
    _require(isinstance(summary, dict) and isinstance(summary.get('cases'), list),
             'A current case summary is required for the forecast basis.')
    cases, seen = [], set()
    for case in summary['cases']:
        _require(isinstance(case, dict), 'Malformed forecast case summary.')
        if (case.get('tenant_id', case.get('tenant')) != tenant
                or case.get('currency') != 'USD' or case.get('status') != 'CANDIDATE'
                or case.get('bucket') != 'approved'
                or case.get('current_decision') != 'APPROVED_FOR_DRAFT'
                or case.get('validity', 'CURRENT') != 'CURRENT'
                or case.get('blocking_request_count', 0)
                or str(case.get('line_id', '')).startswith('LEGACY-')):
            continue
        line = _text(case.get('line_id'), 'approved case ID', 200)
        _require(line not in seen, 'Duplicate approved case in forecast basis.')
        seen.add(line)
        digest = case.get('case_digest')
        _require(isinstance(digest, str) and bool(_SHA256.fullmatch(digest)),
                 'An approved forecast case requires its current case digest.')
        event = _text(case.get('review_event_id'), 'approval event ID', 500)
        net, known = _amount(case.get('net_candidate')), _amount(case.get('known_candidate'))
        _require(net == known and net > Decimal('0'), 'Approved forecast amounts must agree and be positive.')
        cases.append({'tenant_id': tenant, 'line_id': line,
            'supplier_id': _text(case.get('supplier_id'), 'supplier ID', 200),
            'contract_id': _text(case.get('contract_id'), 'contract ID', 200),
            'case_digest': digest, 'review_event_id': event,
            'assessment_id': case.get('assessment_id'),
            'workflow_thread_id': case.get('workflow_thread_id'),
            'amount_usd': format(net, '.2f')})
    cases.sort(key=lambda item: item['line_id'])
    with localcontext() as context:
        context.prec = 60
        total = sum((_amount(case['amount_usd']) for case in cases), Decimal('0'))
    basis = {'policy_version': POLICY_VERSION, 'tenant_id': tenant, 'currency': 'USD',
        'case_count': len(cases), 'amount_usd': format(total, '.2f'), 'cases': cases}
    return {**basis, 'basis_digest': _digest(basis)}


def _portfolio_basis(conn, summary, tenant):
    from .legacy_archive import _FIELDS

    current = basis_from_summary(summary, tenant=tenant)
    cases = [{**case, 'basis_kind': 'CURRENT_APPROVED_DRAFT'} for case in current['cases']]
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='legacy_cases'").fetchone()
    if exists:
        rows = conn.execute('''SELECT c.*,b.provenance,b.synthetic AS batch_synthetic,
            b.schema_version,b.source_system AS batch_source_system,b.payload_sha256
            FROM legacy_cases c LEFT JOIN legacy_archive_batches b ON c.archive_id=b.archive_id
            WHERE c.tenant_id=? AND c.currency='USD' ORDER BY c.line_id''', (tenant,))
        for row in rows:
            row = dict(row)
            _require(row['status'] == 'MIGRATED_APPROVED' and row['locked'] == 1
                     and row['synthetic'] == 1 and row['batch_synthetic'] == 1
                     and row['schema_version'] == 1
                     and row['source_system'] == row['batch_source_system'],
                     'Only locked, synthetic migrated approvals may enter the portfolio forecast.')
            _require(row['case_sha256'] == _digest({field: row[field] for field in _FIELDS['cases']}),
                     'Archived forecast source hash does not match its original record.')
            line = _text(row['line_id'], 'archived case ID', 200)
            _require(line.startswith('LEGACY-'), 'Archived forecast cases require the locked legacy namespace.')
            for field in ('case_sha256', 'payload_sha256'):
                _require(isinstance(row[field], str) and bool(_SHA256.fullmatch(row[field])),
                         'Archived forecast source hashes must be valid SHA-256 digests.')
            amount = _amount(row['historical_approved_amount'])
            cases.append({'tenant_id': tenant, 'line_id': line,
                'supplier_id': _text(row['supplier_id'], 'archived supplier ID', 200),
                'contract_id': _text(row['contract_id'], 'archived contract ID', 200),
                'archive_id': _text(row['archive_id'], 'archive ID', 200),
                'case_sha256': row['case_sha256'], 'archive_payload_sha256': row['payload_sha256'],
                'source_record_id': _text(row['source_record_id'], 'archive source record ID', 200),
                'source_system': _text(row['source_system'], 'archive source system', 200),
                'provenance': _text(row['provenance'], 'archive provenance', 1000),
                'origin_year': row['origin_year'], 'status': 'MIGRATED_APPROVED',
                'locked': True, 'synthetic': True, 'basis_kind': 'MIGRATED_APPROVED',
                'amount_usd': format(amount, '.2f')})
    cases.sort(key=lambda case: case['line_id'])
    _require(len({case['line_id'] for case in cases}) == len(cases),
             'A portfolio forecast case may appear only once.')
    with localcontext() as context:
        context.prec = 60
        total = sum((_amount(case['amount_usd']) for case in cases), Decimal('0'))
    basis = {'policy_version': POLICY_ALL_APPROVED, 'tenant_id': tenant, 'currency': 'USD',
        'case_count': len(cases), 'amount_usd': format(total, '.2f'), 'cases': cases}
    return {**basis, 'basis_digest': _digest(basis)}


def portfolio_basis(db, summary, *, tenant='DEMO'):
    """Combine approved drafts and locked migrated approvals for reporting only.

    Their original approval/source identities remain distinct in the snapshot;
    archived rows never enter the calculation engine or a new review workflow.
    """
    with _read_only(db) as conn:
        _guard(conn, tenant)
        return _portfolio_basis(conn, summary, tenant)


def _validated_basis(basis):
    _require(isinstance(basis, dict), 'A forecast basis is required.')
    fields = {'policy_version', 'tenant_id', 'currency', 'case_count', 'amount_usd', 'cases', 'basis_digest'}
    _require(set(basis) == fields and basis['policy_version'] in {POLICY_VERSION, POLICY_ALL_APPROVED}
             and basis['currency'] == 'USD' and isinstance(basis['cases'], list)
             and type(basis['case_count']) is int and basis['case_count'] == len(basis['cases']),
             'Malformed forecast basis.')
    raw = {key: value for key, value in basis.items() if key != 'basis_digest'}
    _require(basis['basis_digest'] == _digest(raw), 'Forecast basis digest does not match its snapshot.')
    with localcontext() as context:
        context.prec = 60
        total = sum((_amount(case['amount_usd']) for case in basis['cases']), Decimal('0'))
    _require(total == _amount(basis['amount_usd']), 'Forecast basis total does not reconcile.')
    return basis


def compute_forecast(basis, recovery_percentage):
    """Apply a human percentage once to the aggregate approved USD basis."""
    basis = _validated_basis(basis)
    rate = Decimal(_percentage(recovery_percentage))
    with localcontext() as context:
        context.prec = 60
        projected = (Decimal(basis['amount_usd']) * rate / Decimal('100')).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP)
    return format(projected, '.2f')


@contextmanager
def _read_only(db):
    path = Path(db)
    _require(path.is_file(), 'An existing synthetic database is required.')
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        yield conn


def _guard(conn, tenant):
    row = conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
    _require(row is not None, 'A synthetic import manifest is required.')
    manifest = json.loads(row[0])
    _require(manifest.get('synthetic') is True, 'SYNTHETIC_ONLY: forecasts require a synthetic demo database.')
    _require(tenant in manifest.get('tenants', []), 'Forecast tenant is not in the synthetic manifest.')
    return manifest


def _table(scope):
    _require(isinstance(scope, str) and scope in _SCOPES, 'Forecast scope must be current or all_approved.')
    return _SCOPES[scope]


def _exists(conn, scope='current'):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (_table(scope),)).fetchone() is not None


def _latest(conn, tenant, scope='current'):
    table = _table(scope)
    return conn.execute(f'SELECT * FROM {table} WHERE tenant_id=? ORDER BY sequence DESC LIMIT 1',
                        (tenant,)).fetchone() if _exists(conn, scope) else None


def _fresh_basis(db, checkpoints, tenant, scope='current'):
    from .invoice_review import load_worklist
    summary = load_worklist(db, checkpoints, tenant=tenant)['summary']
    return (basis_from_summary(summary, tenant=tenant) if scope == 'current'
            else portfolio_basis(db, summary, tenant=tenant))


def _report(basis, saved):
    assumption = dict(saved) if saved is not None else None
    status = 'NOT_SET'
    projected = None
    if assumption is not None:
        assumption['basis'] = json.loads(assumption.pop('basis_json'))
        _validated_basis(assumption['basis'])
        _require(assumption['basis_digest'] == assumption['basis']['basis_digest']
                 and assumption['projected_cash_usd'] == compute_forecast(
                     assumption['basis'], assumption['recovery_percentage']),
                 'Saved forecast assumption failed reconciliation.')
        status = 'CURRENT' if assumption['basis_digest'] == basis['basis_digest'] else 'STALE_BASIS'
        if status == 'CURRENT':
            projected = assumption['projected_cash_usd']
    scope = 'current' if basis['policy_version'] == POLICY_VERSION else 'all_approved'
    return {'status': status, 'scope': scope, 'label': LABEL, 'notice': NOTICE, 'basis': basis,
        'assumption': assumption, 'projected_cash_usd': projected, 'realized_recovery': '0.00'}


def get_forecast(db, checkpoints, *, tenant='DEMO', summary=None, scope='current'):
    """Read a forecast without creating schema, assumptions, reviews or workflows.

    A supplied summary is useful for display reuse only. Saving always obtains a
    fresh uncached basis, regardless of the summary displayed by the caller.
    """
    _table(scope)
    with _read_only(db) as conn:
        _guard(conn, tenant)
        saved = _latest(conn, tenant, scope)
        if summary is not None:
            basis = (basis_from_summary(summary, tenant=tenant) if scope == 'current'
                     else _portfolio_basis(conn, summary, tenant))
    if summary is None:
        basis = _fresh_basis(db, checkpoints, tenant, scope)
    return _report(basis, saved)


def _schema(conn, scope='current'):
    table = _table(scope)
    prefix = 'forecast' if scope == 'current' else 'forecast_all_approved'
    conn.execute(f'''CREATE TABLE IF NOT EXISTS {table} (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, forecast_id TEXT UNIQUE NOT NULL,
        tenant_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, payload_digest TEXT NOT NULL,
        basis_digest TEXT NOT NULL, basis_json TEXT NOT NULL, recovery_percentage TEXT NOT NULL,
        reason TEXT NOT NULL, reviewer TEXT NOT NULL, as_of TEXT NOT NULL, horizon_date TEXT NOT NULL,
        projected_cash_usd TEXT NOT NULL, created_at TEXT NOT NULL,
        UNIQUE(tenant_id,idempotency_key))''')
    for operation in ('UPDATE', 'DELETE'):
        conn.execute(f'''CREATE TRIGGER IF NOT EXISTS {prefix}_no_{operation.lower()}
            BEFORE {operation} ON {table}
            BEGIN SELECT RAISE(ABORT,'Forecast assumptions are immutable'); END''')
    conn.execute(f'''CREATE TRIGGER IF NOT EXISTS {prefix}_no_replace
        BEFORE INSERT ON {table} WHEN EXISTS (
            SELECT 1 FROM {table} WHERE forecast_id=NEW.forecast_id
            OR (tenant_id=NEW.tenant_id AND idempotency_key=NEW.idempotency_key)
            OR sequence=NEW.sequence)
        BEGIN SELECT RAISE(ABORT,'Forecast assumptions are immutable'); END''')


@contextmanager
def _checkpoint_lock(checkpoints):
    path = Path(checkpoints)
    if not path.exists():
        yield
        return
    _require(path.is_file(), 'Checkpoint path must be a file.')
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=rw', uri=True)) as conn:
        conn.execute('PRAGMA busy_timeout=5000')
        conn.execute('BEGIN IMMEDIATE')
        try:
            yield
        finally:
            conn.rollback()


def save_forecast(db, checkpoints, *, expected_basis_digest, recovery_percentage,
                  reason, reviewer, as_of, horizon_date, idempotency_key, tenant='DEMO', scope='current'):
    """Append an explicitly reviewed assumption against a freshly verified basis."""
    table = _table(scope)
    tenant = _text(tenant, 'tenant', 200)
    rate = _percentage(recovery_percentage)
    reason = _text(reason, 'forecast reason')
    reviewer = _text(reviewer, 'reviewer label', 200)
    key = _text(idempotency_key, 'idempotency key', 200)
    start, end = _date(as_of, 'Forecast as-of date'), _date(horizon_date, 'Forecast horizon date')
    _require(end >= start, 'Forecast horizon date must be on or after its as-of date.')
    _require(isinstance(expected_basis_digest, str) and bool(_SHA256.fullmatch(expected_basis_digest)),
             'The displayed forecast basis digest is required.')
    payload = {'tenant_id': tenant, 'basis_digest': expected_basis_digest,
        'recovery_percentage': rate, 'reason': reason, 'reviewer': reviewer,
        'as_of': as_of, 'horizon_date': horizon_date}
    if scope != 'current':
        payload['scope'] = scope
    _require(Path(db).is_file(), 'An existing synthetic database is required.')
    with connect(db) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        manifest = _guard(conn, tenant)
        _require(start >= _date(manifest['as_of'], 'Evidence as-of date'),
                 'Forecast as-of date cannot precede the evidence snapshot.')
        with _checkpoint_lock(checkpoints):
            basis = _fresh_basis(db, checkpoints, tenant, scope)
            if _exists(conn, scope):
                previous = conn.execute(f'SELECT * FROM {table} WHERE tenant_id=? AND idempotency_key=?',
                                        (tenant, key)).fetchone()
                if previous:
                    _require(previous['payload_digest'] == _digest(payload),
                             'Forecast idempotency key was already used for a different assumption.')
                    return _report(basis, previous)
            _require(basis['basis_digest'] == expected_basis_digest,
                     'STALE_BASIS: approved cases changed; refresh and review the forecast again.')
            _require(basis['case_count'] > 0, 'A cash forecast requires current approved-for-draft USD cases.')
            _schema(conn, scope)
            record = {'forecast_id': 'forecast:' + uuid.uuid4().hex, 'tenant_id': tenant,
                'idempotency_key': key, 'payload_digest': _digest(payload),
                'basis_digest': basis['basis_digest'], 'basis_json': _encode(basis),
                'recovery_percentage': rate, 'reason': reason, 'reviewer': reviewer,
                'as_of': as_of, 'horizon_date': horizon_date,
                'projected_cash_usd': compute_forecast(basis, rate),
                'created_at': datetime.now(timezone.utc).isoformat()}
            fields = list(record)
            conn.execute('INSERT INTO ' + table + ' (' + ','.join(fields) + ') VALUES (' +
                         ','.join('?' for _ in fields) + ')', tuple(record[field] for field in fields))
            saved = conn.execute(f'SELECT * FROM {table} WHERE forecast_id=?',
                                 (record['forecast_id'],)).fetchone()
            return _report(basis, saved)
