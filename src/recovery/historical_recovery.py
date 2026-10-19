"""Explicit invented legacy receipts, separate from current financial findings.

This overlay models the user's synthetic 2024/2025 collection scenario. It does
not record a current settlement or change engine, archive, review or forecast
rows. A historical receipt collects 100% of its frozen, already discounted cash
forecast. Its whole approved case then leaves the remaining forecast basis.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, localcontext
import json
from pathlib import Path
import uuid

from . import forecast as f
from .storage import connect


POLICY_VERSION = 'synthetic-historical-realization-v1'
NOTICE = ('Invented 2024/2025 historical cash receipts for the synthetic demo; '
          'not evidence of actual collections or new current recoveries.')
_HEADERS = 'historical_recovery_receipts'
_ITEMS = 'historical_recovery_items'
_HEADER_FIELDS = ('receipt_id', 'tenant_id', 'idempotency_key', 'payload_digest',
    'policy_version', 'synthetic', 'basis_digest', 'basis_json', 'forecast_id',
    'forecast_json', 'recovery_percentage', 'reviewer', 'reason', 'as_of',
    'reconciliation_json', 'created_at')
_ITEM_FIELDS = ('receipt_id', 'tenant_id', 'line_id', 'archive_id', 'source_system',
    'source_record_id', 'case_sha256', 'archive_payload_sha256', 'origin_year',
    'effective_date', 'approved_amount_usd', 'realized_amount_usd', 'source_json')


def _exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _sum(values):
    with localcontext() as context:
        context.prec = 60
        return format(sum((f._amount(value) for value in values), Decimal('0')), '.2f')


def _cash(amount, percentage):
    with localcontext() as context:
        context.prec = 60
        return format((f._amount(amount) * Decimal(percentage) / Decimal('100')).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP), '.2f')


def _archive_cases(basis):
    cases = [case for case in basis['cases'] if case['basis_kind'] == 'MIGRATED_APPROVED']
    f._require(all(type(case.get('origin_year')) is int and case['origin_year'] in (2024, 2025)
                   and case.get('locked') is True and case.get('synthetic') is True
                   and case.get('status') == 'MIGRATED_APPROVED' for case in cases),
               'Historical realization requires locked synthetic 2024/2025 approvals.')
    return cases


def _validate_archives(conn, cases):
    """Reconstruct complete original imports, including their linked sources."""
    from .legacy_archive import _FIELDS, _sha, _validated

    for archive_id in sorted({case['archive_id'] for case in cases}):
        batch = conn.execute('SELECT * FROM legacy_archive_batches WHERE archive_id=?', (archive_id,)).fetchone()
        f._require(batch is not None, 'Historical realization archive source is missing.')
        payload = {'schema_version': batch['schema_version'], 'synthetic': batch['synthetic'] == 1,
            'archive_id': archive_id, 'source_system': batch['source_system'], 'provenance': batch['provenance']}
        for group, fields in _FIELDS.items():
            rows = conn.execute(f'SELECT {",".join(fields)} FROM legacy_{group} WHERE archive_id=?', (archive_id,))
            payload[group] = [dict(row) for row in rows]
        f._require(_sha(_validated(payload)) == batch['payload_sha256'],
                   'Historical realization archive payload hash failed reconciliation.')


def _allocate(cases, percentage):
    """Largest-remainder cents reconcile exactly to each year's aggregate."""
    result, years = [], {}
    with localcontext() as context:
        context.prec = 60
        for year in (2024, 2025):
            selected = sorted((case for case in cases if case['origin_year'] == year),
                              key=lambda case: case['line_id'])
            approved = _sum(case['amount_usd'] for case in selected)
            target = Decimal(_cash(approved, percentage))
            allocations = []
            for case in selected:
                exact = f._amount(case['amount_usd']) * Decimal(percentage) / Decimal('100')
                base = exact.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
                allocations.append([case, base, exact - base])
            pennies = int((target - sum((row[1] for row in allocations), Decimal('0'))) * Decimal('100'))
            ranked = sorted(allocations, key=lambda row: (-row[2], row[0]['line_id']))
            f._require(0 <= pennies <= len(ranked), 'Historical allocation failed reconciliation.')
            for row in ranked[:pennies]:
                row[1] += Decimal('0.01')
            result.extend((row[0], format(row[1], '.2f')) for row in allocations)
            years[str(year)] = {'case_count': len(selected), 'approved_amount_usd': approved,
                'realized_recovery_usd': format(target, '.2f')}
    return result, years


def _schema(conn):
    conn.execute(f'''CREATE TABLE IF NOT EXISTS {_HEADERS} (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, receipt_id TEXT UNIQUE NOT NULL,
        tenant_id TEXT NOT NULL,idempotency_key TEXT NOT NULL,payload_digest TEXT NOT NULL,
        policy_version TEXT NOT NULL,synthetic INTEGER NOT NULL CHECK(synthetic=1),
        basis_digest TEXT NOT NULL,basis_json TEXT NOT NULL,forecast_id TEXT NOT NULL,
        forecast_json TEXT NOT NULL,recovery_percentage TEXT NOT NULL,
        reviewer TEXT NOT NULL,reason TEXT NOT NULL,as_of TEXT NOT NULL,
        reconciliation_json TEXT NOT NULL,created_at TEXT NOT NULL,record_digest TEXT NOT NULL,
        UNIQUE(tenant_id,idempotency_key))''')
    conn.execute(f'''CREATE TABLE IF NOT EXISTS {_ITEMS} (
        receipt_id TEXT NOT NULL REFERENCES {_HEADERS}(receipt_id),tenant_id TEXT NOT NULL,
        line_id TEXT NOT NULL,archive_id TEXT NOT NULL,source_system TEXT NOT NULL,
        source_record_id TEXT NOT NULL,case_sha256 TEXT NOT NULL,archive_payload_sha256 TEXT NOT NULL,
        origin_year INTEGER NOT NULL CHECK(origin_year IN (2024,2025)),effective_date TEXT NOT NULL,
        approved_amount_usd TEXT NOT NULL,realized_amount_usd TEXT NOT NULL,source_json TEXT NOT NULL,
        item_digest TEXT NOT NULL,PRIMARY KEY(tenant_id,line_id),
        UNIQUE(tenant_id,source_system,source_record_id))''')
    for table in (_HEADERS, _ITEMS):
        for operation in ('UPDATE', 'DELETE'):
            conn.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()}
                BEFORE {operation} ON {table}
                BEGIN SELECT RAISE(ABORT,'Historical realization records are immutable'); END''')
        identity = ('receipt_id=NEW.receipt_id OR '
                    '(tenant_id=NEW.tenant_id AND idempotency_key=NEW.idempotency_key) OR sequence=NEW.sequence'
                    if table == _HEADERS else '(tenant_id=NEW.tenant_id AND line_id=NEW.line_id) OR '
                    '(tenant_id=NEW.tenant_id AND source_system=NEW.source_system AND source_record_id=NEW.source_record_id)')
        conn.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_replace
            BEFORE INSERT ON {table} WHEN EXISTS (SELECT 1 FROM {table} WHERE {identity})
            BEGIN SELECT RAISE(ABORT,'Historical realization records are immutable'); END''')


def _item(record, case, realized):
    raw = {'receipt_id': record['receipt_id'], 'tenant_id': record['tenant_id'],
        'line_id': case['line_id'], 'archive_id': case['archive_id'], 'source_system': case['source_system'],
        'source_record_id': case['source_record_id'], 'case_sha256': case['case_sha256'],
        'archive_payload_sha256': case['archive_payload_sha256'], 'origin_year': case['origin_year'],
        'effective_date': f"{case['origin_year']}-12-31", 'approved_amount_usd': case['amount_usd'],
        'realized_amount_usd': realized, 'source_json': f._encode(case)}
    return {**raw, 'item_digest': f._digest(raw)}


def _receipt_report(row):
    return {'receipt_id': row['receipt_id'], 'tenant_id': row['tenant_id'], 'synthetic': True,
        'notice': NOTICE, 'forecast_id': row['forecast_id'], 'basis_digest': row['basis_digest'],
        'recovery_percentage': row['recovery_percentage'], 'realization_percentage_of_frozen_forecast': '100',
        'reviewer': row['reviewer'], 'reason': row['reason'], 'as_of': row['as_of'],
        'created_at': row['created_at'], **json.loads(row['reconciliation_json'])}


def _read_ledger(conn, basis):
    present = [_exists(conn, table) for table in (_HEADERS, _ITEMS)]
    f._require(present[0] == present[1], 'Historical realization schema is incomplete.')
    if not present[0]:
        return [], []
    canonical = {case['line_id']: case for case in _archive_cases(basis)}
    headers = [dict(row) for row in conn.execute(
        f'SELECT * FROM {_HEADERS} WHERE tenant_id=? ORDER BY sequence', (basis['tenant_id'],))]
    items = [dict(row) for row in conn.execute(
        f'SELECT * FROM {_ITEMS} WHERE tenant_id=? ORDER BY line_id', (basis['tenant_id'],))]
    receipts, seen = [], set()
    for header in headers:
        raw = {field: header[field] for field in _HEADER_FIELDS}
        f._require(header['record_digest'] == f._digest(raw) and header['synthetic'] == 1
                   and header['policy_version'] == POLICY_VERSION,
                   'Historical realization receipt hash failed reconciliation.')
        saved_basis = f._validated_basis(json.loads(header['basis_json']))
        frozen = json.loads(header['forecast_json'])
        f._require(saved_basis['policy_version'] == f.POLICY_ALL_APPROVED
                   and saved_basis['tenant_id'] == basis['tenant_id']
                   and header['basis_digest'] == saved_basis['basis_digest']
                   and header['forecast_id'] == frozen['forecast_id']
                   and header['recovery_percentage'] == frozen['recovery_percentage']
                   and Decimal(f._percentage(header['recovery_percentage'])) > Decimal('0'),
                   'Historical realization forecast identity failed reconciliation.')
        source = conn.execute('SELECT * FROM forecast_all_approved_assumptions WHERE forecast_id=?',
                              (header['forecast_id'],)).fetchone()
        f._require(source is not None and dict(source) == frozen
                   and frozen['basis_json'] == header['basis_json'],
                   'Historical realization frozen forecast no longer matches its immutable source.')
        f._report(saved_basis, source)
        payload = {'tenant_id': header['tenant_id'], 'basis_digest': header['basis_digest'],
            'reviewer': f._text(header['reviewer'], 'reviewer label', 200),
            'reason': f._text(header['reason'], 'historical realization reason'),
            'as_of': header['as_of'], 'synthetic_confirmed': True}
        f._require(header['payload_digest'] == f._digest(payload),
                   'Historical realization review payload failed reconciliation.')
        dated = f._date(header['as_of'], 'Historical realization as-of date')
        f._require(dated >= f._date(frozen['as_of'], 'Frozen forecast as-of date'),
                   'Historical realization review date precedes its frozen forecast.')
        selected = [row for row in items if row['receipt_id'] == header['receipt_id']]
        saved_cases = {case['line_id']: case for case in _archive_cases(saved_basis)}
        source_cases = []
        for item in selected:
            line = item['line_id']
            f._require(line not in seen and line in saved_cases and line in canonical
                       and saved_cases[line] == canonical[line]
                       and json.loads(item['source_json']) == canonical[line],
                       'Historical realization source identity failed reconciliation.')
            seen.add(line)
            source_cases.append(canonical[line])
            f._require(dated >= f._date(item['effective_date'], 'Historical receipt effective date'),
                       'Historical receipt cannot occur after its review date.')
        allocated, years = _allocate(source_cases, header['recovery_percentage'])
        expected = {case['line_id']: _item(header, case, amount) for case, amount in allocated}
        f._require(bool(selected) and all(item == expected[item['line_id']] for item in selected),
                   'Historical realization item hash or amount failed reconciliation.')
        reconciliation = {'case_count': len(selected), 'by_year': years,
            'approved_amount_usd': _sum(row['approved_amount_usd'] for row in selected),
            'realized_recovery_usd': _sum(row['realized_amount_usd'] for row in selected)}
        f._require(json.loads(header['reconciliation_json']) == reconciliation,
                   'Historical realization receipt totals failed reconciliation.')
        receipts.append(_receipt_report(header))
    f._require(len(seen) == len(items), 'Historical realization contains orphaned receipt items.')
    return receipts, items


def get_recovery_overview(db, checkpoints, *, tenant='DEMO', summary=None):
    """Read all-year approvals, outstanding forecast and invented old receipts."""
    report = f.get_forecast(db, checkpoints, tenant=tenant, summary=summary, scope='all_approved')
    basis = report['basis']
    with f._read_only(db) as conn:
        f._guard(conn, tenant)
        _validate_archives(conn, _archive_cases(basis))
        receipts, items = _read_ledger(conn, basis)
    realized_lines = {item['line_id'] for item in items}
    outstanding = _sum(case['amount_usd'] for case in basis['cases'] if case['line_id'] not in realized_lines)
    years = {str(year): {'case_count': sum(row['origin_year'] == year for row in items),
        'approved_amount_usd': _sum(row['approved_amount_usd'] for row in items if row['origin_year'] == year),
        'realized_recovery_usd': _sum(row['realized_amount_usd'] for row in items if row['origin_year'] == year)}
        for year in (2024, 2025)}
    realized = _sum(row['realized_amount_usd'] for row in items)
    return {'forecast': report, 'total_approved_usd': basis['amount_usd'],
        'outstanding_approved_usd': outstanding,
        'estimated_cash_usd': (_cash(outstanding, report['assumption']['recovery_percentage'])
                               if report['status'] == 'CURRENT' else None),
        'realized_recovery_usd': realized, 'current_realized_recovery_usd': '0.00',
        'historical_receipts': {'synthetic': True, 'notice': NOTICE, 'receipt_count': len(receipts),
            'case_count': len(items), 'realized_recovery_usd': realized, 'by_year': years, 'receipts': receipts}}


def save_historical_realization(db, checkpoints, *, expected_basis_digest, reviewer, reason,
                                as_of, idempotency_key, tenant='DEMO', synthetic_confirmed=False):
    """Append an explicitly confirmed synthetic receipt against a fresh basis."""
    f._require(synthetic_confirmed is True, 'SYNTHETIC_ONLY: explicitly confirm the invented historical receipt scenario.')
    tenant = f._text(tenant, 'tenant', 200)
    reviewer = f._text(reviewer, 'reviewer label', 200)
    reason = f._text(reason, 'historical realization reason')
    key = f._text(idempotency_key, 'idempotency key', 200)
    dated = f._date(as_of, 'Historical realization as-of date')
    f._require(isinstance(expected_basis_digest, str) and bool(f._SHA256.fullmatch(expected_basis_digest)),
               'The displayed all-approved forecast basis digest is required.')
    payload = {'tenant_id': tenant, 'basis_digest': expected_basis_digest, 'reviewer': reviewer,
        'reason': reason, 'as_of': as_of, 'synthetic_confirmed': True}
    f._require(Path(db).is_file(), 'An existing synthetic database is required.')
    with connect(db) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        manifest = f._guard(conn, tenant)
        f._require(dated >= f._date(manifest['as_of'], 'Evidence as-of date'),
                   'Historical realization as-of date cannot precede the evidence snapshot.')
        with f._checkpoint_lock(checkpoints):
            basis = f._fresh_basis(db, checkpoints, tenant, 'all_approved')
            historical = _archive_cases(basis)
            _validate_archives(conn, historical)
            receipts, items = _read_ledger(conn, basis)
            if _exists(conn, _HEADERS):
                previous = conn.execute(f'SELECT * FROM {_HEADERS} WHERE tenant_id=? AND idempotency_key=?',
                                        (tenant, key)).fetchone()
                if previous:
                    f._require(previous['payload_digest'] == f._digest(payload),
                               'Historical realization idempotency key was already used for a different review.')
                    return _receipt_report(previous)
            f._require(basis['basis_digest'] == expected_basis_digest,
                       'STALE_BASIS: approved cases changed; refresh and review historical realization again.')
            saved = f._latest(conn, tenant, 'all_approved')
            report = f._report(basis, saved)
            f._require(report['status'] == 'CURRENT',
                       'A current saved all-approved forecast is required before historical realization.')
            percentage = report['assumption']['recovery_percentage']
            f._require(Decimal(f._percentage(percentage)) > Decimal('0'),
                       'Historical realization requires a positive saved forecast percentage.')
            f._require(dated >= f._date(saved['as_of'], 'Frozen forecast as-of date'),
                       'Historical realization review date cannot precede its frozen forecast.')
            seen = {row['line_id'] for row in items}
            selected = [case for case in historical if case['line_id'] not in seen]
            f._require(bool(selected), 'No unrealized historical approvals remain; existing receipts are never counted twice.')
            f._require(all(dated >= f._date(f"{case['origin_year']}-12-31", 'Receipt effective date') for case in selected),
                       'Historical receipt effective dates cannot follow the review date.')
            allocation, years = _allocate(selected, percentage)
            total = _sum(amount for _, amount in allocation)
            f._require(Decimal(total) > Decimal('0'), 'Historical realization requires a positive cash amount.')
            reconciliation = {'case_count': len(selected), 'by_year': years,
                'approved_amount_usd': _sum(case['amount_usd'] for case in selected), 'realized_recovery_usd': total}
            record = {'receipt_id': 'historical-receipt:' + uuid.uuid4().hex, 'tenant_id': tenant,
                'idempotency_key': key, 'payload_digest': f._digest(payload), 'policy_version': POLICY_VERSION,
                'synthetic': 1, 'basis_digest': basis['basis_digest'], 'basis_json': f._encode(basis),
                'forecast_id': saved['forecast_id'], 'forecast_json': f._encode(dict(saved)),
                'recovery_percentage': percentage, 'reviewer': reviewer, 'reason': reason, 'as_of': as_of,
                'reconciliation_json': f._encode(reconciliation), 'created_at': datetime.now(timezone.utc).isoformat()}
            record['record_digest'] = f._digest(record)
            _schema(conn)
            fields = list(record)
            conn.execute(f'INSERT INTO {_HEADERS} ({",".join(fields)}) VALUES ({",".join("?" for _ in fields)})',
                         tuple(record[field] for field in fields))
            for case, amount in allocation:
                item = _item(record, case, amount)
                fields = list(item)
                conn.execute(f'INSERT INTO {_ITEMS} ({",".join(fields)}) VALUES ({",".join("?" for _ in fields)})',
                             tuple(item[field] for field in fields))
            _read_ledger(conn, basis)
            return _receipt_report(record)
