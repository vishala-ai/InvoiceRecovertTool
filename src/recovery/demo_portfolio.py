"""Build a separate, explicitly invented portfolio through real review workflows.

The original import manifest/source digest remain baseline provenance anchors.
All additions are reconciled separately in ``demo_portfolio`` metadata. This
preserves current approval authority for untouched original invoice scopes.
No model, external system, supplier communication or settlement is involved.
"""
from __future__ import annotations

from contextlib import closing
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid

from .schemas import TABLE_KEYS, TABLE_SCHEMAS, validate_row
from .storage import connect, metadata


PROFILE_ID = 'synthetic-scaled-2026-portfolio-v1'
AS_OF = '2026-10-15'
REVIEWER = 'User-authorized synthetic portfolio fixture owner'
FIXTURE_NOTE = ('Manually curated synthetic fixture terms for the user-authorized demo. '
                'No AI extraction or independent real-contract verification is claimed.')
APPROVAL_NOTE = ('The user explicitly authorized approval of these eligible synthetic '
                 'demonstration cases for draft preparation. No supplier submission, '
                 'independent SME review, cash receipt or realized recovery is asserted.')


def _encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      default=lambda blob: {'sqlite_blob_hex': blob.hex()})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_db(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    return closing(connection)


def _safe_path(path):
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        _require(part not in ('.', '..') and not part.lower().startswith('.env'), 'Unsafe demo path')
        current /= part
        _require(not current.is_symlink(), 'Demo paths must not contain symlinks')
    return path


def _rows(conn, table):
    return [dict(row) for row in conn.execute('SELECT * FROM "' + table.replace('"', '""') + '"')]


def database_fingerprints(path):
    """Logical hashes include checkpoint BLOBs without depending on WAL file bytes."""
    with _read_db(path) as conn:
        names = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {name: hashlib.sha256(_encode(sorted(_rows(conn, name), key=_encode)).encode()).hexdigest()
                for name in names}


def _copy_database(source, target):
    _require(not target.exists(), 'Refusing to replace an existing synthetic database')
    target.parent.mkdir(parents=True, exist_ok=True)
    with _read_db(source) as original, closing(sqlite3.connect(target)) as copied:
        original.backup(copied)
        _require(copied.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'SQLite backup failed integrity verification')


def current_portfolio_plan(*, vendor_count=35, approved=500, awaiting=250, evidence=250):
    """Return schema-valid invented records; decisions are targets, not saved states."""
    _require(type(vendor_count) is int and 1 <= vendor_count <= 500, 'Invalid synthetic vendor count')
    _require(all(type(value) is int and value >= 0 for value in (approved, awaiting, evidence))
             and 0 < approved + awaiting + evidence <= 10000, 'Invalid synthetic case counts')
    tables = {table: [] for table in ('suppliers', 'purchase_orders', 'invoice_lines', 'service_entries', 'payments')}
    for number in range(1, vendor_count + 1):
        tables['suppliers'].append({'tenant_id': 'DEMO', 'supplier_id': f'DEMO-SCALE-V{number:03d}',
            'name': f'Synthetic Portfolio Vendor {number:03d}'})
    scopes, cases = {}, []
    for number in range(1, approved + awaiting + evidence + 1):
        supplier = f'DEMO-SCALE-V{(number - 1) % vendor_count + 1:03d}'
        target = 'APPROVED_FOR_DRAFT' if number <= approved else 'AWAITING_APPROVAL' if number <= approved + awaiting else 'NEEDS_REVIEW'
        line = f'SCALE-2026-{number:04d}'
        # A request changes its entire rate-scope input identity. Give evidence
        # cases independent contracts so no other case inherits that blocker.
        contract = ('CTR-SCALE-EVIDENCE-' + f'{number:04d}' if target == 'NEEDS_REVIEW'
                    else 'CTR-SCALE-VALID-' + supplier.rsplit('V', 1)[1])
        scope = {'supplier_id': supplier, 'contract_id': contract, 'role': 'ENGINEER', 'location': 'REMOTE'}
        scopes[(supplier, contract)] = scope
        po_id = 'PO-' + contract
        billed = format(Decimal('100') * Decimal('110.00'), '.2f')
        tables['invoice_lines'].append({'tenant_id': 'DEMO', 'line_id': line,
            'invoice_id': f'SCALE-INV-2026-{number:04d}', 'line_number': '1', **scope,
            'po_id': po_id, 'unit': 'HOUR', 'service_type': 'REGULAR', 'currency': 'USD',
            'billed_hours': '100', 'billed_rate': '110.00', 'billed_amount': billed,
            'invoice_date': '2026-09-30',
            'description': 'INVENTED 2026 synthetic portfolio case; not a real supplier claim.'})
        tables['service_entries'].append({'tenant_id': 'DEMO', 'entry_id': 'SVC-' + line,
            'line_id': line, 'service_date': '2026-09-15', 'approved_hours': '100', 'approved': 'true'})
        tables['payments'].append({'tenant_id': 'DEMO', 'payment_id': 'PAY-' + line,
            'line_id': line, 'amount': billed, 'currency': 'USD',
            'settled_date': '2026-10-06', 'status': 'SETTLED'})
        cases.append({'line_id': line, 'supplier_id': supplier, 'contract_id': contract,
                      'target': target, 'thread_id': 'ui:scaled-2026:' + line})
    for scope in scopes.values():
        tables['purchase_orders'].append({'tenant_id': 'DEMO', 'po_id': 'PO-' + scope['contract_id'],
            **{key: scope[key] for key in ('supplier_id', 'contract_id', 'role')},
            'currency': 'USD', 'unit': 'HOUR', 'po_rate': '110.00'})
    for table, rows in tables.items():
        for index, row in enumerate(rows, start=2):
            _require(not validate_row(table, row, index), f'Invalid invented source row: {table}/{index}')
    return {'synthetic': True, 'profile_id': PROFILE_ID, 'as_of': AS_OF, 'tables': tables,
            'rate_scopes': list(scopes.values()), 'cases': cases,
            'targets': {'approved': approved, 'awaiting': awaiting, 'evidence': evidence},
            'vendor_count': vendor_count}


def historical_portfolio_payload(*, vendor_count=50, cases_per_year=1000):
    """Migrated historical amounts remain outside operational invoices and money totals."""
    _require(type(vendor_count) is int and 1 <= vendor_count <= 500, 'Invalid historical vendor count')
    _require(type(cases_per_year) is int and 1 <= cases_per_year <= 10000, 'Invalid historical case count')
    suppliers = [{'tenant_id': 'DEMO', 'supplier_id': f'LEGACY-SUP-{number:03d}',
                  'name': f'Legacy Synthetic Vendor {number:03d}'} for number in range(1, vendor_count + 1)]
    contracts, cases = [], []
    for year in (2024, 2025):
        for supplier in suppliers:
            suffix = supplier['supplier_id'].rsplit('-', 1)[1]
            contracts.append({'tenant_id': 'DEMO', 'supplier_id': supplier['supplier_id'],
                'contract_id': f'LEGACY-CTR-{year}-{suffix}', 'title': f'Locked synthetic {year} contract {suffix}',
                'origin_year': year})
        for number in range(1, cases_per_year + 1):
            suffix = f'{(number - 1) % vendor_count + 1:03d}'
            cases.append({'tenant_id': 'DEMO', 'line_id': f'LEGACY-{year}-{number:04d}',
                'supplier_id': f'LEGACY-SUP-{suffix}', 'contract_id': f'LEGACY-CTR-{year}-{suffix}',
                'invoice_id': f'LEGACY-INV-{year}-{number:04d}', 'invoice_date': f'{year}-09-30',
                'origin_year': year, 'currency': 'USD', 'historical_approved_amount': '1000.00',
                'source_record_id': f'LEGACY-AP-{year}-{number:04d}',
                'description': 'Invented migrated approval, locked; not a current recovery.'})
    return {'schema_version': 1, 'synthetic': True, 'archive_id': 'LEGACY-2024-2025-v1',
        'source_system': 'Legacy AP (synthetic)', 'provenance': 'Migrated from old system — locked (synthetic)',
        'suppliers': suppliers, 'contracts': contracts, 'cases': cases}


def _append_facts(db, plan):
    with connect(db) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        _require(json.loads(metadata(conn, 'manifest')).get('synthetic') is True, 'SYNTHETIC_ONLY')
        for table, rows in plan['tables'].items():
            fields = list(TABLE_SCHEMAS[table])
            for position, row in enumerate(rows, start=2):
                _require(not validate_row(table, row, position), 'Synthetic append schema validation failed')
                keys = TABLE_KEYS[table]
                _require(conn.execute('SELECT 1 FROM ' + table + ' WHERE ' + ' AND '.join(key + '=?' for key in keys),
                    tuple(row[key] for key in keys)).fetchone() is None, 'Synthetic append collides with an existing source ID')
            conn.executemany('INSERT INTO ' + table + ' (' + ','.join(fields) + ') VALUES (' +
                ','.join('?' for _ in fields) + ')', [tuple(row[field] for field in fields) for row in rows])
        # These indexes affect only the new profile, never calculation rules.
        for name, table, fields in (
            ('invoice_scope', 'invoice_lines', 'tenant_id,supplier_id,contract_id,role,location'),
            ('document_scope', 'documents', 'tenant_id,supplier_id,contract_id,role,location'),
            ('term_scope', 'reviewed_terms', 'tenant_id,supplier_id,contract_id,role,location'),
            ('source_scope', 'source_versions', 'scope_id'),
            ('term_version_scope', 'approved_term_versions', 'scope_id'),
            ('service_line', 'service_entries', 'tenant_id,line_id,service_date'),
            ('payment_line', 'payments', 'tenant_id,line_id'),
            ('assessment_line', 'assessment_revisions', 'tenant_id,line_id,sequence')):
            conn.execute(f'CREATE INDEX scaled_demo_{name} ON {table}({fields})')


def _curate_terms(db, scopes, notify):
    from . import lifecycle
    from .lifecycle_storage import scope_fingerprint, transaction

    source_hashes = []
    for index, scope in enumerate(scopes, start=1):
        quote = ('ENGINEER services at REMOTE are billed in USD at 100.00 per HOUR '
                 'for approved REGULAR hours, effective 2026-01-01.')
        body = (f'# SYNTHETIC RATE AGREEMENT — NOT A REAL CONTRACT\n\n'
            f"Tenant: DEMO; Supplier: {scope['supplier_id']}; Contract: {scope['contract_id']}\n"
            'Simulated signed date: 2026-01-01\n\n' + quote + '\n\n'
            'This rate term supersedes no earlier term. One complete regular-hour rate '
            'applies only to the stated role, location, currency and service dates.\n'
            'Manually curated by the user-authorized synthetic demo fixture owner. '
            'No AI extraction or real signature verification is represented.\n')
        stage = lifecycle.stage_document(db, 'DEMO', filename=scope['contract_id'] + '.md',
            content=body.encode(), synthetic_confirmed=True)
        source = lifecycle.admit_document(db, 'DEMO', stage['stage_id'], scope=scope,
            title='Synthetic 2026 agreement — ' + scope['contract_id'], signed_date='2026-01-01',
            status='SIGNED', synthetic_confirmed=True, signed_simulation_confirmed=True,
            reviewer=REVIEWER, note=FIXTURE_NOTE, idempotency_key='scaled-demo:source:' + scope['contract_id'])
        values = {'role': 'ENGINEER', 'location': 'REMOTE', 'currency': 'USD', 'unit': 'HOUR',
            'service_type': 'REGULAR', 'rate': '100.00', 'effective_from': '2026-01-01',
            'effective_to': None, 'supersedes': None, 'exact_quote': quote,
            'unresolved_questions': [], 'coverage': 'COMPLETE_SINGLE_RATE'}
        with transaction(db) as conn:
            document = {**source, **source['scope']}
            proposal = lifecycle._insert_proposal(conn, 'DEMO', document, values, values,
                {'provider': 'manual_synthetic_fixture', 'model': None, 'origin': 'USER_AUTHORIZED_CURATED_DEMO'},
                scope_fingerprint(conn, document, include_requests=False), note=FIXTURE_NOTE)
        lifecycle.review_terms(db, 'DEMO', proposal['proposal_id'], proposal['digest'],
            'approve', REVIEWER, FIXTURE_NOTE, 'scaled-demo:term-review:' + scope['contract_id'],
            coverage_confirmed=True)
        source_hashes.append({'source_version': source['version_id'], 'doc_id': source['doc_id'],
            'sha256': source['sha256'], 'supplier_id': scope['supplier_id'], 'contract_id': scope['contract_id'],
            'curation': 'manual_synthetic_fixture'})
        if index % 50 == 0 or index == len(scopes):
            notify({'phase': 'manually_curated_terms', 'completed': index, 'total': len(scopes)})
    return source_hashes


def _seed_reviews(db, checkpoints, cases, notify):
    from .lifecycle import reassess
    from .workflow import run_persistent

    receipts = []
    for index, case in enumerate(cases, start=1):
        started = run_persistent(db, checkpoints, case['thread_id'], tenant='DEMO',
            line_id=case['line_id'], use_llm=False)
        _require(started['next'] == ['approval'] and started['state']['case']['status'] == 'CANDIDATE'
                 and started['state']['brief']['mode'] == 'offline_template',
                 'A synthetic case did not reach a real validated offline review pause')
        receipt = {'line_id': case['line_id'], 'thread_id': case['thread_id'], 'target': case['target'],
                   'assessment_id': started['state']['assessment_id'], 'draft_mode': 'offline_template'}
        if case['target'] != 'AWAITING_APPROVAL':
            decision = 'approve' if case['target'] == 'APPROVED_FOR_DRAFT' else 'request_more'
            note = APPROVAL_NOTE if decision == 'approve' else (
                'Synthetic reviewer requests confirmation of this case-specific signed rate agreement. '
                'Keep the case blocked until the supporting agreement is attached and reviewed.')
            completed = run_persistent(db, checkpoints, case['thread_id'], tenant='DEMO',
                decision={'decision': decision, 'reviewer': REVIEWER, 'note': note, 'evidence_type': 'CONTRACT_RATE'})
            _require(not completed['next'], 'A completed synthetic review did not finish')
            event = completed['state']['outcome']
            _require(event['decision'] == ('APPROVED_FOR_DRAFT' if decision == 'approve' else 'NEEDS_MORE_EVIDENCE'),
                     'The existing workflow rejected the intended synthetic decision')
            receipt['review_event_id'] = event['event_id']
            if decision == 'request_more':
                revised = reassess(db, 'DEMO', case['line_id'], prior_review_event_id=event['event_id'],
                    idempotency_key='scaled-demo:evidence-reassessment:' + case['line_id'])
                _require(revised['case_json']['status'] == 'NEEDS_REVIEW'
                         and revised['case_json']['reason'] == 'OPEN_EVIDENCE_REQUEST',
                         'An evidence request did not remain a real assessment blocker')
        receipts.append(receipt)
        if index % 50 == 0 or index == len(cases):
            notify({'phase': 'actual_review_workflows', 'completed': index, 'total': len(cases)})
    return receipts


def _portfolio_state(db, checkpoints):
    from .dashboard import summarize_cases
    from .engine import audit
    from .lifecycle import list_assessments, list_requests
    from .workflow import list_review_workflows

    with _read_db(db) as conn:
        report = audit(conn, 'DEMO')
        events = _rows(conn, 'review_events')
    workflows = list_review_workflows(db, checkpoints, tenant='DEMO')
    requests = list_requests(db, 'DEMO')
    summary = summarize_cases(report['cases'], events=events, workflows=workflows,
        assessments=list_assessments(db, 'DEMO'), requests=requests)
    return report, summary, events, workflows, requests


def create_scaled_portfolio(source_db, output_directory, *, plan=None, archive=None, progress=None):
    """Publish a complete new local profile; source data and checkpoints stay untouched."""
    from .legacy_archive import import_archive
    from .lifecycle_storage import validate_source_projections

    started = time.perf_counter()
    notify = progress or (lambda event: None)
    source = _safe_path(source_db)
    output = _safe_path(output_directory)
    source_cp = source.with_name(source.stem + '-ui-checkpoints.sqlite3')
    _require(source.is_file() and source_cp.is_file(), 'Existing synthetic source and matching checkpoints are required')
    destinations = (output / 'portfolio.sqlite3', output / 'portfolio-ui-checkpoints.sqlite3', output / 'portfolio-profile.json')
    _require(not any(path.exists() for path in destinations), 'Scaled profile already exists; refusing replacement')
    with _read_db(source) as conn:
        source_manifest = metadata(conn, 'manifest')
        manifest = json.loads(source_manifest)
        _require(manifest.get('synthetic') is True, 'SYNTHETIC_ONLY: source is not a synthetic demo')
        _require(manifest.get('dataset_id') == 'synthetic-million-approved-demo-v1'
                 and manifest.get('as_of') == AS_OF, 'Use the existing 2026 million-dollar synthetic profile')
        validate_source_projections(conn)
        _require(not conn.execute("SELECT 1 FROM metadata WHERE key='demo_portfolio'").fetchone(),
                 'Source is already an expanded profile')
        original_digest = metadata(conn, 'source_digest')
        original_rows = {table: _rows(conn, table) for table in (*TABLE_SCHEMAS,
            'source_versions', 'approved_term_versions', 'review_events', 'assessment_revisions',
            'case_review_assessments', 'evidence_requests', 'evidence_request_events')}
    before = {'database': database_fingerprints(source), 'checkpoints': database_fingerprints(source_cp)}
    parent_report, parent_summary, _, _, _ = _portfolio_state(source, source_cp)
    original_approvals = {case['line_id']: (case['case_digest'], case['review_event_id'])
        for case in parent_summary['cases'] if case['current_decision'] == 'APPROVED_FOR_DRAFT'}
    plan = current_portfolio_plan() if plan is None else plan
    _require(plan.get('synthetic') is True and plan.get('profile_id') == PROFILE_ID
             and plan.get('as_of') == AS_OF, 'Invalid synthetic cohort plan')
    archive = historical_portfolio_payload() if archive is None else archive
    output.mkdir(parents=True, exist_ok=True)
    backup_dir = output / 'before' / uuid.uuid4().hex
    backup_dir.mkdir(parents=True)
    _copy_database(source, backup_dir / 'source.sqlite3')
    _copy_database(source_cp, backup_dir / 'source-ui-checkpoints.sqlite3')
    with tempfile.TemporaryDirectory(prefix='.portfolio-build-', dir=output) as temporary:
        stage = Path(temporary)
        db, checkpoints = stage / 'portfolio.sqlite3', stage / 'portfolio-ui-checkpoints.sqlite3'
        _copy_database(backup_dir / 'source.sqlite3', db)
        _copy_database(backup_dir / 'source-ui-checkpoints.sqlite3', checkpoints)
        _append_facts(db, plan)
        notify({'phase': 'schema_validated_facts_appended', 'cases': len(plan['cases']), 'vendors': plan['vendor_count']})
        hashes = _curate_terms(db, plan['rate_scopes'], notify)
        receipts = _seed_reviews(db, checkpoints, plan['cases'], notify)
        archive_receipt = import_archive(db, archive)
        report, summary, events, workflows, requests = _portfolio_state(db, checkpoints)
        by_line = {case['line_id']: case for case in summary['cases']}
        for case in plan['cases']:
            current = by_line[case['line_id']]
            expected_bucket = {'APPROVED_FOR_DRAFT': 'approved', 'AWAITING_APPROVAL': 'approval', 'NEEDS_REVIEW': 'evidence'}[case['target']]
            _require(current['bucket'] == expected_bucket, 'Actual saved review state differs from the synthetic cohort plan')
        for line, identity in original_approvals.items():
            current = by_line[line]
            _require(current['current_decision'] == 'APPROVED_FOR_DRAFT'
                     and (current['case_digest'], current['review_event_id']) == identity,
                     'An existing approval lost its original current authority')
        _require(report['realized_recovery'] == '0.00', 'No new realized recovery is allowed')
        with connect(db) as conn, conn:
            _require(metadata(conn, 'manifest') == source_manifest and metadata(conn, 'source_digest') == original_digest,
                     'Original import provenance anchors changed')
            for table, original in original_rows.items():
                present = {_encode(row) for row in _rows(conn, table)}
                _require(all(_encode(row) in present for row in original), 'Original history/source record changed: ' + table)
            for table, rows in plan['tables'].items():
                keys = TABLE_KEYS[table]
                for row in rows:
                    stored = conn.execute('SELECT * FROM ' + table + ' WHERE ' + ' AND '.join(key + '=?' for key in keys),
                        tuple(row[key] for key in keys)).fetchone()
                    _require(stored is not None and dict(stored) == row, 'Synthetic source/import reconciliation failed')
            reconciliation = {'new_row_counts': {table: len(rows) for table, rows in plan['tables'].items()},
                'new_row_digests': {table: hashlib.sha256(_encode(sorted(rows, key=_encode)).encode()).hexdigest()
                                   for table, rows in plan['tables'].items()},
                'new_invoice_total_usd': format(sum((Decimal(row['billed_amount']) for row in plan['tables']['invoice_lines']), Decimal('0')), '.2f'),
                'new_settled_payment_total_usd': format(sum((Decimal(row['amount']) for row in plan['tables']['payments']), Decimal('0')), '.2f'),
                'source_hashes': hashes, 'matched': True}
            profile = {'synthetic': True, 'profile_id': PROFILE_ID, 'as_of': AS_OF,
                'description': 'Expanded invented synthetic portfolio; current 2026 workflow cases and a separately locked legacy archive.',
                'current_cohort': {'cases': len(plan['cases']), 'vendors': plan['vendor_count'], **plan['targets']},
                'reconciliation': reconciliation, 'archive': archive_receipt,
                'original_import_anchors_preserved': True, 'term_preparation': 'manual_synthetic_fixture',
                'new_realized_recovery': '0.00'}
            conn.execute('INSERT INTO metadata VALUES (?,?)', ('demo_portfolio', _encode(profile)))
            validate_source_projections(conn)
            _require(conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Final SQLite integrity check failed')
            _require(not conn.execute('PRAGMA foreign_key_check').fetchall(), 'Final foreign-key check failed')
        _require(database_fingerprints(source) == before['database']
                 and database_fingerprints(source_cp) == before['checkpoints'],
                 'Source data changed while building; new profile has not been published')
        result = {'status': 'PASSED', 'profile_id': PROFILE_ID,
            'database': str(destinations[0]), 'checkpoints': str(destinations[1]), 'current_invoice_lines': report['case_count'],
            'new_case_cohort': plan['targets'], 'current_counts': summary['counts'],
            'candidate_total_usd': report['candidate_total'], 'approved_for_draft_usd': summary['totals']['approved_draft'],
            'new_realized_recovery': '0.00', 'expected_cash_recovery': None,
            'original_current_approvals_preserved': len(original_approvals), 'original_review_events_preserved': len(original_rows['review_events']),
            'case_review_events': len(events), 'new_review_events': len(events) - len(original_rows['review_events']),
            'new_request_count': sum(request['line_id'].startswith('SCALE-2026-') for request in requests),
            'real_pending_workflow_count': sum(workflow['next'] == ['approval'] and workflow['state']['line_id'].startswith('SCALE-2026-') for workflow in workflows),
            'archive': archive_receipt, 'source_unchanged': True, 'model_calls': 0,
            'source_fingerprints': before, 'reconciliation': reconciliation,
            'workflow_receipts': receipts, 'elapsed_seconds': round(time.perf_counter() - started, 3)}
        # The profile marker is published last. The launcher must select only a
        # completed profile; an interrupted build never replaces existing files.
        _require(not any(path.exists() for path in destinations), 'Scaled output appeared during construction')
        _copy_database(db, destinations[0])
        _copy_database(checkpoints, destinations[1])
        destinations[2].write_text(json.dumps(profile, indent=2) + '\n')
        (output / 'generator-result.json').write_text(json.dumps(result, indent=2) + '\n')
        notify({'phase': 'complete_profile_published', 'cases': report['case_count'], 'elapsed_seconds': result['elapsed_seconds']})
        return result
