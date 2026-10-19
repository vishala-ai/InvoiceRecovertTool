"""All-year reporting must not expand current claims or reuse an old forecast."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from recovery.forecast import (POLICY_ALL_APPROVED, POLICY_VERSION, basis_from_summary,
    compute_forecast, get_forecast, portfolio_basis, save_forecast)
from recovery.legacy_archive import import_archive
from recovery.storage import TABLES, connect, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
try:
    GRAPH_AVAILABLE = importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
except ModuleNotFoundError:
    GRAPH_AVAILABLE = False


def reporting_archive():
    value = {'schema_version': 1, 'synthetic': True, 'archive_id': 'LEGACY-GRAND-TOTAL-SAFETY',
        'source_system': 'Old AP (synthetic)', 'provenance': 'Synthetic migrated approval; reporting only.',
        'suppliers': [], 'contracts': [], 'cases': []}
    for tenant in ('DEMO', 'OTHER'):
        value['suppliers'].append({'tenant_id': tenant, 'supplier_id': 'LEGACY-VENDOR',
                                    'name': 'Invented migrated vendor'})
        for year in (2024, 2025):
            value['contracts'].append({'tenant_id': tenant, 'supplier_id': 'LEGACY-VENDOR',
                'contract_id': f'LEGACY-CONTRACT-{year}', 'title': f'Invented {year} agreement',
                'origin_year': year})
    for tenant, year, suffix, currency, amount in (
            ('DEMO', 2024, 'USD-A', 'USD', '100.05'),
            ('DEMO', 2025, 'USD-B', 'USD', '200.05'),
            ('DEMO', 2024, 'EUR', 'EUR', '999.99'),
            ('OTHER', 2024, 'CANARY', 'USD', '9999.99')):
        value['cases'].append({'tenant_id': tenant, 'line_id': f'LEGACY-{year}-{suffix}',
            'supplier_id': 'LEGACY-VENDOR', 'contract_id': f'LEGACY-CONTRACT-{year}',
            'invoice_id': f'LEGACY-INVOICE-{year}-{suffix}', 'invoice_date': f'{year}-07-01',
            'origin_year': year, 'currency': currency, 'historical_approved_amount': amount,
            'source_record_id': f'OLD-{year}-{suffix}', 'description': 'Invented archived approval.'})
    return value


def approved_case(line='L004', **changes):
    return {'tenant_id': 'DEMO', 'line_id': line, 'supplier_id': 'SUP-A', 'contract_id': 'CTR-A',
        'status': 'CANDIDATE', 'currency': 'USD', 'bucket': 'approved',
        'current_decision': 'APPROVED_FOR_DRAFT', 'case_digest': 'a' * 64,
        'review_event_id': line + ':review', 'assessment_id': 'assessment:' + line,
        'workflow_thread_id': 'thread:' + line, 'net_candidate': '600.00',
        'known_candidate': '600.00', 'blocking_request_count': 0, **changes}


class GrandTotalForecastSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.db, self.cp = self.root / 'synthetic.sqlite3', self.root / 'checkpoints.sqlite3'
        load_snapshot(self.db, ROOT / 'data')

    def records(self, tables):
        with connect(self.db) as conn:
            return {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table}')]
                    for table in tables}

    def archive_records(self):
        return self.records(('legacy_archive_batches', 'legacy_suppliers', 'legacy_contracts', 'legacy_cases'))

    def save(self, *, scope, key='same-human-action', **changes):
        values = {'expected_basis_digest': get_forecast(self.db, self.cp, scope=scope)['basis']['basis_digest'],
            'recovery_percentage': '80', 'reason': 'Explicit all-year synthetic reporting assumption.',
            'reviewer': 'Synthetic finance reviewer', 'as_of': '2026-10-18',
            'horizon_date': '2026-12-31', 'idempotency_key': key, 'scope': scope, **changes}
        return save_forecast(self.db, self.cp, **values)

    def approve_current(self, number=1):
        from recovery.lifecycle import enabled, migrate
        from recovery.workflow import run_persistent

        if not enabled(self.db):
            migrate(self.db, backup_dir=self.root / 'backups')
        thread = f'grand-total-safety:{number}'
        run_persistent(self.db, self.cp, thread, line_id='L004', use_llm=False)
        run_persistent(self.db, self.cp, thread, decision={'decision': 'approve',
            'reviewer': 'Synthetic accountant', 'note': 'Reviewed this current synthetic case and its offsets.'})

    def test_all_year_basis_excludes_nonusd_other_tenant_and_stale_current_without_double_count(self):
        import_archive(self.db, reporting_archive())
        original = {'cases': [approved_case(),
            approved_case('STALE', validity='STALE'),
            approved_case('EVIDENCE', blocking_request_count=1),
            approved_case('PENDING', bucket='approval', current_decision=None),
            approved_case('EUR-CURRENT', currency='EUR'),
            approved_case('CANARY-CURRENT', tenant_id='OTHER'),
            # Even a caller misclassifying an archived line cannot add it a
            # second time through the current-approved branch.
            approved_case('LEGACY-2024-USD-A')]}
        snapshot = copy.deepcopy(original)
        current = basis_from_summary(original)
        combined = portfolio_basis(self.db, original)
        self.assertEqual(original, snapshot)
        self.assertEqual(current['amount_usd'], '600.00')
        self.assertEqual(current['case_count'], 1)
        self.assertEqual(combined['amount_usd'], '900.10')
        self.assertEqual(combined['case_count'], 3)
        self.assertEqual(combined['policy_version'], POLICY_ALL_APPROVED)
        self.assertEqual(current['policy_version'], POLICY_VERSION)
        self.assertNotEqual(current['basis_digest'], combined['basis_digest'])
        self.assertEqual({row['basis_kind'] for row in combined['cases']},
                         {'CURRENT_APPROVED_DRAFT', 'MIGRATED_APPROVED'})
        self.assertEqual(compute_forecast(combined, '80'), '720.08')
        self.assertEqual(portfolio_basis(self.db, original, tenant='OTHER')['amount_usd'], '10599.99')

    def test_archive_source_identity_and_origin_are_preserved_without_new_claim_identity(self):
        import_archive(self.db, reporting_archive())
        before = self.archive_records()
        combined = portfolio_basis(self.db, {'cases': []})
        self.assertEqual(combined['amount_usd'], '300.10')
        for row in combined['cases']:
            self.assertEqual(row['status'], 'MIGRATED_APPROVED')
            self.assertIs(row['locked'], True)
            self.assertIs(row['synthetic'], True)
            self.assertIn(row['origin_year'], (2024, 2025))
            self.assertEqual(row['archive_id'], 'LEGACY-GRAND-TOTAL-SAFETY')
            self.assertEqual(len(row['case_sha256']), 64)
            self.assertEqual(len(row['archive_payload_sha256']), 64)
            self.assertIn('reporting only', row['provenance'])
            self.assertNotIn('review_event_id', row)
            self.assertNotIn('assessment_id', row)
        self.assertEqual(self.archive_records(), before)

    def test_corrupt_archive_source_hash_withholds_the_broader_basis(self):
        import_archive(self.db, reporting_archive())
        with connect(self.db) as conn, conn:
            row = dict(conn.execute("SELECT * FROM legacy_cases WHERE tenant_id='DEMO' AND currency='USD' LIMIT 1").fetchone())
            row.update(line_id='LEGACY-CORRUPT', source_record_id='OLD-CORRUPT')
            # A new directly inserted row reusing a different record's hash is
            # not an authentic imported source. Existing archives remain locked.
            conn.execute(f'INSERT INTO legacy_cases ({",".join(row)}) '
                f'VALUES ({",".join("?" for _ in row)})', tuple(row.values()))
        with self.assertRaisesRegex(ValueError, 'source hash'):
            portfolio_basis(self.db, {'cases': []})

    def test_archive_only_reporting_forecast_does_not_create_current_claims_or_checkpoints(self):
        from recovery.engine import audit

        import_archive(self.db, reporting_archive())
        sources = self.records((*TABLES, 'metadata', 'review_events'))
        archive = self.archive_records()
        with patch('recovery.forecast.basis_from_summary', wraps=basis_from_summary):
            report = self.save(scope='all_approved')
        self.assertEqual(report['scope'], 'all_approved')
        self.assertEqual(report['basis']['amount_usd'], '300.10')
        self.assertEqual(report['projected_cash_usd'], '240.08')
        self.assertEqual(report['realized_recovery'], '0.00')
        self.assertEqual(get_forecast(self.db, self.cp)['status'], 'NOT_SET')
        self.assertFalse(self.cp.exists())
        self.assertEqual(self.archive_records(), archive)
        self.assertEqual(self.records((*TABLES, 'metadata', 'review_events')), sources)
        with connect(self.db) as conn:
            result = audit(conn)
        self.assertEqual(result['case_count'], 20)
        self.assertEqual(result['candidate_total'], '80707.50')
        self.assertEqual(result['realized_recovery'], '0.00')

    @unittest.skipUnless(GRAPH_AVAILABLE, 'Install graph extra for actual current-approval isolation')
    def test_saved_current_forecast_never_expands_implicitly_and_explicit_all_save_is_separate(self):
        self.approve_current()
        original = self.save(scope='current')
        self.assertEqual(original['projected_cash_usd'], '480.00')
        old_rows = self.records(('forecast_assumptions',))
        import_archive(self.db, reporting_archive())
        archive = self.archive_records()
        sources = self.records((*TABLES, 'metadata', 'review_events', 'assessment_revisions'))
        all_report = get_forecast(self.db, self.cp, scope='all_approved')
        self.assertEqual(all_report['status'], 'NOT_SET')
        self.assertIsNone(all_report['projected_cash_usd'])
        self.assertEqual(all_report['basis']['amount_usd'], '900.10')
        self.assertEqual(get_forecast(self.db, self.cp)['projected_cash_usd'], '480.00')
        with self.assertRaisesRegex(ValueError, 'STALE_BASIS'):
            self.save(scope='all_approved', expected_basis_digest=original['basis']['basis_digest'])
        explicit = self.save(scope='all_approved')  # same key string, separate reporting scope
        self.assertEqual(explicit['projected_cash_usd'], '720.08')
        self.assertEqual(explicit['assumption']['recovery_percentage'], '80')
        self.assertNotEqual(explicit['assumption']['forecast_id'], original['assumption']['forecast_id'])
        self.assertEqual(self.records(('forecast_assumptions',)), old_rows)
        self.assertEqual(get_forecast(self.db, self.cp)['assumption']['forecast_id'],
                         original['assumption']['forecast_id'])
        self.assertEqual(self.archive_records(), archive)
        self.assertEqual(self.records((*TABLES, 'metadata', 'review_events', 'assessment_revisions')), sources)

    @unittest.skipUnless(GRAPH_AVAILABLE, 'Install graph extra for actual approval revision changes')
    def test_same_dollar_new_current_approval_invalidates_only_the_saved_basis(self):
        self.approve_current()
        import_archive(self.db, reporting_archive())
        original = self.save(scope='all_approved')
        archive = self.archive_records()
        self.approve_current(2)
        changed = get_forecast(self.db, self.cp, scope='all_approved')
        self.assertEqual(changed['basis']['amount_usd'], '900.10')
        self.assertEqual(changed['status'], 'STALE_BASIS')
        self.assertIsNone(changed['projected_cash_usd'])
        replay = self.save(scope='all_approved', expected_basis_digest=original['basis']['basis_digest'])
        self.assertEqual(replay['status'], 'STALE_BASIS')
        self.assertIsNone(replay['projected_cash_usd'])
        self.assertEqual(replay['assumption']['forecast_id'], original['assumption']['forecast_id'])
        refreshed = self.save(scope='all_approved', key='explicit-new-basis')
        self.assertEqual(refreshed['projected_cash_usd'], '720.08')
        self.assertEqual(self.archive_records(), archive)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM forecast_all_approved_assumptions').fetchone()[0], 2)

    def test_new_archive_record_stales_all_year_forecast_but_does_not_rewrite_it(self):
        import_archive(self.db, reporting_archive())
        first = self.save(scope='all_approved')
        more = reporting_archive()
        more['archive_id'] += '-ADDED'
        for group in ('suppliers', 'contracts', 'cases'):
            for row in more[group]:
                for field in ('supplier_id', 'contract_id', 'line_id', 'invoice_id', 'source_record_id'):
                    if field in row:
                        row[field] += '-ADDED'
        import_archive(self.db, more)
        stale = get_forecast(self.db, self.cp, scope='all_approved')
        self.assertEqual(stale['status'], 'STALE_BASIS')
        self.assertIsNone(stale['projected_cash_usd'])
        self.assertEqual(stale['basis']['amount_usd'], '600.20')
        self.assertEqual(stale['assumption']['forecast_id'], first['assumption']['forecast_id'])
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM forecast_all_approved_assumptions').fetchone()[0], 1)

    def test_all_scope_assumptions_remain_immutable_and_invalid_scope_creates_nothing(self):
        import_archive(self.db, reporting_archive())
        self.save(scope='all_approved')
        with connect(self.db) as conn:
            row = conn.execute('SELECT * FROM forecast_all_approved_assumptions').fetchone()
            for sql, values in (
                    ('UPDATE forecast_all_approved_assumptions SET recovery_percentage=?', ('99',)),
                    ('DELETE FROM forecast_all_approved_assumptions', ()),
                    (f'INSERT OR REPLACE INTO forecast_all_approved_assumptions VALUES '
                     f'({",".join("?" for _ in row)})', tuple(row))):
                with self.subTest(sql=sql), self.assertRaisesRegex(sqlite3.IntegrityError, 'immutable'):
                    with conn:
                        conn.execute(sql, values)
            before = list(conn.iterdump())
        for scope in ('all', 'all_approved; DROP TABLE legacy_cases', None, []):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                get_forecast(self.db, self.cp, scope=scope)
        with connect(self.db) as conn:
            self.assertEqual(list(conn.iterdump()), before)


if __name__ == '__main__':
    unittest.main()
