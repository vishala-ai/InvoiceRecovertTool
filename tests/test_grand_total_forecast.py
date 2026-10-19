"""Grand-total forecasts retain approval provenance and isolated assumptions."""
from __future__ import annotations

import copy
from pathlib import Path
import sqlite3
import tempfile
import unittest

from recovery import forecast
from recovery.demo_portfolio import historical_portfolio_payload
from recovery.engine import audit, evaluate_line
from recovery.legacy_archive import import_archive
from recovery.review import record_review
from recovery.storage import connect, load_snapshot


ROOT = Path(__file__).resolve().parents[1]


class GrandTotalForecastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.db = self.directory / 'synthetic.sqlite3'
        self.checkpoints = self.directory / 'absent-checkpoints.sqlite3'
        load_snapshot(self.db, ROOT / 'data')
        with connect(self.db) as conn:
            case = evaluate_line(conn, 'DEMO', 'L004')
            record_review(conn, event_id='approved:review', thread_id='approved', tenant='DEMO',
                line_id='L004', decision='approve', reviewer='Synthetic reviewer',
                note='Reviewed isolated fixture.', case_digest=case['case_digest'])
        self.archive = historical_portfolio_payload(vendor_count=2, cases_per_year=2)

    def save(self, scope='current', key='review:1', **changes):
        basis = forecast.get_forecast(self.db, self.checkpoints, scope=scope)['basis']
        values = {'expected_basis_digest': basis['basis_digest'], 'recovery_percentage': '80',
            'reason': 'Explicit uncalibrated synthetic grand-total scenario.',
            'reviewer': 'Synthetic scenario owner', 'as_of': '2026-10-18',
            'horizon_date': '2026-12-31', 'idempotency_key': key}
        values.update(changes)
        return forecast.save_forecast(self.db, self.checkpoints, scope=scope, **values)

    def rows(self, table):
        with connect(self.db) as conn:
            return [tuple(row) for row in conn.execute('SELECT * FROM ' + table)]

    def test_current_assumptions_do_not_silently_expand_and_both_scopes_remain_independent(self):
        current = self.save()
        original_rows = self.rows('forecast_assumptions')
        with connect(self.db) as conn:
            original_schema = conn.execute("SELECT sql FROM sqlite_master WHERE name='forecast_assumptions'").fetchone()[0]
        import_archive(self.db, self.archive)
        grand = forecast.get_forecast(self.db, self.checkpoints, scope='all_approved')
        self.assertEqual(grand['scope'], 'all_approved')
        self.assertEqual(grand['status'], 'NOT_SET')
        self.assertIsNone(grand['projected_cash_usd'])
        self.assertEqual(grand['basis']['amount_usd'], '4600.00')
        self.assertEqual(grand['basis']['case_count'], 5)
        self.assertEqual(grand['basis']['policy_version'], forecast.POLICY_ALL_APPROVED)
        self.assertNotEqual(grand['basis']['basis_digest'], current['basis']['basis_digest'])
        saved = self.save('all_approved')
        self.assertEqual(saved['status'], 'CURRENT')
        self.assertEqual(saved['projected_cash_usd'], '3680.00')
        self.assertEqual(self.save('all_approved'), saved)
        self.assertEqual(forecast.get_forecast(self.db, self.checkpoints), current)
        self.assertEqual(self.rows('forecast_assumptions'), original_rows)
        self.assertEqual(len(self.rows('forecast_all_approved_assumptions')), 1)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT sql FROM sqlite_master WHERE name='forecast_assumptions'").fetchone()[0], original_schema)

    def test_composition_preserves_archive_source_hashes_and_never_creates_new_approvals(self):
        import_archive(self.db, self.archive)
        before = {table: self.rows(table) for table in ('review_events', 'legacy_cases', 'legacy_contracts',
            'legacy_suppliers', 'legacy_archive_batches', 'invoice_lines', 'reviewed_terms')}
        with connect(self.db) as conn:
            financial_before = audit(conn)
            archive_ids = {row['line_id']: dict(row) for row in conn.execute('SELECT * FROM legacy_cases')}
        grand = self.save('all_approved')
        for case in grand['basis']['cases']:
            if case['line_id'] in archive_ids:
                source = archive_ids[case['line_id']]
                self.assertEqual(case['archive_id'], source['archive_id'])
                self.assertEqual(case['case_sha256'], source['case_sha256'])
                self.assertEqual(case['source_record_id'], source['source_record_id'])
                self.assertEqual(case['amount_usd'], source['historical_approved_amount'])
                self.assertEqual(case['status'], 'MIGRATED_APPROVED')
                self.assertTrue(case['locked'])
                self.assertNotIn('review_event_id', case)
        for table, rows in before.items():
            self.assertEqual(self.rows(table), rows, table)
        with connect(self.db) as conn:
            self.assertEqual(audit(conn), financial_before)
            self.assertEqual(financial_before['candidate_total'], '80707.50')
            self.assertEqual(financial_before['realized_recovery'], '0.00')
        self.assertFalse(self.checkpoints.exists())

    def test_new_archive_approval_invalidates_only_grand_scope_and_requires_explicit_resave(self):
        import_archive(self.db, self.archive)
        current = self.save()
        original = self.save('all_approved')
        additional = historical_portfolio_payload(vendor_count=1, cases_per_year=1)
        additional['archive_id'] = 'LEGACY-SECOND-v1'
        for group in ('suppliers', 'contracts', 'cases'):
            for row in additional[group]:
                for field, value in row.items():
                    if isinstance(value, str) and value.startswith('LEGACY-'):
                        row[field] = value.replace('LEGACY-', 'LEGACY-SECOND-', 1)
        import_archive(self.db, additional)
        changed = forecast.get_forecast(self.db, self.checkpoints, scope='all_approved')
        self.assertEqual(changed['status'], 'STALE_BASIS')
        self.assertIsNone(changed['projected_cash_usd'])
        self.assertEqual(changed['basis']['amount_usd'], '6600.00')
        self.assertEqual(forecast.get_forecast(self.db, self.checkpoints), current)
        with self.assertRaisesRegex(ValueError, 'STALE_BASIS'):
            self.save('all_approved', key='stale-attempt', expected_basis_digest=original['basis']['basis_digest'])
        replay = self.save('all_approved', expected_basis_digest=original['basis']['basis_digest'])
        self.assertEqual(replay['status'], 'STALE_BASIS')
        self.assertEqual(replay['assumption']['forecast_id'], original['assumption']['forecast_id'])
        refreshed = self.save('all_approved', key='review:2')
        self.assertEqual(refreshed['projected_cash_usd'], '5280.00')
        self.assertEqual(len(self.rows('forecast_all_approved_assumptions')), 2)

    def test_scope_is_explicit_even_without_archive_and_invalid_scope_creates_no_schema(self):
        current = self.save()
        grand = forecast.get_forecast(self.db, self.checkpoints, scope='all_approved')
        self.assertEqual(grand['basis']['amount_usd'], current['basis']['amount_usd'])
        self.assertNotEqual(grand['basis']['basis_digest'], current['basis']['basis_digest'])
        self.assertEqual(grand['status'], 'NOT_SET')
        for scope in ('historical', '', None, False, ['all_approved']):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, 'Forecast scope'):
                forecast.get_forecast(self.db, self.checkpoints, scope=scope)
        with connect(self.db) as conn:
            self.assertFalse(conn.execute("SELECT 1 FROM sqlite_master WHERE name='forecast_all_approved_assumptions'").fetchone())

    def test_grand_assumption_is_immutable_and_current_display_digest_cannot_authorize_it(self):
        import_archive(self.db, self.archive)
        current = forecast.get_forecast(self.db, self.checkpoints)
        with self.assertRaisesRegex(ValueError, 'STALE_BASIS'):
            self.save('all_approved', expected_basis_digest=current['basis']['basis_digest'])
        saved = self.save('all_approved')
        with connect(self.db) as conn:
            row = conn.execute('SELECT * FROM forecast_all_approved_assumptions').fetchone()
            for sql, args in (
                ("UPDATE forecast_all_approved_assumptions SET reason='changed'", ()),
                ('DELETE FROM forecast_all_approved_assumptions', ()),
                ('INSERT OR REPLACE INTO forecast_all_approved_assumptions VALUES (' + ','.join('?' for _ in row) + ')', tuple(row))):
                with self.subTest(sql=sql), self.assertRaisesRegex(sqlite3.IntegrityError, 'immutable'):
                    conn.execute(sql, args)
                conn.rollback()
        self.assertEqual(forecast.get_forecast(self.db, self.checkpoints, scope='all_approved'), saved)

    def test_portfolio_basis_ignores_non_usd_and_is_input_order_independent(self):
        archive = copy.deepcopy(self.archive)
        archive['cases'][0]['currency'] = 'EUR'
        import_archive(self.db, archive)
        from recovery.invoice_review import load_worklist
        summary = load_worklist(self.db, self.checkpoints)['summary']
        basis = forecast.portfolio_basis(self.db, summary)
        self.assertEqual(basis['amount_usd'], '3600.00')
        self.assertEqual(basis['case_count'], 4)
        self.assertEqual(forecast.portfolio_basis(self.db, {**summary, 'cases': list(reversed(summary['cases']))}), basis)
        self.assertEqual(forecast.compute_forecast(basis, '80'), '2880.00')
