"""Synthetic historical receipts freeze cash once and leave current facts alone."""
from __future__ import annotations

import copy
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from recovery import forecast
from recovery.demo_portfolio import historical_portfolio_payload
from recovery.engine import audit, evaluate_line
from recovery.historical_recovery import get_recovery_overview, save_historical_realization
from recovery.legacy_archive import import_archive
from recovery.review import record_review
from recovery.storage import TABLES, connect, load_snapshot


ROOT = Path(__file__).resolve().parents[1]


class HistoricalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.db = self.directory / 'synthetic.sqlite3'
        self.cp = self.directory / 'absent-checkpoints.sqlite3'
        load_snapshot(self.db, ROOT / 'data')
        with connect(self.db) as conn:
            case = evaluate_line(conn, 'DEMO', 'L004')
            record_review(conn, event_id='approved:review', thread_id='approved', tenant='DEMO',
                line_id='L004', decision='approve', reviewer='Synthetic reviewer',
                note='Reviewed isolated synthetic fixture.', case_digest=case['case_digest'])
        self.archive = historical_portfolio_payload(vendor_count=2, cases_per_year=2)
        import_archive(self.db, self.archive)

    def snapshot(self, names):
        with connect(self.db) as conn:
            return {name: [tuple(row) for row in conn.execute(f'SELECT * FROM {name}')] for name in names}

    def forecast(self, percentage='80', key='forecast:1', scope='all_approved'):
        basis = forecast.get_forecast(self.db, self.cp, scope=scope)['basis']
        return forecast.save_forecast(self.db, self.cp, scope=scope,
            expected_basis_digest=basis['basis_digest'], recovery_percentage=percentage,
            reason='User-selected synthetic demo rate.', reviewer='Synthetic reviewer',
            as_of='2026-10-18', horizon_date='2026-12-31', idempotency_key=key)

    def realize(self, **changes):
        basis = forecast.get_forecast(self.db, self.cp, scope='all_approved')['basis']
        values = {'expected_basis_digest': basis['basis_digest'], 'reviewer': 'Synthetic reviewer',
            'reason': 'User explicitly chose invented historical receipts at 100% of the 80% cash forecast.',
            'as_of': '2026-10-18', 'idempotency_key': 'receipt:1', 'synthetic_confirmed': True}
        return save_historical_realization(self.db, self.cp, **{**values, **changes})

    def test_read_is_side_effect_free_and_missing_forecast_is_not_estimated(self):
        before = self.snapshot((*TABLES, 'metadata', 'review_events', 'legacy_cases', 'legacy_archive_batches'))
        with connect(self.db) as conn:
            schema_before = list(conn.execute('SELECT name,sql FROM sqlite_master ORDER BY name'))
        result = get_recovery_overview(self.db, self.cp)
        self.assertEqual(result['total_approved_usd'], '4600.00')
        self.assertEqual(result['outstanding_approved_usd'], '4600.00')
        self.assertIsNone(result['estimated_cash_usd'])
        self.assertEqual(result['realized_recovery_usd'], '0.00')
        self.assertEqual(result['historical_receipts']['case_count'], 0)
        self.assertEqual(self.snapshot(before), before)
        with connect(self.db) as conn:
            self.assertEqual(list(conn.execute('SELECT name,sql FROM sqlite_master ORDER BY name')), schema_before)
        self.assertFalse(self.cp.exists())

    def test_historical_forecast_collected_once_and_excluded_fully_from_outstanding(self):
        self.forecast(scope='current')
        saved = self.forecast()
        tables = (*TABLES, 'metadata', 'review_events', 'legacy_cases', 'legacy_suppliers',
            'legacy_contracts', 'legacy_archive_batches', 'forecast_assumptions', 'forecast_all_approved_assumptions')
        original = self.snapshot(tables)
        receipt = self.realize()
        self.assertEqual(receipt['realization_percentage_of_frozen_forecast'], '100')
        self.assertEqual(receipt['recovery_percentage'], '80')
        self.assertEqual(receipt['case_count'], 4)
        self.assertEqual(receipt['realized_recovery_usd'], '3200.00')
        self.assertEqual(receipt['forecast_id'], saved['assumption']['forecast_id'])
        result = get_recovery_overview(self.db, self.cp)
        self.assertEqual(result['total_approved_usd'], '4600.00')
        self.assertEqual(result['outstanding_approved_usd'], '600.00')
        self.assertEqual(result['estimated_cash_usd'], '480.00')
        self.assertEqual(result['realized_recovery_usd'], '3200.00')
        self.assertEqual(result['current_realized_recovery_usd'], '0.00')
        self.assertEqual(result['historical_receipts']['receipt_count'], 1)
        for year in ('2024', '2025'):
            self.assertEqual(result['historical_receipts']['by_year'][year],
                {'case_count': 2, 'approved_amount_usd': '2000.00', 'realized_recovery_usd': '1600.00'})
        self.assertEqual(self.realize(), receipt)
        self.assertEqual(self.snapshot(tables), original)
        self.assertEqual(forecast.get_forecast(self.db, self.cp, scope='all_approved'), saved)
        with connect(self.db) as conn:
            result = audit(conn)
            self.assertEqual(result['candidate_total'], '80707.50')
            self.assertEqual(result['realized_recovery'], '0.00')
            for row in conn.execute('SELECT * FROM historical_recovery_items'):
                self.assertEqual(row['effective_date'], str(row['origin_year']) + '-12-31')
                self.assertEqual(row['realized_amount_usd'], '800.00')
        self.assertFalse(self.cp.exists())

    def test_prior_current_forecast_cannot_authorize_historical_receipts(self):
        self.forecast(scope='current')
        with self.assertRaisesRegex(ValueError, 'current saved all-approved forecast'):
            self.realize()
        with connect(self.db) as conn:
            self.assertFalse(conn.execute("SELECT 1 FROM sqlite_master WHERE name='historical_recovery_receipts'").fetchone())

    def test_receipts_freeze_original_rate_but_remaining_estimate_uses_new_rate_once(self):
        self.forecast()
        receipt = self.realize()
        self.forecast('90', key='forecast:2')
        result = get_recovery_overview(self.db, self.cp)
        self.assertEqual(result['estimated_cash_usd'], '540.00')
        self.assertEqual(result['realized_recovery_usd'], '3200.00')
        self.assertEqual(result['historical_receipts']['receipts'], [receipt])
        with self.assertRaisesRegex(ValueError, 'never counted twice'):
            self.realize(idempotency_key='receipt:2')
        self.assertEqual(self.realize(), receipt)

    def test_invalid_review_payloads_and_zero_forecasts_leave_no_receipt_schema(self):
        self.forecast()
        for changes, message in (({'synthetic_confirmed': False}, 'SYNTHETIC_ONLY'),
                ({'synthetic_confirmed': 1}, 'SYNTHETIC_ONLY'), ({'reviewer': ''}, 'nonempty'),
                ({'reason': ''}, 'nonempty'), ({'as_of': '2025-12-31'}, 'evidence snapshot'),
                ({'as_of': '2026-01-50'}, 'valid calendar date'),
                ({'expected_basis_digest': 'a' * 64}, 'STALE_BASIS'),
                ({'tenant': 'NOT-IN-MANIFEST'}, 'synthetic manifest')):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.realize(**changes)
        self.forecast('0', key='forecast:zero')
        with self.assertRaisesRegex(ValueError, 'positive saved forecast percentage'):
            self.realize()
        with connect(self.db) as conn:
            self.assertFalse(conn.execute("SELECT 1 FROM sqlite_master WHERE name='historical_recovery_receipts'").fetchone())

    def test_retry_changed_payload_rejected_and_records_are_immutable(self):
        self.forecast()
        receipt = self.realize()
        with self.assertRaisesRegex(ValueError, 'different review'):
            self.realize(reason='Changed review must use a new key.')
        with connect(self.db) as conn:
            for name in ('historical_recovery_receipts', 'historical_recovery_items'):
                row = conn.execute(f'SELECT * FROM {name} LIMIT 1').fetchone()
                for sql, args in ((f'UPDATE {name} SET tenant_id=tenant_id', ()),
                    (f'DELETE FROM {name}', ()),
                    (f'INSERT OR REPLACE INTO {name} VALUES ({",".join("?" for _ in row)})', tuple(row))):
                    with self.subTest(sql=sql), self.assertRaisesRegex(sqlite3.IntegrityError, 'immutable'):
                        with conn:
                            conn.execute(sql, args)
        self.assertEqual(self.realize(), receipt)

    def test_fractional_cents_are_deterministic_and_match_aggregate_per_year(self):
        more = copy.deepcopy(self.archive)
        more['archive_id'] += '-CENTS'
        for group in ('suppliers', 'contracts', 'cases'):
            for row in more[group]:
                for field in ('supplier_id', 'contract_id', 'line_id', 'invoice_id', 'source_record_id'):
                    if field in row:
                        row[field] += '-CENTS'
                if group == 'cases':
                    row['historical_approved_amount'] = '0.01'
        import_archive(self.db, more)
        self.forecast('33.333333')
        self.realize()
        result = get_recovery_overview(self.db, self.cp)
        # 2000.02 × 33.333333% = 666.6733266666 -> 666.67 for each year.
        for year in ('2024', '2025'):
            self.assertEqual(result['historical_receipts']['by_year'][year]['realized_recovery_usd'], '666.67')
        self.assertEqual(result['realized_recovery_usd'], '1333.34')
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM historical_recovery_items').fetchone()[0], 8)
        self.assertEqual(get_recovery_overview(self.db, self.cp), result)

    def test_transaction_rolls_back_header_and_items_if_insertion_fails(self):
        self.forecast()
        with patch('recovery.historical_recovery._item', side_effect=RuntimeError('injected save failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected save failure'):
                self.realize()
        with connect(self.db) as conn:
            self.assertFalse(conn.execute("SELECT 1 FROM sqlite_master WHERE name='historical_recovery_receipts'").fetchone())
        self.assertEqual(get_recovery_overview(self.db, self.cp)['realized_recovery_usd'], '0.00')

    def test_tampered_item_is_withheld_even_if_update_guard_was_removed(self):
        self.forecast()
        self.realize()
        with connect(self.db) as conn, conn:
            conn.execute('DROP TRIGGER historical_recovery_items_no_update')
            conn.execute("UPDATE historical_recovery_items SET realized_amount_usd='999.99'")
        with self.assertRaisesRegex(ValueError, 'item hash or amount'):
            get_recovery_overview(self.db, self.cp)

    def test_tampered_import_batch_cannot_enter_the_historical_receipt_ledger(self):
        self.forecast()
        with connect(self.db) as conn, conn:
            conn.execute('DROP TRIGGER legacy_archive_batches_immutable_update')
            conn.execute("UPDATE legacy_archive_batches SET payload_sha256=?", ('b' * 64,))
        with self.assertRaisesRegex(ValueError, 'archive payload hash'):
            get_recovery_overview(self.db, self.cp)
        with self.assertRaisesRegex(ValueError, 'archive payload hash'):
            self.realize()


if __name__ == '__main__':
    unittest.main()
