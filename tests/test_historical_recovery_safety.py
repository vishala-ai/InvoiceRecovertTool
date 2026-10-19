"""Independent regression checks for the invented historical receipt overlay."""
from __future__ import annotations

import copy
from pathlib import Path
import sqlite3
import tempfile
import unittest

from recovery import forecast
from recovery.demo_portfolio import historical_portfolio_payload
from recovery.engine import audit, evaluate_line
from recovery.historical_recovery import get_recovery_overview, save_historical_realization
from recovery.legacy_archive import import_archive
from recovery.review import record_review
from recovery.storage import connect, load_snapshot


ROOT = Path(__file__).resolve().parents[1]


class HistoricalRecoverySafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.db, self.cp = self.root / 'synthetic.sqlite3', self.root / 'absent-checkpoints.sqlite3'
        load_snapshot(self.db, ROOT / 'data')
        self.approve('L004', 'initial')
        self.archive = historical_portfolio_payload(vendor_count=2, cases_per_year=2)
        import_archive(self.db, self.archive)

    def approve(self, line, identity):
        with connect(self.db) as conn:
            case = evaluate_line(conn, 'DEMO', line)
            record_review(conn, event_id=identity + ':review', thread_id=identity,
                tenant='DEMO', line_id=line, decision='approve', reviewer='Synthetic accountant',
                note='Independent historical reporting regression fixture.', case_digest=case['case_digest'])

    def rows(self):
        with connect(self.db) as conn:
            names = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            return {name: [tuple(row) for row in conn.execute('SELECT * FROM "' + name + '"')]
                    for name in names}

    def save_forecast(self, rate='80', scope='all_approved', key='forecast:first'):
        basis = forecast.get_forecast(self.db, self.cp, scope=scope)['basis']
        return forecast.save_forecast(self.db, self.cp, scope=scope,
            expected_basis_digest=basis['basis_digest'], recovery_percentage=rate,
            reason='Explicit uncalibrated synthetic collection scenario.', reviewer='Synthetic owner',
            as_of='2026-10-18', horizon_date='2026-12-31', idempotency_key=key)

    def realize(self, **changes):
        values = {'expected_basis_digest': forecast.get_forecast(
                self.db, self.cp, scope='all_approved')['basis']['basis_digest'],
            'reviewer': 'Synthetic archive receipt reviewer',
            'reason': 'Invented historical receipts at 100% of the frozen cash forecast.',
            'as_of': '2026-10-18', 'idempotency_key': 'receipt:first', 'synthetic_confirmed': True,
            **changes}
        return save_historical_realization(self.db, self.cp, **values)

    def test_received_forecast_moves_once_without_reducing_the_approved_total(self):
        frozen = self.save_forecast()
        self.assertEqual(frozen['projected_cash_usd'], '3680.00')
        receipt = self.realize()
        report = get_recovery_overview(self.db, self.cp)
        self.assertEqual(receipt['realization_percentage_of_frozen_forecast'], '100')
        self.assertEqual(receipt['recovery_percentage'], '80')
        self.assertEqual(report['total_approved_usd'], '4600.00')
        self.assertEqual(report['outstanding_approved_usd'], '600.00')
        self.assertEqual(report['estimated_cash_usd'], '480.00')
        self.assertEqual(report['realized_recovery_usd'], '3200.00')
        self.assertEqual(report['historical_receipts']['case_count'], 4)
        self.assertEqual(report['historical_receipts']['by_year'], {
            '2024': {'case_count': 2, 'approved_amount_usd': '2000.00', 'realized_recovery_usd': '1600.00'},
            '2025': {'case_count': 2, 'approved_amount_usd': '2000.00', 'realized_recovery_usd': '1600.00'}})
        self.assertEqual(forecast.get_forecast(self.db, self.cp, scope='all_approved'), frozen)

    def test_later_ninety_percent_forecast_cannot_reprice_received_cash(self):
        first = self.save_forecast()
        receipt = self.realize()
        receipt_rows = {name: rows for name, rows in self.rows().items() if name.startswith('historical_recovery_')}
        updated = self.save_forecast('90', key='forecast:ninety')
        report = get_recovery_overview(self.db, self.cp)
        self.assertEqual(report['forecast']['assumption']['forecast_id'], updated['assumption']['forecast_id'])
        self.assertEqual(report['estimated_cash_usd'], '540.00')
        self.assertEqual(report['realized_recovery_usd'], '3200.00')
        self.assertEqual(report['historical_receipts']['receipts'][0], receipt)
        self.assertEqual(receipt['forecast_id'], first['assumption']['forecast_id'])
        self.assertEqual({name: rows for name, rows in self.rows().items()
                          if name.startswith('historical_recovery_')}, receipt_rows)
        # Retrying the original authorized receipt remains tied to 80% even
        # though the latest planning assumption now uses 90%.
        self.assertEqual(self.realize(), receipt)

    def test_stale_forecast_withholds_remaining_estimate_but_keeps_received_cash(self):
        self.save_forecast()
        receipt = self.realize()
        self.approve('L011', 'additional-approval')
        report = get_recovery_overview(self.db, self.cp)
        self.assertEqual(report['forecast']['status'], 'STALE_BASIS')
        self.assertEqual(report['total_approved_usd'], '4700.00')
        self.assertEqual(report['outstanding_approved_usd'], '700.00')
        self.assertIsNone(report['estimated_cash_usd'])
        self.assertEqual(report['realized_recovery_usd'], '3200.00')
        self.assertEqual(report['historical_receipts']['receipts'][0], receipt)
        self.save_forecast('90', key='forecast:refreshed')
        self.assertEqual(get_recovery_overview(self.db, self.cp)['estimated_cash_usd'], '630.00')

    def test_original_forecasts_archives_journal_and_engine_remain_unchanged(self):
        current = self.save_forecast(scope='current', key='forecast:current')
        all_years = self.save_forecast()
        before = self.rows()
        with connect(self.db) as conn:
            financial = audit(conn)
        self.realize()
        report = get_recovery_overview(self.db, self.cp)
        after = self.rows()
        self.assertEqual({name: after[name] for name in before}, before)
        self.assertEqual(set(after) - set(before), {'historical_recovery_receipts', 'historical_recovery_items'})
        self.assertEqual(forecast.get_forecast(self.db, self.cp), current)
        self.assertEqual(forecast.get_forecast(self.db, self.cp, scope='all_approved'), all_years)
        self.assertEqual(current['realized_recovery'], '0.00')
        self.assertEqual(all_years['realized_recovery'], '0.00')
        self.assertEqual(report['current_realized_recovery_usd'], '0.00')
        self.assertFalse(self.cp.exists())
        with connect(self.db) as conn:
            self.assertEqual(audit(conn), financial)
        self.assertEqual(financial['candidate_total'], '80707.50')
        self.assertEqual(financial['realized_recovery'], '0.00')

    def test_duplicate_review_cannot_create_a_second_receipt_or_reopen_closed_amounts(self):
        self.save_forecast()
        receipt = self.realize()
        before = self.rows()
        self.assertEqual(self.realize(), receipt)
        with self.assertRaisesRegex(ValueError, 'different review'):
            self.realize(reason='Changed payload under an already used review identity.')
        with self.assertRaisesRegex(ValueError, 'never counted twice'):
            self.realize(idempotency_key='receipt:another-click')
        self.assertEqual(self.rows(), before)
        with connect(self.db) as conn:
            for table in ('historical_recovery_receipts', 'historical_recovery_items'):
                row = conn.execute('SELECT * FROM ' + table + ' LIMIT 1').fetchone()
                for sql in ('DELETE FROM ' + table, 'UPDATE ' + table + " SET tenant_id='OTHER'",
                            'INSERT OR REPLACE INTO ' + table + ' VALUES (' + ','.join('?' for _ in row) + ')'):
                    values = tuple(row) if sql.startswith('INSERT') else ()
                    with self.subTest(table=table, sql=sql), self.assertRaisesRegex(sqlite3.IntegrityError, 'immutable'):
                        with conn:
                            conn.execute(sql, values)
        self.assertEqual(self.rows(), before)

    def test_zero_future_forecast_does_not_erase_previously_received_cash(self):
        self.save_forecast()
        self.realize()
        self.save_forecast('0', key='forecast:zero')
        report = get_recovery_overview(self.db, self.cp)
        self.assertEqual(report['estimated_cash_usd'], '0.00')
        self.assertEqual(report['realized_recovery_usd'], '3200.00')
        self.assertEqual(report['outstanding_approved_usd'], '600.00')

    def test_other_tenant_and_currency_do_not_enter_the_usd_receipt_rollup(self):
        more = copy.deepcopy(self.archive)
        more['archive_id'] += '-CANARIES'
        for group in ('suppliers', 'contracts', 'cases'):
            for row in more[group]:
                row['tenant_id'] = 'OTHER'
                for field in ('supplier_id', 'contract_id', 'line_id', 'invoice_id', 'source_record_id'):
                    if field in row:
                        row[field] += '-CANARY'
        # A DEMO EUR record is also irrelevant to the USD metric.
        for group in ('suppliers', 'contracts', 'cases'):
            copies = copy.deepcopy(more[group])
            for row in copies:
                row['tenant_id'] = 'DEMO'
                if group == 'cases':
                    row['currency'] = 'EUR'
            more[group].extend(copies)
        import_archive(self.db, more)
        self.save_forecast()
        self.realize()
        report = get_recovery_overview(self.db, self.cp)
        self.assertEqual(report['total_approved_usd'], '4600.00')
        self.assertEqual(report['realized_recovery_usd'], '3200.00')
        self.assertEqual(report['historical_receipts']['case_count'], 4)
        other = get_recovery_overview(self.db, self.cp, tenant='OTHER')
        self.assertEqual(other['total_approved_usd'], '4000.00')
        self.assertEqual(other['realized_recovery_usd'], '0.00')
        self.assertIsNone(other['estimated_cash_usd'])

    def test_cent_allocations_reconcile_per_year_without_per_case_double_rounding(self):
        # Independent tiny amounts reveal double-rounding errors which whole
        # dollar demonstration rows cannot detect.
        tiny_db = self.root / 'tiny.sqlite3'
        load_snapshot(tiny_db, ROOT / 'data')
        tiny = copy.deepcopy(self.archive)
        for row in tiny['cases']:
            row['historical_approved_amount'] = '0.01'
        import_archive(tiny_db, tiny)
        basis = forecast.get_forecast(tiny_db, self.cp, scope='all_approved')['basis']
        forecast.save_forecast(tiny_db, self.cp, scope='all_approved',
            expected_basis_digest=basis['basis_digest'], recovery_percentage='50',
            reason='Tiny synthetic cent-allocation fixture.', reviewer='Synthetic reviewer',
            as_of='2026-10-18', horizon_date='2026-12-31', idempotency_key='tiny:forecast')
        receipt = save_historical_realization(tiny_db, self.cp,
            expected_basis_digest=basis['basis_digest'], reviewer='Synthetic reviewer',
            reason='Invented historical collection fixture.', as_of='2026-10-18',
            idempotency_key='tiny:receipt', synthetic_confirmed=True)
        report = get_recovery_overview(tiny_db, self.cp)
        self.assertEqual(receipt['realized_recovery_usd'], '0.02')
        self.assertEqual(receipt['by_year']['2024']['realized_recovery_usd'], '0.01')
        self.assertEqual(receipt['by_year']['2025']['realized_recovery_usd'], '0.01')
        self.assertEqual(report['estimated_cash_usd'], '0.00')
        self.assertEqual(report['outstanding_approved_usd'], '0.00')
        self.assertEqual(report['realized_recovery_usd'], '0.02')


if __name__ == '__main__':
    unittest.main()
