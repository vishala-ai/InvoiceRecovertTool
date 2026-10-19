"""Cash forecasting is a reviewed assumption, never a change to recovery facts."""
from __future__ import annotations

import copy
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from recovery import forecast
from recovery.engine import audit, evaluate_line
from recovery.review import record_review
from recovery.storage import connect, load_snapshot


ROOT = Path(__file__).resolve().parents[1]


def approved_case(line='L001', amount='100.00'):
    return {'tenant_id': 'DEMO', 'line_id': line, 'supplier_id': 'SUP-A', 'contract_id': 'CTR-A',
        'status': 'CANDIDATE', 'currency': 'USD', 'bucket': 'approved',
        'current_decision': 'APPROVED_FOR_DRAFT', 'case_digest': 'a' * 64,
        'review_event_id': line + ':review', 'assessment_id': 'assessment:' + line,
        'workflow_thread_id': 'thread:' + line, 'net_candidate': amount,
        'known_candidate': amount, 'blocking_request_count': 0}


class ForecastCalculationTests(unittest.TestCase):
    def test_aggregate_decimal_forecast_is_explicit_and_does_not_modify_basis(self):
        summary = {'cases': [approved_case(amount='1500000.00')]}
        before = copy.deepcopy(summary)
        basis = forecast.basis_from_summary(summary)
        self.assertEqual(forecast.compute_forecast(basis, '80'), '1200000.00')
        self.assertEqual(forecast.compute_forecast(basis, '0'), '0.00')
        self.assertEqual(forecast.compute_forecast(basis, '100.000000'), '1500000.00')
        self.assertEqual(summary, before)

    def test_basis_identity_tracks_approval_and_assessment_even_at_same_dollars(self):
        case = approved_case()
        baseline = forecast.basis_from_summary({'cases': [case]})
        for field, changed in (('assessment_id', 'assessment:new'), ('case_digest', 'b' * 64),
                               ('review_event_id', 'other-review'), ('workflow_thread_id', 'other-thread')):
            altered = {**case, field: changed}
            with self.subTest(field=field):
                revised = forecast.basis_from_summary({'cases': [altered]})
                self.assertEqual(revised['amount_usd'], baseline['amount_usd'])
                self.assertNotEqual(revised['basis_digest'], baseline['basis_digest'])

    def test_percentage_rejects_floats_bools_nonfinite_exponents_and_out_of_range(self):
        basis = forecast.basis_from_summary({'cases': [approved_case()]})
        for value in (80, 80.0, True, None, Decimal('80'), '-1', '100.000001', '101',
                      'NaN', 'Infinity', '1e2', ' 80 ', '1.1234567', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                forecast.compute_forecast(basis, value)

    def test_tampered_basis_and_duplicate_approved_lines_are_rejected(self):
        basis = forecast.basis_from_summary({'cases': [approved_case()]})
        basis['amount_usd'] = '999.00'
        with self.assertRaisesRegex(ValueError, 'digest'):
            forecast.compute_forecast(basis, '80')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            forecast.basis_from_summary({'cases': [approved_case(), approved_case()]})


class ForecastPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.db = self.directory / 'synthetic.sqlite3'
        self.checkpoints = self.directory / 'absent-checkpoints.sqlite3'
        load_snapshot(self.db, ROOT / 'data')
        self.approve('L004', 'original-approved')

    def approve(self, line, thread):
        with connect(self.db) as conn:
            case = evaluate_line(conn, 'DEMO', line)
            return record_review(conn, event_id=thread + ':review', thread_id=thread,
                tenant='DEMO', line_id=line, decision='approve', reviewer='Synthetic test reviewer',
                note='Reviewed isolated synthetic case.', case_digest=case['case_digest'])

    def table_rows(self):
        with connect(self.db) as conn:
            names = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            return {name: [tuple(row) for row in conn.execute('SELECT * FROM "' + name + '"')]
                    for name in names}

    def save(self, **overrides):
        values = {'expected_basis_digest': forecast.get_forecast(self.db, self.checkpoints)['basis']['basis_digest'],
            'recovery_percentage': '80', 'reason': 'Hypothetical collection assumption for this synthetic demo.',
            'reviewer': 'Synthetic forecast owner', 'as_of': '2026-10-18', 'horizon_date': '2027-01-18',
            'idempotency_key': 'forecast-review:1'}
        values.update(overrides)
        return forecast.save_forecast(self.db, self.checkpoints, **values)

    def test_read_is_nonmutating_and_not_set_is_distinct_from_reviewed_zero_percent(self):
        before = self.table_rows()
        initial = forecast.get_forecast(self.db, self.checkpoints)
        self.assertEqual(initial['status'], 'NOT_SET')
        self.assertIsNone(initial['projected_cash_usd'])
        self.assertIsNone(initial['assumption'])
        self.assertEqual(initial['basis']['amount_usd'], '600.00')
        self.assertEqual(self.table_rows(), before)
        self.assertFalse(self.checkpoints.exists())
        saved = self.save(recovery_percentage='0')
        self.assertEqual(saved['status'], 'CURRENT')
        self.assertEqual(saved['projected_cash_usd'], '0.00')
        self.assertEqual(saved['assumption']['reviewer'], 'Synthetic forecast owner')
        self.assertEqual(saved['assumption']['horizon_date'], '2027-01-18')
        self.assertIn('uncalibrated', saved['notice'])

    def test_append_and_replay_preserve_financial_inputs_review_journal_and_case_digests(self):
        before = self.table_rows()
        with connect(self.db) as conn:
            financial_before = audit(conn)
        first = self.save()
        second = self.save()
        self.assertEqual(first, second)
        self.assertEqual(first['projected_cash_usd'], '480.00')
        self.assertEqual(first['basis']['case_count'], 1)
        self.assertEqual(first['assumption']['sequence'], 1)
        after = self.table_rows()
        self.assertEqual({name: after[name] for name in before}, before)
        self.assertEqual(len(after['forecast_assumptions']), 1)
        with connect(self.db) as conn:
            self.assertEqual(audit(conn), financial_before)
            self.assertEqual(financial_before['candidate_total'], '80707.50')
            self.assertEqual(financial_before['realized_recovery'], '0.00')
        with self.assertRaisesRegex(ValueError, 'different assumption'):
            self.save(recovery_percentage='81')
        self.assertEqual(self.table_rows(), after)

    def test_new_approval_invalidates_saved_forecast_and_requires_explicit_new_review(self):
        first = self.save()
        self.approve('L011', 'second-approved')
        changed = forecast.get_forecast(self.db, self.checkpoints)
        self.assertEqual(changed['status'], 'STALE_BASIS')
        self.assertIsNone(changed['projected_cash_usd'])
        self.assertEqual(changed['assumption']['forecast_id'], first['assumption']['forecast_id'])
        self.assertEqual(changed['basis']['amount_usd'], '700.00')
        with self.assertRaisesRegex(ValueError, 'STALE_BASIS'):
            self.save(expected_basis_digest=first['basis']['basis_digest'], idempotency_key='stale-attempt')
        refreshed = self.save(recovery_percentage='50', idempotency_key='forecast-review:2')
        self.assertEqual(refreshed['status'], 'CURRENT')
        self.assertEqual(refreshed['projected_cash_usd'], '350.00')
        self.assertEqual(refreshed['assumption']['sequence'], 2)
        self.assertEqual(len(self.table_rows()['forecast_assumptions']), 2)

    def test_bad_inputs_reject_without_creating_forecast_tables(self):
        for changes in ({'reason': ' '}, {'reviewer': ''}, {'as_of': '2026-02-30'},
                        {'as_of': '2026-10-14'}, {'horizon_date': '2026-10-17'},
                        {'horizon_date': '2027-1-18'}, {'idempotency_key': ' '},
                        {'recovery_percentage': 50.0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.save(**changes)
            self.assertNotIn('forecast_assumptions', self.table_rows())

    def test_synthetic_guard_runs_before_current_case_loader_or_schema_creation(self):
        with connect(self.db) as conn, conn:
            manifest = json.loads(conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
            manifest['synthetic'] = False
            conn.execute("UPDATE metadata SET value=? WHERE key='manifest'", (json.dumps(manifest),))
        before = self.table_rows()
        with patch('recovery.forecast._fresh_basis', side_effect=AssertionError('Must not load case state')):
            for action in (lambda: forecast.get_forecast(self.db, self.checkpoints),
                           lambda: forecast.save_forecast(self.db, self.checkpoints,
                            expected_basis_digest='a' * 64, recovery_percentage='80', reason='Synthetic assumption',
                            reviewer='Test owner', as_of='2026-10-18', horizon_date='2027-01-18',
                            idempotency_key='blocked')):
                with self.assertRaisesRegex(ValueError, 'SYNTHETIC_ONLY'):
                    action()
        self.assertEqual(self.table_rows(), before)

    def test_empty_basis_rejects_and_missing_database_never_creates_directories(self):
        empty_db = self.directory / 'no-approvals.sqlite3'
        load_snapshot(empty_db, ROOT / 'data')
        basis = forecast.get_forecast(empty_db, self.checkpoints)['basis']
        with self.assertRaisesRegex(ValueError, 'requires current approved'):
            forecast.save_forecast(empty_db, self.checkpoints, expected_basis_digest=basis['basis_digest'],
                recovery_percentage='80', reason='Synthetic assumption', reviewer='Test owner',
                as_of='2026-10-18', horizon_date='2027-01-18', idempotency_key='empty')
        with connect(empty_db) as conn:
            self.assertFalse(conn.execute("SELECT 1 FROM sqlite_master WHERE name='forecast_assumptions'").fetchone())
        missing = self.directory / 'missing' / 'never.sqlite3'
        with self.assertRaises(ValueError):
            forecast.get_forecast(missing, self.checkpoints)
        self.assertFalse(missing.parent.exists())

    def test_saved_assumptions_reject_update_delete_and_replace(self):
        self.save()
        with connect(self.db) as conn:
            row = conn.execute('SELECT * FROM forecast_assumptions').fetchone()
            for sql, parameters in (
                ("UPDATE forecast_assumptions SET reason='changed'", ()),
                ('DELETE FROM forecast_assumptions', ()),
                ('INSERT OR REPLACE INTO forecast_assumptions VALUES (' + ','.join('?' for _ in row) + ')', tuple(row))):
                with self.subTest(sql=sql), self.assertRaisesRegex(sqlite3.IntegrityError, 'immutable'):
                    conn.execute(sql, parameters)
                conn.rollback()
            self.assertEqual(dict(conn.execute('SELECT * FROM forecast_assumptions').fetchone()), dict(row))
