"""Independent forecast eligibility, precision and stale-assumption regressions."""
from __future__ import annotations

import copy
from decimal import Decimal, localcontext
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from recovery.forecast import basis_from_summary, compute_forecast, get_forecast, save_forecast
from recovery.storage import TABLES, connect, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
try:
    GRAPH_AVAILABLE = importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
except ModuleNotFoundError:
    GRAPH_AVAILABLE = False


def approved_case(line='L001', amount='100.00', **changes):
    return {'tenant_id': 'DEMO', 'line_id': line, 'supplier_id': 'SUP-A', 'contract_id': 'CTR-A',
        'status': 'CANDIDATE', 'currency': 'USD', 'bucket': 'approved',
        'current_decision': 'APPROVED_FOR_DRAFT', 'case_digest': 'a' * 64,
        'review_event_id': line + ':review', 'assessment_id': 'assessment:' + line,
        'workflow_thread_id': 'thread:' + line, 'net_candidate': amount,
        'known_candidate': amount, 'blocking_request_count': 0, **changes}


class ForecastFinancialSafetyTests(unittest.TestCase):
    def test_only_current_approved_usd_cases_enter_the_basis(self):
        changes = [dict(tenant_id='OTHER'), dict(currency='EUR'), dict(status='MIGRATED_APPROVED'),
            dict(line_id='LEGACY-2025-001'), dict(status='NEEDS_REVIEW'), dict(bucket='evidence'),
            dict(bucket='reassessment'), dict(validity='STALE'), dict(blocking_request_count=1),
            dict(current_decision='REJECTED'), dict(current_decision='NEEDS_MORE_EVIDENCE'),
            dict(current_decision=None), dict(bucket='approval'), dict(bucket='preparation')]
        cases = [approved_case()]
        cases.extend(approved_case(line=f'EXCLUDED-{index}', amount='999999.99', **change)
                     for index, change in enumerate(changes))
        basis = basis_from_summary({'cases': cases})
        self.assertEqual(basis['case_count'], 1)
        self.assertEqual(basis['amount_usd'], '100.00')
        self.assertEqual([row['line_id'] for row in basis['cases']], ['L001'])
        self.assertEqual(compute_forecast(basis, '50'), '50.00')

    def test_eligible_rows_with_invalid_financial_or_approval_identity_fail_closed(self):
        changes = [dict(case_digest=None), dict(case_digest='not-a-sha256'),
            dict(review_event_id=''), dict(supplier_id=''), dict(contract_id=None),
            dict(net_candidate=100.0), dict(known_candidate=100.0),
            dict(net_candidate='NaN'), dict(known_candidate='Infinity'),
            dict(net_candidate='-1.00', known_candidate='-1.00'),
            dict(net_candidate='0.00', known_candidate='0.00'),
            dict(net_candidate='100.00', known_candidate='99.99')]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                basis_from_summary({'cases': [approved_case(**change)]})
        case = approved_case()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            basis_from_summary({'cases': [case, dict(case)]})

    def test_round_once_on_aggregate_and_preserve_snapshot_identity(self):
        cases = [approved_case('PENNY-1', '0.05'), approved_case('PENNY-2', '0.05')]
        original = copy.deepcopy(cases)
        first = basis_from_summary({'cases': cases})
        second = basis_from_summary({'cases': list(reversed(cases))})
        self.assertEqual(first['basis_digest'], second['basis_digest'])
        self.assertEqual(compute_forecast(first, '5'), '0.01')
        self.assertEqual(cases, original)
        for change in (dict(assessment_id='assessment:new'),
                       dict(review_event_id='new-thread:review'),
                       dict(workflow_thread_id='new-thread')):
            changed = copy.deepcopy(cases)
            changed[0].update(change)
            self.assertNotEqual(first['basis_digest'],
                basis_from_summary({'cases': changed})['basis_digest'])

    def test_percentage_is_strict_and_zero_is_a_real_projection(self):
        basis = basis_from_summary({'cases': [approved_case()]})
        for rate in (None, False, 50, 50.0, Decimal('50'), '-1', '100.000001',
                     'NaN', 'Infinity', '1e2', ' 50 ', '0.0000001'):
            with self.subTest(rate=repr(rate)), self.assertRaises(ValueError):
                compute_forecast(basis, rate)
        self.assertEqual(compute_forecast(basis, '0'), '0.00')
        self.assertEqual(compute_forecast(basis, '100'), '100.00')

    def test_forecast_preserves_exact_percentage_under_low_ambient_decimal_precision(self):
        basis = basis_from_summary({'cases': [approved_case(amount='1000000.00')]})
        self.assertEqual(compute_forecast(basis, '12.345678'), '123456.78')
        with localcontext() as context:
            context.prec = 5
            self.assertEqual(compute_forecast(basis, '12.345678'), '123456.78')

    def test_modified_basis_cannot_reuse_a_previous_digest(self):
        basis = basis_from_summary({'cases': [approved_case()]})
        basis['amount_usd'] = '1000000.00'
        with self.assertRaisesRegex(ValueError, 'digest'):
            compute_forecast(basis, '50')


@unittest.skipUnless(GRAPH_AVAILABLE, 'Install graph extra for fresh-basis persistence verification')
class ForecastPersistenceSafetyTests(unittest.TestCase):
    def setUp(self):
        from recovery.lifecycle import migrate

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.db, self.cp = self.root / 'synthetic.sqlite3', self.root / 'checkpoints.sqlite3'
        load_snapshot(self.db, ROOT / 'data')
        migrate(self.db, backup_dir=self.root / 'backups')
        self.counter = 0
        self.approve_current()
        for target in ('recovery.ai.resolve_explanation_config',
                       'recovery.ollama_provider._json_request', 'recovery.embeddings.embed_texts'):
            self.enterContext(patch(target, side_effect=AssertionError('Forecast must not call a provider')))

    def approve_current(self):
        from recovery.workflow import run_persistent

        self.counter += 1
        thread = f'forecast-safety:{self.counter}'
        run_persistent(self.db, self.cp, thread, line_id='L004', use_llm=False)
        run_persistent(self.db, self.cp, thread, decision={'decision': 'approve',
            'reviewer': 'Synthetic accountant', 'note': 'Checked the supplied dated rates and offsets.'})

    def financial_state(self):
        with connect(self.db) as conn:
            return {table: [tuple(row) for row in conn.execute(f'SELECT * FROM {table}')]
                    for table in (*TABLES, 'metadata', 'review_events', 'assessment_revisions',
                                  'evidence_requests', 'case_review_assessments', 'operation_keys')}

    def current(self):
        return get_forecast(self.db, self.cp)

    def save(self, **changes):
        values = {'expected_basis_digest': self.current()['basis']['basis_digest'],
            'recovery_percentage': '50', 'reason': 'Illustrative synthetic planning assumption; not calibrated.',
            'reviewer': 'Synthetic finance reviewer', 'as_of': '2026-10-18',
            'horizon_date': '2027-01-31', 'idempotency_key': 'forecast-safety:first', **changes}
        return save_forecast(self.db, self.cp, **values)

    def test_unset_forecast_is_read_only_and_does_not_guess_cash(self):
        before = self.financial_state()
        result = self.current()
        self.assertEqual(result['status'], 'NOT_SET')
        self.assertIsNone(result['projected_cash_usd'])
        self.assertIsNone(result['assumption'])
        self.assertEqual(result['basis']['amount_usd'], '600.00')
        self.assertEqual(result['realized_recovery'], '0.00')
        with connect(self.db) as conn:
            self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name='forecast_assumptions'").fetchone())
        self.assertEqual(self.financial_state(), before)

    def test_explicit_zero_forecast_preserves_approval_finance_and_provenance(self):
        before = self.financial_state()
        saved = self.save(recovery_percentage='0')
        self.assertEqual(saved['status'], 'CURRENT')
        self.assertEqual(saved['projected_cash_usd'], '0.00')
        self.assertEqual(saved['realized_recovery'], '0.00')
        assumption = saved['assumption']
        self.assertEqual(assumption['recovery_percentage'], '0')
        self.assertIn('not calibrated', assumption['reason'])
        self.assertEqual(assumption['reviewer'], 'Synthetic finance reviewer')
        self.assertEqual(assumption['as_of'], '2026-10-18')
        self.assertEqual(assumption['horizon_date'], '2027-01-31')
        self.assertEqual(assumption['basis']['amount_usd'], '600.00')
        self.assertIn('uncalibrated', saved['notice'])
        self.assertEqual(self.current()['assumption']['forecast_id'], assumption['forecast_id'])
        self.assertEqual(self.financial_state(), before)

    def test_changed_displayed_basis_rejects_before_creating_any_assumption(self):
        before = self.financial_state()
        with self.assertRaisesRegex(ValueError, 'STALE_BASIS'):
            self.save(expected_basis_digest='0' * 64)
        with connect(self.db) as conn:
            self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name='forecast_assumptions'").fetchone())
        self.assertEqual(self.financial_state(), before)

    def test_same_amount_new_approval_invalidates_projection_and_old_idempotent_replay(self):
        first = self.save()
        original = first['assumption']
        self.assertEqual(first['projected_cash_usd'], '300.00')
        self.approve_current()
        stale = self.current()
        self.assertEqual(stale['basis']['amount_usd'], '600.00')
        self.assertNotEqual(stale['basis']['basis_digest'], original['basis_digest'])
        self.assertEqual(stale['status'], 'STALE_BASIS')
        self.assertIsNone(stale['projected_cash_usd'])
        replay = self.save(expected_basis_digest=original['basis_digest'])
        self.assertEqual(replay['assumption']['forecast_id'], original['forecast_id'])
        self.assertEqual(replay['status'], 'STALE_BASIS')
        self.assertIsNone(replay['projected_cash_usd'])
        refreshed = self.save(idempotency_key='forecast-safety:refreshed', recovery_percentage='25')
        self.assertEqual(refreshed['status'], 'CURRENT')
        self.assertEqual(refreshed['projected_cash_usd'], '150.00')
        with connect(self.db) as conn:
            rows = list(conn.execute('SELECT * FROM forecast_assumptions ORDER BY sequence'))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['forecast_id'], original['forecast_id'])
        self.assertEqual(rows[0]['projected_cash_usd'], '300.00')

    def test_persisted_assumptions_are_immutable_including_replace_and_payload_reuse(self):
        first = self.save()
        with self.assertRaisesRegex(ValueError, 'idempotency'):
            self.save(recovery_percentage='51')
        with connect(self.db) as conn:
            row = conn.execute('SELECT * FROM forecast_assumptions').fetchone()
            for sql, values in (
                    ('UPDATE forecast_assumptions SET recovery_percentage=?', ('99',)),
                    ('DELETE FROM forecast_assumptions', ()),
                    (f'INSERT OR REPLACE INTO forecast_assumptions VALUES ({",".join("?" for _ in row)})', tuple(row))):
                with self.subTest(sql=sql), self.assertRaisesRegex(sqlite3.IntegrityError, 'immutable'):
                    with conn:
                        conn.execute(sql, values)
            self.assertEqual(conn.execute('SELECT count(*) FROM forecast_assumptions').fetchone()[0], 1)
        self.assertEqual(self.current()['assumption']['forecast_id'], first['assumption']['forecast_id'])

    def test_dates_and_tenant_scope_cannot_reuse_a_current_assumption(self):
        for changes in (dict(as_of='2026-10-14'), dict(horizon_date='2026-10-17'),
                        dict(as_of='2026-02-30'), dict(tenant='MISSING')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.save(**changes)
        saved = self.save()
        other = get_forecast(self.db, self.cp, tenant='OTHER')
        self.assertEqual(other['status'], 'NOT_SET')
        self.assertIsNone(other['projected_cash_usd'])
        self.assertEqual(other['basis']['case_count'], 0)
        self.assertEqual(self.current()['assumption']['forecast_id'], saved['assumption']['forecast_id'])


if __name__ == '__main__':
    unittest.main()
