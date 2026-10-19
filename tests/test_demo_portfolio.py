"""Invented portfolio planning and real, isolated offline workflow integration."""
from __future__ import annotations

import copy
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recovery import demo_portfolio as demo
from recovery.schemas import validate_row
from recovery.storage import connect, load_snapshot


ROOT = Path(__file__).resolve().parents[1]
try:
    GRAPH_AVAILABLE = importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
except ModuleNotFoundError:
    GRAPH_AVAILABLE = False
OPTIONAL_AVAILABLE = GRAPH_AVAILABLE and importlib.util.find_spec('pydantic') is not None


class PortfolioPlanTests(unittest.TestCase):
    def test_current_cohort_has_exact_decisions_and_schema_valid_decimal_sources(self):
        plan = demo.current_portfolio_plan()
        self.assertEqual(plan['targets'], {'approved': 500, 'awaiting': 250, 'evidence': 250})
        self.assertEqual(len(plan['tables']['suppliers']), 35)
        self.assertEqual(len(plan['cases']), 1000)
        self.assertEqual(len({case['line_id'] for case in plan['cases']}), 1000)
        self.assertEqual(len({case['thread_id'] for case in plan['cases']}), 1000)
        self.assertEqual(len(plan['rate_scopes']), 285)
        for table, rows in plan['tables'].items():
            for number, row in enumerate(rows, start=2):
                self.assertEqual(validate_row(table, row, number), [])
        invoices = plan['tables']['invoice_lines']
        self.assertTrue(all(row['invoice_date'] == '2026-09-30' for row in invoices))
        self.assertTrue(all(Decimal(row['billed_hours']) * Decimal(row['billed_rate'])
                            == Decimal(row['billed_amount']) for row in invoices))
        self.assertEqual(sum((Decimal(row['billed_amount']) for row in invoices), Decimal('0')),
                         Decimal('11000000.00'))
        self.assertEqual({case['target'] for case in plan['cases'][:500]}, {'APPROVED_FOR_DRAFT'})
        self.assertEqual({case['target'] for case in plan['cases'][500:750]}, {'AWAITING_APPROVAL'})
        self.assertEqual({case['target'] for case in plan['cases'][750:]}, {'NEEDS_REVIEW'})

    def test_evidence_scopes_cannot_invalidate_other_current_case_scopes(self):
        cases = demo.current_portfolio_plan()['cases']
        complete = {(case['supplier_id'], case['contract_id']) for case in cases[:750]}
        evidence = [(case['supplier_id'], case['contract_id']) for case in cases[750:]]
        self.assertEqual(len(set(evidence)), 250)
        self.assertFalse(complete.intersection(evidence))

    def test_historical_payload_is_separate_locked_provenance_and_exact_year_counts(self):
        archive = demo.historical_portfolio_payload()
        self.assertTrue(archive['synthetic'])
        self.assertEqual(archive['provenance'], 'Migrated from old system — locked (synthetic)')
        self.assertEqual(len(archive['suppliers']), 50)
        self.assertEqual(len(archive['contracts']), 100)
        self.assertEqual(len(archive['cases']), 2000)
        for year in (2024, 2025):
            cases = [case for case in archive['cases'] if case['origin_year'] == year]
            self.assertEqual(len(cases), 1000)
            self.assertTrue(all(case['invoice_date'].startswith(str(year)) for case in cases))
            self.assertTrue(all(case['line_id'].startswith('LEGACY-') and 'status' not in case
                                for case in cases))
            self.assertEqual(sum((Decimal(case['historical_approved_amount']) for case in cases), Decimal('0')),
                             Decimal('1000000.00'))

    def test_counts_reject_bool_negative_and_zero_case_plans(self):
        for options in ({'vendor_count': True}, {'approved': -1}, {'awaiting': 1.5},
                        {'approved': 0, 'awaiting': 0, 'evidence': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                demo.current_portfolio_plan(**options)
        for options in ({'vendor_count': 0}, {'cases_per_year': True}, {'cases_per_year': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                demo.historical_portfolio_payload(**options)


@unittest.skipUnless(OPTIONAL_AVAILABLE, 'Install graph and ollama extras for real offline portfolio workflows')
class PortfolioBuildTests(unittest.TestCase):
    def setUp(self):
        from recovery import lifecycle
        from recovery.workflow import run_persistent

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.db = self.directory / 'million.sqlite3'
        self.checkpoints = self.directory / 'million-ui-checkpoints.sqlite3'
        self.output = self.directory / 'scaled'
        load_snapshot(self.db, ROOT / 'data')
        # The isolated test parent uses the original fixture, with an explicit
        # parent-profile label. Approval is then acquired normally, after this
        # metadata is finalized; no live profile or expected-results file is read.
        with connect(self.db) as conn, conn:
            manifest = json.loads(conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
            manifest['dataset_id'] = 'synthetic-million-approved-demo-v1'
            conn.execute("UPDATE metadata SET value=? WHERE key='manifest'", (json.dumps(manifest),))
        lifecycle.migrate(self.db, backup_dir=self.directory / 'backups')
        run_persistent(self.db, self.checkpoints, 'ui:original-approved', line_id='L004', use_llm=False)
        self.original = run_persistent(self.db, self.checkpoints, 'ui:original-approved', decision={
            'decision': 'approve', 'reviewer': 'synthetic-test-owner', 'note': 'Reviewed isolated synthetic baseline.'})
        self.before = {'database': demo.database_fingerprints(self.db),
                       'checkpoints': demo.database_fingerprints(self.checkpoints)}
        self.plan = demo.current_portfolio_plan(vendor_count=2, approved=2, awaiting=1, evidence=1)
        self.archive = demo.historical_portfolio_payload(vendor_count=2, cases_per_year=2)

    def build(self, **kwargs):
        options = {'plan': self.plan, 'archive': self.archive}
        options.update(kwargs)
        with (patch('recovery.ai.resolve_explanation_config', side_effect=AssertionError('No model selection')),
                patch('recovery.ai.explain_case', side_effect=AssertionError('No model generation')),
                patch('recovery.term_extraction.extract_terms', side_effect=AssertionError('No fake extraction'))):
            return demo.create_scaled_portfolio(self.db, self.output, **options)

    def assert_source_unchanged(self):
        self.assertEqual(demo.database_fingerprints(self.db), self.before['database'])
        self.assertEqual(demo.database_fingerprints(self.checkpoints), self.before['checkpoints'])

    def assert_unpublished(self):
        for name in ('portfolio.sqlite3', 'portfolio-ui-checkpoints.sqlite3', 'portfolio-profile.json'):
            self.assertFalse((self.output / name).exists(), name)
        self.assert_source_unchanged()

    def test_actual_workflows_and_locked_archive_preserve_original_review_authority(self):
        from recovery import lifecycle
        from recovery.engine import audit
        from recovery.legacy_archive import list_archived_cases
        from recovery.workflow import get_review_workflow

        result = self.build()
        self.assertEqual(result['status'], 'PASSED')
        self.assertEqual(result['current_invoice_lines'], 24)
        self.assertEqual(result['new_review_events'], 3)
        self.assertEqual(result['new_request_count'], 1)
        self.assertEqual(result['real_pending_workflow_count'], 1)
        self.assertEqual(result['original_current_approvals_preserved'], 1)
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(result['candidate_total_usd'], '83707.50')
        self.assertEqual(result['new_realized_recovery'], '0.00')
        db = Path(result['database'])
        checkpoints = Path(result['checkpoints'])
        self.assertEqual(get_review_workflow(db, checkpoints, 'ui:original-approved'), self.original)
        pending = get_review_workflow(db, checkpoints, 'ui:scaled-2026:SCALE-2026-0003')
        self.assertEqual(pending['next'], ['approval'])
        self.assertEqual(pending['state']['brief']['mode'], 'offline_template')
        report, summary, _, _, requests = demo._portfolio_state(db, checkpoints)
        cases = {case['line_id']: case for case in summary['cases']}
        self.assertEqual([cases[f'SCALE-2026-{number:04d}']['bucket'] for number in range(1, 5)],
                         ['approved', 'approved', 'approval', 'evidence'])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['status'], 'OPEN')
        evidence = next(case for case in report['cases'] if case['line_id'] == 'SCALE-2026-0004')
        self.assertEqual((evidence['status'], evidence['reason']), ('NEEDS_REVIEW', 'OPEN_EVIDENCE_REQUEST'))
        self.assertEqual(len(list_archived_cases(db)), 4)
        with connect(db) as conn:
            self.assertEqual(audit(conn)['case_count'], 24)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM invoice_lines WHERE line_id LIKE 'LEGACY-%'").fetchone()[0], 0)
            metadata = json.loads(conn.execute("SELECT value FROM metadata WHERE key='demo_portfolio'").fetchone()[0])
        self.assertTrue(metadata['reconciliation']['matched'])
        self.assertEqual(metadata['term_preparation'], 'manual_synthetic_fixture')
        sources = [source for source in lifecycle.list_sources(db, 'DEMO')
                   if source['scope']['supplier_id'].startswith('DEMO-SCALE-')]
        self.assertEqual(len(sources), 3)
        self.assertTrue(all(source['origin'] == 'SYNTHETIC_INTAKE' for source in sources))
        proposals = [proposal for proposal in lifecycle.list_proposals(db, 'DEMO')
                     if proposal['scope']['supplier_id'].startswith('DEMO-SCALE-')]
        self.assertTrue(all(proposal['metadata']['provider'] == 'manual_synthetic_fixture'
                            and proposal['status'] == 'TERMS_APPROVED' for proposal in proposals))
        self.assert_source_unchanged()
        before_output = demo.database_fingerprints(db)
        with self.assertRaisesRegex(ValueError, 'refusing replacement'):
            self.build()
        self.assertEqual(demo.database_fingerprints(db), before_output)
        self.assert_source_unchanged()

    def test_failed_workflows_publish_nothing_and_preserve_source_and_checkpoints(self):
        with patch('recovery.demo_portfolio._seed_reviews', side_effect=RuntimeError('synthetic checkpoint fault')):
            with self.assertRaisesRegex(RuntimeError, 'checkpoint fault'):
                self.build()
        self.assert_unpublished()
        self.assertTrue(list((self.output / 'before').glob('*/source.sqlite3')))

    def test_malformed_source_or_collision_aborts_append_and_publishes_nothing(self):
        for change in ('malformed', 'collision'):
            with self.subTest(change=change):
                plan = copy.deepcopy(self.plan)
                if change == 'malformed':
                    plan['tables']['invoice_lines'][0]['billed_amount'] = 'NaN'
                else:
                    plan['tables']['suppliers'][0]['supplier_id'] = 'SUP-A'
                with self.assertRaisesRegex(ValueError, 'schema validation|collides'):
                    self.build(plan=plan)
                self.assert_unpublished()

    def test_synthetic_guard_rejects_before_backup_or_workflow(self):
        with connect(self.db) as conn, conn:
            manifest = json.loads(conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
            manifest['synthetic'] = False
            conn.execute("UPDATE metadata SET value=? WHERE key='manifest'", (json.dumps(manifest),))
        guarded_before = demo.database_fingerprints(self.db)
        with patch('recovery.demo_portfolio._copy_database') as backup:
            with self.assertRaisesRegex(ValueError, 'SYNTHETIC_ONLY'):
                self.build()
            backup.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(demo.database_fingerprints(self.db), guarded_before)
        self.assertEqual(demo.database_fingerprints(self.checkpoints), self.before['checkpoints'])
