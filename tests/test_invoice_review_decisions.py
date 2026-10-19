"""Invoice Review separates current human decisions from engine assessments."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import test_ui as review_ui_helpers

from recovery.storage import connect


@unittest.skipUnless(review_ui_helpers.AVAILABLE,
                     'Install ui, graph and ollama extras for invoice decision UI verification')
class InvoiceReviewDecisionUITests(unittest.TestCase):
    def setUp(self):
        from recovery import lifecycle

        # Compose the existing app harness rather than inheriting its test class,
        # so discovery runs only this module's regressions.
        self.helper = review_ui_helpers.ReviewWorkflowUITests(methodName='runTest')
        self.addCleanup(self.helper.doCleanups)
        self.helper.setUp()
        self.lifecycle = lifecycle
        lifecycle.migrate(self.helper.db,
                          backup_dir=Path(self.helper.temp.name) / 'backups')

    def approve_offline(self, decision='approve'):
        from recovery.workflow import run_persistent

        assessment = self.lifecycle.reassess(self.helper.db, 'DEMO', 'L004',
                                             idempotency_key='decision-ui-initial:' + decision)
        workflow = self.lifecycle.start_linked_review(
            self.helper.db, self.helper.checkpoints, 'DEMO', assessment['assessment_id'],
            idempotency_key='decision-ui-review:' + decision, use_llm=False)
        self.assertEqual(workflow['next'], ['approval'])
        finished = run_persistent(self.helper.db, self.helper.checkpoints,
            workflow['thread_id'], tenant='DEMO', decision={
                'decision': decision, 'reviewer': 'Synthetic finance reviewer',
                'note': 'Checked the supplied synthetic service and contract evidence.'})
        self.assertEqual(finished['state']['outcome']['decision'],
                         review_ui_helpers.OUTCOMES[decision])
        return assessment, finished['state']['outcome']

    def database_contents(self):
        with connect(self.helper.db) as conn:
            return list(conn.iterdump())

    def render_without_side_effects(self):
        before = self.database_contents()
        self.helper.app = self.helper.new_app().run()
        self.helper.assert_clean()
        self.helper.app.run()
        self.helper.assert_clean()
        self.assertEqual(self.database_contents(), before)
        self.helper.generate.assert_not_called()
        self.helper.resolve.assert_not_called()

    def queue_row(self, line_id):
        frame = next(element.value for element in self.helper.app.dataframe
                     if 'Case ID' in element.value.columns)
        return frame.loc[frame['Case ID'] == line_id].iloc[0]

    def test_saved_approval_moves_to_history_without_changing_candidate_assessment(self):
        self.approve_offline()
        self.render_without_side_effects()
        self.assertFalse(self.helper.app.metric)
        self.assertEqual(self.helper.widget('radio', 'Review scope').value, 'Needs review')
        self.assertNotIn('L004', self.helper.visible_lines())
        pending = self.helper.app.dataframe[0].value
        self.assertNotIn('L004', set(pending['Case ID']))
        before = self.database_contents()
        self.helper.widget('radio', 'Review scope').set_value('All cases').run()
        row = self.queue_row('L004')
        self.assertEqual(row['Candidate amount'], 'USD 600.00')
        self.assertEqual(row['Review status'], 'Approved for draft')
        self.assertIn('L004', self.helper.visible_lines())
        self.assertEqual(self.database_contents(), before)

        before = self.database_contents()
        self.helper.open_line('L002')
        self.helper.assert_clean()
        other = self.queue_row('L002')
        self.assertEqual(other['Candidate amount'], 'USD 0.00')
        self.assertEqual(other['Review status'], 'Other assessment')
        self.assertEqual(self.database_contents(), before)
        self.helper.generate.assert_not_called()
        self.helper.resolve.assert_not_called()

    def test_rejected_case_is_excluded_but_request_more_remains_pending(self):
        for decision in ('reject', 'request_more'):
            with self.subTest(decision=decision):
                self.approve_offline(decision)
                self.render_without_side_effects()
                visible = self.helper.visible_lines()
                if decision == 'reject':
                    self.assertNotIn('L004', visible)
                else:
                    self.assertIn('L004', visible)
                    row = self.queue_row('L004')
                    # A new blocking request invalidates the saved assessment;
                    # it remains outstanding work until evidence/reassessment.
                    self.assertEqual(row['Review status'], 'Reassessment required')
                    requests = self.lifecycle.list_requests(self.helper.db, 'DEMO')
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(requests[0]['status'], 'OPEN')

    def test_empty_pending_queue_hides_invoice_details_and_keeps_history_available(self):
        self.approve_offline()
        with connect(self.helper.db) as conn:
            from recovery.engine import audit

            report = audit(conn)
        # Limit the app report boundary to this completed synthetic case without
        # changing fixture rows or the expected baseline financial results.
        report['cases'] = [case for case in report['cases'] if case['line_id'] == 'L004']
        report.update(case_count=1, counts={'CANDIDATE': 1}, candidate_total='600.00')
        with patch('recovery.invoice_review.audit', return_value=report):
            self.render_without_side_effects()
            self.assertFalse(any(item.label == 'Inspect an invoice line'
                                 for item in self.helper.app.selectbox))
            self.assertFalse(self.helper.app.dataframe)
            self.assertFalse(self.helper.app.json)
            self.assertFalse(any(item.label == 'Review next candidate' for item in self.helper.app.button))
            self.assertTrue(any('No invoices are awaiting review.' in item.value
                                for item in self.helper.app.info))
            before = self.database_contents()
            self.helper.widget('radio', 'Review scope').set_value('All cases').run()
            self.helper.assert_clean()
            self.assertEqual(self.helper.visible_lines(), ['L004'])
            self.assertEqual(self.queue_row('L004')['Review status'], 'Approved for draft')
            self.assertEqual(self.database_contents(), before)
        self.helper.generate.assert_not_called()
        self.helper.resolve.assert_not_called()

    def test_same_digest_reassessment_does_not_display_inherited_approval(self):
        previous, event = self.approve_offline()
        current = self.lifecycle.reassess(self.helper.db, 'DEMO', 'L004',
            prior_review_event_id=event['event_id'],
            idempotency_key='decision-ui-new-revision')
        self.assertNotEqual(previous['assessment_id'], current['assessment_id'])
        self.assertEqual(previous['case_digest'], current['case_digest'])

        self.render_without_side_effects()
        self.assertFalse(self.helper.app.metric)
        row = self.queue_row('L004')
        self.assertEqual(row['Candidate amount'], 'USD 600.00')
        self.assertEqual(row['Review status'], 'Prepare explanation')
        with connect(self.helper.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)

    def test_overview_restored_queue_opens_saved_decision_despite_previous_filters(self):
        self.approve_offline()
        self.render_without_side_effects()
        before = self.database_contents()
        self.helper.search('Search vendor name or ID', 'cedar')
        self.helper.widget('selectbox', 'Vendor').select('SUP-C').run()
        self.helper.search('Search invoice or contract', 'L006')
        self.helper.navigate('Overview')
        metrics = {item.label: item.value for item in self.helper.app.metric}
        self.assertEqual({label: metrics[label] for label in ('Total approved value',
            'Estimated cash recovery under demo assumptions', 'Realized recovery')}, {
            'Total approved value': 'USD 600.00',
            'Estimated cash recovery under demo assumptions': 'Not estimated',
            'Realized recovery': 'USD 0.00',
        })
        self.helper.app.button(key='lifecycle:overview:card:approved').click().run()
        self.helper.assert_clean()
        cases = next(frame.value for frame in self.helper.app.dataframe
                     if 'Invoice line' in frame.value.columns)
        self.assertEqual(cases['Invoice line'].tolist(), ['L004'])
        self.assertEqual(cases['Current decision'].tolist(), ['APPROVED_FOR_DRAFT'])
        self.assertEqual(self.database_contents(), before)
        self.assertTrue(self.helper.widget('button', 'Open selected invoice').disabled)
        table = next(frame for frame in self.helper.app.dataframe if 'Invoice line' in frame.value.columns)
        self.helper.app.session_state[table.key] = {'selection': {'rows': [0], 'columns': [], 'cells': []}}
        self.helper.app.run()
        self.helper.assert_clean()
        self.helper.widget('button', 'Open selected invoice').click().run()
        self.helper.assert_clean()
        self.assertIsNone(self.helper.widget('selectbox', 'Search vendor name or ID').value)
        self.assertIsNone(self.helper.widget('selectbox', 'Search invoice or contract').value)
        self.assertEqual(self.helper.widget('radio', 'Review scope').value, 'All cases')
        self.assertEqual(self.helper.selected_line(), 'L004')
        self.assertEqual(self.queue_row('L004')['Review status'], 'Approved for draft')
        self.assertEqual(self.queue_row('L004')['Candidate amount'], 'USD 600.00')
        self.assertEqual(self.database_contents(), before)
        self.helper.generate.assert_not_called()
        self.helper.resolve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
