"""Current-review navigation, using disposable fixtures and real graph checkpoints."""
from __future__ import annotations

from pathlib import Path
import unittest

import test_ui as review_ui_helpers

from recovery.lifecycle import migrate, reassess
from recovery.workflow import run_persistent


@unittest.skipUnless(review_ui_helpers.AVAILABLE,
                     'Install ui, graph and ollama extras for invoice navigation verification')
class InvoiceNavigationTests(unittest.TestCase):
    def setUp(self):
        self.helper = review_ui_helpers.ReviewWorkflowUITests(methodName='runTest')
        self.addCleanup(self.helper.doCleanups)
        self.helper.setUp()
        self.sequence = 0

    def start_offline_review(self, line):
        self.sequence += 1
        thread = f'ui:{self.sequence:020d}:navigation'
        run_persistent(self.helper.db, self.helper.checkpoints, thread, line_id=line)
        return thread

    def complete_candidates(self, *, migrated=True):
        if migrated:
            migrate(self.helper.db, backup_dir=Path(self.helper.temp.name) / 'backups')
        for line in ('L001', 'L004', 'L011', 'L017'):
            thread = self.start_offline_review(line)
            run_persistent(self.helper.db, self.helper.checkpoints, thread,
                           decision={'decision': 'approve', 'reviewer': 'Navigation test',
                                     'note': 'Reviewed the bundled synthetic rate evidence.'})

    def assert_navigation_only(self, previous_events):
        self.helper.assert_clean()
        self.helper.generate.assert_not_called()
        self.helper.resolve.assert_not_called()
        self.assertEqual(self.helper.events(), previous_events)

    def selected_line(self):
        return self.helper.selected_line()

    def test_completed_candidates_leave_pending_queue_but_remain_in_history(self):
        self.complete_candidates()
        previous_events = self.helper.events()
        self.helper.app.run()
        self.assertEqual(self.helper.widget('radio', 'Review scope').value, 'Needs review')
        self.assertTrue({'L001', 'L004', 'L011', 'L017'}.isdisjoint(self.helper.visible_lines()))
        self.assertFalse(any(widget.label == 'Inspect an invoice line' for widget in self.helper.app.selectbox))
        self.helper.widget('radio', 'Review scope').set_value('All cases').run()
        self.assertEqual(len(self.helper.visible_lines()), 20)
        self.helper.open_line('L004')
        self.assertEqual(self.selected_line(), 'L004')
        self.assert_navigation_only(previous_events)

    def assert_l004_returns_to_pending_table(self, previous_events):
        self.helper.app.run()
        self.assertIn('L004', self.helper.visible_lines())
        self.assertTrue({'L001', 'L011', 'L017'}.isdisjoint(self.helper.visible_lines()))
        self.helper.open_line('L006')
        self.assertEqual(self.selected_line(), 'L006')
        self.helper.open_line('L004')
        self.assertEqual(self.selected_line(), 'L004')
        self.assert_navigation_only(previous_events)

    def test_same_digest_reassessment_returns_candidate_to_navigation(self):
        self.complete_candidates()
        previous_events = self.helper.events()
        prior = next(event for event in previous_events if event['line_id'] == 'L004')
        assessment = reassess(self.helper.db, 'DEMO', 'L004',
                              prior_review_event_id=prior['event_id'],
                              idempotency_key='navigation:same-digest-reassessment')
        self.assertEqual(assessment['case_digest'], prior['case_digest'])
        self.assert_l004_returns_to_pending_table(previous_events)

    def test_new_pending_attempt_supersedes_completed_decision_in_navigation(self):
        self.complete_candidates(migrated=False)
        previous_events = self.helper.events()
        thread = self.start_offline_review('L004')
        self.assertEqual(self.helper.checkpoint_states()[thread].next, ('approval',))
        self.assert_l004_returns_to_pending_table(previous_events)

    def test_vendor_contract_filters_constrain_table_and_clear_incompatible_case(self):
        from recovery.engine import audit
        from recovery.storage import connect

        self.helper.app.run()
        self.helper.open_line('L001')
        with connect(self.helper.db) as conn:
            cases = audit(conn)['cases']
        target = next(case for case in cases if case['supplier_id'] == 'SUP-C'
                      and case['status'] in {'CANDIDATE', 'NEEDS_REVIEW'})
        self.helper.widget('selectbox', 'Vendor').select('SUP-C').run()
        self.helper.widget('selectbox', 'Contract').select(target['contract_id']).run()
        expected = {case['line_id'] for case in cases
                    if case['supplier_id'] == 'SUP-C' and case['contract_id'] == target['contract_id']
                    and case['status'] in {'CANDIDATE', 'NEEDS_REVIEW'}}
        self.assertEqual(set(self.helper.visible_lines()), expected)
        self.assertNotEqual(self.selected_line(), 'L001')
        self.helper.open_line(target['line_id'])
        self.assertEqual(self.selected_line(), target['line_id'])
        self.assert_navigation_only([])

    def test_sidebar_has_six_button_pages_and_navigation_is_read_only(self):
        migrate(self.helper.db, backup_dir=Path(self.helper.temp.name) / 'backups')
        self.helper.app.run()
        navigation = [item for item in self.helper.app.button
                      if item.key and item.key.startswith('navigation:')]
        self.assertEqual([item.label for item in navigation],
            ['Overview', 'Invoice Review', 'Contract Term Review', 'Evidence Requests',
             'Ask the Audit Workspace', 'Human Approval Auditing Team'])
        self.assertFalse(any(item.label == 'Navigation' for item in self.helper.app.radio))
        self.assertEqual([item.key for item in navigation if item.proto.type == 'primary'],
                         ['navigation:Invoice review'])
        self.assertFalse(any(widget.label == 'Page' for widget in self.helper.app.selectbox))
        for page in ('Overview', 'Contract term review', 'Evidence requests',
                     'Ask the Audit Workspace', 'Human Approval Auditing Team', 'Invoice review'):
            self.helper.navigate(page)
            active = [item.key for item in self.helper.app.button
                      if item.key and item.key.startswith('navigation:')
                      and item.proto.type == 'primary']
            self.assertEqual(active, ['navigation:' + page])
            self.assert_navigation_only([])
        self.assertFalse(self.helper.checkpoints.exists())


if __name__ == '__main__':
    unittest.main()
