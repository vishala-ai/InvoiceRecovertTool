"""Recover a committed typed evidence request after an interrupted graph save."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import test_ui as review_ui_helpers

from recovery.engine import audit
from recovery.lifecycle import migrate
from recovery.storage import connect


@unittest.skipUnless(review_ui_helpers.AVAILABLE,
                     'Install ui, graph and ollama extras for lifecycle UI recovery verification')
class LifecycleInterruptedSaveUITests(unittest.TestCase):
    def setUp(self):
        # Reuse the real app/provider harness without inheriting or rediscovering
        # the original test class. Its own cleanup stack owns the temp fixture.
        self.helper = review_ui_helpers.ReviewWorkflowUITests(methodName='runTest')
        self.addCleanup(self.helper.doCleanups)
        self.helper.setUp()
        migrate(self.helper.db, backup_dir=Path(self.helper.temp.name) / 'backups')

    def records(self):
        with connect(self.helper.db) as conn:
            return ([dict(row) for row in conn.execute('SELECT * FROM review_events')],
                    [dict(row) for row in conn.execute('SELECT * FROM evidence_requests')])

    def test_payment_request_keeps_original_type_when_finishing_after_checkpoint_failure(self):
        from recovery.review import record_review

        helper = self.helper
        helper.start_review()
        thread_id = next(iter(helper.checkpoint_states()))
        helper.widget('selectbox', 'Missing evidence type (for Needs review)').select('PAYMENT')

        def commit_then_fail(*args, **kwargs):
            record_review(*args, **kwargs)
            raise RuntimeError('Synthetic failure after payment-request commit, before checkpoint')

        with patch('recovery.workflow.record_review', side_effect=commit_then_fail) as write:
            helper.decide('request_more', reviewer='Finance demo',
                note='Confirm the synthetic settled payment allocation before continuing.')
            write.assert_called_once()
            self.assertEqual(write.call_args.kwargs['evidence_type'], 'PAYMENT')

        events, requests = self.records()
        self.assertEqual(len(events), 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(events[0]['decision'], 'NEEDS_MORE_EVIDENCE')
        self.assertEqual(events[0]['event_id'], thread_id + ':review')
        self.assertEqual(requests[0]['review_event_id'], events[0]['event_id'])
        self.assertEqual(requests[0]['request_type'], 'PAYMENT')
        self.assertEqual(requests[0]['status'], 'OPEN')
        self.assertEqual(helper.checkpoint_states()[thread_id].next, ('approval',))
        self.assertFalse(helper.decision_buttons())
        self.assertFalse(helper.app.error)
        self.assertTrue(any('needs more evidence' in item.value.lower() for item in helper.app.success))

        helper.app = helper.new_app().run()
        helper.assert_clean()
        helper.open_line('L001')
        self.assertFalse(helper.decision_buttons())
        self.assertFalse(any(item.label == 'Missing evidence type (for Needs review)'
                             for item in helper.app.selectbox))
        helper.widget('button', 'Finish workflow completion').click().run()
        helper.assert_clean()
        self.assertFalse(helper.app.error)
        state = helper.checkpoint_states()[thread_id]
        self.assertFalse(state.next)
        self.assertEqual(state.values['outcome'], events[0])
        self.assertEqual(self.records(), (events, requests))
        self.assertFalse(helper.decision_buttons())
        self.assertEqual(helper.generate.call_count, 1)
        # Preparation validates provider settings before lifecycle writes, then
        # independently pins them in the graph. Recovery performs neither call.
        self.assertEqual(helper.resolve.call_count, 2)

        # A further application restart preserves the original event/request.
        helper.app = helper.new_app().run()
        helper.assert_clean()
        helper.open_line('L001')
        self.assertEqual(self.records(), (events, requests))
        self.assertFalse(helper.decision_buttons())
        self.assertEqual(helper.generate.call_count, 1)
        with connect(helper.db) as conn:
            report = audit(conn)
        self.assertEqual(report['candidate_total'], '707.50')
        self.assertEqual(report['realized_recovery'], '0.00')
        l001 = next(case for case in report['cases'] if case['line_id'] == 'L001')
        self.assertEqual(l001['status'], 'NEEDS_REVIEW')


if __name__ == '__main__':
    unittest.main()
