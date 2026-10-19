"""Guarded bulk/single worklist actions against real temporary graph checkpoints."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import test_ui as ui_helpers

from recovery import lifecycle
from recovery.engine import audit
from recovery.invoice_review import (decide_selected, load_worklist, prepare_review,
                                    retry_review, selection_token)
from recovery.storage import connect


@unittest.skipUnless(ui_helpers.AVAILABLE, 'Install UI and graph extras for worklist verification')
class InvoiceWorklistFixture(unittest.TestCase):
    def setUp(self):
        self.helper = ui_helpers.ReviewWorkflowUITests(methodName='runTest')
        self.addCleanup(self.helper.doCleanups)
        self.helper.setUp()
        self.db, self.cp = self.helper.db, self.helper.checkpoints
        lifecycle.migrate(self.db, backup_dir=Path(self.helper.temp.name) / 'backups')

    def cases(self):
        return {case['line_id']: case for case in load_worklist(self.db, self.cp)['cases']}

    def prepare(self, line, key=None):
        case = self.cases()[line]
        return prepare_review(self.db, self.cp, case, action_id=key or line)

    def decide(self, tokens, **kwargs):
        return decide_selected(self.db, self.cp, tokens, supplier_id='SUP-A',
            contract_id='CTR-A', reviewer='Synthetic accountant',
            note='Checked dated rates, all signed amendments and recorded offsets.', **kwargs)

    def assert_baseline(self, event_count):
        with connect(self.db) as conn:
            report = audit(conn)
            count = conn.execute('SELECT count(*) FROM review_events').fetchone()[0]
        self.assertEqual(count, event_count)
        self.assertEqual(report['candidate_total'], '80707.50')
        self.assertEqual(report['realized_recovery'], '0.00')
        self.helper.resolve.assert_not_called()
        self.helper.generate.assert_not_called()



class InvoiceWorklistTests(InvoiceWorklistFixture):
    def test_reading_and_filtering_do_not_prepare_a_review(self):
        with connect(self.db) as conn:
            before = list(conn.iterdump())
        first, second = self.cases(), self.cases()
        self.assertEqual(first, second)
        self.assertFalse(self.cp.exists())
        self.assertFalse(any(case['ready'] for case in first.values()))
        self.assertIsNone(first['L006']['known_candidate'])
        with connect(self.db) as conn:
            self.assertEqual(list(conn.iterdump()), before)
        self.assert_baseline(0)

    def test_offline_pauses_are_eligible_and_bulk_approvals_are_idempotent(self):
        for line in ('L001', 'L004'):
            result = self.prepare(line)
            self.assertEqual(result['next'], ['approval'])
        cases = self.cases()
        self.assertTrue(cases['L001']['ready'])
        self.assertTrue(cases['L004']['ready'])
        tokens = [selection_token(cases[line]) for line in ('L001', 'L004')]
        results = self.decide(tokens)
        self.assertEqual([row['outcome'] for row in results], ['APPROVED_FOR_DRAFT'] * 2)
        self.assertEqual([row['outcome'] for row in self.decide(tokens)], ['Already saved'] * 2)
        alternate = self.decide(tokens, decision='reject')
        self.assertTrue(all(row['outcome'] == 'Not saved' for row in alternate))
        self.assert_baseline(2)

    def test_batch_scope_and_duplicate_validation_happen_before_any_writes(self):
        self.prepare('L004')
        cases = self.cases()
        token = selection_token(cases['L004'])
        for tokens in ([token, token], [token, selection_token(cases['L006'])]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.decide(tokens)
        self.assert_baseline(0)

    def test_partial_outcomes_do_not_approve_unprepared_or_unknown_cases(self):
        self.prepare('L004')
        cases = self.cases()
        results = self.decide([selection_token(cases[line]) for line in ('L001', 'L004')])
        self.assertEqual([row['outcome'] for row in results], ['Not saved', 'APPROVED_FOR_DRAFT'])
        self.assertIn('Prepare', results[0]['reason'])
        self.assertFalse(cases['L006']['ready'])
        self.assert_baseline(1)

    def test_same_digest_new_assessment_and_new_attempt_reject_frozen_selection(self):
        self.prepare('L004')
        previous = self.cases()['L004']
        token = selection_token(previous)
        assessment = lifecycle.reassess(self.db, 'DEMO', 'L004', idempotency_key='new-revision')
        self.assertEqual(assessment['case_digest'], token['case_digest'])
        result = self.decide([token])[0]
        self.assertEqual(result['outcome'], 'Not saved')
        lifecycle.start_linked_review(self.db, self.cp, 'DEMO', assessment['assessment_id'],
            idempotency_key='new-attempt', use_llm=False)
        self.assertTrue(self.cases()['L004']['ready'])
        self.assertEqual(self.decide([token])[0]['outcome'], 'Not saved')
        self.assert_baseline(0)

    def test_old_committed_approval_is_not_reported_for_new_same_digest_revision(self):
        self.prepare('L004')
        self.decide([selection_token(self.cases()['L004'])])
        lifecycle.reassess(self.db, 'DEMO', 'L004', idempotency_key='after-approval')
        current = self.cases()['L004']
        self.assertIsNone(current['current_decision'])
        result = self.decide([selection_token(current)])[0]
        self.assertEqual(result['outcome'], 'Not saved')
        self.assertIn('earlier assessment', result['reason'])
        self.assert_baseline(1)

    def test_failed_checkpoint_returns_committed_receipt_and_allows_exact_completion(self):
        self.prepare('L004')
        token = selection_token(self.cases()['L004'])
        from recovery.review import record_review

        def commit_then_fail(*args, **kwargs):
            record_review(*args, **kwargs)
            raise RuntimeError('Synthetic checkpoint failure after commit')

        with patch('recovery.workflow.record_review', side_effect=commit_then_fail):
            result = self.decide([token])[0]
        self.assertEqual(result['outcome'], 'APPROVED_FOR_DRAFT')
        self.assertTrue(result['completion_pending'])
        self.assertEqual(self.decide([token])[0]['outcome'], 'Already saved')
        self.assertFalse(self.cases()['L004']['completion_pending'])
        self.assert_baseline(1)

    def test_consent_and_index_preflight_precede_any_preparation_mutation(self):
        case = self.cases()['L004']
        with connect(self.db) as conn:
            before = list(conn.iterdump())
        with self.assertRaisesRegex(ValueError, 'Allow'):
            prepare_review(self.db, self.cp, case, action_id='no-consent', use_llm=True)
        missing = Path(self.helper.temp.name) / 'not-an-index'
        with self.assertRaises((ValueError, OSError)):
            prepare_review(self.db, self.cp, case, action_id='bad-index',
                           retrieval_index=str(missing), allow_embeddings=True)
        with patch('recovery.ai.resolve_explanation_config', side_effect=ValueError('Missing local model')):
            with self.assertRaisesRegex(ValueError, 'Missing local model'):
                prepare_review(self.db, self.cp, case, action_id='bad-model',
                               use_llm=True, allow_provider=True)
        self.assertFalse(self.cp.exists())
        with connect(self.db) as conn:
            self.assertEqual(list(conn.iterdump()), before)
        self.assert_baseline(0)

    def test_retry_never_switches_to_a_newer_unconsented_attempt(self):
        self.prepare('L004')
        displayed = self.cases()['L004']
        assessment_id = displayed['assessment_id']
        lifecycle.start_linked_review(self.db, self.cp, 'DEMO', assessment_id,
            idempotency_key='another-attempt', use_llm=False)
        with patch('recovery.workflow.retry_preparation') as retry:
            with self.assertRaisesRegex(ValueError, 'cannot be retried'):
                retry_review(self.db, self.cp, displayed)
            retry.assert_not_called()
        self.assert_baseline(0)


@unittest.skipUnless(ui_helpers.AVAILABLE, 'Install UI and graph extras for worklist verification')
class InvoiceTableSelectionTests(InvoiceWorklistFixture):
    def start_table(self):
        self.helper.app.run()
        self.helper.widget('selectbox', 'Vendor').select('SUP-A').run()
        self.helper.widget('selectbox', 'Contract').select('CTR-A').run()

    def select_line(self, line, selected=True):
        helper = self.helper
        table = helper.invoice_table()
        position = helper.visible_lines().index(line)
        helper.app.session_state[table.key] = {'edited_rows': {position: {'Select': selected}},
                                              'added_rows': [], 'deleted_rows': []}
        helper.app.run()
        helper.assert_clean()

    def assert_single_mode(self, line):
        self.assertEqual(self.helper.selected_line(), line)
        self.assertTrue(self.helper.decision_buttons())
        self.assertFalse(any(item.label == 'Approve selected for draft'
                             for item in self.helper.app.button))
        self.assertFalse(any(item.label == 'Bulk reviewer label'
                             for item in self.helper.app.text_input))

    def assert_bulk_mode(self):
        self.assertIsNone(self.helper.selected_line())
        self.assertFalse(self.helper.decision_buttons())
        self.assertTrue(any(item.label == 'Approve selected for draft'
                            for item in self.helper.app.button))
        self.assertFalse(any(item.label == 'Evidence review note'
                             for item in self.helper.app.text_area))

    def test_single_checkbox_and_multiple_selection_show_only_their_own_decision_form(self):
        self.prepare('L004')
        self.prepare('L011')
        self.start_table()
        self.helper.open_line('L001')
        self.select_line('L004')
        self.assert_single_mode('L004')
        self.select_line('L011')
        self.assert_bulk_mode()
        self.select_line('L011', selected=False)
        self.assert_single_mode('L004')
        self.assert_baseline(0)

    def test_explicit_open_exits_bulk_and_clears_checked_rows(self):
        self.prepare('L004')
        self.prepare('L011')
        self.start_table()
        self.select_line('L004')
        self.select_line('L011')
        self.assert_bulk_mode()
        self.helper.open_line('L011')
        self.assert_single_mode('L011')
        self.assertFalse(self.helper.invoice_table().value['Select'].any())
        self.assert_baseline(0)

    def test_clear_selection_closes_the_checkbox_single_or_bulk_review(self):
        self.prepare('L004')
        self.prepare('L011')
        self.start_table()
        self.select_line('L004')
        self.assert_single_mode('L004')
        self.helper.widget('button', 'Clear selection').click().run()
        self.assertIsNone(self.helper.selected_line())
        self.assertFalse(self.helper.decision_buttons())
        self.select_line('L004')
        self.select_line('L011')
        self.assert_bulk_mode()
        self.helper.widget('button', 'Clear selection').click().run()
        self.assertIsNone(self.helper.selected_line())
        self.assertFalse(self.helper.decision_buttons())
        self.assertFalse(any(item.label == 'Approve selected for draft'
                             for item in self.helper.app.button))
        self.assert_baseline(0)

    def test_deselecting_the_only_checked_case_does_not_restore_an_older_open_case(self):
        self.prepare('L004')
        self.start_table()
        self.helper.open_line('L001')
        self.select_line('L004')
        self.assert_single_mode('L004')
        self.select_line('L004', selected=False)
        self.assertIsNone(self.helper.selected_line())
        self.assertFalse(self.helper.decision_buttons())
        self.assert_baseline(0)

    def test_bulk_to_single_preserves_the_original_review_token_after_a_new_attempt(self):
        self.prepare('L004')
        self.prepare('L011')
        self.start_table()
        self.select_line('L004')
        self.select_line('L011')
        before = self.cases()['L004']
        lifecycle.start_linked_review(self.db, self.cp, 'DEMO', before['assessment_id'],
            idempotency_key='concurrent-bulk-to-single-attempt', use_llm=False)
        self.select_line('L011', selected=False)
        self.assert_single_mode('L004')
        self.assertTrue(all(button.disabled for button in self.helper.decision_buttons()))
        self.assertTrue(any('changed' in item.value.lower() for item in self.helper.app.warning))
        self.assert_baseline(0)

    def test_select_all_only_selects_ready_cases_in_the_current_contract(self):
        self.prepare('L004')
        self.prepare('L011')
        self.start_table()
        self.helper.widget('button', 'Select all eligible in this contract').click().run()
        selected = self.helper.invoice_table().value
        self.assertEqual(set(selected.loc[selected['Select'], 'Case ID']), {'L004', 'L011'})
        self.assertTrue(any('USD 700.00' in item.value for item in self.helper.app.caption))
        self.helper.widget('text_input', 'Bulk reviewer label').set_value('Accountant')
        self.helper.widget('text_area', 'Bulk evidence review note').set_value('Reviewed both cases and offsets.')
        self.helper.widget('button', 'Approve selected for draft').click().run()
        self.helper.assert_clean()
        self.assertEqual(len(self.helper.events()), 2)
        self.assertTrue({'L004', 'L011'}.isdisjoint(self.helper.visible_lines()))
        receipts = next(item.value for item in self.helper.app.dataframe
                        if 'Result' in item.value.columns)
        self.assertEqual(set(receipts['Case ID']), {'L004', 'L011'})
        self.assertEqual(receipts['Result'].tolist(), ['APPROVED_FOR_DRAFT'] * 2)
        self.assert_baseline(2)

    def test_single_checkbox_can_prepare_a_review_and_save_an_individual_decision(self):
        self.start_table()
        self.select_line('L004')
        self.assert_single_mode('L004')
        self.assertTrue(all(button.disabled for button in self.helper.decision_buttons()))
        self.helper.widget('button', 'Prepare review').click().run()
        self.helper.widget('button', 'Prepare review workflow').click().run()
        self.helper.assert_clean()
        self.assert_single_mode('L004')
        self.assertFalse(self.helper.invoice_table().value['Select'].any())
        self.assertFalse(self.helper.widget('button', 'Approve for draft').disabled)
        self.helper.widget('text_input', 'Reviewer label').set_value('Synthetic accountant')
        self.helper.widget('text_area', 'Evidence review note').set_value('Checked this one case.')
        self.helper.widget('button', 'Approve for draft').click().run()
        self.helper.assert_clean()
        self.assert_baseline(1)

    def test_single_checkbox_reassessment_keeps_the_case_open_without_a_stale_selection(self):
        self.start_table()
        self.select_line('L004')
        self.helper.widget('button', 'Reassess with current evidence').click().run()
        self.helper.assert_clean()
        self.assert_single_mode('L004')
        self.assertFalse(self.helper.invoice_table().value['Select'].any())
        self.assertFalse(any('changed after you opened' in item.value for item in self.helper.app.warning))
        self.assert_baseline(0)

    def test_bulk_preview_keeps_original_attempt_when_a_new_ready_thread_appears(self):
        self.prepare('L004')
        self.prepare('L011')
        self.start_table()
        self.select_line('L004')
        self.select_line('L011')
        before = self.cases()['L004']
        self.helper.widget('text_input', 'Bulk reviewer label').set_value('Accountant')
        self.helper.widget('text_area', 'Bulk evidence review note').set_value('Reviewed displayed attempt.')
        lifecycle.start_linked_review(self.db, self.cp, 'DEMO', before['assessment_id'],
            idempotency_key='concurrent-bulk-attempt', use_llm=False)
        self.helper.widget('button', 'Approve selected for draft').click().run()
        self.helper.assert_clean()
        self.assertTrue(any('changed' in item.value.lower() for item in self.helper.app.warning))
        self.assertTrue(self.helper.widget('button', 'Approve selected for draft').disabled)
        self.assert_baseline(0)

    def test_single_form_does_not_adopt_a_newer_ready_thread_on_submit(self):
        self.prepare('L004')
        self.start_table()
        self.helper.open_line('L004')
        before = self.cases()['L004']
        self.helper.widget('text_input', 'Reviewer label').set_value('Accountant')
        self.helper.widget('text_area', 'Evidence review note').set_value('Reviewed displayed single attempt.')
        lifecycle.start_linked_review(self.db, self.cp, 'DEMO', before['assessment_id'],
            idempotency_key='concurrent-single-attempt', use_llm=False)
        self.helper.widget('button', 'Approve for draft').click().run()
        self.helper.assert_clean()
        self.assertTrue(all(button.disabled for button in self.helper.decision_buttons()))
        self.assertTrue(any('changed' in item.value.lower() for item in self.helper.app.warning))
        self.assert_baseline(0)

    def test_filter_changes_clear_selection_even_after_returning_to_old_contract(self):
        self.prepare('L004')
        self.start_table()
        self.select_line('L004')
        self.helper.widget('selectbox', 'Vendor').select('SUP-C').run()
        self.helper.widget('selectbox', 'Vendor').select('SUP-A').run()
        self.helper.widget('selectbox', 'Contract').select('CTR-A').run()
        self.assertFalse(self.helper.invoice_table().value['Select'].any())
        self.assertFalse(any(item.label == 'Approve selected for draft' for item in self.helper.app.button))
        self.assert_baseline(0)

    def test_preparation_settings_do_not_carry_consent_to_a_new_attempt(self):
        self.helper.app.run()
        self.helper.open_line('L004')
        self.helper.ai_settings()
        self.helper.provider_consent().check().run()
        self.assertFalse(self.helper.widget('button', 'Prepare review workflow').disabled)
        # Another explicit local attempt appears while the first panel is open.
        self.prepare('L004', key='other-preparation')
        self.helper.widget('button', 'Prepare review workflow').click().run()
        self.helper.assert_clean()
        self.assertFalse(any(item.label == 'Prepare review workflow' for item in self.helper.app.button))
        self.assert_baseline(0)

    def test_unknown_single_selected_amount_is_not_reported_as_zero_or_bulk_approved(self):
        self.helper.app.run()
        self.select_line('L006')
        self.assertEqual(self.helper.selected_line(), 'L006')
        calculation = next(item.value for item in self.helper.app.dataframe
                           if 'Calculation' in item.value.columns)
        self.assertEqual(calculation.iloc[-1]['Amount'], 'Not established')
        self.assertFalse(any('Selected known candidate total:' in item.value
                             for item in self.helper.app.caption))
        self.assertFalse(self.helper.decision_buttons())
        self.assertFalse(any(item.label == 'Approve selected for draft'
                             for item in self.helper.app.button))
        self.assert_baseline(0)


if __name__ == '__main__':
    unittest.main()
