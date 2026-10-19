"""Read-only Streamlit chat interactions against isolated synthetic databases."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import test_ui as ui_helpers

from recovery.storage import connect, load_snapshot


ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec('streamlit') is not None


@unittest.skipUnless(AVAILABLE, 'Install the UI extra to execute chatbot page checks')
class ChatbotUITests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest
        from recovery.chatbot import answer_question

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'synthetic.sqlite3'
        self.cp = self.db.with_name('synthetic-ui-checkpoints.sqlite3')
        load_snapshot(self.db, ROOT / 'data')
        self.before = self.contents(self.db)
        self.live_answer = answer_question
        self.answer = self.start_patch(patch('recovery.chatbot.answer_question',
                                            return_value=self.response()))
        self.network = self.start_patch(patch('recovery.ollama_provider._json_request',
            side_effect=AssertionError('UI tests must not contact a provider')))
        self.app = AppTest.from_string(
            'from pathlib import Path\n'
            'import streamlit as st\n'
            'from recovery.chatbot_ui import render_chatbot\n'
            f'db = Path(st.session_state.get("test_db", {str(self.db)!r}))\n'
            'render_chatbot(st, db, tenant=st.session_state.get("test_tenant", "DEMO"))\n',
            default_timeout=30)

    def start_patch(self, mock_patch):
        result = mock_patch.start()
        self.addCleanup(mock_patch.stop)
        return result

    @staticmethod
    def contents(db):
        with connect(db) as conn:
            return list(conn.iterdump())

    @staticmethod
    def response(**changes):
        value = {'intent': 'case', 'answer': 'L001 has a supported candidate discrepancy of USD 80,000.00.',
            'rows': [{'Case': 'L001', 'Vendor': 'SUP-A', 'Status': 'Candidate'}],
            'citations': [{'doc_id': 'DOC-A2', 'quote': 'Synthetic signed rate: USD 120.00.',
                           'source_sha256': 'a' * 64}],
            'actions': [{'label': 'Open case L001', 'page': 'Invoice review', 'line_id': 'L001'}],
            'scope': {'line_id': 'L001', 'supplier_id': 'SUP-A', 'contract_id': 'CTR-A'}}
        value.update(changes)
        return value

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def ask(self, question):
        field = next(item for item in self.app.selectbox
                     if item.label == 'Ask about a case, contract or vendor')
        # AppTest does not yet serialize accept_new_options values absent from
        # its option list; emulate the browser's accepted custom text.
        displayed = field.format_func(question)
        if displayed not in field.options:
            field.options.append(displayed)
        field.set_value(question)
        self.button('Send message').click().run()
        self.clean()

    def clean(self):
        self.assertFalse(self.app.exception)
        self.network.assert_not_called()
        self.assertFalse(self.cp.exists())
        self.assertEqual(self.contents(self.db), self.before)

    def text(self):
        return '\n'.join(element.value for element in self.app.text)

    def test_open_and_rerender_never_query_or_call_a_provider(self):
        self.app.run().run()
        self.answer.assert_not_called()
        self.assertFalse(self.app.checkbox)
        field = next(item for item in self.app.selectbox
                     if item.label == 'Ask about a case, contract or vendor')
        self.assertTrue(field.proto.accept_new_options)
        self.assertTrue(any(option.startswith('Explain case L001') for option in field.options))
        self.assertIsNone(field.value)
        self.assertEqual(len(self.app.chat_message), 0)
        self.clean()

    def test_explicit_prompt_shows_readable_answer_rows_and_exact_citation(self):
        self.app.run()
        self.button('Explain case L001').click().run()
        self.answer.assert_called_once_with(self.db.resolve(), self.cp.resolve(),
            'Explain case L001', tenant='DEMO', context=None,
            use_llm=False)
        self.assertIn('supported candidate discrepancy', self.text())
        self.assertIn('Synthetic signed rate: USD 120.00.', self.text())
        self.assertEqual(self.app.dataframe[0].value['Case'].tolist(), ['L001'])
        self.assertEqual(self.app.expander[0].label, 'Supporting contract evidence')
        self.assertFalse(self.app.get('json'))
        self.app.run()
        self.answer.assert_called_once()
        self.clean()

    def test_result_table_has_explicit_column_help_and_reports_capped_matches(self):
        from recovery.chatbot_ui import TABLE_HELP, TABLE_LABELS

        row = {field: field for field in TABLE_LABELS}
        self.answer.return_value = self.response(intent='cases', rows=[row], total=75)
        self.app.run()
        self.ask('Show pending cases')
        table = self.app.dataframe[0]
        config = json.loads(table.proto.columns)
        self.assertTrue(config.pop('_index')['hidden'])
        self.assertEqual(set(config), set(TABLE_LABELS.values()))
        self.assertEqual(set(TABLE_HELP), set(TABLE_LABELS.values()))
        for column in table.value.columns:
            self.assertEqual(config[column]['help'], TABLE_HELP[column])
        self.assertEqual(table.value.iloc[0].to_dict(),
                         {TABLE_LABELS[field]: value for field, value in row.items()})
        self.assertTrue(any('Showing 1 of 75 matching records' in item.value for item in self.app.caption))
        self.assertTrue(any('Matching invoice cases' in item.value for item in self.app.markdown))
        self.clean()

    def test_clarification_table_explains_next_step_and_defines_each_column(self):
        from recovery.chatbot_ui import TABLE_HELP

        self.answer.return_value = self.response(intent='clarification', rows=[], choices=[
            {'supplier_id': 'SUP-A', 'contract_id': 'CTR-A'},
            {'supplier_id': 'SUP-B', 'contract_id': 'CTR-B'}], actions=[])
        self.app.run()
        self.ask('Show contracts')
        table = self.app.dataframe[0]
        config = json.loads(table.proto.columns)
        self.assertEqual(table.value['Vendor ID'].tolist(), ['SUP-A', 'SUP-B'])
        for column in table.value.columns:
            self.assertEqual(config[column]['help'], TABLE_HELP[column])
        self.assertTrue(any('Choose the intended record' in item.value for item in self.app.markdown))
        self.assertTrue(any('Include one exact vendor, contract or case identifier' in item.value
                            for item in self.app.caption))
        self.clean()

    def test_no_model_control_and_clearing_conversation_resets_messages(self):
        self.app.run()
        self.assertFalse(self.app.checkbox)
        self.assertFalse(any('Optional: sends' in element.value for element in self.app.caption))
        scope_key = hashlib.sha256(repr((str(self.db.resolve()), 'DEMO')).encode()).hexdigest()[:16]
        self.app.session_state[f'audit_chat:{scope_key}:local_model'] = True
        self.answer.assert_not_called()
        self.ask('Explain case L001')
        self.assertFalse(self.answer.call_args.kwargs['use_llm'])
        self.assertIn('L001', self.text())
        self.button('Clear conversation').click().run()
        self.assertEqual(len(self.app.chat_message), 0)
        self.assertFalse(self.app.checkbox)
        self.assertEqual(self.app.session_state['audit_workspace_chat']['messages'], [])
        self.assertIsNone(self.app.session_state['audit_workspace_chat']['context'])
        self.clean()

    def test_followups_receive_validated_scope_and_memory_is_bounded(self):
        self.app.run()
        self.ask('Explain case L001')
        for _ in range(7):
            self.ask('What contract supports that case?')
        self.assertEqual(self.answer.call_count, 8)
        self.assertEqual(self.answer.call_args.kwargs['context'], self.response()['scope'])
        self.assertEqual(len(self.app.session_state['audit_workspace_chat']['messages']), 12)
        self.clean()

    def test_switching_database_or_tenant_clears_answers(self):
        self.app.run()
        self.ask('Explain case L001')
        second = Path(self.temp.name) / 'other-synthetic.sqlite3'
        shutil.copyfile(self.db, second)
        self.app.session_state['test_db'] = str(second)
        self.app.run()
        self.assertEqual(len(self.app.chat_message), 0)
        self.assertFalse(self.app.checkbox)
        self.ask('Find vendors and contacts')
        self.assertEqual(self.answer.call_args.args[0], second.resolve())
        self.app.session_state['test_tenant'] = 'CANARY'
        self.app.run()
        self.assertEqual(len(self.app.chat_message), 0)
        self.assertFalse(self.app.checkbox)
        self.assertIsNone(self.app.session_state['audit_workspace_chat']['context'])
        self.clean()

    def test_changed_portfolio_invalidates_previous_answer_and_scope(self):
        self.app.run()
        self.ask('Explain case L001')
        stat = self.db.stat()
        os.utime(self.db, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.app.run()
        self.assertEqual(len(self.app.chat_message), 0)
        self.assertTrue(any('portfolio has changed' in element.value for element in self.app.info))
        self.assertIsNone(self.app.session_state['audit_workspace_chat']['context'])
        self.answer.assert_called_once()
        self.clean()

    def test_portfolio_change_during_answer_withholds_result(self):
        def changed(*args, **kwargs):
            stat = self.db.stat()
            os.utime(self.db, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            return self.response()

        self.answer.side_effect = changed
        self.app.run()
        self.ask('Explain case L001')
        self.assertIn('An answer was withheld.', self.text())
        self.assertNotIn('supported candidate discrepancy', self.text())
        self.assertFalse(self.app.dataframe)
        self.assertFalse(any(button.label == 'Open case L001' for button in self.app.button))
        self.clean()

    def test_unavailable_answer_fails_closed_without_workflow_writes(self):
        self.answer.side_effect = RuntimeError('The local model is unavailable.')
        self.app.run()
        self.ask('Explain case L001')
        self.assertIn('An answer was withheld.', self.text())
        self.assertIn('local model is unavailable', self.text())
        self.assertFalse(any(button.label.startswith('Open case') for button in self.app.button))
        self.clean()

    def test_record_navigation_uses_exact_case_and_contract_handoffs(self):
        self.answer.return_value = self.response(actions=[
            {'label': 'Open case L001', 'page': 'Invoice review', 'line_id': 'L001'},
            {'label': 'Open contract CTR-A', 'page': 'Contract term review',
             'supplier_id': 'SUP-A', 'contract_id': 'CTR-A'},
            {'label': 'Not an allowed action', 'page': 'Send email'}])
        self.app.run()
        self.ask('Explain case L001')
        self.assertFalse(any(button.label == 'Not an allowed action' for button in self.app.button))
        self.button('Open case L001').click().run()
        self.assertEqual(self.app.session_state['recovery_page'], 'Invoice review')
        self.assertEqual(self.app.session_state[f'invoice_selection:{self.db.resolve()}'], 'L001')
        self.button('Open contract CTR-A').click().run()
        self.assertEqual(self.app.session_state['recovery_page'], 'Contract term review')
        handoff = self.app.session_state[f'chatbot_open_contract:{self.db.resolve()}:DEMO']
        self.assertEqual(handoff['supplier_id'], 'SUP-A')
        self.assertEqual(handoff['contract_id'], 'CTR-A')
        self.clean()

    def test_source_text_is_not_rendered_as_unsafe_html(self):
        self.answer.return_value = self.response(
            answer='<script>alert("synthetic")</script>',
            citations=[{'doc_id': 'DOC-A2', 'quote': '<img src="http://invalid.test/">'}])
        self.app.run()
        self.ask('Explain case L001')
        self.assertIn('<script>', self.text())
        self.assertIn('<img src=', self.text())
        self.assertFalse(self.app.get('html'))
        self.clean()

    def test_real_service_answers_all_starter_prompts_without_provider_or_writes(self):
        self.answer.side_effect = self.live_answer
        self.app.run()
        for label in ('Explain case L001', 'Find vendors and contacts',
                      'Show pending cases', 'Show portfolio totals'):
            with self.subTest(question=label):
                self.button(label).click().run()
                result = self.app.session_state['audit_workspace_chat']['messages'][-1]['content']
                self.assertNotEqual(result['intent'], 'withheld', result['answer'])
                self.assertNotEqual(result['intent'], 'help', result['answer'])
                self.assertFalse(self.app.get('json'))
                self.clean()
                self.button('Clear conversation').click().run()
        self.assertEqual(self.answer.call_count, 4)

    def test_suggestion_selection_waits_for_send_and_rerenders_do_not_resubmit(self):
        self.app.run()
        field = next(item for item in self.app.selectbox
                     if item.label == 'Ask about a case, contract or vendor')
        self.assertTrue(any(option.startswith('Show contracts for SUP-A') for option in field.options))
        self.assertFalse(any(option.startswith('Show contracts for SUP-X') for option in field.options))
        field.select('Show contracts for SUP-A')
        self.answer.assert_not_called()
        self.button('Send message').click().run()
        self.answer.assert_called_once()
        self.assertEqual(self.answer.call_args.args[2], 'Show contracts for SUP-A')
        self.app.run()
        self.answer.assert_called_once()
        self.clean()

    def test_long_custom_question_is_rejected_without_service_calls(self):
        self.app.run()
        self.ask('x' * 1001)
        self.answer.assert_not_called()
        self.assertTrue(any('1,000 characters' in item.value for item in self.app.warning))
        self.assertEqual(len(self.app.chat_message), 0)

    def test_changed_tenant_refreshes_question_suggestions(self):
        self.app.run()
        self.app.session_state['test_tenant'] = 'OTHER'
        self.app.run()
        options = next(item.options for item in self.app.selectbox
                       if item.label == 'Ask about a case, contract or vendor')
        self.assertFalse(any('AsterTech' in option for option in options))
        self.assertTrue(any('Isolation Canary' in option for option in options))
        self.clean()


@unittest.skipUnless(ui_helpers.AVAILABLE,
                     'Install UI and graph extras to execute chat navigation checks')
class ChatbotAppNavigationTests(unittest.TestCase):
    def setUp(self):
        from recovery.legacy_archive import import_archive
        from recovery.lifecycle import migrate
        from recovery.vendor_contacts import seed_vendor_contacts
        from test_scaled_portfolio_ui import archive_payload

        self.helper = ui_helpers.ReviewWorkflowUITests(methodName='runTest')
        self.addCleanup(self.helper.doCleanups)
        self.helper.setUp()
        self.db, self.cp, self.app = self.helper.db, self.helper.checkpoints, self.helper.app
        migrate(self.db, backup_dir=Path(self.helper.temp.name) / 'backups')
        import_archive(self.db, archive_payload())
        seed_vendor_contacts(self.db, synthetic_confirmed=True)
        self.before = ChatbotUITests.contents(self.db)
        self.answer = self.helper.start_patch(patch('recovery.chatbot.answer_question',
                                                  return_value=ChatbotUITests.response()))

    def clean(self):
        self.helper.assert_clean()
        self.assertFalse(self.app.error, [item.value for item in self.app.error])
        self.helper.resolve.assert_not_called()
        self.helper.generate.assert_not_called()
        self.assertEqual(ChatbotUITests.contents(self.db), self.before)
        self.assertFalse(self.cp.exists())

    def chat(self):
        from recovery.workspace_ui import CHAT_PAGE

        self.app.run()
        self.helper.navigate(CHAT_PAGE)

    def ask(self, question):
        field = next(item for item in self.app.selectbox
                     if item.label == 'Ask about a case, contract or vendor')
        displayed = field.format_func(question)
        if displayed not in field.options:
            field.options.append(displayed)
        field.set_value(question)
        self.helper.widget('button', 'Send message').click().run()
        self.clean()

    def test_chat_navigation_button_is_available_and_read_only(self):
        from recovery.workspace_ui import CHAT_PAGE

        self.chat()
        self.assertEqual(self.app.button(key='navigation:' + CHAT_PAGE).label, CHAT_PAGE)
        self.answer.assert_not_called()
        self.assertFalse(self.app.checkbox)
        self.clean()

    def test_open_case_displays_exact_current_invoice_without_preparing_review(self):
        self.chat()
        self.ask('Explain case L001')
        self.helper.widget('button', 'Open case L001').click().run()
        self.assertEqual(self.app.session_state['recovery_page'], 'Invoice review')
        self.assertEqual(self.helper.selected_line(), 'L001')
        self.assertTrue(any('L001' in element.value for element in self.app.subheader))
        self.clean()

    def test_open_contract_selects_supplier_contract_cases_and_fictional_poc(self):
        from recovery.vendor_contacts import list_vendor_contacts

        self.answer.return_value = ChatbotUITests.response(actions=[{
            'label': 'Open contract CTR-B', 'page': 'Contract term review',
            'supplier_id': 'SUP-B', 'contract_id': 'CTR-B'}])
        self.chat()
        self.ask('Show contract CTR-B')
        self.helper.widget('button', 'Open contract CTR-B').click().run()
        self.assertEqual(self.app.session_state['recovery_page'], 'Contract term review')
        vendors = next(frame.value for frame in self.app.dataframe
                       if 'Vendor POC (fictional)' in frame.value.columns)
        contacts = {row['supplier_id']: row['poc_name'] for row in list_vendor_contacts(self.db)}
        self.assertEqual(vendors.set_index('Vendor ID')['Vendor POC (fictional)'].to_dict(),
                         {supplier: contacts[supplier] for supplier in ('SUP-A', 'SUP-B', 'SUP-C')})
        contracts = next(frame.value for frame in self.app.dataframe
                         if 'Document status' in frame.value.columns)
        self.assertEqual(set(contracts['Contract']), {'CTR-B'})
        cases = next(frame.value for frame in self.app.dataframe
                     if 'Financial assessment' in frame.value.columns)
        self.assertEqual(set(cases['Contract']), {'CTR-B'})
        self.assertTrue(any('Selected vendor: BeaconStaff' in item.value for item in self.app.caption))
        self.assertTrue(any(item.label == 'Inspect selected contract terms' for item in self.app.expander))
        self.clean()

    def test_evidence_result_opens_existing_read_only_pending_queue(self):
        self.answer.return_value = ChatbotUITests.response(
            actions=[{'label': 'Open Evidence Requests', 'page': 'Evidence requests'}],
            rows=[], citations=[], scope={})
        self.chat()
        self.ask('Show pending evidence requests')
        self.helper.widget('button', 'Open Evidence Requests').click().run()
        self.assertEqual(self.app.session_state['recovery_page'], 'Evidence requests')
        self.assertTrue(any(item.value == 'Evidence Requests' for item in self.app.header))
        self.clean()

    def test_open_historical_case_reaches_exact_locked_invoice_beyond_first_page(self):
        case_id = 'LEGACY-L-2025-0125'
        self.answer.return_value = ChatbotUITests.response(actions=[{
            'label': 'Open historical case', 'page': 'Invoice review', 'line_id': case_id,
            'supplier_id': 'LEGACY-V-NORTH', 'contract_id': 'LEGACY-C-2025', 'origin_year': 2025}])
        self.chat()
        self.ask('Show ' + case_id)
        self.helper.widget('button', 'Open historical case').click().run()
        self.assertEqual(self.app.session_state['recovery_page'], 'Invoice review')
        self.assertEqual(self.helper.widget('selectbox', 'Case cohort').value, 'Migrated · 2025 (locked)')
        self.assertTrue(any(case_id in item.value and 'locked' in item.value for item in self.app.subheader))
        self.assertFalse(self.helper.decision_buttons())
        self.assertFalse(any(item.label.startswith(('Prepare review', 'Approve selected'))
                             for item in self.app.button))
        self.clean()


if __name__ == '__main__':
    unittest.main()
