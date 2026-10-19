"""Real Streamlit → LangGraph → local journal flows, with synthetic provider responses."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from auth_support import provision_account, sign_in_app
from recovery.engine import audit, evaluate_line
from recovery.storage import connect, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
try:
    AVAILABLE = (importlib.util.find_spec('streamlit') is not None
                 and importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
                 and importlib.util.find_spec('pydantic') is not None)
except ModuleNotFoundError:
    AVAILABLE = False

CONFIG = {'provider': 'ollama', 'model': 'llama3.2:latest',
          'model_digest': 'a' * 64, 'prompt_version': 'supplier-explanation-v2-facts',
          'base_url': 'http://127.0.0.1:11434', 'context_tokens': 16384}
BUTTONS = {'approve': 'Approve for draft', 'reject': 'Reject',
           'request_more': 'Needs review'}
OUTCOMES = {'approve': 'APPROVED_FOR_DRAFT', 'reject': 'REJECTED',
            'request_more': 'NEEDS_MORE_EVIDENCE'}


@unittest.skipUnless(AVAILABLE, 'Install ui, graph and ollama extras to verify the integrated UI')
class ReviewWorkflowUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'synthetic.sqlite3'
        self.checkpoints = self.db.with_name('synthetic-ui-checkpoints.sqlite3')
        self.auth_db = Path(self.temp.name) / 'accounts.sqlite3'
        provision_account(self.auth_db)
        load_snapshot(self.db, ROOT / 'data')
        self.start_patch(patch.dict(os.environ, {'RECOVERY_DB': str(self.db),
                                                'RECOVERY_AUTH_DB': str(self.auth_db)}))
        self.resolve = self.start_patch(patch('recovery.ai.resolve_explanation_config',
                                              return_value=CONFIG))
        self.generate = self.start_patch(patch('recovery.ollama_provider.generate',
                                               side_effect=self.provider_response))
        self.network = self.start_patch(patch('recovery.ollama_provider._json_request',
            side_effect=AssertionError('UI regression must not make network requests')))
        self.cloud = self.start_patch(patch('recovery.ai._model',
            side_effect=AssertionError('UI regression must not call a cloud model')))
        self.app = self.new_app()

    def start_patch(self, mock_patch):
        result = mock_patch.start()
        self.addCleanup(mock_patch.stop)
        return result

    def new_app(self):
        from streamlit.testing.v1 import AppTest

        return sign_in_app(AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30),
                           self.auth_db)

    @staticmethod
    def provider_response(**kwargs):
        """A valid synthetic provider answer; actual schema/fact/citation validation runs."""
        payload = json.loads(kwargs['user'])
        answer = {'claims': payload['required_claims'],
                  'next_action': payload['required_next_action'],
                  'citations': [{key: citation[key] for key in ('doc_id', 'quote')}
                                for citation in payload['case']['citations']]}
        return {'message': {'content': json.dumps(answer)}, 'prompt_eval_count': 100}

    def widget(self, kind, label):
        return next(item for item in getattr(self.app, kind) if item.label == label)

    def search(self, label, value):
        """Confirm native typeahead text, including AppTest's unsupported new values."""
        widget = self.widget('selectbox', label)
        if value and value not in widget.options:
            # AppTest serializes only listed options, while the real browser
            # also submits new strings when accept_new_options is enabled.
            self.assertTrue(widget.proto.accept_new_options)
            widget.options.append(value)
        widget.set_value(value or None).run()

    def navigate(self, page):
        self.app.button(key='navigation:' + page).click().run()
        self.assert_clean()

    def decision_buttons(self):
        return [button for button in self.app.button if button.label in BUTTONS.values()]

    def drafts(self):
        """Inspect persisted drafts while the UI presents their readable views."""
        from recovery.workflow import list_review_workflows

        selected = self.selected_line()
        if not selected:
            return []
        workflows = list_review_workflows(self.db, self.checkpoints,
                                           tenant='DEMO', line_id=selected)
        if not workflows:
            return []
        current = max(workflows, key=lambda row: (row.get('created_at') or '', row['thread_id']))
        brief = current['state'].get('brief')
        return [brief] if brief and brief.get('mode') == 'llm_rag_draft' else []

    def invoice_table(self):
        return next(element for element in self.app.dataframe
                    if element.key and element.key.startswith('invoice_table:'))

    def visible_lines(self):
        return self.invoice_table().value['Case ID'].tolist()

    def selected_line(self):
        try:
            return self.app.session_state[f'invoice_selection:{self.db.resolve()}']
        except KeyError:
            return None

    def open_line(self, line):
        """Deliver the same single-row Open edit as the invoice table."""
        table = self.invoice_table()
        position = self.visible_lines().index(line)
        self.app.session_state[table.key] = {
            'edited_rows': {position: {'Open': True}}, 'added_rows': [], 'deleted_rows': []}
        self.app.run()
        self.assert_clean()
        self.assertEqual(self.selected_line(), line)

    def preparation_settings(self):
        if not any(item.label == 'AI explanation' for item in self.app.checkbox):
            self.widget('button', 'Prepare review').click().run()

    def ai_settings(self):
        self.preparation_settings()
        if not self.widget('checkbox', 'AI explanation').value:
            self.widget('checkbox', 'AI explanation').check().run()

    def provider_consent(self):
        return next(item for item in self.app.checkbox
                    if item.label.startswith('Allow sending this synthetic case'))

    def assert_no_provider_consent(self):
        self.assertTrue(all(not item.value for item in self.app.checkbox
                            if item.label.startswith('Allow sending this synthetic case')))

    def assert_no_enabled_preparation(self):
        self.assertTrue(all(button.disabled for button in self.app.button
                            if button.label in {'Prepare review workflow', 'Retry preparation'}))

    def assert_no_actionable_decisions(self):
        self.assertTrue(all(button.disabled for button in self.decision_buttons()))

    def checkpoint_states(self):
        if not self.checkpoints.exists():
            return {}
        from langgraph.checkpoint.sqlite import SqliteSaver
        from recovery.workflow import build_graph

        with SqliteSaver.from_conn_string(str(self.checkpoints)) as saver:
            threads = {entry.config['configurable']['thread_id'] for entry in saver.list(None)}
            graph = build_graph(self.db, saver)
            return {thread: graph.get_state({'configurable': {'thread_id': thread}})
                    for thread in threads}

    def events(self):
        with connect(self.db) as conn:
            result = audit(conn)
            self.assertEqual(result['case_count'], 20)
            self.assertEqual(result['candidate_total'], '80707.50')
            self.assertEqual(result['realized_recovery'], '0.00')
            return [dict(row) for row in conn.execute('SELECT * FROM review_events')]

    def assert_clean(self):
        self.assertFalse(self.app.exception)
        self.network.assert_not_called()
        self.cloud.assert_not_called()

    def start_review(self, line='L001'):
        self.app.run()
        if line not in self.visible_lines():
            # Repeating completed work is an explicit history action. The
            # default pending queue intentionally excludes those invoices.
            self.widget('radio', 'Review scope').set_value('All cases').run()
        self.open_line(line)
        fresh_review = [item for item in self.app.checkbox
                        if item.label == f'Start a fresh review of {line}']
        if fresh_review:
            fresh_review[0].check().run()
        self.ai_settings()
        self.provider_consent().check().run()
        label = ('Retry preparation' if any(item.label == 'Retry preparation'
                                           for item in self.app.button)
                 else 'Prepare review workflow')
        self.widget('button', label).click().run()
        self.assert_clean()

    def decide(self, decision, reviewer='Synthetic reviewer', note='Reviewed the signed amendment.'):
        self.widget('text_input', 'Reviewer label').set_value(reviewer)
        self.widget('text_area', 'Evidence review note').set_value(note)
        self.widget('button', BUTTONS[decision]).click().run()
        self.assert_clean()

    def calculation_tables(self):
        return [element.value for element in self.app.dataframe
                if list(element.value.columns) == ['Calculation', 'Amount']]

    def saved_calculation(self):
        table = self.calculation_tables()[-1]
        return dict(zip(table['Calculation'], table['Amount'], strict=True))

    def service_tables(self):
        return [element.value for element in self.app.dataframe
                if {'Service date', 'Approved hours', 'Approved hourly rate',
                    'Reviewed term', 'Source document'}.issubset(element.value.columns)]

    def test_readable_explanation_uses_business_facts_without_raw_json(self):
        self.start_review()
        headings = {element.value for kind in ('subheader', 'markdown')
                    for element in getattr(self.app, kind)}
        for heading in ('Invoice calculation', 'Supporting evidence', 'Prepared review'):
            self.assertTrue(any(heading in value for value in headings), heading)
        for element in self.app.dataframe:
            column_config = json.loads(element.proto.columns)
            self.assertTrue(all(column_config[name].get('help')
                                for name in element.value.columns), column_config)
        self.assertEqual(self.saved_calculation(), {
            'Invoice amount': 'USD 1,040,000.00',
            'Amount under approved terms': 'USD 960,000.00',
            'Difference before credits and refunds': 'USD 80,000.00',
            'Recorded pricing credits and refunds': 'USD 0.00',
            'Remaining candidate for review': 'USD 80,000.00',
        })
        self.assertFalse(self.app.json)
        self.assertFalse(any('JSON' in element.label for element in self.app.expander))
        self.assertEqual(len(self.drafts()), 1)
        brief = self.drafts()[0]
        displayed_text = [element.value for kind in ('text', 'markdown')
                          for element in getattr(self.app, kind)]
        self.assertNotIn(brief['summary'], displayed_text)
        for citation in brief['source_citations']:
            self.assertIn(citation['quote'], [element.value for element in self.app.text])
        self.assertEqual(len(self.decision_buttons()), 3)
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.events(), [])

    def test_source_warning_markup_is_escaped_and_source_quote_remains_plain_text(self):
        warning = 'Check **terms** ![image](https://example.invalid/pixel) <b>note</b>'
        quote = '<b>Literal signed quotation</b> ![source](https://example.invalid/source)'

        def assessment_with_untrusted_text(*args, **kwargs):
            case = evaluate_line(*args, **kwargs)
            if case['line_id'] == 'L001':
                # Exercise the presentation boundary without changing sources or validation rules.
                case['warnings'].append(warning)
                case['citations'][0]['quote'] = quote
            return case

        with patch('recovery.engine.evaluate_line', side_effect=assessment_with_untrusted_text):
            self.app.run()
            self.open_line('L001')
        self.assert_clean()
        escaped_warning = (r'Check \*\*terms\*\* '
                           r'\!\[image\]\(https\:\/\/example\.invalid\/pixel\) '
                           r'\<b\>note\<\/b\>')
        markdown = [element.value for element in self.app.markdown]
        self.assertIn(escaped_warning, markdown)
        self.assertNotIn(warning, markdown)
        self.assertIn(quote, [element.value for element in self.app.text])
        self.assertNotIn(quote, markdown)
        self.assertTrue(all(not element.proto.allow_html for element in self.app.markdown
                            if element.value == escaped_warning))
        self.assert_no_actionable_decisions()
        self.generate.assert_not_called()
        self.assertEqual(self.events(), [])

    def test_readable_explanation_distinguishes_gross_offset_and_remaining_amount(self):
        self.start_review('L004')
        amounts = self.saved_calculation()
        self.assertEqual(amounts['Invoice amount'], 'USD 13,000.00')
        self.assertEqual(amounts['Amount under approved terms'], 'USD 12,000.00')
        self.assertEqual(amounts['Difference before credits and refunds'], 'USD 1,000.00')
        self.assertEqual(amounts['Recorded pricing credits and refunds'], 'USD 400.00')
        self.assertEqual(amounts['Remaining candidate for review'], 'USD 600.00')
        self.assertEqual(self.events(), [])

    def test_readable_explanation_preserves_each_service_date_rate(self):
        self.start_review('L011')
        services = self.service_tables()[-1]
        self.assertEqual(services['Service date'].tolist(), ['2026-06-30', '2026-07-01'])
        rates = services['Approved hourly rate'].tolist()
        self.assertTrue(any('USD 130.00' in str(value) for value in rates))
        self.assertTrue(any('USD 120.00' in str(value) for value in rates))
        self.assertEqual(len(set(services['Reviewed term'])), 2)
        self.assertEqual(set(services['Source document']), {'DOC-A1', 'DOC-A2'})
        self.assertEqual(self.saved_calculation()['Remaining candidate for review'], 'USD 100.00')
        self.assertEqual(self.events(), [])

    def test_readable_unresolved_explanation_preserves_unknown_amounts_and_abstention(self):
        self.app.run()
        self.open_line('L006')
        amounts = self.saved_calculation()
        self.assertEqual(amounts['Amount under approved terms'], 'Not established')
        self.assertEqual(amounts['Difference before credits and refunds'], 'Not established')
        self.assertFalse(self.decision_buttons())
        self.assertFalse(self.drafts())
        self.generate.assert_not_called()
        self.assertEqual(self.events(), [])

    def test_readable_saved_explanation_restores_without_generation_or_journal_writes(self):
        self.start_review('L004')
        amounts = self.saved_calculation()
        services = self.service_tables()[-1].to_dict('records')
        before = self.drafts()
        self.app = self.new_app()
        self.app.run()
        self.open_line('L004')
        self.assert_clean()
        self.assertEqual(self.saved_calculation(), amounts)
        self.assertEqual(self.service_tables()[-1].to_dict('records'), services)
        self.assertEqual(self.drafts(), before)
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.events(), [])

    def test_readable_stale_draft_uses_its_saved_case_instead_of_current_amounts(self):
        self.start_review()
        old_draft = self.drafts()[0]
        fixture = Path(self.temp.name) / 'revised-synthetic'
        shutil.copytree(ROOT / 'data', fixture)
        path = fixture / 'invoice_lines.csv'
        with path.open(newline='') as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames
            rows = list(reader)
        line = next(row for row in rows if row['line_id'] == 'L001')
        line.update(billed_rate='131', billed_amount='1048000.00')
        with path.open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        payments_path = fixture / 'payments.csv'
        with payments_path.open(newline='') as source:
            reader = csv.DictReader(source)
            payment_fields = reader.fieldnames
            payments = list(reader)
        next(row for row in payments if row['line_id'] == 'L001')['amount'] = '1048000.00'
        with payments_path.open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=payment_fields)
            writer.writeheader()
            writer.writerows(payments)
        load_snapshot(self.db, fixture)
        self.app = self.new_app()
        self.app.run()
        self.open_line('L001')
        self.assert_clean()
        calculations = self.calculation_tables()
        current = dict(zip(calculations[0]['Calculation'], calculations[0]['Amount'], strict=True))
        self.assertEqual(current['Remaining candidate for review'], 'USD 88,000.00')
        self.assertEqual(self.saved_calculation()['Remaining candidate for review'], 'USD 80,000.00')
        self.assertEqual(self.drafts(), [old_draft])
        self.assertTrue(any('changed' in element.value.lower() for element in self.app.warning))
        self.assertTrue(all(button.disabled for button in self.decision_buttons()))
        self.assert_no_provider_consent()
        self.assertEqual(self.generate.call_count, 1)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM review_events').fetchone()[0], 0)

    def test_no_model_or_review_before_case_specific_consent(self):
        self.app.run()
        self.assert_clean()
        self.assert_no_enabled_preparation()
        self.assertFalse(self.decision_buttons())
        self.resolve.assert_not_called()
        self.generate.assert_not_called()
        self.assertFalse(self.checkpoint_states())
        self.assertEqual(self.events(), [])

        self.open_line('L001')
        self.ai_settings()
        self.assertFalse(self.provider_consent().value)
        self.assertTrue(self.widget('button', 'Prepare review workflow').disabled)
        self.resolve.assert_not_called()
        self.generate.assert_not_called()

        self.start_review()
        self.assertEqual(self.resolve.call_count, 1)
        for request in self.resolve.call_args_list:
            self.assertEqual(request.kwargs, dict(provider='ollama', model='llama3.2:latest',
                base_url=CONFIG['base_url'], context_tokens=CONFIG['context_tokens']))
        self.assertEqual(self.generate.call_count, 1)
        payload = json.loads(self.generate.call_args.kwargs['user'])
        self.assertEqual(payload['case']['line_id'], 'L001')
        self.assertEqual(payload['case']['net_candidate'], '80000.00')
        self.assertEqual({doc['doc_id'] for doc in payload['retrieved_evidence']},
                         {'DOC-A1', 'DOC-A2'})
        draft = self.drafts()[0]
        self.assertEqual(draft['composition'], 'validated_fact_template')
        self.assertIn('Net candidate: USD 80000.00.', draft['summary'])
        self.assertEqual(len(self.decision_buttons()), 3)
        self.assertEqual(self.events(), [])

    def test_standard_preparation_uses_no_model_and_pauses_for_human_decision(self):
        self.app.run()
        self.open_line('L001')
        self.preparation_settings()
        self.assertFalse(self.widget('checkbox', 'AI explanation').value)
        self.assert_no_provider_consent()
        self.widget('button', 'Prepare review workflow').click().run()
        self.assert_clean()
        state = next(iter(self.checkpoint_states().values()))
        self.assertEqual(state.next, ('approval',))
        self.assertFalse(state.values['use_llm'])
        self.assertEqual(state.values['brief']['mode'], 'offline_template')
        self.assertEqual(state.values['case']['net_candidate'], '80000.00')
        self.assertTrue(all(not button.disabled for button in self.decision_buttons()))
        self.resolve.assert_not_called()
        self.generate.assert_not_called()
        self.assertEqual(self.events(), [])

    def test_provider_switch_requires_new_consent(self):
        self.app.run()
        self.open_line('L001')
        self.ai_settings()
        self.provider_consent().check().run()
        self.assertFalse(self.widget('button', 'Prepare review workflow').disabled)
        self.widget('selectbox', 'Explanation provider').select('openai').run()
        self.assertFalse(self.provider_consent().value)
        self.assertTrue(self.widget('button', 'Prepare review workflow').disabled)
        self.widget('selectbox', 'Explanation provider').select('ollama').run()
        self.assertFalse(self.provider_consent().value)
        self.assertTrue(self.widget('button', 'Prepare review workflow').disabled)
        self.resolve.assert_not_called()
        self.generate.assert_not_called()
        self.assert_clean()
        self.assertEqual(self.events(), [])

    def test_line_selection_requires_new_consent_and_never_shows_another_paused_case(self):
        self.start_review()
        self.open_line('L011')
        self.assert_no_provider_consent()
        self.assert_no_enabled_preparation()
        self.assert_no_actionable_decisions()
        self.assertEqual(self.drafts(), [])
        self.assertEqual(self.generate.call_count, 1)
        self.ai_settings()
        self.assertFalse(self.provider_consent().value)
        self.assertTrue(self.widget('button', 'Prepare review workflow').disabled)
        self.provider_consent().check().run()
        self.widget('button', 'Prepare review workflow').click().run()
        self.assert_clean()
        states = self.checkpoint_states()
        self.assertEqual(len(states), 2)
        self.assertEqual({state.values['line_id'] for state in states.values()}, {'L001', 'L011'})
        self.assertTrue(all(state.next == ('approval',) for state in states.values()))
        self.decide('request_more')
        self.assertEqual([event['line_id'] for event in self.events()], ['L011'])

    def test_pause_and_validated_draft_survive_rerun_and_new_app_session(self):
        self.start_review()
        before = self.drafts()
        states = self.checkpoint_states()
        self.assertEqual(len(states), 1)
        thread, saved = next(iter(states.items()))
        self.assertTrue(thread.startswith('ui:'))
        self.assertEqual(saved.next, ('approval',))
        self.assertEqual(saved.values['brief'], before[0])
        self.assertEqual(saved.values['explanation_config'], CONFIG)
        self.assert_no_enabled_preparation()
        self.app.run()
        self.assertEqual(self.drafts(), before)
        self.app = self.new_app()
        self.app.run()
        self.open_line('L001')
        self.assert_clean()
        self.assertEqual(self.drafts(), before)
        self.assertEqual(len(self.decision_buttons()), 3)
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.resolve.call_count, 1)
        self.assertEqual(self.events(), [])
        self.decide('approve')
        self.assertEqual(self.events()[0]['thread_id'], thread)

    def test_all_three_decisions_save_reviewer_note_and_stable_event_once(self):
        for decision, expected in OUTCOMES.items():
            with self.subTest(decision=decision):
                self.app = self.new_app()
                self.start_review()
                paused = {thread: state for thread, state in self.checkpoint_states().items()
                          if state.next == ('approval',)}
                self.assertEqual(len(paused), 1)
                thread, before = next(iter(paused.items()))
                note = f'Synthetic evidence reviewed: {decision}'
                self.decide(decision, reviewer='Finance demo', note=note)
                matching = [event for event in self.events() if event['thread_id'] == thread]
                self.assertEqual(len(matching), 1)
                event = matching[0]
                self.assertEqual(event['event_id'], thread + ':review')
                self.assertEqual(event['decision'], expected)
                self.assertEqual(event['reviewer'], 'Finance demo')
                self.assertEqual(event['note'], note)
                self.assertEqual(event['case_digest'], before.values['case']['case_digest'])
                self.assertFalse(self.decision_buttons())
                self.app.run()
                self.app = self.new_app()
                self.app.run()
                self.assert_clean()
                self.assertEqual([row for row in self.events() if row['thread_id'] == thread],
                                 [event])
                self.assertEqual(self.checkpoint_states()[thread].values['outcome'], event)
        self.assertEqual(len(self.events()), 3)
        self.assertEqual(self.generate.call_count, 3)

    def assert_next_candidate_after_decision(self, decision):
        self.start_review()
        self.decide(decision)
        previous_events = self.events()
        self.assertEqual(len(previous_events), 1)
        self.assertEqual(previous_events[0]['decision'], OUTCOMES[decision])
        if decision == 'request_more':
            # Request-more is retained as outstanding evidence work.
            self.assertIn('L001', self.visible_lines())
        else:
            self.assertNotIn('L001', self.visible_lines())
        self.open_line('L004')
        self.assert_clean()
        self.assertEqual(self.selected_line(), 'L004')
        self.assertEqual(self.drafts(), [])
        self.assert_no_actionable_decisions()
        self.assert_no_provider_consent()
        self.assert_no_enabled_preparation()
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.resolve.call_count, 1)
        self.assertEqual(len(self.checkpoint_states()), 1)
        self.assertEqual(self.events(), previous_events)
        self.assertEqual(self.saved_calculation()['Remaining candidate for review'], 'USD 600.00')

        self.ai_settings()
        self.provider_consent().check().run()
        self.widget('button', 'Prepare review workflow').click().run()
        self.assert_clean()
        self.assertEqual(self.generate.call_count, 2)
        payload = json.loads(self.generate.call_args.kwargs['user'])
        self.assertEqual(payload['case']['line_id'], 'L004')
        self.assertEqual(payload['case']['net_candidate'], '600.00')
        self.assertEqual({doc['doc_id'] for doc in payload['retrieved_evidence']},
                         {'DOC-A1', 'DOC-A2'})
        self.assertEqual(len(self.drafts()), 1)
        self.assertIn('Invoice line L004.', self.drafts()[0]['summary'])
        self.assertIn('Net candidate: USD 600.00.', self.drafts()[0]['summary'])
        self.assertNotIn('Invoice line L001.', self.drafts()[0]['summary'])
        self.assertEqual(len(self.decision_buttons()), 3)
        self.assertEqual(self.events(), previous_events)
        self.assertEqual({state.values['line_id'] for state in self.checkpoint_states().values()},
                         {'L001', 'L004'})

    def test_approved_review_can_open_next_case_without_reusing_the_draft(self):
        self.assert_next_candidate_after_decision('approve')

    def test_rejected_review_can_open_next_case_without_reusing_the_draft(self):
        self.assert_next_candidate_after_decision('reject')

    def test_request_more_evidence_can_advance_without_reusing_the_draft(self):
        self.assert_next_candidate_after_decision('request_more')

    def test_completed_invoice_requires_explicit_fresh_review_confirmation(self):
        self.start_review()
        self.decide('approve')
        previous_events = self.events()
        self.assertNotIn('L001', self.visible_lines())
        self.widget('radio', 'Review scope').set_value('All cases').run()
        self.open_line('L001')
        self.assertTrue(self.widget('button', 'Prepare review').disabled)
        self.widget('checkbox', 'Start a fresh review of L001').check().run()
        self.ai_settings()
        self.assertTrue(self.widget('button', 'Prepare review workflow').disabled)
        self.provider_consent().check().run()
        self.assertFalse(self.widget('button', 'Prepare review workflow').disabled)
        self.widget('button', 'Prepare review workflow').click().run()
        self.assert_clean()
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(self.events(), previous_events)
        self.assertEqual(len(self.decision_buttons()), 3)
        self.assertEqual({state.values['line_id'] for state in self.checkpoint_states().values()},
                         {'L001'})
        self.assertEqual(len(self.checkpoint_states()), 2)

    def test_pending_table_excludes_saved_decisions_after_an_app_restart(self):
        self.start_review()
        self.decide('approve')
        self.start_review('L004')
        self.decide('reject')
        previous_events = self.events()
        self.app = self.new_app()
        self.app.run()
        self.assertNotIn('L001', self.visible_lines())
        self.assertNotIn('L004', self.visible_lines())
        self.open_line('L011')
        self.open_line('L017')
        self.assert_clean()
        self.assertEqual(self.selected_line(), 'L017')
        self.assertEqual(self.drafts(), [])
        self.assert_no_provider_consent()
        self.assert_no_enabled_preparation()
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(self.events(), previous_events)

    def test_no_approval_or_preparation_after_all_current_candidates_are_reviewed(self):
        for line in ('L001', 'L004', 'L011', 'L017'):
            with self.subTest(line=line):
                self.start_review(line)
                self.decide('approve')
        previous_events = self.events()
        self.assertEqual({event['line_id'] for event in previous_events},
                         {'L001', 'L004', 'L011', 'L017'})
        self.assertFalse(self.decision_buttons())
        self.assert_no_enabled_preparation()
        self.assertFalse({'L001', 'L004', 'L011', 'L017'} & set(self.visible_lines()))
        self.assertEqual(self.generate.call_count, 4)
        self.app = self.new_app()
        self.app.run()
        self.assert_clean()
        self.assertFalse(self.decision_buttons())
        self.assert_no_enabled_preparation()
        self.assertFalse({'L001', 'L004', 'L011', 'L017'} & set(self.visible_lines()))
        self.assertEqual(self.generate.call_count, 4)
        self.assertEqual(self.events(), previous_events)

    def test_blank_reviewer_or_note_cannot_save_a_decision(self):
        self.start_review()
        for reviewer, note in [('', 'Checked synthetic evidence'), ('Finance demo', '   ')]:
            with self.subTest(reviewer=reviewer, note=note):
                self.decide('approve', reviewer=reviewer, note=note)
                self.assertTrue(self.app.error)
                self.assertEqual(self.events(), [])
                self.assertEqual(len(self.decision_buttons()), 3)
        self.assertEqual(self.generate.call_count, 1)

    def test_committed_decision_survives_checkpoint_failure_without_alternate_choices(self):
        from recovery.review import record_review

        self.start_review()
        thread = next(iter(self.checkpoint_states()))

        def commit_then_fail(*args, **kwargs):
            record_review(*args, **kwargs)
            raise RuntimeError('Simulated failure after journal commit, before checkpoint')

        with patch('recovery.workflow.record_review', side_effect=commit_then_fail) as write:
            self.decide('approve', reviewer='Finance demo', note='Reviewed all synthetic evidence.')
            write.assert_called_once()
        events = self.events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['decision'], 'APPROVED_FOR_DRAFT')
        self.assertEqual(events[0]['event_id'], thread + ':review')
        self.assertFalse(self.decision_buttons())
        self.assertFalse(self.app.error)
        self.assertTrue(any('Approved For Draft' in item.value for item in self.app.success))
        self.assertEqual(self.checkpoint_states()[thread].next, ('approval',))

        self.app = self.new_app()
        self.app.run()
        self.assert_clean()
        # A restart prioritizes finishing the already committed decision;
        # recovery must not require another decision or provider call.
        self.assertIn('L001', self.visible_lines())
        self.open_line('L001')
        self.assert_clean()
        self.assertFalse(self.decision_buttons())
        self.assertTrue(any('Approved For Draft' in item.value for item in self.app.success))
        self.widget('button', 'Finish workflow completion').click().run()
        self.assert_clean()
        state = self.checkpoint_states()[thread]
        self.assertFalse(state.next)
        self.assertEqual(state.values['outcome'], events[0])
        self.assertEqual(self.events(), events)
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.resolve.call_count, 1)
        self.assertFalse(self.decision_buttons())

    def test_contradicted_draft_with_genuine_quotes_never_reaches_review_pause(self):
        def contradicted(**kwargs):
            response = self.provider_response(**kwargs)
            draft = json.loads(response['message']['content'])
            next(claim for claim in draft['claims']
                 if claim['fact_id'] == 'case.net_candidate')['value'] = '0.00'
            response['message']['content'] = json.dumps(draft)
            return response

        self.generate.side_effect = contradicted
        self.start_review()
        self.assertTrue(self.app.error)
        self.assertFalse(self.drafts())
        self.assert_no_actionable_decisions()
        state = next(iter(self.checkpoint_states().values()))
        self.assertEqual(state.next, ('draft',))
        self.assertNotIn('brief', state.values)
        self.assertNotIn('outcome', state.values)
        self.assertEqual(state.values['case']['net_candidate'], '80000.00')
        self.app.run()
        self.app = self.new_app()
        self.app.run()
        self.assert_clean()
        self.assertFalse(self.decision_buttons())
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.events(), [])

    def test_provider_unavailable_withholds_draft_without_cloud_fallback(self):
        self.generate.side_effect = RuntimeError('Local Ollama is unavailable')
        self.start_review()
        self.assertTrue(any('unavailable' in item.value for item in self.app.error))
        self.assertFalse(self.drafts())
        self.assert_no_actionable_decisions()
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.events(), [])

    def test_failed_provider_preflight_never_generates_or_exposes_decision_controls(self):
        self.resolve.side_effect = RuntimeError('Selected local model is not installed')
        self.start_review()
        self.assertTrue(any('not installed' in item.value for item in self.app.error))
        self.assertFalse(self.drafts())
        self.assert_no_actionable_decisions()
        self.assertFalse(self.checkpoint_states())
        self.generate.assert_not_called()
        self.assertEqual(self.events(), [])

    def test_explicit_retry_reuses_pinned_thread_after_withheld_draft(self):
        self.generate.side_effect = RuntimeError('Local Ollama is unavailable')
        self.start_review()
        failed_thread = next(iter(self.checkpoint_states()))
        self.assert_no_actionable_decisions()
        self.generate.side_effect = self.provider_response
        self.start_review()
        self.assert_clean()
        states = self.checkpoint_states()
        self.assertEqual(len(states), 1)
        self.assertEqual(states[failed_thread].next, ('approval',))
        self.decide('reject')
        self.assertEqual(self.events()[0]['thread_id'], failed_thread)
        self.assertEqual(self.generate.call_count, 2)

    def test_unresolved_evidence_case_cannot_be_approved_after_a_valid_draft(self):
        from recovery.workflow import run_persistent

        # Restore a valid CLI-prepared explanation; the invoice UI itself
        # sends noncandidates to evidence handling instead of approval prep.
        run_persistent(self.db, self.checkpoints, 'cli:L006', line_id='L006',
            use_llm=True, provider='ollama', model=CONFIG['model'])
        self.app.run()
        self.open_line('L006')
        self.assertEqual(self.drafts()[0]['composition'], 'validated_fact_template')
        self.assertFalse(self.decision_buttons())
        state = next(iter(self.checkpoint_states().values()))
        self.assertEqual(state.values['case']['status'], 'NEEDS_REVIEW')
        self.assertFalse(state.next)
        self.assertNotIn('outcome', state.values)
        self.assertEqual(self.events(), [])

    def test_reimported_source_requires_new_consent_and_review_before_approval(self):
        self.start_review()
        old_thread, old = next(iter(self.checkpoint_states().items()))
        fixture = Path(self.temp.name) / 'changed-synthetic'
        shutil.copytree(ROOT / 'data', fixture)
        path = fixture / 'invoice_lines.csv'
        with path.open(newline='') as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames
            rows = list(reader)
        next(row for row in rows if row['line_id'] == 'L001')['description'] += ' revised'
        with path.open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        load_snapshot(self.db, fixture)
        # A user may click a form rendered before re-import; the server must recheck.
        self.decide('approve')
        self.assertEqual(self.events(), [])
        self.assertTrue(self.app.error or self.app.warning)
        self.assert_no_provider_consent()
        self.assertTrue(all(button.disabled for button in self.decision_buttons()))
        self.assert_no_enabled_preparation()
        self.ai_settings()
        self.assertFalse(self.provider_consent().value)
        self.provider_consent().check().run()
        self.widget('button', 'Prepare review workflow').click().run()
        self.assert_clean()
        self.assertEqual(self.generate.call_count, 2)
        new = [state for thread, state in self.checkpoint_states().items() if thread != old_thread]
        self.assertEqual(len(new), 1)
        self.assertNotEqual(old.values['case']['case_digest'], new[0].values['case']['case_digest'])
        self.assertNotEqual(old.values['case']['source_digest'], new[0].values['case']['source_digest'])
        self.decide('approve')
        self.assertEqual(len(self.events()), 1)
        self.assertNotEqual(self.events()[0]['thread_id'], old_thread)
