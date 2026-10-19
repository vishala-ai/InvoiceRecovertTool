"""Hosted explanation workflow and Cloud controls, with no external requests."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from recovery.engine import audit
from recovery.storage import connect, load_snapshot


ROOT = Path(__file__).resolve().parents[1]
try:
    GRAPH = (importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
             and importlib.util.find_spec('pydantic') is not None)
except ModuleNotFoundError:
    GRAPH = False
UI = GRAPH and importlib.util.find_spec('streamlit') is not None
SECRET = 'test-only-placeholder-not-a-real-api-key'
REVIEW = {'decision': 'approve', 'reviewer': 'synthetic-reviewer',
          'note': 'Reviewed all synthetic contractual evidence'}


def hosted_response(**kwargs):
    """Simulated provider output still passes through real independent validation."""
    payload = json.loads(kwargs['user'])
    content = {'claims': payload['required_claims'],
               'next_action': payload['required_next_action'],
               'citations': [{key: value[key] for key in ('doc_id', 'quote')}
                             for value in payload['case']['citations']]}
    return {'model': kwargs['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
        'message': {'role': 'assistant', 'content': json.dumps(content)}}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 80, 'total_tokens': 180}}


class TemporaryPortfolio(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'synthetic.sqlite3'
        self.checkpoints = Path(self.temp.name) / 'checkpoints.sqlite3'
        load_snapshot(self.db, ROOT / 'data')
        env = patch.dict(os.environ, {'GROQ_API_KEY': SECRET, 'RECOVERY_CLOUD': '0',
                                      'GROQ_MODEL': 'llama-3.3-70b-versatile'})
        env.start()
        self.addCleanup(env.stop)

    def assert_finances_and_events(self, count=0):
        with connect(self.db) as conn:
            result = audit(conn)
            self.assertEqual(result['candidate_total'], '80707.50')
            self.assertEqual(result['realized_recovery'], '0.00')
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM review_events').fetchone()[0], count)


@unittest.skipUnless(GRAPH, 'Install graph and explanation extras to execute Groq workflow tests')
class GroqWorkflowTests(TemporaryPortfolio):
    def flow(self, thread='groq-test', **kwargs):
        from recovery.workflow import run_persistent
        return run_persistent(self.db, self.checkpoints, thread, **kwargs)

    def test_complete_evidence_validated_before_pause_then_one_human_decision(self):
        from recovery.groq_provider import BASE_URL, DEFAULT_MODEL
        with patch('recovery.groq_provider.generate', side_effect=hosted_response) as generate, \
             patch('recovery.ollama_provider._json_request') as local_network, \
             patch('recovery.ai._model') as openai:
            paused = self.flow(line_id='L001', use_llm=True, provider='groq')
            self.assertEqual(paused['next'], ['approval'])
            state = paused['state']
            self.assertEqual(state['explanation_config'], {
                'provider': 'groq', 'model': DEFAULT_MODEL, 'base_url': BASE_URL,
                'context_tokens': 16384, 'model_digest': None,
                'prompt_version': 'supplier-explanation-v2-facts'})
            self.assertNotIn(SECRET, json.dumps(state))
            prompt = json.loads(generate.call_args.kwargs['user'])
            self.assertEqual({row['doc_id'] for row in prompt['retrieved_evidence']},
                             {'DOC-A1', 'DOC-A2'})
            self.assertTrue(all(row['body'] and row['sha256']
                                for row in prompt['retrieved_evidence']))
            self.assertEqual(state['brief']['composition'], 'validated_fact_template')
            self.assertEqual(state['brief']['validation']['status'], 'PASSED')
            self.assert_finances_and_events()
            finished = self.flow(decision=REVIEW)
            self.assertEqual(finished['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
            self.assertEqual(finished['state']['explanation_config'], state['explanation_config'])
            with self.assertRaises(ValueError):
                self.flow(decision=REVIEW)
            generate.assert_called_once()
            local_network.assert_not_called()
            openai.assert_not_called()
        self.assert_finances_and_events(1)

    def test_failed_hosted_draft_retries_its_pinned_settings_without_resolving_again(self):
        from recovery.workflow import get_review_workflow, retry_preparation
        with patch('recovery.groq_provider.generate', side_effect=RuntimeError('Hosted unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'Hosted unavailable'):
                self.flow(line_id='L001', use_llm=True, provider='groq')
        failed = get_review_workflow(self.db, self.checkpoints, 'groq-test')
        self.assertEqual(failed['next'], ['draft'])
        with patch.dict(os.environ, {'GROQ_MODEL': 'other-server-model'}), \
             patch('recovery.ai.resolve_explanation_config') as resolve, \
             patch('recovery.groq_provider.generate', side_effect=hosted_response) as generate:
            retried = retry_preparation(self.db, self.checkpoints, 'groq-test',
                                         tenant='DEMO', line_id='L001')
            self.assertEqual(retried['next'], ['approval'])
            self.assertEqual(retried['state']['explanation_config'],
                             failed['state']['explanation_config'])
            self.assertEqual(generate.call_args.kwargs['model'], 'llama-3.3-70b-versatile')
            resolve.assert_not_called()
        self.assert_finances_and_events()

    def test_genuine_quotes_do_not_allow_contradictory_financial_claims_to_pause_for_approval(self):
        from recovery.workflow import get_review_workflow

        def contradicted(**kwargs):
            response = hosted_response(**kwargs)
            draft = json.loads(response['choices'][0]['message']['content'])
            next(row for row in draft['claims']
                 if row['fact_id'] == 'case.net_candidate')['value'] = '0.00'
            response['choices'][0]['message']['content'] = json.dumps(draft)
            return response

        with patch('recovery.groq_provider.generate', side_effect=contradicted):
            with self.assertRaisesRegex(ValueError, 'contradicted'):
                self.flow(line_id='L011', use_llm=True, provider='groq')
        failed = get_review_workflow(self.db, self.checkpoints, 'groq-test')
        self.assertEqual(failed['next'], ['draft'])
        self.assertNotIn('brief', failed['state'])
        with self.assertRaisesRegex(ValueError, 'No pending human-review interrupt'):
            self.flow(decision=REVIEW)
        self.assert_finances_and_events()

    def test_checkpoint_rejects_noncanonical_hosted_endpoint_or_model_digest(self):
        from recovery.ai import resolve_explanation_config
        from recovery.workflow import _pinned_explanation_config
        config = resolve_explanation_config(provider='groq')
        self.assertEqual(_pinned_explanation_config({'explanation_config': config}), config)
        for changed in ({'base_url': 'https://other.invalid/openai/v1'},
                        {'model_digest': 'unexpected-hosted-digest'}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'Unpinned Groq'):
                _pinned_explanation_config({'explanation_config': {**config, **changed}})

    def test_cloud_rejects_local_settings_before_provider_or_index_resolution(self):
        with patch.dict(os.environ, {'RECOVERY_CLOUD': '1'}), \
             patch('recovery.ai.resolve_explanation_config') as resolve, \
             patch('recovery.semantic_retrieval.resolve_index_config') as index:
            for options in ({'use_llm': True, 'provider': 'ollama', 'model': 'llama3.2:latest'},
                            {'use_llm': True, 'provider': 'openai'},
                            {'use_llm': True, 'provider': 'groq', 'model': 'other-model'},
                            {'retrieval_index': 'unused-local-index'}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'Cloud'):
                    self.flow(line_id='L001', **options)
            resolve.assert_not_called()
            index.assert_not_called()
        self.assertFalse(self.checkpoints.exists())
        self.assert_finances_and_events()

    def test_cloud_rejects_resume_when_server_model_changed(self):
        with patch('recovery.groq_provider.generate', side_effect=hosted_response):
            self.flow(line_id='L001', use_llm=True, provider='groq')
        with patch.dict(os.environ, {'RECOVERY_CLOUD': '1', 'GROQ_MODEL': 'different-model'}):
            with self.assertRaisesRegex(ValueError, 'prepare a fresh hosted review'):
                self.flow(decision=REVIEW)
        self.assert_finances_and_events()

    def test_cloud_finishes_exact_committed_decision_after_server_model_changes(self):
        from recovery.review import record_review

        def commit_then_fail(*args, **kwargs):
            record_review(*args, **kwargs)
            raise RuntimeError('Synthetic checkpoint interruption after journal commit')

        with patch.dict(os.environ, {'RECOVERY_CLOUD': '1'}), \
             patch('recovery.groq_provider.generate', side_effect=hosted_response) as generate:
            self.flow(line_id='L001', use_llm=True, provider='groq')
            with patch('recovery.workflow.record_review', side_effect=commit_then_fail):
                with self.assertRaisesRegex(RuntimeError, 'after journal commit'):
                    self.flow(decision=REVIEW)
            self.assert_finances_and_events(1)
            with patch.dict(os.environ, {'GROQ_MODEL': 'different-server-model'}):
                completed = self.flow(decision=REVIEW)
                self.assertFalse(completed['paused'])
                self.assertEqual(completed['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
            generate.assert_called_once()
        self.assert_finances_and_events(1)

    def test_cloud_retry_rejects_legacy_local_provider_before_any_generation(self):
        from recovery.ai import PROMPT_VERSION
        from recovery.workflow import retry_preparation
        config = {'provider': 'ollama', 'model': 'llama3.2:latest',
                  'base_url': 'http://127.0.0.1:11434', 'context_tokens': 16384,
                  'model_digest': 'test-local-model-digest', 'prompt_version': PROMPT_VERSION}
        with patch('recovery.ai.resolve_explanation_config', return_value=config), \
             patch('recovery.ai.explain_case', side_effect=RuntimeError('Local unavailable')):
            with self.assertRaises(RuntimeError):
                self.flow(line_id='L001', use_llm=True, provider='ollama', model=config['model'])
        with patch.dict(os.environ, {'RECOVERY_CLOUD': '1'}), \
             patch('recovery.ai.explain_case') as explain:
            with self.assertRaisesRegex(ValueError, 'prepare a fresh hosted review'):
                retry_preparation(self.db, self.checkpoints, 'groq-test',
                                  tenant='DEMO', line_id='L001')
            explain.assert_not_called()
        self.assert_finances_and_events()

    def test_cli_groq_choice_preserves_validation_and_rejects_local_endpoint_overrides(self):
        from recovery.cli import main
        args = ['--db', str(self.db), 'explain', '--line', 'L001', '--llm', '--provider', 'groq']
        with patch('recovery.groq_provider.generate', side_effect=hosted_response), \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(args), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result['brief']['provider'], 'groq')
        self.assertEqual(result['brief']['validation']['status'], 'PASSED')
        self.assertEqual(set(result['retrieved_doc_ids']), {'DOC-A1', 'DOC-A2'})
        for option in (['--ollama-url', 'http://127.0.0.1:11434'], ['--context-tokens', '8192']):
            with self.subTest(option=option), redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as raised:
                    main(args + option)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('require --provider ollama', errors.getvalue())
        self.assert_finances_and_events()


@unittest.skipUnless(UI, 'Install ui, graph and explanation extras to execute Cloud controls tests')
class GroqPreparationUITests(TemporaryPortfolio):
    def component(self, cloud):
        from streamlit.testing.v1 import AppTest
        env = patch.dict(os.environ, {'RECOVERY_CLOUD': '1' if cloud else '0',
                                      'TEST_PORTFOLIO': str(self.db),
                                      'TEST_CHECKPOINTS': str(self.checkpoints)})
        env.start()
        self.addCleanup(env.stop)
        app = AppTest.from_string('''
import os
from pathlib import Path
import streamlit as st
from recovery.invoice_review import load_worklist
from recovery.invoice_ui import _prepare
db = Path(os.environ['TEST_PORTFOLIO'])
checkpoints = Path(os.environ['TEST_CHECKPOINTS'])
case = load_worklist(db, checkpoints, line_id='L001')['cases'][0]
_prepare(st, db, checkpoints, case, 'DEMO')
''', default_timeout=30)
        app.run()
        self.assertFalse(list(app.exception))
        self.widget(app, 'button', 'Prepare review').click().run()
        self.assertFalse(list(app.exception))
        return app

    @staticmethod
    def widget(app, kind, label):
        return next(item for item in getattr(app, kind) if item.label == label)

    def test_cloud_uses_fixed_hosted_model_and_requires_case_consent_before_generation(self):
        from recovery.workflow import list_review_workflows
        with patch('recovery.groq_provider.generate', side_effect=hosted_response) as generate, \
             patch('recovery.ollama_provider._json_request') as local_network:
            app = self.component(cloud=True)
            self.assertFalse(self.widget(app, 'checkbox', 'AI explanation').value)
            self.assertFalse(any(item.label == 'Use local evidence search' for item in app.checkbox))
            self.widget(app, 'checkbox', 'AI explanation').check().run()
            self.assertFalse(list(app.exception))
            self.assertFalse(any(item.label == 'Explanation provider' for item in app.selectbox))
            self.assertFalse(any(item.label == 'Explanation model' for item in app.text_input))
            self.assertFalse(any('embedding' in item.label.lower() for item in app.checkbox))
            consent = next(item for item in app.checkbox if item.label.startswith('Allow sending'))
            self.assertIn('Groq (hosted Llama)', consent.label)
            self.assertFalse(consent.value)
            self.assertTrue(self.widget(app, 'button', 'Prepare review workflow').disabled)
            generate.assert_not_called()
            self.assertFalse(self.checkpoints.exists())
            consent.check().run()
            self.widget(app, 'button', 'Prepare review workflow').click().run()
            self.assertFalse(list(app.exception))
            self.assertEqual(generate.call_args.kwargs['model'], 'llama-3.3-70b-versatile')
            self.assertEqual(generate.call_args.kwargs['base_url'], 'https://api.groq.com/openai/v1')
            paused = list_review_workflows(self.db, self.checkpoints)[0]
            self.assertEqual(paused['next'], ['approval'])
            self.assertNotIn('retrieval_config', paused['state'])
            local_network.assert_not_called()
        self.assert_finances_and_events()

    def test_cloud_standard_preparation_is_offline_without_embedding_controls(self):
        with patch('recovery.ai.resolve_explanation_config') as resolve, \
             patch('recovery.groq_provider.generate') as generate:
            app = self.component(cloud=True)
            self.assertFalse(any('embedding' in item.label.lower() for item in app.checkbox))
            self.widget(app, 'button', 'Prepare review workflow').click().run()
            self.assertFalse(list(app.exception))
            resolve.assert_not_called()
            generate.assert_not_called()
        from recovery.workflow import list_review_workflows
        state = list_review_workflows(self.db, self.checkpoints)[0]['state']
        self.assertFalse(state['use_llm'])
        self.assertEqual(state['brief']['mode'], 'offline_template')
        self.assert_finances_and_events()

    def test_cloud_displays_safe_authentication_error_without_draft_or_fallback(self):
        from recovery.workflow import list_review_workflows

        connection = Mock()
        connection.sock = None
        response = connection.getresponse.return_value
        response.status = 401
        response.read1.side_effect = AssertionError('Do not read a rejected provider response')
        response.getheader.side_effect = AssertionError('Do not read rejected provider headers')
        with patch('recovery.groq_provider._DeadlineConnection', return_value=connection), \
             patch('recovery.ollama_provider._json_request') as local_network, \
             patch('recovery.ai._model') as openai:
            app = self.component(cloud=True)
            self.widget(app, 'checkbox', 'AI explanation').check().run()
            next(item for item in app.checkbox if item.label.startswith('Allow sending')).check().run()
            self.widget(app, 'button', 'Prepare review workflow').click().run()
            self.assertFalse(list(app.exception))
            errors = [item.value for item in app.error]
            self.assertEqual(len(errors), 1)
            self.assertIn('Groq authentication failed (HTTP 401)', errors[0])
            self.assertNotIn(SECRET, errors[0])
            connection.request.assert_called_once()
            response.read1.assert_not_called()
            response.getheader.assert_not_called()
            local_network.assert_not_called()
            openai.assert_not_called()
        workflow = list_review_workflows(self.db, self.checkpoints)[0]
        self.assertEqual(workflow['next'], ['draft'])
        self.assertFalse(workflow['state'].get('brief'))
        self.assertFalse(workflow['state'].get('outcome'))
        self.assertNotIn(SECRET, json.dumps(workflow['state']))
        self.assert_finances_and_events()

    def test_local_controls_keep_existing_providers_and_add_hosted_llama_with_fresh_consent(self):
        with patch('recovery.ai.resolve_explanation_config') as resolve:
            app = self.component(cloud=False)
            self.assertTrue(any(item.label == 'Use local evidence search' for item in app.checkbox))
            self.widget(app, 'checkbox', 'AI explanation').check().run()
            provider = self.widget(app, 'selectbox', 'Explanation provider')
            self.assertEqual(provider.options, ['Local Ollama', 'OpenAI (cloud)', 'Groq (hosted Llama)'])
            consent = next(item for item in app.checkbox if item.label.startswith('Allow sending'))
            consent.check().run()
            self.widget(app, 'selectbox', 'Explanation provider').select('groq').run()
            self.assertFalse(list(app.exception))
            self.assertEqual(self.widget(app, 'text_input', 'Explanation model').value,
                             'llama-3.3-70b-versatile')
            consent = next(item for item in app.checkbox if item.label.startswith('Allow sending'))
            self.assertFalse(consent.value)
            self.assertIn('Groq', consent.label)
            self.assertTrue(self.widget(app, 'button', 'Prepare review workflow').disabled)
            resolve.assert_not_called()
        self.assert_finances_and_events()
