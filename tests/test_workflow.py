"""Offline integration tests; skipped until the graph extra is installed."""
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from recovery.engine import audit
from recovery.storage import load_snapshot, connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW = {'decision': 'approve', 'reviewer': 'demo', 'note': 'Reviewed synthetic evidence'}
OLLAMA_CONFIG = {'provider': 'ollama', 'model': 'llama3.2:latest',
                 'base_url': 'http://127.0.0.1:11434', 'context_tokens': 16384,
                 'model_digest': 'sha256:' + 'a' * 64,
                 'prompt_version': 'supplier-explanation-v2-facts'}

try:
    AVAILABLE = importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
except ModuleNotFoundError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Install graph extra to verify real LangGraph checkpoint integration')
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'data.sqlite3'
        self.checkpoints = Path(self.temp.name) / 'checkpoints.sqlite3'
        load_snapshot(self.db, PROJECT_ROOT / 'data')

    def flow(self, thread='test-flow', **kwargs):
        from recovery.workflow import run_persistent
        return run_persistent(self.db, self.checkpoints, thread, **kwargs)

    def reviews(self):
        with connect(self.db) as conn:
            self.assertEqual(audit(conn)['realized_recovery'], '0.00')
            return [dict(row) for row in conn.execute('SELECT * FROM review_events')]

    def test_pause_restart_resume_and_no_duplicate_approval(self):
        first = self.flow(line_id='L001')
        self.assertTrue(first['paused'])
        final = self.flow(decision=REVIEW)
        self.assertFalse(final['paused'])
        self.assertEqual(final['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
        self.assertEqual(len(self.reviews()), 1)
        with self.assertRaises(ValueError):
            self.flow(decision=REVIEW)

    def test_workflow_discovery_does_not_create_missing_files(self):
        from recovery.workflow import get_review_workflow, list_review_workflows

        missing_dir = Path(self.temp.name) / 'not-created'
        missing_db = missing_dir / 'data.sqlite3'
        missing_checkpoints = missing_dir / 'checkpoints.sqlite3'
        self.assertIsNone(get_review_workflow(missing_db, missing_checkpoints, 'ui:missing'))
        self.assertEqual(list_review_workflows(missing_db, missing_checkpoints), [])
        self.assertFalse(missing_dir.exists())

    def test_saved_review_is_read_without_models_nodes_or_database_writes(self):
        from recovery.workflow import get_review_workflow, list_review_workflows

        started = self.flow('ui:read-only', line_id='L001')
        before = self.checkpoints.read_bytes()
        nonexistent_db = Path(self.temp.name) / 'never-opened.sqlite3'
        with patch('recovery.workflow.connect') as connect_source:
            with patch('recovery.workflow.record_review') as review:
                with patch('recovery.ai.resolve_explanation_config') as resolve:
                    with patch('recovery.ai.explain_case') as explain:
                        loaded = get_review_workflow(nonexistent_db, self.checkpoints,
                            'ui:read-only', line_id='L001')
                        listed = list_review_workflows(nonexistent_db, self.checkpoints,
                            line_id='L001')
                        self.assertEqual(loaded, started)
                        self.assertEqual(listed, [started])
                        connect_source.assert_not_called()
                        review.assert_not_called()
                        resolve.assert_not_called()
                        explain.assert_not_called()
        self.assertFalse(nonexistent_db.exists())
        self.assertEqual(self.checkpoints.read_bytes(), before)
        self.assertEqual(self.reviews(), [])

    def test_workflow_discovery_scopes_tenant_line_and_thread(self):
        from recovery.workflow import get_review_workflow, list_review_workflows

        first = self.flow('ui:L001', line_id='L001')
        second = self.flow('ui:L011', line_id='L011')
        self.assertEqual(list_review_workflows(self.db, self.checkpoints), [first, second])
        self.assertEqual(list_review_workflows(self.db, self.checkpoints, line_id='L011'),
                         [second])
        self.assertEqual(list_review_workflows(self.db, self.checkpoints, tenant='OTHER'), [])
        with self.assertRaisesRegex(ValueError, 'selected tenant'):
            get_review_workflow(self.db, self.checkpoints, 'ui:L001', tenant='OTHER')
        with self.assertRaisesRegex(ValueError, 'selected invoice line'):
            get_review_workflow(self.db, self.checkpoints, 'ui:L001', line_id='L011')
        self.assertIsNone(get_review_workflow(self.db, self.checkpoints, 'ui:unknown'))
        with self.assertRaisesRegex(ValueError, 'nonempty thread'):
            get_review_workflow(self.db, self.checkpoints, '   ')
        self.assertEqual(self.reviews(), [])

    def test_saved_completed_decision_is_available_after_reconstruction(self):
        from recovery.workflow import get_review_workflow, list_review_workflows

        self.flow('ui:completed', line_id='L001')
        completed = self.flow('ui:completed', decision={**REVIEW, 'decision': 'request_more'})
        self.assertFalse(completed['paused'])
        self.assertEqual(completed['state']['outcome']['decision'], 'NEEDS_MORE_EVIDENCE')
        self.assertEqual(get_review_workflow(self.db, self.checkpoints, 'ui:completed'), completed)
        self.assertEqual(list_review_workflows(self.db, self.checkpoints), [completed])
        self.assertEqual(len(self.reviews()), 1)

    def test_failed_draft_discovery_does_not_claim_a_human_interrupt_or_retry(self):
        from recovery.workflow import get_review_workflow, list_review_workflows

        with patch('recovery.ai.resolve_explanation_config', return_value=OLLAMA_CONFIG):
            with patch('recovery.ai.explain_case', side_effect=RuntimeError('synthetic failure')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                    self.flow('ui:failed', line_id='L001', use_llm=True, provider='ollama',
                              model='llama3.2:latest')
        with patch('recovery.ai.explain_case') as explain:
            failed = get_review_workflow(self.db, self.checkpoints, 'ui:failed')
            self.assertEqual(failed['next'], ['draft'])
            self.assertNotIn('brief', failed['state'])
            self.assertNotIn('outcome', failed['state'])
            self.assertEqual(list_review_workflows(self.db, self.checkpoints), [failed])
            explain.assert_not_called()
        self.assertEqual(self.reviews(), [])

    def test_empty_checkpoint_database_is_not_migrated_during_discovery(self):
        from recovery.workflow import get_review_workflow, list_review_workflows

        sqlite3.connect(self.checkpoints).close()
        before = self.checkpoints.read_bytes()
        self.assertIsNone(get_review_workflow(self.db, self.checkpoints, 'ui:missing'))
        self.assertEqual(list_review_workflows(self.db, self.checkpoints), [])
        self.assertEqual(self.checkpoints.read_bytes(), before)
        self.assertEqual(self.reviews(), [])

    def test_read_discovery_supports_uri_characters_and_checkpoint_symlink(self):
        from recovery.workflow import get_review_workflow, list_review_workflows

        started = self.flow('ui:special-path', line_id='L001')
        special_path = self.checkpoints.with_name('checkpoints #?.sqlite3')
        self.checkpoints.rename(special_path)
        alias = special_path.with_name('checkpoint-alias.sqlite3')
        alias.symlink_to(special_path)
        self.assertEqual(get_review_workflow(self.db, alias, 'ui:special-path'), started)
        self.assertEqual(list_review_workflows(self.db, special_path), [started])

    def test_legacy_checkpoint_reads_do_not_migrate_task_path(self):
        from recovery.workflow import get_review_workflow

        started = self.flow('ui:legacy-storage', line_id='L001')
        with sqlite3.connect(self.checkpoints) as conn:
            conn.execute('ALTER TABLE writes DROP COLUMN task_path')
        self.assertEqual(get_review_workflow(self.db, self.checkpoints, 'ui:legacy-storage'), started)
        with sqlite3.connect(self.checkpoints) as conn:
            columns = {row[1] for row in conn.execute('PRAGMA table_info(writes)')}
        self.assertNotIn('task_path', columns)

    def test_cli_checkpoint_survives_separate_processes(self):
        def command(*args):
            result = subprocess.run(
                [sys.executable, '-m', 'recovery', '--db', str(self.db), *args,
                 '--thread', 'process-flow', '--checkpoints', str(self.checkpoints)],
                cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=30)
            return json.loads(result.stdout)

        first = command('graph-start', '--line', 'L001')
        self.assertTrue(first['paused'])
        self.assertEqual(self.reviews(), [])
        final = command('graph-resume', '--decision', 'approve', '--reviewer', 'demo',
                        '--note', REVIEW['note'])
        self.assertFalse(final['paused'])
        self.assertEqual(final['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
        self.assertEqual(len(self.reviews()), 1)

    def test_stale_case_requires_a_new_review(self):
        first = self.flow(line_id='L001')
        with connect(self.db) as conn, conn:
            conn.execute("UPDATE invoice_lines SET description='changed synthetic evidence' "
                         "WHERE tenant_id='DEMO' AND line_id='L001'")
        with self.assertRaisesRegex(ValueError, 'STALE_CASE'):
            self.flow(decision=REVIEW)
        self.assertEqual(self.reviews(), [])

        refreshed = self.flow('refreshed-flow', line_id='L001')
        self.assertTrue(refreshed['paused'])
        self.assertNotEqual(first['state']['case']['case_digest'],
                            refreshed['state']['case']['case_digest'])
        final = self.flow('refreshed-flow', decision=REVIEW)
        self.assertEqual(final['state']['outcome']['case_digest'],
                         refreshed['state']['case']['case_digest'])
        self.assertEqual(len(self.reviews()), 1)

    def test_reject_and_request_more_do_not_approve(self):
        for decision, outcome in [('reject', 'REJECTED'),
                                  ('request_more', 'NEEDS_MORE_EVIDENCE')]:
            with self.subTest(decision=decision):
                self.assertTrue(self.flow(decision, line_id='L001')['paused'])
                final = self.flow(decision, decision={**REVIEW, 'decision': decision})
                self.assertFalse(final['paused'])
                self.assertEqual(final['state']['outcome']['decision'], outcome)
        self.assertEqual({row['decision'] for row in self.reviews()},
                         {'REJECTED', 'NEEDS_MORE_EVIDENCE'})

    def test_other_tenant_cannot_resume_paused_review(self):
        self.flow(line_id='L001')
        with self.assertRaisesRegex(ValueError, 'selected tenant'):
            self.flow(tenant='OTHER', decision=REVIEW)
        self.assertEqual(self.reviews(), [])
        final = self.flow(decision=REVIEW)
        self.assertEqual(final['state']['outcome']['tenant_id'], 'DEMO')
        self.assertEqual(len(self.reviews()), 1)

    def test_replay_after_journal_commit_is_idempotent(self):
        from recovery.review import record_review

        self.flow(line_id='L001')

        def commit_then_fail(*args, **kwargs):
            record_review(*args, **kwargs)
            # Simulate a failure after the journal commits but before the node returns
            # its outcome for checkpointing. Resume must replay the same event ID.
            raise RuntimeError('simulated failure before outcome checkpoint')

        with patch('recovery.workflow.record_review', side_effect=commit_then_fail) as write:
            with self.assertRaisesRegex(RuntimeError, 'before outcome checkpoint'):
                self.flow(decision=REVIEW)
            self.assertEqual(write.call_count, 1)
        committed = self.reviews()
        self.assertEqual(len(committed), 1)
        final = self.flow(decision=REVIEW)
        self.assertFalse(final['paused'])
        self.assertEqual(final['state']['outcome'], committed[0])
        self.assertEqual(self.reviews(), committed)

    def test_llm_config_survives_failed_draft_and_reconstructed_graph(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from recovery.workflow import build_graph

        with patch('recovery.ai.resolve_explanation_config', return_value=OLLAMA_CONFIG) as resolve:
            with patch('recovery.ai.explain_case', side_effect=RuntimeError('local model unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'local model unavailable'):
                    self.flow(line_id='L001', use_llm=True, provider='ollama', model='llama3.2:latest')
            resolve.assert_called_once_with(provider='ollama', model='llama3.2:latest',
                base_url='http://127.0.0.1:11434', context_tokens=16384)
        self.assertEqual(self.reviews(), [])

        config = {'configurable': {'thread_id': 'test-flow'}}
        # No resolver is allowed during retry: provider, model, digest and prompt
        # version must all come from the stored state rather than ambient defaults.
        with patch('recovery.ai.resolve_explanation_config') as resolve:
            with patch('recovery.ai.explain_case', return_value={'mode': 'mock_local_draft'}) as explain:
                with SqliteSaver.from_conn_string(str(self.checkpoints)) as saver:
                    graph = build_graph(self.db, saver)
                    prior = graph.get_state(config)
                    self.assertEqual(prior.values['explanation_config'], OLLAMA_CONFIG)
                    original_case = prior.values['case']
                # Preparation is a separate explicit retry. A review decision
                # cannot resume a failed generation node as a human interrupt.
                with self.assertRaises(ValueError):
                    self.flow(decision=REVIEW)
                from recovery.workflow import retry_preparation
                paused = retry_preparation(self.db, self.checkpoints, 'test-flow',
                                             tenant='DEMO', line_id='L001')
                self.assertTrue(paused['paused'])
                self.assertEqual(paused['state']['case'], original_case)
                self.assertEqual(paused['state']['explanation_config'], OLLAMA_CONFIG)
                self.assertEqual(self.reviews(), [])
                self.assertEqual(explain.call_args.kwargs, OLLAMA_CONFIG)
                self.assertEqual(explain.call_count, 1)
                final = self.flow(decision=REVIEW)
                self.assertFalse(final['paused'])
                self.assertEqual(final['state']['explanation_config'], OLLAMA_CONFIG)
                self.assertEqual(explain.call_count, 1)
            resolve.assert_not_called()
        self.assertEqual(len(self.reviews()), 1)

    def test_resolution_failure_does_not_fall_back_or_create_a_case(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from recovery.workflow import build_graph

        with patch('recovery.ai.resolve_explanation_config',
                   side_effect=RuntimeError('local model unavailable')) as resolve:
            with patch('recovery.ai.explain_case') as explain:
                with self.assertRaisesRegex(RuntimeError, 'local model unavailable'):
                    self.flow(line_id='L001', use_llm=True, provider='ollama', model='llama3.2:latest')
                explain.assert_not_called()
            self.assertEqual(resolve.call_count, 1)
            self.assertEqual(resolve.call_args.kwargs['provider'], 'ollama')
        with SqliteSaver.from_conn_string(str(self.checkpoints)) as saver:
            state = build_graph(self.db, saver).get_state(
                {'configurable': {'thread_id': 'test-flow'}})
            self.assertFalse(state.values)
        self.assertEqual(self.reviews(), [])

    def test_contradictory_fact_draft_blocks_approval_despite_genuine_quotes(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from recovery.engine import evaluate_line
        from recovery.explanation_facts import build_fact_catalog
        from recovery.retrieval import retrieve_evidence
        from recovery.workflow import build_graph

        with connect(self.db) as conn:
            case = evaluate_line(conn, 'DEMO', 'L011')
            documents = retrieve_evidence(conn, 'DEMO', 'L011')
        draft = {
            'claims': [{'fact_id': fact['fact_id'], 'value': fact['value']}
                       for fact in build_fact_catalog(case, documents)],
            'next_action': 'review_candidate',
            'citations': [{'doc_id': cite['doc_id'], 'quote': cite['quote']}
                          for cite in case['citations']],
        }
        bodies = {document['doc_id']: document['body'] for document in documents}
        for citation in draft['citations']:
            self.assertIn(citation['quote'], bodies[citation['doc_id']])
        next(claim for claim in draft['claims']
             if claim['fact_id'] == 'case.net_candidate')['value'] = '0.00'
        response = {'message': {'content': json.dumps(draft)}, 'prompt_eval_count': 100}
        with patch('recovery.ai.resolve_explanation_config', return_value=OLLAMA_CONFIG):
            with patch('recovery.ollama_provider.generate', return_value=response) as generate:
                with patch('recovery.ollama_provider._json_request') as network:
                    with self.assertRaisesRegex(ValueError, 'contradicted'):
                        self.flow(line_id='L011', use_llm=True, provider='ollama',
                                  model='llama3.2:latest')
                    network.assert_not_called()
                generate.assert_called_once()
        with SqliteSaver.from_conn_string(str(self.checkpoints)) as saver:
            state = build_graph(self.db, saver).get_state(
                {'configurable': {'thread_id': 'test-flow'}})
            self.assertEqual(state.values['case'], case)
            self.assertEqual(state.values['case']['status'], 'CANDIDATE')
            self.assertEqual(state.values['case']['net_candidate'], '100.00')
            self.assertEqual(state.next, ('draft',))
            self.assertTrue(all(not task.interrupts for task in state.tasks))
            self.assertNotIn('brief', state.values)
            self.assertNotIn('outcome', state.values)
        self.assertEqual(self.reviews(), [])

    def test_legacy_unpinned_llm_checkpoint_requires_new_thread(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from recovery.workflow import build_graph

        self.flow(line_id='L001')
        with SqliteSaver.from_conn_string(str(self.checkpoints)) as saver:
            graph = build_graph(self.db, saver)
            graph.update_state({'configurable': {'thread_id': 'test-flow'}}, {'use_llm': True})
        with patch('recovery.ai.resolve_explanation_config') as resolve:
            with patch('recovery.ai.explain_case') as explain:
                with self.assertRaisesRegex(ValueError, 'Unpinned LLM workflow'):
                    self.flow(decision=REVIEW)
                explain.assert_not_called()
            resolve.assert_not_called()
        self.assertEqual(self.reviews(), [])

    def test_legacy_v1_pinned_draft_cannot_resume_into_approval(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from recovery.workflow import build_graph

        # Simulate a previously paused v1 draft. It already has a provider/model
        # pin, but its free-prose contract predates the new fact validation gate.
        self.flow(line_id='L001')
        legacy_config = {**OLLAMA_CONFIG, 'prompt_version': 'supplier-explanation-v1'}
        with SqliteSaver.from_conn_string(str(self.checkpoints)) as saver:
            graph = build_graph(self.db, saver)
            graph.update_state({'configurable': {'thread_id': 'test-flow'}},
                               {'use_llm': True, 'explanation_config': legacy_config})
        with patch('recovery.ai.resolve_explanation_config') as resolve:
            with patch('recovery.ai.explain_case') as explain:
                with self.assertRaisesRegex(ValueError, 'prompt version'):
                    self.flow(decision=REVIEW)
                explain.assert_not_called()
            resolve.assert_not_called()
        self.assertEqual(self.reviews(), [])

    def test_cli_provider_options_require_explicit_llm_and_local_provider(self):
        from recovery.cli import main

        for command in (['explain', '--line', 'L001'],
                        ['graph-start', '--line', 'L001', '--thread', 'option-check']):
            for option in (['--provider', 'ollama'], ['--model', 'llama3.2:latest'],
                           ['--ollama-url', OLLAMA_CONFIG['base_url']],
                           ['--context-tokens', '16384']):
                with self.subTest(command=command[0], option=option[0]):
                    with redirect_stderr(io.StringIO()) as errors:
                        with self.assertRaises(SystemExit) as raised:
                            main(command + option)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn('require --llm', errors.getvalue())
            for option in (['--ollama-url', OLLAMA_CONFIG['base_url']],
                           ['--context-tokens', '16384']):
                with self.subTest(command=command[0], local_option=option[0]):
                    with redirect_stderr(io.StringIO()) as errors:
                        with self.assertRaises(SystemExit):
                            main(command + ['--llm'] + option)
                    self.assertIn('require --provider ollama', errors.getvalue())

    def test_cli_explicit_ollama_config_reaches_explanation_only(self):
        from recovery.cli import main

        with patch('recovery.ai.resolve_explanation_config', return_value=OLLAMA_CONFIG) as resolve:
            with patch('recovery.ai.explain_case', return_value={'mode': 'mock_local_draft'}) as explain:
                with redirect_stdout(io.StringIO()) as output:
                    result = main(['--db', str(self.db), 'explain', '--line', 'L001', '--llm',
                        '--provider', 'ollama', '--model', 'llama3.2:latest',
                        '--ollama-url', OLLAMA_CONFIG['base_url'], '--context-tokens', '16384'])
                self.assertEqual(result, 0)
                resolve.assert_called_once_with(provider='ollama', model='llama3.2:latest',
                    base_url=OLLAMA_CONFIG['base_url'], context_tokens=16384)
                self.assertEqual(explain.call_args.kwargs, OLLAMA_CONFIG)
                case = json.loads(output.getvalue())['case']
                self.assertEqual(case['net_candidate'], '80000.00')
                self.assertEqual(case, explain.call_args.args[0])
        self.assertEqual(self.reviews(), [])

    def test_cli_default_is_offline_and_plain_llm_retains_openai_selection(self):
        from recovery.cli import main

        selected = {**OLLAMA_CONFIG, 'provider': 'openai',
                    'model': 'configured-openai-model', 'model_digest': None}
        args = ['--db', str(self.db), 'explain', '--line', 'L001']
        with patch('recovery.ai.resolve_explanation_config', return_value=selected) as resolve:
            with patch('recovery.ai.explain_case', return_value={'mode': 'mock_cloud_draft'}) as explain:
                with redirect_stdout(io.StringIO()) as output:
                    main(args)
                self.assertEqual(json.loads(output.getvalue())['brief']['mode'], 'offline_template')
                resolve.assert_not_called()
                explain.assert_not_called()
                with redirect_stdout(io.StringIO()):
                    main(args + ['--llm'])
                resolve.assert_called_once_with(provider='openai', model=None,
                    base_url=OLLAMA_CONFIG['base_url'], context_tokens=16384)
                self.assertEqual(explain.call_args.kwargs, selected)
        self.assertEqual(self.reviews(), [])
