"""Real local Qdrant/graph integration with synthetic, mocked embedding vectors."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recovery.cli import main
from recovery.engine import audit
from recovery.lifecycle import migrate
from recovery.storage import connect, load_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING = {'model': 'demo-embed:latest', 'model_digest': 'a' * 64,
             'base_url': 'http://127.0.0.1:11434'}
REVIEW = {'decision': 'approve', 'reviewer': 'synthetic-reviewer',
          'note': 'Reviewed complete synthetic amendments, settlement and offsets.'}

try:
    AVAILABLE = (importlib.util.find_spec('qdrant_client') is not None
                 and importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None)
except ModuleNotFoundError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Install vector and graph extras for real semantic workflow integration')
class SemanticWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # macOS temp paths may traverse /var -> /private/var. The index API
        # intentionally rejects aliases, so exercise it with the real temp path.
        root = Path(self.temp.name).resolve()
        self.db = root / 'data.sqlite3'
        self.checkpoints = root / 'checkpoints.sqlite3'
        self.index = root / 'evidence-index'
        load_snapshot(self.db, PROJECT_ROOT / 'data')
        migrate(self.db, backup_dir=root / 'backups')
        self.baseline = self.audit()
        self.assertEqual(self.baseline['case_count'], 20)
        self.assertEqual(self.baseline['candidate_total'], '80707.50')
        self.assertEqual(self.baseline['realized_recovery'], '0.00')

        self.http = self.enterContext(patch('recovery.ollama_provider._json_request',
            side_effect=AssertionError('Integration tests must never make HTTP calls')))
        self.resolve = self.enterContext(patch('recovery.semantic_retrieval.resolve_embedding_config',
            side_effect=lambda model, base_url: copy.deepcopy(EMBEDDING)))
        self.embed = self.enterContext(patch('recovery.semantic_retrieval.embed_texts',
            side_effect=self.vectors))

    @staticmethod
    def vectors(texts, config):
        if config != EMBEDDING:
            raise AssertionError('Index and query must use the same pinned embedding config')
        # Each fixture is shorter than one chunk. A1 outranks A2 for the fixed
        # query vector; every unrelated document remains subject to real filters.
        return [[0.0, 1.0] if 'Document ID:' in text and 'Document ID: DOC-A1' not in text
                else [1.0, 0.0] for text in texts]

    def audit(self):
        with connect(self.db) as conn:
            return audit(conn)

    def reviews(self):
        with connect(self.db) as conn:
            return [dict(row) for row in conn.execute('SELECT * FROM review_events ORDER BY event_id')]

    def cli(self, *args):
        with redirect_stdout(io.StringIO()) as output:
            result = main(['--db', str(self.db), *args])
        self.assertEqual(result, 0)
        return json.loads(output.getvalue())

    def build(self):
        return self.cli('index-evidence', '--index', str(self.index),
                        '--embedding-model', EMBEDDING['model'], '--allow-local-embedding')

    def flow(self, thread='semantic-flow', **kwargs):
        from recovery.workflow import run_persistent
        return run_persistent(self.db, self.checkpoints, thread, **kwargs)

    def assert_unchanged(self):
        self.assertEqual(self.audit(), self.baseline)
        self.http.assert_not_called()

    def test_cli_index_and_search_require_consent_before_any_embedding_or_http(self):
        commands = [
            ['index-evidence', '--index', str(self.index), '--embedding-model', EMBEDDING['model']],
            ['search-evidence', '--index', str(self.index), '--line', 'L001', '--query', 'rate'],
        ]
        for command in commands:
            with self.subTest(command=command[0]), redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as raised:
                    main(['--db', str(self.db), *command])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('--allow-local-embedding', errors.getvalue())
        self.assertFalse(self.index.exists())
        self.resolve.assert_not_called()
        self.embed.assert_not_called()
        self.assertEqual(self.reviews(), [])
        self.assert_unchanged()

    def test_cli_build_and_search_real_index_preserve_both_sources_with_one_chunk_hit(self):
        built = self.build()
        self.assertEqual(built['embedding'], EMBEDDING)
        self.assertTrue(built['synthetic_only'])
        self.resolve.assert_called_once_with(EMBEDDING['model'], EMBEDDING['base_url'])
        searched = self.cli('search-evidence', '--index', str(self.index), '--line', 'L001',
                            '--query', 'approved hourly rate', '--limit', '1',
                            '--allow-local-embedding')
        self.assertEqual(searched['generation'], built['generation'])
        self.assertEqual(searched['complete_signed_document_ids'], ['DOC-A1', 'DOC-A2'])
        self.assertTrue(searched['ranking_only'])
        self.assertEqual(len(searched['matches']), 1)
        self.assertEqual(searched['matches'][0]['doc_id'], 'DOC-A1')
        self.assertEqual([doc['doc_id'] for doc in searched['documents']], ['DOC-A1', 'DOC-A2'])
        self.assertTrue(all(doc['tenant_id'] == 'DEMO' for doc in searched['documents']))
        self.assertEqual(self.reviews(), [])
        self.assert_unchanged()

    def test_cli_offline_explain_uses_semantic_ranking_without_dropping_signed_evidence(self):
        self.build()
        self.embed.reset_mock()
        with patch('recovery.ai.resolve_explanation_config') as explain_config:
            with patch('recovery.ai.explain_case') as explain:
                result = self.cli('explain', '--line', 'L001', '--retrieval-index', str(self.index),
                                  '--allow-local-embedding')
        explain_config.assert_not_called()
        explain.assert_not_called()
        self.assertEqual(result['retrieved_doc_ids'], ['DOC-A1', 'DOC-A2'])
        self.assertEqual(result['brief']['mode'], 'offline_template')
        self.assertEqual(result['case']['net_candidate'], '80000.00')
        self.assertEqual(len([call for call in self.embed.call_args_list if call.args[0]]), 1)
        self.assertEqual(self.reviews(), [])
        self.assert_unchanged()

    def test_graph_pins_generation_reads_without_embeddings_and_approves_once(self):
        from recovery.workflow import get_review_workflow, list_review_workflows

        built = self.build()
        started = self.flow(line_id='L001', use_llm=False, retrieval_index=self.index)
        self.assertTrue(started['paused'])
        self.assertEqual(started['next'], ['approval'])
        self.assertEqual(started['state']['retrieval_config'],
                         {'index_path': str(self.index), 'generation': built['generation']})
        self.assertEqual([doc['doc_id'] for doc in started['state']['documents']], ['DOC-A1', 'DOC-A2'])
        self.assertEqual(started['state']['brief']['mode'], 'offline_template')
        self.assertFalse(started['state']['use_llm'])
        self.assertEqual(self.reviews(), [])

        before = self.checkpoints.read_bytes()
        self.embed.reset_mock()
        self.resolve.reset_mock()
        for _ in range(2):
            self.assertEqual(get_review_workflow(self.db, self.checkpoints, 'semantic-flow'), started)
            self.assertEqual(list_review_workflows(self.db, self.checkpoints), [started])
        self.assertEqual(self.checkpoints.read_bytes(), before)
        self.embed.assert_not_called()
        self.resolve.assert_not_called()

        approved = self.flow(decision=REVIEW)
        self.assertFalse(approved['paused'])
        self.assertEqual(approved['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
        self.assertEqual(approved['state']['retrieval_config'], started['state']['retrieval_config'])
        self.assertEqual(approved['state']['documents'], started['state']['documents'])
        committed = self.reviews()
        self.assertEqual(len(committed), 1)
        # Empty-list calls validate saved config locally. Resume must not embed
        # the query again or resolve a different embedding model.
        self.assertTrue(all(not call.args[0] for call in self.embed.call_args_list))
        self.resolve.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'No pending human-review interrupt'):
            self.flow(decision=REVIEW)
        self.assertEqual(self.reviews(), committed)
        self.assertEqual(get_review_workflow(self.db, self.checkpoints, 'semantic-flow'), approved)
        self.assert_unchanged()

    def test_graph_review_replay_after_commit_is_idempotent_with_pinned_index(self):
        from recovery.review import record_review

        self.build()
        started = self.flow(line_id='L001', use_llm=False, retrieval_index=self.index)
        self.embed.reset_mock()

        def commit_then_fail(*args, **kwargs):
            record_review(*args, **kwargs)
            raise RuntimeError('synthetic failure after journal commit')

        with patch('recovery.workflow.record_review', side_effect=commit_then_fail):
            with self.assertRaisesRegex(RuntimeError, 'after journal commit'):
                self.flow(decision=REVIEW)
        committed = self.reviews()
        self.assertEqual(len(committed), 1)
        replayed = self.flow(decision=REVIEW)
        self.assertEqual(replayed['state']['outcome'], committed[0])
        self.assertEqual(replayed['state']['retrieval_config'], started['state']['retrieval_config'])
        self.assertEqual(self.reviews(), committed)
        self.assertTrue(all(not call.args[0] for call in self.embed.call_args_list))
        self.assert_unchanged()

    def _assert_committed_replay_after_index_change(self, *, rebuild):
        from recovery.review import record_review

        self.build()
        started = self.flow(line_id='L001', use_llm=False, retrieval_index=self.index)

        def commit_then_fail(*args, **kwargs):
            record_review(*args, **kwargs)
            raise RuntimeError('synthetic failure after journal commit')

        with patch('recovery.workflow.record_review', side_effect=commit_then_fail):
            with self.assertRaisesRegex(RuntimeError, 'after journal commit'):
                self.flow(decision=REVIEW)
        committed = self.reviews()
        self.assertEqual(len(committed), 1)
        if rebuild:
            replaced = self.build()
            self.assertNotEqual(replaced['generation'], started['state']['retrieval_config']['generation'])
        else:
            # The original path is now unavailable; retain files in the test's
            # temporary directory so this simulates availability, not data loss.
            self.index.rename(self.index.with_name('unavailable-index'))
        self.embed.reset_mock()

        with patch('recovery.workflow.record_review', wraps=record_review) as record:
            finished = self.flow(decision=REVIEW)
        record.assert_called_once()
        self.assertEqual(record.call_args.kwargs['event_id'], committed[0]['event_id'])
        self.assertEqual(record.call_args.kwargs['case_digest'], committed[0]['case_digest'])
        self.assertFalse(finished['paused'])
        self.assertEqual(finished['state']['outcome'], committed[0])
        self.assertEqual(finished['state']['retrieval_config'], started['state']['retrieval_config'])
        # Even after semantic preflight is bypassed, the original journal API
        # must reject a different choice, reviewer or note for this stable key.
        original = dict(event_id=committed[0]['event_id'], thread_id='semantic-flow',
                        tenant='DEMO', line_id='L001', case_digest=committed[0]['case_digest'],
                        **REVIEW)
        for change in ({'decision': 'reject'}, {'reviewer': 'different-reviewer'},
                       {'note': 'different decision context'}):
            with self.subTest(change=change), connect(self.db) as conn:
                with self.assertRaisesRegex(ValueError, 'already used'):
                    record_review(conn, **{**original, **change})
        self.assertEqual(self.reviews(), committed)
        self.assertTrue(all(not call.args[0] for call in self.embed.call_args_list))
        self.assert_unchanged()

    def test_committed_review_finishes_after_semantic_index_rebuild(self):
        self._assert_committed_replay_after_index_change(rebuild=True)

    def test_committed_review_finishes_when_semantic_index_is_unavailable(self):
        self._assert_committed_replay_after_index_change(rebuild=False)

    def test_rebuilt_generation_blocks_pending_approval_before_journal_write(self):
        from recovery.workflow import get_review_workflow

        first = self.build()
        started = self.flow(line_id='L001', use_llm=False, retrieval_index=self.index)
        rebuilt = self.build()
        self.assertNotEqual(rebuilt['generation'], first['generation'])
        self.assertEqual(rebuilt['embedding'], first['embedding'])
        self.assertEqual(rebuilt['source_fingerprint'], first['source_fingerprint'])
        self.embed.reset_mock()
        with patch('recovery.workflow.record_review') as record:
            with self.assertRaisesRegex(ValueError, 'STALE_SEMANTIC_INDEX'):
                self.flow(decision=REVIEW)
            record.assert_not_called()
        saved = get_review_workflow(self.db, self.checkpoints, 'semantic-flow')
        self.assertEqual(saved['state']['retrieval_config'], started['state']['retrieval_config'])
        self.assertEqual(saved['state']['case'], started['state']['case'])
        self.assertNotIn('outcome', saved['state'])
        self.assertTrue(all(not call.args[0] for call in self.embed.call_args_list))
        self.assertEqual(self.reviews(), [])
        self.assert_unchanged()


if __name__ == '__main__':
    unittest.main()
