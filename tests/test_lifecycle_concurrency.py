"""Linked-review dispatch races and interrupted preparation, with no live providers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import tempfile
from threading import Event, Lock
import unittest
from unittest.mock import patch

import test_lifecycle as fixtures
from recovery import lifecycle
from recovery.explanation_facts import build_fact_catalog, validate_fact_draft
from recovery.review import record_review
from recovery.storage import connect, load_snapshot
from recovery.workflow import get_review_workflow, run_persistent

DATA = Path(__file__).resolve().parents[1] / 'data'
try:
    GRAPH_AVAILABLE = importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
except ModuleNotFoundError:
    GRAPH_AVAILABLE = False
PINNED_CONFIG = {'provider': 'ollama', 'model': 'synthetic-local:1b',
                 'base_url': 'http://127.0.0.1:11434', 'context_tokens': 16384,
                 'model_digest': 'a' * 64, 'prompt_version': 'supplier-explanation-v2-facts'}


@unittest.skipUnless(GRAPH_AVAILABLE, 'Install graph extra for linked review concurrency checks')
class LifecycleConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'synthetic.sqlite3'
        self.checkpoint = self.directory / 'checkpoints.sqlite3'
        load_snapshot(self.db, DATA)
        lifecycle.migrate(self.db, backup_dir=self.directory / 'backups')
        self.assessment = lifecycle.reassess(self.db, 'DEMO', 'L004',
                                              idempotency_key='current-assessment')

    def start(self, **kwargs):
        options = {'idempotency_key': 'same-linked-start', 'use_llm': False}
        options.update(kwargs)
        return lifecycle.start_linked_review(self.db, self.checkpoint, 'DEMO',
                                             self.assessment['assessment_id'], **options)

    def test_concurrent_same_key_starts_dispatch_once_and_share_saved_review(self):
        first_dispatch = Event()
        second_attempt = Event()
        second_dispatch = Event()
        release_first = Event()
        counter_lock = Lock()
        dispatches = []

        def controlled_dispatch(*args, **kwargs):
            with counter_lock:
                dispatches.append(args[2])
                ordinal = len(dispatches)
            if ordinal == 1:
                first_dispatch.set()
                if not release_first.wait(5):
                    raise AssertionError('Concurrent start test did not release first dispatch')
            else:
                second_dispatch.set()
            return run_persistent(*args, **kwargs)

        def second_start():
            second_attempt.set()
            return self.start()

        with patch('recovery.workflow.run_persistent', side_effect=controlled_dispatch):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(self.start)
                try:
                    self.assertTrue(first_dispatch.wait(5))
                    second = pool.submit(second_start)
                    self.assertTrue(second_attempt.wait(5))
                    # The first dispatch is deliberately held before its checkpoint
                    # exists. A duplicate caller must wait, not start another graph.
                    duplicated_while_pending = second_dispatch.wait(0.2)
                finally:
                    release_first.set()
                views = [first.result(timeout=10), second.result(timeout=10)]
        self.assertFalse(duplicated_while_pending)
        self.assertEqual(len(dispatches), 1)
        self.assertEqual(views[0], views[1])
        self.assertTrue(views[0]['paused'])
        self.assertEqual(views[0]['next'], ['approval'])
        self.assertEqual(get_review_workflow(self.db, self.checkpoint, views[0]['thread_id']), views[0])
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_intents').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)
        saved = run_persistent(self.db, self.checkpoint, views[0]['thread_id'], decision={
            'decision': 'approve', 'reviewer': 'synthetic-reviewer',
            'note': 'Reviewed one saved synthetic assessment after concurrent starts.'})
        self.assertEqual(saved['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
        self.assertEqual(self.start(), saved)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)

    def failed_preparation(self):
        options = {'use_llm': True, 'provider': 'ollama', 'model': 'synthetic-local:1b'}
        with patch('recovery.ai.resolve_explanation_config', return_value=PINNED_CONFIG):
            with patch('recovery.ai.explain_case',
                       side_effect=RuntimeError('Injected draft generation failure')):
                with self.assertRaisesRegex(RuntimeError, 'Injected'):
                    self.start(**options)
        with connect(self.db) as conn:
            intent = dict(conn.execute('SELECT * FROM review_intents').fetchone())
        saved = get_review_workflow(self.db, self.checkpoint, intent['thread_id'])
        self.assertEqual(saved['next'], ['draft'])
        self.assertNotIn('brief', saved['state'])
        self.assertNotIn('outcome', saved['state'])
        return options, saved

    def test_partial_checkpoint_retry_returns_preparation_without_regeneration_or_approval(self):
        options, pending = self.failed_preparation()
        with patch('recovery.ai.resolve_explanation_config') as configuration:
            with patch('recovery.ai.explain_case') as generation:
                with patch('recovery.workflow.run_persistent') as dispatch:
                    retried = self.start(**options)
        configuration.assert_not_called()
        generation.assert_not_called()
        dispatch.assert_not_called()
        self.assertEqual(retried, pending)
        self.assertEqual(retried['next'], ['draft'])
        self.assertNotIn('brief', retried['state'])
        self.assertNotIn('outcome', retried['state'])
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_intents').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)
        self.assertEqual(retried['state']['explanation_config'], PINNED_CONFIG)

    def test_failed_preparation_is_not_a_human_interrupt_and_cannot_accept_decision(self):
        _, pending = self.failed_preparation()
        with patch('recovery.ai.explain_case') as generation:
            with self.assertRaises(ValueError):
                run_persistent(self.db, self.checkpoint, pending['thread_id'], decision={
                    'decision': 'approve', 'reviewer': 'synthetic-reviewer',
                    'note': 'There is no reviewed draft or pending human interrupt.'})
        generation.assert_not_called()
        self.assertEqual(get_review_workflow(self.db, self.checkpoint, pending['thread_id']), pending)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)

    def test_explicit_retry_preparation_uses_pinned_model_and_reaches_one_human_review(self):
        options, pending = self.failed_preparation()

        def fixed_generation(case, documents, **_kwargs):
            claims = [{'fact_id': fact['fact_id'], 'value': fact['value']}
                      for fact in build_fact_catalog(case, documents)]
            return validate_fact_draft({'claims': claims, 'next_action': 'review_candidate',
                'citations': [{'doc_id': item['doc_id'], 'quote': item['quote']}
                              for item in case['citations']]}, case, documents)

        with patch('recovery.ai.resolve_explanation_config') as configuration:
            with patch('recovery.ai.explain_case', side_effect=fixed_generation) as generation:
                paused = self.start(**options, retry_preparation=True)
        configuration.assert_not_called()
        self.assertEqual(generation.call_count, 1)
        self.assertEqual(generation.call_args.kwargs, PINNED_CONFIG)
        self.assertEqual(paused['thread_id'], pending['thread_id'])
        self.assertEqual(paused['next'], ['approval'])
        self.assertNotIn('outcome', paused['state'])
        self.assertTrue(paused['state']['brief']['summary'])
        with patch('recovery.ai.explain_case') as generation:
            self.assertEqual(self.start(**options, retry_preparation=True), paused)
        generation.assert_not_called()
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)
        saved = run_persistent(self.db, self.checkpoint, paused['thread_id'], decision={
            'decision': 'approve', 'reviewer': 'synthetic-reviewer',
            'note': 'Reviewed fresh explanation after explicit generation retry.'})
        self.assertEqual(saved['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
        with patch('recovery.ai.explain_case') as generation:
            self.assertEqual(self.start(**options, retry_preparation=True), saved)
        generation.assert_not_called()
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)

    def test_same_start_key_cannot_be_reused_with_another_configuration(self):
        saved = self.start()
        with patch('recovery.workflow.run_persistent') as dispatch:
            with self.assertRaisesRegex(ValueError, 'different payload'):
                self.start(model='another-synthetic-local:1b')
        dispatch.assert_not_called()
        self.assertEqual(get_review_workflow(self.db, self.checkpoint, saved['thread_id']), saved)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_intents').fetchone()[0], 1)

    def test_pending_intent_cannot_dispatch_after_its_assessment_becomes_stale(self):
        with patch('recovery.workflow.run_persistent',
                   side_effect=RuntimeError('Injected failure before first checkpoint')):
            with self.assertRaisesRegex(RuntimeError, 'Injected'):
                self.start()
        staged = lifecycle.stage_document(self.db, 'DEMO', filename='synthetic-pending.md',
            content=b'SYNTHETIC ENGINEER rate amendment at REMOTE; pricing effect needs review.',
            synthetic_confirmed=True)
        lifecycle.admit_document(self.db, 'DEMO', staged['stage_id'],
            scope={'supplier_id': 'SUP-A', 'contract_id': 'CTR-A',
                   'role': 'ENGINEER', 'location': 'REMOTE'},
            title='Synthetic amendment arriving after review intent', signed_date='2026-07-20',
            status='SIGNED', synthetic_confirmed=True, signed_simulation_confirmed=True,
            reviewer='synthetic-reviewer', note='Potentially governing signed synthetic evidence.',
            idempotency_key='changed-source-after-intent')
        with patch('recovery.workflow.run_persistent') as dispatch:
            with self.assertRaises(ValueError):
                self.start()
        dispatch.assert_not_called()
        self.assertFalse(self.checkpoint.exists())
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_intents').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)

    def test_same_input_reassessment_blocks_old_paused_thread_and_requires_new_review(self):
        old = self.start()
        old_assessment = old['state']['assessment_id']
        self.assertEqual(old_assessment, self.assessment['assessment_id'])
        revised = lifecycle.reassess(self.db, 'DEMO', 'L004',
                                       idempotency_key='same-input-new-assessment')
        self.assertNotEqual(revised['assessment_id'], old_assessment)
        self.assertEqual(revised['case_digest'], old['state']['case']['case_digest'])
        with self.assertRaisesRegex(ValueError, 'newer assessment'):
            run_persistent(self.db, self.checkpoint, old['thread_id'], decision={
                'decision': 'approve', 'reviewer': 'synthetic-reviewer',
                'note': 'An unchanged amount cannot bypass the new assessment revision.'})
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)
        self.assessment = revised
        fresh = self.start(idempotency_key='same-input-fresh-review')
        self.assertNotEqual(fresh['thread_id'], old['thread_id'])
        self.assertEqual(fresh['state']['assessment_id'], revised['assessment_id'])
        self.assertEqual(fresh['next'], ['approval'])
        self.assertNotIn('outcome', fresh['state'])
        saved = run_persistent(self.db, self.checkpoint, fresh['thread_id'], decision={
            'decision': 'approve', 'reviewer': 'synthetic-reviewer',
            'note': 'Reviewed the latest assessment as a separate decision.'})
        self.assertEqual(saved['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
        with connect(self.db) as conn:
            edges = [dict(row) for row in conn.execute('SELECT * FROM case_review_assessments')]
            self.assertEqual(len(edges), 1)
            self.assertEqual(edges[0]['assessment_id'], revised['assessment_id'])
            self.assertEqual(edges[0]['thread_id'], fresh['thread_id'])
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)

    def test_same_input_new_assessment_does_not_inherit_completed_approval(self):
        old = self.start()
        approved = run_persistent(self.db, self.checkpoint, old['thread_id'], decision={
            'decision': 'approve', 'reviewer': 'synthetic-reviewer',
            'note': 'Approved this specific earlier assessment revision.'})
        self.assertEqual(approved['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
        revised = lifecycle.reassess(self.db, 'DEMO', 'L004',
                                       idempotency_key='reassess-after-approval')
        self.assertEqual(revised['case_digest'], old['state']['case']['case_digest'])
        self.assessment = revised
        fresh = self.start(idempotency_key='review-after-historical-approval',
                             prior_thread_id=old['thread_id'])
        self.assertEqual(fresh['state']['assessment_id'], revised['assessment_id'])
        self.assertEqual(fresh['next'], ['approval'])
        self.assertNotIn('outcome', fresh['state'])
        self.assertEqual(get_review_workflow(self.db, self.checkpoint, old['thread_id']), approved)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)
            self.assertFalse(conn.execute('SELECT 1 FROM case_review_assessments WHERE assessment_id=?',
                                          (revised['assessment_id'],)).fetchone())
            historical = conn.execute('SELECT assessment_id FROM case_review_assessments').fetchone()[0]
            self.assertEqual(historical, old['state']['assessment_id'])

    def test_graph_payment_evidence_request_type_is_preserved_and_replay_cannot_change_it(self):
        paused = self.start()
        decision = {'decision': 'request_more', 'reviewer': 'synthetic-reviewer',
                    'note': 'Need separate payment allocation evidence.', 'evidence_type': 'PAYMENT'}
        saved = run_persistent(self.db, self.checkpoint, paused['thread_id'], decision=decision)
        self.assertFalse(saved['paused'])
        outcome = saved['state']['outcome']
        self.assertEqual(outcome['decision'], 'NEEDS_MORE_EVIDENCE')
        requests = lifecycle.list_requests(self.db, 'DEMO')
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['request_type'], 'PAYMENT')
        self.assertEqual(requests[0]['review_event_id'], outcome['event_id'])
        arguments = {'event_id': outcome['event_id'], 'thread_id': paused['thread_id'],
                     'tenant': 'DEMO', 'line_id': 'L004', **decision,
                     'case_digest': paused['state']['case']['case_digest'],
                     'assessment_id': paused['state']['assessment_id']}
        with connect(self.db) as conn:
            self.assertEqual(record_review(conn, **arguments), outcome)
            with self.assertRaisesRegex(ValueError, 'different evidence request type'):
                record_review(conn, **{**arguments, 'evidence_type': 'CONTRACT_RATE'})
        self.assertEqual(lifecycle.list_requests(self.db, 'DEMO'), requests)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT count(*) FROM case_review_assessments').fetchone()[0], 1)


@unittest.skipUnless(importlib.util.find_spec('pydantic') is not None,
                     'Install ollama extra for term-proposal source-support verification')
class LifecycleSourceSupportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'synthetic.sqlite3'
        load_snapshot(self.db, DATA)
        lifecycle.migrate(self.db, backup_dir=self.directory / 'backups')

    def generate(self, values):
        with patch('recovery.term_extraction.extract_terms',
                   side_effect=lambda source, **_kwargs: fixtures.extraction(source, values)):
            return lifecycle.generate_proposal(self.db, 'DEMO', 'DOC-A2', model='synthetic-local:1b')

    def approve(self, proposal, key):
        return lifecycle.review_terms(self.db, 'DEMO', proposal['proposal_id'],
            proposal_digest=proposal['digest'], decision='approve', reviewer='synthetic-reviewer',
            note='Checked complete supported source scope and quotations.',
            idempotency_key=key, coverage_confirmed=True)

    def test_valid_but_unquoted_effective_dates_cannot_be_promoted(self):
        with connect(self.db) as conn:
            terms_before = [tuple(row) for row in conn.execute('SELECT * FROM reviewed_terms')]
        for index, change in enumerate(({'effective_from': '2026-07-02'},
                                        {'effective_to': '2026-12-31'})):
            proposal = self.generate({**fixtures.values(), **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.approve(proposal, f'unsupported-date:{index}')
        with connect(self.db) as conn:
            self.assertEqual([tuple(row) for row in conn.execute('SELECT * FROM reviewed_terms')],
                             terms_before)
            self.assertEqual(conn.execute('SELECT count(*) FROM term_review_events').fetchone()[0], 0)

    def test_rate_quote_must_also_establish_currency_and_hourly_unit(self):
        with connect(self.db) as conn:
            terms_before = [tuple(row) for row in conn.execute('SELECT * FROM reviewed_terms')]
        for index, quote in enumerate(('120.00', 'USD at 120.00', '120.00 per HOUR')):
            proposal = self.generate({**fixtures.values(), 'exact_quote': quote})
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                self.approve(proposal, f'incomplete-rate-quote:{index}')
        with connect(self.db) as conn:
            self.assertEqual([tuple(row) for row in conn.execute('SELECT * FROM reviewed_terms')],
                             terms_before)
            self.assertEqual(conn.execute('SELECT count(*) FROM term_review_events').fetchone()[0], 0)

    def test_supersession_target_cannot_match_only_a_substring_of_another_term_id(self):
        body = fixtures.QUOTE_115.replace('TERM-A2', 'TERM-A20')
        staged = lifecycle.stage_document(self.db, 'DEMO', filename='synthetic-target.md',
            content=body.encode(), synthetic_confirmed=True)
        source = lifecycle.admit_document(self.db, 'DEMO', staged['stage_id'],
            scope=fixtures.SCOPE, title='Synthetic amendment referencing a different term',
            signed_date='2026-07-20', status='SIGNED', synthetic_confirmed=True,
            signed_simulation_confirmed=True, reviewer='synthetic-reviewer',
            note='Known synthetic scope; explicit target still needs term review.',
            idempotency_key='different-explicit-target')
        proposed = fixtures.values(rate='115.00', effective_from='2026-08-01',
                                    supersedes='TERM-A2', quote=body)
        with patch('recovery.term_extraction.extract_terms',
                   side_effect=lambda source, **_kwargs: fixtures.extraction(source, proposed)):
            proposal = lifecycle.generate_proposal(self.db, 'DEMO', source['doc_id'],
                                                    model='synthetic-local:1b')
        with connect(self.db) as conn:
            terms_before = [tuple(row) for row in conn.execute('SELECT * FROM reviewed_terms')]
        with self.assertRaises(ValueError):
            self.approve(proposal, 'substring-supersession')
        with connect(self.db) as conn:
            self.assertEqual([tuple(row) for row in conn.execute('SELECT * FROM reviewed_terms')],
                             terms_before)
            self.assertEqual(conn.execute('SELECT count(*) FROM term_review_events').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
