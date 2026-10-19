"""Streamlit opt-in ranking with real local Qdrant and synthetic provider responses."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import test_ui as ui_helpers

from recovery.lifecycle import migrate
from recovery.semantic_retrieval import build_index, resolve_index_config
from recovery.storage import connect
from recovery.workflow import list_review_workflows


SEMANTIC_LABEL = 'Use local evidence search'
EMBEDDING_CONSENT = 'Allow local embedding calls for this review'
EMBEDDING_CONFIG = {'model': 'synthetic-embedding:latest', 'model_digest': 'b' * 64,
                    'base_url': 'http://127.0.0.1:11434'}
AVAILABLE = ui_helpers.AVAILABLE and importlib.util.find_spec('qdrant_client') is not None


def database_contents(path):
    if not path.exists():
        return None
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        return '\n'.join(connection.iterdump())
    finally:
        connection.close()


@unittest.skipUnless(AVAILABLE, 'Install ui, graph, ollama and vector extras for semantic UI checks')
class SemanticReviewUITests(unittest.TestCase):
    # Reuse the established isolated fixture/provider helpers, without inheriting
    # or rediscovering the existing test suite through a second TestCase class.
    start_patch = ui_helpers.ReviewWorkflowUITests.start_patch
    new_app = ui_helpers.ReviewWorkflowUITests.new_app
    provider_response = staticmethod(ui_helpers.ReviewWorkflowUITests.provider_response)
    widget = ui_helpers.ReviewWorkflowUITests.widget
    invoice_table = ui_helpers.ReviewWorkflowUITests.invoice_table
    visible_lines = ui_helpers.ReviewWorkflowUITests.visible_lines
    selected_line = ui_helpers.ReviewWorkflowUITests.selected_line
    open_line = ui_helpers.ReviewWorkflowUITests.open_line
    preparation_settings = ui_helpers.ReviewWorkflowUITests.preparation_settings
    ai_settings = ui_helpers.ReviewWorkflowUITests.ai_settings
    provider_consent = ui_helpers.ReviewWorkflowUITests.provider_consent
    decision_buttons = ui_helpers.ReviewWorkflowUITests.decision_buttons
    events = ui_helpers.ReviewWorkflowUITests.events
    assert_clean = ui_helpers.ReviewWorkflowUITests.assert_clean

    def setUp(self):
        ui_helpers.ReviewWorkflowUITests.setUp(self)
        self.index = Path(self.temp.name).resolve() / 'semantic-index'
        self.start_patch(patch.dict('os.environ', {'RECOVERY_RETRIEVAL_INDEX': str(self.index)}))
        self.embedding_requests = []
        self.embed = self.start_patch(patch('recovery.semantic_retrieval.embed_texts',
                                           side_effect=self.fake_embeddings))
        self.embedding_config = self.start_patch(patch(
            'recovery.semantic_retrieval.resolve_embedding_config', return_value=EMBEDDING_CONFIG))

    def fake_embeddings(self, texts, config):
        self.assertEqual(config, EMBEDDING_CONFIG)
        if texts:
            self.embedding_requests.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    def build(self, *, migrated=False):
        if migrated:
            migrate(self.db, backup_dir=Path(self.temp.name) / 'backups')
        with connect(self.db) as conn:
            result = build_index(conn, 'DEMO', self.index, model=EMBEDDING_CONFIG['model'])
        self.embedding_requests.clear()
        self.embed.reset_mock()
        return result

    def start_semantic_review(self):
        self.app.run()
        self.open_line('L001')
        self.ai_settings()
        self.widget('checkbox', SEMANTIC_LABEL).check().run()
        self.widget('checkbox', EMBEDDING_CONSENT).check().run()
        self.assertEqual(self.embedding_requests, [])
        self.assertEqual(self.provider_consent().label,
            'Allow sending this synthetic case and its complete scoped evidence to local Ollama')
        self.provider_consent().check().run()
        self.widget('button', 'Prepare review workflow').click().run()
        self.assert_clean()
        self.assertFalse(self.app.error, [item.value for item in self.app.error])

    def current_workflow(self):
        workflows = list_review_workflows(self.db, self.checkpoints, tenant='DEMO', line_id='L001')
        return max(workflows, key=lambda workflow: workflow['thread_id'])

    def assert_stale_controls(self):
        self.assert_clean()
        self.assertTrue(any('evidence search' in item.value.lower()
                            for kind in ('info', 'warning') for item in getattr(self.app, kind)))
        self.assertEqual(len(self.decision_buttons()), 3)
        self.assertTrue(all(button.disabled for button in self.decision_buttons()))
        self.assertEqual(self.events(), [])

    def test_default_render_and_start_do_not_read_index_or_embed(self):
        from recovery.semantic_retrieval import validate_index_config

        with patch('recovery.semantic_retrieval.resolve_index_config',
                   side_effect=AssertionError('Default review must not read a semantic index')) as resolve, \
             patch('recovery.semantic_retrieval.validate_index_config',
                   wraps=validate_index_config) as validate:
            self.app.run()
            self.open_line('L001')
            self.ai_settings()
            self.assertFalse(self.widget('checkbox', SEMANTIC_LABEL).value)
            self.assertFalse(any(item.label == EMBEDDING_CONSENT for item in self.app.checkbox))
            self.provider_consent().check().run()
            self.widget('button', 'Prepare review workflow').click().run()
            self.assert_clean()
            self.assertEqual(len(self.decision_buttons()), 3)
            self.app.run()
            resolve.assert_not_called()
            validate.assert_not_called()
            self.embed.assert_not_called()
        self.assertNotIn('retrieval_config', self.current_workflow()['state'])
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.events(), [])

    def test_selected_ranking_pins_index_and_retains_full_contract_and_amendment(self):
        self.build()
        expected_config = resolve_index_config(self.index)
        self.start_semantic_review()
        current = self.current_workflow()
        self.assertEqual(current['next'], ['approval'])
        self.assertEqual(current['state']['retrieval_config'], expected_config)
        self.assertEqual({document['doc_id'] for document in current['state']['documents']},
                         {'DOC-A1', 'DOC-A2'})
        self.assertEqual(len(self.embedding_requests), 1)
        self.assertEqual(self.generate.call_count, 1)
        # Prepared settings are restored from the checkpoint, not exposed as
        # mutable technical controls in the accountant's saved review.
        self.assertFalse(any(item.label == SEMANTIC_LABEL for item in self.app.checkbox))
        self.assertTrue(all(not button.disabled for button in self.decision_buttons()))
        self.assertEqual(self.events(), [])

    def test_selected_search_requires_separate_embedding_consent(self):
        self.build()
        self.app.run()
        self.open_line('L001')
        self.ai_settings()
        self.widget('checkbox', SEMANTIC_LABEL).check().run()
        self.provider_consent().check().run()
        self.assertFalse(self.widget('checkbox', EMBEDDING_CONSENT).value)
        self.assertTrue(self.widget('button', 'Prepare review workflow').disabled)
        self.assertEqual(self.embedding_requests, [])
        self.embed.assert_not_called()
        self.generate.assert_not_called()
        self.assertFalse(self.checkpoints.exists())
        self.assertEqual(self.events(), [])

    def test_migrated_review_uses_existing_linked_workflow_with_pinned_index(self):
        self.build(migrated=True)
        self.start_semantic_review()
        current = self.current_workflow()
        self.assertEqual(current['next'], ['approval'])
        self.assertEqual(current['state']['retrieval_config'], resolve_index_config(self.index))
        self.assertIn('assessment_id', current['state'])
        self.assertEqual({document['doc_id'] for document in current['state']['documents']},
                         {'DOC-A1', 'DOC-A2'})
        self.assertEqual(len(self.embedding_requests), 1)
        self.assertEqual(self.events(), [])

    def test_restoring_selected_review_is_read_only_without_embedding_or_explanation_requests(self):
        self.build()
        self.start_semantic_review()
        before = (database_contents(self.db), database_contents(self.checkpoints))
        pinned = self.current_workflow()['state']['retrieval_config']
        self.embedding_requests.clear()
        self.embed.reset_mock()
        self.generate.reset_mock()
        self.app.run()
        self.app = self.new_app().run()
        self.open_line('L001')
        self.assert_clean()
        self.assertEqual(self.current_workflow()['state']['retrieval_config'], pinned)
        self.assertFalse(any(item.label == SEMANTIC_LABEL for item in self.app.checkbox))
        self.assertTrue(all(not button.disabled for button in self.decision_buttons()))
        self.assertEqual(self.embedding_requests, [])
        self.generate.assert_not_called()
        self.assertEqual(before, (database_contents(self.db), database_contents(self.checkpoints)))

    def test_new_published_generation_disables_old_review_without_new_journal_events(self):
        self.build(migrated=True)
        self.start_semantic_review()
        old_config = self.current_workflow()['state']['retrieval_config']
        before = (database_contents(self.db), database_contents(self.checkpoints))
        replacement = self.build()
        self.assertNotEqual(old_config['generation'], replacement['generation'])
        self.generate.reset_mock()
        self.app.run()
        self.assert_stale_controls()
        self.assertEqual(self.embedding_requests, [])
        self.generate.assert_not_called()
        self.assertEqual(self.current_workflow()['state']['retrieval_config'], old_config)
        self.assertEqual(before, (database_contents(self.db), database_contents(self.checkpoints)))

    def test_missing_saved_index_disables_old_review_without_new_provider_calls(self):
        self.build()
        self.start_semantic_review()
        before = (database_contents(self.db), database_contents(self.checkpoints))
        (self.index / 'evidence-index.json').unlink()
        self.embedding_requests.clear()
        self.generate.reset_mock()
        self.app.run()
        self.assert_stale_controls()
        self.assertEqual(self.embedding_requests, [])
        self.generate.assert_not_called()
        self.assertEqual(before, (database_contents(self.db), database_contents(self.checkpoints)))

    def test_missing_selected_index_fails_before_starting_a_review(self):
        self.app.run()
        self.open_line('L001')
        self.ai_settings()
        self.widget('checkbox', SEMANTIC_LABEL).check().run()
        self.widget('checkbox', EMBEDDING_CONSENT).check().run()
        self.provider_consent().check().run()
        self.widget('button', 'Prepare review workflow').click().run()
        self.assert_clean()
        self.assertTrue(any('Review not prepared' in item.value for item in self.app.error))
        self.assertEqual(self.embedding_requests, [])
        self.generate.assert_not_called()
        self.assertFalse(self.checkpoints.exists())
        self.assertEqual(self.events(), [])

    def test_demo_profile_suggests_index_without_inference_or_automatic_consent(self):
        with patch.dict('os.environ', {'RECOVERY_RETRIEVAL_INDEX': str(self.index)}):
            self.app.run()
            self.open_line('L001')
            self.ai_settings()
            self.assertFalse(self.widget('checkbox', SEMANTIC_LABEL).value)
            self.assertFalse(self.provider_consent().value)
            self.widget('checkbox', SEMANTIC_LABEL).check().run()
            self.assertFalse(self.widget('checkbox', EMBEDDING_CONSENT).value)
            self.assertFalse(self.provider_consent().value)
        self.assert_clean()
        self.embedding_config.assert_not_called()
        self.embed.assert_not_called()
        self.generate.assert_not_called()
        self.assertFalse(self.index.exists())
        self.assertFalse(self.checkpoints.exists())
        self.assertEqual(self.events(), [])


if __name__ == '__main__':
    unittest.main()
