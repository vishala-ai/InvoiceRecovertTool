"""Migrated approvals cannot enter current calculation or review mutation paths."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recovery import lifecycle
from recovery.legacy_archive import import_archive
from recovery.storage import connect, load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def archive_payload():
    return {'schema_version': 1, 'synthetic': True, 'archive_id': 'LEGACY-TEST-BATCH',
        'source_system': 'Legacy AP (synthetic)',
        'provenance': 'Migrated from old system — locked (synthetic)',
        'suppliers': [{'tenant_id': 'DEMO', 'supplier_id': 'LEGACY-SUP-1', 'name': 'Old synthetic vendor'}],
        'contracts': [{'tenant_id': 'DEMO', 'supplier_id': 'LEGACY-SUP-1',
            'contract_id': 'LEGACY-CTR-1', 'title': 'Historical agreement', 'origin_year': 2024}],
        'cases': [{'tenant_id': 'DEMO', 'line_id': 'LEGACY-LINE-1',
            'supplier_id': 'LEGACY-SUP-1', 'contract_id': 'LEGACY-CTR-1',
            'invoice_id': 'LEGACY-INV-1', 'invoice_date': '2024-06-30', 'origin_year': 2024,
            'currency': 'USD', 'historical_approved_amount': '1250.50',
            'source_record_id': 'AP-2024-1', 'description': 'Migrated from old system — locked'}]}


class ArchiveLockGuardTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = Path(directory.name) / 'synthetic.sqlite3'
        self.checkpoints = Path(directory.name) / 'never-created.sqlite3'
        load_snapshot(self.db, ROOT / 'data')
        lifecycle.migrate(self.db, backup_dir=Path(directory.name) / 'backups')
        import_archive(self.db, archive_payload())

    def test_archive_rejects_reassessment_and_forged_evidence_request(self):
        with self.assertRaisesRegex(ValueError, 'MIGRATED_APPROVED_READ_ONLY'):
            lifecycle.reassess(self.db, 'DEMO', 'LEGACY-LINE-1', idempotency_key='archive-reassess')
        with connect(self.db) as conn:
            with self.assertRaisesRegex(ValueError, 'MIGRATED_APPROVED_READ_ONLY'):
                lifecycle.request_for_review(conn, {'tenant_id': 'DEMO', 'line_id': 'LEGACY-LINE-1'})
            self.assertEqual(conn.execute('SELECT count(*) FROM evidence_requests').fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM operation_keys WHERE operation_key='archive-reassess'").fetchone()[0], 0)

    def test_archived_start_and_retry_fail_before_optional_integrations(self):
        from recovery.workflow import run_persistent, retry_preparation
        with patch('recovery.ai.resolve_explanation_config',
                   side_effect=AssertionError('Locked history must never reach a provider')) as model:
            with self.assertRaisesRegex(ValueError, 'MIGRATED_APPROVED_READ_ONLY'):
                run_persistent(self.db, self.checkpoints, 'blocked', tenant='DEMO',
                    line_id='LEGACY-LINE-1', use_llm=True, provider='ollama', model='llama3.2:latest')
            # The lock is checked before importing or opening the optional saver.
            with self.assertRaisesRegex(ValueError, 'MIGRATED_APPROVED_READ_ONLY'):
                retry_preparation(self.db, self.checkpoints, 'blocked', tenant='DEMO', line_id='LEGACY-LINE-1')
            model.assert_not_called()
        self.assertFalse(self.checkpoints.exists())

    def test_direct_assessment_save_cannot_assign_current_state_to_locked_history(self):
        from recovery.engine import evaluate_line
        from recovery.lifecycle_storage import save_assessment
        with connect(self.db) as conn:
            forged = dict(evaluate_line(conn, 'DEMO', 'L001'), line_id='LEGACY-LINE-1')
            before = conn.execute('SELECT count(*) FROM assessment_revisions').fetchone()[0]
            with self.assertRaisesRegex(ValueError, 'MIGRATED_APPROVED_READ_ONLY'):
                save_assessment(conn, forged)
            self.assertEqual(conn.execute('SELECT count(*) FROM assessment_revisions').fetchone()[0], before)
