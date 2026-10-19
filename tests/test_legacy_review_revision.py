"""Legacy checkpoints must not bypass migrated assessment revision authority."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

from recovery.engine import evaluate_line
from recovery.lifecycle import migrate, reassess
from recovery.review import record_review
from recovery.storage import connect, load_snapshot

try:
    GRAPH_AVAILABLE = importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
except ModuleNotFoundError:
    GRAPH_AVAILABLE = False

DATA = Path(__file__).resolve().parents[1] / 'data'
REVIEW = {'decision': 'approve', 'reviewer': 'demo', 'note': 'Reviewed synthetic evidence'}


@unittest.skipUnless(GRAPH_AVAILABLE, 'Install graph extra for legacy checkpoint integration')
class LegacyReviewRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'synthetic.sqlite3'
        self.checkpoints = self.directory / 'checkpoints.sqlite3'
        load_snapshot(self.db, DATA)
        self.paused = self.flow(line_id='L004')
        self.assertEqual(self.paused['next'], ['approval'])
        self.assertNotIn('assessment_id', self.paused['state'])
        migrate(self.db, backup_dir=self.directory / 'backups')
        with connect(self.db) as conn:
            self.baseline = dict(conn.execute('''SELECT * FROM assessment_revisions
                WHERE tenant_id='DEMO' AND line_id='L004' ORDER BY sequence LIMIT 1''').fetchone())
            self.assertEqual(conn.execute('SELECT count(*) FROM operation_keys').fetchone()[0], 0)
        self.assertEqual(self.baseline['case_digest'], self.paused['state']['case']['case_digest'])

    def flow(self, **kwargs):
        from recovery.workflow import run_persistent
        return run_persistent(self.db, self.checkpoints, 'legacy-pending', **kwargs)

    def test_immediate_post_migration_resume_links_baseline_assessment(self):
        finished = self.flow(decision=REVIEW)
        self.assertFalse(finished['paused'])
        self.assertEqual(finished['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
        with connect(self.db) as conn:
            links = [dict(row) for row in conn.execute('SELECT * FROM case_review_assessments')]
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0]['assessment_id'], self.baseline['assessment_id'])
            self.assertEqual(links[0]['event_id'], 'legacy-pending:review')

    def test_same_input_new_revision_blocks_legacy_decision_without_writes(self):
        revised = reassess(self.db, 'DEMO', 'L004', idempotency_key='new-explicit-assessment')
        self.assertEqual(revised['case_digest'], self.baseline['case_digest'])
        self.assertNotEqual(revised['assessment_id'], self.baseline['assessment_id'])
        with connect(self.db) as conn:
            before = '\n'.join(conn.iterdump())
        with self.assertRaisesRegex(ValueError, 'STALE_CASE.*newer assessment revision'):
            self.flow(decision=REVIEW)
        with connect(self.db) as conn:
            self.assertEqual('\n'.join(conn.iterdump()), before)
            self.assertEqual(evaluate_line(conn, 'DEMO', 'L004')['case_digest'],
                             self.baseline['case_digest'])

    def test_committed_legacy_decision_replay_survives_new_revision(self):
        finished = self.flow(decision=REVIEW)
        saved = finished['state']['outcome']
        reassess(self.db, 'DEMO', 'L004', idempotency_key='after-committed-review')
        with connect(self.db) as conn:
            before = '\n'.join(conn.iterdump())
            replayed = record_review(conn, event_id=saved['event_id'],
                thread_id=saved['thread_id'], tenant='DEMO', line_id='L004',
                case_digest=saved['case_digest'], **REVIEW)
            self.assertEqual(replayed, saved)
            self.assertEqual('\n'.join(conn.iterdump()), before)
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT count(*) FROM case_review_assessments')
                             .fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
