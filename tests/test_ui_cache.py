"""Display caching observes SQLite changes but never authorizes a decision."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from recovery import ui_cache

ROOT = Path(__file__).resolve().parents[1]
try:
    UI_AVAILABLE = importlib.util.find_spec('streamlit') is not None
    GRAPH_AVAILABLE = importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
except ModuleNotFoundError:
    UI_AVAILABLE = GRAPH_AVAILABLE = False


class PortfolioStampTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.db, self.cp = self.root / 'portfolio.sqlite3', self.root / 'checkpoints.sqlite3'

    def test_missing_file_stamp_is_read_only_and_detects_later_checkpoint_creation(self):
        missing = ui_cache.portfolio_stamp(self.db, self.cp)
        self.assertFalse(self.db.exists())
        self.assertFalse(self.cp.exists())
        self.assertEqual(len(missing), 4)
        self.cp.write_bytes(b'Synthetic checkpoint placeholder')
        self.assertNotEqual(ui_cache.portfolio_stamp(self.db, self.cp), missing)

    def test_stamp_detects_same_size_changes_and_file_replacement(self):
        self.db.write_bytes(b'Synthetic original')
        first = ui_cache.portfolio_stamp(self.db, self.cp)
        stat = self.db.stat()
        self.db.write_bytes(b'Synthetic modified')
        os.utime(self.db, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        second = ui_cache.portfolio_stamp(self.db, self.cp)
        self.assertNotEqual(first, second)
        replacement = self.root / 'replacement.sqlite3'
        replacement.write_bytes(self.db.read_bytes())
        stat = self.db.stat()
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        os.replace(replacement, self.db)
        self.assertNotEqual(second, ui_cache.portfolio_stamp(self.db, self.cp))

    def test_empty_and_missing_wals_are_equivalent_but_nonempty_wals_are_not(self):
        original = ui_cache.portfolio_stamp(self.db, self.cp)
        for database in (self.db, self.cp):
            wal = database.with_name(database.name + '-wal')
            wal.touch()
            self.assertEqual(ui_cache.portfolio_stamp(self.db, self.cp), original)
            wal.write_bytes(b'Synthetic committed WAL frame')
            self.assertNotEqual(ui_cache.portfolio_stamp(self.db, self.cp), original)
            wal.write_bytes(b'')
            self.assertEqual(ui_cache.portfolio_stamp(self.db, self.cp), original)


@unittest.skipUnless(UI_AVAILABLE, 'Install UI extra to execute the real Streamlit display cache')
class DisplayWorklistCacheTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st

        self.st = st
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.db, self.cp = self.root / 'portfolio.sqlite3', self.root / 'checkpoints.sqlite3'
        with sqlite3.connect(self.db) as conn:
            conn.execute('CREATE TABLE cache_marker(value TEXT)')
            conn.execute("INSERT INTO cache_marker VALUES ('synthetic')")
        self.enterContext(patch.object(ui_cache, '_cached_worklist', None))
        self.addCleanup(self.clear_display_cache)

    @staticmethod
    def clear_display_cache():
        if ui_cache._cached_worklist is not None:
            ui_cache._cached_worklist.clear()

    def display(self, tenant='DEMO'):
        return ui_cache.display_worklist(self.st, self.db, self.cp, tenant=tenant)

    def test_unchanged_reads_use_real_cache_without_creating_a_checkpoint(self):
        value = {'cases': [], 'report': {'candidate_total': '80707.50', 'realized_recovery': '0.00'}}
        stamp = ui_cache.portfolio_stamp(self.db, self.cp)
        with patch('recovery.invoice_review.load_worklist', return_value=value) as load:
            first, second = self.display(), self.display()
            self.assertEqual(first, value)
            self.assertEqual(second, value)
            load.assert_called_once_with(str(self.db), str(self.cp), tenant='DEMO')
            # Streamlit returns copies; editing a display cannot poison its cache.
            first['report']['candidate_total'] = '0.00'
            self.assertEqual(self.display()['report']['candidate_total'], '80707.50')
        self.assertEqual(ui_cache.portfolio_stamp(self.db, self.cp), stamp)
        self.assertFalse(self.cp.exists())

    def test_database_and_checkpoint_file_changes_each_invalidate_the_display(self):
        with patch('recovery.invoice_review.load_worklist', return_value={'cases': []}) as load:
            self.display()
            self.display()
            self.assertEqual(load.call_count, 1)
            with sqlite3.connect(self.db) as conn:
                conn.execute("INSERT INTO cache_marker VALUES ('changed synthetic portfolio')")
            self.display()
            self.assertEqual(load.call_count, 2)
            with sqlite3.connect(self.cp) as conn:
                conn.execute('CREATE TABLE synthetic_checkpoint(value TEXT)')
            self.display()
            self.assertEqual(load.call_count, 3)
            with sqlite3.connect(self.cp) as conn:
                conn.execute("INSERT INTO synthetic_checkpoint VALUES ('changed review attempt')")
            self.display()
            self.assertEqual(load.call_count, 4)

    def test_database_and_checkpoint_wal_commits_invalidate_even_when_main_file_is_unchanged(self):
        for database in (self.db, self.cp):
            with self.subTest(database=database.name):
                self.clear_display_cache()
                with sqlite3.connect(database) as writer:
                    writer.execute('PRAGMA journal_mode=WAL')
                    writer.execute('PRAGMA wal_autocheckpoint=0')
                    writer.execute('CREATE TABLE IF NOT EXISTS wal_marker(value TEXT)')
                    writer.commit()
                    with patch('recovery.invoice_review.load_worklist', return_value={'cases': []}) as load:
                        self.display()
                        self.display()
                        self.assertEqual(load.call_count, 1)
                        main_stamp = database.stat().st_mtime_ns, database.stat().st_size
                        writer.execute("INSERT INTO wal_marker VALUES ('synthetic WAL commit')")
                        writer.commit()
                        self.assertTrue(database.with_name(database.name + '-wal').exists())
                        self.assertEqual((database.stat().st_mtime_ns, database.stat().st_size), main_stamp)
                        self.display()
                        self.assertEqual(load.call_count, 2)

    def test_tenant_scope_is_part_of_the_cache_key(self):
        with patch('recovery.invoice_review.load_worklist', return_value={'cases': []}) as load:
            self.display('DEMO')
            self.display('OTHER')
            self.display('DEMO')
            self.assertEqual(load.call_count, 2)
            self.assertEqual([item.kwargs['tenant'] for item in load.call_args_list], ['DEMO', 'OTHER'])

    def test_write_during_load_withholds_snapshot_and_does_not_cache_a_partial_result(self):
        def changing_load(*args, **kwargs):
            with sqlite3.connect(self.db) as conn:
                conn.execute("INSERT INTO cache_marker VALUES ('concurrent synthetic write')")
            return {'cases': []}

        with patch('recovery.invoice_review.load_worklist', side_effect=changing_load) as load:
            with self.assertRaisesRegex(RuntimeError, 'changed while loading'):
                self.display()
            self.assertEqual(load.call_count, 1)
        with patch('recovery.invoice_review.load_worklist', return_value={'cases': []}) as load:
            self.assertEqual(self.display(), {'cases': []})
            load.assert_called_once()

    def test_first_read_may_create_an_empty_checkpoint_wal_without_false_staleness(self):
        def reader_creates_empty_wal(*args, **kwargs):
            self.cp.with_name(self.cp.name + '-wal').touch()
            return {'cases': []}

        with patch('recovery.invoice_review.load_worklist', side_effect=reader_creates_empty_wal) as load:
            self.assertEqual(self.display(), {'cases': []})
            self.assertEqual(self.display(), {'cases': []})
            load.assert_called_once()

    def test_semantic_index_is_revalidated_on_every_hit_without_database_writes_or_embeddings(self):
        case = {'line_id': 'L001', 'status': 'CANDIDATE', 'net_candidate': '100.00',
            'case_digest': 'synthetic-digest', 'bucket': 'approval', 'blocking_request_count': 0,
            'current_decision': None, 'committed': None, 'ready': True, 'decision_error': None,
            'workflow': {'next': ['approval'], 'state': {'tenant': 'DEMO', 'line_id': 'L001',
                'case': {'case_digest': 'synthetic-digest'}, 'use_llm': False,
                'brief': {'mode': 'offline_template'}, 'retrieval_config': {'generation': 'synthetic-old'}}}}
        before = ui_cache.portfolio_stamp(self.db, self.cp)
        with patch('recovery.invoice_review.load_worklist', return_value={'cases': [case]}) as load, \
                patch('recovery.semantic_retrieval.validate_index_config',
                    side_effect=[None, ValueError('synthetic index rebuilt/deleted'), None]) as validate, \
                patch('recovery.embeddings.embed_texts', side_effect=AssertionError('Display must not embed')):
            self.assertTrue(self.display()['cases'][0]['ready'])
            changed = self.display()['cases'][0]
            self.assertFalse(changed['ready'])
            self.assertIn('changed or is unavailable', changed['decision_error'])
            self.assertTrue(self.display()['cases'][0]['ready'])
            self.assertEqual(load.call_count, 1)
            self.assertEqual(validate.call_count, 3)
        self.assertEqual(ui_cache.portfolio_stamp(self.db, self.cp), before)

    @unittest.skipUnless(GRAPH_AVAILABLE, 'Install graph extra for uncached decision integration')
    def test_actual_decision_uses_fresh_line_read_and_rejects_an_old_cached_attempt(self):
        from recovery import lifecycle
        from recovery.invoice_review import decide_selected, prepare_review, selection_token, load_worklist
        from recovery.storage import connect, load_snapshot

        # Replace only this test's disposable marker DB with the supplied fixture.
        self.db.unlink()
        load_snapshot(self.db, ROOT / 'data')
        lifecycle.migrate(self.db, backup_dir=self.root / 'backups')
        original = next(row for row in load_worklist(self.db, self.cp)['cases'] if row['line_id'] == 'L004')
        prepare_review(self.db, self.cp, original, action_id='cache-original-review')
        cached = next(row for row in self.display()['cases'] if row['line_id'] == 'L004')
        self.assertTrue(cached['ready'])
        lifecycle.start_linked_review(self.db, self.cp, 'DEMO', cached['assessment_id'],
            idempotency_key='cache-newer-same-digest-attempt', use_llm=False)
        with patch('recovery.ui_cache.display_worklist',
                   side_effect=AssertionError('A decision must not read display cache')), \
                patch('recovery.invoice_review.audit',
                      side_effect=AssertionError('Single decision must not audit the entire portfolio')):
            results = decide_selected(self.db, self.cp, [selection_token(cached)],
                supplier_id='SUP-A', contract_id='CTR-A', reviewer='Synthetic reviewer',
                note='This displayed attempt has been superseded.')
        self.assertEqual(results[0]['outcome'], 'Not saved')
        self.assertIn('changed after selection', results[0]['reason'])
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
