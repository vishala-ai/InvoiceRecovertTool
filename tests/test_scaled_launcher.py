"""The fixed local launcher never substitutes a different portfolio or index."""
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from recovery.demo_portfolio import AS_OF, PROFILE_ID


spec = importlib.util.spec_from_file_location('scaled_demo_launcher',
    Path(__file__).resolve().parents[1] / 'scripts/run_demo.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ScaledLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(launcher, 'SCALED_DIRECTORY', self.directory).start()
        self.marker = self.directory / 'portfolio-profile.json'
        self.profile = {'synthetic': True, 'profile_id': PROFILE_ID, 'as_of': AS_OF,
            'current_cohort': {'cases': 1000, 'vendors': 35, 'approved': 500,
                               'awaiting': 250, 'evidence': 250},
            'reconciliation': {'matched': True}}

    def seed(self):
        self.marker.write_text(json.dumps(self.profile))
        database = self.directory / 'portfolio.sqlite3'
        with sqlite3.connect(database) as conn:
            conn.executescript('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);'
                               'CREATE TABLE invoice_lines (tenant_id TEXT, line_id TEXT);'
                               'CREATE TABLE suppliers (tenant_id TEXT, supplier_id TEXT);')
            conn.executemany('INSERT INTO metadata VALUES (?,?)', [
                ('manifest', json.dumps({'synthetic': True, 'as_of': AS_OF})),
                ('demo_portfolio', json.dumps(self.profile))])
            conn.executemany('INSERT INTO invoice_lines VALUES (?,?)',
                             [('DEMO', f'SCALE-2026-{i:04d}') for i in range(1000)])
            conn.executemany('INSERT INTO suppliers VALUES (?,?)',
                             [('DEMO', f'DEMO-SCALE-V{i:03d}') for i in range(35)])
        with sqlite3.connect(self.directory / 'portfolio-ui-checkpoints.sqlite3'):
            pass
        patch('recovery.legacy_archive.archive_summary', return_value={'case_count': 2000}).start()
        return database

    def test_missing_profile_fails_without_creating_or_falling_back(self):
        with self.assertRaisesRegex(ValueError, 'never creates'):
            launcher.verify_profile()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_matching_scaled_profile_is_read_only_and_has_no_inherited_semantic_index(self):
        database = self.seed()
        before = database.read_bytes()
        result = launcher.verify_profile()
        self.assertIsNone(result['index'])
        self.assertEqual(result['locked_historical_cases'], 2000)
        self.assertEqual(database.read_bytes(), before)
        with (patch.dict('os.environ', {'RECOVERY_RETRIEVAL_INDEX': '/old/million/index'}),
              patch('os.chdir'), patch('os.execvpe') as execute, redirect_stdout(io.StringIO())):
            self.assertEqual(launcher.main([]), 0)
        environment = execute.call_args.args[2]
        self.assertNotIn('RECOVERY_RETRIEVAL_INDEX', environment)
        self.assertEqual(environment['RECOVERY_DB'], str(database))
        self.assertIn('8515', execute.call_args.args[1])

    def test_marker_database_disagreement_rejects_profile(self):
        self.seed()
        self.profile['description'] = 'different published generation'
        self.marker.write_text(json.dumps(self.profile))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            launcher.verify_profile()

    def test_incomplete_current_cohort_rejects_profile(self):
        database = self.seed()
        with sqlite3.connect(database) as conn:
            conn.execute("DELETE FROM invoice_lines WHERE line_id='SCALE-2026-0000'")
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            launcher.verify_profile()

    def test_explicit_million_profile_retains_its_original_index_validation(self):
        with (patch.object(launcher, 'DATABASE', self.directory / 'missing-million.sqlite3'),
              redirect_stderr(io.StringIO()) as errors):
            self.assertEqual(launcher.main(['--profile', 'million', '--check']), 2)
        self.assertIn('million-dollar', errors.getvalue())
        self.assertEqual(list(self.directory.iterdir()), [])
