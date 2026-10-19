"""Commit-before-rerun behavior using real SQLite and the explicit PG double."""
from contextlib import closing
import importlib.util
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from recovery import cloud_runtime as runtime, local_auth
from recovery.cloud_store import CloudStore
from recovery.engine import audit
from recovery.storage import connect, load_snapshot
from test_cloud_store import CERTIFICATE, DSN, _PostgresProtocol

ROOT = Path(__file__).resolve().parents[1]
KEY = 'gsk_isolated_cloud_runtime_test_only'
PASSWORD = 'Isolated cloud owner test passphrase 2026'
try:
    GRAPH = importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
except ModuleNotFoundError:
    GRAPH = False
UI = importlib.util.find_spec('streamlit') is not None


class Rerun(BaseException):
    pass


class Stop(BaseException):
    pass


class CloudSettingsTests(unittest.TestCase):
    def settings(self):
        return {'cloud': {'database_url': DSN, 'root_cert_pem': CERTIFICATE},
                'groq': {'api_key': KEY}}

    def test_secrets_have_no_unsafe_repr_and_no_dotenv_or_network_is_read(self):
        settings = runtime.load_settings(self.settings())
        self.assertEqual(settings.model, 'llama-3.3-70b-versatile')
        self.assertNotIn(DSN, repr(settings))
        self.assertNotIn(KEY, repr(settings))
        self.assertNotIn(CERTIFICATE, repr(settings))

    def test_missing_and_placeholder_configuration_reports_no_values(self):
        for values in ({}, {'cloud': {}}, self.settings() | {'groq': {'api_key': 'SET_PRIVATELY_IN_STREAMLIT_SECRETS'}}):
            with self.subTest(values=bool(values)):
                with self.assertRaises(runtime.CloudRuntimeError) as caught:
                    runtime.load_settings(values)
                self.assertNotIn(DSN, str(caught.exception))
                self.assertNotIn(KEY, str(caught.exception))

    def test_environment_is_scoped_and_removes_local_embedding_configuration(self):
        settings = runtime.load_settings(self.settings())
        files = {'portfolio': Path('/synthetic/portfolio.sqlite3'), 'accounts': Path('/synthetic/accounts.sqlite3')}
        with patch.dict(os.environ, {'RECOVERY_CLOUD': '0', 'RECOVERY_RETRIEVAL_INDEX': '/local/index'}):
            previous = dict(os.environ)
            with runtime.cloud_environment(settings, files):
                self.assertEqual(os.environ['RECOVERY_CLOUD'], '1')
                self.assertEqual(os.environ['GROQ_API_KEY'], KEY)
                self.assertEqual(os.environ['PYTHON_DOTENV_DISABLED'], '1')
                self.assertNotIn('RECOVERY_RETRIEVAL_INDEX', os.environ)
            self.assertEqual(dict(os.environ), previous)


@unittest.skipUnless(hasattr(hashlib, 'scrypt'), 'Use uv Python with scrypt for Cloud account checks')
class CloudRuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.server = _PostgresProtocol()
        self.settings = runtime.CloudSettings(DSN, CERTIFICATE, KEY,
            owner={'username': 'cloud-owner', 'display_name': 'Cloud owner', 'password': PASSWORD})
        self.factory = lambda dsn, root_cert_pem: CloudStore(dsn, root_cert_pem=root_cert_pem,
                                                          connector=self.server.connect)
        self.directory = self.root / 'worker'
        self.state = {}

    def seed(self, *, accounts=False):
        with self.factory(DSN, root_cert_pem=CERTIFICATE).workspace(self.settings.workspace_id,
                                                                 self.root / 'seed') as files:
            load_snapshot(files['portfolio'], ROOT / 'data')
            with closing(sqlite3.connect(files['checkpoints'])) as conn:
                conn.execute('VACUUM')
            local_auth.initialize(files['accounts'])
            if accounts:
                local_auth.bootstrap_owner(files['accounts'], 'cloud-owner', 'Cloud owner', PASSWORD)

    def execute(self, runner, **kwargs):
        return runtime.execute_workspace(self.settings, self.state, runner, store_factory=self.factory,
            directory=self.directory, control_exceptions=(Rerun, Stop), **kwargs)

    def test_empty_workspace_fails_closed_without_creating_financial_or_public_owner_data(self):
        self.state.update({'auth:token': 'discard', 'consent:old': True})
        with self.assertRaisesRegex(runtime.CloudRuntimeError, 'initial synthetic seed'):
            self.execute(lambda: self.fail('Uninitialized app executed'))
        self.assertEqual(self.state, {})
        self.assertEqual(self.server.data, {})

    def test_first_owner_uses_private_configuration_and_is_committed_before_stop(self):
        self.seed()
        def render():
            self.assertTrue(local_auth.has_accounts(os.environ['RECOVERY_AUTH_DB']))
            raise Stop()
        with self.assertRaises(Stop):
            self.execute(render)
        with self.factory(DSN, root_cert_pem=CERTIFICATE).workspace(self.settings.workspace_id,
                                                                 self.root / 'after-restart') as files:
            token = local_auth.authenticate(files['accounts'], 'cloud-owner', PASSWORD)
            self.assertEqual(local_auth.current_user(files['accounts'], token)['role'], 'owner')

    def test_missing_private_owner_setup_never_runs_public_registration(self):
        self.seed()
        self.settings = runtime.CloudSettings(DSN, CERTIFICATE, KEY)
        with self.assertRaisesRegex(runtime.CloudRuntimeError, 'private cloud.owner'):
            self.execute(lambda: self.fail('Public account setup executed'))

    def test_existing_accounts_do_not_require_or_reuse_bootstrap_password(self):
        self.seed(accounts=True)
        self.settings = runtime.CloudSettings(DSN, CERTIFICATE, KEY)
        calls = []
        self.execute(lambda: calls.append('rendered'))
        self.assertEqual(calls, ['rendered'])

    def test_commit_failure_discards_pending_success_auth_and_chat_state(self):
        self.seed(accounts=True)
        def render():
            self.state.update({'auth:token': 'new-session', 'auth:account-notice': 'success',
                               'invoice_bulk_result': 'success', 'audit_workspace_chat': 'old answer'})
            raise Rerun()
        self.server.fail_commit = 'before'
        with self.assertRaisesRegex(runtime.CloudRuntimeError, 'confirmation is unavailable'):
            self.execute(render)
        self.assertEqual(self.state, {})

    def test_unhandled_application_error_is_sanitized_and_does_not_publish_partial_state(self):
        self.seed(accounts=True)
        prior = self.server.data[self.settings.workspace_id]['revision']
        def render():
            db = os.environ['RECOVERY_DB']
            with connect(db) as conn, conn:
                conn.execute('CREATE TABLE uncommitted_cloud_change (value TEXT)')
            raise RuntimeError(DSN + KEY)
        with self.assertRaises(runtime.CloudRuntimeError) as caught:
            self.execute(render)
        self.assertNotIn(DSN, str(caught.exception))
        self.assertNotIn(KEY, str(caught.exception))
        self.assertEqual(self.server.data[self.settings.workspace_id]['revision'], prior)

    @unittest.skipUnless(GRAPH, 'Install graph extra for durable review/checkpoint integration')
    def test_pause_decision_and_ambiguous_commit_restore_together_without_new_recovery(self):
        from recovery.workflow import get_review_workflow, run_persistent

        self.seed(accounts=True)
        def start():
            db = Path(os.environ['RECOVERY_DB'])
            cp = db.with_name(db.stem + '-ui-checkpoints.sqlite3')
            result = run_persistent(db, cp, 'cloud-review-L001', line_id='L001', use_llm=False)
            self.assertTrue(result['paused'])
            raise Rerun()
        with self.assertRaises(Rerun):
            self.execute(start)
        def approve():
            db = Path(os.environ['RECOVERY_DB'])
            cp = db.with_name(db.stem + '-ui-checkpoints.sqlite3')
            self.assertTrue(get_review_workflow(db, cp, 'cloud-review-L001')['paused'])
            run_persistent(db, cp, 'cloud-review-L001', decision={
                'decision': 'approve', 'reviewer': 'Cloud synthetic accountant',
                'note': 'Reviewed the complete synthetic signed evidence.'})
            self.state['saved_notice'] = 'approved'
            raise Rerun()
        self.server.fail_commit = 'after'
        with self.assertRaisesRegex(runtime.CloudRuntimeError, 'confirmation is unavailable'):
            self.execute(approve)
        self.assertEqual(self.state, {})
        self.directory = self.root / 'new-worker-after-restart'
        def inspect():
            db = Path(os.environ['RECOVERY_DB'])
            cp = db.with_name(db.stem + '-ui-checkpoints.sqlite3')
            self.assertFalse(get_review_workflow(db, cp, 'cloud-review-L001')['paused'])
            with connect(db) as conn:
                self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)
                report = audit(conn)
                self.assertEqual((report['case_count'], report['candidate_total'], report['realized_recovery']),
                                 (20, '80707.50', '0.00'))
        self.execute(inspect)


@unittest.skipUnless(UI, 'Install ui extra for Cloud registration gate')
class CloudRegistrationUITests(unittest.TestCase):
    def test_cloud_empty_account_store_has_no_public_owner_form(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'RECOVERY_CLOUD': '1', 'CLOUD_TEST_AUTH': str(Path(directory) / 'auth.sqlite3')}):
            app = AppTest.from_string('''
import os
from pathlib import Path
import streamlit as st
from recovery.auth_ui import require_login
require_login(st, Path(os.environ['CLOUD_TEST_AUTH']))
''').run()
            self.assertFalse(app.exception)
            self.assertFalse(app.text_input)
            self.assertFalse(any(item.label == 'Create owner account' for item in app.button))
            self.assertTrue(any('Public account registration is unavailable' in item.value for item in app.error))

    def test_failed_cloud_commit_removes_provisional_ui_and_does_not_replay_action(self):
        from streamlit.testing.v1 import AppTest
        import streamlit as st

        def fail_commit(*args, **kwargs):
            st.sidebar.success('Provisional account creation')
            st.success('Provisional invoice approval')
            st.dataframe([{'Case': 'L001', 'Result': 'Provisional'}])
            st.session_state['auth:token'] = 'unconfirmed-token'
            st.session_state['invoice_bulk_result'] = 'unconfirmed'
            raise runtime.CloudRuntimeError('Save confirmation is unavailable. Refresh to check the saved state.')

        with patch('recovery.cloud_runtime.load_settings', return_value=runtime.CloudSettings(DSN, CERTIFICATE, KEY)), \
             patch('recovery.cloud_runtime.execute_workspace', side_effect=fail_commit) as execute:
            app = AppTest.from_file(str(ROOT / 'cloud/streamlit_app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.success)
        self.assertFalse(app.dataframe)
        self.assertFalse(app.sidebar.success)
        self.assertNotIn('auth:token', app.session_state)
        self.assertNotIn('invoice_bulk_result', app.session_state)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual([button.label for button in app.button], ['Refresh current saved state'])
        self.assertTrue(any('Save confirmation is unavailable' in item.value for item in app.error))


if __name__ == '__main__':
    unittest.main()
