"""Real password gate, account controls and isolation before portfolio reads."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from recovery import local_auth
from recovery.storage import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec('streamlit') is not None and hasattr(hashlib, 'scrypt')
PASSWORD = 'Only for isolated auth UI tests 42!'
NEW_PASSWORD = 'A replacement isolated passphrase 89!'


@unittest.skipUnless(AVAILABLE, 'Install ui extra and use uv Python with scrypt for sign-in UI checks')
class AuthenticationUITests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest

        # Load component dependencies before patching the unauthenticated read
        # boundary; imports must not retain a test's temporary mocked alias.
        for module in ('recovery.invoice_ui', 'recovery.lifecycle_ui'):
            importlib.import_module(module)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.auth = self.directory / 'auth.sqlite3'
        self.db = self.directory / 'synthetic.sqlite3'
        load_snapshot(self.db, ROOT / 'data')
        environment = patch.dict(os.environ, {'RECOVERY_DB': str(self.db),
                                               'RECOVERY_AUTH_DB': str(self.auth)})
        environment.start()
        self.addCleanup(environment.stop)
        self.new_app = lambda: AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30)
        self.app = self.new_app()

    def provision(self):
        local_auth.initialize(self.auth)
        local_auth.bootstrap_owner(self.auth, 'local-owner', 'Local owner', PASSWORD)

    def widget(self, kind, label, app=None):
        return next(item for item in getattr(self.app if app is None else app, kind)
                    if item.label == label)

    def login(self, username='local-owner', password=PASSWORD, app=None):
        app = self.app if app is None else app
        app.run()
        self.widget('text_input', 'Username', app).set_value(username)
        self.widget('text_input', 'Password', app).set_value(password)
        self.widget('button', 'Sign in', app).click().run()
        self.assertFalse(app.exception)

    def no_portfolio(self, app=None):
        app = self.app if app is None else app
        self.assertFalse(app.exception)
        self.assertFalse(app.dataframe)
        self.assertFalse(app.metric)
        self.assertFalse(any(button.key and button.key.startswith('navigation:')
                             for button in app.button))

    def contents(self):
        with sqlite3.connect(self.db) as conn:
            return '\n'.join(conn.iterdump())

    def test_first_visit_stops_before_any_portfolio_reader_and_has_no_default_password(self):
        with patch('recovery.storage.connect', side_effect=AssertionError('Unauthenticated read')), \
             patch('recovery.invoice_ui.render_invoice_review',
                   side_effect=AssertionError('Unauthenticated review')):
            self.app.run()
        self.no_portfolio()
        self.assertFalse(local_auth.has_accounts(self.auth))
        self.assertIn('Set up your workspace', [item.value for item in self.app.subheader])
        for field in self.app.text_input:
            self.assertEqual(field.value, '')
            if 'password' in field.label.lower():
                self.assertEqual(field.proto.type, 1)

    def test_first_owner_setup_requires_matching_password_then_explicit_signin(self):
        self.app.run()
        self.widget('text_input', 'Your name').set_value('Local owner')
        self.widget('text_input', 'Username').set_value('local-owner')
        self.widget('text_input', 'Password').set_value(PASSWORD)
        self.widget('text_input', 'Confirm password').set_value('A different test passphrase')
        self.widget('button', 'Create owner account').click().run()
        self.assertFalse(local_auth.has_accounts(self.auth))
        self.assertTrue(any('do not match' in item.value for item in self.app.error))
        self.widget('text_input', 'Your name').set_value('Local owner')
        self.widget('text_input', 'Username').set_value('local-owner')
        self.widget('text_input', 'Password').set_value(PASSWORD)
        self.widget('text_input', 'Confirm password').set_value(PASSWORD)
        self.widget('button', 'Create owner account').click().run()
        self.no_portfolio()
        self.assertTrue(local_auth.has_accounts(self.auth))
        self.assertNotIn('auth:token', self.app.session_state)
        self.assertTrue(any(item.label == 'Sign in' for item in self.app.button))

    def test_wrong_password_cannot_open_workspace_and_valid_signin_does_not_change_portfolio(self):
        self.provision()
        before = self.contents()
        with patch('recovery.storage.connect', side_effect=AssertionError('Unauthenticated read')):
            self.login(password='A valid-length but wrong password')
        self.no_portfolio()
        self.assertTrue(any('Sign-in unsuccessful' in item.value for item in self.app.error))
        self.login()
        self.assertTrue(any(button.key == 'navigation:Overview' for button in self.app.button))
        self.assertTrue(any(button.label == 'Sign out' for button in self.app.button))
        self.assertEqual(self.contents(), before)

    def test_signout_revokes_session_and_clears_chat_selection_consent_and_credentials(self):
        self.provision()
        self.login()
        token = self.app.session_state['auth:token']
        for key in ('chat:old-answer', 'invoice_selection:old', 'consent:old'):
            self.app.session_state[key] = 'previous account private state'
        with patch('recovery.storage.connect', side_effect=AssertionError('Read after signout')):
            self.widget('button', 'Sign out').click().run()
        self.no_portfolio()
        self.assertIsNone(local_auth.current_user(self.auth, token))
        remaining = self.app.session_state
        for key in ('auth:token', 'chat:old-answer', 'invoice_selection:old', 'consent:old'):
            self.assertNotIn(key, remaining)

    def test_revoked_session_cannot_render_another_page_or_retain_previous_account_data(self):
        self.provision()
        self.login()
        token = self.app.session_state['auth:token']
        local_auth.logout(self.auth, token)
        self.app.session_state['chat:old-answer'] = 'discard this'
        with patch('recovery.storage.connect', side_effect=AssertionError('Revoked read')):
            self.app.button(key='navigation:Invoice review').click().run()
        self.no_portfolio()
        self.assertNotIn('chat:old-answer', self.app.session_state)
        self.assertTrue(any('session ended' in item.value for item in self.app.info))

    def test_account_store_failure_fails_closed_before_data_access(self):
        self.auth.write_text('This is not a database')
        self.auth.chmod(0o600)
        with patch('recovery.storage.connect', side_effect=AssertionError('Broken-store read')):
            self.app.run()
        self.no_portfolio()
        self.assertTrue(any('Portfolio access remains closed' in item.value for item in self.app.error))

    def test_owner_adds_accountant_without_granting_account_administration(self):
        self.provision()
        before = self.contents()
        self.login()
        self.widget('text_input', 'Accountant name').set_value('Demo accountant')
        self.widget('text_input', 'New username').set_value('new-accountant')
        self.widget('text_input', 'Temporary password').set_value(PASSWORD)
        self.widget('button', 'Add accountant').click().run()
        self.assertFalse(self.app.exception)
        accountant = self.new_app()
        self.login(username='new-accountant', app=accountant)
        self.assertTrue(any(button.key == 'navigation:Overview' for button in accountant.button))
        self.assertFalse(any(button.label == 'Add accountant' for button in accountant.button))
        self.assertFalse(any(button.label == 'Disable selected account' for button in accountant.button))
        self.assertEqual(self.contents(), before)

    def test_password_change_ends_session_and_old_password_stops_working(self):
        self.provision()
        self.login()
        token = self.app.session_state['auth:token']
        self.widget('text_input', 'Current password').set_value(PASSWORD)
        self.widget('text_input', 'New password').set_value(NEW_PASSWORD)
        self.widget('text_input', 'Confirm new password').set_value(NEW_PASSWORD)
        self.widget('button', 'Change password').click().run()
        self.no_portfolio()
        self.assertIsNone(local_auth.current_user(self.auth, token))
        self.assertIsNone(local_auth.authenticate(self.auth, 'local-owner', PASSWORD))
        self.login(password=NEW_PASSWORD)
        self.assertTrue(any(button.label == 'Sign out' for button in self.app.button))

    def test_owner_disables_account_and_its_open_session_loses_workspace_access(self):
        self.provision()
        owner = local_auth.authenticate(self.auth, 'local-owner', PASSWORD)
        local_auth.create_account(self.auth, owner, 'demo-accountant', 'Demo accountant', PASSWORD)
        self.login()
        accountant = self.new_app()
        self.login(username='demo-accountant', app=accountant)
        self.widget('selectbox', 'Account to disable').set_value('demo-accountant').run()
        self.widget('button', 'Disable selected account').click().run()
        self.assertFalse(self.app.exception)
        with patch('recovery.storage.connect', side_effect=AssertionError('Disabled-account read')):
            accountant.button(key='navigation:Invoice review').click().run()
        self.no_portfolio(accountant)
        self.assertIsNone(local_auth.authenticate(self.auth, 'demo-accountant', PASSWORD))

    def test_owner_reset_revokes_accountant_session_and_requires_replacement_password(self):
        self.provision()
        owner = local_auth.authenticate(self.auth, 'local-owner', PASSWORD)
        local_auth.create_account(self.auth, owner, 'demo-accountant', 'Demo accountant', PASSWORD)
        prior_token = local_auth.authenticate(self.auth, 'demo-accountant', PASSWORD)
        self.login()
        self.widget('selectbox', 'Accountant password to reset').set_value('demo-accountant')
        self.widget('text_input', 'Replacement temporary password').set_value(NEW_PASSWORD)
        self.widget('button', 'Reset accountant password').click().run()
        self.assertFalse(self.app.exception)
        self.assertIsNone(local_auth.current_user(self.auth, prior_token))
        self.assertIsNone(local_auth.authenticate(self.auth, 'demo-accountant', PASSWORD))
        self.assertIsNotNone(local_auth.authenticate(self.auth, 'demo-accountant', NEW_PASSWORD))

    def test_auth_database_switch_clears_old_session_without_reusing_authority(self):
        self.provision()
        self.login()
        self.app.session_state['chat:old-answer'] = 'old account answer'
        second_auth = self.directory / 'another.sqlite3'
        with patch.dict(os.environ, {'RECOVERY_AUTH_DB': str(second_auth)}), \
             patch('recovery.storage.connect', side_effect=AssertionError('Cross-store read')):
            self.app.run()
        self.no_portfolio()
        self.assertFalse(local_auth.has_accounts(second_auth))
        self.assertNotIn('auth:token', self.app.session_state)
        self.assertNotIn('chat:old-answer', self.app.session_state)


if __name__ == '__main__':
    unittest.main()
