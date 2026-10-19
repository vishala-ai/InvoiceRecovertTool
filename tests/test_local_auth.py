"""Real local credential/session checks use isolated authentication stores only."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import threading
import unittest
from unittest import mock

from recovery import local_auth as auth


PASSWORD = 'Synthetic owner test passphrase 2026'
NEW_PASSWORD = 'A different synthetic passphrase 2027'
ACCOUNTANT_PASSWORD = 'Synthetic accountant passphrase 2026'


class LocalAuthStorageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.db = self.directory / 'accounts' / 'local.sqlite3'

    def test_initialization_is_private_empty_and_idempotent(self):
        auth.initialize(self.db)
        self.assertFalse(auth.has_accounts(self.db))
        self.assertEqual(stat.S_IMODE(self.db.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.db.parent.stat().st_mode), 0o700)
        before = self.db.read_bytes()
        auth.initialize(self.db)
        self.assertEqual(self.db.read_bytes(), before)
        with sqlite3.connect(self.db) as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables, {'auth_meta', 'auth_accounts', 'auth_sessions', 'auth_throttles'})

    def test_business_database_is_rejected_without_mutation(self):
        self.db.parent.mkdir(mode=0o700)
        with sqlite3.connect(self.db) as conn:
            conn.execute('CREATE TABLE invoice_lines (id TEXT PRIMARY KEY)')
            conn.execute("INSERT INTO invoice_lines VALUES ('L001')")
        self.db.chmod(0o600)
        before = self.db.read_bytes()
        with self.assertRaisesRegex(auth.AuthError, 'separate'):
            auth.initialize(self.db)
        self.assertEqual(self.db.read_bytes(), before)

    def test_non_sqlite_file_and_incompatible_schema_are_not_overwritten(self):
        self.db.parent.mkdir(mode=0o700)
        self.db.write_bytes(b'not an account database')
        self.db.chmod(0o600)
        before = self.db.read_bytes()
        with self.assertRaises(auth.AuthError):
            auth.initialize(self.db)
        self.assertEqual(self.db.read_bytes(), before)
        self.db.unlink()
        auth.initialize(self.db)
        with sqlite3.connect(self.db) as conn:
            conn.execute('PRAGMA ignore_check_constraints=ON')
            conn.execute('UPDATE auth_meta SET schema_version=2')
        before = self.db.read_bytes()
        with self.assertRaises(auth.AuthError):
            auth.initialize(self.db)
        self.assertEqual(self.db.read_bytes(), before)

    def test_symbolic_link_file_parent_and_journal_are_rejected(self):
        auth.initialize(self.db)
        linked = self.directory / 'linked.sqlite3'
        linked.symlink_to(self.db)
        with self.assertRaises(auth.AuthError):
            auth.initialize(linked)
        linked_parent = self.directory / 'linked-accounts'
        linked_parent.symlink_to(self.db.parent, target_is_directory=True)
        with self.assertRaises(auth.AuthError):
            auth.initialize(linked_parent / 'other.sqlite3')
        journal = Path(f'{self.db}-journal')
        journal.symlink_to(self.db)
        with self.assertRaises(auth.AuthError):
            auth.initialize(self.db)

    def test_hard_link_public_file_and_shared_writable_directory_are_rejected(self):
        auth.initialize(self.db)
        linked = self.db.parent / 'hard-linked.sqlite3'
        os.link(self.db, linked)
        with self.assertRaises(auth.AuthError):
            auth.initialize(self.db)
        linked.unlink()
        self.db.chmod(0o644)
        with self.assertRaises(auth.AuthError):
            auth.initialize(self.db)
        self.db.chmod(0o600)
        self.db.parent.chmod(0o777)
        with self.assertRaises(auth.AuthError):
            auth.initialize(self.db)

    def test_queries_never_initialize_missing_store(self):
        with self.assertRaises(auth.AuthError):
            auth.has_accounts(self.db)
        self.assertFalse(self.db.parent.exists())
        self.assertIsNone(auth.current_user(self.db, 'not-a-token'))
        self.assertFalse(self.db.exists())

    def test_unsupported_scrypt_fails_closed_without_a_default_or_weaker_hash(self):
        auth.initialize(self.db)
        with mock.patch.object(auth, 'hashlib', object()):
            with self.assertRaisesRegex(auth.AuthError, 'start the app with uv'):
                auth.bootstrap_owner(self.db, 'owner', 'Demo owner', PASSWORD)
        with mock.patch.object(auth.hashlib, 'scrypt', create=True) as unavailable:
            unavailable.side_effect = ValueError('unavailable')
            with self.assertRaises(auth.AuthError):
                auth.bootstrap_owner(self.db, 'owner', 'Demo owner', PASSWORD)
        self.assertFalse(auth.has_accounts(self.db))


@unittest.skipUnless(hasattr(hashlib, 'scrypt'), 'OpenSSL scrypt is unavailable; use uv Python')
class LocalAuthAccountTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.db = self.directory / 'accounts' / 'local.sqlite3'
        auth.initialize(self.db)
        clock_patch = mock.patch.object(auth, '_now', return_value=1_700_000_000)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.owner = auth.bootstrap_owner(self.db, 'Owner', 'Demo Owner', PASSWORD)

    def rows(self, table):
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid')]

    def owner_token(self):
        token = auth.authenticate(self.db, 'owner', PASSWORD)
        self.assertIsInstance(token, str)
        return token

    def accountant(self, owner_token=None, username='accountant'):
        auth.create_account(self.db, owner_token or self.owner_token(), username,
                            'Demo Accountant', ACCOUNTANT_PASSWORD)
        token = auth.authenticate(self.db, username, ACCOUNTANT_PASSWORD)
        self.assertIsInstance(token, str)
        return token

    def test_bootstrap_owner_has_no_implicit_session_or_public_secret(self):
        self.assertEqual(self.owner, {'username': 'owner', 'display_name': 'Demo Owner',
                                     'role': 'owner', 'active': True,
                                     'created_at': '2023-11-14T22:13:20+00:00'})
        self.assertTrue(auth.has_accounts(self.db))
        self.assertEqual(self.rows('auth_sessions'), [])
        with self.assertRaises(PermissionError):
            auth.bootstrap_owner(self.db, 'other-owner', 'Another owner', PASSWORD)
        self.assertEqual(len(self.rows('auth_accounts')), 1)

    def test_salted_owasp_scrypt_and_tokens_are_never_stored_in_plaintext(self):
        token = self.owner_token()
        auth.create_account(self.db, token, 'accountant', 'Demo Accountant', PASSWORD)
        rows = self.rows('auth_accounts')
        self.assertEqual(len({row['salt'] for row in rows}), 2)
        self.assertEqual(len({row['password_hash'] for row in rows}), 2)
        for row in rows:
            self.assertEqual(len(row['salt']), 16)
            self.assertEqual(len(row['password_hash']), 32)
            expected = hashlib.scrypt(PASSWORD.encode(), salt=row['salt'],
                                      n=32768, r=8, p=3, maxmem=64*1024*1024, dklen=32)
            self.assertEqual(row['password_hash'], expected)
        sessions = self.rows('auth_sessions')
        self.assertEqual(sessions[0]['token_hash'], hashlib.sha256(token.encode()).hexdigest())
        stored = self.db.read_bytes()
        self.assertNotIn(PASSWORD.encode(), stored)
        self.assertNotIn(token.encode(), stored)

    def test_login_normalizes_only_username_and_uses_exact_password(self):
        token = auth.authenticate(self.db, '  OWNER ', PASSWORD)
        self.assertEqual(auth.current_user(self.db, token)['username'], 'owner')
        self.assertIsNone(auth.authenticate(self.db, 'owner', PASSWORD + ' '))
        self.assertIsNone(auth.authenticate(self.db, 'unknown', PASSWORD))
        self.assertIsNone(auth.authenticate(self.db, "owner' OR 1=1 --", PASSWORD))
        self.assertEqual(len(self.rows('auth_accounts')), 1)

    def test_unknown_names_and_wrong_passwords_take_the_same_scrypt_path(self):
        real_derive = auth._derive
        with mock.patch.object(auth, '_derive', wraps=real_derive) as derive:
            self.assertIsNone(auth.authenticate(self.db, 'owner', NEW_PASSWORD))
            self.assertIsNone(auth.authenticate(self.db, 'nonexistent', NEW_PASSWORD))
        self.assertEqual(derive.call_count, 2)
        self.assertEqual([call.args[0] for call in derive.call_args_list],
                         [NEW_PASSWORD.encode(), NEW_PASSWORD.encode()])

    def test_malformed_or_unbounded_login_inputs_are_rejected_before_hashing(self):
        with mock.patch.object(auth, '_derive') as derive:
            for username, password in [('x'*100000, PASSWORD), ('owner', 'x'*100000),
                                       ('owner', None), (None, PASSWORD), ('owner', 'short')]:
                self.assertIsNone(auth.authenticate(self.db, username, password))
        derive.assert_not_called()
        self.assertEqual(self.rows('auth_sessions'), [])

    def test_unicode_passphrase_is_supported_without_truncation(self):
        token = self.owner_token()
        password = 'A long Unicode passphrase 🦉 café 日本語'
        auth.create_account(self.db, token, 'unicode', 'Demo Unicode', password)
        signed_in = auth.authenticate(self.db, 'unicode', password)
        self.assertEqual(auth.current_user(self.db, signed_in)['username'], 'unicode')
        self.assertIsNone(auth.authenticate(self.db, 'unicode', password[:-1]))

    def test_owner_alone_creates_lists_and_disables_accounts(self):
        owner = self.owner_token()
        accountant = self.accountant(owner)
        public = auth.list_accounts(self.db, owner)
        self.assertEqual({row['username'] for row in public}, {'owner', 'accountant'})
        self.assertEqual({row['role'] for row in public}, {'owner', 'accountant'})
        self.assertTrue(all(set(row) == set(self.owner) for row in public))
        before = self.rows('auth_accounts')
        for token in (accountant, 'forged-token'):
            with self.assertRaises(PermissionError):
                auth.create_account(self.db, token, 'intruder', 'Intruder', PASSWORD)
            with self.assertRaises(PermissionError):
                auth.list_accounts(self.db, token)
            with self.assertRaises(PermissionError):
                auth.disable_account(self.db, token, 'owner')
        self.assertEqual(self.rows('auth_accounts'), before)

    def test_account_fields_are_bounded_validated_and_duplicate_names_do_not_replace(self):
        owner = self.owner_token()
        before = self.rows('auth_accounts')
        invalid = [('', 'Demo', PASSWORD), ('ab', 'Demo', PASSWORD),
                   ('a b', 'Demo', PASSWORD), ('éclair', 'Demo', PASSWORD),
                   ('normal', '', PASSWORD), ('normal', 'x'*81, PASSWORD),
                   ('normal', 'Control\nName', PASSWORD), ('normal', 'Demo', 'too short'),
                   ('normal', 'Demo', ' '*16), ('normal', 'Demo', 'x'*129)]
        for username, name, password in invalid:
            with self.subTest(username=username, name=name[:10]):
                with self.assertRaises(ValueError):
                    auth.create_account(self.db, owner, username, name, password)
        with self.assertRaises(ValueError):
            auth.create_account(self.db, owner, 'OWNER', 'Replacement', NEW_PASSWORD)
        self.assertEqual(self.rows('auth_accounts'), before)

    def test_disabling_account_revokes_every_session_and_prevents_login(self):
        owner = self.owner_token()
        first = self.accountant(owner)
        second = auth.authenticate(self.db, 'accountant', ACCOUNTANT_PASSWORD)
        auth.disable_account(self.db, owner, 'accountant')
        self.assertIsNone(auth.current_user(self.db, first))
        self.assertIsNone(auth.current_user(self.db, second))
        self.assertIsNone(auth.authenticate(self.db, 'accountant', ACCOUNTANT_PASSWORD))
        self.assertFalse(next(row for row in auth.list_accounts(self.db, owner)
                              if row['username'] == 'accountant')['active'])
        self.assertIsNotNone(auth.current_user(self.db, owner))

    def test_last_owner_cannot_be_disabled_and_setup_never_reopens(self):
        owner = self.owner_token()
        before = self.rows('auth_accounts')
        with self.assertRaisesRegex(ValueError, 'last active'):
            auth.disable_account(self.db, owner, 'owner')
        self.assertEqual(self.rows('auth_accounts'), before)
        self.assertIsNotNone(auth.current_user(self.db, owner))
        self.assertTrue(auth.has_accounts(self.db))

    def test_forged_tokens_and_invalid_passwords_cannot_change_credentials(self):
        owner = self.owner_token()
        before = self.rows('auth_accounts')
        with self.assertRaises(PermissionError):
            auth.change_password(self.db, 'forged-token', PASSWORD, NEW_PASSWORD)
        with self.assertRaises(PermissionError):
            auth.change_password(self.db, owner, NEW_PASSWORD, NEW_PASSWORD)
        with self.assertRaises(ValueError):
            auth.change_password(self.db, owner, PASSWORD, 'short')
        self.assertEqual(self.rows('auth_accounts'), before)
        self.assertIsNotNone(auth.current_user(self.db, owner))

    def test_password_change_revokes_all_own_tokens_and_rotates_the_salt(self):
        owner = self.owner_token()
        second = self.owner_token()
        accountant = self.accountant(owner)
        old = self.rows('auth_accounts')[0]
        auth.change_password(self.db, owner, PASSWORD, NEW_PASSWORD)
        self.assertIsNone(auth.current_user(self.db, owner))
        self.assertIsNone(auth.current_user(self.db, second))
        self.assertIsNotNone(auth.current_user(self.db, accountant))
        self.assertIsNone(auth.authenticate(self.db, 'owner', PASSWORD))
        self.assertIsNotNone(auth.authenticate(self.db, 'owner', NEW_PASSWORD))
        new = self.rows('auth_accounts')[0]
        self.assertNotEqual(new['salt'], old['salt'])
        self.assertEqual(new['security_version'], old['security_version'] + 1)

    def test_owner_reset_revokes_target_sessions_without_touching_other_users(self):
        owner = self.owner_token()
        first = self.accountant(owner)
        second = auth.authenticate(self.db, 'accountant', ACCOUNTANT_PASSWORD)
        auth.reset_account_password(self.db, owner, 'accountant', NEW_PASSWORD)
        self.assertIsNone(auth.current_user(self.db, first))
        self.assertIsNone(auth.current_user(self.db, second))
        self.assertIsNotNone(auth.current_user(self.db, owner))
        self.assertIsNone(auth.authenticate(self.db, 'accountant', ACCOUNTANT_PASSWORD))
        self.assertIsNotNone(auth.authenticate(self.db, 'accountant', NEW_PASSWORD))

    def test_reset_requires_fresh_owner_and_active_accountant_and_valid_new_password(self):
        owner = self.owner_token()
        accountant = self.accountant(owner)
        before = self.rows('auth_accounts')
        for token in (accountant, 'forged-token'):
            with self.assertRaises(PermissionError):
                auth.reset_account_password(self.db, token, 'accountant', NEW_PASSWORD)
        for username, password in [('owner', NEW_PASSWORD), ('unknown', NEW_PASSWORD),
                                   ('accountant', 'short')]:
            with self.assertRaises(ValueError):
                auth.reset_account_password(self.db, owner, username, password)
        self.assertEqual(self.rows('auth_accounts'), before)
        self.assertIsNotNone(auth.current_user(self.db, accountant))
        auth.disable_account(self.db, owner, 'accountant')
        with self.assertRaises(ValueError):
            auth.reset_account_password(self.db, owner, 'accountant', NEW_PASSWORD)
        auth.logout(self.db, owner)
        with self.assertRaises(PermissionError):
            auth.reset_account_password(self.db, owner, 'accountant', NEW_PASSWORD)

    def test_logout_invalidates_only_the_given_session_and_replay_is_harmless(self):
        first, second = self.owner_token(), self.owner_token()
        auth.logout(self.db, first)
        auth.logout(self.db, first)
        auth.logout(self.db, 'malformed')
        self.assertIsNone(auth.current_user(self.db, first))
        self.assertIsNotNone(auth.current_user(self.db, second))

    def test_idle_expiry_is_exact_and_invalid_requests_do_not_extend_it(self):
        owner = self.owner_token()
        self.clock.return_value += auth.IDLE_TIMEOUT_SECONDS - 1
        self.assertIsNotNone(auth.current_user(self.db, owner))
        last_seen = self.rows('auth_sessions')[0]['last_seen_at']
        self.clock.return_value += auth.IDLE_TIMEOUT_SECONDS
        forged = owner[:-1] + ('A' if owner[-1] != 'A' else 'B')
        self.assertIsNone(auth.current_user(self.db, forged))
        self.assertEqual(self.rows('auth_sessions')[0]['last_seen_at'], last_seen)
        self.assertIsNone(auth.current_user(self.db, owner))
        self.assertEqual(self.rows('auth_sessions'), [])

    def test_absolute_expiry_cannot_be_kept_alive_by_activity(self):
        owner = self.owner_token()
        for _ in range(auth.ABSOLUTE_TIMEOUT_SECONDS // 600 - 1):
            self.clock.return_value += 600
            self.assertIsNotNone(auth.current_user(self.db, owner))
        self.clock.return_value += 600
        self.assertIsNone(auth.current_user(self.db, owner))

    def test_clock_rollback_fails_closed(self):
        owner = self.owner_token()
        self.clock.return_value -= 1
        self.assertIsNone(auth.current_user(self.db, owner))

    def test_session_count_is_bounded_and_newest_sessions_remain_valid(self):
        tokens = [self.owner_token() for _ in range(auth.MAX_SESSIONS_PER_ACCOUNT + 2)]
        self.assertEqual(len(self.rows('auth_sessions')), auth.MAX_SESSIONS_PER_ACCOUNT)
        self.assertIsNone(auth.current_user(self.db, tokens[0]))
        self.assertIsNone(auth.current_user(self.db, tokens[1]))
        self.assertTrue(all(auth.current_user(self.db, token) for token in tokens[2:]))

    def test_username_throttle_is_persistent_equal_for_known_and_unknown_and_expires(self):
        real_derive = auth._derive
        with mock.patch.object(auth, '_derive', wraps=real_derive) as derive:
            for username in ('owner', 'unknown'):
                for _ in range(auth.USERNAME_ATTEMPT_LIMIT):
                    self.assertIsNone(auth.authenticate(self.db, username, NEW_PASSWORD))
                # Reinitializing represents a new process: persisted counters survive.
                auth.initialize(self.db)
                self.assertIsNone(auth.authenticate(self.db, username, PASSWORD))
        self.assertEqual(derive.call_count, 2 * auth.USERNAME_ATTEMPT_LIMIT)
        self.clock.return_value += auth.ATTEMPT_WINDOW_SECONDS
        self.assertIsNotNone(auth.authenticate(self.db, 'owner', PASSWORD))
        self.assertLessEqual(len(self.rows('auth_throttles')), 2)

    def test_global_throttle_prevents_username_rotation_and_bounds_persisted_rows(self):
        with mock.patch.object(auth, '_derive', return_value=bytes(32)) as derive:
            for index in range(auth.GLOBAL_ATTEMPT_LIMIT + 20):
                self.assertIsNone(auth.authenticate(self.db, f'unknown-{index}', PASSWORD))
        self.assertEqual(derive.call_count, auth.GLOBAL_ATTEMPT_LIMIT)
        self.assertEqual(len(self.rows('auth_throttles')), auth.GLOBAL_ATTEMPT_LIMIT + 1)
        auth.initialize(self.db)
        self.assertIsNone(auth.authenticate(self.db, 'owner', PASSWORD))
        self.clock.return_value += auth.ATTEMPT_WINDOW_SECONDS
        self.assertIsNotNone(auth.authenticate(self.db, 'owner', PASSWORD))
        self.assertEqual(len(self.rows('auth_throttles')), 1)

    def test_success_resets_only_username_failures_and_password_checks_are_throttled(self):
        for _ in range(auth.USERNAME_ATTEMPT_LIMIT - 1):
            self.assertIsNone(auth.authenticate(self.db, 'owner', NEW_PASSWORD))
        owner = self.owner_token()
        for _ in range(auth.USERNAME_ATTEMPT_LIMIT):
            with self.assertRaises(PermissionError):
                auth.change_password(self.db, owner, NEW_PASSWORD, NEW_PASSWORD)
        with mock.patch.object(auth, '_derive') as derive:
            with self.assertRaises(PermissionError):
                auth.change_password(self.db, owner, PASSWORD, NEW_PASSWORD)
        derive.assert_not_called()
        self.assertIsNotNone(auth.current_user(self.db, owner))

    def test_concurrent_first_account_creation_has_exactly_one_owner(self):
        other_db = self.directory / 'race' / 'local.sqlite3'
        auth.initialize(other_db)
        barrier = threading.Barrier(2)
        real_credentials = auth._credentials

        def delayed_credentials(password):
            barrier.wait(timeout=10)
            return real_credentials(password)

        def bootstrap(index):
            try:
                return auth.bootstrap_owner(other_db, f'owner-{index}', 'Demo Owner', PASSWORD)
            except PermissionError:
                return None

        with mock.patch.object(auth, '_credentials', side_effect=delayed_credentials):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(bootstrap, [1, 2]))
        self.assertEqual(sum(result is not None for result in results), 1)
        with sqlite3.connect(other_db) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM auth_accounts').fetchone()[0], 1)

    def test_password_changed_during_login_cannot_issue_a_stale_session(self):
        real_matches = auth._matches

        def concurrent_change(encoded, row):
            result = real_matches(encoded, row)
            with sqlite3.connect(self.db) as conn:
                conn.execute('UPDATE auth_accounts SET security_version=security_version+1')
            return result

        with mock.patch.object(auth, '_matches', side_effect=concurrent_change):
            self.assertIsNone(auth.authenticate(self.db, 'owner', PASSWORD))
        self.assertEqual(self.rows('auth_sessions'), [])


if __name__ == '__main__':
    unittest.main()
