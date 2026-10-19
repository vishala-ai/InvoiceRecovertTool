"""Durable-store protocol checks with real SQLite and an in-memory PG test double.

These tests do NOT open a PostgreSQL connection or establish Supabase behavior.
The fake implements only the transaction/row-lock contract used by CloudStore.
The embedded public certificate is solely a generated parser-test fixture.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import copy
import hashlib
from pathlib import Path
import sqlite3
import stat
import tempfile
import threading
import unittest
from unittest.mock import patch
import zlib

from recovery import cloud_store as store


DSN = 'postgresql://synthetic:do-not-print-secret@db.example.test:5432/postgres'
CERTIFICATE = '''-----BEGIN CERTIFICATE-----
MIICxjCCAa4CCQDwVN6xzHlFnDANBgkqhkiG9w0BAQsFADAlMSMwIQYDVQQDDBpz
eW50aGV0aWMtY2xvdWQtc3RvcmUtdGVzdDAeFw0yNjEwMTkwMDA4MTRaFw0yNjEw
MjEwMDA4MTRaMCUxIzAhBgNVBAMMGnN5bnRoZXRpYy1jbG91ZC1zdG9yZS10ZXN0
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA51dWzU4dyt6kxaqg4SK4
m+83nnr8ByP+8okrU/iGUZPTuBg8EYJ2CAXs147buR61DCspeGOtVQEqaw+TOkKQ
r4zZyHePsd551D3aYfb1RLBSIpUmGeO9CEoY1Rb6kgJTE3nWD3FgZlSfZQ39oBwX
3sPl6csgkluJJcZuxWM7KUZodDgZrjX0qKxHyH6oLr8PoEA+f3tAaF0NzBgenCE/
CcR+7UokH82I8azbvgi89m3mtZoQqUHYDGtxjoydxRiFlXT7jtZ/fESYg1dzpvuw
SI8oH/+fd+6CB7ysJQDb/wzSwpcXL0peZOGmoN+PsaccorT66lKyXSc6ujaubUyA
cQIDAQABMA0GCSqGSIb3DQEBCwUAA4IBAQAgt/ZVWwOANPcMwyvrcVFxTKC6pkax
maUgzN+UkmNfz/vtGLMbXjuNoGDi/zSOuIiOG4qdJzxnQGVXS6kB2S7CwSEk35Y4
kEHlYzXtGu41IiBDPrxErr/lFdrgLSFy7f9UXTiktnfqMG/L0yAv07K8e7AEej7v
nQYJapM+nLG+PnVFt8aQ41bbcbmgiN8nbIE8lI9SAleoR9HkHLN0DsjirT2InuvN
WN2fkNqebAeQCOayLVrkQpv3Mxn6S052SQZszqJhE+dqmhGy81pcU/BPH0Zw4te0
2KxnERZdlnbrRolZzBAR6xmJLxFUIh9Uvc/PrzEiOQyhJZOEvHYYra0O
-----END CERTIFICATE-----'''


class _Result:
    def __init__(self, rows=()):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class _PostgresProtocol:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {}
        self.connections = []
        self.committed_roles = []
        self.fail_commit = None
        self.fail_write = False
        self.fail_connect = False

    def connect(self, dsn, **kwargs):
        if self.fail_connect:
            raise RuntimeError(dsn)
        certificate = Path(kwargs['sslrootcert'])
        connection = _Connection(self)
        connection.options = kwargs
        connection.certificate = certificate.read_text()
        connection.certificate_mode = stat.S_IMODE(certificate.stat().st_mode)
        connection.directory_mode = stat.S_IMODE(certificate.parent.stat().st_mode)
        self.connections.append(connection)
        return connection


class _Connection:
    def __init__(self, server):
        self.server = server
        self.locked = False
        self.closed = False
        self.local = None
        self.workspace = None
        self.changed = []
        self.statements = []

    def execute(self, query, arguments=()):
        query = ' '.join(query.split())
        self.statements.append((query, arguments))
        if query.startswith('SET LOCAL '):
            return _Result()
        if query.startswith('INSERT INTO invoice_recovery_private.workspace_heads '):
            self.workspace = arguments[0]
            return _Result()
        if query.endswith('FOR UPDATE'):
            self.server.lock.acquire()
            self.locked = True
            self.local = copy.deepcopy(self.server.data)
            self.local.setdefault(self.workspace, {'revision': 0, 'files': {}})
            return _Result([(self.local[self.workspace]['revision'],)])
        if query.startswith('SELECT role,sha256,raw_size,payload '):
            return _Result([(role, *row) for role, row
                            in sorted(self.local[self.workspace]['files'].items())])
        if query.startswith('INSERT INTO invoice_recovery_private.workspace_files '):
            if self.server.fail_write:
                raise RuntimeError(DSN)
            workspace, role, digest, size, payload = arguments
            self.local[workspace]['files'][role] = (digest, size, payload)
            self.changed.append(role)
            return _Result()
        if query.startswith('UPDATE invoice_recovery_private.workspace_heads '):
            workspace, expected = arguments
            head = self.local[workspace]
            if head['revision'] != expected:
                return _Result()
            head['revision'] += 1
            return _Result([(head['revision'],)])
        raise AssertionError('Unexpected SQL shape: ' + query)

    def commit(self):
        if self.server.fail_commit == 'before':
            self.server.fail_commit = None
            raise RuntimeError(DSN)
        self.server.data = copy.deepcopy(self.local)
        self.server.committed_roles.append(tuple(self.changed))
        self.rollback()
        if self.server.fail_commit == 'after':
            self.server.fail_commit = None
            raise RuntimeError(DSN)

    def rollback(self):
        if self.locked:
            self.locked = False
            self.server.lock.release()

    def close(self):
        self.rollback()
        self.closed = True


def _seed(files, version='v1'):
    for role, path in files.items():
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute('CREATE TABLE journal (id TEXT PRIMARY KEY, value TEXT)')
            connection.execute('INSERT INTO journal VALUES (?,?)', ('event-1', role + '-' + version))


def _read(path):
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute('SELECT value FROM journal WHERE id=?', ('event-1',)).fetchone()[0]


def _change(files, version):
    for role, path in files.items():
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute('UPDATE journal SET value=? WHERE id=?',
                               (role + '-' + version, 'event-1'))


class DurableCloudStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.server = _PostgresProtocol()
        self.store = store.CloudStore(DSN, root_cert_pem=CERTIFICATE, connector=self.server.connect)
        self.working = self.directory / 'worker-a'

    def seed(self):
        with self.store.workspace('demo', self.working) as files:
            self.assertTrue(all(not path.exists() for path in files.values()))
            _seed(files)

    def test_restart_restores_coupled_review_checkpoint_and_account_state(self):
        self.seed()
        with self.store.workspace('demo', self.working) as files:
            _change(files, 'v2')
        restarted = store.CloudStore(DSN, root_cert_pem=CERTIFICATE, connector=self.server.connect)
        with restarted.workspace('demo', self.directory / 'new-worker') as files:
            self.assertEqual({role: _read(path) for role, path in files.items()},
                             {role: role + '-v2' for role in files})
            self.assertTrue(all(stat.S_IMODE(path.stat().st_mode) == 0o600
                                for path in files.values()))
            self.assertEqual(stat.S_IMODE(files['portfolio'].parent.stat().st_mode), 0o700)
        self.assertEqual(self.server.data['demo']['revision'], 2)
        self.assertEqual(set(self.server.data['demo']['files']), set(store._FILES))

    def test_only_changed_role_is_updated_and_readonly_rerun_keeps_revision(self):
        self.seed()
        with self.store.workspace('demo', self.working) as files:
            self.assertEqual(_read(files['portfolio']), 'portfolio-v1')
        self.assertEqual(self.server.data['demo']['revision'], 1)
        self.assertEqual(self.server.committed_roles[-1], ())
        with self.store.workspace('demo', self.working) as files:
            with closing(sqlite3.connect(files['accounts'])) as connection, connection:
                connection.execute("UPDATE journal SET value='disabled'")
        self.assertEqual(self.server.committed_roles[-1], ('accounts',))
        with self.store.workspace('demo', self.directory / 'second-worker') as files:
            self.assertEqual(_read(files['accounts']), 'disabled')
            self.assertEqual(_read(files['portfolio']), 'portfolio-v1')

    def test_application_exception_rolls_back_all_roles_and_discards_local_mutations(self):
        self.seed()
        before = copy.deepcopy(self.server.data)
        with self.assertRaisesRegex(ValueError, 'application defect'):
            with self.store.workspace('demo', self.working) as files:
                _change(files, 'uncommitted')
                raise ValueError('application defect')
        self.assertEqual(self.server.data, before)
        self.assertTrue(all(not path.exists() for path in files.values()))
        with self.store.workspace('demo', self.working) as files:
            self.assertTrue(all(_read(path) == role + '-v1' for role, path in files.items()))

    def test_no_partial_first_publication_or_fallback_to_unregistered_local_accounts(self):
        self.working.mkdir()
        stale = {role: self.working / name for role, name in store._FILES.items()}
        _seed(stale, 'must-not-import')
        with self.assertRaises(store.CloudStoreError):
            with self.store.workspace('demo', self.working) as files:
                self.assertFalse(any(path.exists() for path in files.values()))
                _seed({'portfolio': files['portfolio']})
        self.assertEqual(self.server.data, {})

    def test_commit_failure_is_sanitized_and_prior_committed_bundle_wins(self):
        self.seed()
        self.server.fail_commit = 'before'
        with self.assertRaisesRegex(store.CloudStoreError, 'confirmation is unavailable') as result:
            with self.store.workspace('demo', self.working) as files:
                _change(files, 'unconfirmed')
        self.assertNotIn('do-not-print-secret', str(result.exception))
        self.assertNotIn('not saved', str(result.exception).lower())
        with self.store.workspace('demo', self.working) as files:
            self.assertTrue(all(_read(path) == role + '-v1' for role, path in files.items()))

    def test_ambiguous_commit_restores_saved_state_without_duplicate_journal_events(self):
        self.seed()
        self.server.fail_commit = 'after'
        with self.assertRaisesRegex(store.CloudStoreError, 'confirmation is unavailable'):
            with self.store.workspace('demo', self.working) as files:
                _change(files, 'actually-committed')
        with self.store.workspace('demo', self.working) as files:
            self.assertTrue(all(_read(path) == role + '-actually-committed'
                                for role, path in files.items()))
            for path in files.values():
                with closing(sqlite3.connect(path)) as connection:
                    self.assertEqual(connection.execute('SELECT count(*) FROM journal').fetchone()[0], 1)
        self.assertEqual(self.server.data['demo']['revision'], 2)

    def test_write_failure_publishes_none_of_the_three_databases(self):
        self.seed()
        before = copy.deepcopy(self.server.data)
        self.server.fail_write = True
        with self.assertRaises(store.CloudStoreError) as error:
            with self.store.workspace('demo', self.working) as files:
                _change(files, 'v2')
        self.assertEqual(self.server.data, before)
        self.assertNotIn('do-not-print-secret', str(error.exception))

    def test_two_workers_serialize_and_second_observes_first_commit(self):
        self.seed()
        entered, attempted, release, second_entered = (threading.Event() for _ in range(4))

        def first():
            with self.store.workspace('demo', self.working) as files:
                _change(files, 'v2')
                entered.set()
                if not release.wait(3):
                    raise AssertionError('test coordination timeout')

        def second():
            attempted.set()
            separate_store = store.CloudStore(DSN, root_cert_pem=CERTIFICATE,
                                             connector=self.server.connect)
            with separate_store.workspace('demo', self.directory / 'worker-b') as files:
                second_entered.set()
                self.assertTrue(all(_read(path) == role + '-v2' for role, path in files.items()))
                _change(files, 'v3')

        with ThreadPoolExecutor(max_workers=2) as executor:
            a = executor.submit(first)
            self.assertTrue(entered.wait(3))
            b = executor.submit(second)
            self.assertTrue(attempted.wait(3))
            self.assertFalse(second_entered.wait(0.05))
            release.set()
            a.result(3)
            b.result(3)
        self.assertEqual(self.server.data['demo']['revision'], 3)

    def test_control_exception_must_be_deferred_until_after_successful_context(self):
        class SimulatedStreamlitControl(BaseException):
            pass

        control = None
        with self.store.workspace('demo', self.working) as files:
            try:
                _seed(files)
                raise SimulatedStreamlitControl()
            except SimulatedStreamlitControl as caught:
                control = caught
        self.assertIsInstance(control, SimulatedStreamlitControl)
        self.assertEqual(self.server.data['demo']['revision'], 1)

    def test_sqlite_backup_includes_committed_wal_pages(self):
        keeper = None
        try:
            with self.store.workspace('demo', self.working) as files:
                _seed(files)
                keeper = sqlite3.connect(files['checkpoints'])
                keeper.execute('PRAGMA journal_mode=WAL')
                keeper.execute("UPDATE journal SET value='checkpoint-in-wal'")
                keeper.commit()
                self.assertTrue(files['checkpoints'].with_name(
                    files['checkpoints'].name + '-wal').exists())
        finally:
            if keeper is not None:
                keeper.close()
        with self.store.workspace('demo', self.directory / 'restart') as files:
            self.assertEqual(_read(files['checkpoints']), 'checkpoint-in-wal')

    def test_tls_settings_certificate_permissions_and_timeouts_are_enforced(self):
        self.seed()
        connection = self.server.connections[-1]
        self.assertEqual(connection.options['sslmode'], 'verify-full')
        self.assertIsNone(connection.options['prepare_threshold'])
        self.assertEqual(connection.options['connect_timeout'], 10)
        self.assertFalse(connection.options['autocommit'])
        self.assertEqual(connection.certificate, CERTIFICATE)
        self.assertEqual(connection.certificate_mode, 0o600)
        self.assertEqual(connection.directory_mode, 0o700)
        self.assertFalse(Path(connection.options['sslrootcert']).exists())
        self.assertTrue(connection.closed)
        statements = [statement for statement, _ in connection.statements]
        self.assertTrue(any('FOR UPDATE' in statement for statement in statements))
        self.assertEqual(sum(statement.startswith('SET LOCAL ') for statement in statements), 3)
        self.assertFalse(any(statement.startswith('CREATE ') for statement in statements))

    def test_connection_errors_do_not_disclose_uri_password(self):
        self.server.fail_connect = True
        with self.assertRaises(store.CloudStoreError) as error:
            with self.store.workspace('demo', self.working):
                self.fail('must not yield')
        self.assertNotIn('do-not-print-secret', str(error.exception))

    def test_hosted_uri_rejects_localhost_and_insecure_or_conflicting_parameters(self):
        for dsn in ('', 'host=db.example.test password=secret',
                    'postgresql://user:pass@localhost/database',
                    'postgresql://user:pass@127.0.0.1/database',
                    'postgresql://user:pass@127.1/database',
                    'postgresql://user:pass@LOCALHOST./database',
                    'postgresql://user:pass@[::1]/database',
                    'postgresql://user:pass@[::ffff:127.0.0.1]/database',
                    DSN + '?sslmode=disable', DSN + '?sslmode=require',
                    DSN + '?hostaddr=127.0.0.1', DSN + '?options=-c%20search_path=public',
                    DSN + '#fragment', DSN + '?sslmode=verify-full&sslmode=verify-full'):
            with self.subTest(dsn_shape='invalid'), self.assertRaises(store.CloudStoreError):
                store.CloudStore(dsn, root_cert_pem=CERTIFICATE, connector=self.server.connect)
        store.CloudStore(DSN + '?sslmode=verify-full', root_cert_pem=CERTIFICATE,
                         connector=self.server.connect)

    def test_missing_malformed_or_private_key_certificates_are_rejected(self):
        for pem in ('', 'not a certificate', '-----BEGIN CERTIFICATE-----\ninvalid',
                    CERTIFICATE + '\n-----BEGIN PRIVATE KEY-----'):
            with self.subTest(kind='invalid certificate'), self.assertRaises(store.CloudStoreError):
                store.CloudStore(DSN, root_cert_pem=pem, connector=self.server.connect)

    def test_workspace_ids_and_symlinked_directories_files_and_sidecars_are_rejected(self):
        for identifier in ('../demo', 'demo/other', 'demo space', '', 'a' * 65):
            with self.subTest(identifier=identifier), self.assertRaises(store.CloudStoreError):
                with self.store.workspace(identifier, self.working):
                    self.fail('must not yield')
        linked = self.directory / 'alias'
        linked.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaises(store.CloudStoreError):
            with self.store.workspace('demo', linked / 'worker'):
                self.fail('must not yield')
        self.working.mkdir()
        outside = self.directory / 'untouched'
        outside.write_text('preserve')
        for name in ('accounts.sqlite3', 'portfolio.sqlite3-wal'):
            alias = self.working / name
            alias.symlink_to(outside)
            with self.assertRaises(store.CloudStoreError):
                with self.store.workspace('demo', self.working):
                    self.fail('must not yield')
            self.assertEqual(outside.read_text(), 'preserve')
            alias.unlink()

    def test_tampered_hash_unknown_role_missing_role_and_non_sqlite_payload_fail_closed(self):
        self.seed()
        original = copy.deepcopy(self.server.data)
        mutations = ('hash', 'role', 'missing', 'sqlite', 'trailing', 'length')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.server.data = copy.deepcopy(original)
                rows = self.server.data['demo']['files']
                digest, size, payload = rows['accounts']
                if mutation == 'hash':
                    rows['accounts'] = ('0' * 64, size, payload)
                elif mutation == 'role':
                    rows['../other'] = rows.pop('accounts')
                elif mutation == 'missing':
                    rows.pop('accounts')
                elif mutation == 'sqlite':
                    raw = b'bad sqlite' * 100
                    rows['accounts'] = (hashlib.sha256(raw).hexdigest(), len(raw), zlib.compress(raw))
                elif mutation == 'trailing':
                    rows['accounts'] = (digest, size, payload + zlib.compress(b'extra stream'))
                elif mutation == 'length':
                    rows['accounts'] = (digest, size + 1, payload)
                with self.assertRaises(store.CloudStoreError):
                    with self.store.workspace('demo', self.directory / mutation):
                        self.fail('must not yield')

    def test_decompression_bomb_and_raw_compressed_and_total_caps_fail_closed(self):
        self.seed()
        original = copy.deepcopy(self.server.data)
        digest, size, payload = original['demo']['files']['accounts']
        self.server.data['demo']['files']['accounts'] = (digest, 100, zlib.compress(b'x' * 200000))
        with self.assertRaises(store.CloudStoreError):
            with self.store.workspace('demo', self.directory / 'bomb'):
                self.fail('must not yield')
        for limit in ('MAX_FILE_BYTES', 'MAX_TOTAL_BYTES', 'MAX_COMPRESSED_BYTES'):
            self.server.data = copy.deepcopy(original)
            with self.subTest(limit=limit), patch.object(store, limit, 100), \
                    self.assertRaises(store.CloudStoreError):
                with self.store.workspace('demo', self.directory / limit):
                    self.fail('must not yield')

    def test_fresh_snapshot_cannot_exceed_storage_cap(self):
        with patch.object(store, 'MAX_FILE_BYTES', 100), self.assertRaises(store.CloudStoreError):
            with self.store.workspace('demo', self.working) as files:
                _seed(files)
        self.assertEqual(self.server.data, {})

    def test_schema_is_private_dml_only_at_runtime_and_revokes_public_api_roles(self):
        sql = (Path(__file__).resolve().parents[1] / 'cloud' / 'schema.sql').read_text()
        self.assertIn('REVOKE ALL ON SCHEMA invoice_recovery_private FROM PUBLIC', sql)
        self.assertIn("ARRAY['anon', 'authenticated']", sql)
        self.assertIn("role IN ('portfolio', 'checkpoints', 'accounts')", sql)
        self.assertNotIn('CREATE ROLE', sql)


if __name__ == '__main__':
    unittest.main()
