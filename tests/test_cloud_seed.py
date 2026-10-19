"""Private publication uses disposable synthetic fixtures and a fake durable store."""
from __future__ import annotations

from contextlib import contextmanager, closing, redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from recovery.cloud_seed import CloudSeedError, check_seed, main, prepare_seed, publish_seed
from recovery.local_auth import has_accounts
from recovery.storage import connect, load_snapshot, metadata

DATA = Path(__file__).resolve().parents[1] / 'data'
try:
    GRAPH_AVAILABLE = importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
except ModuleNotFoundError:
    GRAPH_AVAILABLE = False


class FakeStore:
    def __init__(self, *, existing=False, fail_commit=False):
        self.existing, self.fail_commit = existing, fail_commit
        self.committed = False
        self.saved = None

    @contextmanager
    def workspace(self, workspace_id, directory):
        directory.mkdir(mode=0o700, parents=True)
        files = {'portfolio': directory / 'portfolio.sqlite3',
                 'checkpoints': directory / 'portfolio-ui-checkpoints.sqlite3',
                 'accounts': directory / 'accounts.sqlite3'}
        if self.existing:
            files['portfolio'].write_bytes(b'preexisting remote records')
        yield files
        if self.fail_commit:
            raise RuntimeError('postgres://operator:do-not-log@host.invalid/private')
        self.saved = {role: path.read_bytes() for role, path in files.items()}
        self.committed = True


@unittest.skipUnless(GRAPH_AVAILABLE, 'Install graph extra for real persisted checkpoint seed tests')
class CloudSeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'portfolio.sqlite3'
        self.checkpoints = self.root / 'portfolio-ui-checkpoints.sqlite3'
        load_snapshot(self.source, DATA)
        from recovery.workflow import run_persistent
        run_persistent(self.source, self.checkpoints, 'ui:seed-L001', line_id='L001')
        run_persistent(self.source, self.checkpoints, 'ui:seed-L001', decision={
            'decision': 'approve', 'reviewer': 'Synthetic seed reviewer',
            'note': 'Explicit synthetic approval for private publication fixture.'})

    def fingerprint(self, path):
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
            return {name: sorted(repr(row) for row in conn.execute('SELECT * FROM "' + name + '"'))
                    for name, in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def test_prepare_preserves_all_business_and_checkpoint_rows_and_starts_empty_accounts(self):
        before = self.fingerprint(self.source), self.fingerprint(self.checkpoints)
        unrelated_auth = self.root / 'accounts.sqlite3'
        unrelated_auth.write_bytes(b'this existing account file must never be transferred')
        stage = self.root / 'stage'
        with patch('recovery.ai.explain_case') as model:
            summary = prepare_seed(self.source, stage)
        model.assert_not_called()
        self.assertEqual(summary['current_cases'], 20)
        self.assertEqual(summary['candidate_total_usd'], '80707.50')
        self.assertEqual(summary['new_realized_recovery'], '0.00')
        self.assertEqual(summary['existing_review_events'], 1)
        self.assertEqual(summary['accounts_transferred'], 0)
        self.assertGreater(summary['checkpoint_case_snapshots'], 0)
        self.assertEqual(self.fingerprint(stage / 'portfolio.sqlite3'), before[0])
        self.assertEqual(self.fingerprint(stage / 'portfolio-ui-checkpoints.sqlite3'), before[1])
        self.assertEqual((self.fingerprint(self.source), self.fingerprint(self.checkpoints)), before)
        self.assertFalse(has_accounts(stage / 'accounts.sqlite3'))
        self.assertEqual(stage.stat().st_mode & 0o777, 0o700)
        for path in stage.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(unrelated_auth.read_bytes(), b'this existing account file must never be transferred')

    def test_check_requires_no_credentials_network_or_existing_account_access(self):
        with patch.dict(os.environ, {}, clear=True), patch('recovery.cloud_store.CloudStore') as store:
            result = check_seed(self.source)
        store.assert_not_called()
        self.assertEqual(result['publication'], 'not_attempted')
        self.assertEqual(result['model_calls'], 0)

    def test_source_and_checkpoint_writer_reservations_cover_both_backups(self):
        from recovery.cloud_seed import _backup
        checked = []

        def checked_backup(source, target):
            for path in (self.source, self.checkpoints):
                with closing(sqlite3.connect(path, timeout=0.01)) as competing:
                    with self.assertRaisesRegex(sqlite3.OperationalError, 'locked'):
                        competing.execute('BEGIN IMMEDIATE')
            checked.append(source)
            _backup(source, target)

        with patch('recovery.cloud_seed._backup', side_effect=checked_backup):
            prepare_seed(self.source, self.root / 'reserved-stage')
        self.assertEqual(checked, [self.source, self.checkpoints])

    def test_oversized_input_fails_before_backups_or_network(self):
        with patch('recovery.cloud_seed._MAX_FILE_BYTES', 100), patch(
                'recovery.cloud_seed._backup') as backup:
            with self.assertRaisesRegex(CloudSeedError, 'storage limit'):
                check_seed(self.source)
        backup.assert_not_called()

    def test_success_is_reported_only_after_store_context_commits_all_three_roles(self):
        store = FakeStore()
        result = publish_seed(self.source, workspace_id='private-demo', dsn='operator-secret',
                              root_cert_pem='public certificate', store_factory=lambda *a, **k: store)
        self.assertTrue(store.committed)
        self.assertEqual(set(store.saved), {'portfolio', 'checkpoints', 'accounts'})
        self.assertEqual(result['publication'], 'committed')
        restore = self.root / 'restored'
        restore.mkdir(mode=0o700)
        for role, raw in store.saved.items():
            path = restore / (role + '.sqlite3')
            path.write_bytes(raw)
            path.chmod(0o600)
        self.assertEqual(self.fingerprint(restore / 'portfolio.sqlite3'), self.fingerprint(self.source))
        self.assertEqual(self.fingerprint(restore / 'checkpoints.sqlite3'), self.fingerprint(self.checkpoints))
        self.assertFalse(has_accounts(restore / 'accounts.sqlite3'))

    def test_existing_remote_workspace_cannot_be_replaced(self):
        store = FakeStore(existing=True)
        before = self.fingerprint(self.source), self.fingerprint(self.checkpoints)
        with self.assertRaisesRegex(CloudSeedError, 'refuses replacement'):
            publish_seed(self.source, workspace_id='private-demo', dsn='operator-secret',
                         root_cert_pem='public certificate', store_factory=lambda *a, **k: store)
        self.assertFalse(store.committed)
        self.assertEqual((self.fingerprint(self.source), self.fingerprint(self.checkpoints)), before)

    def test_ambiguous_commit_never_reports_success_or_credentials(self):
        store = FakeStore(fail_commit=True)
        with self.assertRaises(CloudSeedError) as caught:
            publish_seed(self.source, workspace_id='private-demo', dsn='operator-secret',
                         root_cert_pem='public certificate', store_factory=lambda *a, **k: store)
        self.assertIn('Verify the remote workspace', str(caught.exception))
        self.assertNotIn('postgres:', str(caught.exception))
        self.assertNotIn('operator-secret', str(caught.exception))
        self.assertFalse(store.committed)

    def test_non_synthetic_unknown_dataset_and_malformed_manifest_rejected(self):
        with connect(self.source) as conn:
            original = json.loads(metadata(conn, 'manifest'))
        for changes in ({'synthetic': False}, {'synthetic': 1}, {'dataset_id': 'client-export'},
                        {'tenants': {}}):
            with connect(self.source) as conn, conn:
                conn.execute("UPDATE metadata SET value=? WHERE key='manifest'",
                             (json.dumps({**original, **changes}),))
            with self.subTest(changes=changes), self.assertRaises(CloudSeedError):
                check_seed(self.source)

    def test_changed_source_schema_hash_or_row_reconciliation_rejected(self):
        for sql in ("UPDATE documents SET body='replaced untrusted body' WHERE doc_id='DOC-A1'",
                    "UPDATE invoice_lines SET billed_amount='9999.00' WHERE line_id='L001'",
                    'CREATE TABLE auth_accounts(secret TEXT)'):
            copied = self.root / ('copy-' + hashlib.sha256(sql.encode()).hexdigest()[:8] + '.sqlite3')
            shutil.copyfile(self.source, copied)
            shutil.copyfile(self.checkpoints, copied.with_name(copied.stem + '-ui-checkpoints.sqlite3'))
            with closing(sqlite3.connect(copied)) as conn, conn:
                conn.execute(sql)
            with self.subTest(sql=sql), self.assertRaises(CloudSeedError):
                check_seed(copied)

    def test_missing_or_unrelated_checkpoint_file_is_rejected_without_creating_it(self):
        self.checkpoints.unlink()
        with self.assertRaises(CloudSeedError):
            check_seed(self.source)
        self.assertFalse(self.checkpoints.exists())
        with closing(sqlite3.connect(self.checkpoints)) as conn, conn:
            conn.execute('CREATE TABLE unrelated(data TEXT)')
        with self.assertRaises(CloudSeedError):
            check_seed(self.source)

    def test_foreign_checkpoint_source_anchor_is_rejected(self):
        other = self.root / 'other.sqlite3'
        load_snapshot(other, DATA)
        with connect(other) as conn, conn:
            conn.execute("UPDATE metadata SET value=? WHERE key='source_digest'", ('f' * 64,))
        foreign = self.root / 'foreign.sqlite3'
        from recovery.workflow import run_persistent
        run_persistent(other, foreign, 'ui:foreign', line_id='L001')
        shutil.copyfile(foreign, self.checkpoints)
        with self.assertRaisesRegex(CloudSeedError, 'do not match'):
            check_seed(self.source)

    def test_locked_synthetic_archive_reconciles_and_stays_separate_from_current_recovery(self):
        from recovery.demo_portfolio import historical_portfolio_payload
        from recovery.legacy_archive import import_archive
        import_archive(self.source, historical_portfolio_payload(vendor_count=1, cases_per_year=2))
        report = check_seed(self.source)
        self.assertEqual(report['locked_historical_cases'], 4)
        self.assertEqual(report['current_cases'], 20)
        self.assertEqual(report['candidate_total_usd'], '80707.50')
        self.assertEqual(report['new_realized_recovery'], '0.00')
        with connect(self.source) as conn, conn:
            conn.execute('DROP TRIGGER legacy_cases_immutable_update')
        with self.assertRaises(CloudSeedError):
            check_seed(self.source)

    def test_validated_synthetic_intake_paths_are_preserved_without_csv_path_coercion(self):
        from recovery.lifecycle import admit_document, stage_document
        from recovery.lifecycle_storage import migrate, scope_of
        migrate(self.source, backup_dir=self.root / 'backups')
        staged = stage_document(self.source, 'DEMO', 'synthetic-amendment.txt',
                                b'Invented synthetic source requiring human term review.', True)
        with connect(self.source) as conn:
            scope = scope_of(dict(conn.execute("SELECT * FROM invoice_lines WHERE line_id='L001'").fetchone()))
        admit_document(self.source, 'DEMO', staged['stage_id'], scope,
                       'Synthetic additional evidence', '2026-07-01', 'SIGNED', True,
                       'Synthetic reviewer', 'Admitted invented evidence.', 'seed-intake-1',
                       signed_simulation_confirmed=True)
        before = self.fingerprint(self.source)
        self.assertEqual(check_seed(self.source)['current_cases'], 20)
        self.assertEqual(self.fingerprint(self.source), before)

    def test_old_checkpoint_anchor_requires_retained_immutable_case_history(self):
        from recovery.lifecycle_storage import migrate
        migrate(self.source, backup_dir=self.root / 'backups')
        with connect(self.source) as conn, conn:
            retained = json.loads(metadata(conn, 'import_report'))
            retained['source_digest'] = 'f' * 64
            conn.execute("UPDATE metadata SET value=? WHERE key='import_report'", (json.dumps(retained),))
            conn.execute("UPDATE metadata SET value=? WHERE key='source_digest'", ('f' * 64,))
        # The old complete case is retained by the existing review event. It is
        # stale history, not authority transferred to the changed source anchor.
        self.assertEqual(check_seed(self.source)['existing_review_events'], 1)

    def test_account_env_symlink_and_existing_stage_paths_rejected_before_open(self):
        for source in (self.root / '.env', self.root / 'accounts.sqlite3',
                       self.root / 'local-auth' / 'anything.sqlite3'):
            with self.subTest(source=source.name), patch('recovery.cloud_seed.sqlite3.connect') as connect_db:
                with self.assertRaises(CloudSeedError):
                    check_seed(source)
                connect_db.assert_not_called()
        linked = self.root / 'linked.sqlite3'
        linked.symlink_to(self.source)
        with self.assertRaises(CloudSeedError):
            check_seed(linked)
        existing = self.root / 'existing-stage'
        existing.mkdir()
        marker = existing / 'must-remain'
        marker.write_text('keep')
        with self.assertRaises(CloudSeedError):
            prepare_seed(self.source, existing)
        self.assertEqual(marker.read_text(), 'keep')

    def test_cli_check_prints_only_nonsecret_summary_and_never_initializes_store(self):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error), patch.dict(os.environ, {}, clear=True):
            code = main(['--source', str(self.source), '--check'])
        self.assertEqual(code, 0)
        self.assertEqual(error.getvalue(), '')
        result = json.loads(output.getvalue())
        self.assertEqual(result['publication'], 'not_attempted')
        self.assertNotIn(str(self.root), output.getvalue())
        self.assertNotIn('reviewer', output.getvalue())

    def test_cli_refuses_missing_connection_and_never_echoes_failed_provider_details(self):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error), patch.dict(os.environ, {}, clear=True):
            code = main(['--source', str(self.source)])
        self.assertEqual(code, 2)
        self.assertEqual(output.getvalue(), '')
        self.assertIn('RECOVERY_CLOUD_DATABASE_URL', error.getvalue())
        certificate = self.root / 'public-root.crt'
        certificate.write_text('-----BEGIN CERTIFICATE-----\nsynthetic-public-certificate\n')
        with redirect_stdout(output), redirect_stderr(error), patch.dict(os.environ,
                {'RECOVERY_CLOUD_DATABASE_URL': 'do-not-print-secret'}), patch(
                'recovery.cloud_seed.publish_seed', side_effect=RuntimeError('do-not-print-secret')):
            code = main(['--source', str(self.source), '--ca-cert', str(certificate)])
        self.assertEqual(code, 2)
        self.assertNotIn('do-not-print-secret', error.getvalue())
        self.assertNotIn('committed', output.getvalue())

    def test_private_prompt_avoids_connection_secret_in_output_or_environment(self):
        certificate = self.root / 'public-root.crt'
        certificate.write_text('-----BEGIN CERTIFICATE-----\nsynthetic-public-certificate\n')
        output, error = io.StringIO(), io.StringIO()
        secret = 'postgresql://synthetic:private-prompt-test@host.invalid/postgres'
        with redirect_stdout(output), redirect_stderr(error), patch.dict(os.environ, {}, clear=True), \
             patch('getpass.getpass', return_value=secret) as prompt, \
             patch('recovery.cloud_seed.publish_seed', return_value={'publication': 'committed'}) as publish:
            code = main(['--source', str(self.source), '--ca-cert', str(certificate), '--prompt-database-url'])
            self.assertNotIn('RECOVERY_CLOUD_DATABASE_URL', os.environ)
        self.assertEqual(code, 0)
        prompt.assert_called_once()
        self.assertEqual(publish.call_args.kwargs['dsn'], secret)
        self.assertEqual(publish.call_args.kwargs['workspace_id'], 'invoice-recovery-demo')
        self.assertNotIn(secret, output.getvalue() + error.getvalue())


if __name__ == '__main__':
    unittest.main()
