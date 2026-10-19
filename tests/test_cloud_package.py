"""Source-only publication staging, without secret or workspace database reads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_cloud_repository',
    ROOT / 'scripts/prepare_cloud_repository.py')
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


class CloudPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source'
        self.output = Path(self.temp.name) / 'InvoiceRecovertTool'
        self.source.mkdir()
        for name in (*package.ROOT_FILES, package.EXAMPLE,
                     'data/manifest.json', 'data/invoice_lines.csv', 'data/retrieval_eval.jsonl',
                     'cloud/streamlit_app.py', 'cloud/requirements.txt',
                     'src/recovery/engine.py'):
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        config = self.source / package.CONFIG
        config.parent.mkdir(parents=True)
        config.write_text('[server]\naddress = "127.0.0.1"\nport = 8515\nheadless = true\n'
                          '[browser]\ngatherUsageStats = false\n'
                          '[theme]\nbase = "light"\nprimaryColor = "#0f766e"\n',
                          encoding='utf-8')

    def test_prepared_sources_match_and_only_staged_server_settings_are_removed(self):
        before = {path.relative_to(self.source).as_posix(): path.read_bytes()
                  for path in self.source.rglob('*') if path.is_file()}
        report = package.prepare_repository(self.source, self.output)
        self.assertEqual(report['synthetic'], True)
        self.assertEqual(report['dataset_id'], 'synthetic-rate-recovery-v1')
        self.assertEqual(report['file_count'], len(report['files']))
        self.assertEqual(json.loads((self.output / package.MANIFEST).read_text()), report)
        published_files = {path.relative_to(self.output).as_posix()
                           for path in self.output.rglob('*') if path.is_file()}
        self.assertEqual(published_files, {row['path'] for row in report['files']} | {package.MANIFEST})
        for row in report['files']:
            original, prepared = before[row['path']], (self.output / row['path']).read_bytes()
            self.assertEqual(row['source_sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(row['sha256'], hashlib.sha256(prepared).hexdigest())
            self.assertEqual(row['bytes'], len(prepared))
            self.assertNotIn(str(self.source), json.dumps(row))
            if row['path'] != package.CONFIG:
                self.assertEqual(prepared, original)
                self.assertFalse(row['transformed'])
        self.assertEqual((self.output / 'uv.lock').read_bytes(), (ROOT / 'uv.lock').read_bytes())
        local = tomllib.loads(before[package.CONFIG].decode())
        published = tomllib.loads((self.output / package.CONFIG).read_text())
        self.assertEqual(local['server']['address'], '127.0.0.1')
        self.assertEqual(local['server']['port'], 8515)
        self.assertEqual(published, {key: value for key, value in local.items() if key != 'server'})
        for name, content in before.items():
            self.assertEqual((self.source / name).read_bytes(), content)

    def test_retrieval_evaluation_jsonl_is_included_with_its_exact_source_bytes(self):
        report = package.prepare_repository(self.source, self.output)
        name = 'data/retrieval_eval.jsonl'
        self.assertIn(name, {row['path'] for row in report['files']})
        self.assertEqual((self.output / name).read_bytes(), (ROOT / name).read_bytes())

    def test_secret_names_hidden_files_and_workspace_data_are_never_read(self):
        excluded = [
            '.env', '.env.production', '.env.example', '.streamlit/secrets.toml',
            'cloud/secrets.toml', 'cloud/credentials.json', 'data/private-key.txt',
            'data/credentials.jsonl', 'data/api-key.jsonl',
            'data/accounts.csv', 'data/cases.sqlite3', 'src/recovery/__pycache__/saved.py',
            'docs/.private.md', 'docs/secrets/recovery.txt', 'scripts/passwords.json',
            'src/.git/config', 'var/local-auth/accounts.sqlite3',
            'var/scaled-demo/portfolio.sqlite3', 'examples/database.json',
            '.venv/package.py', 'unrelated/private.md',
        ]
        for name in excluded:
            target = self.source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('This excluded fixture must never be opened.')
        read = package._read_regular

        def guarded(path):
            path = Path(path)
            if path.is_relative_to(self.source):
                relative = path.relative_to(self.source).as_posix()
                self.assertNotIn(relative, excluded, 'Excluded source was opened')
            return read(path)

        with patch.object(package, '_read_regular', side_effect=guarded) as reads:
            report = package.prepare_repository(self.source, self.output)
        self.assertGreater(reads.call_count, 0)
        self.assertTrue(all(row['path'] not in excluded for row in report['files']))
        self.assertTrue((self.output / package.EXAMPLE).is_file())
        for name in excluded:
            self.assertFalse((self.output / name).exists())

    def test_symlinked_files_and_directories_are_not_followed_or_copied(self):
        private = Path(self.temp.name) / 'outside'
        private.mkdir()
        (private / 'not-for-publication.md').write_text('Excluded symlink target fixture')
        docs = self.source / 'docs'
        docs.mkdir()
        (docs / 'linked.md').symlink_to(private / 'not-for-publication.md')
        (docs / 'linked-directory').symlink_to(private, target_is_directory=True)
        read = package._read_regular

        def guarded(path):
            self.assertFalse(Path(path).is_symlink())
            self.assertFalse(Path(path).is_relative_to(private))
            return read(path)

        with patch.object(package, '_read_regular', side_effect=guarded):
            report = package.prepare_repository(self.source, self.output)
        self.assertFalse(any('linked' in row['path'] for row in report['files']))
        self.assertFalse((self.output / 'docs/linked.md').exists())
        self.assertFalse((self.output / 'docs/linked-directory').exists())

    def test_non_synthetic_and_invalid_fixture_manifests_fail_before_staging(self):
        manifest = json.loads((self.source / 'data/manifest.json').read_text())
        for changed in ({**manifest, 'synthetic': False},
                        {**manifest, 'tenants': {}},
                        {**manifest, 'unexpected': 'not admitted'}):
            with self.subTest(changed=changed):
                (self.source / 'data/manifest.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'synthetic-only fixture manifest'):
                    package.prepare_repository(self.source, self.output)
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.output.parent.glob('.cloud-repository-*')), [])

    def test_existing_destination_is_refused_without_modifying_it(self):
        self.output.mkdir()
        (self.output / 'existing.txt').write_text('Preserve this existing repository')
        with self.assertRaisesRegex(ValueError, 'Output already exists'):
            package.prepare_repository(self.source, self.output)
        self.assertEqual((self.output / 'existing.txt').read_text(),
                         'Preserve this existing repository')
        self.assertEqual([path.name for path in self.output.iterdir()], ['existing.txt'])

    def test_credentials_in_public_example_are_rejected_without_publishing(self):
        example = self.source / package.EXAMPLE
        example.write_text(example.read_text().replace('SET_PRIVATELY_IN_STREAMLIT_SECRETS',
                                                     'not-an-approved-placeholder'))
        with self.assertRaisesRegex(ValueError, 'only the approved placeholders'):
            package.prepare_repository(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob('.cloud-repository-*')), [])

    def test_failed_atomic_publish_removes_only_its_own_empty_reservation(self):
        original = (self.source / 'uv.lock').read_bytes()
        with patch.object(package.os, 'replace', side_effect=OSError('Simulated rename failure')):
            with self.assertRaisesRegex(OSError, 'Simulated rename failure'):
                package.prepare_repository(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob('.cloud-repository-*')), [])
        self.assertEqual((self.source / 'uv.lock').read_bytes(), original)

    def test_manifest_change_during_preparation_is_not_published(self):
        original_read = package._read_regular
        reads = 0

        def change_on_second_read(path):
            nonlocal reads
            content = original_read(path)
            if Path(path) == self.source / 'data/manifest.json':
                reads += 1
                if reads == 2:
                    content = content.replace(b'"synthetic": true', b'"synthetic": false')
            return content

        with patch.object(package, '_read_regular', side_effect=change_on_second_read):
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                package.prepare_repository(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob('.cloud-repository-*')), [])
