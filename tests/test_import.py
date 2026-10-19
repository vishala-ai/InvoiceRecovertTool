"""Importer contract and transactional reconciliation regressions, using synthetic data only."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from decimal import Decimal, localcontext
from pathlib import Path
from unittest.mock import patch

from recovery.cli import main
from recovery.engine import audit, evaluate_line
from recovery.review import record_review
from recovery.storage import TABLES, connect, load_snapshot
from recovery.storage import init_schema

DATA = Path(__file__).resolve().parents[1] / 'data'
TOTAL_FIELDS = {'invoice_lines': 'billed_amount', 'payments': 'amount', 'adjustments': 'amount'}


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name).resolve()
        self.db = self.directory / 'test.sqlite3'
        self.fixture_number = 0
        load_snapshot(self.db, DATA)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self):
        self.fixture_number += 1
        target = self.directory / f'fixture-{self.fixture_number}'
        shutil.copytree(DATA, target)
        return target

    def rows(self, directory, table):
        with (directory / f'{table}.csv').open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def write_rows(self, directory, table, fields, rows):
        with (directory / f'{table}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def edit(self, directory, table, **changes):
        fields, rows = self.rows(directory, table)
        rows[0].update(changes)
        self.write_rows(directory, table, fields, rows)

    def manifest(self, directory, change):
        path = directory / 'manifest.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        change(value)
        path.write_text(json.dumps(value), encoding='utf-8')

    def rejected(self, directory, file=None, field=None, row=None):
        with self.assertRaises(ValueError) as raised:
            load_snapshot(self.db, directory)
        report = getattr(raised.exception, 'report', None)
        self.assertIsInstance(report, dict, 'Every rejected import must carry a structured report')
        self.assertEqual(report['schema_version'], 1)
        self.assertEqual(report['status'], 'REJECTED')
        self.assertIn('reconciliation', report)
        self.assertTrue(report['errors'])
        for error in report['errors']:
            self.assertTrue({'code', 'file', 'row', 'field', 'message'} <= error.keys())
            self.assertIsInstance(error['code'], str)
            self.assertTrue(error['code'])
            self.assertIsInstance(error['message'], str)
            self.assertTrue(error['message'])
        matches = [error for error in report['errors']
                   if (file is None or error['file'] == file)
                   and (field is None or error['field'] == field)
                   and (row is None or error['row'] == row)]
        self.assertTrue(matches, f'No error at requested source location: {report["errors"]}')
        return report

    def database_state(self):
        with connect(self.db) as conn:
            return {table: sorted(tuple(row) for row in conn.execute(f'SELECT * FROM "{table}"'))
                    for table in (*TABLES, 'document_fts', 'metadata', 'review_events')}

    def add_review(self):
        with connect(self.db) as conn:
            case = evaluate_line(conn, 'DEMO', 'L001')
            record_review(conn, event_id='import-regression:1', thread_id='import-regression',
                          tenant='DEMO', line_id='L001', decision='approve',
                          reviewer='synthetic-reviewer', note='Synthetic import rollback check',
                          case_digest=case['case_digest'])

    def independent_metrics(self, directory, from_database=False):
        """Compute expected totals independently, without using importer aggregation helpers."""
        row_counts = {}
        totals = {}
        with localcontext() as context, connect(self.db) as conn:
            context.prec = 128
            for table in TABLES:
                records = ([dict(row) for row in conn.execute(f'SELECT * FROM "{table}"')]
                           if from_database else self.rows(directory, table)[1])
                row_counts[table] = len(records)
                if table not in TOTAL_FIELDS:
                    continue
                field = TOTAL_FIELDS[table]
                groups = {}
                for record in records:
                    status = record.get('status', 'ALL')
                    group = groups.setdefault(record['tenant_id'], {}).setdefault(
                        record['currency'], {}).setdefault(status, {field: Decimal('0')})
                    group[field] += Decimal(record[field])
                totals[table] = groups
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        return {'row_counts': row_counts, 'totals': totals, 'as_of': manifest['as_of'],
                'extracted_at': manifest.get('extracted_at')}

    def assert_metrics(self, actual, expected):
        self.assertEqual(actual['row_counts'], expected['row_counts'])
        self.assertEqual(actual['as_of'], expected['as_of'])
        self.assertEqual(actual['extracted_at'], expected['extracted_at'])
        parsed = json.loads(json.dumps(actual['totals']), parse_float=Decimal)
        for tenants in parsed.values():
            for currencies in tenants.values():
                for statuses in currencies.values():
                    for amounts in statuses.values():
                        for field, amount in amounts.items():
                            self.assertIsInstance(amount, str)
                            amounts[field] = Decimal(amount)
        self.assertEqual(parsed, expected['totals'])

    def test_report_reconciles_source_and_database_without_changing_audit(self):
        with connect(self.db) as conn:
            before = audit(conn)
        report = load_snapshot(self.db, DATA)
        self.assertEqual(report['schema_version'], 1)
        self.assertEqual(report['status'], 'IMPORTED')
        self.assertEqual(report['lines'], 20)
        self.assertEqual(report['as_of'], '2026-10-15')
        self.assertIsNone(report['extracted_at'])
        self.assertTrue(report['reconciliation']['matched'])
        self.assert_metrics(report['reconciliation']['source'], self.independent_metrics(DATA))
        self.assert_metrics(report['reconciliation']['imported'],
                            self.independent_metrics(DATA, from_database=True))
        with connect(self.db) as conn:
            after = audit(conn)
        self.assertEqual(after, before)
        self.assertEqual(after['candidate_total'], '80707.50')
        self.assertEqual(after['realized_recovery'], '0.00')

    def test_reconciliation_preserves_fractional_values_and_extraction_timestamp(self):
        directory = self.fixture()
        fields, rows = self.rows(directory, 'payments')
        rows[0]['amount'] = '0.123456'
        rows[1]['amount'] = '0.000001'
        self.write_rows(directory, 'payments', fields, rows)
        self.manifest(directory, lambda value: value.update(extracted_at='2026-10-15T12:30:00Z'))
        report = load_snapshot(self.db, directory)
        self.assertEqual(report['extracted_at'], '2026-10-15T12:30:00Z')
        self.assertTrue(report['reconciliation']['matched'])
        self.assert_metrics(report['reconciliation']['source'], self.independent_metrics(directory))
        self.assert_metrics(report['reconciliation']['imported'],
                            self.independent_metrics(directory, from_database=True))
        total = report['reconciliation']['source']['totals']['payments']['DEMO']['USD']['SETTLED']
        amount = Decimal(total['amount'])
        self.assertNotEqual(amount, amount.quantize(Decimal('0.01')))

    def test_missing_extra_and_duplicate_headers_are_rejected(self):
        for problem in ('missing', 'extra', 'duplicate'):
            with self.subTest(problem=problem):
                directory = self.fixture()
                path = directory / 'payments.csv'
                with path.open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.reader(stream))
                if problem == 'missing':
                    rows = [row[:-1] for row in rows]
                else:
                    rows[0].append('unexpected' if problem == 'extra' else rows[0][0])
                    for row in rows[1:]:
                        row.append('extra-value')
                with path.open('w', newline='', encoding='utf-8') as stream:
                    csv.writer(stream).writerows(rows)
                self.rejected(directory, 'payments.csv', row=1)

    def test_short_long_and_invalid_quoted_csv_rows_are_rejected(self):
        for problem in ('short', 'long', 'quote'):
            with self.subTest(problem=problem):
                directory = self.fixture()
                path = directory / 'payments.csv'
                with path.open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.reader(stream))
                if problem == 'quote':
                    path.write_text(','.join(rows[0]) + '\n"unterminated', encoding='utf-8')
                else:
                    rows[1] = rows[1][:-1] if problem == 'short' else rows[1] + ['extra-value']
                    with path.open('w', newline='', encoding='utf-8') as stream:
                        csv.writer(stream).writerows(rows)
                self.rejected(directory, 'payments.csv', row=2)

    def test_invalid_financial_strings_are_rejected_before_database_write(self):
        before = self.database_state()
        invalid = ('', 'NaN', 'Infinity', '-1', 'twelve',
                   '12345678901234567890123456789.123456789')
        fields = (('invoice_lines', 'billed_hours'), ('invoice_lines', 'billed_rate'),
                  ('invoice_lines', 'billed_amount'), ('service_entries', 'approved_hours'),
                  ('payments', 'amount'), ('adjustments', 'amount'),
                  ('reviewed_terms', 'rate'), ('purchase_orders', 'po_rate'))
        for table, field in fields:
            for value in invalid:
                with self.subTest(table=table, field=field, value=value):
                    directory = self.fixture()
                    self.edit(directory, table, **{field: value})
                    self.rejected(directory, f'{table}.csv', field, row=2)
        self.assertEqual(self.database_state(), before)

    def test_invalid_dates_booleans_enums_and_line_numbers_are_rejected(self):
        cases = (('invoice_lines', 'invoice_date', '2026-02-30'),
                 ('service_entries', 'service_date', '10/15/2026'),
                 ('reviewed_terms', 'effective_from', '2026-13-01'),
                 ('reviewed_terms', 'effective_to', 'not-a-date'),
                 ('documents', 'signed_date', '2026-02-30'),
                 ('payments', 'settled_date', '2026-02-30'),
                 ('adjustments', 'effective_date', '2026-02-30'),
                 ('service_entries', 'approved', 'yes'),
                 ('reviewed_terms', 'reviewed', 'TRUE'),
                 ('payments', 'status', 'UNKNOWN_STATUS'),
                 ('adjustments', 'kind', 'UNKNOWN_KIND'),
                 ('invoice_lines', 'currency', 'usd'),
                 ('invoice_lines', 'line_number', '0'),
                 ('invoice_lines', 'line_number', '1.5'),
                 ('suppliers', 'supplier_id', ''))
        for table, field, value in cases:
            with self.subTest(table=table, field=field, value=value):
                directory = self.fixture()
                self.edit(directory, table, **{field: value})
                self.rejected(directory, f'{table}.csv', field, row=2)

    def test_invalid_manifest_shapes_values_and_json_are_reported(self):
        cases = (lambda value: value.pop('as_of'),
                 lambda value: value.update(as_of='2026-02-30'),
                 lambda value: value.update(extracted_at='not-a-timestamp'),
                 lambda value: value.update(unknown_field=True),
                 lambda value: value.update(tenants=[]),
                 lambda value: value['tenants']['DEMO'].update(payments_complete='true'))
        for change in cases:
            with self.subTest(change=change):
                directory = self.fixture()
                self.manifest(directory, change)
                self.rejected(directory, 'manifest.json')
        for content in ('{"synthetic": true,', '[]'):
            with self.subTest(content=content):
                directory = self.fixture()
                (directory / 'manifest.json').write_text(content, encoding='utf-8')
                self.rejected(directory, 'manifest.json')

    def test_synthetic_guard_runs_before_any_source_file_is_opened(self):
        original_open = Path.open
        before = self.database_state()
        for synthetic in (False, 'true', 1, None):
            with self.subTest(synthetic=synthetic):
                directory = self.fixture()
                self.manifest(directory, lambda value: value.update(synthetic=synthetic))

                def guarded_open(path, *args, **kwargs):
                    if path.resolve().is_relative_to(directory) and path.name != 'manifest.json':
                        self.fail('A source was opened before rejecting the synthetic attestation')
                    return original_open(path, *args, **kwargs)

                with patch.object(Path, 'open', guarded_open):
                    report = self.rejected(directory, 'manifest.json', 'synthetic')
                self.assertEqual(len(report['errors']), 1)
        self.assertEqual(self.database_state(), before)

    def test_missing_required_source_and_document_files_are_reported(self):
        for filename in ('manifest.json', *(f'{table}.csv' for table in TABLES),
                         'contracts/DOC-A1.md'):
            with self.subTest(filename=filename):
                directory = self.fixture()
                (directory / filename).unlink()
                self.rejected(directory)

    def test_unsafe_paths_symlinks_and_document_hash_mismatch_are_rejected(self):
        directory = self.fixture()
        outside = self.directory / 'synthetic-sentinel.md'
        outside.write_text('Synthetic sentinel: must not be read by importer.', encoding='utf-8')
        self.edit(directory, 'documents', path='../synthetic-sentinel.md')
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path.resolve() == outside:
                self.fail('Importer opened a document outside its declared source directory')
            return original_open(path, *args, **kwargs)

        with patch.object(Path, 'open', guarded_open):
            self.rejected(directory, 'documents.csv', 'path', row=2)
        directory = self.fixture()
        with (directory / 'contracts/DOC-A1.md').open('a', encoding='utf-8') as stream:
            stream.write('\nSynthetic integrity corruption.')
        self.rejected(directory, 'documents.csv', 'sha256', row=2)
        for filename in ('manifest.json', 'payments.csv', 'contracts/DOC-A1.md'):
            with self.subTest(symlink=filename):
                directory = self.fixture()
                source = directory / filename
                outside = self.directory / f'synthetic-sentinel-{self.fixture_number}{source.suffix}'
                outside.write_bytes(source.read_bytes())
                source.unlink()
                source.symlink_to(outside)

                def guarded_open(path, *args, **kwargs):
                    if path.resolve() == outside:
                        self.fail('Importer opened a source symlink outside its source directory')
                    return original_open(path, *args, **kwargs)

                with patch.object(Path, 'open', guarded_open):
                    self.rejected(directory)

    def test_orphan_and_cross_tenant_source_references_are_rejected(self):
        cases = (('invoice_lines', 'supplier_id', 'MISSING'),
                 ('documents', 'supplier_id', 'MISSING'),
                 ('purchase_orders', 'supplier_id', 'MISSING'),
                 ('service_entries', 'line_id', 'MISSING'),
                 ('payments', 'line_id', 'MISSING'),
                 ('adjustments', 'line_id', 'MISSING'),
                 ('reviewed_terms', 'doc_id', 'MISSING'),
                 ('reviewed_terms', 'supersedes', 'MISSING'),
                 ('payments', 'tenant_id', 'OTHER'))
        for table, field, value in cases:
            with self.subTest(table=table, field=field):
                directory = self.fixture()
                self.edit(directory, table, **{field: value})
                self.rejected(directory, f'{table}.csv', row=2)

    def test_term_document_scope_and_exact_quote_must_match(self):
        for field, value in (('contract_id', 'MISSING'), ('source_quote', 'Invented exact quote')):
            with self.subTest(field=field):
                directory = self.fixture()
                self.edit(directory, 'reviewed_terms', **{field: value})
                self.rejected(directory, 'reviewed_terms.csv', row=2)

    def test_duplicate_source_ids_in_every_table_are_rejected(self):
        for table in TABLES:
            with self.subTest(table=table):
                directory = self.fixture()
                fields, rows = self.rows(directory, table)
                rows.append(dict(rows[0]))
                self.write_rows(directory, table, fields, rows)
                self.rejected(directory, f'{table}.csv', row=len(rows) + 1)

    def test_duplicate_invoice_business_key_with_different_line_id_is_rejected(self):
        directory = self.fixture()
        fields, rows = self.rows(directory, 'invoice_lines')
        duplicate = dict(rows[0], line_id='L999')
        rows.append(duplicate)
        self.write_rows(directory, 'invoice_lines', fields, rows)
        self.rejected(directory, 'invoice_lines.csv', row=len(rows) + 1)

    def test_rejection_report_collects_independent_errors_with_source_locations(self):
        directory = self.fixture()
        self.edit(directory, 'payments', amount='NaN')
        self.edit(directory, 'service_entries', approved='yes')
        self.edit(directory, 'invoice_lines', invoice_date='2026-02-30')
        report = self.rejected(directory)
        locations = {(error['file'], error['row'], error['field']) for error in report['errors']}
        self.assertTrue({('payments.csv', 2, 'amount'), ('service_entries.csv', 2, 'approved'),
                         ('invoice_lines.csv', 2, 'invoice_date')} <= locations)

    def test_preflight_rejection_preserves_all_sources_fts_metadata_and_review_journal(self):
        self.add_review()
        before = self.database_state()
        directory = self.fixture()
        self.edit(directory, 'invoice_lines', description='A replacement snapshot must not commit')
        self.edit(directory, 'adjustments', amount='NaN')
        self.rejected(directory)
        self.assertEqual(self.database_state(), before)

    def test_reconciliation_detects_silently_dropped_or_altered_rows_and_rolls_back(self):
        self.add_review()
        before = self.database_state()
        triggers = {
            'drop': '''CREATE TRIGGER synthetic_import_change BEFORE INSERT ON payments
                WHEN NEW.payment_id='PAY-001' BEGIN SELECT RAISE(IGNORE); END''',
            'alter': '''CREATE TRIGGER synthetic_import_change AFTER INSERT ON invoice_lines
                WHEN NEW.line_id='L001' BEGIN UPDATE invoice_lines SET
                description='Synthetic silent alteration' WHERE line_id='L001'; END''',
            'late_alter': '''CREATE TRIGGER synthetic_import_change AFTER INSERT ON metadata
                WHEN NEW.key='import_report' BEGIN UPDATE invoice_lines SET
                description='Synthetic alteration after report write' WHERE line_id='L001'; END''',
        }
        for problem, trigger in triggers.items():
            with self.subTest(problem=problem):
                with connect(self.db) as conn, conn:
                    conn.execute(trigger)
                directory = self.fixture()
                self.edit(directory, 'invoice_lines', description='Synthetic replacement snapshot')
                report = self.rejected(directory)
                reconciliation = report['reconciliation']
                self.assertFalse(reconciliation['matched'])
                self.assertNotEqual(reconciliation['source'], reconciliation['imported'])
                if problem in ('alter', 'late_alter'):
                    for field in ('row_counts', 'totals'):
                        self.assertEqual(reconciliation['source'][field],
                                         reconciliation['imported'][field])
                self.assertEqual(self.database_state(), before)
                with connect(self.db) as conn, conn:
                    conn.execute('DROP TRIGGER synthetic_import_change')

    def test_database_write_failure_is_reported_and_preserves_existing_snapshot(self):
        self.add_review()
        before = self.database_state()
        triggers = {
            'abort': '''CREATE TRIGGER synthetic_import_failure BEFORE INSERT ON payments
                BEGIN SELECT RAISE(ABORT, 'Synthetic injected write failure'); END''',
            'invalid_amount': '''CREATE TRIGGER synthetic_import_failure AFTER INSERT ON payments
                BEGIN UPDATE payments SET amount='not-number' WHERE payment_id=NEW.payment_id; END''',
            'missing_manifest_fields': '''CREATE TRIGGER synthetic_import_failure
                AFTER INSERT ON metadata WHEN NEW.key='manifest'
                BEGIN UPDATE metadata SET value='{}' WHERE key='manifest'; END''',
            'invalid_manifest_json': '''CREATE TRIGGER synthetic_import_failure
                AFTER INSERT ON metadata WHEN NEW.key='manifest'
                BEGIN UPDATE metadata SET value='invalid JSON' WHERE key='manifest'; END''',
            'boolean_coerced_to_integer': '''CREATE TRIGGER synthetic_import_failure
                AFTER INSERT ON metadata WHEN NEW.key='manifest'
                BEGIN UPDATE metadata SET value=json_set(value,
                '$.tenants.DEMO.payments_complete', 1) WHERE key='manifest'; END''',
            'missing_source_digest': '''CREATE TRIGGER synthetic_import_failure
                AFTER INSERT ON metadata WHEN NEW.key='source_digest'
                BEGIN DELETE FROM metadata WHERE key='source_digest'; END''',
        }
        for problem, trigger in triggers.items():
            with self.subTest(problem=problem):
                with connect(self.db) as conn, conn:
                    conn.execute(trigger)
                self.rejected(self.fixture())
                self.assertEqual(self.database_state(), before)
                with connect(self.db) as conn, conn:
                    conn.execute('DROP TRIGGER synthetic_import_failure')

    def test_failed_first_import_never_leaves_partial_schema(self):
        directory = self.fixture()
        self.edit(directory, 'payments', amount='NaN')
        fresh = self.directory / 'fresh.sqlite3'
        with self.assertRaises(ValueError) as rejected:
            load_snapshot(fresh, directory)
        self.assertEqual(rejected.exception.report['status'], 'REJECTED')
        self.assertFalse(fresh.exists())

        def schema_then_fail(conn):
            init_schema(conn)
            raise sqlite3.OperationalError('Synthetic schema creation failure')

        with patch('recovery.storage.init_schema', side_effect=schema_then_fail):
            with self.assertRaises(ValueError) as rejected:
                load_snapshot(fresh, DATA)
        self.assertEqual(rejected.exception.report['status'], 'REJECTED')
        with connect(fresh) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
                             .fetchone()[0], 0)

    def test_cli_writes_success_report_as_json(self):
        target = self.directory / 'reports' / 'success.json'
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(['--db', str(self.db), 'init', '--data', str(DATA), '--report', str(target)])
        self.assertEqual(code, 0)
        report = json.loads(target.read_text(encoding='utf-8'))
        self.assertEqual(report, json.loads(stdout.getvalue()))
        self.assertEqual(report['status'], 'IMPORTED')
        self.assertTrue(report['reconciliation']['matched'])

    def test_cli_rejection_is_machine_readable_and_leaves_database_unchanged(self):
        directory = self.fixture()
        self.edit(directory, 'payments', amount='NaN')
        target = self.directory / 'reports' / 'rejected.json'
        before = self.database_state()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main(['--db', str(self.db), 'init', '--data', str(directory), '--report', str(target)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stdout.getvalue(), '')
        report = json.loads(target.read_text(encoding='utf-8'))
        self.assertEqual(report, json.loads(stderr.getvalue()))
        self.assertEqual(report['status'], 'REJECTED')
        self.assertEqual(self.database_state(), before)

    def test_cli_report_cannot_overwrite_database_sources_or_a_directory(self):
        for target_name in ('database', 'directory', 'manifest.json', 'payments.csv',
                            'contracts/DOC-A1.md', 'hardlink_database', 'hardlink_source'):
            with self.subTest(target=target_name):
                directory = self.fixture()
                if target_name.startswith('hardlink_'):
                    target = self.directory / f'report-hardlink-{self.fixture_number}.json'
                    linked = self.db if target_name == 'hardlink_database' else directory / 'payments.csv'
                    os.link(linked, target)
                else:
                    target = (self.db if target_name == 'database' else directory if
                              target_name == 'directory' else directory / target_name)
                before = self.database_state()
                content = target.read_bytes() if target.is_file() else None
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main(['--db', str(self.db), 'init', '--data', str(directory),
                              '--report', str(target)])
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(stdout.getvalue(), '')
                report = json.loads(stderr.getvalue())
                self.assertEqual(report['status'], 'REJECTED')
                self.assertEqual(report['errors'][0]['code'], 'REPORT_PATH_ERROR')
                self.assertEqual(self.database_state(), before)
                if content is not None:
                    self.assertEqual(target.read_bytes(), content)
                if target_name.startswith('hardlink_'):
                    target.unlink()

    def test_cli_late_report_write_failure_preserves_actual_import_outcome(self):
        original_write = Path.write_text
        for outcome in ('IMPORTED', 'REJECTED'):
            with self.subTest(outcome=outcome):
                directory = self.fixture()
                self.edit(directory, 'invoice_lines', description='Synthetic replacement snapshot')
                if outcome == 'REJECTED':
                    self.edit(directory, 'payments', amount='NaN')
                before = self.database_state()
                target = self.directory / 'reports' / f'late-failure-{outcome}.json'

                def failing_write(path, *args, **kwargs):
                    if path.resolve() == target:
                        raise OSError('Synthetic injected report write failure')
                    return original_write(path, *args, **kwargs)

                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(Path, 'write_text', failing_write), \
                        contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main(['--db', str(self.db), 'init', '--data', str(directory),
                              '--report', str(target)])
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(stdout.getvalue(), '')
                self.assertFalse(target.exists())
                report = json.loads(stderr.getvalue())
                self.assertEqual(report['status'], outcome)
                self.assertEqual(report['report_errors'][0]['code'], 'REPORT_WRITE_ERROR')
                if outcome == 'REJECTED':
                    self.assertTrue(any(error['file'] == 'payments.csv' and
                                        error['field'] == 'amount' for error in report['errors']))
                    self.assertEqual(self.database_state(), before)
                else:
                    self.assertEqual(report['errors'], [])
                    self.assertNotEqual(self.database_state(), before)
                    with connect(self.db) as conn:
                        result = audit(conn)
                    self.assertEqual(result['candidate_total'], '80707.50')
                    self.assertEqual(result['realized_recovery'], '0.00')


if __name__ == '__main__':
    unittest.main()
