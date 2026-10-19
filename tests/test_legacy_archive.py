"""Synthetic migration history is immutable and excluded from current finance."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
from decimal import Decimal, localcontext
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from recovery.cli import main
from recovery.engine import audit, evaluate_line
from recovery.legacy_archive import (STATUS, archive_summary, assert_not_archived,
    count_archived_cases, get_archived_case, import_archive, list_archived_cases,
    list_archived_contracts, list_archived_suppliers)
from recovery.review import record_review
from recovery.storage import TABLES, connect, load_snapshot, metadata

DATA = Path(__file__).resolve().parents[1] / 'data'


def payload(*, vendors=2, cases_per_year=2):
    result = {'schema_version': 1, 'synthetic': True, 'archive_id': 'LEGACY-2024-2025-v1',
        'source_system': 'Legacy AP (synthetic)',
        'provenance': 'Migrated from old system — locked (synthetic)',
        'suppliers': [], 'contracts': [], 'cases': []}
    for number in range(1, vendors + 1):
        supplier = f'LEGACY-SUP-{number:03d}'
        result['suppliers'].append({'tenant_id': 'DEMO', 'supplier_id': supplier,
                                    'name': f'Synthetic archived vendor {number:03d}'})
        for year in (2024, 2025):
            result['contracts'].append({'tenant_id': 'DEMO', 'supplier_id': supplier,
                'contract_id': f'LEGACY-CTR-{number:03d}-{year}', 'origin_year': year,
                'title': f'Synthetic prior-year services {year} vendor {number:03d}'})
    for year in (2024, 2025):
        for index in range(1, cases_per_year + 1):
            number = (index - 1) % vendors + 1
            result['cases'].append({'tenant_id': 'DEMO', 'line_id': f'LEGACY-{year}-{index:04d}',
                'supplier_id': f'LEGACY-SUP-{number:03d}',
                'contract_id': f'LEGACY-CTR-{number:03d}-{year}',
                'invoice_id': f'LEGACY-INV-{year}-{index:04d}', 'invoice_date': f'{year}-06-30',
                'origin_year': year, 'currency': 'USD', 'historical_approved_amount': '123.45',
                'source_record_id': f'LEGACY-AP-{year}-{index:04d}',
                'description': 'Invented historical hourly invoice; migrated approval assumed.'})
    return result


class LegacyArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.db = self.root / 'synthetic.sqlite3'
        load_snapshot(self.db, DATA)

    def source_state(self):
        with connect(self.db) as conn:
            return {name: sorted(tuple(row) for row in conn.execute(f'SELECT * FROM {name}'))
                    for name in (*TABLES, 'metadata', 'document_fts', 'review_events')}

    def archive_exists(self):
        with connect(self.db) as conn:
            return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE name='legacy_cases'").fetchone())

    def test_reads_do_not_initialize_missing_databases_or_schema(self):
        before = self.source_state()
        for db in (self.db, self.root / 'missing.sqlite3'):
            self.assertEqual(list_archived_cases(db), [])
            self.assertEqual(list_archived_suppliers(db), [])
            self.assertEqual(list_archived_contracts(db), [])
            self.assertEqual(count_archived_cases(db), 0)
            self.assertIsNone(get_archived_case(db, 'DEMO', 'LEGACY-absent'))
            self.assertEqual(archive_summary(db)['case_count'], 0)
        self.assertFalse((self.root / 'missing.sqlite3').exists())
        self.assertFalse(self.archive_exists())
        self.assertEqual(self.source_state(), before)
        with connect(self.db) as conn:
            assert_not_archived(conn, 'DEMO', 'L001')

    def test_import_preserves_current_source_financial_results_and_original_payload(self):
        original = payload()
        saved = copy.deepcopy(original)
        before = self.source_state()
        receipt = import_archive(self.db, original)
        self.assertEqual(original, saved)
        self.assertEqual(receipt['row_counts'], {'suppliers': 2, 'contracts': 4, 'cases': 4})
        self.assertEqual(receipt['counts_by_year'], {'2024': 2, '2025': 2})
        self.assertFalse(receipt['already_imported'])
        rows = list_archived_cases(self.db)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row['status'] for row in rows}, {STATUS})
        self.assertTrue(all(row['locked'] is True and row['synthetic'] is True for row in rows))
        self.assertEqual({row['provenance'] for row in rows}, {original['provenance']})
        self.assertTrue(all(len(row['case_sha256']) == 64 for row in rows))
        self.assertTrue(all(isinstance(row['historical_approved_amount'], str) for row in rows))
        self.assertEqual(self.source_state(), before)
        with connect(self.db) as conn:
            report = audit(conn)
        self.assertEqual(report['case_count'], 20)
        self.assertEqual(report['candidate_total'], '80707.50')
        self.assertEqual(report['realized_recovery'], '0.00')
        summary = archive_summary(self.db)
        self.assertEqual(summary, {'case_count': 4, 'supplier_count': 2, 'contract_count': 4,
            'counts_by_year': {'2024': 2, '2025': 2}, 'totals_by_currency': {'USD': '493.80'}})

    def test_reimport_is_order_independent_and_cannot_replace_an_archive(self):
        first = payload()
        original = import_archive(self.db, first)
        with connect(self.db) as conn:
            before = list(conn.iterdump())
        reordered = copy.deepcopy(first)
        for group in ('suppliers', 'contracts', 'cases'):
            reordered[group].reverse()
        again = import_archive(self.db, reordered)
        self.assertTrue(again['already_imported'])
        self.assertEqual(again['payload_sha256'], original['payload_sha256'])
        changed = copy.deepcopy(first)
        changed['cases'][0]['historical_approved_amount'] = '999.00'
        with self.assertRaisesRegex(ValueError, 'immutable'):
            import_archive(self.db, changed)
        with connect(self.db) as conn:
            self.assertEqual(list(conn.iterdump()), before)

    def test_all_archive_tables_reject_update_delete_and_replace(self):
        import_archive(self.db, payload())
        with connect(self.db) as conn:
            before = list(conn.iterdump())
            for table in ('legacy_archive_batches', 'legacy_suppliers', 'legacy_contracts', 'legacy_cases'):
                first = conn.execute(f'SELECT * FROM {table} LIMIT 1').fetchone()
                column = first.keys()[0]
                for sql, args in ((f'UPDATE {table} SET {column}={column}', ()),
                                  (f'DELETE FROM {table}', ()),
                                  (f'INSERT OR REPLACE INTO {table} VALUES '
                                   f'({",".join("?" for _ in first)})', tuple(first))):
                    with self.subTest(table=table, sql=sql):
                        with self.assertRaisesRegex(sqlite3.IntegrityError, 'READ_ONLY'):
                            with conn:
                                conn.execute(sql, args)
            self.assertEqual(list(conn.iterdump()), before)

    def test_source_record_replace_cannot_delete_an_existing_case_via_unique_constraint(self):
        import_archive(self.db, payload())
        with connect(self.db) as conn:
            row = dict(conn.execute('SELECT * FROM legacy_cases LIMIT 1').fetchone())
            original = dict(row)
            row['line_id'] = 'LEGACY-different-line'
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'READ_ONLY'):
                with conn:
                    conn.execute(f'INSERT OR REPLACE INTO legacy_cases ({",".join(row)}) '
                        f'VALUES ({",".join("?" for _ in row)})', tuple(row.values()))
            actual = dict(conn.execute('SELECT * FROM legacy_cases WHERE line_id=?',
                                       (original['line_id'],)).fetchone())
            self.assertEqual(actual, original)

    def test_decimal_schema_rejects_coercion_invalid_values_and_extreme_amounts(self):
        for bad in (None, True, 123.45, Decimal('123.45'), 'NaN', 'Infinity', '-1.00',
                    '1e2', '1', '1.0', '1.000', '01.00', '1000000000000000.00'):
            data = payload()
            data['cases'][0]['historical_approved_amount'] = bad
            with self.subTest(value=repr(bad)), self.assertRaisesRegex(ValueError, 'Decimal'):
                import_archive(self.db, data)
        self.assertFalse(self.archive_exists())

    def test_totals_preserve_large_decimal_strings_independent_of_global_precision(self):
        data = payload()
        data['cases'][0]['historical_approved_amount'] = '123456789012345.67'
        data['cases'][1]['currency'] = 'EUR'
        data['cases'][1]['historical_approved_amount'] = '0.01'
        import_archive(self.db, data)
        with localcontext() as context:
            context.prec = 5
            totals = archive_summary(self.db)['totals_by_currency']
        self.assertEqual(totals, {'EUR': '0.01', 'USD': '123456789012592.57'})
        self.assertEqual(get_archived_case(self.db, 'DEMO', data['cases'][0]['line_id'])[
            'historical_approved_amount'], '123456789012345.67')

    def test_schema_identity_reference_and_year_validation_fail_atomically(self):
        changes = [lambda value: value.update(unexpected=True),
            lambda value: value.update(schema_version=True),
            lambda value: value['suppliers'][0].update(supplier_id='SUP-A'),
            lambda value: value['cases'][0].update(status='APPROVED_FOR_DRAFT'),
            lambda value: value['cases'][0].update(origin_year=2026),
            lambda value: value['cases'][0].update(origin_year=True),
            lambda value: value['cases'][0].update(invoice_date='2025-06-30'),
            lambda value: value['cases'][0].update(contract_id='LEGACY-unknown'),
            lambda value: value['cases'][0].update(supplier_id='LEGACY-SUP-002'),
            lambda value: value['cases'][0].update(tenant_id='UNKNOWN'),
            lambda value: value['cases'].append(dict(value['cases'][0])),
            lambda value: value['cases'][1].update(source_record_id=value['cases'][0]['source_record_id'])]
        before = self.source_state()
        for change in changes:
            data = payload()
            change(data)
            with self.subTest(change=changes.index(change)), self.assertRaises(ValueError):
                import_archive(self.db, data)
            self.assertFalse(self.archive_exists())
        self.assertEqual(self.source_state(), before)

    def test_synthetic_guards_reject_payload_and_database_without_creating_schema(self):
        data = payload()
        data['synthetic'] = False
        with self.assertRaisesRegex(ValueError, 'SYNTHETIC_ONLY'):
            import_archive(self.db, data)
        with connect(self.db) as conn, conn:
            manifest = json.loads(metadata(conn, 'manifest'))
            manifest['synthetic'] = False
            conn.execute('UPDATE metadata SET value=? WHERE key=?', (json.dumps(manifest), 'manifest'))
        with self.assertRaisesRegex(ValueError, 'SYNTHETIC_ONLY'):
            import_archive(self.db, payload())
        self.assertFalse(self.archive_exists())
        missing = self.root / 'missing.sqlite3'
        with self.assertRaisesRegex(ValueError, 'existing synthetic'):
            import_archive(missing, payload())
        self.assertFalse(missing.exists())

    def test_existing_and_future_canonical_id_collisions_are_rejected(self):
        data = payload()
        with connect(self.db) as conn, conn:
            conn.execute('INSERT INTO suppliers VALUES (?,?,?)',
                ('DEMO', data['suppliers'][0]['supplier_id'], 'Invented collision vendor'))
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            import_archive(self.db, data)
        self.assertFalse(self.archive_exists())
        load_snapshot(self.db, DATA)
        import_archive(self.db, data)
        with connect(self.db) as conn:
            canonical = dict(conn.execute("SELECT * FROM invoice_lines WHERE tenant_id='DEMO' LIMIT 1").fetchone())
            canonical['line_id'] = data['cases'][0]['line_id']
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'READ_ONLY'):
                with conn:
                    conn.execute(f'INSERT INTO invoice_lines ({",".join(canonical)}) '
                        f'VALUES ({",".join("?" for _ in canonical)})', tuple(canonical.values()))
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'READ_ONLY'):
                with conn:
                    conn.execute('INSERT INTO suppliers VALUES (?,?,?)',
                        ('DEMO', data['suppliers'][0]['supplier_id'], 'Invented collision vendor'))
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)

    def test_failed_second_batch_rolls_back_every_new_record(self):
        first = payload()
        import_archive(self.db, first)
        second = payload()
        second['archive_id'] = 'LEGACY-other-batch'
        with connect(self.db) as conn:
            before = list(conn.iterdump())
        with self.assertRaises(sqlite3.IntegrityError):
            import_archive(self.db, second)
        with connect(self.db) as conn:
            self.assertEqual(list(conn.iterdump()), before)

    def test_contract_and_invoice_collisions_are_rejected_before_archive_creation(self):
        data = payload()
        for table, field, archived_value in (
                ('documents', 'contract_id', data['contracts'][0]['contract_id']),
                ('invoice_lines', 'line_id', data['cases'][0]['line_id'])):
            with self.subTest(table=table):
                load_snapshot(self.db, DATA)
                with connect(self.db) as conn, conn:
                    row = dict(conn.execute(f"SELECT * FROM {table} WHERE tenant_id='DEMO' LIMIT 1").fetchone())
                    if table == 'documents':
                        row['doc_id'] = 'LEGACY-collision-document'
                    row[field] = archived_value
                    conn.execute(f'INSERT INTO {table} ({",".join(row)}) '
                        f'VALUES ({",".join("?" for _ in row)})', tuple(row.values()))
                with self.assertRaisesRegex(ValueError, 'overlaps'):
                    import_archive(self.db, data)
                self.assertFalse(self.archive_exists())

    def test_foreign_keys_reject_cross_contract_archive_rows(self):
        import_archive(self.db, payload())
        with connect(self.db) as conn:
            row = dict(conn.execute('SELECT * FROM legacy_cases LIMIT 1').fetchone())
            row.update(line_id='LEGACY-new-line', source_record_id='LEGACY-new-source',
                       contract_id='LEGACY-missing-contract')
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'FOREIGN KEY'):
                with conn:
                    conn.execute(f'INSERT INTO legacy_cases ({",".join(row)}) '
                        f'VALUES ({",".join("?" for _ in row)})', tuple(row.values()))
            self.assertEqual(conn.execute('SELECT count(*) FROM legacy_cases').fetchone()[0], 4)

    def test_sql_filters_pagination_and_tenant_scope_do_not_calculate_cases(self):
        data = payload()
        data['cases'][0]['description'] = 'Invented 100% underscore_ literal'
        # Same IDs in a different demo tenant remain a distinct archive scope.
        for group in ('suppliers', 'contracts', 'cases'):
            data[group] += [{**row, 'tenant_id': 'OTHER'} for row in data[group]]
        import_archive(self.db, data)
        with patch('recovery.engine.audit', side_effect=AssertionError('Archive must not audit current cases')):
            self.assertEqual(count_archived_cases(self.db), 4)
            self.assertEqual(count_archived_cases(self.db, origin_year=2024), 2)
            self.assertEqual(count_archived_cases(self.db, search='%'), 1)
            self.assertEqual(count_archived_cases(self.db, search='_'), 1)
            self.assertEqual(count_archived_cases(self.db, search="' OR 1=1 --"), 0)
            self.assertEqual(count_archived_cases(self.db, search='vendor 001'), 2)
            first = list_archived_cases(self.db, limit=1)
            second = list_archived_cases(self.db, limit=1, offset=1)
            self.assertNotEqual(first[0]['line_id'], second[0]['line_id'])
            self.assertEqual(first[0]['origin_year'], 2025)
            vendors = list_archived_suppliers(self.db, origin_year=2024)
            self.assertEqual(len(vendors), 2)
            contracts = list_archived_contracts(self.db, 'DEMO', vendors[0]['supplier_id'], origin_year=2024)
            self.assertEqual(len(contracts), 1)
            cases = list_archived_cases(self.db, origin_year=2024,
                supplier_id=vendors[0]['supplier_id'], contract_id=contracts[0]['contract_id'])
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]['tenant_id'], 'DEMO')
            self.assertEqual(archive_summary(self.db, 'OTHER')['case_count'], 4)
            self.assertIsNone(get_archived_case(self.db, 'MISSING', cases[0]['line_id']))
        for options in ({'limit': 0}, {'limit': True}, {'limit': 1001}, {'offset': -1},
                        {'offset': False}, {'origin_year': '2024'}, {'search': None}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                list_archived_cases(self.db, **options)

    def test_archived_ids_cannot_be_reassessed_or_approved_through_existing_services(self):
        from recovery.lifecycle import migrate, reassess

        migrate(self.db, backup_dir=self.root / 'backups')
        import_archive(self.db, payload())
        case = list_archived_cases(self.db)[0]
        with connect(self.db) as conn:
            with self.assertRaisesRegex(ValueError, 'READ_ONLY'):
                assert_not_archived(conn, 'DEMO', case['line_id'])
            assert_not_archived(conn, 'OTHER', case['line_id'])
            with self.assertRaises(ValueError):
                evaluate_line(conn, 'DEMO', case['line_id'])
            with self.assertRaises(ValueError):
                record_review(conn, event_id='legacy-not-an-approval', thread_id='legacy-reuse',
                    tenant='DEMO', line_id=case['line_id'], decision='approve', reviewer='Synthetic test',
                    note='An archived record cannot become a new approval.', case_digest=case['case_sha256'])
        with self.assertRaises(ValueError):
            reassess(self.db, 'DEMO', case['line_id'], idempotency_key='cannot-reassess-archive')
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)
            self.assertEqual(conn.execute('SELECT count(*) FROM assessment_revisions WHERE line_id=?',
                                          (case['line_id'],)).fetchone()[0], 0)

    def test_cli_case_rejects_archive_id_without_creating_workflow_or_approval(self):
        import_archive(self.db, payload())
        line = list_archived_cases(self.db)[0]['line_id']
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(['--db', str(self.db), 'case', '--line', line])
            self.assertEqual(result.exception.code, 2)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)

    @unittest.skipUnless(importlib.util.find_spec('langgraph') is not None, 'Install graph extra for graph guard')
    def test_graph_and_cli_start_reject_before_provider_or_checkpoint_creation(self):
        from recovery.workflow import run_persistent

        import_archive(self.db, payload())
        line = list_archived_cases(self.db)[0]['line_id']
        checkpoints = self.root / 'must-not-create.sqlite3'
        with patch('recovery.ai.resolve_explanation_config', side_effect=AssertionError('No archive provider calls')):
            with self.assertRaisesRegex(ValueError, 'READ_ONLY'):
                run_persistent(self.db, checkpoints, 'archive-cannot-start', line_id=line, use_llm=True,
                               provider='ollama', model='synthetic-model')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main(['--db', str(self.db), 'graph-start', '--line', line,
                        '--thread', 'archive-cannot-start-cli', '--checkpoints', str(checkpoints),
                        '--llm', '--provider', 'ollama', '--model', 'synthetic-model'])
                self.assertEqual(result.exception.code, 2)
        self.assertFalse(checkpoints.exists())

    def test_exact_two_thousand_history_rows_have_fifty_vendors_and_two_year_cohorts(self):
        receipt = import_archive(self.db, payload(vendors=50, cases_per_year=1000))
        self.assertEqual(receipt['row_counts'], {'suppliers': 50, 'contracts': 100, 'cases': 2000})
        self.assertEqual(receipt['counts_by_year'], {'2024': 1000, '2025': 1000})
        self.assertEqual(archive_summary(self.db)['totals_by_currency'], {'USD': '246900.00'})
        self.assertEqual(count_archived_cases(self.db), 2000)
        self.assertEqual(len(list_archived_cases(self.db, limit=50)), 50)
        with connect(self.db) as conn:
            report = audit(conn)
        self.assertEqual(report['case_count'], 20)
        self.assertEqual(report['candidate_total'], '80707.50')
        self.assertEqual(report['realized_recovery'], '0.00')


if __name__ == '__main__':
    unittest.main()
