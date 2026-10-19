"""Paged current work and locked migration history on temporary synthetic data."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sqlite3
import unittest
from unittest.mock import patch

import test_ui as ui_helpers

from recovery.engine import audit
from recovery.legacy_archive import import_archive
from recovery.lifecycle import migrate
from recovery.storage import connect, load_snapshot
from recovery.workflow import list_review_workflows, run_persistent


def contents(path):
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        return '\n'.join(conn.iterdump())
    finally:
        conn.close()


def archive_payload():
    payload = dict(schema_version=1, synthetic=True, archive_id='LEGACY-UI-TEST',
        source_system='Invented old-system migration',
        provenance='Synthetic UI pagination fixture; no current approval or recovery.',
        suppliers=[dict(tenant_id='DEMO', supplier_id='LEGACY-V-NORTH', name='Migration North Synthetic'),
                   dict(tenant_id='DEMO', supplier_id='LEGACY-V-SOUTH', name='Migration South Synthetic')],
        contracts=[], cases=[])
    for year, supplier, count in ((2025, 'LEGACY-V-NORTH', 125), (2024, 'LEGACY-V-SOUTH', 2)):
        contract = f'LEGACY-C-{year}'
        payload['contracts'].append(dict(tenant_id='DEMO', supplier_id=supplier,
            contract_id=contract, title=f'Invented {year} migration contract', origin_year=year))
        for index in range(1, count + 1):
            payload['cases'].append(dict(tenant_id='DEMO', line_id=f'LEGACY-L-{year}-{index:04d}',
                supplier_id=supplier, contract_id=contract, invoice_id=f'LEGACY-I-{year}-{index:04d}',
                invoice_date=f'{year}-07-15', origin_year=year, currency='USD',
                historical_approved_amount='12.34', source_record_id=f'OLD-{year}-{index:04d}',
                description='Invented historical approved record, locked after migration.'))
    return payload


@unittest.skipUnless(ui_helpers.AVAILABLE, 'Install UI and graph extras for scaled portfolio UI')
class ScaledPortfolioUITests(unittest.TestCase):
    def setUp(self):
        self.helper = ui_helpers.ReviewWorkflowUITests(methodName='runTest')
        self.addCleanup(self.helper.doCleanups)
        self.helper.setUp()
        self.db, self.cp = self.helper.db, self.helper.checkpoints
        fixture = Path(self.helper.temp.name) / 'paged-synthetic'
        shutil.copytree(ui_helpers.ROOT / 'data', fixture)
        tables = {}
        for name in ('invoice_lines', 'service_entries', 'payments'):
            with (fixture / (name + '.csv')).open(newline='') as source:
                reader = csv.DictReader(source)
                tables[name] = (reader.fieldnames, list(reader))
        invoice = next(row for row in tables['invoice_lines'][1] if row['line_id'] == 'L004')
        service = next(row for row in tables['service_entries'][1] if row['line_id'] == 'L004')
        payment = next(row for row in tables['payments'][1] if row['line_id'] == 'L004')
        for index in range(1, 121):
            line = f'SCALE-2026-{index:04d}'
            tables['invoice_lines'][1].append(dict(invoice, line_id=line,
                invoice_id=f'SCALE-INV-{index:04d}', billed_hours='1', billed_amount='130.00',
                description=f'Synthetic pagination invoice {index:04d}'))
            tables['service_entries'][1].append(dict(service, line_id=line,
                entry_id=f'SCALE-SVC-{index:04d}', approved_hours='1'))
            tables['payments'][1].append(dict(payment, line_id=line,
                payment_id=f'SCALE-PAY-{index:04d}', amount='130.00'))
        for name, (fields, rows) in tables.items():
            with (fixture / (name + '.csv')).open('w', newline='') as destination:
                writer = csv.DictWriter(destination, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        load_snapshot(self.db, fixture)
        migrate(self.db, backup_dir=Path(self.helper.temp.name) / 'backups')
        self.app = self.helper.app

    def widget(self, kind, label):
        return self.helper.widget(kind, label)

    def search(self, label, value):
        self.helper.search(label, value)

    def clean(self):
        self.helper.assert_clean()
        self.assertFalse(self.app.error, [item.value for item in self.app.error])
        self.helper.resolve.assert_not_called()
        self.helper.generate.assert_not_called()

    def frame(self, column):
        return next(frame for frame in self.app.dataframe if column in frame.value.columns)

    def select_row(self, column, value):
        frame = self.frame(column)
        index = frame.value[column].tolist().index(value)
        self.app.session_state[frame.key] = {'selection': {'rows': [index], 'columns': [], 'cells': []}}
        self.app.run()
        self.clean()

    def baseline(self, events=0):
        with connect(self.db) as conn:
            result = audit(conn)
            count = conn.execute('SELECT count(*) FROM review_events').fetchone()[0]
        self.assertEqual(result['case_count'], 140)
        self.assertEqual(result['candidate_total'], '81907.50')
        self.assertEqual(result['realized_recovery'], '0.00')
        self.assertEqual(count, events)

    def test_invoice_vendor_name_id_and_invoice_search_are_read_only(self):
        before = contents(self.db)
        self.app.run()
        self.assertEqual(len(self.helper.visible_lines()), 50)
        self.search('Search vendor name or ID', 'asterTECH')
        self.assertEqual(set(self.helper.invoice_table().value['Vendor']), {'AsterTech Synthetic Services'})
        self.search('Search invoice or contract', 'SCALE-INV-0001')
        self.assertEqual(self.helper.visible_lines(), ['SCALE-2026-0001'])
        self.search('Search invoice or contract', '')
        self.widget('radio', 'Review scope').set_value('All cases').run()
        self.search('Search vendor name or ID', 'sup-b')
        self.assertEqual(set(self.helper.invoice_table().value['Vendor']), {'BeaconStaff Synthetic Services'})
        self.search('Search vendor name or ID', 'no such synthetic vendor')
        self.assertFalse(any(frame.key and frame.key.startswith('invoice_table:') for frame in self.app.dataframe))
        self.assertIsNone(self.helper.selected_line())
        self.assertEqual(contents(self.db), before)
        self.assertFalse(self.cp.exists())
        self.clean()
        self.baseline()

    def test_invoice_suggestions_cover_all_pages_and_respect_vendor_contract_scope(self):
        before = contents(self.db)
        self.app.run()
        vendor_search = self.widget('selectbox', 'Search vendor name or ID')
        self.assertIsNone(vendor_search.value)
        self.assertTrue(vendor_search.proto.accept_new_options)
        self.assertIn('SUP-A', vendor_search.options)
        self.assertIn('AsterTech Synthetic Services', vendor_search.options)
        self.assertFalse(self.widget('selectbox', 'Vendor').proto.accept_new_options)
        self.widget('selectbox', 'Vendor').select('SUP-A').run()
        self.widget('selectbox', 'Contract').select('CTR-A').run()
        invoice_search = self.widget('selectbox', 'Search invoice or contract')
        self.assertIn('SCALE-2026-0001', invoice_search.options)
        self.assertIn('SCALE-2026-0120', invoice_search.options)
        self.assertNotIn('L006', invoice_search.options)
        self.assertNotIn('CTR-C', invoice_search.options)
        self.assertNotIn('SCALE-2026-0001', self.helper.visible_lines())
        self.helper.open_line('SCALE-2026-0120')
        table = self.helper.invoice_table()
        self.app.session_state[table.key] = {'edited_rows': {0: {'Select': True}},
            'added_rows': [], 'deleted_rows': []}
        self.app.run()
        self.assertTrue(self.helper.invoice_table().value['Select'].any())
        self.search('Search invoice or contract', 'SCALE-2026-0001')
        self.assertEqual(self.helper.visible_lines(), ['SCALE-2026-0001'])
        self.assertIsNone(self.helper.selected_line())
        self.assertFalse(self.helper.invoice_table().value['Select'].any())
        self.search('Search invoice or contract', '')
        self.assertEqual(len(self.helper.visible_lines()), 50)
        self.assertEqual(contents(self.db), before)
        self.assertFalse(self.cp.exists())
        self.clean()
        self.baseline()

    def test_page_changes_clear_open_and_bulk_selection_without_reusing_positions(self):
        self.app.run()
        self.widget('selectbox', 'Vendor').select('SUP-A').run()
        self.widget('selectbox', 'Contract').select('CTR-A').run()
        first = self.helper.visible_lines()
        self.helper.open_line(first[0])
        table = self.helper.invoice_table()
        self.app.session_state[table.key] = {'edited_rows': {0: {'Select': True}},
                                              'added_rows': [], 'deleted_rows': []}
        self.app.run()
        self.assertTrue(self.helper.invoice_table().value['Select'].any())
        self.widget('number_input', 'Invoice page').set_value(2).run()
        second = self.helper.visible_lines()
        self.assertTrue(set(first).isdisjoint(second))
        self.assertIsNone(self.helper.selected_line())
        self.assertFalse(self.helper.invoice_table().value['Select'].any())
        self.widget('number_input', 'Invoice page').set_value(1).run()
        self.assertFalse(self.helper.invoice_table().value['Select'].any())
        self.widget('selectbox', 'Invoice rows per page').select(100).run()
        self.assertEqual(len(self.helper.visible_lines()), 100)
        self.clean()
        self.baseline()

    def test_select_all_approves_only_ready_rows_on_the_displayed_page(self):
        for line in ('SCALE-2026-0120', 'SCALE-2026-0060'):
            run_persistent(self.db, self.cp, 'paged:' + line, line_id=line, use_llm=False)
        self.app.run()
        self.clean()
        self.widget('selectbox', 'Vendor').select('SUP-A').run()
        self.widget('selectbox', 'Contract').select('CTR-A').run()
        self.assertIn('SCALE-2026-0120', self.helper.visible_lines())
        self.assertNotIn('SCALE-2026-0060', self.helper.visible_lines())
        self.widget('button', 'Select all eligible in this contract').click().run()
        table = self.helper.invoice_table().value
        self.assertEqual(table.loc[table['Select'], 'Case ID'].tolist(), ['SCALE-2026-0120'])
        self.widget('text_input', 'Bulk reviewer label').set_value('Synthetic accountant')
        self.widget('text_area', 'Bulk evidence review note').set_value('Checked the one selected visible invoice.')
        self.widget('button', 'Approve selected for draft').click().run()
        self.clean()
        workflows = {row['state']['line_id']: row for row in list_review_workflows(self.db, self.cp)}
        self.assertEqual(workflows['SCALE-2026-0120']['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
        self.assertEqual(workflows['SCALE-2026-0060']['next'], ['approval'])
        self.baseline(events=1)

    def test_contract_vendor_search_and_related_case_pagination(self):
        before = contents(self.db)
        self.app.run()
        self.helper.navigate('Contract term review')
        self.search('Search vendor name or ID', 'astertech')
        self.assertEqual(self.frame('Vendor ID').value['Vendor ID'].tolist(), ['SUP-A'])
        self.select_row('Vendor ID', 'SUP-A')
        self.assertEqual(len(self.frame('Invoice line').value), 50)
        self.search('Search contract or document', 'DOC-A2')
        self.assertEqual(self.frame('Document').value['Document'].tolist(), ['DOC-A2'])
        self.select_row('Document', 'DOC-A2')
        self.search('Search related invoice', 'SCALE-INV-0001')
        self.assertEqual(self.frame('Invoice line').value['Invoice line'].tolist(), ['SCALE-2026-0001'])
        self.assertEqual(contents(self.db), before)
        self.clean()
        self.baseline()

    def test_contract_document_and_related_invoice_suggestions_are_vendor_scoped(self):
        before = contents(self.db)
        self.app.run()
        self.helper.navigate('Contract term review')
        self.search('Search vendor name or ID', 'SUP-A')
        self.select_row('Vendor ID', 'SUP-A')
        contracts = self.widget('selectbox', 'Search contract or document')
        self.assertTrue(contracts.proto.accept_new_options)
        self.assertIn('DOC-A2', contracts.options)
        self.assertIn('CTR-A', contracts.options)
        self.assertNotIn('DOC-C1', contracts.options)
        self.search('Search contract or document', 'DOC-A2')
        self.select_row('Document', 'DOC-A2')
        invoices = self.widget('selectbox', 'Search related invoice')
        self.assertIn('SCALE-2026-0001', invoices.options)
        self.assertIn('SCALE-2026-0120', invoices.options)
        self.assertNotIn('L006', invoices.options)
        self.search('Search related invoice', 'SCALE-2026-0120')
        self.assertEqual(self.frame('Invoice line').value['Invoice line'].tolist(), ['SCALE-2026-0120'])
        self.assertEqual(contents(self.db), before)
        self.assertFalse(self.cp.exists())
        self.clean()
        self.baseline()

    def test_migrated_year_vendor_contract_and_invoice_are_locked_and_sql_only(self):
        import_archive(self.db, archive_payload())
        before = contents(self.db)
        self.app.run()
        self.helper.open_line('L001')
        with patch('recovery.invoice_ui.display_worklist', side_effect=AssertionError('Archive must not assess current work')), \
             patch('recovery.ui_cache.display_worklist', side_effect=AssertionError('Archive must not load current workflows')):
            self.widget('selectbox', 'Case cohort').select('Migrated · 2025 (locked)').run()
            self.assertIsNone(self.helper.selected_line())
            self.search('Search vendor name or ID', 'migration north')
            self.select_row('Vendor ID', 'LEGACY-V-NORTH')
            self.select_row('Contract', 'LEGACY-C-2025')
            first = self.frame('Case ID').value['Case ID'].tolist()
            self.assertEqual(len(first), 50)
            self.select_row('Case ID', first[0])
            self.assertTrue(any('Migrated from old system — locked' in element.value
                                for element in self.app.subheader))
            self.assertFalse(self.helper.decision_buttons())
            self.assertFalse(any('Select' in frame.value.columns for frame in self.app.dataframe))
            self.widget('number_input', 'Migrated invoices page').set_value(2).run()
            self.assertTrue(set(first).isdisjoint(self.frame('Case ID').value['Case ID']))
            self.assertFalse(any(first[0] in element.value for element in self.app.subheader))
            self.widget('selectbox', 'Case cohort').select('Migrated · 2024 (locked)').run()
            self.assertEqual(self.frame('Vendor ID').value['Vendor ID'].tolist(), ['LEGACY-V-SOUTH'])
            self.select_row('Vendor ID', 'LEGACY-V-SOUTH')
            self.assertEqual(set(self.frame('Case ID').value['Year']), {2024})
        self.assertEqual(contents(self.db), before)
        self.assertFalse(self.cp.exists())
        self.clean()
        self.baseline()

    def test_contract_review_archive_search_never_offers_term_or_invoice_actions(self):
        import_archive(self.db, archive_payload())
        before = contents(self.db)
        self.app.run()
        self.helper.navigate('Contract term review')
        with patch('recovery.ui_cache.display_worklist',
                   side_effect=AssertionError('Archived contracts must not assess current work')):
            self.widget('selectbox', 'Case cohort').select('Migrated · 2024 (locked)').run()
            self.search('Search vendor name or ID', 'legacy-v-south')
            self.assertEqual(self.frame('Vendor ID').value['Vendor ID'].tolist(), ['LEGACY-V-SOUTH'])
            self.select_row('Vendor ID', 'LEGACY-V-SOUTH')
            self.select_row('Contract', 'LEGACY-C-2024')
            self.select_row('Case ID', 'LEGACY-L-2024-0001')
        self.assertFalse(any(button.label in {'Approve Terms', 'Reject Proposal',
            'Approve for draft', 'Approve selected for draft', 'Prepare review'} for button in self.app.button))
        self.assertEqual(contents(self.db), before)
        self.assertFalse(self.cp.exists())
        self.clean()
        self.baseline()

    def test_archive_suggestions_are_year_scoped_and_find_invoices_beyond_the_first_page(self):
        import_archive(self.db, archive_payload())
        before = contents(self.db)
        self.app.run()
        with patch('recovery.invoice_ui.display_worklist', side_effect=AssertionError('Archive must not assess current work')), \
             patch('recovery.ui_cache.display_worklist', side_effect=AssertionError('Archive must not load current workflows')):
            self.widget('selectbox', 'Case cohort').select('Migrated · 2025 (locked)').run()
            vendors = self.widget('selectbox', 'Search vendor name or ID')
            self.assertIn('LEGACY-V-NORTH', vendors.options)
            self.assertNotIn('LEGACY-V-SOUTH', vendors.options)
            self.select_row('Vendor ID', 'LEGACY-V-NORTH')
            contracts = self.widget('selectbox', 'Search contract')
            self.assertIn('LEGACY-C-2025', contracts.options)
            self.assertNotIn('LEGACY-C-2024', contracts.options)
            self.select_row('Contract', 'LEGACY-C-2025')
            invoices = self.widget('selectbox', 'Search invoice')
            self.assertIn('LEGACY-L-2025-0125', invoices.options)
            self.assertNotIn('LEGACY-L-2024-0001', invoices.options)
            self.assertNotIn('LEGACY-L-2025-0125', self.frame('Case ID').value['Case ID'].tolist())
            self.search('Search invoice', 'LEGACY-L-2025-0125')
            self.assertEqual(self.frame('Case ID').value['Case ID'].tolist(), ['LEGACY-L-2025-0125'])
            self.search('Search invoice', 'no matching old invoice')
            self.assertFalse(any('Case ID' in frame.value.columns for frame in self.app.dataframe))
            self.search('Search invoice', '')
            self.assertEqual(len(self.frame('Case ID').value), 50)
            self.widget('selectbox', 'Case cohort').select('Migrated · 2024 (locked)').run()
            self.assertNotIn('LEGACY-V-NORTH', self.widget('selectbox', 'Search vendor name or ID').options)
        self.assertEqual(contents(self.db), before)
        self.assertFalse(self.cp.exists())
        self.assertFalse(self.helper.decision_buttons())
        self.clean()
        self.baseline()

    def test_restored_overview_work_queue_is_paged_current_only_and_read_only(self):
        import_archive(self.db, archive_payload())
        before = contents(self.db)
        self.app.run()
        self.helper.navigate('Overview')
        self.app.button(key='lifecycle:overview:card:preparation').click().run()
        self.clean()
        first = self.frame('Invoice line').value['Invoice line'].tolist()
        self.assertEqual(len(first), 50)
        self.assertTrue(all(not line.startswith('LEGACY-') for line in first))
        self.assertTrue(self.widget('button', 'Open selected invoice').disabled)
        self.select_row('Invoice line', first[0])
        self.assertFalse(self.widget('button', 'Open selected invoice').disabled)
        self.widget('number_input', 'Overview cases page').set_value(2).run()
        self.clean()
        second = self.frame('Invoice line').value['Invoice line'].tolist()
        self.assertEqual(len(second), 50)
        self.assertTrue(set(first).isdisjoint(second))
        self.assertTrue(self.widget('button', 'Open selected invoice').disabled)
        self.widget('button', 'Close cases').click().run()
        self.clean()
        self.assertFalse(self.app.dataframe)
        self.assertFalse(any(button.label == 'Open selected invoice' for button in self.app.button))
        self.assertEqual(contents(self.db), before)
        self.assertFalse(self.cp.exists())
        self.baseline()

    def test_archive_values_enter_only_grand_total_reporting_not_current_finance(self):
        self.app.run()
        self.helper.navigate('Overview')
        before = {metric.label: metric.value for metric in self.app.metric}
        self.assertEqual({label: before[label] for label in ('Total approved value',
            'Estimated cash recovery under demo assumptions', 'Realized recovery')}, {
            'Total approved value': 'USD 0.00',
            'Estimated cash recovery under demo assumptions': 'Not estimated',
            'Realized recovery': 'USD 0.00',
        })
        import_archive(self.db, archive_payload())
        after_import = contents(self.db)
        self.app.run()
        after = {metric.label: metric.value for metric in self.app.metric}
        self.assertEqual({label: after[label] for label in ('Total approved value',
            'Estimated cash recovery under demo assumptions', 'Realized recovery')}, {
            'Total approved value': 'USD 1,567.18',  # 127 invented migrated approvals × 12.34
            'Estimated cash recovery under demo assumptions': 'Not estimated',
            'Realized recovery': 'USD 0.00',
        })
        self.assertEqual(len(self.app.metric), 12)
        self.assertEqual((before['Suppliers'], before['Contracts']), ('3', '4'))
        self.assertEqual((after['Suppliers'], after['Contracts']), ('5', '6'))
        from recovery.dashboard import BUCKET_LABELS
        self.assertEqual({label: before[label] for label in BUCKET_LABELS.values()},
                         {label: after[label] for label in BUCKET_LABELS.values()})
        self.assertFalse(self.app.expander)
        self.assertFalse(any('Locked migration archive:' in element.value for element in self.app.info))
        self.assertEqual(contents(self.db), after_import)
        self.clean()
        self.baseline()


if __name__ == '__main__':
    unittest.main()
