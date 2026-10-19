"""Autocomplete values reflect existing synthetic records and never mutate them."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from recovery.chatbot import answer_question
from recovery.demo_portfolio import historical_portfolio_payload
from recovery.legacy_archive import import_archive
from recovery.lifecycle_storage import migrate
from recovery.storage import connect, load_snapshot
from recovery.suggestions import chatbot_suggestions, load_suggestions
from recovery.vendor_contacts import seed_vendor_contacts


ROOT = Path(__file__).resolve().parents[1]


class SuggestionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.db = self.directory / 'synthetic.sqlite3'
        load_snapshot(self.db, ROOT / 'data')

    def snapshot(self):
        with sqlite3.connect(self.db) as conn:
            return '\n'.join(conn.iterdump())

    def test_original_fixture_values_without_optional_tables(self):
        before = self.snapshot()
        values = load_suggestions(self.db)
        self.assertEqual(values['vendor_ids'], ['SUP-A', 'SUP-B', 'SUP-C'])
        self.assertEqual(values['cases'], [f'L{i:03d}' for i in range(1, 21)])
        self.assertIn('CTR-A', values['contracts'])
        self.assertIn('DOC-A2', values['document_ids'])
        self.assertTrue(values['document_titles'])
        self.assertTrue(values['invoices'])
        self.assertEqual(values['poc_names'], [])
        self.assertEqual(values['evidence_requests'], [])
        self.assertTrue(all(isinstance(value, str) for items in values.values() for value in items))
        self.assertEqual(self.snapshot(), before)

    def test_tenant_canary_is_excluded_from_demo_and_supported_in_its_own_scope(self):
        demo = load_suggestions(self.db)
        other = load_suggestions(self.db, tenant='OTHER')
        self.assertNotIn('Isolation Canary', str(demo))
        self.assertIn('Isolation Canary', str(other))
        self.assertNotIn('SUP-B', other['vendor_ids'])
        with self.assertRaisesRegex(ValueError, 'tenant'):
            load_suggestions(self.db, tenant='UNKNOWN')

    def test_vendor_and_contract_filters_cascade_without_widening(self):
        vendor = load_suggestions(self.db, supplier_id='SUP-A')
        scoped = load_suggestions(self.db, supplier_id='SUP-A', contract_id='CTR-A')
        self.assertEqual(scoped['vendor_ids'], ['SUP-A'])
        self.assertEqual(scoped['contracts'], ['CTR-A'])
        self.assertEqual(scoped['cases'], vendor['cases'])
        self.assertEqual(scoped['document_ids'], ['DOC-A1', 'DOC-A2', 'DOC-ADRAFT', 'DOC-NOTE'])
        incompatible = load_suggestions(self.db, supplier_id='SUP-B', contract_id='CTR-A')
        for key in ('vendors', 'contracts', 'cases', 'documents', 'evidence_requests'):
            self.assertEqual(incompatible[key], [])
        unknown = load_suggestions(self.db, supplier_id='SUP-NONEXISTENT')
        self.assertEqual(unknown['cases'], [])

    def test_fictional_contacts_have_raw_search_values_and_visible_labels(self):
        seed_vendor_contacts(self.db, synthetic_confirmed=True, contacts=[
            {'supplier_id': 'SUP-A', 'poc_name': 'Avery Example'}])
        values = load_suggestions(self.db, supplier_id='SUP-A')
        self.assertEqual(values['poc_names'], ['Avery Example'])
        self.assertIn('Avery Example', values['vendor_search'])
        self.assertIn(values['vendor_names'][0], values['vendor_search'])
        self.assertIn('SUP-A', values['vendor_search'])
        self.assertIn('Avery Example', values['vendors'][0])
        self.assertTrue(any('Avery Example' in label for label in values['question_labels']))
        self.assertNotIn('Avery Example', load_suggestions(self.db, tenant='OTHER')['poc_names'])

    def test_historical_cohorts_are_distinct_and_all_includes_current(self):
        payload = historical_portfolio_payload(vendor_count=2, cases_per_year=4)
        import_archive(self.db, payload)
        current = load_suggestions(self.db)
        old = load_suggestions(self.db, cohort=2024)
        newer = load_suggestions(self.db, cohort='2025')
        all_values = load_suggestions(self.db, cohort='all')
        self.assertEqual(len(old['cases']), 4)
        self.assertEqual(len(newer['cases']), 4)
        self.assertFalse(set(old['cases']) & set(newer['cases']))
        self.assertFalse(set(current['cases']) & set(old['cases']))
        self.assertEqual(len(all_values['cases']), 28)
        self.assertEqual(old['documents'], [])
        self.assertEqual(load_suggestions(self.db, cohort=2026), current)
        self.assertIn(f'Explain case {old["cases"][0]}', old['questions'])

    def test_pending_evidence_ids_follow_current_vendor_contract_scope(self):
        # The optional table is isolated here: autocomplete reads an existing
        # request projection and does not create a lifecycle or journal event.
        with connect(self.db) as conn, conn:
            conn.execute('CREATE TABLE evidence_requests (request_id TEXT,tenant_id TEXT,line_id TEXT,status TEXT)')
            conn.executemany('INSERT INTO evidence_requests VALUES (?,?,?,?)', [
                ('REQ-A', 'DEMO', 'L001', 'OPEN'),
                ('REQ-B', 'DEMO', 'L001', 'EVIDENCE_ATTACHED'),
                ('REQ-CLOSED', 'DEMO', 'L001', 'RESOLVED'),
                ('REQ-OTHER', 'OTHER', 'L001', 'OPEN'),
                ('REQ-MISSING', 'DEMO', 'MISSING', 'OPEN'),
            ])
        self.assertEqual(load_suggestions(self.db)['evidence_requests'], ['REQ-A', 'REQ-B'])
        self.assertEqual(load_suggestions(self.db, supplier_id='SUP-B')['evidence_requests'], [])
        self.assertEqual(load_suggestions(self.db, cohort=2024)['evidence_requests'], [])

    def test_all_suggested_base_questions_are_supported_not_writes_or_unknown_ids(self):
        values = load_suggestions(self.db)
        checkpoints = self.directory / 'never-created.sqlite3'
        before = self.snapshot()
        for question in values['questions']:
            with self.subTest(question=question):
                reply = answer_question(self.db, checkpoints, question)
                self.assertNotIn(reply['intent'], ('help', 'read_only', 'not_found', 'withheld'))
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(checkpoints.exists())
        self.assertFalse(any('DOC-' in question or 'REQ-' in question for question in values['questions']))

    def test_bounded_literal_question_matching_supports_exact_case_and_contact_search(self):
        seed_vendor_contacts(self.db, synthetic_confirmed=True, contacts=[
            {'supplier_id': 'SUP-A', 'poc_name': 'Avery Example'}])
        values = load_suggestions(self.db)
        self.assertEqual(chatbot_suggestions(values, 'l001'), ['Explain case L001'])
        self.assertTrue(chatbot_suggestions(values, 'avery example'))
        self.assertEqual(len(chatbot_suggestions(values, limit=3)), 3)
        self.assertEqual(chatbot_suggestions(values, '%'), [])
        self.assertEqual(chatbot_suggestions(values, 'not a known field value'), [])
        self.assertEqual(values, load_suggestions(self.db))

    def test_question_labels_preserve_canonical_scope_for_repeated_contacts(self):
        seed_vendor_contacts(self.db, synthetic_confirmed=True, contacts=[
            {'supplier_id': 'SUP-A', 'poc_name': 'Taylor Example'},
            {'supplier_id': 'SUP-B', 'poc_name': 'Taylor Example'}])
        values = load_suggestions(self.db)
        self.assertEqual(len(values['questions']), len(values['question_labels']))
        labels = dict(zip(values['questions'], values['question_labels']))
        for vendor in ('SUP-A', 'SUP-B'):
            question = f'Who is the POC for {vendor}?'
            self.assertIn('Taylor Example', labels[question])
            self.assertIn(vendor, labels[question])
            self.assertNotIn('Taylor Example', question)
            response = answer_question(self.db, self.directory / 'no-checkpoint.sqlite3', question)
            self.assertEqual(response['scope'], {'supplier_id': vendor})
            self.assertEqual(response['rows'][0]['supplier_id'], vendor)
        matched = chatbot_suggestions(values, 'Taylor Example', limit=50)
        self.assertIn('Who is the POC for SUP-A?', matched)
        self.assertIn('Who is the POC for SUP-B?', matched)
        self.assertIn('CTR-A', labels['Explain case L001'])
        self.assertIn('Invoice:', labels['Explain case L001'])
        self.assertIn('Taylor Example', labels['Explain case L001'])
        self.assertEqual(chatbot_suggestions(values, labels['Explain case L001']), ['Explain case L001'])

    def test_invalid_scope_query_and_limits_fail_closed(self):
        for scope in ({'tenant': "DEMO' OR 1=1"}, {'tenant': ''}, {'cohort': True},
                      {'cohort': '2023'}, {'cohort': []}, {'supplier_id': "SUP-A'"},
                      {'contract_id': 'CTR-A\n'}):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                load_suggestions(self.db, **scope)
        values = load_suggestions(self.db)
        for query in (None, 'x' * 2001, 'bad\nquery'):
            with self.subTest(query=query), self.assertRaises(ValueError):
                chatbot_suggestions(values, query)
        for limit in (0, 51, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                chatbot_suggestions(values, limit=limit)
        for catalog in ({}, {'questions': [None]}, {'questions': ['Show contracts'], 'question_labels': []}, None):
            with self.subTest(catalog=catalog), self.assertRaises(ValueError):
                chatbot_suggestions(catalog)

    def test_missing_database_is_not_created(self):
        missing = self.directory / 'not-created' / 'missing.sqlite3'
        with self.assertRaisesRegex(ValueError, 'existing synthetic database'):
            load_suggestions(missing)
        self.assertFalse(missing.parent.exists())

    def test_malformed_or_nonsynthetic_manifest_is_rejected(self):
        for manifest in ({'synthetic': False}, {'synthetic': 1}, [],
                         {'synthetic': True, 'tenants': ['DEMO']}, 'bad JSON'):
            with connect(self.db) as conn, conn:
                value = manifest if manifest == 'bad JSON' else json.dumps(manifest)
                conn.execute("UPDATE metadata SET value=? WHERE key='manifest'", (value,))
            before = self.snapshot()
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                load_suggestions(self.db)
            self.assertEqual(self.snapshot(), before)

    def test_mutated_source_projection_is_rejected_without_writes(self):
        migrate(self.db, backup_dir=self.directory / 'backups')
        with connect(self.db) as conn, conn:
            conn.execute("UPDATE documents SET path='contracts/changed.md' WHERE tenant_id='DEMO' AND doc_id='DOC-A2'")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'SOURCE_PROJECTION_MISMATCH'):
            load_suggestions(self.db)
        self.assertEqual(self.snapshot(), before)

    def test_incomplete_or_inconsistent_archive_provenance_is_rejected(self):
        with connect(self.db) as conn, conn:
            conn.execute('CREATE TABLE legacy_cases (line_id TEXT)')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            load_suggestions(self.db, cohort='all')

    def test_inconsistent_archived_case_cannot_supply_suggestions(self):
        import_archive(self.db, historical_portfolio_payload(vendor_count=1, cases_per_year=2))
        # Deliberately corrupt an isolated copy, removing only its immutable
        # update trigger; the reader must not trust the remaining projection.
        with sqlite3.connect(self.db) as conn:
            conn.execute('DROP TRIGGER legacy_cases_immutable_update')
            conn.execute("UPDATE legacy_cases SET contract_id='UNKNOWN'")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'scope is inconsistent'):
            load_suggestions(self.db, cohort='all')
        self.assertEqual(self.snapshot(), before)

    def test_catalog_does_not_reuse_another_database_snapshot(self):
        other_db = self.directory / 'other.sqlite3'
        load_snapshot(other_db, ROOT / 'data')
        with connect(other_db) as conn, conn:
            conn.execute("UPDATE suppliers SET name='Different Demo Name' WHERE tenant_id='DEMO' AND supplier_id='SUP-A'")
        self.assertNotIn('Different Demo Name', str(load_suggestions(self.db)))
        self.assertIn('Different Demo Name', str(load_suggestions(other_db)))


if __name__ == '__main__':
    unittest.main()
