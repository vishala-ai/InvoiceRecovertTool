"""Fictional contact seeding never alters source, finance or review identities."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from recovery.demo_portfolio import historical_portfolio_payload
from recovery.engine import audit
from recovery.legacy_archive import import_archive
from recovery.storage import connect, load_snapshot
from recovery.vendor_contacts import (
    PROVENANCE, get_vendor_contact, list_vendor_contacts, seed_vendor_contacts,
)


ROOT = Path(__file__).resolve().parents[1]


class VendorContactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.db = self.directory / 'synthetic.sqlite3'
        load_snapshot(self.db, ROOT / 'data')

    def snapshot(self):
        with connect(self.db) as conn:
            tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name!='vendor_contacts'")]
            return {name: sorted([tuple(row) for row in conn.execute(f'SELECT * FROM "{name}"')],
                                  key=repr) for name in tables}

    def schema(self):
        with connect(self.db) as conn:
            return [tuple(row) for row in conn.execute('SELECT * FROM sqlite_master ORDER BY name')]

    def seed(self, **changes):
        return seed_vendor_contacts(self.db, **{'synthetic_confirmed': True, **changes})

    def test_absent_table_reads_do_not_create_schema_or_database(self):
        before, schema = self.snapshot(), self.schema()
        self.assertEqual(list_vendor_contacts(self.db), [])
        self.assertIsNone(get_vendor_contact(self.db, 'DEMO', 'SUP-A'))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.schema(), schema)
        missing = self.directory / 'missing' / 'database.sqlite3'
        with self.assertRaisesRegex(ValueError, 'existing synthetic database'):
            list_vendor_contacts(missing)
        with self.assertRaisesRegex(ValueError, 'existing synthetic database'):
            seed_vendor_contacts(missing, synthetic_confirmed=True)
        self.assertFalse(missing.parent.exists())

    def test_seed_is_idempotent_and_preserves_all_prior_rows_and_case_digests(self):
        before = self.snapshot()
        with connect(self.db) as conn:
            audit_before = audit(conn)
        result = self.seed()
        self.assertEqual(result['supplier_count'], 3)
        self.assertEqual(result['inserted'], 3)
        contacts = list_vendor_contacts(self.db)
        self.assertEqual({row['supplier_id'] for row in contacts}, {'SUP-A', 'SUP-B', 'SUP-C'})
        self.assertTrue(all(row['synthetic'] is True and row['provenance'] == PROVENANCE
                            for row in contacts))
        repeated = self.seed()
        self.assertEqual((repeated['inserted'], repeated['retained']), (0, 3))
        self.assertEqual(repeated['contacts'], contacts)
        self.assertEqual(get_vendor_contact(self.db, supplier_id='SUP-A'), contacts[0])
        self.assertEqual(self.snapshot(), before)
        with connect(self.db) as conn:
            self.assertEqual(audit(conn), audit_before)
        self.assertEqual(audit_before['candidate_total'], '80707.50')
        self.assertEqual(audit_before['realized_recovery'], '0.00')

    def test_includes_locked_archive_suppliers_without_changing_archive_rows(self):
        payload = historical_portfolio_payload(vendor_count=2, cases_per_year=2)
        import_archive(self.db, payload)
        before = self.snapshot()
        result = self.seed()
        self.assertEqual(result['supplier_count'], 5)
        expected = {'SUP-A', 'SUP-B', 'SUP-C', *(row['supplier_id'] for row in payload['suppliers'])}
        self.assertEqual({row['supplier_id'] for row in result['contacts']}, expected)
        self.assertEqual(self.snapshot(), before)

    def test_contacts_are_scoped_to_selected_manifest_tenant(self):
        demo = self.seed()
        self.assertEqual(list_vendor_contacts(self.db, 'OTHER'), [])
        other = self.seed(tenant='OTHER')
        self.assertTrue(other['contacts'])
        self.assertTrue(all(row['tenant_id'] == 'OTHER' for row in other['contacts']))
        self.assertEqual(list_vendor_contacts(self.db), demo['contacts'])
        # The deliberate cross-tenant canary reuses SUP-A. Its contact stays distinct.
        demo_contact = get_vendor_contact(self.db, 'DEMO', 'SUP-A')
        other_contact = get_vendor_contact(self.db, 'OTHER', 'SUP-A')
        self.assertEqual(demo_contact['tenant_id'], 'DEMO')
        self.assertEqual(other_contact['tenant_id'], 'OTHER')
        self.assertNotEqual(demo_contact['poc_name'], other_contact['poc_name'])
        self.assertIsNone(get_vendor_contact(self.db, 'OTHER', 'SUP-B'))

    def test_default_seed_preserves_custom_fictional_poc_name(self):
        result = self.seed(contacts=[{'supplier_id': 'SUP-A', 'poc_name': 'Avery Demo'}])
        original = result['contacts'][0]
        self.assertEqual(self.seed()['inserted'], 2)
        self.assertEqual(get_vendor_contact(self.db, supplier_id='SUP-A'), original)
        repeated = self.seed(contacts=[{'supplier_id': 'SUP-A', 'poc_name': 'Avery Demo'}])
        self.assertEqual(repeated['inserted'], 0)
        self.assertEqual(repeated['retained'], 1)

    def test_conflicting_explicit_seed_is_rejected_atomically(self):
        self.seed(contacts=[{'supplier_id': 'SUP-A', 'poc_name': 'Avery Demo'}])
        before = list_vendor_contacts(self.db)
        with self.assertRaisesRegex(ValueError, 'CONTACT_CONFLICT'):
            self.seed(contacts=[{'supplier_id': 'SUP-B', 'poc_name': 'New Demo'},
                                {'supplier_id': 'SUP-A', 'poc_name': 'Changed Demo'}])
        self.assertEqual(list_vendor_contacts(self.db), before)

    def test_invalid_inputs_never_create_contact_table(self):
        bad = [False, {}, [None], [{'supplier_id': 'SUP-A'}],
               [{'supplier_id': 'SUP-A', 'poc_name': 'Avery Demo', 'email': 'forbidden'}],
               [{'supplier_id': 'UNKNOWN', 'poc_name': 'Avery Demo'}],
               [{'supplier_id': 'SUP-A', 'poc_name': ''}],
               [{'supplier_id': 'SUP-A', 'poc_name': 'Bad\nName'}],
               [{'supplier_id': 'SUP-A', 'poc_name': 'Bad\x7fName'}],
               [{'supplier_id': 'SUP-A', 'poc_name': 5}],
               [{'supplier_id': 'SUP-A', 'poc_name': 'x' * 201}],
               [{'supplier_id': 'SUP-A', 'poc_name': 'Demo'},
                {'supplier_id': 'SUP-A', 'poc_name': 'Demo'}]]
        before = self.schema()
        for contacts in bad:
            with self.subTest(contacts=contacts), self.assertRaises(ValueError):
                self.seed(contacts=contacts)
            self.assertEqual(self.schema(), before)

    def test_confirmation_and_known_tenant_required(self):
        before = self.schema()
        for confirmation in (False, 1, 'true', None):
            with self.subTest(value=confirmation), self.assertRaisesRegex(ValueError, 'SYNTHETIC_ONLY'):
                self.seed(synthetic_confirmed=confirmation)
        for tenant in ('UNKNOWN', '', None, 'DEMO\n'):
            with self.subTest(tenant=tenant), self.assertRaises(ValueError):
                self.seed(tenant=tenant)
        self.assertEqual(self.schema(), before)

    def test_missing_nonsynthetic_and_malformed_manifests_rejected_before_schema(self):
        for manifest in ({'synthetic': False}, {'synthetic': 1}, 'not JSON', [],
                         {'synthetic': True, 'tenants': ['DEMO']}):
            with connect(self.db) as conn, conn:
                value = manifest if manifest == 'not JSON' else json.dumps(manifest)
                conn.execute("UPDATE metadata SET value=? WHERE key='manifest'", (value,))
            before = self.schema()
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                self.seed()
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                list_vendor_contacts(self.db)
            self.assertEqual(self.schema(), before)
        with connect(self.db) as conn, conn:
            conn.execute("DELETE FROM metadata WHERE key='manifest'")
        with self.assertRaisesRegex(ValueError, 'SYNTHETIC_ONLY'):
            self.seed()

    def test_read_only_connection_supported_and_read_factory_preserved(self):
        self.seed()
        with sqlite3.connect(self.db.resolve().as_uri() + '?mode=ro', uri=True) as conn:
            self.assertIsNone(conn.row_factory)
            before = conn.total_changes
            self.assertEqual(len(list_vendor_contacts(conn)), 3)
            self.assertEqual(get_vendor_contact(conn, 'DEMO', 'SUP-A')['supplier_id'], 'SUP-A')
            self.assertEqual(conn.total_changes, before)
            self.assertIsNone(conn.row_factory)
            self.assertFalse(conn.in_transaction)

    def test_seed_supplied_connection_does_not_commit_another_transaction(self):
        with connect(self.db) as conn:
            conn.execute('BEGIN')
            conn.execute("INSERT INTO metadata VALUES ('uncommitted-test','pending')")
            with self.assertRaisesRegex(ValueError, 'idle database connection'):
                seed_vendor_contacts(conn, synthetic_confirmed=True)
            self.assertTrue(conn.in_transaction)
            conn.rollback()
            self.assertIsNone(conn.execute("SELECT 1 FROM metadata WHERE key='uncommitted-test'").fetchone())
        self.assertEqual(list_vendor_contacts(self.db), [])

    def test_failed_insert_rolls_back_prior_new_contacts(self):
        self.seed(contacts=[])
        with connect(self.db) as conn, conn:
            conn.execute('''CREATE TRIGGER fail_contact BEFORE INSERT ON vendor_contacts
                WHEN NEW.supplier_id='SUP-B' BEGIN SELECT RAISE(ABORT,'test failure'); END''')
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'test failure'):
            self.seed()
        self.assertEqual(list_vendor_contacts(self.db), [])

    def test_failed_first_seed_rolls_back_new_schema_as_well_as_rows(self):
        before = self.schema()
        with connect(self.db) as conn:
            conn.set_authorizer(lambda operation, table, *unused:
                sqlite3.SQLITE_DENY if operation == sqlite3.SQLITE_INSERT
                and table == 'vendor_contacts' else sqlite3.SQLITE_OK)
            with self.assertRaises(sqlite3.DatabaseError):
                seed_vendor_contacts(conn, synthetic_confirmed=True)
            conn.set_authorizer(None)
            self.assertFalse(conn.in_transaction)
        self.assertEqual(self.schema(), before)
        self.assertEqual(list_vendor_contacts(self.db), [])

    def test_saved_unknown_supplier_and_false_provenance_are_rejected(self):
        self.seed()
        with connect(self.db) as conn, conn:
            conn.execute("UPDATE vendor_contacts SET provenance='unverified' WHERE supplier_id='SUP-A'")
        with self.assertRaisesRegex(ValueError, 'synthetic provenance'):
            list_vendor_contacts(self.db)
        with self.assertRaisesRegex(ValueError, 'synthetic provenance'):
            self.seed()
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE vendor_contacts SET provenance=?,supplier_id=? WHERE supplier_id=?',
                         (PROVENANCE, 'UNKNOWN', 'SUP-A'))
        with self.assertRaisesRegex(ValueError, 'unknown scope'):
            list_vendor_contacts(self.db)

    def test_incompatible_table_is_rejected_without_overwriting(self):
        with connect(self.db) as conn, conn:
            conn.execute('CREATE TABLE vendor_contacts (supplier_id TEXT,poc_name TEXT)')
        before = self.schema()
        with self.assertRaisesRegex(ValueError, 'schema is incompatible'):
            self.seed()
        self.assertEqual(self.schema(), before)

    def test_missing_metadata_and_missing_supplier_are_handled(self):
        empty = self.directory / 'empty.sqlite3'
        sqlite3.connect(empty).close()
        with self.assertRaisesRegex(ValueError, 'synthetic import manifest'):
            seed_vendor_contacts(empty, synthetic_confirmed=True)
        for supplier in (None, '', 'SUP-A\n'):
            with self.subTest(supplier=supplier), self.assertRaises(ValueError):
                get_vendor_contact(self.db, supplier_id=supplier)

    def test_cli_requires_explicit_confirmation_and_rejects_missing_database(self):
        command = [sys.executable, str(ROOT / 'scripts/seed_vendor_contacts.py'), '--db', str(self.db)]
        missing = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(missing.returncode, 2)
        self.assertEqual(list_vendor_contacts(self.db), [])
        successful = subprocess.run([*command, '--synthetic-confirmed'], capture_output=True,
                                    text=True, check=False)
        self.assertEqual(successful.returncode, 0, successful.stderr)
        self.assertEqual(json.loads(successful.stdout)['inserted'], 3)
        nonexistent = self.directory / 'not-there.sqlite3'
        rejected = subprocess.run([*command[:-1], str(nonexistent), '--synthetic-confirmed'],
                                  capture_output=True, text=True, check=False)
        self.assertEqual(rejected.returncode, 2)
        self.assertEqual(json.loads(rejected.stderr)['status'], 'REJECTED')
        self.assertFalse(nonexistent.exists())


if __name__ == '__main__':
    unittest.main()
