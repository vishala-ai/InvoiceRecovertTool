"""Chat links select exact saved scoped records without changing their state."""
from pathlib import Path
import tempfile
import unittest

from recovery.chatbot_navigation import consume_chat_target, resolve_chat_target
from recovery.legacy_archive import import_archive
from recovery.storage import connect, load_snapshot
from test_legacy_archive import payload


DATA = Path(__file__).resolve().parents[1] / 'data'


class FakeState:
    def __init__(self):
        self.session_state = {}
        self.warnings = []

    def warning(self, value):
        self.warnings.append(value)


class ChatNavigationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'synthetic.sqlite3'
        load_snapshot(self.db, DATA)
        import_archive(self.db, payload())

    def state(self):
        with connect(self.db) as conn:
            return list(conn.iterdump())

    def test_current_case_and_contract_resolve_canonical_scope_read_only(self):
        before = self.state()
        case = resolve_chat_target(self.db, {'line_id': 'L001'}, kind='case')
        self.assertEqual((case['supplier_id'], case['contract_id'], case['origin_year']),
                         ('SUP-A', 'CTR-A', None))
        contract = resolve_chat_target(self.db, {'supplier_id': 'SUP-A', 'contract_id': 'CTR-A'})
        self.assertIsNone(contract['line_id'])
        vendor = resolve_chat_target(self.db, {'supplier_id': 'SUP-A'})
        self.assertIsNone(vendor['contract_id'])
        self.assertEqual(self.state(), before)

    def test_historical_case_year_comes_from_actual_locked_record(self):
        target = resolve_chat_target(self.db, {'line_id': 'LEGACY-2025-0001'}, kind='case')
        self.assertEqual(target['origin_year'], 2025)
        self.assertEqual(target['supplier_id'], 'LEGACY-SUP-001')
        with self.assertRaises(ValueError):
            resolve_chat_target(self.db, {'line_id': 'LEGACY-2025-0001', 'origin_year': 2024}, kind='case')

    def test_scope_mismatch_unknown_tenant_and_injected_identifier_rejected(self):
        for target, kwargs in [
            ({'line_id': 'L001', 'supplier_id': 'SUP-B'}, {'kind': 'case'}),
            ({'line_id': 'L001', 'contract_id': 'CTR-B'}, {'kind': 'case'}),
            ({'supplier_id': 'SUP-A', 'contract_id': 'CTR-B'}, {}),
            ({'line_id': 'L001'}, {'kind': 'case', 'tenant': 'OTHER'}),
            ({'supplier_id': "SUP-A' OR 1=1 --"}, {}),
        ]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                resolve_chat_target(self.db, target, **kwargs)

    def test_malformed_links_fail_without_creating_database(self):
        for target in (None, [], {}, {'supplier_id': []}, {'supplier_id': 'SUP-A', 'origin_year': True}):
            with self.subTest(target=target), self.assertRaises(ValueError):
                resolve_chat_target(self.db, target)
        missing = self.db.with_name('absent.sqlite3')
        with self.assertRaises(Exception):
            resolve_chat_target(missing, {'supplier_id': 'SUP-A'})
        self.assertFalse(missing.exists())

    def test_handoff_is_scoped_consumed_once_and_selects_locked_cohort(self):
        st = FakeState()
        key = f'chatbot_open_case:{self.db.resolve()}:DEMO'
        st.session_state[key] = {'line_id': 'LEGACY-2025-0001'}
        before = self.state()
        target = consume_chat_target(st, self.db, kind='case', namespace='invoice')
        self.assertEqual(target['origin_year'], 2025)
        self.assertEqual(st.session_state[f'invoice_cohort:{self.db.resolve()}'],
                         'Migrated · 2025 (locked)')
        self.assertNotIn(key, st.session_state)
        self.assertEqual(consume_chat_target(st, self.db, kind='case', namespace='invoice'), target)
        self.assertEqual(self.state(), before)

    def test_invalid_handoff_cannot_reuse_previous_target(self):
        st = FakeState()
        key = f'chatbot_open_contract:{self.db.resolve()}:DEMO'
        st.session_state[key] = {'supplier_id': 'SUP-A'}
        self.assertIsNotNone(consume_chat_target(st, self.db))
        st.session_state[key] = {'supplier_id': 'UNKNOWN'}
        self.assertIsNone(consume_chat_target(st, self.db))
        self.assertEqual(len(st.warnings), 1)


if __name__ == '__main__':
    unittest.main()
