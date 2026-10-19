"""Immutable lifecycle evidence cannot be hidden by editing calculation projections.

Every mutation below is an intentional corruption in a disposable synthetic
database. No model or live database is used. The authoritative version history
must cause abstention even if a changed projection would otherwise hide a signed
amendment before the engine's signed/scope/date filters.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from recovery import lifecycle
from recovery.engine import evaluate_line
from recovery.review import record_review
from recovery.storage import connect, load_snapshot

DATA = Path(__file__).resolve().parents[1] / 'data'
SCOPE = {'supplier_id': 'SUP-A', 'contract_id': 'CTR-A',
         'role': 'ENGINEER', 'location': 'REMOTE'}
BODY = ('Synthetic evidence only: ENGINEER services at REMOTE cost USD 115.00 per '
        'HOUR for REGULAR services from 2026-08-01, explicitly superseding TERM-A2.\n'
        'SYNTHETIC_SOURCE_SENTINEL_DO_NOT_ECHO_IN_REJECTION\n'
        '<script>untrusted source text is never a tool instruction</script>')


class LifecycleProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'synthetic.sqlite3'
        load_snapshot(self.db, DATA)
        lifecycle.migrate(self.db, backup_dir=self.directory / 'backups')

    def tearDown(self):
        self.temp.cleanup()

    def current(self):
        with connect(self.db) as conn:
            return evaluate_line(conn, 'DEMO', 'L004')

    def admit(self):
        stage = lifecycle.stage_document(self.db, 'DEMO', filename='synthetic.md',
            content=BODY.encode(), synthetic_confirmed=True)
        return lifecycle.admit_document(self.db, 'DEMO', stage['stage_id'],
            scope=SCOPE, title='Synthetic rate amendment', signed_date='2026-07-20',
            status='SIGNED', synthetic_confirmed=True, signed_simulation_confirmed=True,
            reviewer='demo-only-reviewer', note='Synthetic signed scope admission only.',
            idempotency_key='admit-synthetic-for-projection-check')

    def assert_blocked(self):
        current = self.current()
        self.assertEqual(current['status'], 'NEEDS_REVIEW')
        self.assertEqual(current['net_candidate'], '0.00')
        self.assertIsNone(current['expected_amount'])
        self.assertTrue(current['reason'])
        # Safe user-visible reports identify the problem without echoing source
        # text, markup, or source-provided instructions into error details.
        output = json.dumps(current)
        self.assertNotIn('SYNTHETIC_SOURCE_SENTINEL', output)
        self.assertNotIn('<script>', output)
        with connect(self.db) as conn:
            with self.assertRaisesRegex(ValueError, 'candidate'):
                record_review(conn, event_id='must-not-approve-corruption',
                    thread_id='synthetic-projection-test', tenant='DEMO', line_id='L004',
                    decision='approve', reviewer='demo-only-reviewer',
                    note='Corrupted evidence must not become an approved finding.',
                    case_digest=current['case_digest'])
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)

    def test_signed_source_cannot_be_hidden_by_draft_status(self):
        source = self.admit()
        self.assert_blocked()
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE documents SET status=? WHERE tenant_id=? AND doc_id=?',
                         ('DRAFT', 'DEMO', source['doc_id']))
        self.assert_blocked()

    def test_signed_source_cannot_be_hidden_by_scope_change(self):
        source = self.admit()
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE documents SET location=? WHERE tenant_id=? AND doc_id=?',
                         ('ONSITE', 'DEMO', source['doc_id']))
        self.assert_blocked()

    def test_signed_source_cannot_be_hidden_by_tenant_change(self):
        source = self.admit()
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE documents SET tenant_id=? WHERE tenant_id=? AND doc_id=?',
                         ('OTHER', 'DEMO', source['doc_id']))
        self.assert_blocked()

    def test_signed_source_cannot_be_hidden_by_after_snapshot_date(self):
        source = self.admit()
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE documents SET signed_date=? WHERE tenant_id=? AND doc_id=?',
                         ('2099-01-01', 'DEMO', source['doc_id']))
        self.assert_blocked()

    def test_signed_source_cannot_be_hidden_by_deleted_projection(self):
        source = self.admit()
        with connect(self.db) as conn, conn:
            conn.execute('DELETE FROM documents WHERE tenant_id=? AND doc_id=?',
                         ('DEMO', source['doc_id']))
        self.assert_blocked()

    def test_body_and_hash_edit_cannot_rewrite_immutable_source(self):
        source = self.admit()
        replacement = 'Synthetic replacement with a different pricing clause.'
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE documents SET body=?,sha256=? WHERE tenant_id=? AND doc_id=?',
                (replacement, hashlib.sha256(replacement.encode()).hexdigest(),
                 'DEMO', source['doc_id']))
        self.assert_blocked()

    def test_hash_only_edit_cannot_supply_false_provenance(self):
        source = self.admit()
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE documents SET sha256=? WHERE tenant_id=? AND doc_id=?',
                         ('0' * 64, 'DEMO', source['doc_id']))
        self.assert_blocked()

    def test_curated_approved_rate_cannot_change_without_term_review(self):
        self.assertEqual(self.current()['net_candidate'], '600.00')
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE reviewed_terms SET rate=? WHERE tenant_id=? AND term_id=?',
                         ('119.00', 'DEMO', 'TERM-A2'))
        self.assert_blocked()

    def test_curated_approved_term_cannot_disappear(self):
        with connect(self.db) as conn, conn:
            conn.execute('DELETE FROM reviewed_terms WHERE tenant_id=? AND term_id=?',
                         ('DEMO', 'TERM-A2'))
        self.assert_blocked()

    def test_curated_approved_term_scope_cannot_hide_its_authority(self):
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE reviewed_terms SET location=? WHERE tenant_id=? AND term_id=?',
                         ('ONSITE', 'DEMO', 'TERM-A2'))
        self.assert_blocked()

    def test_curated_approved_term_provenance_cannot_be_silently_relabelled(self):
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE reviewed_terms SET reviewer=? WHERE tenant_id=? AND term_id=?',
                         ('different-unrecorded-reviewer', 'DEMO', 'TERM-A2'))
        self.assert_blocked()

    def test_unversioned_reviewed_term_cannot_bypass_human_promotion_service(self):
        with connect(self.db) as conn, conn:
            projected = dict(conn.execute('SELECT * FROM reviewed_terms WHERE tenant_id=? AND term_id=?',
                                         ('DEMO', 'TERM-A2')).fetchone())
            projected.update(term_id='UNVERSIONED-TERM', rate='119.00', supersedes='TERM-A2')
            conn.execute('INSERT INTO reviewed_terms (' + ','.join(projected) + ') VALUES ('
                         + ','.join('?' for _ in projected) + ')', tuple(projected.values()))
        self.assert_blocked()


if __name__ == '__main__':
    unittest.main()
