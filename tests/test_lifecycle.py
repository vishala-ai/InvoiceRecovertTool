"""Step 9 lifecycle regressions using isolated synthetic databases only.

All extraction is mocked. The shipped fixture directory and working databases are
never edited; the 800 -> 1300 acceptance scenario is a copied synthetic dataset.
"""
from __future__ import annotations

import copy
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recovery import lifecycle
from recovery.dashboard import summarize_cases
from recovery.engine import audit, evaluate_line
from recovery.review import record_review
from recovery.storage import connect, load_snapshot

DATA = Path(__file__).resolve().parents[1] / 'data'
SCOPE = {'supplier_id': 'SUP-A', 'contract_id': 'CTR-A',
         'role': 'ENGINEER', 'location': 'REMOTE'}
QUOTE_120 = ('ENGINEER services at REMOTE are billed in USD at 120.00 per HOUR '
             'for approved REGULAR hours, effective 2026-07-01.')
QUOTE_115 = ('ENGINEER services at REMOTE are billed in USD at 115.00 per HOUR '
             'for approved REGULAR hours, effective 2026-08-01. '
             'This explicitly supersedes TERM-A2 for this same scope.')
try:
    GRAPH_AVAILABLE = importlib.util.find_spec('langgraph.checkpoint.sqlite') is not None
except ModuleNotFoundError:
    GRAPH_AVAILABLE = False


def values(*, rate='120.00', effective_from='2026-07-01', supersedes='TERM-A1',
           quote=QUOTE_120):
    return {'role': 'ENGINEER', 'location': 'REMOTE', 'currency': 'USD',
            'unit': 'HOUR', 'service_type': 'REGULAR', 'rate': rate,
            'effective_from': effective_from, 'effective_to': None,
            'supersedes': supersedes, 'exact_quote': quote,
            'unresolved_questions': [], 'coverage': 'COMPLETE_SINGLE_RATE'}


def extraction(source, proposal_values):
    original = copy.deepcopy(proposal_values)
    return {'status': 'UNREVIEWED_PROPOSAL', 'doc_id': source['doc_id'],
            'source_sha256': source['sha256'], 'trusted_scope': {
                field: source[field] for field in ('tenant_id', *SCOPE)},
            'proposal': copy.deepcopy(original), 'original_output': original,
            'raw_output': json.dumps(original), 'provider': 'ollama',
            'model': 'synthetic-local:1b', 'model_digest': 'a' * 64,
            'prompt_version': 'supplier-term-extraction-v1',
            'base_url': 'http://127.0.0.1:11434', 'context_tokens': 16384,
            'extracted_at': '2026-10-17T12:00:00+00:00', 'usage': {}}


@unittest.skipUnless(importlib.util.find_spec('pydantic') is not None,
                     'Install ollama extra for persisted term-proposal verification')
class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'synthetic.sqlite3'
        self.backups = self.directory / 'backups'
        self.key_number = 0
        load_snapshot(self.db, DATA)
        lifecycle.migrate(self.db, backup_dir=self.backups)
        self.baseline_sources = lifecycle.list_sources(self.db, 'DEMO')

    def tearDown(self):
        self.temp.cleanup()

    def key(self, kind='test'):
        self.key_number += 1
        return f'{kind}:{self.key_number}'

    def case(self, line='L004'):
        with connect(self.db) as conn:
            return evaluate_line(conn, 'DEMO', line)

    def report(self):
        with connect(self.db) as conn:
            return audit(conn)

    def dashboard_summary(self, *, history=(), workflows=()):
        with connect(self.db) as conn:
            report = audit(conn)
            events = [dict(row) for row in conn.execute('SELECT * FROM review_events')]
        return summarize_cases([*history, *report['cases']], events=events, workflows=workflows,
            assessments=lifecycle.list_assessments(self.db, 'DEMO'),
            requests=lifecycle.list_requests(self.db, 'DEMO'))

    def terms(self):
        with connect(self.db) as conn:
            return [dict(row) for row in conn.execute(
                'SELECT * FROM reviewed_terms ORDER BY tenant_id,term_id')]

    def additive_sources(self):
        return [source for source in lifecycle.list_sources(self.db, 'DEMO')
                if source['origin'] != 'CURATED_FIXTURE']

    def database_snapshot(self, db):
        with connect(db) as conn:
            names = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            return {name: sorted([tuple(row) for row in conn.execute(
                'SELECT * FROM "' + name.replace('"', '""') + '"')], key=repr) for name in names}

    def generate(self, doc_id='DOC-A2', proposal_values=None, *, tenant='DEMO'):
        proposed = values() if proposal_values is None else proposal_values
        with patch('recovery.term_extraction.extract_terms',
                   side_effect=lambda source, **_kwargs: extraction(source, proposed)):
            return lifecycle.generate_proposal(self.db, tenant, doc_id,
                                                model='synthetic-local:1b')

    def approve(self, proposal, **kwargs):
        parameters = dict(proposal_digest=proposal['digest'], decision='approve',
            reviewer='synthetic-reviewer', note='Reviewed source, scope, dates and explicit supersession.',
            idempotency_key=self.key('term-review'), coverage_confirmed=True)
        parameters.update(kwargs)
        return lifecycle.review_terms(self.db, 'DEMO', proposal['proposal_id'], **parameters)

    def stage(self, body=QUOTE_115, *, filename='synthetic-amendment.md', tenant='DEMO'):
        return lifecycle.stage_document(self.db, tenant, filename=filename,
            content=body.encode('utf-8'), synthetic_confirmed=True)

    def admit(self, body=QUOTE_115, *, status='SIGNED', scope=None, replaces_doc_id=None,
              stage=None, idempotency_key=None):
        staged = self.stage(body) if stage is None else stage
        return lifecycle.admit_document(self.db, 'DEMO', staged['stage_id'],
            scope=SCOPE if scope is None else scope, title='Synthetic explicit rate amendment',
            signed_date='2026-07-20' if status == 'SIGNED' else '', status=status,
            synthetic_confirmed=True, signed_simulation_confirmed=status == 'SIGNED',
            reviewer='synthetic-reviewer',
            note='Reviewed synthetic status and exact known invoice scope.',
            idempotency_key=idempotency_key or self.key('admit'),
            replaces_doc_id=replaces_doc_id)

    def new_proposal(self, **kwargs):
        source = self.admit(**kwargs)
        proposed = values(rate='115.00', effective_from='2026-08-01',
                          supersedes='TERM-A2', quote=QUOTE_115)
        return source, self.generate(source['doc_id'], proposed)

    def test_migration_preserves_original_twenty_case_financial_baseline(self):
        with connect(self.db) as conn:
            result = audit(conn)
        self.assertEqual(result['case_count'], 20)
        self.assertEqual(result['counts'], {'CANDIDATE': 4, 'NO_DISCREPANCY': 4,
            'ALREADY_OFFSET': 2, 'NEEDS_REVIEW': 9, 'UNPAID': 1})
        self.assertEqual(result['candidate_total'], '80707.50')
        self.assertEqual(result['realized_recovery'], '0.00')

    def test_old_schema_migration_backup_and_history_survive_restart(self):
        old_db = self.directory / 'old-schema.sqlite3'
        load_snapshot(old_db, DATA)
        with connect(old_db) as conn:
            old_case = evaluate_line(conn, 'DEMO', 'L001')
            event = record_review(conn, event_id='legacy-review', thread_id='legacy-thread',
                tenant='DEMO', line_id='L001', decision='reject', reviewer='synthetic-reviewer',
                note='Legacy immutable synthetic decision.', case_digest=old_case['case_digest'])
        original_tables = self.database_snapshot(old_db)
        backup_dir = self.directory / 'old-backups'
        lifecycle.migrate(old_db, backup_dir=backup_dir)
        lifecycle.migrate(old_db, backup_dir=backup_dir)
        backups = list(backup_dir.glob('*.sqlite3'))
        self.assertTrue(backups, 'Migration must create a verified recoverable database backup')
        self.assertEqual(self.database_snapshot(backups[0]), original_tables)
        migrated_tables = self.database_snapshot(old_db)
        self.assertEqual({name: migrated_tables[name] for name in original_tables}, original_tables)
        for path in (old_db, backups[0]):
            with connect(path) as conn:
                self.assertEqual(dict(conn.execute('SELECT * FROM review_events WHERE event_id=?',
                                                  ('legacy-review',)).fetchone()), event)
                self.assertEqual(audit(conn)['candidate_total'], '80707.50')
                self.assertEqual(conn.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
        self.assertEqual(lifecycle.list_proposals(old_db, 'DEMO'), [])

    def test_failed_migration_rolls_back_schema_and_retains_verified_backup(self):
        old_db = self.directory / 'failed-migration.sqlite3'
        load_snapshot(old_db, DATA)
        with connect(old_db) as conn:
            schema_before = list(conn.execute('SELECT name,sql FROM sqlite_master ORDER BY name'))
            schema_before = [tuple(row) for row in schema_before]
        backup_dir = self.directory / 'failed-migration-backups'
        with patch('recovery.lifecycle_storage.save_assessment',
                   side_effect=RuntimeError('Injected assessment persistence failure')):
            with self.assertRaisesRegex(RuntimeError, 'Injected'):
                lifecycle.migrate(old_db, backup_dir=backup_dir)
        with connect(old_db) as conn:
            self.assertEqual([tuple(row) for row in conn.execute(
                'SELECT name,sql FROM sqlite_master ORDER BY name')], schema_before)
            self.assertEqual(audit(conn)['candidate_total'], '80707.50')
        backups = list(backup_dir.glob('*.sqlite3'))
        self.assertTrue(backups)
        with connect(backups[0]) as conn:
            self.assertEqual(conn.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(audit(conn)['candidate_total'], '80707.50')
        self.assertFalse(lifecycle.enabled(old_db))

    def test_proposal_persists_without_changing_engine_and_has_immutable_original(self):
        before, terms = self.case(), self.terms()
        unknown = values()
        unknown.update(rate=None, coverage='INCOMPLETE',
                       unresolved_questions=['Confirm rate and all exceptions.'])
        generated = self.generate(proposal_values=unknown)
        reloaded = lifecycle.list_proposals(self.db, 'DEMO')[0]
        self.assertEqual(reloaded['proposal_id'], generated['proposal_id'])
        self.assertEqual(reloaded['status'], 'UNREVIEWED_PROPOSAL')
        self.assertEqual(reloaded['values'], unknown)
        self.assertEqual(self.terms(), terms)
        self.assertEqual(self.case(), before)
        edited = lifecycle.edit_proposal(self.db, 'DEMO', reloaded['proposal_id'],
            values=values(), reviewer='synthetic-editor', note='Confirmed the source rate.',
            idempotency_key=self.key('edit'))
        original = next(item for item in lifecycle.list_proposals(self.db, 'DEMO')
                        if item['proposal_id'] == generated['proposal_id'])
        self.assertEqual(original['values'], unknown)
        self.assertEqual(original['original_output'], unknown)
        self.assertEqual(edited['original_output'], original['original_output'])
        self.assertNotEqual(edited['proposal_id'], generated['proposal_id'])
        self.assertGreater(edited['revision'], generated['revision'])
        self.assertEqual(self.terms(), terms)

    def test_fixture_term_reapproval_is_idempotent_without_duplicate_projection(self):
        proposal = self.generate()
        before = self.terms()
        first = self.approve(proposal, idempotency_key='fixture-approval')
        repeated = self.approve(proposal, idempotency_key='fixture-approval')
        self.assertEqual(first, repeated)
        self.assertEqual(self.terms(), before)
        self.assertEqual(lifecycle.list_proposals(self.db, 'DEMO')[0]['status'], 'TERMS_APPROVED')
        self.assertEqual(self.case()['net_candidate'], '600.00')
        with self.assertRaises(ValueError):
            self.approve(proposal, idempotency_key='fixture-approval', note='Conflicting payload')

    def test_quoted_human_term_approval_reaches_engine_and_preserves_service_dates(self):
        with connect(self.db) as conn, conn:
            conn.execute("UPDATE service_entries SET service_date='2026-08-10' WHERE line_id='L004'")
        _, proposal = self.new_proposal()
        self.assertEqual(self.case()['status'], 'NEEDS_REVIEW')
        self.approve(proposal)
        current = self.case()
        self.assertEqual(current['expected_amount'], '11500.00')
        self.assertEqual(current['net_candidate'], '1100.00')
        self.assertEqual(self.report()['realized_recovery'], '0.00')
        # June/July dates cannot inherit the newly approved August rate.
        split = self.case('L011')
        self.assertEqual(split['expected_amount'], '2500.00')
        self.assertEqual(split['net_candidate'], '100.00')

    def test_failed_promotion_rolls_back_projection_version_event_and_key(self):
        _, proposal = self.new_proposal()
        before_terms = self.terms()
        before_sources = lifecycle.list_sources(self.db, 'DEMO')
        with connect(self.db) as conn, conn:
            versions_before = conn.execute('SELECT count(*) FROM approved_term_versions').fetchone()[0]
            events_before = conn.execute('SELECT count(*) FROM term_review_events').fetchone()[0]
            conn.execute('''CREATE TRIGGER inject_failed_term_review BEFORE INSERT ON term_review_events
                BEGIN SELECT RAISE(ABORT, 'injected review event failure'); END''')
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'injected'):
            self.approve(proposal, idempotency_key='atomic-term-approval')
        self.assertEqual(self.terms(), before_terms)
        self.assertEqual(lifecycle.list_sources(self.db, 'DEMO'), before_sources)
        self.assertEqual(lifecycle.list_proposals(self.db, 'DEMO')[0]['status'], 'UNREVIEWED_PROPOSAL')
        with connect(self.db) as conn, conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM approved_term_versions').fetchone()[0],
                             versions_before)
            self.assertEqual(conn.execute('SELECT count(*) FROM term_review_events').fetchone()[0],
                             events_before)
            self.assertFalse(conn.execute('SELECT 1 FROM operation_keys WHERE operation_key=?',
                                          ('atomic-term-approval',)).fetchone())
            conn.execute('DROP TRIGGER inject_failed_term_review')
        self.assertEqual(self.approve(proposal, idempotency_key='atomic-term-approval')['decision'],
                         'TERMS_APPROVED')

    def test_concurrent_approvals_serialize_and_reject_the_stale_proposal(self):
        source, first = self.new_proposal()
        second = self.generate(source['doc_id'], values(rate='115.00',
            effective_from='2026-08-01', supersedes='TERM-A2', quote=QUOTE_115))
        before_terms = self.terms()

        def approve_once(proposal, key):
            try:
                return self.approve(proposal, idempotency_key=key)
            except ValueError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(approve_once, first, 'concurrent-one'),
                       pool.submit(approve_once, second, 'concurrent-two')]
            outcomes = [future.result() for future in pending]
        self.assertEqual(sum(isinstance(outcome, dict) for outcome in outcomes), 1)
        self.assertEqual(sum(isinstance(outcome, ValueError) for outcome in outcomes), 1)
        self.assertEqual(len(self.terms()), len(before_terms) + 1)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM term_review_events').fetchone()[0], 1)
        proposals = lifecycle.list_proposals(self.db, 'DEMO')
        self.assertEqual({proposal['status'] for proposal in proposals},
                         {'TERMS_APPROVED', 'PROPOSAL_STALE'})

    def test_invalid_fields_cannot_be_promoted(self):
        original = self.generate()
        terms = self.terms()
        mutations = [{'rate': 'NaN'}, {'rate': 'Infinity'}, {'rate': '-1'}, {'rate': '1e1000'},
                     {'rate': 120.0}, {'effective_from': '2026-02-30'},
                     {'effective_to': '2025-01-01'}, {'unit': 'DAY'}, {'currency': 'EUR'},
                     {'currency': 'US'}, {'service_type': 'OVERTIME'}, {'role': 'ANALYST'},
                     {'location': 'ONSITE'}, {'supersedes': 'TERM-C1'},
                     {'supersedes': 'NO-SUCH-TERM'}, {'supersedes': 'TERM-A2'},
                     {'coverage': 'INCOMPLETE'}, {'rate': None},
                     {'unresolved_questions': ['Scope not established.']},
                     {'exact_quote': 'The model invented this quotation.'},
                     {'tenant_id': 'OTHER'}, {'supplier_id': 'SUP-B'}]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                edited = lifecycle.edit_proposal(self.db, 'DEMO', original['proposal_id'],
                    values={**values(), **change}, reviewer='synthetic-editor',
                    note='Test unsupported correction.', idempotency_key=self.key('invalid-edit'))
                self.approve(edited)
            self.assertEqual(self.terms(), terms)

    def test_missing_human_note_coverage_or_current_digest_blocks_approval(self):
        proposal = self.generate()
        for change in ({'note': ''}, {'reviewer': ''}, {'coverage_confirmed': False},
                       {'proposal_digest': 'b' * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.approve(proposal, **change)
        self.assertEqual(lifecycle.list_proposals(self.db, 'DEMO')[0]['status'],
                         'UNREVIEWED_PROPOSAL')

    def test_new_signed_scope_evidence_stales_existing_proposal_and_blocks_old_case_approval(self):
        proposal = self.generate()
        previous = self.case()
        self.admit()
        with self.assertRaises(ValueError):
            self.approve(proposal)
        stale = next(item for item in lifecycle.list_proposals(self.db, 'DEMO')
                     if item['proposal_id'] == proposal['proposal_id'])
        self.assertEqual(stale['status'], 'PROPOSAL_STALE')
        self.assertEqual(stale['source_sha256'], proposal['source_sha256'])
        with connect(self.db) as conn:
            with self.assertRaisesRegex(ValueError, 'STALE'):
                record_review(conn, event_id='stale-case', thread_id='stale-thread',
                    tenant='DEMO', line_id='L004', decision='approve', reviewer='synthetic-reviewer',
                    note='Attempt old approval.', case_digest=previous['case_digest'])

    def test_late_model_response_cannot_become_current_proposal(self):
        def late_response(source, **_kwargs):
            self.admit()
            return extraction(source, values())
        with patch('recovery.term_extraction.extract_terms', side_effect=late_response):
            try:
                result = lifecycle.generate_proposal(self.db, 'DEMO', 'DOC-A2',
                                                     model='synthetic-local:1b')
            except ValueError:
                return
        self.assertEqual(result['status'], 'PROPOSAL_STALE')

    def test_changed_source_projection_is_not_approvable_as_original_version(self):
        _, proposal = self.new_proposal()
        before_terms = self.terms()
        changed = QUOTE_115 + '\nChanged after extraction.'
        with connect(self.db) as conn, conn:
            conn.execute('UPDATE documents SET body=?,sha256=? WHERE tenant_id=? AND doc_id=?',
                (changed, hashlib.sha256(changed.encode()).hexdigest(), 'DEMO', proposal['doc_id']))
        with self.assertRaises(ValueError):
            self.approve(proposal)
        self.assertEqual(self.terms(), before_terms)
        self.assertEqual(lifecycle.list_proposals(self.db, 'DEMO')[0]['status'], 'PROPOSAL_STALE')

    def test_wrong_tenant_document_and_proposal_cannot_be_reviewed(self):
        with patch('recovery.term_extraction.extract_terms') as model:
            with self.assertRaises(ValueError):
                lifecycle.generate_proposal(self.db, 'OTHER', 'DOC-A2', model='synthetic-local:1b')
        model.assert_not_called()
        proposal = self.generate()
        with self.assertRaises(ValueError):
            lifecycle.review_terms(self.db, 'OTHER', proposal['proposal_id'],
                proposal_digest=proposal['digest'], decision='approve', reviewer='synthetic-reviewer',
                note='Wrong tenant cannot approve.', idempotency_key='wrong-tenant',
                coverage_confirmed=True)

    def test_unsigned_source_and_incomplete_coverage_never_control_rates(self):
        _, proposal = self.new_proposal(status='DRAFT')
        self.assertEqual(self.case()['net_candidate'], '600.00')
        with self.assertRaises(ValueError):
            self.approve(proposal)
        self.assertEqual(self.case()['net_candidate'], '600.00')

    def test_rejecting_extraction_does_not_resolve_signed_source(self):
        _, proposal = self.new_proposal()
        self.approve(proposal, decision='reject', coverage_confirmed=False)
        self.assertEqual(lifecycle.list_proposals(self.db, 'DEMO')[0]['status'], 'TERMS_REJECTED')
        current = self.case()
        self.assertEqual(current['status'], 'NEEDS_REVIEW')
        self.assertEqual(current['net_candidate'], '0.00')
        self.assertTrue(lifecycle.list_sources(self.db, 'DEMO'))

    def test_upload_validation_rejects_without_changing_source_or_assessment(self):
        before = self.case()
        examples = [{'filename': 'bad.pdf', 'content': b'synthetic text'},
                    {'filename': 'empty.md', 'content': b'  \n'},
                    {'filename': 'binary.md', 'content': b'hello\x00world'},
                    {'filename': 'encoding.txt', 'content': b'\xff\xfe'},
                    {'filename': 'large.md', 'content': b'x' * 262145},
                    {'filename': 'fixture.md', 'content': b'synthetic',
                     'synthetic_confirmed': False}]
        for arguments in examples:
            parameters = {'synthetic_confirmed': True, **arguments}
            with self.subTest(arguments={key: value for key, value in parameters.items()
                                         if key != 'content'}), self.assertRaises(ValueError):
                lifecycle.stage_document(self.db, 'DEMO', **parameters)
        self.assertEqual(lifecycle.list_sources(self.db, 'DEMO'), self.baseline_sources)
        self.assertEqual(self.case(), before)

    def test_supplied_filename_is_metadata_and_never_used_as_a_storage_path(self):
        filename = '../synthetic-fixture-outside.md'
        outside_path = self.directory.parent / 'synthetic-fixture-outside.md'
        self.assertFalse(outside_path.exists())
        staged = self.stage(filename=filename)
        source = self.admit(stage=staged)
        self.assertEqual(source['original_filename'], filename)
        self.assertFalse(outside_path.exists())
        document = next(item for item in lifecycle.list_documents(self.db, 'DEMO')
                        if item['doc_id'] == source['doc_id'])
        self.assertTrue(document['path'].startswith('evidence/source-'))
        self.assertNotIn('..', document['path'])

    def test_duplicate_admission_is_idempotent_and_conflicting_key_rejected(self):
        stage = self.stage()
        first = self.admit(stage=stage, idempotency_key='same-admission')
        repeated = self.admit(stage=stage, idempotency_key='same-admission')
        self.assertEqual(first, repeated)
        self.assertEqual(len(self.additive_sources()), 1)
        with self.assertRaises(ValueError):
            self.admit(stage=stage, status='DRAFT', idempotency_key='same-admission')
        try:
            duplicate = self.admit(stage=self.stage(), idempotency_key='duplicate-bytes')
        except ValueError:
            pass  # A clearly rejected duplicate is also a valid detected-duplicate policy.
        else:
            self.assertEqual(duplicate['doc_id'], first['doc_id'])
        self.assertEqual(len(self.additive_sources()), 1)

    def test_source_versions_preserve_prior_bytes_and_index_incrementally(self):
        first = self.admit()
        second_body = QUOTE_115 + '\nAdditional synthetic clause: versionmarkeromega.'
        second = self.admit(second_body, replaces_doc_id=first['doc_id'])
        sources = self.additive_sources()
        self.assertEqual(len(sources), 2)
        self.assertNotEqual(first['version_id'], second['version_id'])
        stored_first = next(source for source in sources if source['version_id'] == first['version_id'])
        self.assertEqual(stored_first['body'], QUOTE_115)
        self.assertEqual(stored_first['sha256'], hashlib.sha256(QUOTE_115.encode()).hexdigest())
        with connect(self.db) as conn:
            matches = conn.execute("SELECT doc_id FROM document_fts WHERE document_fts MATCH ?",
                                   ('versionmarkeromega',)).fetchall()
            self.assertIn(second['doc_id'], {row[0] for row in matches})
            self.assertTrue(conn.execute("SELECT 1 FROM document_fts WHERE doc_id='DOC-A1'").fetchone())

    def test_bad_scope_or_unknown_tenant_admission_rolls_back(self):
        before = self.case()
        stage = self.stage()
        for scope in ({**SCOPE, 'supplier_id': 'SUP-NO-SUCH'},
                      {**SCOPE, 'contract_id': 'NO-SUCH'}, {**SCOPE, 'role': 'ANALYST'},
                      {**SCOPE, 'location': 'ONSITE'}):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                self.admit(stage=stage, scope=scope)
        with self.assertRaises(ValueError):
            lifecycle.stage_document(self.db, 'NO-SUCH', filename='synthetic.md',
                                     content=b'Synthetic fixture.', synthetic_confirmed=True)
        self.assertEqual(lifecycle.list_sources(self.db, 'DEMO'), self.baseline_sources)
        self.assertEqual(self.case(), before)

    def test_fixture_initialization_cannot_erase_additive_evidence_or_terms(self):
        source, proposal = self.new_proposal()
        self.approve(proposal)
        before = self.case()
        with self.assertRaises(ValueError):
            load_snapshot(self.db, DATA)
        self.assertEqual(self.case(), before)
        self.assertTrue(any(item['doc_id'] == source['doc_id']
                            for item in lifecycle.list_sources(self.db, 'DEMO')))

    def test_same_amount_evidence_change_changes_digest_and_stales_assessment(self):
        assessment = lifecycle.reassess(self.db, 'DEMO', 'L004',
                                       idempotency_key='initial-assessment')
        previous = self.case()
        same_rate_quote = QUOTE_115.replace('115.00', '120.00')
        source = self.admit(same_rate_quote)
        generated = self.generate(source['doc_id'], values(rate='120.00',
            effective_from='2026-08-01', supersedes='TERM-A2', quote=same_rate_quote))
        self.approve(generated)
        current = self.case()
        self.assertEqual(current['net_candidate'], previous['net_candidate'])
        self.assertNotEqual(current['case_digest'], previous['case_digest'])
        history = lifecycle.list_assessments(self.db, 'DEMO', line_id='L004')
        self.assertEqual(next(item for item in history
                              if item['assessment_id'] == assessment['assessment_id'])['validity'],
                         'STALE')

    def test_request_more_is_traceable_idempotent_and_unsupported_type_stays_unresolved(self):
        current = self.case()
        arguments = dict(event_id='more-evidence', thread_id='original-review', tenant='DEMO',
            line_id='L004', decision='request_more', reviewer='synthetic-reviewer',
            note='Need another amendment.', case_digest=current['case_digest'])
        with connect(self.db) as conn:
            first = record_review(conn, **arguments)
            self.assertEqual(record_review(conn, **arguments), first)
        requests = lifecycle.list_requests(self.db, 'DEMO')
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['review_event_id'], first['event_id'])
        self.assertEqual(requests[0]['status'], 'OPEN')
        # Unsupported request kinds may be tracked, but not resolved by contract approval.
        with self.assertRaises(ValueError):
            lifecycle.resolve_request(self.db, 'DEMO', requests[0]['request_id'],
                reviewer='synthetic-reviewer', note='No accepted evidence attached.',
                idempotency_key='invalid-request-resolution')

    def test_unsupported_evidence_type_cannot_be_resolved_by_contract_terms(self):
        current = self.case()
        with connect(self.db) as conn:
            event = record_review(conn, event_id='payment-more', thread_id='payment-prior',
                tenant='DEMO', line_id='L004', decision='request_more', reviewer='synthetic-reviewer',
                note='Missing payment allocation evidence.', case_digest=current['case_digest'],
                evidence_type='PAYMENT')
        request = lifecycle.list_requests(self.db, 'DEMO')[0]
        self.assertEqual(request['review_event_id'], event['event_id'])
        self.assertEqual(request['request_type'], 'PAYMENT')
        source = next(source for source in lifecycle.list_sources(self.db, 'DEMO')
                      if source['doc_id'] == 'DOC-A2')
        lifecycle.attach_evidence(self.db, 'DEMO', request['request_id'], source['version_id'],
            reviewer='synthetic-reviewer', note='Attaching a rate clause does not prove payment.',
            idempotency_key='unsupported-attach')
        with self.assertRaisesRegex(ValueError, 'contract/rate'):
            lifecycle.resolve_request(self.db, 'DEMO', request['request_id'],
                reviewer='synthetic-reviewer', note='Payment still requires separate handling.',
                idempotency_key='unsupported-resolve')
        self.assertEqual(lifecycle.list_requests(self.db, 'DEMO')[0]['status'], 'EVIDENCE_ATTACHED')

    def test_evidence_attachment_cannot_cross_supplier_scope_or_tenant(self):
        current = self.case()
        with connect(self.db) as conn:
            record_review(conn, event_id='scoped-request', thread_id='scoped-prior', tenant='DEMO',
                line_id='L004', decision='request_more', reviewer='synthetic-reviewer',
                note='Need a later applicable amendment.', case_digest=current['case_digest'])
        request = lifecycle.list_requests(self.db, 'DEMO')[0]
        wrong_supplier = next(source for source in lifecycle.list_sources(self.db, 'DEMO')
                              if source['doc_id'] == 'DOC-B1')
        wrong_tenant = lifecycle.list_sources(self.db, 'OTHER')[0]
        for source in (wrong_supplier, wrong_tenant):
            with self.subTest(source=source['doc_id']), self.assertRaises(ValueError):
                lifecycle.attach_evidence(self.db, 'DEMO', request['request_id'], source['version_id'],
                    reviewer='synthetic-reviewer', note='This source cannot support the request.',
                    idempotency_key=self.key('wrong-attachment'))
        self.assertEqual(lifecycle.list_requests(self.db, 'DEMO')[0]['status'], 'OPEN')
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM evidence_request_events').fetchone()[0], 0)

    def acceptance_fixture(self):
        fixture = self.directory / 'acceptance-fixture'
        fixture.mkdir()
        manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
        manifest['dataset_id'] = 'synthetic-step9-acceptance-v1'
        manifest['description'] = 'Isolated synthetic single-invoice Step 9 acceptance dataset.'
        manifest['tenants'] = {'DEMO': manifest['tenants']['DEMO']}
        (fixture / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        selections = {
            'suppliers': lambda row: row['supplier_id'] == 'SUP-A',
            'documents': lambda row: row['doc_id'] in {'DOC-A1', 'DOC-A2'},
            'reviewed_terms': lambda row: row['term_id'] in {'TERM-A1', 'TERM-A2'},
            'purchase_orders': lambda row: row['po_id'] == 'PO-SUP-A',
            'invoice_lines': lambda row: row['line_id'] == 'L004',
            'service_entries': lambda row: row['line_id'] == 'L004',
            'payments': lambda row: row['line_id'] == 'L004',
            'adjustments': lambda row: row['line_id'] == 'L004',
        }
        for table, select in selections.items():
            with (DATA / f'{table}.csv').open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows = [row for row in rows if row['tenant_id'] == 'DEMO' and select(row)]
            for row in rows:
                if table == 'adjustments':
                    row['amount'] = '200.00'
                elif table == 'service_entries':
                    row['service_date'] = '2026-08-10'
                elif table == 'documents':
                    target = fixture / row['path']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(DATA / row['path'], target)
            with (fixture / f'{table}.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        database = self.directory / 'acceptance.sqlite3'
        load_snapshot(database, fixture)
        lifecycle.migrate(database, backup_dir=self.directory / 'acceptance-backups')
        return database

    def test_acceptance_800_to_1300_is_one_current_finding_with_preserved_history(self):
        baseline_db = self.db
        self.db = self.acceptance_fixture()
        initial = lifecycle.reassess(self.db, 'DEMO', 'L004', idempotency_key='initial')
        self.assertEqual(initial['case_json']['expected_amount'], '12000.00')
        self.assertEqual(initial['case_json']['net_candidate'], '800.00')
        self.assertEqual(self.report()['case_count'], 1)
        self.assertEqual(self.report()['candidate_total'], '800.00')
        self.assertEqual(self.dashboard_summary()['totals']['current_candidate'], '800.00')
        with connect(self.db) as conn:
            manifest = json.loads(conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
            self.assertEqual(manifest['dataset_id'], 'synthetic-step9-acceptance-v1')
            self.assertEqual(set(manifest['tenants']), {'DEMO'})
            self.assertEqual(manifest['as_of'], '2026-10-15')
        with connect(self.db) as conn:
            event = record_review(conn, event_id='acceptance-more', thread_id='acceptance-prior',
                tenant='DEMO', line_id='L004', decision='request_more', reviewer='synthetic-reviewer',
                note='Check whether a later applicable amendment exists.',
                case_digest=initial['case_json']['case_digest'], assessment_id=initial['assessment_id'])
        request = lifecycle.list_requests(self.db, 'DEMO')[0]
        source, proposal = self.new_proposal()
        self.assertEqual(self.case()['status'], 'NEEDS_REVIEW')
        lifecycle.attach_evidence(self.db, 'DEMO', request['request_id'], source['version_id'],
            reviewer='synthetic-reviewer', note='Attached the requested synthetic amendment.',
            idempotency_key='acceptance-attach')
        self.assertEqual(lifecycle.list_requests(self.db, 'DEMO')[0]['status'], 'EVIDENCE_ATTACHED')
        with self.assertRaises(ValueError):
            lifecycle.resolve_request(self.db, 'DEMO', request['request_id'],
                reviewer='synthetic-reviewer', note='Cannot resolve unreviewed terms.',
                idempotency_key='acceptance-premature-resolve')
        self.approve(proposal)
        lifecycle.resolve_request(self.db, 'DEMO', request['request_id'],
            reviewer='synthetic-reviewer', note='Reviewed quoted replacement rate and approved its terms.',
            idempotency_key='acceptance-resolve')
        revised = lifecycle.reassess(self.db, 'DEMO', 'L004',
            prior_review_event_id=event['event_id'], idempotency_key='acceptance-reassessment')
        self.assertEqual(revised['case_json']['expected_amount'], '11500.00')
        self.assertEqual(revised['case_json']['gross_discrepancy'], '1500.00')
        self.assertEqual(revised['case_json']['offsets'], '200.00')
        self.assertEqual(revised['case_json']['net_candidate'], '1300.00')
        self.assertEqual(self.report()['case_count'], 1)
        self.assertEqual(self.report()['candidate_total'], '1300.00')
        self.assertEqual(self.report()['realized_recovery'], '0.00')
        history = lifecycle.list_assessments(self.db, 'DEMO', line_id='L004')
        self.assertEqual(len([item for item in history if item['validity'] == 'CURRENT']), 1)
        self.assertEqual(revised['prior_assessment_id'], initial['assessment_id'])
        self.assertEqual(next(item for item in history
                              if item['assessment_id'] == initial['assessment_id'])['validity'], 'STALE')
        summary = self.dashboard_summary(history=(initial['case_json'], revised['case_json']))
        self.assertEqual(summary['case_count'], 1)
        self.assertEqual(summary['totals']['current_candidate'], '1300.00')
        self.assertNotEqual(summary['totals']['current_candidate'], '2100.00')
        with connect(self.db) as conn:
            self.assertEqual(dict(conn.execute('SELECT * FROM review_events WHERE event_id=?',
                                              (event['event_id'],)).fetchone()), event)
        with connect(baseline_db) as conn:
            baseline = audit(conn)
            self.assertEqual(baseline['case_count'], 20)
            self.assertEqual(baseline['candidate_total'], '80707.50')

    @unittest.skipUnless(GRAPH_AVAILABLE, 'Install graph extra for linked lifecycle integration')
    def test_acceptance_completed_request_thread_gets_fresh_linked_review_and_one_new_decision(self):
        from recovery.workflow import get_review_workflow, list_review_workflows, run_persistent

        self.db = self.acceptance_fixture()
        checkpoint = self.directory / 'acceptance-checkpoints.sqlite3'
        initial = lifecycle.reassess(self.db, 'DEMO', 'L004', idempotency_key='graph-initial')
        self.assertEqual(initial['case_json']['net_candidate'], '800.00')
        self.assertEqual(self.report()['case_count'], 1)
        self.assertEqual(self.report()['candidate_total'], '800.00')
        first = run_persistent(self.db, checkpoint, 'prior-completed-review', line_id='L004')
        self.assertTrue(first['paused'])
        previous = run_persistent(self.db, checkpoint, 'prior-completed-review', decision={
            'decision': 'request_more', 'reviewer': 'synthetic-reviewer',
            'note': 'Check whether a later applicable amendment exists.'})
        self.assertFalse(previous['paused'])
        self.assertEqual(previous['state']['outcome']['decision'], 'NEEDS_MORE_EVIDENCE')
        with self.assertRaises(ValueError):
            run_persistent(self.db, checkpoint, 'prior-completed-review', decision={
                'decision': 'approve', 'reviewer': 'synthetic-reviewer',
                'note': 'Completed threads cannot resume for new evidence.'})
        request = lifecycle.list_requests(self.db, 'DEMO')[0]
        source, proposal = self.new_proposal()
        lifecycle.attach_evidence(self.db, 'DEMO', request['request_id'], source['version_id'],
            reviewer='synthetic-reviewer', note='Attached later synthetic rate amendment.',
            idempotency_key='graph-attach')
        self.approve(proposal)
        lifecycle.resolve_request(self.db, 'DEMO', request['request_id'],
            reviewer='synthetic-reviewer', note='Reviewed replacement rate and applicability.',
            idempotency_key='graph-resolve')
        current = lifecycle.reassess(self.db, 'DEMO', 'L004',
            prior_review_event_id=previous['state']['outcome']['event_id'],
            idempotency_key='graph-reassess')
        self.assertEqual(current['case_json']['net_candidate'], '1300.00')
        self.assertEqual(self.report()['case_count'], 1)
        self.assertEqual(self.report()['candidate_total'], '1300.00')
        start_arguments = dict(prior_thread_id='prior-completed-review',
                               idempotency_key='new-linked-review', use_llm=False)
        started = lifecycle.start_linked_review(self.db, checkpoint, 'DEMO',
                                               current['assessment_id'], **start_arguments)
        self.assertTrue(started['paused'])
        self.assertNotEqual(started['thread_id'], 'prior-completed-review')
        self.assertEqual(started['state']['case']['net_candidate'], '1300.00')
        self.assertNotIn('outcome', started['state'])
        repeated = lifecycle.start_linked_review(self.db, checkpoint, 'DEMO',
                                                current['assessment_id'], **start_arguments)
        self.assertEqual(repeated, started)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT count(*) FROM review_intents').fetchone()[0], 1)
        saved = run_persistent(self.db, checkpoint, started['thread_id'], decision={
            'decision': 'approve', 'reviewer': 'synthetic-reviewer',
            'note': 'Reviewed fresh assessment, governing amendment and existing credit.'})
        self.assertEqual(saved['state']['outcome']['decision'], 'APPROVED_FOR_DRAFT')
        self.assertFalse(saved['paused'])
        replay = lifecycle.start_linked_review(self.db, checkpoint, 'DEMO',
                                              current['assessment_id'], **start_arguments)
        self.assertEqual(replay, saved)
        self.assertEqual(get_review_workflow(self.db, checkpoint, 'prior-completed-review'), previous)
        with connect(self.db) as conn:
            events = [dict(row) for row in conn.execute('SELECT * FROM review_events')]
            self.assertEqual(len(events), 2)
            self.assertEqual({event['decision'] for event in events},
                             {'NEEDS_MORE_EVIDENCE', 'APPROVED_FOR_DRAFT'})
        self.assertEqual(self.report()['realized_recovery'], '0.00')
        summary = self.dashboard_summary(history=(initial['case_json'], current['case_json']),
            workflows=list_review_workflows(self.db, checkpoint))
        self.assertEqual(summary['case_count'], 1)
        self.assertEqual(summary['totals']['current_candidate'], '1300.00')
        self.assertEqual(summary['totals']['approved_draft'], '1300.00')
        self.assertNotEqual(summary['totals']['current_candidate'], '2100.00')

    @unittest.skipUnless(GRAPH_AVAILABLE, 'Install graph extra for linked lifecycle integration')
    def test_interrupted_linked_start_reuses_durable_intent_after_restart(self):
        assessment = lifecycle.reassess(self.db, 'DEMO', 'L004', idempotency_key='intent-assessment')
        checkpoint = self.directory / 'intent-checkpoints.sqlite3'
        arguments = dict(idempotency_key='retry-linked-start', use_llm=False)
        with patch('recovery.workflow.run_persistent',
                   side_effect=RuntimeError('Injected failure after durable intent')):
            with self.assertRaisesRegex(RuntimeError, 'Injected'):
                lifecycle.start_linked_review(self.db, checkpoint, 'DEMO',
                                               assessment['assessment_id'], **arguments)
        with connect(self.db) as conn:
            intent = dict(conn.execute('SELECT * FROM review_intents').fetchone())
            self.assertEqual(intent['status'], 'PENDING')
        restarted = lifecycle.start_linked_review(self.db, checkpoint, 'DEMO',
                                                 assessment['assessment_id'], **arguments)
        self.assertEqual(restarted['thread_id'], intent['thread_id'])
        self.assertTrue(restarted['paused'])
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_intents').fetchone()[0], 1)
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 0)


class SourceProjectionIntegrityTests(unittest.TestCase):
    """These source-integrity checks also execute without optional AI packages."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.db = self.directory / 'source-integrity.sqlite3'
        load_snapshot(self.db, DATA)
        lifecycle.migrate(self.db)

    def case(self):
        with connect(self.db) as conn:
            return evaluate_line(conn, 'DEMO', 'L001')

    def admit(self, status):
        stage = lifecycle.stage_document(self.db, 'DEMO', filename='synthetic-integrity.md',
            content=QUOTE_115.encode('utf-8'), synthetic_confirmed=True)
        return lifecycle.admit_document(self.db, 'DEMO', stage['stage_id'], scope=SCOPE,
            title='Synthetic source projection integrity test',
            signed_date='2026-07-20' if status == 'SIGNED' else '', status=status,
            synthetic_confirmed=True, signed_simulation_confirmed=status == 'SIGNED',
            reviewer='synthetic-reviewer', note='Explicit synthetic admission for integrity test.',
            idempotency_key='source-integrity-admit:' + status)

    def test_kind_change_cannot_hide_original_governing_signed_amendment(self):
        before = self.case()
        self.assertEqual(before['status'], 'CANDIDATE')
        self.assertEqual(before['net_candidate'], '80000.00')
        with connect(self.db) as conn, conn:
            conn.execute("UPDATE documents SET kind='NOTE' WHERE tenant_id='DEMO' AND doc_id='DOC-A2'")
        after = self.case()
        self.assertEqual(after['status'], 'NEEDS_REVIEW')
        self.assertEqual(after['reason'], 'SOURCE_PROJECTION_MISMATCH')
        self.assertIsNone(after['expected_amount'])

    def test_original_notes_drafts_and_baseline_case_digest_are_preserved_read_only(self):
        from recovery.lifecycle_storage import validate_source_projections

        before = self.case()
        with connect(self.db) as conn:
            original = tuple(conn.iterdump())
            validate_source_projections(conn)
            self.assertEqual(tuple(conn.iterdump()), original)
            note = conn.execute("SELECT kind,status FROM documents WHERE doc_id='DOC-NOTE'").fetchone()
            self.assertEqual(tuple(note), ('NOTE', 'UNTRUSTED'))
            draft = conn.execute("SELECT kind,status FROM documents WHERE doc_id='DOC-ADRAFT'").fetchone()
            self.assertEqual(tuple(draft), ('RATE_AGREEMENT', 'DRAFT'))
            report = audit(conn)
        self.assertEqual(self.case(), before)
        self.assertEqual(report['candidate_total'], '80707.50')
        self.assertEqual(report['realized_recovery'], '0.00')

    def test_legitimate_admitted_signed_source_is_separate_from_original_import_digest(self):
        from recovery.lifecycle_storage import validate_source_projections

        self.admit('SIGNED')
        with connect(self.db) as conn:
            validate_source_projections(conn)
            self.assertEqual(conn.execute('SELECT count(*) FROM documents').fetchone()[0], 11)
        current = self.case()
        self.assertEqual(current['status'], 'NEEDS_REVIEW')
        self.assertEqual(current['reason'], 'UNREVIEWED_SIGNED_DOCUMENT')

    def test_admitted_source_kind_cannot_be_changed_to_hide_required_review(self):
        source = self.admit('SIGNED')
        with connect(self.db) as conn, conn:
            conn.execute("UPDATE documents SET kind='NOTE' WHERE tenant_id=? AND doc_id=?",
                         ('DEMO', source['doc_id']))
        current = self.case()
        self.assertEqual(current['status'], 'NEEDS_REVIEW')
        self.assertEqual(current['reason'], 'SOURCE_PROJECTION_MISMATCH')

    def test_legitimate_admitted_draft_keeps_original_financial_amount(self):
        from recovery.lifecycle_storage import validate_source_projections

        self.admit('DRAFT')
        with connect(self.db) as conn:
            validate_source_projections(conn)
        current = self.case()
        self.assertEqual(current['status'], 'CANDIDATE')
        self.assertEqual(current['net_candidate'], '80000.00')

    def test_missing_or_malformed_retained_reconciliation_fails_closed(self):
        with connect(self.db) as conn:
            original = conn.execute("SELECT value FROM metadata WHERE key='import_report'").fetchone()[0]
        missing = json.loads(original)
        del missing['reconciliation']['source']['row_digests']['documents']
        wrong_count = json.loads(original)
        wrong_count['reconciliation']['source']['row_counts']['documents'] = True
        malformed_digest = json.loads(original)
        malformed_digest['reconciliation']['source']['row_digests']['documents'] = 'not-a-source-hash'
        for report in ('{}', 'not JSON', json.dumps(missing), json.dumps(wrong_count),
                       json.dumps(malformed_digest)):
            with self.subTest(report_shape=report[:30]):
                with connect(self.db) as conn, conn:
                    conn.execute("UPDATE metadata SET value=? WHERE key='import_report'", (report,))
                current = self.case()
                self.assertEqual(current['status'], 'NEEDS_REVIEW')
                self.assertEqual(current['reason'], 'SOURCE_PROJECTION_MISMATCH')

    def test_original_document_path_corruption_is_not_repaired_or_followed(self):
        from recovery.lifecycle_storage import validate_source_projections

        with connect(self.db) as conn, conn:
            conn.execute("UPDATE documents SET path='contracts/incorrect.md' WHERE tenant_id='DEMO' AND doc_id='DOC-A2'")
        with connect(self.db) as conn, patch('builtins.open', side_effect=AssertionError('No source file reads')):
            with self.assertRaisesRegex(ValueError, 'SOURCE_PROJECTION_MISMATCH'):
                validate_source_projections(conn)
        self.assertEqual(self.case()['reason'], 'SOURCE_PROJECTION_MISMATCH')


if __name__ == '__main__':
    unittest.main()
