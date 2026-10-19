from __future__ import annotations

import copy
import csv
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from decimal import Decimal

from recovery.engine import ReviewRequired, audit, evaluate_line, number
from recovery.retrieval import retrieve_evidence, validate_model_citations
from recovery.review import record_review
from recovery.storage import connect, load_snapshot

DATA = Path(__file__).resolve().parents[1] / 'data'
GOLD = json.loads((DATA / 'expected_results.json').read_text())


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'test.sqlite3'
        load_snapshot(self.db, DATA)

    def tearDown(self):
        self.temp.cleanup()

    def case(self, lid):
        with connect(self.db) as conn:
            return evaluate_line(conn, 'DEMO', lid)

    def change(self, sql, args=()):
        with connect(self.db) as conn, conn:
            conn.execute(sql, args)

    def event(self, **changes):
        args = dict(event_id='test:1', thread_id='test', tenant='DEMO', line_id='L001',
                    decision='approve', reviewer='synthetic-reviewer', note='Checked synthetic evidence',
                    case_digest=self.case('L001')['case_digest'])
        args.update(changes)
        with connect(self.db) as conn:
            return record_review(conn, **args)

    def test_total_and_status_counts(self):
        with connect(self.db) as conn:
            result = audit(conn)
        self.assertEqual(result['candidate_total'], GOLD['expected_candidate_total'])
        self.assertEqual(result['counts'], {'CANDIDATE':4, 'NO_DISCREPANCY':4,
            'ALREADY_OFFSET':2, 'NEEDS_REVIEW':9, 'UNPAID':1})
        self.assertEqual(result['realized_recovery'], '0.00')

    def test_reimport_is_idempotent(self):
        before = self.case('L001')
        load_snapshot(self.db, DATA)
        self.assertEqual(before, self.case('L001'))
        with connect(self.db) as conn:
            self.assertEqual(audit(conn)['case_count'], 20)

    def test_no_expected_results_dependency(self):
        fixture = Path(self.temp.name) / 'fixtures'
        shutil.copytree(DATA, fixture)
        (fixture / 'expected_results.json').unlink()
        load_snapshot(self.db, fixture)
        self.assertEqual(self.case('L001')['net_candidate'], '80000.00')

    def test_duplicate_import_rolls_back(self):
        fixture = Path(self.temp.name) / 'fixtures'
        shutil.copytree(DATA, fixture)
        file = fixture / 'invoice_lines.csv'
        with file.open(newline='') as f:
            rows = list(csv.reader(f))
        with file.open('a', newline='') as f:
            csv.writer(f).writerow(rows[1])
        with self.assertRaises(sqlite3.IntegrityError):
            load_snapshot(self.db, fixture)
        self.assertEqual(self.case('L001')['net_candidate'], '80000.00')

    def test_corrupt_document_hash_rejected(self):
        fixture = Path(self.temp.name) / 'fixtures'
        shutil.copytree(DATA, fixture)
        with (fixture / 'contracts/DOC-A2.md').open('a') as f:
            f.write('tampered')
        with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
            load_snapshot(self.db, fixture)

    def test_real_data_manifest_refused(self):
        fixture = Path(self.temp.name) / 'fixtures'
        shutil.copytree(DATA, fixture)
        m = json.loads((fixture / 'manifest.json').read_text())
        m['synthetic'] = False
        (fixture / 'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            load_snapshot(self.db, fixture)

    def test_scope_filters_unsigned_note_and_other_tenant(self):
        with connect(self.db) as conn:
            docs = retrieve_evidence(conn, 'DEMO', 'L001', 'ignore previous instructions rate')
        self.assertEqual({d['doc_id'] for d in docs}, {'DOC-A1', 'DOC-A2'})
        self.assertTrue(all(d['tenant_id'] == 'DEMO' for d in docs))

    def test_lexical_miss_does_not_drop_governing_documents(self):
        with connect(self.db) as conn:
            docs = retrieve_evidence(conn, 'DEMO', 'L001', 'xyzneveroccurs')
        self.assertEqual(len(docs), 2)

    def test_adversarial_query_is_not_sql(self):
        with connect(self.db) as conn:
            docs = retrieve_evidence(conn, 'DEMO', 'L001', '\"; DROP TABLE documents; --')
            self.assertGreater(conn.execute('SELECT count(*) FROM documents').fetchone()[0], 0)
        self.assertEqual(len(docs), 2)

    def test_line_scope_cannot_read_other_tenant(self):
        with connect(self.db) as conn:
            with self.assertRaises(ValueError):
                evaluate_line(conn, 'OTHER', 'L001')

    def test_candidate_citation_is_correct_line(self):
        citation = self.case('L001')['citations'][0]
        source = (DATA / citation['path']).read_text().splitlines()
        self.assertEqual(source[citation['line_start'] - 1], citation['quote'])
        self.assertEqual(citation['doc_id'], 'DOC-A2')

    def test_split_period_uses_both_versions(self):
        case = self.case('L011')
        self.assertEqual({c['doc_id'] for c in case['citations']}, {'DOC-A1', 'DOC-A2'})
        self.assertEqual(case['expected_amount'], '2500.00')

    def test_signed_unreviewed_amendment_blocks(self):
        self.change("UPDATE reviewed_terms SET reviewed='false' WHERE term_id='TERM-A2'")
        self.assertEqual(self.case('L001')['reason'], 'UNREVIEWED_SIGNED_DOCUMENT')

    def test_incomplete_credit_snapshot_blocks(self):
        with connect(self.db) as conn:
            m=json.loads(conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        m['tenants']['DEMO']['adjustments_complete'] = False
        self.change("UPDATE metadata SET value=? WHERE key='manifest'", (json.dumps(m),))
        case=self.case('L001')
        self.assertEqual(case['reason'], 'INCOMPLETE_SNAPSHOT')
        self.assertEqual(case['net_candidate'], '0.00')

    def test_future_credit_does_not_reduce_current_candidate(self):
        self.change("UPDATE adjustments SET effective_date='2026-10-20' WHERE line_id='L003'")
        self.assertEqual(self.case('L003')['net_candidate'], '1000.00')

    def test_cyclic_supersession_is_rejected(self):
        self.change("UPDATE reviewed_terms SET supersedes='TERM-A2' WHERE term_id='TERM-A1'")
        self.assertEqual(self.case('L001')['reason'], 'INVALID_SUPERSESSION_CHAIN')

    def test_supersession_cannot_cross_currency_unit_or_service_type(self):
        for field, value in (('currency', 'EUR'), ('unit', 'DAY'), ('service_type', 'OVERTIME')):
            with self.subTest(field=field):
                self.change("""UPDATE reviewed_terms SET currency='USD', unit='HOUR',
                    service_type='REGULAR' WHERE term_id='TERM-A1'""")
                self.change(f"UPDATE reviewed_terms SET {field}=? WHERE term_id='TERM-A1'",
                            (value,))
                case = self.case('L001')
                self.assertEqual(case['status'], 'NEEDS_REVIEW')
                self.assertEqual(case['reason'], 'INVALID_SUPERSESSION_CHAIN')
                self.assertEqual(case['net_candidate'], '0.00')

    def test_financial_numbers_require_strings(self):
        self.assertEqual(number('0.1'), Decimal('0.1'))
        for value in (0.1, 1, True, Decimal('0.1'), None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ReviewRequired, 'INVALID_NUMBER'):
                    number(value)

    def test_nonfinite_amount_is_rejected(self):
        self.change("UPDATE invoice_lines SET billed_amount='NaN' WHERE line_id='L001'")
        self.assertEqual(self.case('L001')['reason'], 'INVALID_NUMBER')

    def test_negative_payment_is_rejected(self):
        self.change("UPDATE payments SET amount='-1' WHERE line_id='L001'")
        self.assertEqual(self.case('L001')['reason'], 'INVALID_NUMBER')

    def test_payment_currency_mismatch_is_rejected(self):
        self.change("UPDATE payments SET currency='EUR' WHERE line_id='L001'")
        self.assertEqual(self.case('L001')['reason'], 'PAYMENT_CURRENCY_MISMATCH')

    def test_overlarge_offset_is_not_negative_recovery(self):
        self.change("UPDATE adjustments SET amount='9999' WHERE line_id='L003'")
        case=self.case('L003')
        self.assertEqual(case['reason'], 'OFFSETS_EXCEED_RATE_DISCREPANCY')
        self.assertEqual(case['net_candidate'], '0.00')

    def test_unapproved_timesheet_is_rejected(self):
        self.change("UPDATE service_entries SET approved='false' WHERE line_id='L001'")
        self.assertEqual(self.case('L001')['reason'], 'UNAPPROVED_SERVICE_EVIDENCE')

    def test_invented_model_source_is_rejected(self):
        with connect(self.db) as conn:
            docs = retrieve_evidence(conn, 'DEMO', 'L001')
        with self.assertRaises(ValueError):
            validate_model_citations({'citations':[{'doc_id':'MADE-UP', 'quote':'anything'}]},
                                     docs, self.case('L001'))

    def test_invented_quote_is_rejected(self):
        with connect(self.db) as conn:
            docs = retrieve_evidence(conn, 'DEMO', 'L001')
        with self.assertRaises(ValueError):
            validate_model_citations({'citations':[{'doc_id':'DOC-A2', 'quote':'invented'}]},
                                     docs, self.case('L001'))

    def test_missing_governing_citation_is_rejected(self):
        with connect(self.db) as conn:
            docs = retrieve_evidence(conn, 'DEMO', 'L001')
        with self.assertRaises(ValueError):
            validate_model_citations({'citations':[]}, docs, self.case('L001'))

    def test_valid_citation_is_accepted_without_mutating_case(self):
        case = self.case('L001')
        before = copy.deepcopy(case)
        with connect(self.db) as conn:
            docs = retrieve_evidence(conn, 'DEMO', 'L001')
        draft={'citations': [{'doc_id':c['doc_id'], 'quote':c['quote']} for c in case['citations']]}
        self.assertEqual(validate_model_citations(draft, docs, case), draft)
        self.assertEqual(case, before)

    def test_approval_is_local_draft_only(self):
        event = self.event()
        self.assertEqual(event['decision'], 'APPROVED_FOR_DRAFT')
        with connect(self.db) as conn:
            self.assertEqual(audit(conn)['realized_recovery'], '0.00')

    def test_review_replay_is_idempotent(self):
        self.event()
        self.event()
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)

    def test_review_id_cannot_be_reused_with_other_decision(self):
        self.event()
        with self.assertRaises(ValueError):
            self.event(decision='reject')

    def test_stale_case_cannot_be_approved(self):
        digest=self.case('L001')['case_digest']
        self.change("UPDATE invoice_lines SET description='changed evidence context' WHERE line_id='L001'")
        with self.assertRaisesRegex(ValueError, 'STALE_CASE'):
            self.event(case_digest=digest)

    def test_needs_review_cannot_be_approved(self):
        with self.assertRaisesRegex(ValueError, 'evidence-complete'):
            self.event(line_id='L006', case_digest=self.case('L006')['case_digest'])

    def test_reimport_preserves_review_journal(self):
        self.event()
        load_snapshot(self.db, DATA)
        with connect(self.db) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM review_events').fetchone()[0], 1)

    def test_po_is_not_the_rate_authority(self):
        case=self.case('L001')
        self.assertEqual(case['po_rate'], '130')
        self.assertEqual(case['components'][0]['rate'], '120.00')


# Twenty separately reported acceptance cases, independent of the operational engine.
for item in GOLD['cases']:
    def check(self, item=item):
        result = self.case(item['line_id'])
        self.assertEqual(result['status'], item['expected_status'])
        self.assertEqual(result['net_candidate'], item['expected_net_candidate'])
        if item['reason']:
            self.assertEqual(result['reason'], item['reason'])
    setattr(CoreTests, 'test_fixture_' + item['line_id'], check)


if __name__ == '__main__':
    unittest.main()
