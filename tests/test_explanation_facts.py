"""Independent explanation regressions against bundled synthetic source cases.

These checks exercise exact claim support, not general natural-language entailment.
No provider, network, expected-results file, or real supplier data is used.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from recovery.engine import audit, evaluate_line
from recovery.explanation_facts import (
    build_fact_catalog,
    evaluate_claims,
    validate_fact_draft,
)
from recovery.retrieval import retrieve_evidence
from recovery.storage import connect, load_snapshot


DATA = Path(__file__).resolve().parents[1] / 'data'


class ExplanationFactRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'synthetic.sqlite3'
        load_snapshot(self.db, DATA)

    def inputs(self, line_id):
        with connect(self.db) as conn:
            return (evaluate_line(conn, 'DEMO', line_id),
                    retrieve_evidence(conn, 'DEMO', line_id))

    @staticmethod
    def claims(catalog):
        return [{'fact_id': fact['fact_id'], 'value': fact['value']} for fact in catalog]

    def draft(self, case, documents):
        return {
            'claims': self.claims(build_fact_catalog(case, documents)),
            'next_action': ('review_candidate' if case['status'] == 'CANDIDATE' else
                            'resolve_evidence' if case['status'] == 'NEEDS_REVIEW' else
                            'retain_assessment'),
            'citations': [{'doc_id': cite['doc_id'], 'quote': cite['quote']}
                          for cite in case['citations']],
        }

    def test_l011_keeps_both_service_date_rates_and_all_financial_facts(self):
        case, documents = self.inputs('L011')
        catalog = build_fact_catalog(case, documents)
        facts = {fact['fact_id']: fact for fact in catalog}
        expected = {
            'case.line_id': 'L011',
            'case.status': 'CANDIDATE',
            'case.reason': 'POTENTIAL_RATE_OVERPAYMENT',
            'case.currency': 'USD',
            'case.billed_amount': '2600.00',
            'case.expected_amount': '2500.00',
            'case.gross_discrepancy': '100.00',
            'case.offsets': '0.00',
            'case.net_candidate': '100.00',
            'case.as_of': '2026-10-15',
            'case.po_rate': '130',
            'case.realized_recovery': '0.00',
            'component.SVC-011-1.rate': '130.00',
            'component.SVC-011-2.rate': '120.00',
        }
        for fact_id, value in expected.items():
            with self.subTest(fact_id=fact_id):
                self.assertEqual(facts[fact_id]['value'], value)
        for service_id, date, term, document in [
                ('SVC-011-1', '2026-06-30', 'TERM-A1', 'DOC-A1'),
                ('SVC-011-2', '2026-07-01', 'TERM-A2', 'DOC-A2')]:
            text = facts[f'component.{service_id}.rate']['text']
            for token in (date, '10', term, document):
                self.assertIn(token, text)

    def test_complete_exact_claims_are_supported_without_mutating_inputs(self):
        case, documents = self.inputs('L011')
        original = copy.deepcopy((case, documents))
        catalog = build_fact_catalog(case, documents)
        claims = self.claims(catalog)
        before = copy.deepcopy((catalog, claims))
        report = evaluate_claims(claims, catalog)
        self.assertEqual(report['status'], 'PASSED')
        self.assertEqual(report['missing_fact_ids'], [])
        self.assertTrue(report['financial_consistency'])
        self.assertEqual({claim['classification'] for claim in report['claims']}, {'supported'})
        self.assertEqual((case, documents), original)
        self.assertEqual((catalog, claims), before)

    def test_exact_strings_reject_changed_amount_status_currency_and_rate(self):
        case, documents = self.inputs('L011')
        catalog = build_fact_catalog(case, documents)
        for fact_id, wrong in [
                ('case.net_candidate', '0.00'),
                ('case.net_candidate', '100'),
                ('case.status', 'NO_DISCREPANCY'),
                ('case.currency', 'EUR'),
                ('case.expected_amount', '2600.00'),
                ('component.SVC-011-2.rate', '130.00'),
                ('case.realized_recovery', '100.00')]:
            claims = self.claims(catalog)
            next(item for item in claims if item['fact_id'] == fact_id)['value'] = wrong
            with self.subTest(fact_id=fact_id, wrong=wrong):
                report = evaluate_claims(claims, catalog)
                self.assertEqual(report['status'], 'WITHHELD')
                self.assertFalse(report['financial_consistency'])
                self.assertIn({'fact_id': fact_id, 'classification': 'contradicted'},
                              report['claims'])

    def test_unknown_fact_cannot_supply_missing_amendment_or_approval_claim(self):
        case, documents = self.inputs('L011')
        catalog = build_fact_catalog(case, documents)
        for fact_id, value in [('case.amendment_absent', 'true'),
                               ('case.approval_state', 'APPROVED')]:
            with self.subTest(fact_id=fact_id):
                report = evaluate_claims(
                    self.claims(catalog) + [{'fact_id': fact_id, 'value': value}], catalog)
                self.assertEqual(report['status'], 'WITHHELD')
                self.assertIn({'fact_id': fact_id, 'classification': 'unsupported'},
                              report['claims'])

    def test_missing_any_required_fact_or_duplicate_claim_withholds_draft(self):
        case, documents = self.inputs('L011')
        catalog = build_fact_catalog(case, documents)
        complete = self.claims(catalog)
        for fact in catalog:
            with self.subTest(missing=fact['fact_id']):
                claims = [item for item in complete if item['fact_id'] != fact['fact_id']]
                report = evaluate_claims(claims, catalog)
                self.assertEqual(report['status'], 'WITHHELD')
                self.assertIn(fact['fact_id'], report['missing_fact_ids'])
        report = evaluate_claims(complete + [copy.deepcopy(complete[0])], catalog)
        self.assertEqual(report['status'], 'WITHHELD')

    def test_malformed_claims_are_withheld_rather_than_interpreted(self):
        case, documents = self.inputs('L011')
        catalog = build_fact_catalog(case, documents)
        for claims in [None, {}, '100.00', [None], [{}],
                       [{'fact_id': 'case.net_candidate'}],
                       [{'fact_id': 'case.net_candidate', 'value': 100.0}],
                       [{'fact_id': ['case.net_candidate'], 'value': '100.00'}]]:
            with self.subTest(claims=claims):
                self.assertEqual(evaluate_claims(claims, catalog)['status'], 'WITHHELD')

    def test_l004_existing_credit_is_separate_from_gross_and_net_candidate(self):
        case, documents = self.inputs('L004')
        catalog = build_fact_catalog(case, documents)
        facts = {fact['fact_id']: fact['value'] for fact in catalog}
        self.assertEqual(facts['case.gross_discrepancy'], '1000.00')
        self.assertEqual(facts['case.offsets'], '400.00')
        self.assertEqual(facts['case.net_candidate'], '600.00')
        draft = self.draft(case, documents)
        for claim in draft['claims']:
            if claim['fact_id'] == 'case.net_candidate':
                claim['value'] = '1000.00'
        with self.assertRaises(ValueError):
            validate_fact_draft(draft, case, documents)

    def test_unresolved_conflicting_terms_remain_null_and_needs_review(self):
        case, documents = self.inputs('L006')
        catalog = build_fact_catalog(case, documents)
        facts = {fact['fact_id']: fact['value'] for fact in catalog}
        self.assertEqual(facts['case.status'], 'NEEDS_REVIEW')
        self.assertEqual(facts['case.reason'], 'CONFLICTING_TERMS')
        for field in ('expected_amount', 'gross_discrepancy', 'offsets', 'po_rate'):
            self.assertIsNone(facts[f'case.{field}'])
        self.assertEqual(facts['case.net_candidate'], '0.00')
        draft = self.draft(case, documents)
        validated = validate_fact_draft(draft, case, documents)
        self.assertIn('NEEDS_REVIEW', validated['summary'])
        self.assertIn('CONFLICTING_TERMS', validated['summary'])
        for claim in draft['claims']:
            if claim['fact_id'] == 'case.expected_amount':
                claim['value'] = '0.00'
        with self.assertRaises(ValueError):
            validate_fact_draft(draft, case, documents)

    def test_authentic_quotes_do_not_rescue_l011_false_no_discrepancy_claim(self):
        case, documents = self.inputs('L011')
        original = copy.deepcopy((case, documents))
        draft = self.draft(case, documents)
        bodies = {document['doc_id']: document['body'] for document in documents}
        for citation in draft['citations']:
            self.assertIn(citation['quote'], bodies[citation['doc_id']])
        for claim in draft['claims']:
            if claim['fact_id'] == 'case.status':
                claim['value'] = 'NO_DISCREPANCY'
            elif claim['fact_id'] in {'case.gross_discrepancy', 'case.net_candidate'}:
                claim['value'] = '0.00'
        with self.assertRaises(ValueError):
            validate_fact_draft(draft, case, documents)
        self.assertEqual((case, documents), original)
        with connect(self.db) as conn:
            result = audit(conn)
            self.assertEqual(result['candidate_total'], '80707.50')
            self.assertEqual(result['realized_recovery'], '0.00')
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM review_events').fetchone()[0], 0)

    def test_rendered_l011_narrative_preserves_rates_and_rejects_prose_injection(self):
        case, documents = self.inputs('L011')
        draft = self.draft(case, documents)
        result = validate_fact_draft(draft, case, documents)
        visible = json.dumps({key: result[key] for key in
                              ('summary', 'next_action', 'limitations', 'citations')})
        for token in ('L011', 'CANDIDATE', '100.00', 'USD', '2026-06-30',
                      '2026-07-01', '130.00', '120.00', 'DOC-A1', 'DOC-A2'):
            self.assertIn(token, visible)
        draft['summary'] = 'No amendment or discrepancy evidence exists.'
        with self.assertRaises(ValueError):
            validate_fact_draft(draft, case, documents)

    def test_component_rate_requires_matching_term_document_quote_and_hash(self):
        case, documents = self.inputs('L011')
        examples = []
        changed = copy.deepcopy(case)
        changed['components'][1]['term_id'] = 'UNSUPPORTED-TERM'
        examples.append((changed, documents))
        changed = copy.deepcopy(case)
        changed['citations'][1]['quote'] = 'An invented rate clause.'
        examples.append((changed, documents))
        changed = copy.deepcopy(case)
        changed['citations'][1]['sha256'] = '0' * 64
        examples.append((changed, documents))
        for changed_case, changed_documents in examples:
            with self.subTest(case=changed_case):
                with self.assertRaises(ValueError):
                    build_fact_catalog(changed_case, changed_documents)

    def test_warnings_are_required_and_missing_scalar_is_not_invented(self):
        case, documents = self.inputs('L011')
        case['warnings'] = ['Future adjustment excluded: SYNTHETIC-FUTURE']
        catalog = build_fact_catalog(case, documents)
        warning = next(fact for fact in catalog if fact['fact_id'] == 'warning.0')
        self.assertEqual(warning['value'], case['warnings'][0])
        report = evaluate_claims([claim for claim in self.claims(catalog)
                                 if claim['fact_id'] != 'warning.0'], catalog)
        self.assertEqual(report['status'], 'WITHHELD')
        self.assertIn('warning.0', report['missing_fact_ids'])
        del case['billed_amount']
        with self.assertRaises(ValueError):
            build_fact_catalog(case, documents)

    def test_malformed_citations_are_rejected_by_direct_validator(self):
        case, documents = self.inputs('L011')
        draft = self.draft(case, documents)
        for citations in (None, [None], [{}], [{'doc_id': [], 'quote': 'text'}]):
            with self.subTest(citations=citations), self.assertRaises(ValueError):
                validate_fact_draft({**draft, 'citations': citations}, case, documents)


if __name__ == '__main__':
    unittest.main()
