"""Offline eval regressions; no model calls or independent SME accuracy claims."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recovery.engine import audit, evaluate_line
from recovery.explanation_facts import DraftRejected, build_fact_catalog, validate_fact_draft
from recovery.retrieval import retrieve_evidence
from recovery.storage import connect, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'evaluate_explanations', ROOT / 'scripts/evaluate_explanations.py')
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class ExplanationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='test-explanation-eval-')
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'synthetic.sqlite3'
        load_snapshot(self.db, ROOT / 'data')
        self.corpus = evaluation.load_corpus()

    def regressions(self):
        with connect(self.db) as conn:
            return {row['id']: row for row in evaluation.evaluate_regressions(conn, self.corpus)}

    def test_offline_corpus_passes_without_provider_or_review_events(self):
        with patch('recovery.ai.resolve_explanation_config') as config, \
                patch('recovery.ai.explain_case') as generate:
            result = evaluation.run_evaluation()
        config.assert_not_called()
        generate.assert_not_called()
        self.assertEqual(result['status'], 'PASSED')
        self.assertEqual((result['passed'], result['total']), (21, 21))
        self.assertEqual(result['generation_cases_attempted'], 0)
        self.assertEqual(result['label_status'], 'PROVISIONAL_PENDING_SME_REVIEW')
        self.assertEqual(result['held_out_sme_validation'], 'NOT_PERFORMED')
        self.assertEqual(result['general_semantic_faithfulness'], 'NOT_MEASURED')
        self.assertEqual(result['invariants']['candidate_total'], '80707.50')
        self.assertEqual(result['invariants']['realized_recovery'], '0.00')
        self.assertEqual(result['invariants']['review_events'], 0)

    def test_l011_both_rates_and_false_zero_remain_distinct(self):
        results = self.regressions()
        correct = results['l011_both_service_rate_periods']
        self.assertTrue(correct['passed'])
        self.assertEqual(correct['governing_doc_ids'], ['DOC-A1', 'DOC-A2'])
        self.assertTrue(correct['claim_validation']['financial_consistency'])
        false_zero = results['l011_false_no_discrepancy']['claim_validation']
        self.assertEqual(false_zero['status'], 'WITHHELD')
        self.assertFalse(false_zero['financial_consistency'])
        bad = {c['fact_id']: c['classification'] for c in false_zero['claims']
               if c['classification'] != 'supported'}
        self.assertEqual(bad, {'case.status': 'contradicted',
                               'case.net_candidate': 'contradicted'})

    def test_unsupported_claim_and_missing_rate_are_separate_from_financial_errors(self):
        results = self.regressions()
        unknown = results['l011_unsupported_no_amendment']['claim_validation']
        self.assertEqual(unknown['status'], 'WITHHELD')
        self.assertTrue(unknown['financial_consistency'])
        self.assertIn({'fact_id': 'evidence.no_amendment', 'classification': 'unsupported'},
                      [{k: c[k] for k in ('fact_id', 'classification')} for c in unknown['claims']])
        missing = results['l011_late_rate_missing']['claim_validation']
        self.assertEqual(missing['missing_fact_ids'], ['component.SVC-011-2.rate'])
        self.assertFalse(missing['financial_consistency'])

    def test_valid_claims_do_not_rescue_bad_citations(self):
        results = self.regressions()
        for fixture_id in ('l011_omitted_governing_amendment_citation',
                           'l011_fabricated_quote', 'l011_unrelated_source_citation',
                           'l011_incomplete_but_exact_governing_quote'):
            with self.subTest(fixture=fixture_id):
                result = results[fixture_id]
                self.assertTrue(result['passed'])
                self.assertEqual(result['claim_validation']['status'], 'PASSED')
                self.assertEqual(result['citation_validation'], 'WITHHELD')

    def test_fixed_labels_detect_actual_fact_and_expected_classification_changes(self):
        self.corpus['cases'][0]['fact_assertions']['case.net_candidate'] = '0.00'
        with connect(self.db) as conn:
            result = evaluation.evaluate_regressions(conn, self.corpus)[0]
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['fixed_fact_values'])
        fixture = next(f for f in self.corpus['cases'] if f['id'] == 'l011_false_no_discrepancy')
        fixture['expected']['classifications'] = {}
        results = self.regressions()
        self.assertFalse(results['l011_false_no_discrepancy']['checks']['claim_classifications'])

    def test_audit_drift_fails_instead_of_changing_expected_total(self):
        with connect(self.db) as conn:
            original = audit(conn)
        changed = deepcopy(original)
        changed['candidate_total'] = '0.00'
        with patch.object(evaluation, 'audit', side_effect=[original, changed]):
            result = evaluation.run_evaluation()
        self.assertEqual(result['status'], 'FAILED')
        self.assertFalse(result['invariants']['passed'])
        self.assertFalse(result['invariants']['audit_unchanged'])

    def test_local_run_requires_explicit_opt_in_and_model(self):
        with patch('recovery.ai.resolve_explanation_config') as config, \
                patch('recovery.ai.explain_case') as generate:
            with self.assertRaisesRegex(ValueError, '--run-local'):
                evaluation.run_evaluation(model='synthetic-model')
            with self.assertRaisesRegex(ValueError, '--model'):
                evaluation.run_evaluation(run_local=True)
        config.assert_not_called()
        generate.assert_not_called()

    def test_local_invalid_generated_claim_is_withheld_and_database_unchanged(self):
        with connect(self.db) as conn:
            case = evaluate_line(conn, 'DEMO', 'L011')
            documents = retrieve_evidence(conn, 'DEMO', 'L011')
            claims = [{'fact_id': f['fact_id'], 'value': f['value']}
                      for f in build_fact_catalog(case, documents)]
            next(c for c in claims if c['fact_id'] == 'case.net_candidate')['value'] = '0.00'
            brief = {'claims': claims,
                     'citations': [{'doc_id': c['doc_id'], 'quote': c['quote']}
                                   for c in case['citations']]}
            before = audit(conn)
            with patch('recovery.ai.explain_case', return_value=brief):
                result = evaluation.evaluate_local(conn, {'provider': 'ollama'}, ['L011'])[0]
            self.assertEqual(result['status'], 'WITHHELD')
            self.assertFalse(result['claim_validation']['financial_consistency'])
            self.assertEqual(result['citation_validation'], 'PASSED')
            self.assertEqual(audit(conn), before)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM review_events').fetchone()[0], 0)

    def test_local_provider_failure_is_not_reported_as_quality_pass(self):
        with connect(self.db) as conn, patch('recovery.ai.explain_case',
                                             side_effect=RuntimeError('Synthetic unavailable')):
            result = evaluation.evaluate_local(conn, {'provider': 'ollama'}, ['L011'])[0]
        self.assertEqual(result['status'], 'WITHHELD')
        self.assertEqual(result['citation_validation'], 'NOT_ESTABLISHED')
        self.assertEqual(result['general_semantic_accuracy'], 'NOT_MEASURED')
        self.assertTrue(result['input_immutability'])

    def test_local_rejection_preserves_safe_validation_details(self):
        with connect(self.db) as conn:
            case = evaluate_line(conn, 'DEMO', 'L011')
            documents = retrieve_evidence(conn, 'DEMO', 'L011')
            claims = [{'fact_id': f['fact_id'], 'value': f['value']}
                      for f in build_fact_catalog(case, documents)
                      if f['fact_id'] != 'component.SVC-011-2.rate']
            draft = {'claims': claims, 'next_action': 'review_candidate',
                     'citations': [{'doc_id': c['doc_id'], 'quote': c['quote']}
                                   for c in case['citations']]}
            with self.assertRaises(DraftRejected) as caught:
                validate_fact_draft(draft, case, documents)
            with patch('recovery.ai.explain_case', side_effect=caught.exception):
                result = evaluation.evaluate_local(conn, {'provider': 'ollama'}, ['L011'])[0]
        self.assertEqual(result['status'], 'WITHHELD')
        self.assertEqual(result['claim_validation'], caught.exception.validation)
        self.assertEqual(result['claim_validation']['missing_fact_ids'],
                         ['component.SVC-011-2.rate'])
        self.assertFalse(result['claim_validation']['financial_consistency'])
        self.assertEqual(result['citation_validation'], 'NOT_ESTABLISHED')
        self.assertNotIn('brief', result)
        self.assertTrue(result['input_immutability'])

    def test_unreviewed_and_duplicate_labels_are_rejected(self):
        for change in ('status', 'duplicate_id'):
            corpus = deepcopy(self.corpus)
            if change == 'status':
                corpus['label_status'] = 'SME_APPROVED'
            else:
                corpus['cases'].append(deepcopy(corpus['cases'][0]))
            path = Path(self.temp.name) / f'{change}.json'
            path.write_text(json.dumps(corpus), encoding='utf-8')
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluation.load_corpus(path)

    def test_unknown_test_corruption_does_not_silently_pass(self):
        with self.assertRaisesRegex(ValueError, 'Unknown test claim'):
            evaluation.fixture_claims([], {'changes': [{'op': 'invent'}]})
        with self.assertRaisesRegex(ValueError, 'absent or duplicate'):
            evaluation.fixture_claims([], {'changes': [
                {'op': 'replace', 'fact_id': 'absent', 'value': '0.00'}]})

    def test_cli_refuses_source_fixture_output_before_writing(self):
        path = ROOT / 'tests/fixtures/explanation_eval.json'
        before = path.read_bytes()
        with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            evaluation.main(['--output', str(path)])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(path.read_bytes(), before)

    def test_report_target_rejects_secret_names_symlinks_and_hard_links(self):
        with tempfile.TemporaryDirectory(prefix='test-report-path-') as directory, \
                patch.object(evaluation, 'ROOT', Path(directory).resolve()):
            root = Path(directory).resolve() / 'var' / 'reports'
            root.mkdir(parents=True)
            self.assertEqual(evaluation.report_target(root / 'new.json'), root / 'new.json')
            regular = root / 'regular.json'
            regular.write_text('{"sentinel": true}\n', encoding='utf-8')
            symlink = root / 'linked.json'
            symlink.symlink_to(regular)
            hardlink = root / 'hard.json'
            hardlink.hardlink_to(regular)
            linked_directory = root / 'linked-directory'
            linked_directory.symlink_to(root, target_is_directory=True)
            for target in (symlink, hardlink, linked_directory / 'new.json',
                           root / '.env.json', root / '../escape.json'):
                with self.subTest(target=target.name), self.assertRaises(ValueError):
                    evaluation.report_target(target)
            self.assertEqual(regular.read_text(encoding='utf-8'), '{"sentinel": true}\n')


if __name__ == '__main__':
    unittest.main()
