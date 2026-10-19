"""Controlled synthetic claim checks; not a general semantic-faithfulness benchmark.

The default run is offline. The developer-authored labels are provisional and require
SME review; these cases are visible regression fixtures, not an independent held-out set.
An optional local run uses installed Ollama weights and a disposable synthetic database.
No reviews, approvals, source updates, or realized recoveries are recorded.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile

from recovery.engine import audit, evaluate_line
from recovery.explanation_facts import (
    DraftRejected, build_fact_catalog, evaluate_claims, validate_fact_draft,
)
from recovery.retrieval import retrieve_evidence
from recovery.storage import connect, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'tests/fixtures/explanation_eval.json'
EXPECTED_COUNTS = {'CANDIDATE': 4, 'NO_DISCREPANCY': 4, 'ALREADY_OFFSET': 2,
                   'NEEDS_REVIEW': 9, 'UNPAID': 1}


def load_corpus(path: Path = CORPUS) -> dict:
    """Load only the test-side labels; operational modules never call this function."""
    corpus = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(corpus, dict) or corpus.get('version') != 1
            or corpus.get('label_status') != 'PROVISIONAL_PENDING_SME_REVIEW'
            or not isinstance(corpus.get('label_author'), str)
            or not isinstance(corpus.get('local_line_ids'), list)
            or not corpus['local_line_ids']
            or not all(isinstance(line, str) for line in corpus['local_line_ids'])
            or not isinstance(corpus.get('cases'), list) or not corpus['cases']):
        raise ValueError('Invalid or unlabelled explanation evaluation corpus')
    seen = set()
    for fixture in corpus['cases']:
        if not isinstance(fixture, dict) or not isinstance(fixture.get('id'), str):
            raise ValueError('Every evaluation case requires an ID')
        if fixture['id'] in seen:
            raise ValueError('Evaluation case IDs must be unique')
        seen.add(fixture['id'])
        if (not isinstance(fixture.get('line_id'), str)
                or not isinstance(fixture.get('expected'), dict)
                or not isinstance(fixture.get('fact_assertions'), dict)
                or not isinstance(fixture.get('changes', []), list)):
            raise ValueError('Malformed explanation evaluation case')
        expected = fixture['expected']
        if (set(expected) != {'claim_status', 'classifications', 'missing_fact_ids',
                             'financial_consistency', 'citation_validation', 'draft_status'}
                or expected['claim_status'] not in {'PASSED', 'WITHHELD'}
                or expected['draft_status'] not in {'PASSED', 'WITHHELD'}
                or expected['citation_validation'] not in {'PASSED', 'WITHHELD'}
                or not isinstance(expected['classifications'], dict)
                or not all(isinstance(key, str) and value in {'contradicted', 'unsupported'}
                           for key, value in expected['classifications'].items())
                or not isinstance(expected['missing_fact_ids'], list)
                or not all(isinstance(key, str) for key in expected['missing_fact_ids'])
                or type(expected['financial_consistency']) is not bool):
            raise ValueError('Malformed explanation evaluation expectations')
    return corpus


def fixture_claims(catalog: list[dict], fixture: dict) -> list[dict]:
    """Copy the complete catalog and apply explicitly labelled test corruptions."""
    claims = [{'fact_id': fact['fact_id'], 'value': fact['value']} for fact in catalog]
    for change in fixture.get('changes', []):
        if not isinstance(change, dict) or change.get('op') not in {'replace', 'remove', 'add'}:
            raise ValueError('Unknown test claim operation')
        fact_id = change.get('fact_id')
        if not isinstance(fact_id, str):
            raise ValueError('Test claim changes require a fact ID')
        matches = [claim for claim in claims if claim['fact_id'] == fact_id]
        if change['op'] in {'remove', 'replace'} and len(matches) != 1:
            raise ValueError('Test corruption refers to an absent or duplicate fact')
        if change['op'] == 'remove':
            claims.remove(matches[0])
        elif change['op'] == 'replace':
            matches[0]['value'] = change['value']
        else:
            claims.append({'fact_id': fact_id, 'value': change['value']})
    return claims


def fixture_citations(case: dict, fixture: dict) -> list[dict]:
    citations = [{'doc_id': c['doc_id'], 'quote': c['quote']} for c in case['citations']]
    mutation = fixture.get('citation_change')
    if mutation is None:
        return citations
    if mutation == 'omit_governing_amendment':
        return [c for c in citations if c['doc_id'] != 'DOC-A2']
    if mutation == 'fabricated_quote':
        if not citations:
            raise ValueError('Fabricated-quote fixture needs a governing citation')
        citations[0]['quote'] = 'This exact sentence does not appear in the signed fixture.'
        return citations
    if mutation == 'incomplete_exact_quote':
        if not citations or len(citations[0]['quote']) < 2:
            raise ValueError('Incomplete-quote fixture needs a governing quotation')
        # This is a real substring, so source membership alone would accept it.
        citations[0]['quote'] = citations[0]['quote'][0]
        return citations
    if mutation == 'unknown_document':
        return citations + [{'doc_id': 'OTHER-TENANT-DOC', 'quote': 'Unrelated evidence.'}]
    raise ValueError('Unknown test citation operation')


def _action(case: dict) -> str:
    return ('review_candidate' if case['status'] == 'CANDIDATE' else
            'resolve_evidence' if case['status'] == 'NEEDS_REVIEW' else 'retain_assessment')


def check_citations(citations: list[dict], case: dict, documents: list[dict],
                    catalog: list[dict]) -> str:
    """Isolate citation checks using known-good claims and the allowed case action."""
    try:
        validate_fact_draft({
            'claims': [{'fact_id': f['fact_id'], 'value': f['value']} for f in catalog],
            'next_action': _action(case), 'citations': citations,
        }, case, documents)
    except ValueError:
        return 'WITHHELD'
    return 'PASSED'


def evaluate_regressions(conn: sqlite3.Connection, corpus: dict) -> list[dict]:
    results = []
    for fixture in corpus['cases']:
        case = evaluate_line(conn, 'DEMO', fixture['line_id'])
        documents = retrieve_evidence(conn, 'DEMO', fixture['line_id'])
        original = deepcopy((case, documents))
        catalog = build_fact_catalog(case, documents)
        facts = {fact['fact_id']: fact['value'] for fact in catalog}
        claims = fixture_claims(catalog, fixture)
        validation = evaluate_claims(claims, catalog)
        citations = fixture_citations(case, fixture)
        citation_status = check_citations(citations, case, documents, catalog)
        draft_status = 'PASSED'
        try:
            validate_fact_draft({'claims': claims, 'next_action': _action(case),
                                 'citations': citations}, case, documents)
        except ValueError:
            draft_status = 'WITHHELD'
        classifications = {c['fact_id']: c['classification'] for c in validation['claims']
                           if c['classification'] != 'supported'}
        expected = fixture['expected']
        checks = {
            'fixed_fact_values': all(key in facts and facts[key] == value
                                     for key, value in fixture['fact_assertions'].items()),
            'claim_status': validation['status'] == expected['claim_status'],
            'claim_classifications': classifications == expected['classifications'],
            'required_coverage': sorted(validation['missing_fact_ids'])
                == sorted(expected['missing_fact_ids']),
            'financial_consistency': validation['financial_consistency']
                is expected['financial_consistency'],
            'citation_validation': citation_status == expected['citation_validation'],
            'draft_status': draft_status == expected['draft_status'],
            'input_immutability': (case, documents) == original,
        }
        results.append({
            'id': fixture['id'], 'line_id': fixture['line_id'],
            'passed': all(checks.values()), 'checks': checks,
            'claim_validation': validation, 'citation_validation': citation_status,
            'draft_status': draft_status,
            'required_fact_count': len(catalog),
            'governing_doc_ids': sorted({c['doc_id'] for c in case['citations']}),
        })
    return results


def evaluate_local(conn: sqlite3.Connection, config: dict, line_ids: list[str]) -> list[dict]:
    # Lazy import keeps the deterministic evaluation runnable without optional packages.
    from recovery.ai import explain_case

    results = []
    for line_id in line_ids:
        case = evaluate_line(conn, 'DEMO', line_id)
        documents = retrieve_evidence(conn, 'DEMO', line_id)
        original = deepcopy((case, documents))
        result = {'line_id': line_id, 'status': 'WITHHELD',
                  'general_semantic_accuracy': 'NOT_MEASURED'}
        try:
            brief = explain_case(case, documents, **config)
            catalog = build_fact_catalog(case, documents)
            validation = evaluate_claims(brief['claims'], catalog)
            citation_status = check_citations(brief['citations'], case, documents, catalog)
            result.update(brief=brief, claim_validation=validation,
                          citation_validation=citation_status,
                          status='PASSED' if (validation['status'] == 'PASSED'
                                              and citation_status == 'PASSED') else 'WITHHELD')
        except DraftRejected as exc:
            result.update(error=str(exc), claim_validation=exc.validation,
                          citation_validation=exc.validation['citation_check'])
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            result['error'] = str(exc)
            result['citation_validation'] = 'NOT_ESTABLISHED'
        result['input_immutability'] = (case, documents) == original
        if not result['input_immutability']:
            result['status'] = 'FAILED'
        results.append(result)
    return results


def run_evaluation(*, corpus_path: Path = CORPUS, run_local: bool = False,
                   model: str | None = None, base_url: str = 'http://127.0.0.1:11434',
                   context_tokens: int = 16384) -> dict:
    if run_local and not model:
        raise ValueError('--model is required with --run-local')
    if model and not run_local:
        raise ValueError('--model requires --run-local; no generation request was made')
    corpus = load_corpus(corpus_path)
    report = {
        'evaluation': 'synthetic_controlled_claim_regressions', 'status': 'FAILED',
        'label_status': corpus['label_status'], 'label_author': corpus['label_author'],
        'scope': 'Exact controlled-claim alignment, financial consistency, required coverage, '
                 'and citation validation are reported separately.',
        'general_semantic_faithfulness': 'NOT_MEASURED',
        'held_out_sme_validation': 'NOT_PERFORMED', 'generation_cases_attempted': 0,
        'limitations': [
            'Developer-authored provisional labels require independent SME review.',
            'Visible synthetic regression fixtures do not establish real-data accuracy.',
            'No semantic model judge is installed, calibrated, or used.',
            'No review, approval, claim, payment, or recovery action is performed.',
        ],
    }
    with tempfile.TemporaryDirectory(prefix='recovery-explanation-eval-') as directory:
        db = Path(directory) / 'synthetic.sqlite3'
        load_snapshot(db, ROOT / 'data')
        with connect(db) as conn:
            before = audit(conn)
            report['regressions'] = evaluate_regressions(conn, corpus)
            report['local_generation'] = {'requested': run_local, 'status': 'NOT_RUN', 'cases': []}
            if run_local:
                from recovery.ai import resolve_explanation_config

                try:
                    config = resolve_explanation_config(provider='ollama', model=model,
                                                        base_url=base_url,
                                                        context_tokens=context_tokens)
                    report['local_generation']['configuration'] = config
                    line_ids = corpus['local_line_ids']
                    report['generation_cases_attempted'] = len(line_ids)
                    local = evaluate_local(conn, config, line_ids)
                    report['local_generation'].update(
                        cases=local, status='PASSED' if all(c['status'] == 'PASSED' for c in local)
                        else 'NEEDS_IMPROVEMENT')
                except (ValueError, RuntimeError, OSError) as exc:
                    report['local_generation'].update(status='FAILED', error=str(exc))
            after = audit(conn)
            report['invariants'] = {
                'audit_unchanged': before == after,
                'case_count': after['case_count'], 'candidate_total': after['candidate_total'],
                'realized_recovery': after['realized_recovery'], 'counts': after['counts'],
                'review_events': conn.execute('SELECT COUNT(*) FROM review_events').fetchone()[0],
            }
            invariant_pass = (before == after and after['case_count'] == 20
                              and after['candidate_total'] == '80707.50'
                              and after['realized_recovery'] == '0.00'
                              and after['counts'] == EXPECTED_COUNTS
                              and report['invariants']['review_events'] == 0)
            report['invariants']['passed'] = invariant_pass
    report['passed'] = sum(result['passed'] for result in report['regressions'])
    report['total'] = len(report['regressions'])
    if (report['passed'] == report['total'] and report['invariants']['passed']
            and report['local_generation']['status'] in {'NOT_RUN', 'PASSED'}):
        report['status'] = 'PASSED'
    return report


def report_target(path: Path) -> Path:
    """Restrict reports to ordinary, unlinked JSON files within project var/."""
    target = path.absolute()
    output_root = ROOT / 'var'
    try:
        relative = target.relative_to(output_root)
    except ValueError:
        raise ValueError('--output must be a JSON file under the project var directory') from None
    if (target.suffix != '.json' or '..' in relative.parts
            or any(part.lower().startswith('.env') for part in relative.parts)):
        raise ValueError('--output must be a separate report file without secret paths')
    current = output_root
    for part in ('', *relative.parts):
        if part:
            current = current / part
        if current.is_symlink():
            raise ValueError('--output must not contain symbolic links')
    if target.exists() and (not target.is_file() or target.stat().st_nlink > 1):
        raise ValueError('--output must be an ordinary file without hard links')
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-local', action='store_true',
                        help='Explicitly allow synthetic prompts to loopback Ollama')
    parser.add_argument('--model', help='Already installed local model; no download occurs')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--context-tokens', type=int, default=16384)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'var/faithfulness-implementation/evaluation.json')
    args = parser.parse_args(argv)
    try:
        output = report_target(args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        report = run_evaluation(run_local=args.run_local, model=args.model,
                                base_url=args.ollama_url, context_tokens=args.context_tokens)
    except (ValueError, RuntimeError, OSError, sqlite3.DatabaseError) as exc:
        report = {'evaluation': 'synthetic_controlled_claim_regressions',
                  'status': 'FAILED', 'error': str(exc)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
