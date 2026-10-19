"""Opt-in local-generation smoke check using only the bundled synthetic fixtures.

This is an integration check, not an evaluation of explanation accuracy on real data.
It never promotes terms, records reviews, or modifies the user's audit database.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from recovery.ai import explain_case, resolve_explanation_config
from recovery.engine import audit, evaluate_line
from recovery.retrieval import retrieve_evidence
from recovery.storage import connect, load_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-local', action='store_true',
                        help='Explicitly allow synthetic prompts to the loopback Ollama server')
    parser.add_argument('--model', required=True)
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--context-tokens', type=int, default=16384)
    parser.add_argument('--output', type=Path,
                        default=Path('var/ollama-implementation/live-smoke.json'))
    args = parser.parse_args()
    if not args.run_local:
        parser.error('--run-local is required; no generation request was made')

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.suffix != '.json' or output == root / 'data' or root / 'data' in output.parents:
        parser.error('--output must be a JSON file outside the bundled source fixtures')
    report = {'check': 'opt_in_synthetic_ollama_smoke', 'status': 'FAILED', 'cases': [],
              'validation_scope': 'transport_schema_governing_quotes_closed_facts_and_financial_immutability',
              'semantic_quality': 'ARBITRARY_PROSE_NOT_GENERATED_OR_GRADED',
              'limitations': ['Closed fact matching does not establish real-world source correctness.',
                              'No real supplier data or recovery actions are evaluated.']}
    try:
        config = resolve_explanation_config(provider='ollama', model=args.model,
                                            base_url=args.ollama_url,
                                            context_tokens=args.context_tokens)
        report['configuration'] = config
        with tempfile.TemporaryDirectory(prefix='recovery-ollama-smoke-') as directory:
            db = Path(directory) / 'synthetic.sqlite3'
            load_snapshot(db, root / 'data')
            with connect(db) as conn:
                before = audit(conn)
                for line_id in ('L001', 'L011'):
                    case = evaluate_line(conn, 'DEMO', line_id)
                    documents = retrieve_evidence(conn, 'DEMO', line_id)
                    original = deepcopy((case, documents))
                    check = {'line_id': line_id,
                             'retrieved_doc_ids': [d['doc_id'] for d in documents]}
                    try:
                        check['brief'] = explain_case(case, documents, **config)
                        if (case, documents) != original:
                            raise AssertionError('Provider mutated authoritative inputs')
                        check['status'] = 'PASSED'
                    except (ValueError, RuntimeError, OSError, AssertionError) as exc:
                        check['status'] = 'FAILED'
                        check['error'] = str(exc)
                    report['cases'].append(check)
                after = audit(conn)
                report['audit_unchanged'] = before == after
                report['candidate_total'] = after['candidate_total']
                report['realized_recovery'] = after['realized_recovery']
                report['case_count'] = after['case_count']
                report['counts'] = after['counts']
                report['review_events'] = conn.execute(
                    'SELECT COUNT(*) FROM review_events').fetchone()[0]
                if (before == after and after['candidate_total'] == '80707.50'
                        and after['realized_recovery'] == '0.00'
                        and report['review_events'] == 0
                        and all(c['status'] == 'PASSED' for c in report['cases'])):
                    report['status'] = 'PASSED'
    except (ValueError, RuntimeError, OSError) as exc:
        report['error'] = str(exc)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
