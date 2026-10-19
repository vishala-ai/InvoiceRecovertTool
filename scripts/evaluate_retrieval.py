"""Document-coverage smoke test, NOT an LLM quality or ranking benchmark."""
import json
from pathlib import Path
from recovery.storage import connect
from recovery.retrieval import retrieve_evidence

root = Path(__file__).resolve().parents[1]
fixtures = [json.loads(line) for line in (root/'data/retrieval_eval.jsonl').read_text().splitlines()]
results=[]
with connect(root/'var/recovery.sqlite3') as conn:
    for fixture in fixtures:
        docs=retrieve_evidence(conn, 'DEMO', fixture['line_id'], fixture['question'])
        retrieved={d['doc_id'] for d in docs}
        passed=set(fixture['required_docs']).issubset(retrieved) and not (
            set(fixture['forbidden_docs']) & retrieved)
        results.append({'id':fixture['id'], 'passed':passed, 'retrieved':sorted(retrieved)})
report={'evaluation':'scoped_document_coverage_smoke_test', 'passed':sum(x['passed'] for x in results),
        'total':len(results), 'provider_calls':0, 'results':results}
print(json.dumps(report, indent=2))
raise SystemExit(0 if all(x['passed'] for x in results) else 1)
