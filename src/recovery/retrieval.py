"""Scoped lexical retrieval. Retrieval is not limited to a possibly incomplete top-k."""
from __future__ import annotations

import re
import sqlite3

from .storage import one_line, scoped_documents, metadata
import json


def retrieve_evidence(conn: sqlite3.Connection, tenant: str, line_id: str,
                      query: str = 'hourly rate effective amendment supersedes', *,
                      retrieval_config: dict | None = None) -> list[dict]:
    line = one_line(conn, tenant, line_id)
    as_of = json.loads(metadata(conn, 'manifest'))['as_of']
    # The full scoped signed-document set is retained even when lexical ranking misses a clause.
    documents = [d for d in scoped_documents(conn, line) if d['signed_date'] <= as_of]
    tokens = re.findall(r'[A-Za-z0-9_]+', query)[:24]
    scores = {}
    if tokens:
        expression = ' OR '.join(f'"{t}"' for t in tokens)
        rows = conn.execute('''SELECT f.doc_id, bm25(document_fts) AS score
            FROM document_fts f JOIN documents d
            ON f.tenant_id=d.tenant_id AND f.doc_id=d.doc_id
            WHERE document_fts MATCH ? AND d.tenant_id=? AND d.supplier_id=?
            AND d.contract_id=? AND d.role=? AND d.location=?
            AND d.status='SIGNED' AND d.kind='RATE_AGREEMENT' ''',
            (expression, *(line[k] for k in
                           ('tenant_id', 'supplier_id', 'contract_id', 'role', 'location'))))
        scores = {row['doc_id']: row['score'] for row in rows}
    documents.sort(key=lambda d: (scores.get(d['doc_id'], float('inf')), d['doc_id']))
    if retrieval_config is not None:
        from .semantic_retrieval import rank_evidence
        return rank_evidence(conn, tenant, line_id, documents, retrieval_config,
                             query=query)['documents']
    return documents


def deterministic_brief(case: dict) -> dict:
    """Offline template, deliberately NOT labeled an LLM/RAG-generated answer."""
    return {
        'mode': 'offline_template',
        'summary': f"{case['status']}: {case['reason']}. This is a candidate assessment, not a claim.",
        'next_action': ('Check applicability, full settlement and all offsets before preparing a draft.'
                        if case['status'] == 'CANDIDATE' else
                        'Resolve the evidence gap or retain the documented non-recovery result.'),
        'limitations': ['Synthetic fixtures only.', 'No supplier communication or recovery occurred.'],
        'citations': [{'doc_id': c['doc_id'], 'quote': c['quote']} for c in case['citations']],
    }


def validate_model_citations(brief: dict, documents: list[dict], case: dict) -> dict:
    """Validate source membership/exact quotes, not the semantic truth of generated prose."""
    allowed = {d['doc_id']: d['body'] for d in documents}
    cited = set()
    for citation in brief.get('citations', []):
        did, quote = citation.get('doc_id'), citation.get('quote')
        if did not in allowed or not quote or quote not in allowed[did]:
            raise ValueError('Model returned an unsupported citation; draft withheld')
        cited.add(did)
    required = {c['doc_id'] for c in case.get('citations', [])}
    if not required.issubset(cited):
        raise ValueError('Model omitted governing evidence; draft withheld')
    return brief
