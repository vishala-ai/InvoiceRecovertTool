"""Bounded, read-only audit workspace questions over synthetic records.

Routing is deterministic; no model receives SQL, tools, approval capabilities,
conversation instructions or authority to calculate money. Optional Ollama calls
use only the existing closed-claim case explanation and its independent checks.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .dashboard import summarize_cases
from .engine import evaluate_line
from .explanation_facts import ACTIONS, build_fact_catalog, required_action, validate_fact_draft
from .historical_recovery import get_recovery_overview
from .lifecycle_storage import active, validate_projections, validate_source_projections
from .presentation import assessment_view, amount
from .retrieval import retrieve_evidence
from .vendor_contacts import list_vendor_contacts
from .workflow import list_review_workflows

MAX_QUESTION = 2000
MAX_ROWS = 50
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}\Z')
_WRITE = re.compile(r'\b(approve|reject|delete|update|insert|send|submit|pay|reassess|resolve|change|execute)\b', re.I)
_UNTRUSTED = re.compile(r'\b(ignore|override|bypass)\b|\b(system prompt|api key|\.env|union select|drop table)\b|\bselect\b.+\bfrom\b', re.I)
_STATUS = {
    'approved': r'\bapproved\b', 'rejected': r'\brejected\b',
    'approval': r'\b(pending|awaiting approval|needs approval)\b',
    'evidence': r'\b(evidence required|requiring evidence|needs evidence|needing evidence|missing evidence|need more evidence|needs review|requiring review)\b',
    'preparation': r'\b(prepare explanation|unprepared)\b',
    'reassessment': r'\b(reassessment|stale)\b',
}
_HELP = ('Ask about a vendor or its contact, contracts, invoice cases, pending approvals, '
         'missing evidence, or portfolio recovery totals. For example: "Explain L001", '
         '"Show cases for SUP-A", "Who is the POC for SUP-A?", or "Show approved value and cash recovery".')


@contextmanager
def _read(db):
    path = Path(db)
    if not path.is_file():
        raise ValueError('An existing synthetic database is required.')
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA query_only=ON')
    try:
        yield conn
    finally:
        conn.close()


def _exists(conn, table):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _reply(intent, answer, *, rows=(), citations=(), actions=(), scope=None, **extra):
    return {'intent': intent, 'answer': answer, 'rows': list(rows), 'citations': list(citations),
            'actions': list(actions), 'scope': scope or {}, **extra}


def _limited(intent, answer, rows, *, scope, actions=(), **extra):
    total = len(rows)
    return _reply(intent, answer + (f' Showing the first {MAX_ROWS} of {total} matches.'
                  if total > MAX_ROWS else f' {total} matching record(s).'),
                  rows=rows[:MAX_ROWS], scope=scope, actions=actions,
                  total=total, truncated=total > MAX_ROWS, **extra)


def _guard(conn, tenant):
    if not isinstance(tenant, str) or not _ID.fullmatch(tenant):
        raise ValueError('Invalid demo tenant scope.')
    row = conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
    manifest = json.loads(row[0]) if row else {}
    if manifest.get('synthetic') is not True:
        raise ValueError('SYNTHETIC_ONLY: the assistant requires synthetic records.')
    if tenant not in manifest.get('tenants', {}):
        raise ValueError('The selected tenant is not in the synthetic manifest.')
    validate_source_projections(conn)
    return manifest


def _catalog(conn, tenant):
    vendors = {r['supplier_id']: dict(r) for r in conn.execute(
        'SELECT supplier_id,name FROM suppliers WHERE tenant_id=?', (tenant,))}
    contracts = {(r['supplier_id'], r['contract_id']) for r in conn.execute(
        'SELECT DISTINCT supplier_id,contract_id FROM invoice_lines WHERE tenant_id=?', (tenant,))}
    contracts.update((r['supplier_id'], r['contract_id']) for r in conn.execute(
        'SELECT DISTINCT supplier_id,contract_id FROM documents WHERE tenant_id=?', (tenant,)))
    cases = {r['line_id']: dict(r) for r in conn.execute(
        'SELECT line_id,supplier_id,contract_id FROM invoice_lines WHERE tenant_id=?', (tenant,))}
    if _exists(conn, 'legacy_cases'):
        for row in conn.execute('SELECT supplier_id,name FROM legacy_suppliers WHERE tenant_id=?', (tenant,)):
            vendors.setdefault(row['supplier_id'], dict(row))
        contracts.update((r['supplier_id'], r['contract_id']) for r in conn.execute(
            'SELECT DISTINCT supplier_id,contract_id FROM legacy_cases WHERE tenant_id=?', (tenant,)))
        for row in conn.execute('SELECT line_id,supplier_id,contract_id FROM legacy_cases WHERE tenant_id=?', (tenant,)):
            cases.setdefault(row['line_id'], dict(row))
    return vendors, contracts, cases


def _mentioned(question, values):
    return [value for value in sorted(set(values))
            if re.search(r'(?<![\w.:-])' + re.escape(value) + r'(?![\w.:-])', question, re.I)]


def _scope(question, context, vendors, contracts, cases, contacts):
    scope = dict(context or {})
    if (not isinstance(context, (dict, type(None))) or set(scope) - {'supplier_id', 'contract_id', 'line_id'}
            or any(not isinstance(v, str) or not _ID.fullmatch(v) for v in scope.values())):
        raise ValueError('Context must contain valid supplier_id, contract_id or line_id values only.')
    explicit = any(_mentioned(question, values) for values in (vendors, {c[1] for c in contracts}, cases))
    if explicit or not re.search(r'\b(this|that|it|its|same|selected)\b', question, re.I):
        scope = {}
    elif re.search(r'\bcontracts?\b', question, re.I):
        scope.pop('line_id', None)
    elif re.search(r'\b(vendors?|suppliers?)\b', question, re.I):
        scope.pop('line_id', None)
        scope.pop('contract_id', None)
    for field, values in (('supplier_id', vendors), ('contract_id', {c[1] for c in contracts}), ('line_id', cases)):
        found = _mentioned(question, values)
        if len(found) > 1:
            return scope, _reply('clarification', f'Choose one {field.replace("_id", "")} for this question.',
                                 choices=[{field: value} for value in found])
        if found:
            scope[field] = found[0]
    # Unknown explicit IDs must not silently widen into a portfolio query.
    tokens = re.findall(r'(?<![\w-])(?:SUP-[A-Za-z0-9_.:-]+|CTR-[A-Za-z0-9_.:-]+|L\d+[A-Za-z0-9_.:-]*|LEGACY-[A-Za-z0-9_.:-]+|SCALE-[A-Za-z0-9_.:-]+|DEMO-[A-Za-z0-9_.:-]+)', question, re.I)
    known = {value.casefold() for value in (*vendors, *(c[1] for c in contracts), *cases)}
    if any(token.casefold() not in known for token in tokens):
        return scope, _reply('not_found', 'That ID was not found in the selected demo tenant. Check the exact vendor, contract or case ID.')
    # Resolve names independently of IDs: an ID cannot silently suppress a
    # second named vendor. Remove known IDs before token matching so LEGACY case
    # identifiers do not accidentally match every vendor named "Legacy".
    name_question = question
    for value in sorted({*vendors, *(pair[1] for pair in contracts), *cases}, key=len, reverse=True):
        name_question = re.sub(r'(?<![\w.:-])' + re.escape(value) + r'(?![\w.:-])', ' ', name_question, flags=re.I)
    searchable = {key: [vendor['name']] for key, vendor in vendors.items()}
    for contact in contacts:
        searchable.setdefault(contact['supplier_id'], []).append(contact['poc_name'])
    words = set(re.findall(r'[a-z0-9]+', name_question.casefold()))
    stop = {'synthetic', 'services', 'supplier', 'suppliers', 'vendor', 'vendors', 'contract', 'contracts',
            'case', 'cases', 'invoice', 'invoices', 'the', 'for', 'show', 'find', 'list', 'search', 'display',
            'poc', 'name', 'contact', 'contacts', 'portfolio', 'overall', 'value', 'total', 'totals',
            'cash', 'recovery', 'recoveries', 'savings', 'approved', 'approval', 'approvals', 'draft',
            'pending', 'awaiting', 'rejected', 'evidence', 'required', 'review', 'reviews', 'current',
            'historical', 'legacy', 'migrated', 'all', 'year', 'years', 'count', 'number', 'our', 'team',
            'accounting', 'finance', 'auditing', 'manager', 'managers', 'this', 'that', 'selected'}
    matches = set()
    for key, names in searchable.items():
        for name in names:
            terms = [word for word in re.findall(r'[a-z0-9]+', name.casefold()) if len(word) >= 3 and word not in stop]
            if (terms and name.casefold() in name_question.casefold()) or any(
                    word not in stop and len(word) >= 3 and term.startswith(word)
                    for word in words for term in terms):
                matches.add(key)
    if scope.get('supplier_id') and matches:
        matches.add(scope['supplier_id'])
    if len(matches) > 1:
        return scope, _reply('clarification', 'Several vendors match. Choose the exact vendor before continuing.',
            choices=[{'supplier_id': key, 'vendor_name': vendors[key]['name']} for key in sorted(matches)])
    if matches:
        scope['supplier_id'] = next(iter(matches))
    if scope.get('line_id'):
        case = cases.get(scope['line_id'])
        if case is None:
            raise ValueError('The case is not in the selected tenant.')
        for field in ('supplier_id', 'contract_id'):
            if field in scope and scope[field] != case[field]:
                raise ValueError('The case conflicts with the selected vendor or contract.')
            scope[field] = case[field]
    if scope.get('supplier_id') and scope['supplier_id'] not in vendors:
        raise ValueError('The vendor is not in the selected tenant.')
    if scope.get('contract_id'):
        matches = [pair for pair in contracts if pair[1] == scope['contract_id']]
        if scope.get('supplier_id'):
            matches = [pair for pair in matches if pair[0] == scope['supplier_id']]
        if not matches:
            raise ValueError('The contract conflicts with the selected vendor or tenant.')
        if len(matches) > 1:
            return scope, _reply('clarification', 'Choose the vendor for this contract.',
                                 choices=[{'supplier_id': pair[0], 'contract_id': pair[1]} for pair in matches])
        scope['supplier_id'] = matches[0][0]
    if not scope:
        requested = re.search(r'(?<![-\w])(?:for|from)\s+(.+?)[?.!]*$', question, re.I)
        unknown_named = re.search(r'\b(?:vendor|supplier|contract|case|invoice)\s+(?:named\s+)?([A-Z][A-Za-z0-9_-]+)', question)
        if (requested and not re.fullmatch(r'(?:(?:all )?(?:vendors?|suppliers?|contracts?|cases?|invoices?|years)|draft(?: cases| invoices)?|review|approval|the portfolio|2024|2025|2026)', requested[1].strip(), re.I)) or unknown_named:
            return scope, _reply('not_found', 'No matching vendor, contact, contract or case was found. Use an exact ID or a recorded vendor/contact name.')
    return scope, None


def _summary(db, checkpoints, tenant, scope):
    """Same dashboard projection as Invoice Review, using read-only connections."""
    with _read(db) as conn:
        where, args = ['tenant_id=?'], [tenant]
        for field in ('supplier_id', 'contract_id', 'line_id'):
            if scope.get(field):
                where.append(field + '=?')
                args.append(scope[field])
        ids = [r[0] for r in conn.execute('SELECT line_id FROM invoice_lines WHERE '
                                        + ' AND '.join(where) + ' ORDER BY line_id', args)]
        cases = [evaluate_line(conn, tenant, line_id) for line_id in ids]
        selected = set(ids)
        events = [dict(r) for r in conn.execute(
            'SELECT * FROM review_events WHERE tenant_id=? ORDER BY recorded_at,event_id', (tenant,))
                  if r['line_id'] in selected]
        requests = [dict(r) for r in conn.execute('SELECT * FROM evidence_requests WHERE tenant_id=?', (tenant,))
                    if r['line_id'] in selected] if _exists(conn, 'evidence_requests') else []
        assessments = []
        if _exists(conn, 'assessment_revisions'):
            latest, digests = set(), {c['line_id']: c['case_digest'] for c in cases}
            for row in conn.execute('SELECT * FROM assessment_revisions WHERE tenant_id=? ORDER BY sequence DESC', (tenant,)):
                if row['line_id'] not in selected:
                    continue
                record = dict(row)
                record['case_json'] = json.loads(record['case_json'])
                record['validity'] = ('CURRENT' if row['line_id'] not in latest
                    and row['case_digest'] == digests[row['line_id']] else 'STALE')
                latest.add(row['line_id'])
                assessments.append(record)
    workflows = list_review_workflows(db, checkpoints, tenant=tenant, line_id=scope.get('line_id'))
    workflows = [w for w in workflows if w.get('state', {}).get('line_id') in selected]
    return summarize_cases(cases, events=events, workflows=workflows,
                           assessments=assessments, requests=requests), events, requests, workflows


def _validate_contract_sources(conn, tenant, scope):
    # Validate immutable provenance before any mutable status/date/scope filter
    # can hide a previously known amendment from the retrieved set.
    validate_source_projections(conn)
    if not active(conn):
        return
    known_scopes = [json.loads(row['scope_json']) for row in conn.execute(
        'SELECT DISTINCT scope_json FROM source_versions WHERE tenant_id=?', (tenant,))]
    for known_scope in known_scopes:
        if all(not scope.get(field) or known_scope.get(field) == scope[field]
               for field in ('supplier_id', 'contract_id')):
            validate_projections(conn, known_scope)


def _documents(conn, tenant, scope, as_of):
    _validate_contract_sources(conn, tenant, scope)
    clauses, args = ["tenant_id=?", "status='SIGNED'", "kind='RATE_AGREEMENT'", 'signed_date<=?'], [tenant, as_of]
    for field in ('supplier_id', 'contract_id'):
        if scope.get(field):
            clauses.append(field + '=?')
            args.append(scope[field])
    docs = [dict(r) for r in conn.execute('SELECT * FROM documents WHERE ' + ' AND '.join(clauses) + ' ORDER BY doc_id', args)]
    _check_sources(docs)
    return docs


def _check_sources(docs):
    for doc in docs:
        if hashlib.sha256(doc['body'].encode()).hexdigest() != doc['sha256']:
            raise ValueError('A source hash does not match. The answer is withheld until the evidence is checked.')
    if sum(len(d['body']) for d in docs) > 80000:
        raise ValueError('The complete signed evidence is too large for this answer. Select one contract or case.')


def _source_citations(docs, governing=()):
    citations = []
    by_id = {d['doc_id']: d for d in docs}
    for item in governing:
        doc = by_id.get(item['doc_id'])
        if doc is None or not item.get('quote') or item['quote'] not in doc['body']:
            raise ValueError('A governing quotation is unsupported. The answer is withheld.')
        citations.append({'doc_id': doc['doc_id'], 'quote': item['quote'], 'source_sha256': doc['sha256'], 'kind': 'governing'})
    covered = {c['doc_id'] for c in citations}
    for doc in docs:
        if doc['doc_id'] not in covered:
            citations.append({'doc_id': doc['doc_id'], 'quote': doc['body'], 'source_sha256': doc['sha256'], 'kind': 'supporting_signed_source'})
    return citations


def _validated_explanation(brief, case, documents):
    action = required_action(case)
    if brief.get('next_action') != ACTIONS[action]:
        raise ValueError('Unsupported explanation action; draft withheld.')
    return validate_fact_draft({'claims': brief['claims'], 'citations': brief['citations'],
                                'next_action': action}, case, documents)


def _case_answer(db, checkpoints, tenant, scope, vendors, contacts, use_llm, model):
    from .legacy_archive import get_archived_case
    archived = get_archived_case(db, tenant, scope['line_id'])
    if archived:
        row = {key: archived[key] for key in ('line_id', 'supplier_id', 'contract_id', 'origin_year', 'currency', 'historical_approved_amount', 'description')}
        return _reply('case', f"{row['line_id']} is an approved, locked synthetic {row['origin_year']} migration record. "
            f"Its historical approved amount is {amount(row['historical_approved_amount'], row['currency'])}. It does not require a new approval.",
            rows=[row], scope=scope, actions=[{'label': 'Open historical case', 'page': 'Invoice review',
                'line_id': row['line_id'], 'supplier_id': row['supplier_id'], 'contract_id': row['contract_id'], 'origin_year': row['origin_year']}])
    summary, events, requests, workflows = _summary(db, checkpoints, tenant, scope)
    if not summary['cases']:
        return _reply('not_found', 'No case matched this exact scope.', scope=scope)
    case = summary['cases'][0]
    view = assessment_view(case)
    with _read(db) as conn:
        _validate_contract_sources(conn, tenant, scope)
        documents = retrieve_evidence(conn, tenant, scope['line_id'])
    _check_sources(documents)
    facts = build_fact_catalog(case, documents)
    contact = next((c['poc_name'] for c in contacts if c['supplier_id'] == case['supplier_id']), 'Not recorded')
    paragraphs = [f"{vendors[case['supplier_id']]['name']} ({case['supplier_id']}), contract {case['contract_id']}. Vendor POC: {contact}.",
                  view['finding'], view['reason']]
    paragraphs.extend(f"{r['Calculation']}: {r['Amount']}." for r in view['calculations'])
    for service in view['services']:
        paragraphs.append(f"Service {service['Service entry']} on {service['Service date']}: {service['Approved hours']} approved hours "
            f"at {service['Approved hourly rate']} per hour; reviewed term {service['Reviewed term']} in {service['Source document']}.")
    paragraphs.append(f"Current review: {case['bucket_label']}. Snapshot: {case['as_of']}.")
    if events:
        latest = events[-1]
        paragraphs.append(f"Latest saved decision: {latest['decision']} by {latest['reviewer']}. "
                          + ('It applies to the current assessment.' if case.get('review_event_id') == latest['event_id']
                             else 'It does not authorize the current assessment; a fresh review may be needed.'))
    open_requests = [r for r in requests if r['status'] in {'OPEN', 'EVIDENCE_ATTACHED'}]
    if open_requests:
        paragraphs.append(f"{len(open_requests)} evidence request(s) remain open or awaiting disposition. Attaching evidence alone does not resolve a request.")
        paragraphs.extend(f"{r['request_type']}: {r['details']}" for r in open_requests)
    mode = 'offline_facts'
    notice = 'Python facts and original signed wording; no model call or review action.'
    if use_llm:
        from .ai import explain_case
        brief = explain_case(deepcopy(case), deepcopy(documents), provider='ollama', model=model)
        # Revalidate even a saved/provider adapter result before exposing any prose.
        checked = _validated_explanation(brief, case, documents)
        with _read(db) as conn:
            fresh_case = evaluate_line(conn, tenant, scope['line_id'])
            fresh_docs = retrieve_evidence(conn, tenant, scope['line_id'])
        fresh_summary, _, _, _ = _summary(db, checkpoints, tenant, scope)
        fresh_review = fresh_summary['cases'][0]
        review_fields = ('bucket', 'review_event_id', 'assessment_id', 'workflow_thread_id', 'blocking_request_count')
        if (fresh_case['case_digest'] != case['case_digest'] or fresh_docs != documents
                or any(fresh_review.get(key) != case.get(key) for key in review_fields)):
            raise ValueError('The case or evidence changed during generation. Refresh and ask again; the draft is withheld.')
        paragraphs.extend([checked['summary'], checked['next_action']])
        mode, notice = 'validated_ollama_facts', 'Local Ollama closed claims passed independent fact/citation checks; arbitrary prose accuracy is not asserted.'
    else:
        # Show a current saved explanation only after rerunning the fact checks.
        for workflow in workflows:
            state = workflow.get('state', {})
            brief = state.get('brief', {})
            if (state.get('case', {}).get('case_digest') == case['case_digest']
                    and brief.get('composition') == 'validated_fact_template'):
                try:
                    checked = _validated_explanation(brief, case, documents)
                except (ValueError, KeyError, TypeError):
                    continue
                paragraphs.append('Saved validated explanation: ' + checked['summary'])
                break
    paragraphs.append('New current realized recovery: USD 0.00. Approval is separate from supplier acceptance and cash collection.')
    row = {'line_id': case['line_id'], 'vendor_name': vendors[case['supplier_id']]['name'], 'supplier_id': case['supplier_id'],
           'contract_id': case['contract_id'], 'poc_name': contact, 'assessment': case['status'], 'review_status': case['bucket_label'],
           'currency': case['currency'], 'candidate_amount': case['known_candidate'], 'case_digest': case['case_digest'], 'as_of': case['as_of']}
    return _reply('case', '\n\n'.join(paragraphs), rows=[row], scope=scope,
        citations=_source_citations(documents, case['citations']), mode=mode, notice=notice,
        evidence_document_ids=[d['doc_id'] for d in documents], fact_count=len(facts),
        actions=[{'label': 'Open case', 'page': 'Invoice review', 'line_id': case['line_id']},
                 {'label': 'Open contract', 'page': 'Contract term review', 'supplier_id': case['supplier_id'], 'contract_id': case['contract_id']}])


def _vendor_answer(vendors, contracts, cases, contacts, scope):
    by_supplier = {c['supplier_id']: c for c in contacts}
    rows = [{'supplier_id': key, 'vendor_name': vendor['name'],
             'poc_name': by_supplier.get(key, {}).get('poc_name', 'Not recorded'),
             'contract_count': sum(pair[0] == key for pair in contracts),
             'case_count': sum(c['supplier_id'] == key for c in cases.values())}
            for key, vendor in sorted(vendors.items()) if not scope.get('supplier_id') or key == scope['supplier_id']]
    return _limited('vendors', 'Synthetic vendor directory and recorded points of contact. Names are fictional; no message is sent.', rows, scope=scope,
        actions=[{'label': 'Open vendor contracts', 'page': 'Contract term review', 'supplier_id': scope['supplier_id']}]
        if scope.get('supplier_id') else [])


def answer_question(db, checkpoints, question, *, tenant='DEMO', context=None,
                    use_llm=False, model='llama3.2:latest'):
    """Answer one bounded question; all data/approval/checkpoint reads are read-only.

    Expected unavailable provider/dependency/source conditions return WITHHELD,
    never invented output or a fallback falsely labelled as a model response.
    """
    if (not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION
            or any(ord(char) < 32 for char in question)):
        return _reply('invalid', f'Enter a question of 1–{MAX_QUESTION} characters, on one line.')
    if type(use_llm) is not bool or not isinstance(model, str) or not model.strip() or len(model) > 200:
        return _reply('invalid', 'Use a valid local model name and an explicit true/false local-generation option.')
    question = question.strip()
    if _WRITE.search(question) or _UNTRUSTED.search(question):
        return _reply('read_only', 'This assistant searches and explains records. Use Invoice Review for decisions, or Evidence Requests for evidence actions. It cannot execute instructions, SQL, approvals, payments or communications.')
    try:
        with _read(db) as conn:
            manifest = _guard(conn, tenant)
            vendors, contracts, cases = _catalog(conn, tenant)
            contacts = list_vendor_contacts(conn, tenant=tenant)
            explicit_contracts = _mentioned(question, {pair[1] for pair in contracts})
            if len(explicit_contracts) == 1:
                _validate_contract_sources(conn, tenant, {'contract_id': explicit_contracts[0]})
            scope, clarification = _scope(question, context, vendors, contracts, cases, contacts)
        if clarification:
            return clarification
        if re.search(r'\b(poc|contact|point of contact)\b', question, re.I):
            return _vendor_answer(vendors, contracts, cases, contacts, scope)
        if scope.get('line_id'):
            return _case_answer(db, checkpoints, tenant, scope, vendors, contacts,
                use_llm and bool(re.search(r'\b(explain|explanation|why|story|details?)\b', question, re.I)), model)
        lower = question.casefold()
        if re.search(r"\b(no|not|except|exclude|excluding|without|isn't|aren't|unapproved)\b", lower):
            return _reply('clarification', 'Use one positive status filter: approved, pending approval, rejected, or needs evidence.', scope=scope)
        if any(int(year) not in {2024, 2025, 2026} for year in re.findall(r'\b20[0-9]{2}\b', lower)):
            return _reply('clarification', 'The synthetic workspace contains 2024/2025 locked history and current 2026 cases. Choose one of these years.', scope=scope)
        if re.search(r'\b(evidence requests?|pending evidence|open requests?)\b', lower):
            if re.search(r'\b(resolved|closed|approved|rejected)\b', lower):
                return _reply('clarification', 'This search shows pending evidence requests only. Use one vendor or contract to narrow the pending queue.', scope=scope)
            _, _, requests, _ = _summary(db, checkpoints, tenant, scope)
            rows = [{key: request[key] for key in ('request_id', 'line_id', 'request_type', 'status', 'details')}
                    for request in requests if request['status'] in {'OPEN', 'EVIDENCE_ATTACHED'}]
            return _limited('evidence', 'Pending evidence requests. Attachments await explicit disposition; resolution and reassessment are separate steps.',
                rows, scope=scope, actions=[{'label': 'Open Evidence Requests', 'page': 'Evidence requests'}])
        statuses = [key for key, pattern in _STATUS.items() if re.search(pattern, lower)]
        if (len(statuses) > 1 or re.search(r"\b(not|except|exclude|excluding|without|isn't|aren't)\b", lower)):
            return _reply('clarification', 'Use one positive status filter: approved, pending approval, rejected, or needs evidence.', scope=scope)
        years = {int(y) for y in re.findall(r'\b(2024|2025|2026)\b', lower)}
        if re.search(r'\b(total|totals|cash|recoveries|recovery|portfolio value|approved value|savings)\b', lower) and not re.search(r'\b(cases|invoices)\b', lower):
            if scope or years or any(status != 'approved' for status in statuses):
                return _reply('clarification', 'Cash forecasts and historical receipts are currently available for the complete all-year USD portfolio. Ask for portfolio recovery totals, or ask for cases within this vendor/contract.', scope=scope)
            summary, _, _, _ = _summary(db, checkpoints, tenant, {})
            report = get_recovery_overview(db, checkpoints, tenant=tenant, summary=summary)
            rows = [{'measure': 'Total approved value', 'currency': 'USD', 'amount': report['total_approved_usd']},
                    {'measure': 'Estimated cash still to recover', 'currency': 'USD', 'amount': report['estimated_cash_usd']},
                    {'measure': 'Invented historical realized recovery', 'currency': 'USD', 'amount': report['realized_recovery_usd']},
                    {'measure': 'New current realized recovery', 'currency': 'USD', 'amount': report['current_realized_recovery_usd']}]
            return _reply('totals', '\n'.join(f"{r['measure']}: {amount(r['amount'], 'USD')}." for r in rows)
                + '\nApproved value is separate from cash. Historical demo receipts are excluded from the remaining estimate and are not actual collections.', rows=rows,
                notice=report['historical_receipts']['notice'])
        if re.search(r'\b(cases?|invoices?|pending|approved|rejected|evidence|review)\b', lower) or statuses:
            summary, _, _, _ = _summary(db, checkpoints, tenant, scope)
            bucket = statuses[0] if statuses else None
            current = [] if (years and 2026 not in years) or re.search(r'\b(historical|legacy|migrated)\b', lower) and not re.search(r'\ball\b', lower) else summary['cases']
            rows = [{'line_id': c['line_id'], 'supplier_id': c['supplier_id'], 'vendor_name': vendors[c['supplier_id']]['name'],
                     'contract_id': c['contract_id'], 'status': c['bucket_label'], 'assessment': c['status'],
                     'currency': c['currency'], 'candidate_amount': c['known_candidate'], 'locked': False}
                    for c in current if not bucket or c['bucket'] == bucket]
            if (years & {2024, 2025} or re.search(r'\b(historical|legacy|migrated|all years)\b', lower)) and bucket in {None, 'approved'}:
                from .legacy_archive import list_archived_cases
                archived = list_archived_cases(db, tenant, supplier_id=scope.get('supplier_id'), contract_id=scope.get('contract_id'))
                rows.extend({'line_id': c['line_id'], 'supplier_id': c['supplier_id'], 'vendor_name': c['supplier_name'],
                    'contract_id': c['contract_id'], 'status': 'Migrated approved — locked', 'currency': c['currency'],
                    'historical_approved_amount': c['historical_approved_amount'], 'origin_year': c['origin_year'], 'locked': True}
                    for c in archived if not years or c['origin_year'] in years)
            return _limited('cases', 'Cases in the selected scope. Unresolved amounts remain unknown; migrated approvals are locked.', rows, scope=scope,
                actions=[{'label': 'Open Invoice Review', 'page': 'Invoice review', **scope}])
        if scope.get('contract_id') or re.search(r'\b(contracts?|amendments?|terms?|rate|wording)\b', lower):
            with _read(db) as conn:
                if scope.get('contract_id'):
                    # A quoted answer requires the complete exact-contract set.
                    docs = _documents(conn, tenant, scope, manifest['as_of'])
                    source_counts = {}
                    for document in docs:
                        key = document['supplier_id'], document['contract_id']
                        source_counts[key] = source_counts.get(key, 0) + 1
                else:
                    validate_source_projections(conn)
                    # A directory is metadata only. Loading every portfolio body
                    # would needlessly trigger the complete-answer size guard.
                    clauses = ["tenant_id=?", "status='SIGNED'", "kind='RATE_AGREEMENT'", 'signed_date<=?']
                    args = [tenant, manifest['as_of']]
                    if scope.get('supplier_id'):
                        clauses.append('supplier_id=?')
                        args.append(scope['supplier_id'])
                    source_counts = {(r['supplier_id'], r['contract_id']): r['source_count']
                        for r in conn.execute('SELECT supplier_id,contract_id,count(*) AS source_count '
                            'FROM documents WHERE ' + ' AND '.join(clauses)
                            + ' GROUP BY supplier_id,contract_id', args)}
                    docs = []
            rows = [{'supplier_id': supplier, 'vendor_name': vendors[supplier]['name'], 'contract_id': contract,
                     'signed_documents': source_counts.get((supplier, contract), 0)}
                    for supplier, contract in sorted(contracts) if not scope.get('supplier_id') or supplier == scope['supplier_id']
                    if not scope.get('contract_id') or contract == scope['contract_id']]
            # Original wording is returned only within one exact contract; a directory is not a legal answer.
            citations = _source_citations(docs) if scope.get('contract_id') else []
            with _read(db) as conn:
                historical = {(r['supplier_id'], r['contract_id']): r['origin_year'] for r in conn.execute(
                    'SELECT supplier_id,contract_id,origin_year FROM legacy_contracts WHERE tenant_id=?', (tenant,))} if _exists(conn, 'legacy_contracts') else {}
            for row in rows:
                if (row['supplier_id'], row['contract_id']) in historical:
                    row['origin_year'] = historical[row['supplier_id'], row['contract_id']]
            actions = [{'label': 'Open contract', 'page': 'Contract term review', 'supplier_id': r['supplier_id'], 'contract_id': r['contract_id'],
                        **({'origin_year': r['origin_year']} if 'origin_year' in r else {})} for r in rows[:5]]
            return _limited('contracts', ('Complete known signed rate documents through ' + manifest['as_of']
                + ' are quoted below. Applicability depends on the invoice service date and reviewed terms.' if scope.get('contract_id')
                else 'Contract directory. Select an exact contract ID to read all its signed source wording. Historical contracts may have no current signed source text.'),
                rows, scope=scope, citations=citations, actions=actions)
        if scope.get('supplier_id') or re.search(r'\b(vendors?|suppliers?|poc|contact|point of contact)\b', lower):
            return _vendor_answer(vendors, contracts, cases, contacts, scope)
        return _reply('help', _HELP)
    except (ValueError, RuntimeError, OSError, sqlite3.Error, ImportError, KeyError, TypeError):
        # Do not disclose raw provider payloads, paths, SQL or arbitrary source instructions.
        return _reply('withheld', 'The answer could not be verified in the selected scope. Check the exact IDs and current evidence; if local generation was enabled, verify Ollama is available and retry.',
                      notice='No approval, evidence request, forecast or checkpoint was created or changed.')
