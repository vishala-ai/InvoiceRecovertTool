"""Accountant worklist and guarded decisions through the existing workflow.

Reading a worklist never prepares a review. A selection captures case/attempt
identity, not authority: every decision re-reads the canonical case and journal.
"""
from __future__ import annotations

from decimal import Decimal
import sqlite3
import time
import uuid

from .dashboard import _validated_brief, summarize_cases
from .engine import audit, evaluate_line
from .lifecycle import enabled, list_assessments, list_requests
from .review import DECISIONS
from .storage import connect
from .workflow import list_review_workflows, run_persistent

ERRORS = (ValueError, RuntimeError, ImportError, OSError, sqlite3.Error)
IDENTITY_FIELDS = ('line_id', 'supplier_id', 'contract_id', 'case_digest',
                   'assessment_id', 'workflow_thread_id', 'currency', 'net_candidate',
                   'status', 'known_candidate')


def load_worklist(db, checkpoints, *, tenant='DEMO', line_id=None):
    """Load current financial cases and saved review status without inference."""
    with connect(db) as conn:
        if line_id is None:
            report = audit(conn, tenant)
        else:
            case = evaluate_line(conn, tenant, line_id)
            report = {'tenant_id': tenant, 'synthetic': True, 'case_count': 1,
                'candidate_total': case['net_candidate'] if case['status'] == 'CANDIDATE' else '0.00',
                'currency': 'USD', 'counts': {case['status']: 1},
                'realized_recovery': '0.00', 'cases': [case]}
        where = 'tenant_id=?' + (' AND line_id=?' if line_id is not None else '')
        args = (tenant, line_id) if line_id is not None else (tenant,)
        events = [dict(row) for row in conn.execute(
            f'SELECT * FROM review_events WHERE {where} ORDER BY recorded_at,event_id', args)]
        vendors = {row['supplier_id']: row['name'] for row in conn.execute(
            'SELECT supplier_id,name FROM suppliers WHERE tenant_id=?', (tenant,))}
    workflows = list_review_workflows(db, checkpoints, tenant=tenant, line_id=line_id)
    migrated = enabled(db)
    summary = summarize_cases(report['cases'], events=events, workflows=workflows,
        assessments=list_assessments(db, tenant, line_id) if migrated else (),
        requests=list_requests(db, tenant, line_id) if migrated else ())
    by_thread = {workflow['thread_id']: workflow for workflow in workflows}
    by_event = {event['event_id']: event for event in events}
    for case in summary['cases']:
        case['vendor_name'] = vendors.get(case['supplier_id'], case['supplier_id'])
        workflow = by_thread.get(case['workflow_thread_id'])
        case['workflow'] = workflow
        case['committed'] = (by_event.get(workflow['thread_id'] + ':review')
                             if workflow else None)
        case['decision_error'] = decision_error(db, case, tenant=tenant)
        case['ready'] = case['decision_error'] is None
        case['completion_pending'] = bool(case['committed'] and workflow
            and workflow['next'] == ['approval'] and not workflow['state'].get('outcome'))
    return {'report': report, 'cases': summary['cases'], 'events': events, 'summary': summary}


def decision_error(db, case, *, tenant='DEMO'):
    """Preflight only; record_review remains the transactional final guard."""
    if case.get('status') != 'CANDIDATE' or Decimal(case['net_candidate']) <= 0:
        return 'This invoice needs evidence or has no eligible candidate amount.'
    if case.get('blocking_request_count'):
        return 'Resolve the open evidence request, then reassess this invoice.'
    if case.get('current_decision') or case.get('committed'):
        return 'A decision is already saved for this review.'
    workflow = case.get('workflow')
    if not workflow or workflow['next'] != ['approval'] or workflow['state'].get('outcome'):
        return 'Prepare a review before recording a decision.'
    state = workflow['state']
    if (state.get('tenant') != tenant or state.get('line_id') != case['line_id']
            or state.get('case', {}).get('case_digest') != case['case_digest']
            or case.get('bucket') != 'approval'
            or not _validated_brief(state, case['case_digest'])):
        return 'The saved review is out of date. Reassess and prepare a fresh review.'
    if state.get('retrieval_config') is not None:
        try:
            from .semantic_retrieval import validate_index_config
            with connect(db) as conn:
                validate_index_config(conn, tenant, state['retrieval_config'])
        except ERRORS:
            return 'The saved evidence search has changed or is unavailable. Prepare a fresh review.'
    return None


def selection_token(case):
    return {key: case.get(key) for key in IDENTITY_FIELDS}


def decide_selected(db, checkpoints, selections, *, supplier_id, contract_id,
                    reviewer, note, decision='approve', evidence_type='CONTRACT_RATE',
                    tenant='DEMO'):
    """Save each explicitly selected review separately and report partial outcomes.

    A batch cannot widen beyond one selected vendor/contract. No selected table
    amount is used in calculation or journal writes; each paused workflow resumes
    through its ordinary stable event and transactional stale-case checks.
    """
    if decision not in DECISIONS or not reviewer.strip() or not note.strip():
        raise ValueError('Enter a reviewer label and evidence review note.')
    if not supplier_id or not contract_id or not selections:
        raise ValueError('Select one vendor, one contract and at least one invoice.')
    ids = [item.get('line_id') for item in selections]
    if len(ids) != len(set(ids)):
        raise ValueError('An invoice may appear only once in a selection.')
    if any(item.get('supplier_id') != supplier_id or item.get('contract_id') != contract_id
           for item in selections):
        raise ValueError('All selected invoices must belong to the selected vendor and contract.')
    results = []
    for selected in selections:
        line = selected['line_id']
        attempted = False
        try:
            current = next((row for row in load_worklist(db, checkpoints, tenant=tenant, line_id=line)['cases']
                            if row['line_id'] == line), None)
            if current is None or selection_token(current) != selected:
                raise ValueError('The invoice or review changed after selection. Select it again.')
            committed = current['committed']
            if committed:
                if (committed['case_digest'] != current['case_digest']
                        or current.get('review_event_id') != committed['event_id']
                        or current.get('current_decision') != committed['decision']):
                    raise ValueError('The saved decision belongs to an earlier assessment; it cannot approve this invoice.')
                if (committed['decision'] != DECISIONS[decision]
                        or committed['reviewer'] != reviewer or committed['note'] != note):
                    raise ValueError('A different decision is already saved for this review.')
                if decision == 'request_more' and enabled(db):
                    with connect(db) as conn:
                        request = conn.execute('SELECT request_type FROM evidence_requests '
                            'WHERE tenant_id=? AND review_event_id=?',
                            (tenant, committed['event_id'])).fetchone()
                    if request is None or request['request_type'] != evidence_type:
                        raise ValueError('A different evidence request is already saved for this review.')
                if current['completion_pending']:
                    attempted = True
                    finish_committed(db, checkpoints, current, tenant=tenant)
                results.append({'line_id': line, 'outcome': 'Already saved',
                                'event_id': committed['event_id']})
                continue
            if current['decision_error']:
                raise ValueError(current['decision_error'])
            attempted = True
            result = run_persistent(db, checkpoints, current['workflow_thread_id'], tenant=tenant,
                decision={'decision': decision, 'reviewer': reviewer, 'note': note,
                          'evidence_type': evidence_type})
            results.append({'line_id': line, 'outcome': DECISIONS[decision],
                            'event_id': result['state']['outcome']['event_id']})
        except ERRORS as exc:
            # A checkpoint save can fail after record_review commits. The
            # journal wins; do not leave alternate decisions available for it.
            if not attempted:
                results.append({'line_id': line, 'outcome': 'Not saved', 'reason': str(exc)})
                continue
            with connect(db) as conn:
                event = conn.execute('SELECT * FROM review_events WHERE event_id=? '
                    'AND tenant_id=? AND line_id=? AND case_digest=?',
                    (str(selected.get('workflow_thread_id')) + ':review', tenant,
                     line, selected.get('case_digest'))).fetchone()
            if (event and event['thread_id'] == selected.get('workflow_thread_id')
                    and event['decision'] == DECISIONS[decision]
                    and event['reviewer'] == reviewer and event['note'] == note):
                results.append({'line_id': line, 'outcome': event['decision'],
                    'event_id': event['event_id'], 'completion_pending': True})
            else:
                results.append({'line_id': line, 'outcome': 'Not saved', 'reason': str(exc)})
    return results


def finish_committed(db, checkpoints, case, *, tenant='DEMO'):
    """Complete an interrupted checkpoint using only the exact journal payload."""
    event = case['committed']
    if not event or not case['completion_pending']:
        raise ValueError('No saved decision needs workflow completion.')
    evidence_type = 'CONTRACT_RATE'
    if enabled(db):
        with connect(db) as conn:
            row = conn.execute('SELECT request_type FROM evidence_requests '
                'WHERE tenant_id=? AND review_event_id=?', (tenant, event['event_id'])).fetchone()
        if row:
            evidence_type = row['request_type']
    return run_persistent(db, checkpoints, event['thread_id'], tenant=tenant,
        decision={'decision': next(key for key, value in DECISIONS.items()
                                   if value == event['decision']),
                  'reviewer': event['reviewer'], 'note': event['note'],
                  'evidence_type': evidence_type})


def reassess_case(db, line_id, *, action_id, tenant='DEMO'):
    from .lifecycle import reassess
    with connect(db) as conn:
        event = conn.execute('SELECT event_id FROM review_events WHERE tenant_id=? '
            'AND line_id=? ORDER BY recorded_at DESC,event_id DESC LIMIT 1',
            (tenant, line_id)).fetchone()
    return reassess(db, tenant, line_id, prior_review_event_id=event['event_id'] if event else None,
                    idempotency_key='invoice-reassess:' + action_id)


def prepare_review(db, checkpoints, case, *, action_id, tenant='DEMO', use_llm=False,
                   provider='ollama', model='llama3.2:latest',
                   base_url='http://127.0.0.1:11434', context_tokens=16384,
                   retrieval_index=None, allow_provider=False, allow_embeddings=False):
    if use_llm and not allow_provider:
        raise ValueError('Allow the selected explanation provider before preparing this review.')
    if retrieval_index is not None and not allow_embeddings:
        raise ValueError('Allow local embedding calls before preparing this review.')
    current = next(row for row in load_worklist(db, checkpoints, tenant=tenant, line_id=case['line_id'])['cases']
                   if row['line_id'] == case['line_id'])
    if selection_token(current) != selection_token(case) or current.get('blocking_request_count'):
        raise ValueError('The invoice changed or has an open evidence request. Resolve and reassess it.')
    if current['status'] != 'CANDIDATE':
        raise ValueError('Resolve the missing evidence before preparing an approval review.')
    if retrieval_index is not None:
        from .semantic_retrieval import resolve_index_config, validate_index_config
        config = resolve_index_config(retrieval_index)
        with connect(db) as conn:
            validate_index_config(conn, tenant, config)
        retrieval_index = config['index_path']
    if use_llm and enabled(db):
        from .ai import resolve_explanation_config
        # Explicit consent precedes provider metadata checks. Fail before
        # creating an assessment, durable intent, or checkpoint database.
        resolve_explanation_config(provider=provider, model=model or None,
                                   base_url=base_url, context_tokens=context_tokens)
    options = dict(use_llm=use_llm, provider=provider, model=model or None,
                   base_url=base_url, context_tokens=context_tokens, retrieval_index=retrieval_index)
    if enabled(db):
        from .lifecycle import start_linked_review
        assessment = reassess_case(db, case['line_id'], action_id=action_id, tenant=tenant)
        with connect(db) as conn:
            event = conn.execute('SELECT thread_id FROM review_events WHERE tenant_id=? AND line_id=? '
                'ORDER BY recorded_at DESC,event_id DESC LIMIT 1', (tenant, case['line_id'])).fetchone()
        return start_linked_review(db, checkpoints, tenant, assessment['assessment_id'],
            prior_thread_id=event['thread_id'] if event else None,
            idempotency_key='invoice-prepare:' + action_id, **options)
    return run_persistent(db, checkpoints, f'ui:{time.time_ns():020d}:{uuid.uuid4().hex}',
        tenant=tenant, line_id=case['line_id'], expected_case_digest=case['case_digest'], **options)


def retry_review(db, checkpoints, case, *, tenant='DEMO', allow_provider=False,
                 allow_embeddings=False):
    from .workflow import retry_preparation
    current = next(row for row in load_worklist(db, checkpoints, tenant=tenant, line_id=case['line_id'])['cases']
                   if row['line_id'] == case['line_id'])
    workflow = current.get('workflow')
    state = workflow['state'] if workflow else {}
    if (selection_token(current) != selection_token(case)
            or not workflow or not workflow['next'] or workflow['next'] == ['approval']
            or state.get('outcome') or state.get('case', {}).get('case_digest') != current['case_digest']
            or current.get('blocking_request_count') or current['status'] != 'CANDIDATE'
            or current.get('bucket') == 'reassessment'):
        raise ValueError('This preparation cannot be retried. Reassess and prepare a fresh review.')
    if state.get('use_llm') and not allow_provider:
        raise ValueError('Allow the saved explanation provider before retrying.')
    if state.get('retrieval_config') is not None and not allow_embeddings:
        raise ValueError('Allow local embedding calls before retrying.')
    return retry_preparation(db, checkpoints, workflow['thread_id'],
                             tenant=tenant, line_id=case['line_id'])
