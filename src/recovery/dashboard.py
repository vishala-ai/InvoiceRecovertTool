"""Read-only case metrics. Review events and historical versions are not new cases.

The caller supplies engine assessments and persisted records; this module never
reads a database, invokes a model, or changes review state. Amounts are Decimal
strings. A USD portfolio view is supplied alongside separate currency subtotals;
there is deliberately no cross-currency grand total or cash forecast.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json

BUCKET_LABELS = {
    'approved': 'Approved for draft',
    'rejected': 'Rejected',
    'evidence': 'Needs evidence / review',
    'approval': 'Awaiting approval',
    'preparation': 'Prepare explanation',
    'reassessment': 'Reassessment required',
    'other': 'Other assessment',
}
_DECISIONS = {'APPROVED_FOR_DRAFT', 'REJECTED', 'NEEDS_MORE_EVIDENCE'}
_REQUEST_STATES = {'OPEN': 'open', 'EVIDENCE_ATTACHED': 'evidence_attached',
                   'RESOLVED': 'resolved'}
_ZERO = Decimal('0')
_CENT = Decimal('0.01')


def _amount(value):
    # Never coerce floats, bools, nulls, NaN or Infinity into financial values.
    if not isinstance(value, str):
        return None
    try:
        result = Decimal(value)
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= _ZERO else None


def _money(value):
    return format(value.quantize(_CENT, rounding=ROUND_HALF_UP), '.2f')


def _scope(record):
    return record.get('tenant_id', record.get('tenant')), record.get('line_id')


def _case(record):
    value = record.get('case', record.get('case_json', record.get('result')))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def _order(record, index):
    """Prefer persisted ordering; input order breaks ties for legacy records.

    Legacy workflow views have no creation timestamp. UI thread IDs contain
    their creation nanoseconds; callers must supply other legacy threads in
    creation order when neither a sequence nor a timestamp is available.
    """
    sequence = record.get('sequence', record.get('attempt_sequence'))
    if type(sequence) is int:
        return 2, sequence, index
    for field in ('started_at', 'created_at', 'recorded_at'):
        timestamp = record.get(field)
        if isinstance(timestamp, str):
            try:
                parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if parsed.utcoffset() is not None:
                    # Integer microseconds, without float timestamp arithmetic.
                    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
                    delta = parsed.astimezone(timezone.utc) - epoch
                    micros = ((delta.days * 86400 + delta.seconds) * 1000000
                              + delta.microseconds)
                    return 1, micros, index
            except ValueError:
                pass
    thread = record.get('thread_id', '')
    if isinstance(thread, str) and thread.startswith('ui:'):
        stamp = thread.split(':', 2)[1]
        if stamp.isdigit():
            return 1, int(stamp) // 1000, index
    return 0, index, index


def _latest(records):
    return max(enumerate(records), key=lambda pair: _order(pair[1], pair[0]))[1]


def _validated_brief(state, digest):
    brief = state.get('brief')
    if not isinstance(brief, dict) or not brief:
        return False
    if not state.get('use_llm'):
        return brief.get('mode') == 'offline_template'
    from .ai import PROMPT_VERSION
    checks = brief.get('validation', {})
    return (state.get('explanation_config', {}).get('prompt_version') == PROMPT_VERSION
            and brief.get('composition') == 'validated_fact_template'
            and brief.get('case_digest') == digest
            and isinstance(checks, dict) and checks.get('status') == 'PASSED'
            and checks.get('financial_consistency') is True
            and checks.get('citation_check') == 'PASSED'
            and checks.get('action_check') == 'PASSED')


def _candidate(case):
    # NEEDS_REVIEW's engine contribution of zero means unresolved, not no saving.
    if case.get('status') == 'NEEDS_REVIEW':
        return None
    if case.get('status') == 'CANDIDATE':
        return _amount(case.get('net_candidate'))
    return _ZERO


def _historical_amount(assessment, workflow, case):
    for value in (_case(assessment or {}), (workflow or {}).get('state', {}).get('case', {}),
                  case):
        if value.get('status') == 'CANDIDATE':
            return _candidate(value), value.get('currency') or 'UNSPECIFIED'
    return None, None


def summarize_cases(cases: list[dict], *, events: list[dict], workflows=(),
                    assessments=(), requests=()) -> dict:
    """Count each tenant/invoice line once using its latest applicable attempt.

    Cases are current engine results. Repeated inputs for one line are replaced
    by the latest sequence/input, never added. Assessment records may contain a
    decoded ``case``/``case_json`` or serialized ``case_json``. A STALE persisted
    assessment, mismatching current digest, or stale latest review is excluded
    from current dollar totals until reassessment. Open requests remain blocking
    across revisions until explicitly resolved; attaching evidence is insufficient.

    ``events`` must be the case-review journal, not term-review approvals.
    The journal is authoritative even after an interrupted checkpoint save.
    A fresh attempt suppresses a prior decision, including a prior approval.
    Explicit assessment revisions also require a matching workflow assessment ID,
    even when their calculated digest/amount is unchanged. Legacy reviews may
    remain applicable to the initial migrated baseline only.
    """
    selected = {}
    grouped = defaultdict(list)
    for case in cases:
        grouped[_scope(case)].append(case)
    for key, versions in grouped.items():
        selected[key] = _latest(versions)

    event_groups, workflow_groups, assessment_groups, request_groups = (
        defaultdict(list) for _ in range(4))
    for event in events:
        if event.get('decision') in _DECISIONS:
            event_groups[_scope(event)].append(event)
    for workflow in workflows:
        workflow_groups[_scope(workflow.get('state', {}))].append(workflow)
    for assessment in assessments:
        assessment_groups[_scope(assessment)].append(assessment)
    for request in requests:
        request_groups[_scope(request)].append(request)

    decorated = []
    request_counts = dict(open=0, evidence_attached=0, resolved=0, blocking_cases=0)
    for key, case in selected.items():
        digest = case.get('case_digest')
        assessment = _latest(assessment_groups[key]) if assessment_groups[key] else None
        workflow = _latest(workflow_groups[key]) if workflow_groups[key] else None
        state = workflow.get('state', {}) if workflow else {}
        saved_case = state.get('case', {})
        current_assessment = bool(assessment and assessment.get('validity') == 'CURRENT'
                                  and assessment.get('case_digest') == digest)
        revision_compatible = True
        if current_assessment:
            workflow_assessment = state.get('assessment_id')
            revision_compatible = (workflow_assessment == assessment.get('assessment_id')
                if workflow_assessment else not assessment.get('prior_assessment_id'))
        stale = bool(case.get('validity') == 'STALE' or (assessment and (
            assessment.get('validity') != 'CURRENT'
            or assessment.get('case_digest') != digest)))
        stale = stale or bool(not current_assessment and saved_case
                              and saved_case.get('case_digest') != digest)
        applicable = [event for event in event_groups[key]
                      if revision_compatible and digest and event.get('case_digest') == digest
                      and (workflow is None or event.get('thread_id') == workflow.get('thread_id'))]
        event = _latest(applicable) if applicable else None
        decision = event.get('decision') if event else None

        blocking = []
        # A retry returns the same request ID. Be defensive if a caller supplied
        # repeated snapshots; use its latest persisted state rather than events.
        deduped_requests = {}
        for index, request in enumerate(request_groups[key]):
            request_key = request.get('request_id', ('row', index))
            deduped_requests[request_key] = request
        for request in deduped_requests.values():
            status = request.get('status')
            if status in _REQUEST_STATES:
                request_counts[_REQUEST_STATES[status]] += 1
            if status in {'OPEN', 'EVIDENCE_ATTACHED'} and request.get('blocking', True):
                blocking.append(request)
        request_counts['blocking_cases'] += bool(blocking)

        if stale:
            bucket = 'reassessment'
        elif case.get('status') == 'NEEDS_REVIEW' or blocking or decision == 'NEEDS_MORE_EVIDENCE':
            bucket = 'evidence'
        elif decision == 'APPROVED_FOR_DRAFT' and case.get('status') == 'CANDIDATE':
            bucket = 'approved'
        elif decision == 'REJECTED':
            bucket = 'rejected'
        elif (case.get('status') == 'CANDIDATE' and workflow and revision_compatible
              and workflow.get('next') == ['approval'] and not state.get('outcome')
              and saved_case.get('case_digest') == digest and _validated_brief(state, digest)):
            bucket = 'approval'
        elif case.get('status') == 'CANDIDATE':
            bucket = 'preparation'
        else:
            bucket = 'other'
        amount = _candidate(case)
        historical, historical_currency = _historical_amount(assessment, workflow, case)
        decorated.append({**case, 'bucket': bucket, 'bucket_label': BUCKET_LABELS[bucket],
            'current_decision': decision if not stale else None,
            'review_event_id': event.get('event_id') if event and not stale else None,
            'assessment_id': assessment.get('assessment_id') if assessment else None,
            'workflow_thread_id': workflow.get('thread_id') if workflow else None,
            'known_candidate': _money(amount) if amount is not None else None,
            'historical_candidate': (_money(historical)
                                     if historical is not None and stale else None),
            'historical_currency': historical_currency if stale else None,
            'blocking_request_count': len(blocking)})

    counts = {bucket: 0 for bucket in BUCKET_LABELS}
    currencies = ({'USD'} | {case.get('currency') or 'UNSPECIFIED' for case in decorated}
                  | {case['historical_currency'] for case in decorated
                     if case['historical_currency'] is not None})
    per_currency = {}
    buckets = {bucket: {'count': 0, 'known_total': '0.00', 'unknown_count': 0,
                        'per_currency': {}} for bucket in BUCKET_LABELS}
    for currency in sorted(currencies):
        values = {key: _ZERO for key in ('current_candidate', 'approved_draft',
                  'rejected_candidate', 'held_candidate', 'historical_stale_candidate')}
        bucket_values = {bucket: {'count': 0, 'known_total': _ZERO, 'unknown_count': 0}
                         for bucket in BUCKET_LABELS}
        for case in decorated:
            if case['bucket'] == 'reassessment' and case['historical_currency'] == currency:
                historical = _amount(case['historical_candidate'])
                if historical is not None:
                    values['historical_stale_candidate'] += historical
            if (case.get('currency') or 'UNSPECIFIED') != currency:
                continue
            bucket = case['bucket']
            amount = _amount(case['known_candidate'])
            bucket_values[bucket]['count'] += 1
            bucket_values[bucket]['unknown_count'] += amount is None
            if amount is not None:
                bucket_values[bucket]['known_total'] += amount
                if case['status'] == 'CANDIDATE' and bucket != 'reassessment':
                    values['current_candidate'] += amount
                    target = {'approved': 'approved_draft', 'rejected': 'rejected_candidate',
                              'evidence': 'held_candidate'}.get(bucket)
                    if target:
                        values[target] += amount
        formatted_buckets = {}
        for bucket, value in bucket_values.items():
            formatted = {**value, 'known_total': _money(value['known_total'])}
            formatted_buckets[bucket] = formatted
            counts[bucket] += value['count']
            buckets[bucket]['count'] += value['count']
            buckets[bucket]['unknown_count'] += value['unknown_count']
            buckets[bucket]['per_currency'][currency] = formatted
            if currency == 'USD':
                buckets[bucket]['known_total'] = formatted['known_total']
        per_currency[currency] = {**{key: _money(value) for key, value in values.items()},
            'currency': currency, 'realized_recovery': '0.00', 'expected_cash': None,
            'bucket_totals': formatted_buckets}
    return {'case_count': len(decorated), 'counts': counts, 'buckets': buckets,
            'cases': decorated, 'totals': per_currency['USD'], 'per_currency': per_currency,
            'request_counts': request_counts,
            'notices': ['Approved-for-draft value is part of calculated candidates, not extra savings.',
                        'Rejected and held findings remain in the calculated candidate total.',
                        'Unknown amounts are not established; they are not zero savings.',
                        'Expected cash recovery is not estimated without a reviewed forecasting policy.',
                        'Contract-term approval is separate from approval of an invoice finding.',
                        'Reviewer labels are demo-only; these counts represent cases, not approvers.']}
