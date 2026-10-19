"""Local demo review journal; reviewer names are not authenticated identities."""
from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

from .engine import evaluate_line

DECISIONS = {'approve': 'APPROVED_FOR_DRAFT', 'reject': 'REJECTED',
             'request_more': 'NEEDS_MORE_EVIDENCE'}


def record_review(conn: sqlite3.Connection, *, event_id: str, thread_id: str,
                  tenant: str, line_id: str, decision: str, reviewer: str,
                  note: str, case_digest: str, evidence_type: str = 'CONTRACT_RATE',
                  assessment_id: str | None = None) -> dict:
    if decision not in DECISIONS:
        raise ValueError('Decision must be approve, reject, or request_more')
    if not all(x.strip() for x in (event_id, thread_id, reviewer, note, case_digest)):
        raise ValueError('A review requires IDs, reviewer, note, and the displayed case digest')
    # Serialize the freshness check and the append. No external side effect occurs.
    with conn:
        conn.execute('BEGIN IMMEDIATE')
        from .legacy_archive import assert_not_archived
        assert_not_archived(conn, tenant, line_id)
        existing = conn.execute('SELECT * FROM review_events WHERE event_id=?', (event_id,)).fetchone()
        if existing:
            identical = all(existing[k] == v for k, v in {
                'thread_id': thread_id, 'tenant_id': tenant, 'line_id': line_id,
                'decision': DECISIONS[decision], 'reviewer': reviewer, 'note': note,
                'case_digest': case_digest}.items())
            if not identical:
                raise ValueError('Review event ID was already used for a different decision')
            from .lifecycle_storage import active
            if decision == 'request_more' and active(conn):
                request=conn.execute('''SELECT request_type FROM evidence_requests WHERE tenant_id=?
                    AND review_event_id=?''',(tenant,event_id)).fetchone()
                if request and request['request_type']!=evidence_type:
                    raise ValueError('Review event ID already used for a different evidence request type')
            return dict(existing)
        current = evaluate_line(conn, tenant, line_id)
        if current['case_digest'] != case_digest:
            raise ValueError('STALE_CASE: refresh and review the new evidence before deciding')
        from .lifecycle_storage import active
        if active(conn):
            if assessment_id is None:
                linked=conn.execute('''SELECT result_json FROM operation_keys WHERE tenant_id=?
                    AND operation_key=?''',(tenant,'workflow-assessment:'+thread_id)).fetchone()
                if linked:
                    import json
                    assessment_id=json.loads(linked[0])['assessment_id']
                else:
                    # A pre-migration checkpoint has no assessment link. It may
                    # retain authority only over the initial migrated baseline,
                    # never inherit a later revision with an identical digest.
                    baseline=conn.execute('''SELECT assessment_id FROM assessment_revisions
                        WHERE tenant_id=? AND line_id=? ORDER BY sequence LIMIT 1''',
                        (tenant,line_id)).fetchone()
                    assessment_id=baseline[0] if baseline else None
            latest=conn.execute('''SELECT assessment_id FROM assessment_revisions WHERE tenant_id=?
                AND line_id=? ORDER BY sequence DESC LIMIT 1''',(tenant,line_id)).fetchone()
            if not latest or latest[0]!=assessment_id:
                raise ValueError('STALE_CASE: a newer assessment revision requires a new review')
        if decision == 'approve' and current['status'] != 'CANDIDATE':
            raise ValueError('Only an evidence-complete candidate can be approved for a draft')
        event = dict(event_id=event_id, thread_id=thread_id, tenant_id=tenant, line_id=line_id,
                     decision=DECISIONS[decision], reviewer=reviewer, note=note,
                     case_digest=case_digest, recorded_at=datetime.now(timezone.utc).isoformat())
        conn.execute('INSERT INTO review_events VALUES (?,?,?,?,?,?,?,?,?)', tuple(event.values()))
        if active(conn) and assessment_id:
            conn.execute('INSERT INTO case_review_assessments VALUES (?,?,?,?)',
                         (event_id,tenant,assessment_id,thread_id))
        if decision == 'request_more':
            from .lifecycle import request_for_review
            request_for_review(conn, event, evidence_type)
    return event
