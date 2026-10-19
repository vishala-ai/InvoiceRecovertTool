"""Fictional reviewer labels and a read-only view of actual journal activity."""
from __future__ import annotations

from pathlib import Path
import sqlite3

MANAGERS = (
    {'name': 'Elena Park', 'role': 'Audit Manager', 'focus': 'Evidence quality and review consistency'},
    {'name': 'Marcus Vale', 'role': 'Finance Escalation Manager', 'focus': 'Complex financial review questions'},
)
ACCOUNTANTS = tuple({'accountant_id': f'ACC-{index:03d}', 'name': name, 'focus': focus}
    for index, (name, focus) in enumerate((
        ('Maya Chen', 'Supplier rate review'), ('Arjun Mehta', 'Contract amendments'),
        ('Sofia Reyes', 'Pricing credits'), ('Noah Bennett', 'Service-date evidence'),
        ('Amara Okafor', 'Invoice reconciliation'), ('Leo Martins', 'Payment evidence'),
        ('Priya Nair', 'Contract scope'), ('Ethan Brooks', 'Draft quality review'),
        ('Lina Haddad', 'Evidence follow-up'), ('Oliver Reed', 'Review documentation'),
    ), 1))
CUSTOM_REVIEWER = 'CUSTOM'


def reviewer_label(accountant_id):
    person = next(person for person in ACCOUNTANTS if person['accountant_id'] == accountant_id)
    return f"{person['name']} · {person['accountant_id']} (fictional)"


def journal_activity(db, *, tenant='DEMO', reviewer=None, offset=0, limit=50):
    """Read saved events only; no directory entry is assigned retroactively."""
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('Journal pages must use a nonnegative offset and 1–100 rows')
    conn = sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA query_only=ON')
        where, values = 'tenant_id=?', [tenant]
        if reviewer is not None:
            where += ' AND reviewer=?'
            values.append(reviewer)
        counts = {row['decision']: row['n'] for row in conn.execute(
            f'SELECT decision,count(*) AS n FROM review_events WHERE {where} GROUP BY decision', values)}
        rows = [dict(row) for row in conn.execute(
            f'SELECT * FROM review_events WHERE {where} '
            'ORDER BY recorded_at DESC,event_id DESC LIMIT ? OFFSET ?', [*values, limit, offset])]
        return {'event_count': sum(counts.values()), 'counts': counts, 'events': rows}
    finally:
        conn.close()
