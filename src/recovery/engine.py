"""Deterministic pricing checks. No model, network, float arithmetic, or generated SQL."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from .storage import metadata, one_line, scoped_documents
from .lifecycle_storage import coverage, input_revision, open_requests, validate_projections

CENT = Decimal('0.01')
ZERO = Decimal('0')
SCOPE = ('tenant_id', 'supplier_id', 'contract_id', 'role', 'location')
TERM_SCOPE = (*SCOPE, 'currency', 'unit', 'service_type')


class ReviewRequired(ValueError):
    pass


def number(value: str) -> Decimal:
    if not isinstance(value, str):
        raise ReviewRequired('INVALID_NUMBER')
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ReviewRequired('INVALID_NUMBER') from exc
    if not result.is_finite() or result < ZERO:
        raise ReviewRequired('INVALID_NUMBER')
    return result


def money(value: Decimal) -> str:
    return format(value.quantize(CENT, rounding=ROUND_HALF_UP), '.2f')


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ReviewRequired('INVALID_DATE') from exc


def active_term(terms: list[dict], day: str, currency: str) -> dict:
    """Resolve explicit supersession within exact scope; never 'latest file wins'."""
    service_date = parse_date(day)
    eligible = [t for t in terms if parse_date(t['effective_from']) <= service_date
                and (not t['effective_to'] or parse_date(t['effective_to']) >= service_date)]
    if not eligible:
        raise ReviewRequired('MISSING_TERMS')
    by_id = {t['term_id']: t for t in terms}
    superseded: set[str] = set()
    for term in eligible:
        ancestor = term['supersedes']
        seen = {term['term_id']}
        while ancestor:
            if ancestor in seen or ancestor not in by_id:
                raise ReviewRequired('INVALID_SUPERSESSION_CHAIN')
            if any(by_id[ancestor][field] != term[field] for field in TERM_SCOPE):
                raise ReviewRequired('INVALID_SUPERSESSION_CHAIN')
            seen.add(ancestor)
            superseded.add(ancestor)
            ancestor = by_id[ancestor]['supersedes']
    winners = [t for t in eligible if t['term_id'] not in superseded]
    if len(winners) != 1:
        raise ReviewRequired('CONFLICTING_TERMS')
    winner = winners[0]
    if winner['currency'] != currency:
        raise ReviewRequired('TERM_CURRENCY_MISMATCH')
    if winner['unit'] != 'HOUR' or winner['service_type'] != 'REGULAR':
        raise ReviewRequired('UNSUPPORTED_TERM_SCOPE')
    return winner


def evaluate_line(conn: sqlite3.Connection, tenant: str, line_id: str) -> dict:
    from .legacy_archive import assert_not_archived
    assert_not_archived(conn, tenant, line_id)
    line = one_line(conn, tenant, line_id)
    manifest = json.loads(metadata(conn, 'manifest'))
    result = {
        'tenant_id': tenant, 'line_id': line_id, 'invoice_id': line['invoice_id'],
        'supplier_id': line['supplier_id'], 'contract_id': line['contract_id'],
        'currency': line['currency'], 'description': line['description'],
        'as_of': manifest['as_of'], 'source_digest': metadata(conn, 'source_digest'),
        'status': 'NEEDS_REVIEW', 'reason': '', 'billed_amount': line['billed_amount'],
        'expected_amount': None, 'gross_discrepancy': None, 'offsets': None,
        'net_candidate': '0.00', 'components': [], 'citations': [], 'po_rate': None,
        'warnings': [],
    }
    revision = input_revision(conn, line)
    if revision:
        result['input_revision'] = revision
    try:
        try:
            validate_projections(conn, line)
        except ValueError as exc:
            raise ReviewRequired(str(exc)) from exc
        if open_requests(conn, tenant, line_id):
            raise ReviewRequired('OPEN_EVIDENCE_REQUEST')
        _calculate(conn, line, manifest, result)
    except ReviewRequired as exc:
        result['status'] = 'NEEDS_REVIEW'
        result['reason'] = str(exc)
        result['net_candidate'] = '0.00'
    # Prevent approval of stale findings after re-import or rule changes.
    result['case_digest'] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def _calculate(conn: sqlite3.Connection, line: dict, manifest: dict, result: dict) -> None:
    tenant, lid = line['tenant_id'], line['line_id']
    as_of = parse_date(manifest['as_of'])
    complete = manifest['tenants'][tenant]
    for field in ('contracts_complete', 'payments_complete', 'adjustments_complete'):
        if complete.get(field) is not True:
            raise ReviewRequired('INCOMPLETE_SNAPSHOT')
    if line['currency'] != 'USD':
        raise ReviewRequired('UNSUPPORTED_CURRENCY')
    if line['unit'] != 'HOUR' or line['service_type'] != 'REGULAR':
        raise ReviewRequired('UNSUPPORTED_SCOPE')
    if parse_date(line['invoice_date']) > as_of:
        raise ReviewRequired('INVOICE_AFTER_SNAPSHOT')
    hours, billed_rate, billed = (number(line[x]) for x in
                                 ('billed_hours', 'billed_rate', 'billed_amount'))
    if hours == ZERO or billed == ZERO:
        raise ReviewRequired('ZERO_VALUE_LINE')
    if money(hours * billed_rate) != money(billed):
        raise ReviewRequired('INVOICE_EXTENSION_MISMATCH')
    paid = ZERO
    for payment in conn.execute('SELECT * FROM payments WHERE tenant_id=? AND line_id=?',
                                (tenant, lid)):
        if payment['status'] != 'SETTLED' or parse_date(payment['settled_date']) > as_of:
            continue
        if payment['currency'] != line['currency']:
            raise ReviewRequired('PAYMENT_CURRENCY_MISMATCH')
        paid += number(payment['amount'])
    if paid == ZERO:
        result.update(status='UNPAID', reason='NO_SETTLED_PAYMENT')
        return
    # Deliberately do not invent allocation of an overcharge to a partial payment.
    if money(paid) != money(billed):
        raise ReviewRequired('PARTIAL_OR_EXCESS_PAYMENT')
    services = [dict(r) for r in conn.execute(
        'SELECT * FROM service_entries WHERE tenant_id=? AND line_id=? ORDER BY service_date,entry_id',
        (tenant, lid))]
    if not services:
        raise ReviewRequired('MISSING_SERVICE_EVIDENCE')
    if any(s['approved'] != 'true' for s in services):
        raise ReviewRequired('UNAPPROVED_SERVICE_EVIDENCE')
    if any(parse_date(s['service_date']) > as_of for s in services):
        raise ReviewRequired('SERVICE_AFTER_SNAPSHOT')
    if sum((number(s['approved_hours']) for s in services), ZERO) != hours:
        raise ReviewRequired('HOURS_MISMATCH')
    docs = [d for d in scoped_documents(conn, line) if parse_date(d['signed_date']) <= as_of]
    try:
        covered, ignored = coverage(conn, docs)
    except ValueError as exc:
        raise ReviewRequired(str(exc)) from exc
    docs = [doc for doc in docs if doc['doc_id'] not in ignored]
    doc_by_id = {d['doc_id']: d for d in docs}
    terms = [dict(r) for r in conn.execute('''SELECT * FROM reviewed_terms
        WHERE tenant_id=? AND supplier_id=? AND contract_id=? AND role=? AND location=?''',
        tuple(line[k] for k in SCOPE)) if r['doc_id'] in doc_by_id]
    # Signed but unreviewed amendments cannot be silently omitted from the financial decision.
    reviewed_doc_ids = {t['doc_id'] for t in terms if t['reviewed'] == 'true' and t['reviewer']} | covered
    # Additive sources need explicit full-scope human coverage, not just one projected rate row.
    from .lifecycle_storage import active
    if active(conn):
        additive = {row[0] for row in conn.execute(
            "SELECT doc_id FROM source_versions WHERE tenant_id=? AND origin!='CURATED_FIXTURE'",
            (tenant,))}
        reviewed_doc_ids -= additive - covered
    if any(d['doc_id'] not in reviewed_doc_ids for d in docs):
        raise ReviewRequired('UNREVIEWED_SIGNED_DOCUMENT')
    terms = [t for t in terms if t['reviewed'] == 'true' and t['reviewer']]
    if not terms:
        raise ReviewRequired('MISSING_TERMS')
    expected = ZERO
    citations = {}
    for service in services:
        term = active_term(terms, service['service_date'], line['currency'])
        if term['source_quote'] not in doc_by_id[term['doc_id']]['body']:
            raise ReviewRequired('INVALID_TERM_PROVENANCE')
        units, rate = number(service['approved_hours']), number(term['rate'])
        expected += units * rate
        result['components'].append({
            'service_entry_id': service['entry_id'], 'service_date': service['service_date'],
            'hours': service['approved_hours'], 'rate': term['rate'],
            'term_id': term['term_id'], 'doc_id': term['doc_id'],
        })
        doc = doc_by_id[term['doc_id']]
        offset = doc['body'].index(term['source_quote'])
        first_line = doc['body'][:offset].count('\n') + 1
        citations[term['term_id']] = {'term_id': term['term_id'], 'doc_id': doc['doc_id'],
            'path': doc['path'], 'sha256': doc['sha256'], 'line_start': first_line,
            'line_end': first_line + term['source_quote'].count('\n'),
            'quote': term['source_quote']}
    expected = Decimal(money(expected))
    gross = max(ZERO, billed - expected)
    result.update(expected_amount=money(expected), gross_discrepancy=money(gross),
                  citations=list(citations.values()))
    po = conn.execute('SELECT * FROM purchase_orders WHERE tenant_id=? AND po_id=?',
                      (tenant, line['po_id'])).fetchone()
    if po and all(po[k] == line[k] for k in ('supplier_id', 'contract_id', 'role', 'currency', 'unit')):
        result['po_rate'] = po['po_rate']
    offsets = ZERO
    for adjustment in conn.execute('SELECT * FROM adjustments WHERE tenant_id=? AND line_id=?',
                                   (tenant, lid)):
        if parse_date(adjustment['effective_date']) > as_of:
            result['warnings'].append(f'Future adjustment excluded: {adjustment["adjustment_id"]}')
            continue
        if adjustment['status'] == 'VOID':
            continue
        if adjustment['status'] != 'APPLIED':
            raise ReviewRequired('PENDING_ADJUSTMENT')
        if adjustment['currency'] != line['currency']:
            raise ReviewRequired('ADJUSTMENT_CURRENCY_MISMATCH')
        if adjustment['reason'] != 'RATE_PRICING' or adjustment['kind'] not in (
                'APPLIED_CREDIT', 'RECEIVED_REFUND'):
            raise ReviewRequired('AMBIGUOUS_ADJUSTMENT')
        offsets += number(adjustment['amount'])
    result['offsets'] = money(offsets)
    if offsets > gross:
        raise ReviewRequired('OFFSETS_EXCEED_RATE_DISCREPANCY')
    net = max(ZERO, gross - offsets)
    if gross == ZERO:
        result.update(status='NO_DISCREPANCY', reason='BILLED_AT_OR_BELOW_APPROVED_RATE')
    elif net == ZERO:
        result.update(status='ALREADY_OFFSET', reason='EXISTING_CREDIT_OR_REFUND')
    else:
        result.update(status='CANDIDATE', reason='POTENTIAL_RATE_OVERPAYMENT', net_candidate=money(net))


def audit(conn: sqlite3.Connection, tenant: str = 'DEMO') -> dict:
    rows = conn.execute('SELECT line_id FROM invoice_lines WHERE tenant_id=? ORDER BY line_id',
                        (tenant,)).fetchall()
    cases = [evaluate_line(conn, tenant, row['line_id']) for row in rows]
    total = sum((Decimal(c['net_candidate']) for c in cases if c['status'] == 'CANDIDATE'), ZERO)
    return {'tenant_id': tenant, 'synthetic': True, 'case_count': len(cases),
            'candidate_total': money(total), 'currency': 'USD',
            'counts': dict(Counter(c['status'] for c in cases)),
            'realized_recovery': '0.00', 'cases': cases}
