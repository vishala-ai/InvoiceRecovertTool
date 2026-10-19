"""Read-only, tenant-scoped suggestions from the selected synthetic portfolio.

This is an entity catalogue, not a search engine or a source of financial facts.
It never creates schemas, assesses cases, calls providers or changes decisions.
UI controls may use the raw values while the chatbot uses supported questions.
"""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3

from .lifecycle_storage import validate_source_projections
from .vendor_contacts import list_vendor_contacts


_KEYS = ('vendors', 'vendor_ids', 'vendor_names', 'poc_names', 'vendor_search',
         'contracts', 'cases', 'invoices', 'documents', 'document_ids',
         'document_titles', 'evidence_requests', 'questions', 'question_labels')
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}\Z')
_ARCHIVE = {'legacy_archive_batches', 'legacy_suppliers', 'legacy_contracts', 'legacy_cases'}
_GLOBAL = ('Show portfolio totals', 'Find vendors and contacts', 'Show pending cases',
           'Show approved cases', 'Show rejected cases', 'Show cases requiring evidence',
           'Show pending evidence requests', 'Show contracts')


def _id(value, label):
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f'Invalid suggestion {label}.')
    return value


def _cohort(value):
    if type(value) is int:
        value = str(value)
    if value in ('current', '2026'):
        return 'current'
    if value in ('all', '2024', '2025'):
        return value
    raise ValueError('Suggestion cohort must be current, all, 2024, 2025 or 2026.')


def _sorted(values):
    return sorted(set(values), key=lambda value: (value.casefold(), value))


def _manifest(conn, tenant):
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        manifest = json.loads(row[0]) if row else None
    except (sqlite3.Error, ValueError, TypeError) as exc:
        raise ValueError('SYNTHETIC_ONLY: suggestions require a valid synthetic manifest.') from exc
    if not isinstance(manifest, dict) or manifest.get('synthetic') is not True:
        raise ValueError('SYNTHETIC_ONLY: suggestions require a synthetic portfolio.')
    if not isinstance(manifest.get('tenants'), dict) or tenant not in manifest['tenants']:
        raise ValueError('Suggestion tenant is not in the synthetic manifest.')
    validate_source_projections(conn)


def _archive(conn, tenant, tables):
    """Validate provenance before presenting immutable legacy identities."""
    if not tables.intersection(_ARCHIVE):
        return [], [], []
    if not _ARCHIVE.issubset(tables):
        raise ValueError('Synthetic archive suggestion scope is incomplete.')
    suppliers = [dict(row) for row in conn.execute(
        'SELECT * FROM legacy_suppliers WHERE tenant_id=?', (tenant,))]
    contracts = [dict(row) for row in conn.execute(
        'SELECT * FROM legacy_contracts WHERE tenant_id=?', (tenant,))]
    cases = [dict(row) for row in conn.execute(
        'SELECT * FROM legacy_cases WHERE tenant_id=?', (tenant,))]
    batches = {row['archive_id']: dict(row) for row in conn.execute('SELECT * FROM legacy_archive_batches')}
    supplier_scope = {(row['archive_id'], row['supplier_id']) for row in suppliers}
    contract_scope = {(row['archive_id'], row['supplier_id'], row['contract_id'], row['origin_year'])
                      for row in contracts}
    for row in (*suppliers, *contracts, *cases):
        batch = batches.get(row['archive_id'], {})
        if batch.get('synthetic') != 1:
            raise ValueError('SYNTHETIC_ONLY: archive suggestion provenance is unverified.')
    for row in contracts:
        if ((row['archive_id'], row['supplier_id']) not in supplier_scope
                or row['origin_year'] not in (2024, 2025)):
            raise ValueError('Archive contract suggestion scope is inconsistent.')
    for row in cases:
        if (row['synthetic'] != 1 or row['locked'] != 1 or row['status'] != 'MIGRATED_APPROVED'
                or (row['archive_id'], row['supplier_id'], row['contract_id'], row['origin_year']) not in contract_scope
                or row['source_system'] != batches[row['archive_id']]['source_system']):
            raise ValueError('Archive case suggestion scope is inconsistent.')
    return suppliers, contracts, cases


def load_suggestions(db, *, tenant='DEMO', cohort='current', supplier_id=None,
                     contract_id=None):
    """Return raw field values and supported chat prompts for one display scope.

    ``current``/``2026`` select operational records; ``2024``/``2025`` select
    locked archive records; ``all`` combines them. Optional vendor and contract
    filters cascade to cases, invoices, sources and pending evidence requests.
    No records from another tenant or cohort enter the result. Optional contact,
    archive and lifecycle tables need not exist in an original fixture database.
    """
    _id(tenant, 'tenant')
    cohort = _cohort(cohort)
    for label, value in (('vendor', supplier_id), ('contract', contract_id)):
        if value is not None:
            _id(value, label)
    path = Path(db)
    if not path.is_file():
        raise ValueError('Suggestions require an existing synthetic database.')
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        _manifest(conn, tenant)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        suppliers, contracts, cases, documents, requests = [], [], [], [], []
        if cohort in ('current', 'all'):
            suppliers = [dict(row) for row in conn.execute(
                'SELECT supplier_id,name FROM suppliers WHERE tenant_id=?', (tenant,))]
            cases = [dict(row) for row in conn.execute(
                'SELECT line_id,invoice_id,supplier_id,contract_id FROM invoice_lines WHERE tenant_id=?', (tenant,))]
            documents = [dict(row) for row in conn.execute(
                'SELECT doc_id,path,supplier_id,contract_id FROM documents WHERE tenant_id=?', (tenant,))]
            contracts = [{'supplier_id': row['supplier_id'], 'contract_id': row['contract_id']}
                         for row in (*cases, *documents)]
            if 'evidence_requests' in tables:
                requests = [dict(row) for row in conn.execute('''SELECT r.request_id,l.supplier_id,l.contract_id
                    FROM evidence_requests r JOIN invoice_lines l
                    ON l.tenant_id=r.tenant_id AND l.line_id=r.line_id
                    WHERE r.tenant_id=? AND r.status IN ('OPEN','EVIDENCE_ATTACHED')''', (tenant,))]
        if cohort != 'current':
            old_suppliers, old_contracts, old_cases = _archive(conn, tenant, tables)
            if cohort != 'all':
                old_contracts = [row for row in old_contracts if row['origin_year'] == int(cohort)]
                old_cases = [row for row in old_cases if row['origin_year'] == int(cohort)]
                selected_suppliers = {row['supplier_id'] for row in old_contracts}
                old_suppliers = [row for row in old_suppliers if row['supplier_id'] in selected_suppliers]
            suppliers.extend(old_suppliers)
            contracts.extend(old_contracts)
            cases.extend(old_cases)

        def selected(row):
            return ((supplier_id is None or row['supplier_id'] == supplier_id)
                    and (contract_id is None or row['contract_id'] == contract_id))

        contracts = [row for row in contracts if selected(row)]
        cases = [row for row in cases if selected(row)]
        documents = [row for row in documents if selected(row)]
        requests = [row for row in requests if selected(row)]
        contract_suppliers = {row['supplier_id'] for row in contracts}
        suppliers = [row for row in suppliers
                     if (supplier_id is None or row['supplier_id'] == supplier_id)
                     and (contract_id is None or row['supplier_id'] in contract_suppliers)]
        vendor_ids = {row['supplier_id'] for row in suppliers}
        contacts = {row['supplier_id']: row['poc_name'] for row in list_vendor_contacts(conn, tenant)
                    if row['supplier_id'] in vendor_ids}

    result = {key: [] for key in _KEYS}
    for row in sorted(suppliers, key=lambda row: row['supplier_id']):
        contact = contacts.get(row['supplier_id'])
        result['vendors'].append(' — '.join(filter(None, (row['supplier_id'], row['name'], contact))))
        result['vendor_ids'].append(row['supplier_id'])
        result['vendor_names'].append(row['name'])
        if contact:
            result['poc_names'].append(contact)
    result['vendor_search'] = _sorted([*result['vendor_ids'], *result['vendor_names'], *result['poc_names']])
    result['contracts'] = _sorted(row['contract_id'] for row in contracts)
    result['cases'] = _sorted(row['line_id'] for row in cases)
    result['invoices'] = _sorted(row['invoice_id'] for row in cases)
    result['documents'] = _sorted(f"{row['doc_id']} — {PurePosixPath(row['path']).name}" for row in documents)
    result['document_ids'] = _sorted(row['doc_id'] for row in documents)
    result['document_titles'] = _sorted(PurePosixPath(row['path']).name for row in documents)
    result['evidence_requests'] = _sorted(row['request_id'] for row in requests)
    # Labels are presentation-only search aliases. Submit the canonical ID
    # question, never the displayed name/POC, so repeated contact names cannot
    # silently widen the selected vendor or force an avoidable clarification.
    names = {row['supplier_id']: row['name'] for row in suppliers}

    def vendor_label(vendor):
        parts = [vendor, names.get(vendor, '')]
        if contacts.get(vendor):
            parts.append('POC: ' + contacts[vendor])
        return ' — '.join(part for part in parts if part)

    labels = {question: question for question in _GLOBAL}
    for vendor in result['vendor_ids']:
        for question in (f'Show contracts for {vendor}', f'Show cases for {vendor}',
                         f'Who is the POC for {vendor}?'):
            labels[question] = question + ' · ' + vendor_label(vendor)
    contract_vendors = {}
    for row in contracts:
        contract_vendors.setdefault(row['contract_id'], set()).add(row['supplier_id'])
    for contract in result['contracts']:
        vendors = sorted(contract_vendors[contract])
        for vendor in vendors:
            question = f'Explain contract {contract}'
            if len(vendors) > 1:
                question += f' for {vendor}'
            labels[question] = question + ' · ' + vendor_label(vendor)
    for row in sorted(cases, key=lambda row: row['line_id']):
        question = f"Explain case {row['line_id']}"
        labels[question] = (question + ' · ' + vendor_label(row['supplier_id'])
                            + f" · Contract: {row['contract_id']} · Invoice: {row['invoice_id']}")
    result['questions'] = list(labels)
    result['question_labels'] = list(labels.values())
    return result


def chatbot_suggestions(catalog, query='', *, limit=20):
    """Bound a literal, case-insensitive question match without running a query."""
    if (not isinstance(query, str) or len(query) > 2000
            or any(ord(character) < 32 for character in query)):
        raise ValueError('Suggestion query must be text of at most 2000 characters.')
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError('Suggestion limit must be between 1 and 50.')
    if not isinstance(catalog, dict) or not isinstance(catalog.get('questions'), list):
        raise ValueError('Suggestions require a loaded catalogue.')
    questions = catalog['questions']
    labels = catalog.get('question_labels', questions)
    if (not isinstance(labels, list) or len(labels) != len(questions)
            or any(not isinstance(value, str) for value in (*questions, *labels))):
        raise ValueError('Suggestion questions and labels must be parallel text lists.')
    term = query.strip().casefold()
    matches = [(position, question, label) for position, (question, label)
               in enumerate(zip(questions, labels)) if term in label.casefold() or term in question.casefold()]
    return [question for _, question, _ in sorted(
        matches, key=lambda item: (not item[2].casefold().startswith(term), item[0]))[:limit]]
