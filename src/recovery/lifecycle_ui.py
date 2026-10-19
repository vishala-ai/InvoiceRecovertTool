"""Readable localhost views for the synthetic evidence and review lifecycle.

The database is authoritative. Rendering a page never generates proposals, approves
terms, starts a workflow, or saves a reviewer decision.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .presentation import amount
from .storage import connect
from .ui_guidance import column_help, read_only_table, table_intro
from .workspace_ui import TEAM_PAGE, page_description


PAGES = ('Overview', 'Contract term review', 'Evidence requests', TEAM_PAGE)
EDIT_FIELDS = (
    ('role', 'Service role'), ('location', 'Service location'),
    ('currency', 'Currency'), ('unit', 'Unit'), ('service_type', 'Service type'),
    ('rate', 'Hourly rate'), ('effective_from', 'Effective from (YYYY-MM-DD)'),
    ('effective_to', 'Effective to (optional, YYYY-MM-DD)'),
    ('supersedes', 'Explicitly supersedes reviewed term ID (optional)'),
)
ACTION_ERRORS = (ValueError, RuntimeError, ImportError, OSError, sqlite3.Error)
TERM_STATUS = {'UNREVIEWED_PROPOSAL': 'Awaiting human term review',
               'TERMS_APPROVED': 'Terms approved for calculation input',
               'TERMS_REJECTED': 'Proposal rejected',
               'PROPOSAL_STALE': 'Source context changed — a fresh proposal is required'}
REQUEST_STATUS = {'OPEN': 'Open — evidence is still missing',
                  'EVIDENCE_ATTACHED': 'Evidence attached — reviewer disposition required',
                  'RESOLVED': 'Resolved by a reviewer'}
REQUEST_TYPES = {'CONTRACT_RATE': 'Contract or rate terms', 'PAYMENT': 'Payment evidence',
                 'CREDIT': 'Credit or refund evidence', 'SERVICE': 'Approved service records',
                 'OTHER': 'Other evidence'}


def _key(action, *values):
    """Stable action identity survives reruns; conflicting payloads remain distinct."""
    encoded = json.dumps(values, sort_keys=True, ensure_ascii=False, default=str).encode()
    return 'ui:' + action + ':' + hashlib.sha256(encoded).hexdigest()


def _rows(db_path, query, arguments=()):
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute(query, arguments)]


def _notice(st, db_path):
    notice = st.session_state.pop(f'lifecycle_notice:{Path(db_path).resolve()}', None)
    if notice:
        st.success(notice)


def _action(st, db_path, label, callback, *, after=None):
    try:
        with st.spinner(label):
            result = callback()
        if after is not None:
            after(result)
        st.session_state[f'lifecycle_notice:{Path(db_path).resolve()}'] = label + ' Saved locally.'
        st.rerun()
        return result
    except ACTION_ERRORS as exc:
        st.error(str(exc))
    return None


def _reviewer(st, prefix):
    st.caption('Reviewer labels identify demo history only; they are not authenticated identities.')
    reviewer = st.text_input('Reviewer label (demo only)', key=prefix + ':reviewer')
    note = st.text_area('Review rationale', key=prefix + ':note')
    return reviewer.strip(), note.strip()


def _navigate(st, page, db_path, *, line_id=None, doc_id=None):
    """Only selects a view; callback runs before the navigation widget rerenders."""
    st.session_state['recovery_page'] = page
    if line_id is not None:
        st.session_state[f'invoice_selection:{Path(db_path).resolve()}'] = line_id
        if page == 'Invoice review':
            st.session_state[f'invoice_cohort:{Path(db_path).resolve()}'] = 'Current · 2026'
            st.session_state[f'invoice_last_cohort:{Path(db_path).resolve()}'] = None
            st.session_state[f'invoice_review_scope:{Path(db_path).resolve()}'] = 'All cases'
            st.session_state.pop(f'invoice_statuses:{Path(db_path).resolve()}:All cases', None)
            for name in ('invoice_vendor_search', 'invoice_case_search', 'invoice_last_open',
                         'invoice_table_descriptor'):
                st.session_state.pop(f'{name}:{Path(db_path).resolve()}', None)
    if doc_id is not None:
        st.session_state['lifecycle:terms:document'] = doc_id


def _filter_overview(st, bucket):
    st.session_state['lifecycle:overview:active-bucket'] = bucket


def _close_overview_cases(st):
    st.session_state.pop('lifecycle:overview:active-bucket', None)


def _align_request_reassessment(st, db_path, request):
    """Selecting a different request selects its invoice and prior decision too."""
    key = f'lifecycle:reassess:request:{Path(db_path).resolve()}'
    if st.session_state.get(key) != request['request_id']:
        line_id = request['line_id']
        st.session_state['lifecycle:reassess:line'] = line_id
        st.session_state[f'lifecycle:reassess:prior:{line_id}'] = request['review_event_id']
        st.session_state[key] = request['request_id']


def _source(st, document):
    st.text(document['body'])


def select_table_row(st, rows, *, key, row_ids=None, preferred_id=None, page_label=None,
                     table_title=None, table_description=None):
    """Select an actual dataframe row without reusing another filtered row's position."""
    if not rows:
        return None
    identities = list(range(len(rows))) if row_ids is None else list(row_ids)
    if len(identities) != len(rows):
        raise ValueError('Table identities must match its visible rows')
    from .invoice_ui import page_bounds
    offset, limit = page_bounds(st, len(rows), key=_key('pages', key, identities),
                                label=page_label or 'Table')
    visible = rows[offset:offset + limit]
    visible_ids = identities[offset:offset + limit]
    table_intro(st, table_title or page_label or 'Available records',
        table_description or 'Review the records below and select one row to inspect its details. '
        'Hover over a column heading for its meaning.', row_count=len(visible), total_count=len(rows))
    table_key = _key('table', key, visible_ids)
    event = st.dataframe(visible, hide_index=True, use_container_width=True,
        on_select='rerun', selection_mode='single-row', key=table_key,
        column_config=column_help(st, visible[0]))
    selected = event.get('selection', {}).get('rows', [])
    if len(selected) == 1 and type(selected[0]) is int and 0 <= selected[0] < len(visible):
        return offset + selected[0]
    return identities.index(preferred_id) if preferred_id in visible_ids else None


def _proposal_fields(st, values, prefix):
    edited = {}
    for name, label in EDIT_FIELDS:
        edited[name] = st.text_input(label, value=values.get(name) or '',
                                      key=prefix + ':' + name).strip() or None
    edited['exact_quote'] = st.text_area('Exact supporting quotation',
        value=values.get('exact_quote', ''), key=prefix + ':quote')
    questions = st.text_area('Unresolved applicability questions (one per line)',
        value='\n'.join(values.get('unresolved_questions', [])), key=prefix + ':questions')
    edited['unresolved_questions'] = [line.strip() for line in questions.splitlines() if line.strip()]
    edited['coverage'] = st.selectbox('Review completeness',
        ['INCOMPLETE', 'COMPLETE_SINGLE_RATE'],
        index=1 if values.get('coverage') == 'COMPLETE_SINGLE_RATE' else 0,
        format_func=lambda value: {'INCOMPLETE': 'Incomplete — additional review required',
            'COMPLETE_SINGLE_RATE': 'One complete regular-hour rate and exact scope'}[value],
        key=prefix + ':coverage')
    return edited


def render_overview(st, db_path, *, tenant='DEMO'):
    from .dashboard import BUCKET_LABELS
    from .forecast import LABEL as FORECAST_LABEL
    from .historical_recovery import get_recovery_overview
    from .ui_cache import display_worklist

    page_description(st, 'Overview')
    checkpoint_path = Path(db_path).with_name(Path(db_path).stem + '-ui-checkpoints.sqlite3')
    summary = display_worklist(st, db_path, checkpoint_path, tenant=tenant)['summary']
    overview = get_recovery_overview(db_path, checkpoint_path, summary=summary, tenant=tenant)
    forecast = overview['forecast']
    st.html("""<style>
        [data-testid="stAppDeployButton"] { display: none; }
        .st-key-overview-usd-portfolio {
            background-color: #eff6ff;
            border: 1px solid #dbeafe;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            box-sizing: border-box;
        }
        .st-key-overview-usd-portfolio h3 { color: #1e40af; }
        .st-key-overview-usd-portfolio [data-testid="stMetricLabel"] p,
        .st-key-overview-usd-portfolio [data-testid="stMetricValue"] { color: #1e293b; }
        .st-key-overview-usd-portfolio [data-testid="stCaptionContainer"] p { color: #475569; }
        .st-key-overview-portfolio-scope, .st-key-overview-review-workload,
        .st-key-overview-case-list {
            border-radius: 12px; padding: 1.25rem 1.5rem; box-sizing: border-box;
        }
        .st-key-overview-portfolio-scope {
            background: #ecfdf5; border: 1px solid #a7f3d0;
        }
        .st-key-overview-portfolio-scope h3 { color: #047857; }
        .st-key-overview-review-workload {
            background: #fffbeb; border: 1px solid #fde68a;
        }
        .st-key-overview-review-workload h3 { color: #6b21a8; }
        .st-key-overview-case-list { background: #f8fafc; border: 1px solid #e2e8f0; }
        .st-key-overview-case-list h3 { color: #334155; }
        .st-key-overview-review-workload [data-testid="stMetricValue"],
        .st-key-overview-portfolio-scope [data-testid="stMetricValue"] { color: #1e293b; }
        .st-key-overview-workload-approved { border-top: 4px solid #10b981; background: #f0fdf4; }
        .st-key-overview-workload-rejected { border-top: 4px solid #f43f5e; background: #fff1f2; }
        .st-key-overview-workload-evidence { border-top: 4px solid #f59e0b; background: #fffbeb; }
        .st-key-overview-workload-approval { border-top: 4px solid #3b82f6; background: #eff6ff; }
        .st-key-overview-workload-preparation { border-top: 4px solid #64748b; background: #f8fafc; }
        .st-key-overview-workload-reassessment { border-top: 4px solid #8b5cf6; background: #f5f3ff; }
        .st-key-overview-workload-other { border-top: 4px solid #14b8a6; background: #f0fdfa; }
        .st-key-overview-review-workload [class*="st-key-overview-workload-"] {
            border-radius: 10px; padding: .9rem; height: 100%;
        }
    </style>""")
    with st.container(key='overview-usd-portfolio'):
        st.subheader('GRAND TOTALS (USD)')
        st.caption('Compare the value approved for drafting with the cash still expected and the '
                   'completed synthetic recovery history. These are separate measures across all years.')
        approved_value, expected_cash, realized = st.columns(3)
        approved_value.metric('Total approved value', amount(overview['total_approved_usd'], 'USD'),
            help='Supported USD values approved for drafting, including locked approved historical '
                 'records. Approval authorizes a draft; it does not establish cash received.')
        expected_cash.metric(FORECAST_LABEL, amount(overview['estimated_cash_usd'], 'USD')
            if overview['estimated_cash_usd'] is not None else 'Not estimated',
            help='Remaining expected USD cash based on the saved planning percentage and eligible '
                 'approved records. Completed historical receipts are excluded to prevent double counting. '
                 'A missing or stale planning assumption withholds the estimate.')
        realized.metric('Realized recovery', amount(overview['realized_recovery_usd'], 'USD'),
            help='Invented receipts recorded in the separate synthetic historical scenario. '
                 'No actual cash collection is asserted, and current operational recovery remains zero.')
        if overview['historical_receipts']['case_count']:
            st.caption('Realized recovery includes invented 2024–2025 demo receipts, at 100% of their '
                       'already-discounted estimate. Those completed records are excluded from '
                       'estimated cash recovery. No real cash receipts are asserted.')
        if forecast['status'] == 'CURRENT':
            st.caption(f"Remaining estimated cash: {forecast['assumption']['recovery_percentage']}% × "
                       f"{amount(overview['outstanding_approved_usd'], 'USD')}.")
        elif forecast['status'] == 'STALE_BASIS':
            st.warning('The approved-case basis changed. The remaining estimate is withheld until '
                       'a reviewed forecast is saved for the current approved records.')
    portfolio = _rows(db_path, '''SELECT DISTINCT supplier_id,contract_id FROM invoice_lines
        WHERE tenant_id=?''', (tenant,))
    if _rows(db_path, "SELECT name FROM sqlite_master WHERE type='table' AND name='legacy_contracts'"):
        portfolio.extend(_rows(db_path, '''SELECT DISTINCT supplier_id,contract_id
            FROM legacy_contracts WHERE tenant_id=?''', (tenant,)))
    with st.container(key='overview-portfolio-scope'):
        st.subheader('PORTFOLIO COVERAGE')
        st.caption('See the supplier and contract coverage behind the portfolio totals, across current '
                   'records and locked migrated history.')
        approved_records, suppliers, contracts = st.columns(3)
        approved_cases = forecast['basis']['cases']
        current_approved = sum(case['basis_kind'] == 'CURRENT_APPROVED_DRAFT'
                               for case in approved_cases)
        historical_approved = sum(case['basis_kind'] == 'MIGRATED_APPROVED'
                                  for case in approved_cases)
        approved_records.metric('Total approved records', f"{forecast['basis']['case_count']:,}",
            help='Eligible USD approved records across all years supporting the Grand Totals: '
                 'current approved drafts plus locked migrated approvals, including completed '
                 'historical records. Historical records remain read-only.')
        approved_records.caption(f'{current_approved:,} current / {historical_approved:,} historical')
        suppliers.metric('Suppliers', len({row['supplier_id'] for row in portfolio}),
            help='Distinct supplier IDs represented by current invoice lines or locked historical '
                 'contracts in the selected demo scope. A supplier is counted once across all years.')
        contracts.metric('Contracts', len({(row['supplier_id'], row['contract_id']) for row in portfolio}),
            help='Distinct supplier-and-contract combinations across all years. Signed amendments '
                 'remain part of their parent contract and do not increase this count.')
        st.caption('Across all years. Amendments belong to their parent contract and are not extra contracts.')
    with st.container(key='overview-review-workload'):
        st.subheader('REVIEW WORKLOAD')
        st.caption('Current cases only. Migrated approved history is locked and needs no approval. '
                   'Use the queues to plan the next review action. Choose View cases to inspect a queue; '
                   'awaiting approval cases are available on Invoice Review.')
        workload_help = {
            'approved': 'Current cases with a valid approval for drafting. This is a case count, '
                        'not a count of approvers or collected payments.',
            'rejected': 'Current cases with a valid reviewer rejection. Their amounts are not '
                        'included in approved draft value.',
            'evidence': 'Current cases blocked by incomplete evidence or a request for further review. '
                        'Obtain the missing evidence before seeking approval.',
            'approval': 'Current cases with a prepared validated explanation paused for human review. '
                        'Open Invoice Review to approve, reject or request more evidence.',
            'preparation': 'Current supported candidates that need a validated explanation and a '
                           'human-review pause before a decision can be recorded.',
            'reassessment': 'Cases whose evidence or review context changed. Run a fresh assessment '
                            'before relying on a prior decision.',
            'other': 'Current cases outside the actionable review queues, such as no discrepancy, '
                     'already offset or unpaid assessments. No supported draft approval is implied.',
        }
        keys = list(BUCKET_LABELS)
        for start in (0, 4):
            for column, key in zip(st.columns(4), keys[start:start + 4], strict=False):
                with column.container(key='overview-workload-' + key):
                    label = 'Current approved for draft' if key == 'approved' else BUCKET_LABELS[key]
                    st.metric(label, summary['counts'].get(key, 0),
                              help=workload_help[key])
                    if key != 'approval':
                        st.button('View cases', key=f'lifecycle:overview:card:{key}',
                            on_click=_filter_overview, args=(st, key), width='stretch')
        st.caption('Each invoice line is counted once. Counts represent cases, not individual approvers.')
    selected = st.session_state.get('lifecycle:overview:active-bucket')
    if selected == 'approval':
        _close_overview_cases(st)
        return
    if selected not in BUCKET_LABELS:
        return
    with st.container(key='overview-case-list'):
        st.subheader(BUCKET_LABELS[selected].upper() + ' CASES')
        st.button('Close cases', on_click=_close_overview_cases, args=(st,),
                  key='lifecycle:overview:close-cases')
        cases = [case for case in summary['cases'] if case['bucket'] == selected]
        if not cases:
            st.info('There are no cases in this queue.')
            return
        rows = [{'Invoice line': case['line_id'], 'Supplier': case['supplier_id'],
            'Work queue': BUCKET_LABELS[case['bucket']],
            'Assessment': case['status'].replace('_', ' ').capitalize(),
            'Candidate': amount(case['net_candidate'], case['currency'])
                if case.get('known_candidate') is not None else 'Not established',
            'Current decision': case.get('current_decision') or 'No current decision'}
            for case in cases]
        chosen = select_table_row(st, rows, key=f'overview:{Path(db_path).resolve()}:{selected}',
            row_ids=[case['line_id'] for case in cases], page_label='Overview cases',
            table_title='Cases in this work queue',
            table_description='Each row is one current invoice-line case in the selected queue. '
            'Compare its financial assessment and latest valid decision, then select a row to open Invoice Review. '
            'Candidate amounts describe supported discrepancies, not collected cash.')
        st.button('Open selected invoice', disabled=chosen is None,
            on_click=_navigate, args=(st, 'Invoice review', db_path),
            kwargs={'line_id': cases[chosen]['line_id'] if chosen is not None else None},
            key='lifecycle:overview:open-invoice')


def _render_saved_proposals(st, db_path, tenant, chosen, document):
    from . import lifecycle

    proposals = [item for item in lifecycle.list_proposals(db_path, tenant) if item['doc_id'] == chosen]
    if not proposals:
        st.info('No saved term suggestion is awaiting review for this document.')
        return
    proposal_ids = [item['proposal_id'] for item in proposals]
    pending = st.session_state.pop(f'lifecycle:proposal:pending:{chosen}', None)
    if pending in proposal_ids:
        st.session_state[f'lifecycle:proposal:{chosen}'] = pending
    selected = st.selectbox('Saved proposal revision', proposal_ids,
        format_func=lambda proposal_id: next(
            f"Revision {item['revision']} · {TERM_STATUS.get(item['status'], item['status'])}"
            for item in proposals if item['proposal_id'] == proposal_id), key=f'lifecycle:proposal:{chosen}')
    proposal = next(item for item in proposals if item['proposal_id'] == selected)
    values = proposal['values']
    st.subheader(f"Saved proposal — revision {proposal['revision']}")
    st.text('Status: ' + TERM_STATUS.get(proposal['status'], proposal['status']))
    review = proposal.get('review') or {}
    if review:
        st.text(f"Decision reviewer (demo only): {review.get('reviewer', 'Not recorded')}")
        st.text(f"Review rationale: {review.get('note', 'Not recorded')}")
        st.caption('This is the saved term-review decision. It does not approve an invoice finding.')
    scope = proposal.get('scope', {})
    read_only_table(st, [{'Scope field': key.replace('_', ' ').capitalize(), 'Bound value': str(value)}
        for key, value in scope.items()], title='Proposal applicability',
        description='These saved boundaries identify the supplier, contract and service context '
        'for this proposal. Check them against the original document before reviewing a rate.')
    read_only_table(st, [{'Proposed term': label, 'Saved value': values.get(name) or 'Not established'}
        for name, label in EDIT_FIELDS], title='Saved proposed terms',
        description='This matrix shows the exact saved revision being reviewed. Proposed values '
        'are not approved calculation inputs until a reviewer explicitly approves complete terms.')
    if values.get('unresolved_questions'):
        st.warning('Unresolved applicability questions block approval.')
        for question in values['unresolved_questions']:
            st.text(question)
    if values.get('coverage') != 'COMPLETE_SINGLE_RATE':
        st.warning('This proposal does not cover one complete supported regular-hour rate. '
                   'Unhandled rates, exceptions and pricing clauses must remain unresolved.')
    st.markdown('**Exact supporting quotation**')
    st.text(values.get('exact_quote', ''))
    with st.expander('Edit a new revision for review'):
        with st.form(f'lifecycle:edit:{selected}'):
            edited = _proposal_fields(st, values, f'lifecycle:edit:{selected}')
            editor, edit_note = _reviewer(st, f'lifecycle:edit:{selected}')
            save = st.form_submit_button('Save Edits for Review',
                disabled=proposal['status'] != 'UNREVIEWED_PROPOSAL')
            if proposal['status'] != 'UNREVIEWED_PROPOSAL':
                st.caption('This saved revision is complete or outdated. A new suggestion needs separate review.')
        if save:
            if not editor or not edit_note:
                st.error('Enter a demo reviewer label and correction rationale to save a new revision.')
            else:
                _action(st, db_path, 'Edited revision saved.', lambda: lifecycle.edit_proposal(
                    db_path, tenant, selected, values=edited, reviewer=editor, note=edit_note,
                    idempotency_key=_key('edit', selected, edited, editor, edit_note)),
                    after=lambda result: st.session_state.__setitem__(
                        f'lifecycle:proposal:pending:{chosen}', result['proposal_id']))
    st.caption('Approve or reject the saved revision shown above. Edits inside the form are not '
               'part of that revision until Save Edits for Review is clicked.')
    with st.form(f'lifecycle:terms:decision:{selected}'):
        reviewer, note = _reviewer(st, f'lifecycle:decision:{selected}')
        coverage = st.checkbox('I checked the complete source: this proposal covers its one regular-hour '
            'rate and exact scope, including dates and explicit supersession.',
            key=f'lifecycle:coverage:{selected}')
        can_approve = (proposal['status'] == 'UNREVIEWED_PROPOSAL'
                       and values.get('coverage') == 'COMPLETE_SINGLE_RATE'
                       and not values.get('unresolved_questions') and document.get('status') == 'SIGNED')
        approve = st.form_submit_button('Approve Terms', disabled=not can_approve)
        reject = st.form_submit_button('Reject Proposal',
                                       disabled=proposal['status'] != 'UNREVIEWED_PROPOSAL')
    if approve or reject:
        if not reviewer or not note:
            st.error('Enter a demo reviewer label and review rationale before saving a term decision.')
        elif approve and not coverage:
            st.error('Confirm the complete source and supported scope before approving terms.')
        else:
            decision = 'approve' if approve else 'reject'
            _action(st, db_path, 'Term decision recorded.', lambda: lifecycle.review_terms(
                db_path, tenant, selected, proposal_digest=proposal['digest'], decision=decision,
                reviewer=reviewer, note=note, coverage_confirmed=coverage,
                idempotency_key=_key('term-review', selected, proposal['digest'], decision,
                                     reviewer, note, coverage)))


def render_term_review(st, db_path, *, tenant='DEMO'):
    from . import lifecycle
    from .chatbot_navigation import consume_chat_target
    from .ui_cache import display_worklist
    from .vendor_contacts import list_vendor_contacts

    page_description(st, 'Contract term review')
    from .invoice_ui import _search_matches, cohort_filter, render_archive
    chat_target = consume_chat_target(st, db_path, tenant=tenant)
    year = cohort_filter(st, Path(db_path), 'contracts')
    if year is not None:
        render_archive(st, Path(db_path), tenant=tenant, origin_year=year, namespace='contracts')
        return
    st.caption('Select a vendor, a contract document and an affected case. '
               'Approving rate terms is separate from approving an invoice.')
    documents = [doc for doc in lifecycle.list_documents(db_path, tenant)
                 if doc['kind'] == 'RATE_AGREEMENT']
    lines = _rows(db_path, 'SELECT * FROM invoice_lines WHERE tenant_id=? ORDER BY line_id', (tenant,))
    supplier_names = {row['supplier_id']: row['name'] for row in _rows(db_path,
        'SELECT supplier_id,name FROM suppliers WHERE tenant_id=?', (tenant,))}
    contacts = {row['supplier_id']: row['poc_name']
                for row in list_vendor_contacts(db_path, tenant)}
    suppliers = sorted({line['supplier_id'] for line in lines} | {doc['supplier_id'] for doc in documents})
    terms = _rows(db_path, 'SELECT * FROM reviewed_terms WHERE tenant_id=? ORDER BY effective_from,term_id', (tenant,))
    prefix = f'lifecycle:terms:{Path(db_path).resolve()}:{tenant}'
    if chat_target and chat_target['origin_year'] is None:
        prefix += ':chat:' + chat_target['token']
    else:
        chat_target = None
    from .search_ui import autocomplete_search

    vendor_query = autocomplete_search(st, 'Search vendor name or ID',
        [value for supplier in suppliers for value in (supplier, supplier_names.get(supplier))],
        key=prefix + ':vendor_search')
    suppliers = [supplier for supplier in suppliers
                 if _search_matches(vendor_query, supplier, supplier_names.get(supplier))]
    preferred_doc = st.session_state.get('lifecycle:terms:document')
    if preferred_doc is None and chat_target and chat_target.get('contract_id'):
        preferred_doc = next((doc['doc_id'] for doc in documents
            if doc['supplier_id'] == chat_target['supplier_id']
            and doc['contract_id'] == chat_target['contract_id']), None)
    preferred = next((doc for doc in documents if doc['doc_id'] == preferred_doc), None)
    vendor_rows = [{'Vendor': supplier_names.get(supplier, supplier), 'Vendor ID': supplier,
        'Vendor POC (fictional)': contacts.get(supplier, 'Not recorded'),
        'Contracts': len({line['contract_id'] for line in lines if line['supplier_id'] == supplier}),
        'Contract documents': sum(doc['supplier_id'] == supplier for doc in documents)}
        for supplier in suppliers]
    st.subheader('Vendors')
    vendor_index = select_table_row(st, vendor_rows, key=prefix + ':vendors:' + vendor_query,
        row_ids=suppliers, preferred_id=(preferred['supplier_id'] if preferred else
            chat_target['supplier_id'] if chat_target else None),
        page_label='Vendors', table_title='Supplier directory',
        table_description='Each row groups a supplier with its fictional point of contact and '
        'recorded contract coverage. Select a vendor to narrow the documents and related cases below. '
        'Contact names do not confer approval authority.')
    if vendor_index is None:
        st.info('Select a vendor row to see its contracts and amendments.')
        return
    supplier = suppliers[vendor_index]
    st.caption('Selected vendor: ' + supplier_names.get(supplier, supplier))
    choices = [doc for doc in documents if doc['supplier_id'] == supplier]
    present = {doc['contract_id'] for doc in choices}
    for contract in sorted({line['contract_id'] for line in lines if line['supplier_id'] == supplier} - present):
        choices.append({'doc_id': None, 'contract_id': contract, 'supplier_id': supplier})
    contract_query = autocomplete_search(st, 'Search contract or document',
        [value for doc in choices for value in (doc['contract_id'], doc.get('doc_id'), doc.get('title'))],
        key=prefix + ':contract_search:' + supplier)
    choices = [doc for doc in choices if _search_matches(contract_query,
        doc['contract_id'], doc.get('doc_id'), doc.get('title'))]
    if not choices:
        st.info('No contract documents match this search.')
        return
    choices.sort(key=lambda doc: (doc['contract_id'], doc.get('signed_date', ''), doc['doc_id'] or ''))
    document_ids = [doc['doc_id'] or 'missing:' + doc['contract_id'] for doc in choices]
    contract_rows = []
    for doc in choices:
        approved = [term for term in terms if term['doc_id'] == doc['doc_id'] and term['reviewed'] == 'true']
        contract_rows.append({'Contract': doc['contract_id'],
            'Document': doc['doc_id'] or 'Contract evidence missing',
            'Document status': doc.get('status', 'Not available').replace('_', ' ').capitalize(),
            'Signed date': doc.get('signed_date') or 'Not recorded',
            'Role': doc.get('role', 'See related cases'), 'Location': doc.get('location', 'See related cases'),
            'Rate terms': 'Approved terms available' if approved else 'Terms need review'})
    st.subheader('Contracts and amendments')
    preferred_document_id = preferred_doc
    if preferred_document_id is None and chat_target and chat_target.get('contract_id'):
        preferred_document_id = 'missing:' + chat_target['contract_id']
    contract_index = select_table_row(st, contract_rows, key=prefix + ':contracts:' + supplier,
        row_ids=document_ids, preferred_id=preferred_document_id, page_label='Contract documents',
        table_title='Contract and amendment register',
        table_description='Review the selected supplier’s contract documents, signature status and '
        'rate-review readiness. Each amendment appears separately under its parent contract. '
        'Selecting a document narrows related cases to that contract; it does not exclude other '
        'applicable amendments from calculations.')
    document = choices[contract_index] if contract_index is not None else None
    checkpoints = Path(db_path).with_name(Path(db_path).stem + '-ui-checkpoints.sqlite3')
    report = display_worklist(st, db_path, checkpoints, tenant=tenant)['report']
    related_ids = {line['line_id'] for line in lines if line['supplier_id'] == supplier
        and (document is None or line['contract_id'] == document['contract_id'])}
    related = [case for case in report['cases'] if case['line_id'] in related_ids]
    invoice_query = autocomplete_search(st, 'Search related invoice',
        [value for case in related for value in
         (case['line_id'], case['invoice_id'], case['contract_id'])],
        key=prefix + ':invoice_search:' + supplier)
    related = [case for case in related if _search_matches(invoice_query,
        case['line_id'], case['invoice_id'], case['contract_id'], case['description'])]
    st.subheader('Related cases')
    st.caption('All cases for this vendor are shown until a contract row is selected. '
               'A selected contract includes every service role; calculations retain all applicable amendments.')
    if related:
        index = select_table_row(st, [{'Invoice': case['invoice_id'], 'Invoice line': case['line_id'],
            'Contract': case['contract_id'],
            'Financial assessment': case['status'].replace('_', ' ').capitalize(),
            'Candidate amount': amount(case['net_candidate'], case['currency'])
                if case['status'] != 'NEEDS_REVIEW' else 'Not established'} for case in related],
            key=prefix + ':cases:' + supplier + ':' + (document['contract_id'] if document else 'all'),
            row_ids=[case['line_id'] for case in related], page_label='Related cases',
            table_title='Invoice cases linked to this selection',
            table_description='Each row links an invoice line to the selected supplier or contract. '
            'The assessment is calculated from current evidence; “Not established” means an '
            'evidence gap prevents a supported amount. Select a case to continue in Invoice Review.')
        st.button('Open selected invoice', disabled=index is None, on_click=_navigate,
            args=(st, 'Invoice review', db_path),
            kwargs={'line_id': related[index]['line_id'] if index is not None else None},
            key=prefix + ':open-invoice')
    else:
        st.info('No invoice cases share this contract.')
    if document is None:
        st.info('Select a contract or amendment row to inspect its rate terms.')
        return
    chosen = document['doc_id']
    with st.expander('Inspect selected contract terms'):
        approved = [term for term in terms if term['doc_id'] == chosen and term['reviewed'] == 'true']
        if approved:
            st.caption('Existing manually reviewed demonstration terms. These were not automatically extracted by AI.')
            read_only_table(st, [{'Role': term['role'], 'Location': term['location'],
                'Hourly rate': amount(term['rate'], term['currency']), 'Service': term['service_type'].capitalize(),
                'Effective from': term['effective_from'], 'Effective to': term['effective_to'] or 'No end date',
                'Replaces earlier rate': 'Yes — explicit replacement recorded' if term['supersedes'] else 'No',
                'Reviewed by': term['reviewer']} for term in approved], title='Approved rate terms',
                description='These manually reviewed terms are available to the calculation engine. '
                'Service date, role, location and explicit replacement determine which rate applies. '
                'A term approval does not approve an invoice or establish a recovery.')
            for term in approved:
                st.markdown('**Supporting contract wording**')
                st.text(term['source_quote'])
        else:
            st.info('No approved rate terms are available for this document. Affected cases remain subject to evidence checks.')
        if chosen:
            with st.expander('Read original contract document'):
                _source(st, document)
            _render_saved_proposals(st, db_path, tenant, chosen, document)


def _supporting_intake(st, db_path, tenant, line, request=None):
    """Keep guarded synthetic admission beside the case that needs its evidence."""
    from . import lifecycle

    prefix = _key('supporting-evidence', str(Path(db_path).resolve()), tenant, line['line_id'],
                  request['request_id'] if request else None)
    with st.container(key=prefix + ':panel'):
        st.caption('Synthetic TXT or Markdown only. Adding a document does not approve its rates or clear a case.')
        upload = st.file_uploader('Synthetic supporting document', type=['txt', 'md'], key=prefix + ':upload')
        synthetic = st.checkbox('I confirm this document contains synthetic demonstration data only', key=prefix + ':synthetic')
        if st.button('Save document for review', disabled=upload is None or not synthetic, key=prefix + ':stage'):
            _action(st, db_path, 'Supporting document saved.', lambda: lifecycle.stage_document(
                db_path, tenant, filename=upload.name, content=upload.getvalue(), synthetic_confirmed=synthetic))
        stages = lifecycle.list_stages(db_path, tenant)
        if not stages:
            return
        selected = st.selectbox('Saved supporting document', [stage['stage_id'] for stage in stages],
            format_func=lambda value: next(stage['original_filename'] for stage in stages if stage['stage_id'] == value),
            key=prefix + ':staged')
        staged = next(stage for stage in stages if stage['stage_id'] == selected)
        with st.expander('Read supporting document'):
            st.text(staged['body'])
        scope = {key: line[key] for key in ('supplier_id', 'contract_id', 'role', 'location')}
        st.caption(f"For {line['line_id']} · {line['contract_id']} · {line['role']} · {line['location']}")
        admission_prefix = prefix + ':' + selected
        with st.form(admission_prefix + ':admit'):
            title = st.text_input('Document title', value=staged['original_filename'])
            status = st.selectbox('Document status', ['DRAFT', 'SIGNED'],
                format_func=lambda value: 'Unsigned draft' if value == 'DRAFT' else 'Simulated signed agreement')
            signed_date = st.text_input('Signed date (YYYY-MM-DD)')
            attest = st.checkbox('I checked the selected case and contract scope.')
            signed_attest = st.checkbox('For signed documents, I confirm this is a synthetic signature simulation.')
            reviewer, note = _reviewer(st, admission_prefix + ':admission')
            submit = st.form_submit_button('Add document to this case')
        if submit:
            if not synthetic or not attest or (status == 'SIGNED' and not signed_attest):
                st.error('Confirm synthetic data, the case scope and the separate signed-document simulation.')
            elif not reviewer or not note:
                st.error('Enter the reviewer label and a reason for adding this evidence.')
            else:
                def admit():
                    source = lifecycle.admit_document(db_path, tenant, selected, scope=scope, title=title,
                        signed_date=signed_date, status=status, synthetic_confirmed=synthetic,
                        signed_simulation_confirmed=signed_attest, reviewer=reviewer, note=note,
                        idempotency_key=_key('support-admit', selected, scope, title, signed_date, status, reviewer, note))
                    if request is not None:
                        lifecycle.attach_evidence(db_path, tenant, request['request_id'], source['version_id'],
                            reviewer, note, _key('support-attach', request['request_id'], source['version_id'], reviewer, note))
                    return source
                _action(st, db_path, 'Supporting evidence added; term review is still required.', admit)


def render_requests(st, db_path, *, tenant='DEMO'):
    from . import lifecycle
    from .engine import audit, evaluate_line

    page_description(st, 'Evidence requests', compact=True)
    st.caption('Only open requests and requests awaiting evidence review appear here. '
               'Resolved requests stay in local history; reassessment continues on Invoice review.')
    requests = [request for request in lifecycle.list_requests(db_path, tenant)
                if request['status'] in ('OPEN', 'EVIDENCE_ATTACHED')]
    lines = _rows(db_path, 'SELECT * FROM invoice_lines WHERE tenant_id=? ORDER BY line_id', (tenant,))
    by_line = {line['line_id']: line for line in lines}
    names = {row['supplier_id']: row['name'] for row in _rows(db_path,
        'SELECT supplier_id,name FROM suppliers WHERE tenant_id=?', (tenant,))}
    prefix = f'lifecycle:evidence:{Path(db_path).resolve()}:{tenant}'
    target = st.session_state.get(f'lifecycle:evidence:target:{Path(db_path).resolve()}:{tenant}')
    target_line = by_line.get(target)
    if st.session_state.get(prefix + ':aligned') != target and target_line:
        st.session_state[prefix + ':vendor'] = target_line['supplier_id']
        st.session_state[prefix + ':contract'] = target_line['contract_id']
        st.session_state[prefix + ':aligned'] = target
    suppliers = sorted({line['supplier_id'] for line in lines})
    vendor_filter, contract_filter = st.columns(2)
    vendor = vendor_filter.selectbox('Vendor', [''] + suppliers,
        format_func=lambda value: f'{names.get(value, value)} ({value})' if value else 'All vendors',
        key=prefix + ':vendor')
    contracts = sorted({line['contract_id'] for line in lines if not vendor or line['supplier_id'] == vendor})
    if st.session_state.get(prefix + ':contract') not in [''] + contracts:
        st.session_state[prefix + ':contract'] = ''
    contract = contract_filter.selectbox('Contract', [''] + contracts,
        format_func=lambda value: value or 'All contracts', key=prefix + ':contract')
    def in_scope(line):
        return (not vendor or line['supplier_id'] == vendor) and (not contract or line['contract_id'] == contract)
    filtered = [request for request in requests if request['line_id'] in by_line and in_scope(by_line[request['line_id']])]
    st.subheader('Pending evidence requests')
    preferred = next((request['request_id'] for request in filtered if request['line_id'] == target), None)
    selected = select_table_row(st, [{'Vendor': names.get(by_line[item['line_id']]['supplier_id'], by_line[item['line_id']]['supplier_id']),
        'Contract': by_line[item['line_id']]['contract_id'], 'Invoice line': item['line_id'],
        'Evidence needed': REQUEST_TYPES.get(item['request_type'], item['request_type']),
        'Status': REQUEST_STATUS.get(item['status'], item['status']), 'Requested': item['created_at'][:10]}
        for item in filtered], key=prefix + ':requests:' + vendor + ':' + contract,
        row_ids=[item['request_id'] for item in filtered], preferred_id=preferred,
        page_label='Evidence requests', table_title='Evidence work queue',
        table_description='Each row is an unresolved evidence request within the selected vendor '
        'and contract filters. Select a request to inspect what is missing, attach relevant support '
        'and record a reviewed resolution. Closing a request does not reassess or approve the invoice.')
    if not filtered:
        st.info('No pending evidence requests match these filters.')
    with connect(db_path) as conn:
        report = audit(conn, tenant)
    request_lines = {request['line_id'] for request in requests}
    gaps = [case for case in report['cases'] if case['status'] == 'NEEDS_REVIEW'
            and case['line_id'] not in request_lines and in_scope(by_line[case['line_id']])]
    if gaps:
        st.warning(f'{len(gaps)} other case(s) still need evidence or scope review. '
                   'They have no open structured request; continue on Invoice review.')
        gap_line = target if target in {case['line_id'] for case in gaps} else gaps[0]['line_id']
        st.button('Review cases needing evidence', on_click=_navigate,
            args=(st, 'Invoice review', db_path), kwargs={'line_id': gap_line}, key=prefix + ':open-gaps')
    if selected is None:
        if filtered:
            st.info('Select a pending request row to review its supporting evidence.')
        return
    request = filtered[selected]
    line = by_line[request['line_id']]
    st.subheader('Review evidence for ' + request['line_id'])
    st.caption(f"{names.get(line['supplier_id'], line['supplier_id'])} · {line['contract_id']} · "
               f"{REQUEST_STATUS.get(request['status'], request['status'])}")
    st.text(request['details'])
    contract_request = request['request_type'] == 'CONTRACT_RATE'
    actions = ['Attach existing document', 'Add new document', 'Review and resolve', 'Reassess case']
    if not contract_request:
        st.info('This evidence type is tracked, but its payment, credit, service or other correction is not supported in this demo. It remains open.')
        actions = ['Reassess case']
    action = st.selectbox('Next action', actions,
        index=2 if contract_request and request.get('attachments') else 0,
        key=prefix + ':action:' + request['request_id'] + ':' + request['status'],
        help='Show one action for this request. Attachment, term review, resolution and '
             'invoice approval remain separate decisions.')
    if action in ('Attach existing document', 'Review and resolve'):
        contract_documents = {doc['doc_id'] for doc in lifecycle.list_documents(db_path, tenant)
                              if doc['kind'] == 'RATE_AGREEMENT'}
        sources = [source for source in lifecycle.list_sources(db_path, tenant)
            if source['doc_id'] in contract_documents and all(source['scope'][key] == line[key]
                for key in ('tenant_id', 'supplier_id', 'contract_id', 'role', 'location'))]
    if action == 'Attach existing document':
        st.caption('Link a document already saved for this contract. Its terms still require review.')
        with st.form(prefix + ':attach:' + request['request_id']):
            version = st.selectbox('Supporting contract document', [''] + [source['version_id'] for source in sources],
                format_func=lambda value: next((source['doc_id'] for source in sources if source['version_id'] == value), 'Select a document'))
            reviewer, note = _reviewer(st, prefix + ':attachment:' + request['request_id'])
            attach = st.form_submit_button('Attach supporting document')
        if attach:
            if not version:
                st.error('Choose a supporting contract document before attaching evidence.')
            elif version not in {source['version_id'] for source in sources}:
                st.error('That document is no longer available for this request. Select it again.')
            else:
                _action(st, db_path, 'Evidence attached; a reviewed resolution is still required.', lambda: lifecycle.attach_evidence(
                    db_path, tenant, request['request_id'], version, reviewer, note,
                    _key('attach', request['request_id'], version, reviewer, note)))
    elif action == 'Review and resolve':
        attached_versions = {event['version_id'] for event in request.get('attachments', [])}
        attached_sources = [source for source in sources if source['version_id'] in attached_versions]
        if attached_sources:
            version = st.selectbox('Attached contract document',
                [source['version_id'] for source in attached_sources],
                format_func=lambda value: next(source['doc_id'] for source in attached_sources
                                                if source['version_id'] == value),
                key=prefix + ':attached:' + request['request_id'])
            attached_source = next(source for source in attached_sources
                                   if source['version_id'] == version)
            document_id = attached_source['doc_id']
            with st.expander('Read attached document'):
                _source(st, attached_source)
            st.button('Review contract terms', on_click=_navigate,
                args=(st, 'Contract term review', db_path), kwargs={'doc_id': document_id},
                key=prefix + ':terms:' + request['request_id'])
        else:
            st.info('Attach supporting evidence before recording a resolution.')
        with st.form(prefix + ':resolve:' + request['request_id']):
            st.caption('Review all attached evidence and its terms before resolving. '
                       'Resolution does not reassess or approve the invoice.')
            reviewer, note = _reviewer(st, prefix + ':resolution:' + request['request_id'])
            resolve = st.form_submit_button('Resolve evidence request', disabled=not request.get('attachments'))
        if resolve:
            _action(st, db_path, 'Evidence request resolved. Reassess this case on Invoice review.',
                lambda: lifecycle.resolve_request(db_path, tenant, request['request_id'], reviewer, note,
                    _key('resolve', request['request_id'], reviewer, note)),
                after=lambda _result: _navigate(st, 'Invoice review', db_path, line_id=line['line_id']))
    elif action == 'Add new document':
        _supporting_intake(st, db_path, tenant, line, request)
    elif action == 'Reassess case':
        st.caption('Recalculate this invoice using current evidence. Open requests remain blocking.')
        if st.button('Reassess selected case', key=prefix + ':reassess:' + request['request_id']):
            with connect(db_path) as conn:
                current = evaluate_line(conn, tenant, line['line_id'])
            _action(st, db_path, 'Current evidence assessed; an open request remains blocking.', lambda: lifecycle.reassess(
                db_path, tenant, line['line_id'], prior_review_event_id=request['review_event_id'],
                idempotency_key=_key('request-reassess', request['request_id'], current['case_digest'])))
    st.button('Open invoice review', on_click=_navigate, args=(st, 'Invoice review', db_path),
              kwargs={'line_id': line['line_id']}, key=prefix + ':open-invoice')


def render_page(st, db_path, page, *, tenant='DEMO'):
    from .team_ui import render_team

    renderers = {'Overview': render_overview, 'Contract term review': render_term_review,
                 'Evidence requests': render_requests, TEAM_PAGE: render_team}
    if page not in renderers:
        raise ValueError('Unknown lifecycle page')
    _notice(st, db_path)
    try:
        renderers[page](st, db_path, tenant=tenant)
    except ACTION_ERRORS as exc:
        st.error(f'This section could not be loaded: {exc}')
