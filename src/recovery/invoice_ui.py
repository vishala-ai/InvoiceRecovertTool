"""Invoice table and accountant review actions for the localhost demo."""
from __future__ import annotations

import hashlib
import json
import os
import string
import uuid
from decimal import Decimal

from .invoice_review import (ERRORS, decide_selected, finish_committed,
                             prepare_review, reassess_case, retry_review, selection_token)
from .lifecycle import enabled
from .presentation import amount, assessment_view
from .ui_cache import display_worklist
from .team_ui import accountant_choice, reviewer_field
from .workspace_ui import page_description
from .search_ui import autocomplete_search
from .ui_guidance import column_help, read_only_table, table_intro


COHORTS = ('Current · 2026', 'Migrated · 2025 (locked)', 'Migrated · 2024 (locked)')


def cohort_filter(st, db, namespace):
    """Historical migration rows never enter the operational invoice worklist."""
    value = st.selectbox('Case cohort', COHORTS, key=f'{namespace}_cohort:{db.resolve()}')
    return None if value == COHORTS[0] else int(value.split(' · ')[1].split()[0])


def page_bounds(st, count, *, key, label):
    """Bound browser tables and use a distinct page identity after filtering."""
    if count <= 50:
        return 0, 50
    size = st.selectbox(f'{label} rows per page', [50, 100], key=key + ':size')
    pages = (count + size - 1) // size
    page_key = key + f':page:{size}'
    if st.session_state.get(page_key, 1) > pages:
        st.session_state[page_key] = 1
    page = st.number_input(f'{label} page', min_value=1, max_value=pages, step=1,
                           key=page_key)
    start = (page - 1) * size
    st.caption(f'{label}: {start + 1}–{min(start + size, count)} of {count}. '
               f'Page {page} of {pages}.')
    return start, size


def _search_matches(query, *values):
    return not query or query.casefold() in ' '.join(str(value or '') for value in values).casefold()


def render_archive(st, db, *, tenant, origin_year, namespace):
    """Browse a separately imported, immutable synthetic migration archive."""
    from .legacy_archive import (count_archived_cases, get_archived_case,
                                 list_archived_cases, list_archived_contracts,
                                 list_archived_suppliers)
    from .lifecycle_ui import select_table_row
    from .vendor_contacts import list_vendor_contacts

    prefix = f'{namespace}_archive:{db.resolve()}:{tenant}:{origin_year}'
    chat_target = st.session_state.get(f'chatbot_active_target:{db.resolve()}:{tenant}:{namespace}')
    if chat_target and chat_target['origin_year'] == origin_year:
        prefix += ':chat:' + chat_target['token']
    else:
        chat_target = None
    contacts = {row['supplier_id']: row['poc_name']
                for row in list_vendor_contacts(db, tenant)}
    st.info('Migrated from old system — locked')
    st.caption('Browse previously approved records by vendor, contract and invoice. '
               'These historical records are locked and do not require another approval. '
               'They do not enter the current review queue or create a new recovery.')
    suppliers = list_archived_suppliers(db, tenant, origin_year=origin_year)
    query = autocomplete_search(st, 'Search vendor name or ID',
        [value for row in suppliers for value in
         (row['supplier_id'], row.get('supplier_name') or row.get('name'))],
        key=prefix + ':vendor_search')
    suppliers = [row for row in suppliers if _search_matches(query, row['supplier_id'],
        row.get('supplier_name') or row.get('name'))]
    if not suppliers:
        st.info('No migrated vendors match this search and year.')
        return
    vendor_rows = [{'Vendor ID': row['supplier_id'],
        'Vendor': row.get('supplier_name') or row.get('name', row['supplier_id']),
        'Vendor POC (fictional)': contacts.get(row['supplier_id'], 'Not recorded')}
        for row in suppliers]
    selected = select_table_row(st, vendor_rows, key=prefix + ':vendors:' + query,
        row_ids=[row['supplier_id'] for row in suppliers], page_label='Migrated vendors',
        table_title='Historical vendors',
        table_description='Vendors with migrated records in the selected year. '
                          'Select one vendor to see its contracts and previously approved invoices.',
        preferred_id=chat_target['supplier_id'] if chat_target else None)
    if selected is None:
        st.info('Select a migrated vendor row to see its contracts and invoices.')
        return
    supplier = suppliers[selected]['supplier_id']
    contracts = list_archived_contracts(db, tenant, supplier, origin_year=origin_year)
    contract_query = autocomplete_search(st, 'Search contract',
        [value for row in contracts for value in
         (row['contract_id'], row.get('contract_title') or row.get('title'))],
        key=prefix + ':contract_search:' + supplier)
    contracts = [row for row in contracts if _search_matches(contract_query, row['contract_id'],
        row.get('contract_title') or row.get('title'))]
    if not contracts:
        st.info('No migrated contracts match this search.')
        return
    contract_rows = [{'Contract': row['contract_id'],
        'Title': row.get('contract_title') or row.get('title', row['contract_id']),
        'Status': 'Migrated from old system — locked'} for row in contracts]
    contract_index = select_table_row(st, contract_rows,
        key=prefix + ':contracts:' + supplier + ':' + contract_query,
        row_ids=[row['contract_id'] for row in contracts], page_label='Migrated contracts',
        table_title='Historical contracts',
        table_description='Contracts associated with the selected vendor and year. '
                          'Select a contract to narrow the invoice list; these agreements are read-only.',
        preferred_id=chat_target.get('contract_id') if chat_target else None)
    contract = contracts[contract_index]['contract_id'] if contract_index is not None else None
    st.caption('Invoices remain scoped to this vendor until a contract row is selected.')
    invoice_search_key = prefix + ':invoice_search:' + supplier
    if (chat_target and chat_target.get('line_id')
            and invoice_search_key not in st.session_state):
        st.session_state[invoice_search_key] = chat_target['line_id']
    invoice_suggestions = list_archived_cases(db, tenant, origin_year=origin_year,
        supplier_id=supplier, contract_id=contract)
    invoice_query = autocomplete_search(st, 'Search invoice',
        [value for row in invoice_suggestions for value in
         (row['line_id'], row['invoice_id'], row['contract_id'])],
        key=invoice_search_key)
    filters = dict(origin_year=origin_year, supplier_id=supplier,
                   contract_id=contract, search=invoice_query)
    count = count_archived_cases(db, tenant, **filters)
    if not count:
        st.info('No migrated invoices match these filters.')
        return
    context = hashlib.sha256(json.dumps([supplier, contract, invoice_query]).encode()).hexdigest()[:12]
    offset, limit = page_bounds(st, count, key=prefix + ':cases:' + context,
                                label='Migrated invoices')
    cases = list_archived_cases(db, tenant, **filters, limit=limit, offset=offset)
    rows = [{'Case ID': case['line_id'], 'Invoice': case['invoice_id'],
        'Contract': case['contract_id'], 'Year': case['origin_year'],
        'Historical approved amount': amount(case['historical_approved_amount'], case['currency']),
        'Status': 'Migrated approved — locked', 'Description': case['description']} for case in cases]
    chosen = select_table_row(st, rows, key=prefix + ':invoices:' + context + ':' + str(offset),
        row_ids=[case['line_id'] for case in cases],
        table_title='Historical invoices',
        table_description=f'{count:,} previously approved invoice records match the selected '
                          'vendor, contract and year filters. Open a row to inspect its saved '
                          'details. These records cannot be edited or approved again.',
        preferred_id=chat_target.get('line_id') if chat_target else None)
    if chosen is None:
        st.info('Select an archived invoice row to inspect its saved historical record.')
        return
    case = get_archived_case(db, tenant, cases[chosen]['line_id'])
    if case is None:
        st.warning('The selected migration record is unavailable. Select its current archive row again.')
        return
    st.subheader(case['line_id'] + ' · Migrated from old system — locked')
    historical_details = [{'Historical field': label, 'Saved value': value} for label, value in (
        ('Invoice', case['invoice_id']), ('Vendor', case['supplier_name']),
        ('Contract', case['contract_id']), ('Invoice date', case['invoice_date']),
        ('Historical approved amount', amount(case['historical_approved_amount'], case['currency'])),
        ('Migration source', case['source_system']), ('Source record', case['source_record_id']))]
    read_only_table(st, historical_details, title='Saved historical invoice details',
        description='The invoice identity, approved value and origin recorded for this locked '
                    'historical case. This is a saved reference, rather than a new rate calculation.',
        overrides={
            'Historical field': 'The invoice attribute preserved in this historical record.',
            'Saved value': 'The recorded value for the named attribute. The historical approved '
                           'amount is not a new payment or a new recovery.',
        })
    st.text(case['description'])
    st.caption('This invented migration record cannot be edited, reassessed or approved in this workflow.')


def _prose(st, value):
    st.markdown(''.join('\\' + char if char in string.punctuation else char for char in str(value)))


def _table_edits(st, key, rows, selection_key, bulk_key, processed_key, last_open_key, open_token_key):
    edits = st.session_state.get(key, {}).get('edited_rows', {})
    signature = json.dumps(edits, sort_keys=True)
    if st.session_state.get(processed_key) == [key, signature]:
        return
    st.session_state[processed_key] = [key, signature]
    selected = dict(st.session_state.get(bulk_key, {}))
    explicitly_opened = False
    selection_changed = False
    for index, values in edits.items():
        if not str(index).isdigit() or int(index) >= len(rows):
            continue
        line_id = rows[int(index)]['line_id']
        if values.get('Open') is True:
            st.session_state[selection_key] = line_id
            st.session_state[last_open_key] = line_id
            st.session_state[open_token_key] = dict(rows[int(index)])
            explicitly_opened = True
        elif values.get('Open') is False and st.session_state.get(selection_key) == line_id:
            st.session_state.pop(selection_key, None)
            st.session_state.pop(open_token_key, None)
            selected.pop(line_id, None)
            selection_changed = True
        if 'Select' in values:
            selection_changed = True
            if values['Select'] is True:
                selected[line_id] = dict(rows[int(index)])
            else:
                selected.pop(line_id, None)
    if explicitly_opened:
        selected = {}
    elif selection_changed:
        st.session_state.pop(selection_key, None)
        st.session_state.pop(open_token_key, None)
    st.session_state[bulk_key] = selected


def _show_case(st, case):
    view = assessment_view(case)
    st.subheader(f"{case['line_id']} · {view['status']}")
    _prose(st, view['finding'])
    _prose(st, view['reason'])
    if case['status'] == 'NEEDS_REVIEW':
        view['calculations'][-1]['Amount'] = 'Not established'
    read_only_table(st, view['calculations'], title='Invoice calculation',
        description='Compare the billed amount with the amount supported by reviewed terms, '
                    'then account for recorded pricing credits and refunds. The remaining '
                    'candidate is a potential discrepancy for review, not cash recovered.',
        overrides={
            'Calculation': 'The financial step used to assess this invoice line, from its billed '
                           'amount to the remaining candidate after eligible credits and refunds.',
            'Amount': 'The calculated value in the invoice currency. Not established means '
                      'the available evidence cannot support a reliable amount; it is not zero.',
        })
    if view['services']:
        read_only_table(st, view['services'], title='Reviewed rates and service dates',
            description='Each service entry links approved hours to the reviewed rate applicable '
                        'on that service date. Use the term and document references to check '
                        'the supporting wording below; invoice or purchase-order dates do not '
                        'determine the contractual rate.',
            overrides={
                'Service entry': 'The approved service record used for this dated calculation.',
                'Service date': 'The date the work was performed, which determines which '
                                'reviewed rate and superseding amendment applies.',
                'Approved hours': 'Hours supported by the approved service entry. These quantities '
                                  'are calculation inputs and cannot be edited in this table.',
                'Approved hourly rate': 'The reviewed contract rate for this service date and '
                                        'scope, displayed in the invoice currency.',
                'Reviewed term': 'The manually reviewed term authorizing this service-date rate.',
                'Source document': 'The contract or amendment containing the governing quotation.',
            })
    if case['status'] == 'NEEDS_REVIEW':
        st.caption('Not established means evidence is unresolved; it does not mean there is no possible discrepancy.')
    if case.get('po_rate') is not None:
        st.caption('A matching purchase order alone does not establish the correct contract rate.')
    for warning in case.get('warnings', []):
        _prose(st, warning)
    st.markdown('**Supporting evidence**')
    if not case['citations']:
        st.caption('No governing quotation has been established. Check the missing evidence before reviewing an amount.')
    for citation in case['citations']:
        st.text(citation['doc_id'])
        st.text(citation['quote'])


def _evidence_link(st, db, case, tenant):
    if st.button('Review evidence request', key=f'invoice_evidence:{case["line_id"]}'):
        st.session_state[f'lifecycle:evidence:target:{db.resolve()}:{tenant}'] = case['line_id']
        st.session_state['recovery_page'] = 'Evidence requests'
        st.rerun()


def _prepare(st, db, checkpoints, case, tenant, *, checkbox_key=None):
    workflow = case.get('workflow')
    saved = workflow['state'] if workflow else {}
    line_id = case['line_id']
    identity = hashlib.sha256(json.dumps(selection_token(case), sort_keys=True).encode()).hexdigest()[:16]
    panel_key = f'invoice_prepare:{db.resolve()}:{line_id}:{identity}'
    fresh_required = bool(case.get('committed') or (workflow and not workflow['next']))
    fresh_allowed = not fresh_required
    if fresh_required:
        fresh_allowed = st.checkbox(f'Start a fresh review of {line_id}',
            key=f'repeat_review:{workflow["thread_id"]}')
        st.caption('A fresh review retains the previous decision in history.')
    blocked = bool(case.get('blocking_request_count'))
    if st.button('Prepare review', disabled=case['ready'] or not fresh_allowed or blocked,
                 key=f'prepare_review:{line_id}'):
        st.session_state[panel_key] = uuid.uuid4().hex
    if case['ready']:
        st.caption('This review is ready for a decision. Opening another row does not prepare a new review.')
    if not st.session_state.get(panel_key):
        return
    action_id = st.session_state[panel_key]
    unfinished = bool(workflow and workflow['next'] and workflow['next'] != ['approval']
                      and not saved.get('outcome'))
    cloud = os.getenv('RECOVERY_CLOUD') == '1'
    if cloud and unfinished:
        from .workflow import _validate_cloud_state
        try:
            _validate_cloud_state(saved)
        except ValueError:
            st.error('This unfinished review uses settings unavailable in this Cloud workspace. '
                     'Reassess the invoice and prepare a fresh hosted review.')
            return
    ai = st.checkbox('AI explanation', value=bool(saved.get('use_llm')) if unfinished else False,
                     key=f'invoice_ai:{action_id}', disabled=unfinished)
    provider, model = 'ollama', 'llama3.2:latest'
    base_url, context_tokens = 'http://127.0.0.1:11434', 16384
    allow_provider = False
    if ai:
        pinned = saved.get('explanation_config', {}) if unfinished else {}
        if cloud:
            from .groq_provider import BASE_URL, DEFAULT_MODEL
            provider, model = 'groq', pinned.get('model', os.getenv('GROQ_MODEL', DEFAULT_MODEL))
            base_url = BASE_URL
            st.caption(f'Hosted explanation: Groq · {model}. '
                       'The workspace owner configures this model.')
        else:
            providers = ['ollama', 'openai', 'groq']
            provider = st.selectbox('Explanation provider', providers,
                index=providers.index(pinned.get('provider', 'ollama')),
                key=f'invoice_provider:{action_id}', disabled=unfinished,
                format_func=lambda value: {'ollama': 'Local Ollama', 'openai': 'OpenAI (cloud)',
                                          'groq': 'Groq (hosted Llama)'}[value])
            from .groq_provider import DEFAULT_MODEL
            default_model = {'ollama': 'llama3.2:latest', 'openai': '', 'groq': DEFAULT_MODEL}
            model = st.text_input('Explanation model',
                value=pinned.get('model', default_model[provider]),
                key=f'invoice_model:{action_id}:{provider}', disabled=unfinished)
        base_url = pinned.get('base_url', base_url)
        context_tokens = pinned.get('context_tokens', context_tokens)
        consent = ('Allow sending this synthetic case and its complete scoped evidence to '
                   + {'ollama': 'local Ollama', 'openai': 'OpenAI',
                      'groq': 'Groq (hosted Llama)'}[provider])
        allow_provider = st.checkbox(consent, key=f'invoice_provider_consent:{action_id}:{provider}')
    else:
        st.caption('The standard review uses the calculated facts and quotations without a model call.')
    pinned_retrieval = saved.get('retrieval_config') if unfinished else None
    semantic = False
    if not cloud:
        semantic = st.checkbox('Use local evidence search', value=bool(pinned_retrieval),
                               key=f'invoice_semantic:{action_id}', disabled=unfinished)
    retrieval_index = None
    allow_embeddings = False
    if semantic:
        retrieval_index = (pinned_retrieval['index_path'] if pinned_retrieval
                           else os.getenv('RECOVERY_RETRIEVAL_INDEX', 'var/vector-index'))
        allow_embeddings = st.checkbox('Allow local embedding calls for this review',
                                       key=f'invoice_embedding_consent:{action_id}')
        st.caption('Local search ranks the evidence. All known relevant signed sources are retained.')
    ready = (not ai or allow_provider) and (not semantic or allow_embeddings)
    label = 'Retry preparation' if unfinished else 'Prepare review workflow'
    if st.button(label, disabled=not ready or blocked or not fresh_allowed,
                 key=f'invoice_prepare_submit:{action_id}'):
        try:
            with st.spinner('Preparing the review…'):
                if unfinished:
                    retry_review(db, checkpoints, case, tenant=tenant,
                        allow_provider=allow_provider, allow_embeddings=allow_embeddings)
                else:
                    prepare_review(db, checkpoints, case, action_id=action_id, tenant=tenant,
                        use_llm=ai, provider=provider, model=model, base_url=base_url,
                        context_tokens=context_tokens, retrieval_index=retrieval_index,
                        allow_provider=allow_provider, allow_embeddings=allow_embeddings)
            st.session_state.pop(panel_key, None)
            if checkbox_key is not None:
                st.session_state[checkbox_key] = {}
            st.session_state.pop(f'invoice_open_token:{db.resolve()}', None)
            st.rerun()
        except ERRORS as exc:
            st.error(f'Review not prepared: {exc}')


def _decisions(st, db, checkpoints, case, tenant, notice_key, displayed_token):
    workflow = case.get('workflow')
    if case.get('committed'):
        st.success('Saved decision: ' + case['committed']['decision'].replace('_', ' ').title())
        if case['completion_pending']:
            st.warning('The decision is saved. Finish the interrupted workflow using that same decision.')
            if st.button('Finish workflow completion'):
                try:
                    finish_committed(db, checkpoints, case, tenant=tenant)
                    st.rerun()
                except ERRORS as exc:
                    st.error(f'The decision remains saved; workflow completion failed: {exc}')
        return
    if case['decision_error']:
        st.info(case['decision_error'])
    unchanged = displayed_token == selection_token(case)
    if not unchanged:
        st.warning('This invoice or review changed after you opened it. Open its current row again before deciding.')
    reviewer_key = f'invoice_decision:{workflow["thread_id"] if workflow else case["line_id"]}'
    accountant = accountant_choice(st, key=reviewer_key)
    with st.form(reviewer_key):
        reviewer = reviewer_field(st, selected=accountant, label='Reviewer label', key=reviewer_key)
        note = st.text_area('Evidence review note')
        evidence_type = st.selectbox('Missing evidence type (for Needs review)',
            ['CONTRACT_RATE', 'PAYMENT', 'CREDIT', 'SERVICE', 'OTHER'])
        columns = st.columns(3)
        approve = columns[0].form_submit_button('Approve for draft', disabled=not case['ready'] or not unchanged)
        reject = columns[1].form_submit_button('Reject', disabled=not case['ready'] or not unchanged)
        more = columns[2].form_submit_button('Needs review', disabled=not case['ready'] or not unchanged)
    decision = 'approve' if approve else 'reject' if reject else 'request_more' if more else None
    if decision:
        try:
            results = decide_selected(db, checkpoints, [displayed_token],
                supplier_id=case['supplier_id'], contract_id=case['contract_id'], reviewer=reviewer,
                note=note, decision=decision, evidence_type=evidence_type, tenant=tenant)
            if results[0]['outcome'] == 'Not saved':
                st.error(results[0]['reason'])
            else:
                st.session_state[notice_key] = case['line_id'] + ': decision saved locally.'
                st.rerun()
        except ERRORS as exc:
            st.error(str(exc))


def render_invoice_review(st, db, *, tenant='DEMO'):
    from .chatbot_navigation import consume_chat_target

    page_description(st, 'Invoice review')
    consume_chat_target(st, db, tenant=tenant, kind='case', namespace='invoice')
    checkpoints = db.with_name(db.stem + '-ui-checkpoints.sqlite3')
    prefix = str(db.resolve())
    selection_key = f'invoice_selection:{prefix}'
    notice_key = f'invoice_review_notice:{prefix}'
    year = cohort_filter(st, db, 'invoice')
    cohort_key = f'invoice_last_cohort:{prefix}'
    if st.session_state.get(cohort_key) != year:
        for key in list(st.session_state):
            if key.startswith((f'invoice_bulk:{prefix}:', f'invoice_prepare:{prefix}:')):
                st.session_state.pop(key, None)
        for key in (selection_key, f'invoice_open_token:{prefix}',
                    f'invoice_table_descriptor:{prefix}', f'invoice_filter_context:{prefix}',
                    f'invoice_bulk_results:{prefix}'):
            st.session_state.pop(key, None)
    st.session_state[cohort_key] = year
    if year is not None:
        try:
            render_archive(st, db, tenant=tenant, origin_year=year, namespace='invoice')
        except ERRORS as exc:
            st.error(f'Migrated records could not be loaded: {exc}')
        return
    if notice := st.session_state.pop(notice_key, None):
        st.success(notice)
    try:
        worklist = display_worklist(st, db, checkpoints, tenant=tenant)
    except ERRORS as exc:
        st.error(f'Invoice reviews could not be loaded: {exc}')
        return
    cases = worklist['cases']
    by_id = {case['line_id']: case for case in cases}
    descriptor_key = f'invoice_table_descriptor:{prefix}'
    processed = f'invoice_table_processed:{prefix}'
    last_open_key = f'invoice_last_open:{prefix}'
    open_token_key = f'invoice_open_token:{prefix}'
    # Consume the previously displayed table's event before deriving a new
    # table key from its selection. This also makes restored editor events
    # behave like the ordinary browser on_change callback.
    if previous := st.session_state.get(descriptor_key):
        _table_edits(st, previous['key'], previous['rows'], selection_key,
                     previous['bulk_key'], processed, last_open_key, open_token_key)
    vendors = {case['supplier_id']: case['vendor_name'] for case in cases}
    vendor_key, contract_key = f'invoice_vendor:{prefix}', f'invoice_contract:{prefix}'
    scope_key = f'invoice_review_scope:{prefix}'
    # Overview/evidence links set the hidden ID before the table is rendered.
    linked = st.session_state.get(selection_key)
    last_open = st.session_state.get(f'invoice_last_open:{prefix}')
    if linked in by_id and linked != last_open:
        st.session_state[vendor_key] = by_id[linked]['supplier_id']
        st.session_state[contract_key] = by_id[linked]['contract_id']
        st.session_state[open_token_key] = selection_token(by_id[linked])
        if by_id[linked]['bucket'] not in {'preparation', 'approval', 'evidence', 'reassessment'}:
            st.session_state[scope_key] = 'All cases'
    vendor_query = autocomplete_search(st, 'Search vendor name or ID',
        [value for supplier, name in vendors.items() for value in (supplier, name)],
        key=f'invoice_vendor_search:{prefix}')
    matching_vendors = {supplier for supplier, name in vendors.items()
                        if _search_matches(vendor_query, supplier, name)}
    if st.session_state.get(vendor_key) not in ['', *matching_vendors]:
        st.session_state[vendor_key] = ''
    columns = st.columns(2)
    vendor = columns[0].selectbox('Vendor', ['', *sorted(matching_vendors)], key=vendor_key,
        format_func=lambda value: f'{vendors[value]} ({value})' if value else 'All vendors')
    contracts = sorted({case['contract_id'] for case in cases
                        if case['supplier_id'] in matching_vendors
                        and (not vendor or case['supplier_id'] == vendor)})
    contract_limit = len(contracts) > 100 and not vendor
    if contract_limit:
        contracts = []
    if st.session_state.get(contract_key) not in ['', *contracts]:
        st.session_state[contract_key] = ''
    contract = columns[1].selectbox('Contract', ['', *contracts], key=contract_key,
        format_func=lambda value: value or 'All contracts', disabled=contract_limit)
    if contract_limit:
        st.caption('Choose or search for a vendor to narrow the contract list.')
    scope = st.radio('Review scope', ['Needs review', 'All cases'], horizontal=True, key=scope_key)
    suggested_cases = [case for case in cases if case['supplier_id'] in matching_vendors
        and (not vendor or case['supplier_id'] == vendor)
        and (not contract or case['contract_id'] == contract)
        and (scope == 'All cases' or case['bucket'] in {'preparation', 'approval', 'evidence', 'reassessment'}
             or case['completion_pending'])]
    invoice_query = autocomplete_search(st, 'Search invoice or contract',
        [value for case in suggested_cases for value in
         (case['line_id'], case['invoice_id'], case['contract_id'])],
        key=f'invoice_case_search:{prefix}')
    filtered = [case for case in cases if case['supplier_id'] in matching_vendors
        and (not vendor or case['supplier_id'] == vendor)
        and (not contract or case['contract_id'] == contract)
        and _search_matches(invoice_query, case['line_id'], case['invoice_id'],
                            case['contract_id'], case['description'])
        and (scope == 'All cases' or case['bucket'] in {'preparation', 'approval', 'evidence', 'reassessment'}
             or case['completion_pending'])]
    filtered.sort(key=lambda case: (Decimal(case['known_candidate'] or '0'), case['line_id']), reverse=True)
    search_context = hashlib.sha256(json.dumps(
        [vendor, contract, scope, vendor_query, invoice_query]).encode()).hexdigest()[:12]
    offset, limit = page_bounds(st, len(filtered), key=f'invoice_page:{prefix}:{search_context}',
                                label='Invoice')
    visible = filtered[offset:offset + limit]
    ids = [case['line_id'] for case in visible]
    context = hashlib.sha256(json.dumps(
        [vendor, contract, scope, vendor_query, invoice_query, offset, limit]).encode()).hexdigest()[:12]
    bulk_key = f'invoice_bulk:{prefix}:{context}'
    last_context_key = f'invoice_filter_context:{prefix}'
    last_context = st.session_state.get(last_context_key)
    if last_context != context:
        if last_context:
            st.session_state.pop(f'invoice_bulk:{prefix}:{last_context}', None)
        st.session_state.pop(bulk_key, None)
        st.session_state.pop(f'invoice_bulk_results:{prefix}', None)
        st.session_state[last_context_key] = context
    selected = set(st.session_state.get(bulk_key, {})) & set(ids)
    st.caption('Open an invoice or select one checkbox for an individual review. Select two or more '
               'eligible invoices in the same contract for bulk approval. '
               'Approved and rejected invoices remain in All cases. '
               'Select-all applies only to eligible invoices on this displayed page.')
    if st.session_state.get(selection_key) not in ids:
        st.session_state.pop(selection_key, None)
        st.session_state.pop(open_token_key, None)
    if not visible:
        st.info('No invoices are awaiting review.' if scope == 'Needs review'
                else 'No invoices match this vendor and contract.')
        return
    eligible = [case for case in visible if case['ready']]
    left, right = st.columns(2)
    if left.button('Select all eligible in this contract', disabled=not vendor or not contract or not eligible):
        st.session_state[bulk_key] = {case['line_id']: selection_token(case) for case in eligible}
        selected = set(st.session_state[bulk_key])
    if right.button('Clear selection', disabled=not selected):
        st.session_state[bulk_key] = {}
        selected = set()
        st.session_state.pop(selection_key, None)
        st.session_state.pop(open_token_key, None)
    if len(selected) == 1:
        # Preserve the displayed token when moving from bulk back to one case.
        # A fresh workflow arriving during a rerun must still require re-review.
        token = st.session_state[bulk_key][next(iter(selected))]
        st.session_state[selection_key] = token['line_id']
        st.session_state[last_open_key] = token['line_id']
        st.session_state[open_token_key] = dict(token)
    elif selected:
        st.session_state.pop(selection_key, None)
        st.session_state.pop(open_token_key, None)
    opened = st.session_state.get(selection_key)
    rows = [{'Open': case['line_id'] == opened, 'Select': case['line_id'] in selected,
        'Case ID': case['line_id'], 'Vendor': case['vendor_name'], 'Contract': case['contract_id'],
        'Candidate amount': amount(case['known_candidate'], case['currency']),
        'Review status': case['bucket_label'], 'Description': case['description']}
        for case in visible]
    fingerprint = hashlib.sha256(json.dumps([context, ids, opened, sorted(selected)]).encode()).hexdigest()[:12]
    key = f'invoice_table:{prefix}:{fingerprint}'
    st.session_state[descriptor_key] = {'key': key,
        'rows': [selection_token(case) for case in visible], 'bulk_key': bulk_key}
    table_intro(st, 'Invoice cases',
        'Current invoice lines within your vendor, contract and review filters, ordered by '
        'known candidate value. Open a case to inspect its calculation and evidence. Select '
        'one invoice for individual decisions or two or more eligible cases from one vendor '
        'and contract for bulk approval; amounts shown here '
        'are candidates, not recovered cash.', row_count=len(rows), total_count=len(filtered))
    column_overrides = {
        'Open': 'Open this invoice line below to review its calculation, supporting evidence '
                'and individual decisions. Opening a case clears the checkbox selection '
                'and does not approve it.',
        'Select': 'Select one visible invoice for individual review, or two or more for a '
                  'bulk decision. Approval still requires '
                  'a current validated review and no unresolved evidence blockers.',
        'Candidate amount': 'The supported potential discrepancy after eligible credits and '
                            'refunds, in the invoice currency. Not established means evidence '
                            'is unresolved; this is not an expected cash receipt.',
        'Review status': 'The current review stage for this invoice. Its financial assessment '
                         'and any saved approval remain separate.',
    }
    column_config = column_help(st, rows[0], overrides=column_overrides)
    for name in ('Open', 'Select'):
        column_config[name] = st.column_config.CheckboxColumn(name, help=column_overrides[name])
    st.data_editor(rows, key=key, hide_index=True, use_container_width=True,
        column_config=column_config,
        disabled=[name for name in rows[0] if name not in {'Open', 'Select'}],
        on_change=_table_edits, args=(st, key, [selection_token(case) for case in visible],
            selection_key, bulk_key, processed, last_open_key, open_token_key))
    selected = set(st.session_state.get(bulk_key, {})) & set(ids)
    chosen = [case for case in visible if case['line_id'] in selected]
    if len(chosen) > 1:
        st.markdown(f'**{len(chosen)} selected invoice(s)**')
        pinned = st.session_state[bulk_key]
        totals = {}
        unknown = 0
        for line in sorted(selected):
            token = pinned[line]
            if token['known_candidate'] is None:
                unknown += 1
            else:
                totals[token['currency']] = (totals.get(token['currency'], Decimal('0'))
                                              + Decimal(token['known_candidate']))
        if totals:
            st.caption('Selected known candidate total: ' + '; '.join(
                amount(format(total, '.2f'), currency) for currency, total in sorted(totals.items())))
        if unknown:
            st.caption(f'{unknown} selected invoice(s): candidate amount not established.')
        unready = [case for case in chosen if not case['ready']]
        changed = any(pinned[case['line_id']] != selection_token(case) for case in chosen)
        if unready:
            st.warning('Some selected invoices need preparation or evidence. Clear them before bulk approval.')
        if changed:
            st.warning('A selected invoice or review changed. Clear it and select its current row again.')
        reviewer_key = f'invoice_bulk_approval:{prefix}:{context}'
        accountant = accountant_choice(st, key=reviewer_key, bulk=True)
        with st.form(reviewer_key):
            reviewer = reviewer_field(st, selected=accountant, label='Bulk reviewer label', key=reviewer_key)
            note = st.text_area('Bulk evidence review note')
            approve = st.form_submit_button('Approve selected for draft',
                disabled=not vendor or not contract or bool(unready) or changed)
        if approve and vendor and contract and not unready and not changed:
            try:
                outcomes = decide_selected(db, checkpoints, [pinned[case['line_id']] for case in chosen],
                    supplier_id=vendor, contract_id=contract, reviewer=reviewer, note=note, tenant=tenant)
                st.session_state[f'invoice_bulk_results:{prefix}'] = outcomes
                st.session_state[bulk_key] = {}
                st.rerun()
            except ERRORS as exc:
                st.error(str(exc))
    if results := st.session_state.get(f'invoice_bulk_results:{prefix}'):
        read_only_table(st, [{'Case ID': row['line_id'], 'Result': row['outcome'],
                             'Details': row.get('reason', '')} for row in results],
            title='Last bulk approval results',
            description='The individual outcome for each case in the most recent batch. '
                        'A batch can save some decisions while blocking others; inspect '
                        'each result before retrying an unresolved case.',
            overrides={
                'Result': 'Whether this individual case decision was saved, was already saved, '
                          'or could not be saved. Other rows can have different outcomes.',
                'Details': 'The reason for this case outcome, including any evidence, freshness '
                           'or preparation issue to address before another attempt.',
            })
    if len(chosen) > 1:
        return
    line_id = st.session_state.get(selection_key)
    if line_id not in ids:
        st.info('Choose Open beside an invoice to review its calculation and evidence.')
        return
    st.session_state[f'invoice_last_open:{prefix}'] = line_id
    case = by_id[line_id]
    if st.session_state.get(open_token_key, {}).get('line_id') != line_id:
        st.session_state[open_token_key] = selection_token(case)
    _show_case(st, case)
    saved = case.get('workflow', {}).get('state', {}) if case.get('workflow') else {}
    if saved.get('case') and saved['case']['case_digest'] != case['case_digest']:
        st.markdown('**Previous review — historical evidence and amount**')
        st.caption('These were the facts used by the saved review. They do not authorize the current invoice.')
        _show_case(st, saved['case'])
    if case.get('bucket') == 'reassessment':
        st.warning('Evidence or review rules changed. Reassess before preparing a current decision.')
    if case['status'] != 'CANDIDATE' or case.get('blocking_request_count'):
        _evidence_link(st, db, case, tenant)
    if enabled(db) and not case['ready']:
        if st.button('Reassess with current evidence', key=f'invoice_reassess:{line_id}'):
            try:
                reassess_case(db, line_id, action_id=uuid.uuid4().hex, tenant=tenant)
                st.session_state[bulk_key] = {}
                st.session_state.pop(open_token_key, None)
                st.session_state[notice_key] = f'{line_id}: reassessed with current evidence.'
                st.rerun()
            except ERRORS as exc:
                st.error(f'Reassessment not saved: {exc}')
    if case['status'] == 'CANDIDATE':
        _prepare(st, db, checkpoints, case, tenant, checkbox_key=bulk_key)
        _decisions(st, db, checkpoints, case, tenant, notice_key, st.session_state[open_token_key])
    else:
        if case.get('committed'):
            _decisions(st, db, checkpoints, case, tenant, notice_key, st.session_state[open_token_key])
        st.info('Resolve the evidence gap before an approval review can be prepared.')
    if workflow := case.get('workflow'):
        brief = workflow['state'].get('brief')
        if brief and case['ready']:
            st.markdown('**Prepared review**')
            _prose(st, brief.get('next_action', 'Check the supporting evidence before recording a decision.'))
    st.caption('Approval permits a draft only. New realized recovery remains zero. '
               'Reviewer labels are local demo labels, not authenticated identities.')
