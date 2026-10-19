"""Read-only conversational search of the local synthetic audit workspace.

Messages are presentation state only. No chat action approves an invoice, sends
communications, saves an explanation, or creates a review/checkpoint.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from .ui_cache import portfolio_stamp
from .search_ui import autocomplete_search
from .suggestions import load_suggestions
from .ui_guidance import column_help, table_intro


MAX_TURNS = 6
STATE_KEY = 'audit_workspace_chat'
PROMPTS = (
    ('Explain case L001', ':material/receipt_long:'),
    ('Find vendors and contacts', ':material/contact_page:'),
    ('Show pending cases', ':material/pending_actions:'),
    ('Show portfolio totals', ':material/account_balance_wallet:'),
)
ERRORS = (ValueError, RuntimeError, OSError, ImportError, sqlite3.Error)
TABLE_LABELS = {
    'line_id': 'Case', 'supplier_id': 'Vendor ID', 'vendor_name': 'Vendor',
    'contract_id': 'Contract', 'poc_name': 'Vendor POC', 'assessment': 'Assessment',
    'review_status': 'Review status', 'candidate_amount': 'Candidate amount',
    'currency': 'Currency', 'status': 'Status', 'case_count': 'Cases',
    'contract_count': 'Contracts', 'signed_documents': 'Signed documents',
    'origin_year': 'Year', 'historical_approved_amount': 'Historical approved amount',
    'locked': 'Historical lock', 'description': 'Description',
    'measure': 'Measure', 'amount': 'Amount', 'as_of': 'Snapshot date',
    'request_id': 'Evidence request', 'request_type': 'Evidence needed', 'details': 'Details',
}
TABLE_HELP = {
    'Case': 'Invoice-line identifier. Use it to ask about one current or locked historical case.',
    'Vendor ID': 'Recorded supplier identifier linking this vendor to its contracts and cases.',
    'Vendor': 'Synthetic supplier name recorded in the selected portfolio.',
    'Contract': 'Contract identifier linking the supplier, reviewed terms and related cases.',
    'Vendor POC': 'Fictional vendor point-of-contact name stored for this demo. Not recorded means no contact name is available.',
    'Assessment': 'Python calculation outcome based on the invoice, dated reviewed terms, payment evidence and existing offsets.',
    'Review status': 'Current human-review stage for the case. It is separate from the calculation outcome and cash recovery.',
    'Candidate amount': 'Supported discrepancy after allocated applied pricing credits and received refunds. An unresolved amount is not an established saving.',
    'Currency': 'Currency code for the displayed amount. Values in different currencies must not be added together.',
    'Status': 'Recorded case-review or evidence-request stage for this result. Locked migrated approvals need no new approval.',
    'Cases': 'Number of associated current and archived invoice lines in the selected synthetic vendor directory.',
    'Contracts': 'Number of associated contract identifiers, including archived contracts, for this vendor.',
    'Signed documents': 'Known signed rate documents through the source snapshot date. A count alone does not establish applicable reviewed rates.',
    'Year': 'Origin year of the locked synthetic historical record.',
    'Historical approved amount': 'Approved amount imported in the invented locked historical archive. It is separate from a current candidate and from a cash receipt.',
    'Historical lock': 'Whether this result is a migrated historical record protected from new operational review actions.',
    'Description': 'Business description recorded for the synthetic source or migrated record.',
    'Measure': 'Financial measure shown in this row. Approved value, remaining cash estimate and invented historical receipts are separate measures.',
    'Amount': 'Value for the named measure in the stated currency. Historical demo receipts are excluded from the remaining cash estimate.',
    'Snapshot date': 'As-of date used for this calculation and its available source evidence.',
    'Evidence request': 'Stable identifier for an open evidence request or one with an attachment awaiting explicit disposition.',
    'Evidence needed': 'Recorded category of missing evidence. Resolving it and reassessing the invoice are separate steps.',
    'Details': 'Recorded explanation of the evidence gap or follow-up required for this request.',
}
TABLE_GUIDANCE = {
    'vendors': ('Vendor and contact matches',
                'Find the recorded supplier, its fictional point of contact and the number of linked contracts and cases. '
                'Open the vendor\'s contracts to continue reviewing its records.'),
    'contracts': ('Contract matches',
                  'Identify contracts linked to the selected supplier. Ask about an exact contract to read its signed source wording; '
                  'the applicable rate still depends on service dates and reviewed terms.'),
    'case': ('Selected case summary',
             'Confirm the invoice line, supplier and contract behind the explanation above. '
             'Calculation outcomes, human decisions and cash recovery remain separate; migrated cases are locked.'),
    'cases': ('Matching invoice cases',
              'Review invoice lines matching your question and scope. Current review stages show work to do; '
              'migrated approvals are locked history. Open Invoice Review to inspect or decide a current case.'),
    'evidence': ('Pending evidence requests',
                 'Find open requests and attachments awaiting disposition. Attaching evidence alone does not resolve the request '
                 'or approve its case; continue in Evidence Requests for the next action.'),
    'totals': ('Portfolio financial measures',
               'Read the same all-year USD measures as Overview. Approved value is not cash; the remaining estimate excludes '
               'invented historical receipts, and new current realized recovery remains zero.'),
}


def _empty_state(scope, stamp):
    return {'scope': scope, 'stamp': stamp, 'messages': [], 'context': None}


def _action_valid(action):
    if not isinstance(action, dict) or not isinstance(action.get('label'), str):
        return False
    if action.get('page') == 'Invoice review':
        return action.get('line_id') is None or (
            isinstance(action['line_id'], str) and bool(action['line_id']))
    if action.get('page') == 'Contract term review':
        return isinstance(action.get('supplier_id'), str) and bool(action['supplier_id'])
    return action.get('page') == 'Evidence requests'


def _open_action(st, db, tenant, action):
    """A result can select an existing view, never invoke a workflow mutation."""
    if not _action_valid(action):
        return
    prefix = str(Path(db).resolve())
    if action['page'] == 'Invoice review':
        from .lifecycle_ui import _navigate

        _navigate(st, 'Invoice review', db, line_id=action.get('line_id'))
        if not action.get('line_id'):
            st.session_state.pop(f'invoice_selection:{prefix}', None)
            st.session_state[f'invoice_vendor:{prefix}'] = action.get('supplier_id') or ''
            st.session_state[f'invoice_contract:{prefix}'] = action.get('contract_id') or ''
            st.session_state[f'invoice_cohort:{prefix}'] = 'Current · 2026'
            st.session_state[f'invoice_review_scope:{prefix}'] = 'All cases'
            st.session_state.pop(f'invoice_statuses:{prefix}:All cases', None)
            for name in ('invoice_vendor_search', 'invoice_case_search', 'invoice_last_open',
                         'invoice_table_descriptor'):
                st.session_state.pop(f'{name}:{prefix}', None)
        if action.get('origin_year') in (2024, 2025):
            st.session_state[f'chatbot_open_case:{prefix}:{tenant}'] = dict(action)
            st.session_state[f'invoice_cohort:{prefix}'] = (
                f'Migrated · {action["origin_year"]} (locked)')
    elif action['page'] == 'Contract term review':
        st.session_state[f'chatbot_open_contract:{prefix}:{tenant}'] = {
            key: action.get(key) for key in ('supplier_id', 'contract_id', 'origin_year')}
        st.session_state['recovery_page'] = 'Contract term review'
    else:
        st.session_state['recovery_page'] = 'Evidence requests'


def _table_rows(rows):
    return [{TABLE_LABELS.get(field, field): value for field, value in row.items()
             if field != 'case_digest'} for row in rows]


def _render_response(st, db, tenant, response, *, key):
    # Untrusted source wording is always displayed as text, never unsafe HTML.
    st.text(response['answer'])
    if response.get('notice'):
        st.caption(str(response['notice']))
    if response.get('rows'):
        rows = _table_rows(response['rows'])
        title, description = TABLE_GUIDANCE.get(response.get('intent'), (
            'Matching audit records',
            'Inspect the recorded facts supporting this answer. Narrow your question to a vendor, contract or invoice line '
            'for more detail, and use the relevant review page to make a decision.'))
        table_intro(st, title, description, row_count=len(rows), total_count=response.get('total'))
        columns = list(dict.fromkeys(field for row in rows for field in row))
        st.dataframe(rows, hide_index=True, width='stretch',
                     column_config=column_help(st, columns, overrides=TABLE_HELP))
    if response.get('choices'):
        choices = _table_rows(response['choices'])
        table_intro(st, 'Choose the intended record',
            'Several records match your question. Include one exact vendor, contract or case identifier in your next question '
            'so the answer uses the intended records; no review decision has been made.', row_count=len(choices))
        columns = list(dict.fromkeys(field for row in choices for field in row))
        st.dataframe(choices, hide_index=True, width='stretch',
                     column_config=column_help(st, columns, overrides=TABLE_HELP))
    if response.get('citations'):
        with st.expander('Supporting contract evidence'):
            for citation in response['citations']:
                st.text(str(citation.get('doc_id', 'Source document')))
                st.text(str(citation.get('quote', '')))
                if citation.get('source_sha256'):
                    st.caption('Source SHA-256: ' + str(citation['source_sha256']))
    for index, action in enumerate(response.get('actions') or []):
        if _action_valid(action):
            st.button(action['label'], key=f'{key}:open:{index}',
                icon=':material/open_in_new:', on_click=_open_action,
                args=(st, db, tenant, action))


def render_chatbot(st, db, *, tenant='DEMO'):
    """Render explicit user questions; ordinary rerenders make no service calls."""
    from . import chatbot
    from .workspace_ui import CHAT_PAGE, page_description

    db = Path(db).resolve()
    checkpoints = db.with_name(db.stem + '-ui-checkpoints.sqlite3')
    scope = (str(db), tenant)
    scope_key = hashlib.sha256(repr(scope).encode()).hexdigest()[:16]
    prefix = 'audit_chat:' + scope_key
    stamp = portfolio_stamp(db, checkpoints)
    state = st.session_state.get(STATE_KEY)
    stale = bool(state and state.get('scope') == scope and state.get('stamp') != stamp)
    if state is None or state.get('scope') != scope or stale:
        state = _empty_state(scope, stamp)
        st.session_state[STATE_KEY] = state
        st.session_state.pop(prefix + ':question', None)

    page_description(st, CHAT_PAGE)
    st.caption('Search vendors and their contacts, explore contracts and cases, or ask '
               'for a supported explanation. Decisions remain in Invoice Review.')
    if stale:
        st.info('The portfolio has changed. Previous answers were cleared so the next '
                'question uses the current records.')
    if st.button('Clear conversation', key=prefix + ':clear',
                 icon=':material/delete_sweep:'):
        state = _empty_state(scope, stamp)
        st.session_state[STATE_KEY] = state
        st.session_state.pop(prefix + ':question', None)

    prompt = None
    if not state['messages']:
        with st.container(border=True):
            st.markdown('**What would you like to explore?**')
            columns = st.columns(2)
            for index, (label, icon) in enumerate(PROMPTS):
                if columns[index % 2].button(label, key=f'{prefix}:prompt:{index}',
                                             icon=icon, width='stretch'):
                    prompt = label

    for index, message in enumerate(state['messages']):
        with st.chat_message(message['role']):
            if message['role'] == 'user':
                st.text(message['content'])
            else:
                _render_response(st, db, tenant, message['content'],
                                 key=f'{prefix}:message:{index}')

    try:
        catalog = load_suggestions(db, tenant=tenant, cohort='all')
        suggestions = catalog['questions']
        labels = dict(zip(suggestions, catalog.get('question_labels', suggestions)))
    except ERRORS:
        suggestions = [label for label, _ in PROMPTS]
        labels = {}
    response_area = st.container()
    with st.form(prefix + ':compose', clear_on_submit=True):
        entered = autocomplete_search(st, 'Ask about a case, contract or vendor', suggestions,
            key=prefix + ':question', placeholder='Start typing a vendor, contract, case or question',
            format_func=lambda value: labels.get(value, value))
        st.caption('Choose a suggestion or enter your own question, then select Send message.')
        submitted = st.form_submit_button('Send message', icon=':material/send:', type='primary')
    question = (entered if submitted else None) or prompt
    if not question:
        return
    if len(question) > 1000:
        st.warning('Please keep your question to 1,000 characters or fewer.')
        return
    with response_area.chat_message('user'):
        st.text(question)
    with response_area.chat_message('assistant'):
        try:
            with st.spinner('Checking the local audit records…'):
                response = chatbot.answer_question(db, checkpoints, question, tenant=tenant,
                    context=state['context'], use_llm=False)
            if portfolio_stamp(db, checkpoints) != stamp:
                raise RuntimeError('The portfolio changed while answering. Ask again '
                                   'to see current information.')
            if not isinstance(response, dict) or not isinstance(response.get('answer'), str):
                raise ValueError('The answer could not be validated. Please ask again.')
        except ERRORS as exc:
            response = {'answer': 'An answer was withheld. ' + str(exc),
                        'scope': {}, 'intent': 'withheld'}
        _render_response(st, db, tenant, response, key=f'{prefix}:new-message')
    state['messages'].extend([{'role': 'user', 'content': question},
                              {'role': 'assistant', 'content': response}])
    state['messages'] = state['messages'][-MAX_TURNS * 2:]
    state['context'] = response.get('scope') or None
    st.session_state[STATE_KEY] = state
