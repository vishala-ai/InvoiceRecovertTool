"""Business page labels, descriptions and button navigation for the local demo."""

from .branding import PRODUCT_MARK, PRODUCT_NAME

TEAM_PAGE = 'Human Approval Auditing Team'
CHAT_PAGE = 'Ask the Audit Workspace'
NAVIGATION = (
    ('Overview', 'Overview', ':material/dashboard:'),
    ('Invoice review', 'Invoice Review', ':material/receipt_long:'),
    ('Contract term review', 'Contract Term Review', ':material/description:'),
    ('Evidence requests', 'Evidence Requests', ':material/folder_open:'),
    (CHAT_PAGE, CHAT_PAGE, ':material/chat:'),
    (TEAM_PAGE, TEAM_PAGE, ':material/groups:'),
)
DESCRIPTIONS = {
    'Overview': 'See grand totals for approved value, estimated cash recovery and realized '
        'recovery across all years, supplier and contract coverage, and the current review queues. '
        'Completed demo receipts appear only in realized recovery.',
    'Invoice review': 'Find invoices by vendor or contract, inspect the calculation and evidence, '
        'then record a human decision. Approval permits a draft; it does not record money recovered.',
    'Contract term review': 'Inspect contracts, amendments and manually reviewed rate terms, '
        'then open the affected invoices. Term approval is separate from invoice approval.',
    'Evidence requests': 'Work through missing evidence, attach supporting documents and record '
        'a separate reviewed resolution before reassessing an invoice.',
    CHAT_PAGE: 'Ask about a vendor, contact, contract, invoice case or portfolio value. '
        'Answers use saved records and cited evidence; approval decisions stay in Invoice Review.',
    TEAM_PAGE: 'View the fictional demo team, choose an accountant label for future approvals '
        'and inspect the actual local decision journal.',
}

PAGE_USAGE = {
    'Overview': (
        '**Understand value:** compare approved value, the remaining cash estimate and recorded '
        'demo receipts. Approval is a review milestone; it is separate from receiving cash.',
        '**Understand coverage:** supplier and contract counts cover the portfolio across all '
        'years. Amendments remain grouped under their parent contract.',
        '**Choose your next task:** use the workload cards to inspect a queue, then open an '
        'invoice. Awaiting approvals are reviewed on the Invoice Review page.',
    ),
    'Invoice review': (
        '**Find the right invoice:** choose the year, search for a vendor and narrow the '
        'contract. Use suggestions or your own search text to find a specific case.',
        '**Inspect the finding:** choose Open or select one checkbox, compare the calculation with the '
        'reviewed service-date rates and read the supporting evidence. Unresolved amounts '
        'remain Not established.',
        '**Record a decision:** approve for draft, reject or request more evidence after '
        'inspection. Two or more checked invoices show bulk approval instead of individual '
        'decisions, within one vendor and contract on the displayed page. Opening a case '
        'clears the checkbox selection. Migrated history is locked.',
    ),
    'Contract term review': (
        '**Follow the relationship:** select a vendor, then a contract or amendment. The '
        'related cases update to match that selection.',
        '**Check applicability:** inspect the original wording, service role, location and '
        'effective dates alongside the manually reviewed terms. A draft document is not '
        'an approved rate.',
        '**Continue the review:** open a related invoice to assess its discrepancy. '
        'Approval of rate terms and approval of an invoice are separate decisions.',
    ),
    'Evidence requests': (
        '**Prioritize outstanding requests:** filter by vendor or contract and select a '
        'pending request. Resolved requests are retained in history and leave this queue.',
        '**Review the missing evidence:** inspect the request type and supporting documents '
        'before recording the evidence disposition. Attaching a document alone does not '
        'resolve the request.',
        '**Return to the case:** after evidence is resolved, reassess the invoice against '
        'the updated records and complete its separate decision in Invoice Review.',
    ),
    CHAT_PAGE: (
        '**Ask a focused question:** type a vendor, contact, contract or invoice identifier. '
        'Choose a suggested question or write your own, then select Send message.',
        '**Understand the answer:** read the result summary and expand supporting contract '
        'evidence where available. Repeated contact names are distinguished by vendor ID.',
        '**Continue in the right workspace:** use a result link to inspect its record. '
        'Chat is read-only; approval decisions remain in Invoice Review.',
    ),
    TEAM_PAGE: (
        '**Understand responsibilities:** use the fictional escalation directory to see '
        'which review issues belong with audit management or finance.',
        '**Set future ownership:** choose an accountant label for your next invoice decisions. '
        'Changing this preference preserves the reviewers recorded on earlier decisions.',
        '**Inspect recorded activity:** filter the journal by reviewer and read each saved '
        'decision and rationale. Repeated reviews can create several events for one invoice. '
        'The demo roster does not provide sign-in or send notifications.',
    ),
}


def select_page(st, page):
    st.session_state['recovery_page'] = page


def render_navigation(st):
    pages = [page for page, _, _ in NAVIGATION]
    if st.session_state.get('recovery_page') not in pages:
        st.session_state['recovery_page'] = 'Invoice review'
    selected = st.session_state['recovery_page']
    st.html('''<style>
      .st-key-workspace-navigation { padding: 0 2px; }
      .workspace-sidebar-brand { display: flex; align-items: center; gap: .75rem;
        padding: .25rem .4rem 1.35rem; border-bottom: 1px solid #2a4057; margin-bottom: .6rem; }
      .workspace-sidebar-mark { color: #d8fff7; background: #19545c; border: 1px solid #28727a;
        width: 37px; height: 37px; display: flex; align-items: center; justify-content: center;
        border-radius: 10px; font-size: .92rem; font-weight: 750; letter-spacing: -.06em;
        flex-shrink: 0; }
      .workspace-sidebar-brand > div:last-child { min-width: 0; }
      .workspace-sidebar-name { color: #f0f6ff; font-size: .95rem; font-weight: 700;
        line-height: 1.35; overflow-wrap: break-word; }
      .workspace-sidebar-subtitle { color: #a1b9cd; font-size: .71rem; margin-top: .15rem; }
      .workspace-navigation-label { color: #94b0c7; font-size: .65rem; font-weight: 650;
        letter-spacing: .13em; padding: .7rem .5rem .4rem; }
      .st-key-workspace-navigation button { justify-content: flex-start; text-align: left;
        border-radius: 9px; min-height: 46px; white-space: normal; transition: background .15s ease;
        border: 1px solid transparent; font-weight: 550; padding: .65rem .8rem; }
      .st-key-workspace-navigation button[kind="secondary"],
      .st-key-workspace-navigation [data-testid="stBaseButton-secondary"] {
        background: transparent; color: #b9cce0; border-color: transparent; }
      .st-key-workspace-navigation button[kind="secondary"]:hover,
      .st-key-workspace-navigation [data-testid="stBaseButton-secondary"]:hover {
        background: #203a53; color: #fff; border-color: #2c4b65; }
      .st-key-workspace-navigation button[kind="primary"],
      .st-key-workspace-navigation [data-testid="stBaseButton-primary"] {
        background: #176876; color: #fff; border-color: #2b8790;
        box-shadow: inset 3px 0 0 #7ce5d4; }
      .st-key-workspace-navigation button[kind="primary"]:hover,
      .st-key-workspace-navigation [data-testid="stBaseButton-primary"]:hover {
        background: #1a7884; border-color: #4da3a8; }
    </style>''')
    with st.sidebar.container(key='workspace-navigation'):
        st.html('''<div class="workspace-sidebar-brand">
          <div class="workspace-sidebar-mark" aria-hidden="true">''' + PRODUCT_MARK + '''</div>
          <div><div class="workspace-sidebar-name">''' + PRODUCT_NAME + '''</div>
          <div class="workspace-sidebar-subtitle">Finance review workspace</div></div>
        </div><div class="workspace-navigation-label">WORKSPACE</div>''')
        for page, label, icon in NAVIGATION:
            st.button(label, icon=icon, key='navigation:' + page, width='stretch',
                type='primary' if selected == page else 'secondary',
                on_click=select_page, args=(st, page))
    return selected


def page_description(st, page):
    label = next(label for name, label, _ in NAVIGATION if name == page)
    st.header(label)
    st.caption(DESCRIPTIONS[page])
    with st.container(key='page-usage-guide', border=True):
        st.markdown('**How to use this page**')
        st.markdown('\n'.join('- ' + item for item in PAGE_USAGE[page]))
