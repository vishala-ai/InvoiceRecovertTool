"""Fictional directory and real local approval history, without identity claims."""
from __future__ import annotations

from .team import ACCOUNTANTS, CUSTOM_REVIEWER, MANAGERS, journal_activity, reviewer_label
from .ui_guidance import read_only_table
from .workspace_ui import TEAM_PAGE, page_description


def accountant_choice(st, *, key, bulk=False):
    preferred = st.session_state.get('preferred_demo_accountant', CUSTOM_REVIEWER)
    ids = [CUSTOM_REVIEWER, *[person['accountant_id'] for person in ACCOUNTANTS]]
    return st.selectbox('Bulk approval accountant' if bulk else 'Approval accountant', ids,
        index=ids.index(preferred) if preferred in ids else 0, key=key + ':accountant:' + preferred,
        format_func=lambda value: 'Custom reviewer label' if value == CUSTOM_REVIEWER else reviewer_label(value))


def reviewer_field(st, *, selected, label, key):
    if selected == CUSTOM_REVIEWER:
        return st.text_input(label, key=key + ':custom')
    value = reviewer_label(selected)
    st.text_input(label, value=value, disabled=True, key=key + ':' + selected)
    return value


def _remember_accountant(st):
    # Keep the preference outside widget-owned state, which Streamlit removes
    # when the directory selectbox is no longer rendered on another page.
    st.session_state['preferred_demo_accountant'] = st.session_state['team_default_accountant_choice']


def render_team(st, db_path, *, tenant='DEMO'):
    from .invoice_ui import page_bounds

    page_description(st, TEAM_PAGE)
    st.info('Fictional local demo roster. Reviewer labels are not logins, authenticated identities or access permissions.')
    st.subheader('Escalation managers')
    st.caption('Demonstration contacts only. Escalation routing and notifications are not implemented.')
    for column, manager in zip(st.columns(2), MANAGERS, strict=True):
        with column.container(border=True):
            st.markdown('**' + manager['role'] + '**')
            st.text(manager['name'] + ' (fictional)')
            st.caption(manager['focus'])
    st.subheader('Accountants for future invoice approvals')
    read_only_table(st, [{'Accountant ID': person['accountant_id'], 'Name': person['name'] + ' (fictional)',
        'Review focus': person['focus']} for person in ACCOUNTANTS],
        title='Accountant directory',
        description='Use this fictional directory to choose a reviewer label for future decisions in this browser session. '
                    'The review focus is a demonstration specialty; existing journal entries keep their original reviewers.',
        overrides={
            'Accountant ID': 'Stable identifier for a fictional accountant in this demo directory.',
            'Name': 'Fictional accountant name. It is a reviewer label, not an authenticated identity.',
            'Review focus': 'Suggested area of review expertise for this fictional accountant; it does not grant permissions.',
        })
    ids = [CUSTOM_REVIEWER, *[person['accountant_id'] for person in ACCOUNTANTS]]
    preferred = st.session_state.get('preferred_demo_accountant', CUSTOM_REVIEWER)
    st.selectbox('Accountant for future approvals', ids, key='team_default_accountant_choice',
        index=ids.index(preferred) if preferred in ids else 0,
        on_change=_remember_accountant, args=(st,),
        format_func=lambda value: 'Custom reviewer label' if value == CUSTOM_REVIEWER else reviewer_label(value))
    st.subheader('Approval activity')
    st.caption('Actual saved invoice-review events, including previous demo reviewer labels. '
               'Repeated reviews can create multiple events; these are not current portfolio case counts.')
    selected = st.selectbox('Activity reviewer', ids, key='team_activity_reviewer',
        format_func=lambda value: 'All recorded reviewers' if value == CUSTOM_REVIEWER else reviewer_label(value))
    reviewer = None if selected == CUSTOM_REVIEWER else reviewer_label(selected)
    summary = journal_activity(db_path, tenant=tenant, reviewer=reviewer)
    for column, (label, count) in zip(st.columns(4), (
        ('Recorded decisions', summary['event_count']),
        ('Approved for draft', summary['counts'].get('APPROVED_FOR_DRAFT', 0)),
        ('Rejected', summary['counts'].get('REJECTED', 0)),
        ('More evidence requested', summary['counts'].get('NEEDS_MORE_EVIDENCE', 0))), strict=True):
        column.metric(label, count,
            help='Saved decision events matching the reviewer filter. A case reviewed more than once can contribute '
                 'multiple events; this is not a count of distinct current cases.')
    if not summary['event_count']:
        st.info('No saved decisions match this reviewer filter. The directory does not create approval activity.')
        return
    offset, limit = page_bounds(st, summary['event_count'],
        key=f'team_activity:{db_path}:{tenant}:{selected}', label='Approval activity')
    events = journal_activity(db_path, tenant=tenant, reviewer=reviewer, offset=offset, limit=limit)['events']
    labels = {reviewer_label(person['accountant_id']) for person in ACCOUNTANTS}
    read_only_table(st, [{'Recorded at': event['recorded_at'], 'Invoice line': event['line_id'],
        'Decision': event['decision'].replace('_', ' ').title(), 'Recorded reviewer': event['reviewer'],
        'Attribution': 'Fictional directory label' if event['reviewer'] in labels else 'Original demo reviewer label',
        'Review note': event['note']} for event in events],
        title='Saved review decisions',
        description='Read the saved decisions on this page of the selected reviewer\'s activity. '
                    'This is an audit journal: later evidence or reassessment can require a new review, '
                    'so an older approval does not establish a case\'s current status.',
        overrides={
            'Recorded at': 'Timestamp saved with this review event. The journal is ordered newest first.',
            'Invoice line': 'Invoice-line identifier for the case reviewed in this event.',
            'Decision': 'Decision saved at this review event. Approval for draft does not mean supplier acceptance or cash recovery.',
            'Recorded reviewer': 'Reviewer label recorded at the time of the decision; older labels are preserved.',
            'Attribution': 'Whether the saved reviewer matches the fictional directory or retains an original demo reviewer label.',
            'Review note': 'Reviewer\'s saved explanation or evidence note for this event, shown as originally recorded.',
        })
