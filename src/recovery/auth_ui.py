"""Password sign-in for the loopback-only synthetic workspace.

Authentication state is kept on the Streamlit server, separate from portfolio
data and fictional reviewer labels. No default credential or public sign-up is
provided after the first local owner has been created.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

from . import local_auth
from .branding import PRODUCT_NAME


def _clear_session(st, *, notice=None):
    # Clear selections, chat, consent, drafts and credentials between accounts.
    for key in list(st.session_state):
        del st.session_state[key]
    if notice:
        st.session_state['auth:notice'] = notice


def _password_help():
    return 'Use a unique passphrase of 15–128 characters. Spaces are allowed.'


def _render_entry(st, database):
    from .presentation import render_brand

    render_brand(st)
    st.html('<style>[data-testid="stAppDeployButton"] { display: none; }</style>')
    first_account = not local_auth.has_accounts(database)
    from .cloud_mode import is_cloud

    if first_account and is_cloud():
        st.error('Cloud owner provisioning is required. Ask the workspace operator to complete '
                 'the private setup. Public account registration is unavailable.')
        return
    with st.container(key='auth-shell'):
        intro, entry = st.columns([1.08, 1], gap='large')
        with intro.container(key='auth-intro'):
            st.caption(PRODUCT_NAME + ' · Finance operations')
            st.header('Clear evidence. Confident reviews.')
            st.markdown('One workspace to investigate invoice discrepancies, understand '
                        'contract terms and keep human decisions connected to their evidence.')
            st.markdown('- **Review with context** — invoices, rates and supporting contracts.\n'
                        '- **Stay in control** — approve a draft, reject or request evidence.\n'
                        '- **Track portfolio value** — approved amounts, forecasts and demo receipts.')
            st.caption('Local synthetic pilot · No supplier communications or payment actions.')
        with entry.container(key='auth-card'):
            st.subheader('Set up your workspace' if first_account else 'Welcome back')
            st.caption('Create the first local owner account. The owner can add accountant accounts '
                       'after signing in.' if first_account else
                       'Sign in with your local account to open the audit workspace.')
            notice = st.session_state.pop('auth:notice', None)
            if notice:
                st.info(notice)
            if first_account:
                with st.form('auth:setup', clear_on_submit=True):
                    name = st.text_input('Your name', max_chars=80, key='auth:setup:name')
                    username = st.text_input('Username', max_chars=64, key='auth:setup:username',
                        help='3–64 characters: letters, numbers, periods, underscores or hyphens.')
                    password = st.text_input('Password', type='password', max_chars=128,
                        help=_password_help(), key='auth:setup:password')
                    confirm = st.text_input('Confirm password', type='password', max_chars=128,
                        key='auth:setup:confirm')
                    submitted = st.form_submit_button('Create owner account', type='primary',
                                                      width='stretch')
                if submitted:
                    if password != confirm:
                        st.error('The passwords do not match. Please enter them again.')
                    else:
                        try:
                            local_auth.bootstrap_owner(database, username, name, password)
                        except (ValueError, PermissionError) as exc:
                            st.error(str(exc))
                        else:
                            _clear_session(st, notice='Your owner account is ready. Sign in to continue.')
                            st.rerun()
            else:
                with st.form('auth:login', clear_on_submit=True):
                    username = st.text_input('Username', max_chars=64, key='auth:login:username')
                    password = st.text_input('Password', type='password', max_chars=128,
                                             key='auth:login:password')
                    submitted = st.form_submit_button('Sign in', type='primary', width='stretch')
                if submitted:
                    token = local_auth.authenticate(database, username, password)
                    if token:
                        _clear_session(st)
                        st.session_state['auth:token'] = token
                        st.session_state['auth:database'] = str(database.resolve())
                        st.session_state['recovery_page'] = 'Overview'
                        st.rerun()
                    else:
                        st.error('Sign-in unsuccessful. Check your credentials or try again later.')
                st.caption('Need access or a password reset? Ask the workspace owner.')
            st.divider()
            st.caption(('Accounts are saved in the private Cloud workspace. ' if is_cloud() else
                        'Accounts are stored on this computer. ') + 'All signed-in accounts share the '
                       'synthetic demo workspace. Company SSO and per-vendor permissions are not configured.')


def require_login(st, database):
    """Return a freshly validated account, or stop before business data is read."""
    database = Path(database)
    try:
        local_auth.initialize(database)
        token = st.session_state.get('auth:token')
        same_database = st.session_state.get('auth:database') == str(database.resolve())
        user = local_auth.current_user(database, token) if token and same_database else None
        if user:
            return user
        if token:
            # Revoke a token only in its originating store, never in a new store.
            if same_database:
                local_auth.logout(database, token)
            _clear_session(st, notice='Your session ended. Sign in to continue.')
        _render_entry(st, database)
    except (OSError, sqlite3.Error, ValueError):
        _clear_session(st)
        st.error('Sign-in is unavailable. Ask the local workspace owner to check the account store. '
                 'Portfolio access remains closed.')
    st.stop()


def render_account_menu(st, database, user):
    """Account administration is separate from fictional invoice reviewer labels."""
    database = Path(database)
    token = st.session_state['auth:token']
    account_notice = st.session_state.pop('auth:account-notice', None)
    with st.sidebar.container(key='session-account'):
        st.caption('SIGNED IN')
        st.text(user['display_name'])
        st.caption('@' + user['username'] + ' · ' + ('Workspace owner' if user['role'] == 'owner'
                                                   else 'Accountant'))
        with st.popover('Account settings', icon=':material/manage_accounts:', width='stretch'):
            st.markdown('**Your account**')
            if account_notice:
                st.success(account_notice)
            st.caption('Workspace password access · 30-minute idle timeout · 8-hour maximum session. '
                       'A browser reload may require signing in again.')
            with st.form('auth:change-password', clear_on_submit=True):
                current = st.text_input('Current password', type='password', max_chars=128)
                new = st.text_input('New password', type='password', max_chars=128,
                                    help=_password_help())
                confirm = st.text_input('Confirm new password', type='password', max_chars=128)
                change = st.form_submit_button('Change password')
            if change:
                if new != confirm:
                    st.error('The new passwords do not match.')
                else:
                    try:
                        local_auth.change_password(database, token, current, new)
                    except (ValueError, PermissionError) as exc:
                        st.error(str(exc))
                    else:
                        _clear_session(st, notice='Password changed. Sign in with your new password.')
                        st.rerun()
            if user['role'] == 'owner':
                st.divider()
                st.markdown('**Manage local accounts**')
                st.caption('The owner manages sign-in accounts. Accountants can use all six demo '
                           'pages; these roles do not restrict individual vendors or contracts.')
                with st.form('auth:add-account', clear_on_submit=True):
                    name = st.text_input('Accountant name', max_chars=80)
                    username = st.text_input('New username', max_chars=64)
                    password = st.text_input('Temporary password', type='password', max_chars=128,
                                            help=_password_help())
                    add = st.form_submit_button('Add accountant')
                if add:
                    try:
                        local_auth.create_account(database, token, username, name, password)
                    except (ValueError, PermissionError) as exc:
                        st.error(str(exc))
                    else:
                        st.session_state['auth:account-notice'] = (
                            'Account created. Share access details privately and ask the '
                            'accountant to change their password after signing in.')
                        st.rerun()
                accounts = local_auth.list_accounts(database, token)
                others = [account for account in accounts if account['username'] != user['username']
                          and account.get('active', True)]
                st.caption(f'{len(accounts)} local accounts registered. Disabled accounts cannot sign in.')
                selected = st.selectbox('Account to disable', [account['username'] for account in others],
                    index=None, placeholder='Choose an active accountant')
                if st.button('Disable selected account', disabled=selected is None):
                    try:
                        local_auth.disable_account(database, token, selected)
                    except (ValueError, PermissionError) as exc:
                        st.error(str(exc))
                    else:
                        st.session_state['auth:account-notice'] = (
                            'Account disabled. Its existing sessions have ended.')
                        st.rerun()
                with st.form('auth:reset-password', clear_on_submit=True):
                    target = st.selectbox('Accountant password to reset',
                        [account['username'] for account in others], index=None,
                        placeholder='Choose an active accountant')
                    replacement = st.text_input('Replacement temporary password', type='password',
                        max_chars=128, help=_password_help())
                    reset = st.form_submit_button('Reset accountant password', disabled=not others)
                if reset:
                    if target is None:
                        st.error('Choose an active accountant first.')
                    else:
                        try:
                            local_auth.reset_account_password(database, token, target, replacement)
                        except (ValueError, PermissionError) as exc:
                            st.error(str(exc))
                        else:
                            st.session_state['auth:account-notice'] = (
                                'Password reset. All sessions for this accountant have ended. '
                                'Share the replacement privately and ask them to change it.')
                            st.rerun()
            st.caption('Invoice reviewer labels and the fictional team directory remain separate '
                       'from sign-in accounts. Existing journal entries keep their original labels.')
        if st.button('Sign out', icon=':material/logout:', width='stretch'):
            local_auth.logout(database, token)
            _clear_session(st, notice='You have signed out.')
            st.rerun()
