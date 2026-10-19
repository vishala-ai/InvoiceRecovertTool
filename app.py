"""Optional localhost accountant workspace."""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from recovery.auth_ui import render_account_menu, require_login
from recovery.branding import PRODUCT_NAME
from recovery.invoice_ui import render_invoice_review
from recovery.lifecycle import enabled
from recovery.lifecycle_ui import render_page
from recovery.storage import connect, metadata
from recovery.presentation import render_brand, render_theme
from recovery.workspace_ui import CHAT_PAGE, TEAM_PAGE, render_navigation


st.set_page_config(page_title=PRODUCT_NAME, layout='wide')
render_theme(st)
AUTH_DB = Path(os.getenv('RECOVERY_AUTH_DB', 'var/local-auth/accounts.sqlite3'))
user = require_login(st, AUTH_DB)
render_brand(st, signed_in=True)
page = render_navigation(st)
render_account_menu(st, AUTH_DB, user)
DB = Path(os.getenv('RECOVERY_DB', 'var/recovery.sqlite3'))
if not DB.exists():
    st.info('Initialize the supplied synthetic snapshot first: uv run recover init')
    st.stop()
with connect(DB) as conn:
    manifest = json.loads(metadata(conn, 'manifest'))
st.caption(f'Synthetic demonstration · As of {manifest["as_of"]} · '
           'Candidate amounts and approved drafts are not cash recoveries.')
if page == CHAT_PAGE:
    from recovery.chatbot_ui import render_chatbot

    render_chatbot(st, DB, tenant='DEMO')
    st.stop()
if page != 'Invoice review':
    if page != TEAM_PAGE and not enabled(DB):
        st.info('Enable the local evidence lifecycle with a verified backup first: '
                f'uv run recover --db {DB} migrate')
        st.stop()
    render_page(st, DB, page, tenant='DEMO')
    st.stop()
render_invoice_review(st, DB, tenant='DEMO')
