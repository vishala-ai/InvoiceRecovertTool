"""Streamlit Community Cloud entrypoint; local app.py keeps its existing launcher."""
from pathlib import Path
import runpy

import streamlit as st
from streamlit.runtime.scriptrunner_utils.exceptions import RerunException, StopException

from recovery.branding import PRODUCT_NAME
from recovery.cloud_runtime import (CloudRuntimeError, clear_session, execute_workspace,
                                     load_settings)
from recovery.presentation import render_brand, render_theme


st.set_page_config(page_title=PRODUCT_NAME, layout='wide')
problem = st.session_state.get('cloud:request-error')
if problem:
    render_theme(st)
    render_brand(st, cloud=True)
    st.error(problem)
    if st.button('Refresh current saved state', type='primary'):
        clear_session(st.session_state)
        st.rerun()
    st.stop()
try:
    settings = load_settings(st.secrets)
    execute_workspace(settings, st.session_state,
        lambda: runpy.run_path(str(Path(__file__).resolve().parents[1] / 'app.py'), run_name='__main__'),
        control_exceptions=(RerunException, StopException))
except CloudRuntimeError as exc:
    clear_session(st.session_state)
    # A fresh full rerun removes provisional main/sidebar output. It never
    # automatically repeats a submitted decision or provider request.
    st.session_state['cloud:request-error'] = str(exc)
    st.rerun()
