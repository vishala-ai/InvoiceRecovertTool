"""An explicit human assumption for the demo forecast; no automatic saves."""
from __future__ import annotations

from datetime import date
import hashlib
import json
import sqlite3

from .forecast import NOTICE, get_forecast, save_forecast
from .presentation import amount


def read_forecast(db, checkpoints, summary, *, tenant='DEMO', scope='current'):
    return get_forecast(db, checkpoints, tenant=tenant, summary=summary, scope=scope)


def render_forecast_assumption(st, db, checkpoints, forecast, *, tenant='DEMO', scope='current',
                               recovery_overview=None):
    basis, saved = forecast['basis'], forecast['assumption'] or {}
    prior_id = saved.get('forecast_id', 'not-set')
    prefix = f'forecast:{db}:{tenant}:{scope}:{basis["basis_digest"]}:{prior_id}'
    notice_key = f'forecast_notice:{db}:{tenant}:{scope}'
    if notice := st.session_state.pop(notice_key, None):
        st.success(notice)
    with st.expander('Set a demo cash-recovery assumption'):
        st.caption(NOTICE)
        records = 'approved records across all years' if scope == 'all_approved' else 'current approved cases'
        if forecast['status'] == 'CURRENT':
            has_receipts = recovery_overview and recovery_overview['historical_receipts']['case_count']
            forecast_amount = recovery_overview['outstanding_approved_usd'] if has_receipts else basis['amount_usd']
            scope_label = 'in approved records still awaiting recovery' if has_receipts else (
                f"across {basis['case_count']} {records}")
            st.caption(f"{saved['recovery_percentage']}% × {amount(forecast_amount, 'USD')} "
                f"{scope_label}. "
                f"As of {saved['as_of']}; forecast horizon {saved['horizon_date']}.")
            if has_receipts:
                st.caption('Saved forecasts remain in history. Invented historical receipts are fixed '
                           'at their original forecast amount; changing this assumption affects '
                           'only the remaining estimate and never revalues those receipts.')
            st.text('Assumption reviewer: ' + saved['reviewer'])
            st.text('Assumption rationale: ' + saved['reason'])
        elif forecast['status'] == 'STALE_BASIS':
            st.warning('The approved-case basis changed. The previous assumption is retained in history; '
                       'save a reviewed assumption for the current basis before showing an estimate.')
        value_type = 'approved value' if scope == 'all_approved' else 'approved-for-draft value'
        st.caption(f'The percentage is the assumed share of {value_type} ultimately '
                   'recovered as cash by the horizon, including collection uncertainty. '
                   'The initial 80% is illustrative and editable; saving requires human review.')
        st.caption(f"Basis: {basis['case_count']} {records}, {amount(basis['amount_usd'], 'USD')}. "
                   'Pending and rejected records are excluded.')
        with st.form(prefix + ':form'):
            percentage = st.text_input('Assumed recovery percentage (0–100)',
                value=saved.get('recovery_percentage', '80'), key=prefix + ':percentage')
            reviewer = st.text_input('Forecast reviewer label', key=prefix + ':reviewer')
            reason = st.text_area('Reason for this demo assumption', key=prefix + ':reason')
            as_of = st.text_input('Forecast as-of date (YYYY-MM-DD)',
                value=saved.get('as_of', date.today().isoformat()), key=prefix + ':as_of')
            horizon = st.text_input('Forecast horizon (YYYY-MM-DD)',
                value=saved.get('horizon_date', '2026-12-31'), key=prefix + ':horizon')
            save = st.form_submit_button('Save reviewed forecast assumption', disabled=not basis['case_count'])
        if save:
            payload = dict(expected_basis_digest=basis['basis_digest'], recovery_percentage=percentage,
                reviewer=reviewer, reason=reason, as_of=as_of, horizon_date=horizon)
            intent = {'scope': scope, 'prior_forecast_id': prior_id, **payload}
            identity = hashlib.sha256(json.dumps(intent, sort_keys=True).encode()).hexdigest()
            try:
                save_forecast(db, checkpoints, tenant=tenant, scope=scope,
                    idempotency_key='ui-forecast:' + identity, **payload)
                st.session_state[notice_key] = 'Reviewed demo forecast assumption saved locally.'
                st.rerun()
            except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
                st.error(f'Forecast assumption not saved: {exc}')
