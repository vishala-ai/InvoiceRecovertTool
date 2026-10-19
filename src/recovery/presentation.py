"""Plain-language views of assessed facts; no calculation or model calls."""
from __future__ import annotations

from decimal import Decimal

from .branding import PRODUCT_MARK, PRODUCT_NAME

STATUS_LABELS = {
    'CANDIDATE': 'Potential rate overpayment',
    'NEEDS_REVIEW': 'More evidence needed',
    'NO_DISCREPANCY': 'No rate discrepancy identified',
    'ALREADY_OFFSET': 'Already covered by credits or refunds',
    'UNPAID': 'No settled payment',
}
REASONS = {
    'POTENTIAL_RATE_OVERPAYMENT':
        'The invoice amount exceeds the amount under the approved service-date rates.',
    'BILLED_AT_OR_BELOW_APPROVED_RATE':
        'The invoice was billed at or below the approved rate.',
    'EXISTING_CREDIT_OR_REFUND':
        'Recorded pricing credits or refunds already cover the rate discrepancy.',
    'NO_SETTLED_PAYMENT': 'No settled payment was found for this invoice line.',
    'CONFLICTING_TERMS':
        'Conflicting reviewed terms leave the applicable service-date rate unresolved.',
    'UNREVIEWED_SIGNED_DOCUMENT':
        'A signed document has not yet been reviewed for its effect on the rate.',
    'MISSING_TERMS': 'No applicable reviewed rate terms were found for the service date.',
    'MISSING_SERVICE_EVIDENCE': 'Approved service records are missing.',
    'PARTIAL_OR_EXCESS_PAYMENT':
        'Recorded settled payments do not match the full invoice amount.',
    'PENDING_ADJUSTMENT': 'A credit or adjustment is still pending and needs review.',
    'INCOMPLETE_SNAPSHOT': 'The supplied contract, payment or adjustment records are incomplete.',
}


def amount(value: str | None, currency: str) -> str:
    """Format existing monetary strings without turning unknown amounts into zero."""
    return 'Not established' if value is None else f'{currency} {Decimal(value):,.2f}'


def assessment_view(case: dict) -> dict:
    """Use the provided snapshot, including when a saved review has become stale."""
    currency, status = case['currency'], case['status']
    candidate = amount(case['net_candidate'], currency)
    if status == 'CANDIDATE':
        finding = (f"Invoice line {case['line_id']} has a potential rate overpayment of "
                   f'{candidate} after recorded pricing credits and refunds. '
                   'This is a candidate for review, not money recovered.')
    elif status == 'NEEDS_REVIEW':
        finding = (f"Invoice line {case['line_id']} needs more evidence before a reliable "
                   'candidate amount can be established. No amount from this line is included '
                   'in the candidate total; this does not establish that there is no overpayment.')
    elif status == 'ALREADY_OFFSET':
        finding = (f"The rate discrepancy for invoice line {case['line_id']} is already covered "
                   'by recorded pricing credits or refunds. No candidate amount remains.')
    elif status == 'UNPAID':
        finding = (f"Invoice line {case['line_id']} has no settled payment. "
                   'This assessment does not count an unpaid invoice as an overpayment.')
    else:
        finding = (f"Invoice line {case['line_id']} was billed at or below the approved rate. "
                   'No rate discrepancy is identified by the current assessment.')
    labels = (
        ('billed_amount', 'Invoice amount'),
        ('expected_amount', 'Amount under approved terms'),
        ('gross_discrepancy', 'Difference before credits and refunds'),
        ('offsets', 'Recorded pricing credits and refunds'),
        ('net_candidate', 'Remaining candidate for review'),
    )
    calculations = [{'Calculation': label, 'Amount': amount(case[key], currency)}
                    for key, label in labels]
    services = [{'Service entry': part['service_entry_id'],
                 'Service date': part['service_date'],
                 'Approved hours': f"{Decimal(part['hours']):,f}",
                 'Approved hourly rate': f"{currency} {Decimal(part['rate']):,f}",
                 'Reviewed term': part['term_id'], 'Source document': part['doc_id']}
                for part in case['components']]
    return {'status': STATUS_LABELS[status], 'finding': finding,
            'reason': REASONS.get(case['reason'], case['reason'].replace('_', ' ').capitalize()),
            'calculations': calculations, 'services': services}


_THEME = '''<style>
  :root {
    --recovery-navy: #122239;
    --recovery-ink: #172b43;
    --recovery-muted: #5d6e82;
    --recovery-teal: #0f766e;
    --recovery-border: #dfe7ef;
  }
  .stApp { background: #f4f7fb; color: var(--recovery-ink); }
  [data-testid="stHeader"] { background: rgba(244,247,251,.96); }
  [data-testid="stMainBlockContainer"] {
    max-width: 1500px; padding: 3.5rem 2.75rem 4rem;
  }
  [data-testid="stMain"] h1, [data-testid="stMain"] h2,
  [data-testid="stMain"] h3 { color: var(--recovery-ink); letter-spacing: -.025em; }
  [data-testid="stMain"] h2 { font-size: 1.9rem; font-weight: 700; }
  [data-testid="stMain"] h3 { font-size: 1.2rem; font-weight: 650; }
  [data-testid="stCaptionContainer"] p { color: var(--recovery-muted); line-height: 1.6; }
  [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stMain"] [data-testid="stMarkdownContainer"] li { line-height: 1.65; }
  [data-testid="stMain"] [data-testid="stMetricLabel"] p { color: #53657b; font-size: .85rem; }
  [data-testid="stMain"] [data-testid="stMetricValue"] {
    color: #182e48; font-weight: 650; letter-spacing: -.035em;
  }
  [data-testid="stMain"] [data-testid="stDataFrame"],
  [data-testid="stMain"] [data-testid="stDataEditor"] {
    border-radius: 12px; overflow: hidden; background: #fff;
    box-shadow: 0 2px 6px #13243b05;
  }
  [data-testid="stMain"] [data-testid="stForm"] {
    background: #fff; border-color: var(--recovery-border); border-radius: 16px;
  }
  [data-testid="stMain"] [data-testid="stTextInputRootElement"],
  [data-testid="stMain"] [data-testid="stTextInput"] [data-baseweb="input"] {
    background: #f7f9fc; border: 1px solid #ccd8e5; border-radius: 8px;
  }
  [data-testid="stMain"] [data-testid="stTextInputRootElement"]:focus-within,
  [data-testid="stMain"] [data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
    border-color: #0f766e; box-shadow: 0 0 0 2px #0f766e1c;
  }
  [data-testid="stMain"] [data-testid="stExpander"] {
    background: #fff; border-color: var(--recovery-border); border-radius: 12px;
  }
  .stButton button, .stFormSubmitButton button {
    border-radius: 9px; min-height: 2.6rem; font-weight: 600;
    transition: background .15s ease, border-color .15s ease;
  }
  [data-testid="stMain"] .stButton button[kind="secondary"],
  [data-testid="stMain"] .stFormSubmitButton button[kind="secondary"] {
    background: #fff; border-color: #cbd7e3; color: #23415e;
  }
  [data-testid="stMain"] .stButton button[kind="secondary"]:hover,
  [data-testid="stMain"] .stFormSubmitButton button[kind="secondary"]:hover {
    background: #edf6f6; border-color: #7bb6b0; color: #0f645e;
  }
  [data-testid="stSidebar"] {
    background: var(--recovery-navy); border-right: 1px solid #20354d;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #e6eef7; }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #a9bdd2; }
  [data-testid="stSidebar"] .stButton button {
    background: #1c324c; color: #e6eef7; border-color: #304960;
  }
  [data-testid="stSidebar"] .stButton button:hover {
    background: #28435e; color: #fff; border-color: #4b718b;
  }
  [data-testid="stSidebar"] [data-testid="stIconMaterial"] { color: inherit; }
  .recovery-brand {
    display: flex; align-items: center; justify-content: space-between;
    gap: 1.5rem; padding: .2rem 0 1.2rem; margin-bottom: .8rem;
    border-bottom: 1px solid #dce5ee;
  }
  .recovery-brand__identity { display: flex; align-items: center; gap: .8rem; min-width: 0; }
  .recovery-brand__mark {
    display: inline-flex; align-items: center; justify-content: center;
    width: 42px; height: 42px; border-radius: 12px; color: #fff;
    font-weight: 750; font-size: 1.15rem; letter-spacing: -.06em;
    background: linear-gradient(135deg, #17547a, #0f766e);
    box-shadow: 0 5px 12px #0f766e18; flex-shrink: 0;
  }
  .recovery-brand__name { color: #19324d; font-size: 1.1rem; font-weight: 750;
    line-height: 1.35; overflow-wrap: break-word; }
  .recovery-brand__context {
    color: #64758a; font-size: .72rem; font-weight: 600;
    letter-spacing: .095em; text-transform: uppercase; margin-top: .1rem;
  }
  .recovery-brand__badge {
    display: inline-flex; align-items: center; gap: .45rem;
    padding: .4rem .7rem; border: 1px solid #cde5df; border-radius: 30px;
    color: #29645b; background: #eaf6f1; font-size: .7rem;
    font-weight: 700; letter-spacing: .055em; white-space: nowrap;
  }
  .recovery-brand__badge::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%; background: #16846e;
  }
  .st-key-page-usage-guide {
    background: #fff; border: 1px solid #dce7ef; border-left: 3px solid #218c8b;
    border-radius: 12px; padding: 1rem 1.3rem; margin-bottom: .8rem;
    box-shadow: 0 2px 8px #152c4803;
  }
  .st-key-page-usage-guide p, .st-key-page-usage-guide li { color: #4c6177; font-size: .9rem; }
  .st-key-page-usage-guide li + li { margin-top: .35rem; }
  .st-key-page-usage-guide strong { color: #203c58; }
  .st-key-auth-shell { max-width: 1080px; margin: 2rem auto; }
  .st-key-auth-intro {
    background: linear-gradient(145deg, #13283f 0%, #16435a 60%, #12625e 100%);
    border: 1px solid #21495b; border-radius: 20px;
    padding: 2.2rem; min-height: 410px; box-shadow: 0 10px 35px #1222390d;
  }
  [data-testid="stMain"] .st-key-auth-intro h2,
  [data-testid="stMain"] .st-key-auth-intro h3,
  [data-testid="stMain"] .st-key-auth-intro p,
  [data-testid="stMain"] .st-key-auth-intro li,
  [data-testid="stMain"] .st-key-auth-intro strong { color: #f0f7ff; }
  [data-testid="stMain"] .st-key-auth-intro [data-testid="stCaptionContainer"] p { color: #c6d9e8; }
  .st-key-auth-card {
    background: #fff; border: 1px solid #dce5ee; border-radius: 20px;
    padding: 2rem; box-shadow: 0 10px 35px #12223908;
  }
  .st-key-auth-card [data-testid="stForm"] { border: 0; padding: 0; }
  .st-key-session-account {
    background: #192f48; border: 1px solid #2a425b; border-radius: 12px;
    padding: .9rem; margin-top: 1.5rem;
  }
  @media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding: 3.5rem 1.25rem 3rem; }
    .recovery-brand { align-items: flex-start; gap: .7rem; flex-wrap: wrap; }
    .recovery-brand__badge { font-size: .61rem; padding: .35rem .5rem; }
    .recovery-brand__name { font-size: 1rem; }
    .recovery-brand__context { font-size: .62rem; }
    .st-key-auth-shell { margin: 1rem auto; }
    .st-key-auth-intro, .st-key-auth-card { padding: 1.5rem; min-height: auto; }
  }
  @media (prefers-reduced-motion: reduce) {
    .stButton button, .stFormSubmitButton button { transition: none; }
  }
</style>'''


def render_theme(st):
    """Apply the workspace's static, responsive visual system."""
    st.html(_THEME)


def render_brand(st, *, signed_in=False, cloud=None):
    """Render the fixed product identity without exposing portfolio records."""
    from .cloud_mode import is_cloud

    context = 'Finance operations · Audit workspace' if signed_in else 'Finance operations · Sign in'
    cloud = is_cloud() if cloud is None else cloud
    badge = 'CLOUD · SYNTHETIC PILOT' if cloud else 'LOCAL · SYNTHETIC PILOT'
    st.html('''<div class="recovery-brand">
      <div class="recovery-brand__identity">
        <span class="recovery-brand__mark" aria-hidden="true">''' + PRODUCT_MARK + '''</span>
        <div><div class="recovery-brand__name">''' + PRODUCT_NAME + '''</div>
        <div class="recovery-brand__context">''' + context + '''</div></div>
      </div>
      <span class="recovery-brand__badge">''' + badge + '''</span>
    </div>''')
