"""Business descriptions and column help for the local review workspace.

Presentation only: existing values are displayed without calculations, provider
calls, decisions or database access. Money remains its original formatted text.
"""
from __future__ import annotations


COLUMN_HELP = {
    'Open': 'Open this invoice to inspect its calculation and evidence. This does not approve it.',
    'Select': 'Include this visible invoice in a proposed bulk decision. Eligibility is checked again before saving.',
    'Case': 'Invoice-line case identifier. Use it to open or ask about this specific assessment.',
    'Case ID': 'Unique invoice-line case identifier. Each line is reviewed separately.',
    'Invoice line': 'The individual invoice line assessed or referenced by this record.',
    'Invoice': 'Supplier invoice identifier. One invoice can contain several separately assessed lines.',
    'Supplier': 'Supplier identifier associated with the invoice line.',
    'Vendor ID': 'Recorded supplier identifier used to link its contracts and invoice cases.',
    'Vendor': 'Recorded supplier name. Use the vendor ID to distinguish similar names.',
    'Vendor POC': 'Fictional vendor point of contact for this synthetic portfolio; no message is sent.',
    'Vendor POC (fictional)': 'Invented vendor point of contact, included for demonstration and lookup only.',
    'Contract': 'Contract identifier linking rate documents, amendments and related invoice lines.',
    'Contracts': 'Distinct contracts in this row or scope. Amendments are grouped under their parent contract.',
    'Cases': 'Invoice-line cases associated with this record. Cases are different from review events.',
    'Contract documents': 'Recorded contract and amendment documents for this vendor, including documents awaiting review.',
    'Document': 'Source document identifier for this contract or amendment. A missing label means evidence is unavailable.',
    'Title': 'Recorded title of the selected contract or source record.',
    'Document status': 'Recorded document state, such as signed or draft. Signed documents still require applicability review.',
    'Signed documents': 'Count of known signed rate documents in this contract scope; ranking does not establish completeness.',
    'Signed date': 'Recorded signature date. Applicable rates are selected using service dates and reviewed terms.',
    'Source document': 'Source identifier supporting the reviewed hourly rate for this service entry.',
    'Role': 'Service role to which the reviewed rate or document applies.',
    'Location': 'Service location used when matching an invoice to applicable reviewed terms.',
    'Service': 'Type of service covered by the reviewed rate, such as regular hourly work.',
    'Rate terms': 'Whether manually reviewed approved terms are available for this document.',
    'Hourly rate': 'Human-reviewed contractual hourly rate in its stated currency. It is not a saving.',
    'Effective from': 'First service date covered by the reviewed rate.',
    'Effective to': 'Last covered service date, where recorded. Explicit supersession rules still apply.',
    'Replaces earlier rate': 'Earlier reviewed term explicitly superseded within the same applicable scope.',
    'Reviewed by': 'Saved demo reviewer label for the reviewed term; this is not an authenticated identity.',
    'Reviewed term': 'Identifier of the manually reviewed rate term selected for this service date and scope.',
    'Work queue': 'Current review queue for this invoice line, based on its latest assessment and decision.',
    'Assessment': 'Financial assessment of the invoice line. A candidate is a potential discrepancy requiring review.',
    'Financial assessment': 'Current calculation finding. Missing evidence withholds a reliable candidate amount.',
    'Review status': 'Current case workflow position, such as preparation, awaiting approval or evidence required.',
    'Current decision': 'Latest applicable human decision. A draft approval does not establish cash received.',
    'Status': 'Recorded state for this table. Read the table description to distinguish document, case and request states.',
    'Candidate': 'Supported net discrepancy after allocated applied pricing credits and received refunds; not cash recovered.',
    'Candidate amount': 'Potential net amount for review. Not established means evidence is unresolved, rather than zero.',
    'Currency': 'Recorded currency. Amounts in different currencies must not be added without an approved conversion policy.',
    'Calculation': 'Calculation component explaining the invoice amount, contractual amount, offsets or remaining candidate.',
    'Amount': 'Existing monetary value for the measure named on this row. Read its currency and measure before comparison.',
    'Service entry': 'Identifier of an approved service record supporting the assessed hours and service date.',
    'Service date': 'Date the service was performed; this selects the applicable reviewed rate and amendment.',
    'Approved hours': 'Approved service hours used by the calculation, preserved with Decimal precision.',
    'Approved hourly rate': 'Reviewed rate selected for this service date and exact contract scope, in the stated currency.',
    'Description': 'Saved business context for the case or record. Synthetic and migration labels describe its provenance.',
    'Year': 'Historical origin year of the locked migrated record.',
    'Historical approved amount': 'Invented previously approved amount carried in locked migration history; not a new current approval.',
    'Historical lock': 'Whether this migrated record is read-only and excluded from the current approval process.',
    'Historical field': 'Field from the saved migrated invoice record. Historical records cannot be edited or reassessed here.',
    'Saved value': 'Previously recorded value shown for inspection. Displaying it does not approve or change it.',
    'Scope field': 'Applicability dimension binding a proposal to its vendor, contract, role, location or currency.',
    'Bound value': 'Saved applicability value. It must match the invoice scope before a term can be approved.',
    'Proposed term': 'Term proposed for human review. A proposal is not an approved calculation input.',
    'Result': 'Outcome for this individual bulk-review attempt. Other invoices in the batch may have different outcomes.',
    'Details': 'Reason or explanation associated with this row. Check any unmet requirements before taking the next step.',
    'Evidence request': 'Identifier of a saved request for missing or incomplete evidence.',
    'Evidence needed': 'Evidence category required to progress the case, such as contract, payment, credit or service records.',
    'Requested': 'Date the evidence request was created; resolution and reassessment are separate steps.',
    'Snapshot date': 'As-of date for the assessed records or reported financial measure.',
    'Measure': 'Name of the reported financial measure. Approved value, estimated cash and realized receipts are distinct.',
    'Accountant ID': 'Stable identifier for a fictional member of the demonstration accountant roster.',
    'Name': 'Fictional team member name for the local demonstration.',
    'Review focus': 'Example responsibility or area of review for this fictional accountant.',
    'Recorded at': 'Saved timestamp of the review event, including its recorded timezone where present.',
    'Decision': 'Human decision saved for this event. Approval permits a draft and does not record received cash.',
    'Recorded reviewer': 'Reviewer label saved with this event. Existing labels are retained when the directory preference changes.',
    'Attribution': 'Whether the saved label matches the fictional directory or an earlier demo reviewer label.',
    'Review note': 'Rationale saved by the reviewer for this event; previous notes remain in history.',
}


def column_help(st, columns, *, overrides=None):
    """Add native header help without changing column values or inferred types."""
    definitions = {**COLUMN_HELP, **(overrides or {})}
    return {name: st.column_config.Column(name, help=definitions.get(name,
        'Recorded field for the matching record. Read this table\'s description for its scope and next step.'))
        for name in columns}


def table_intro(st, title, description, *, row_count=None, total_count=None):
    """Explain a table and report only counts supplied by its existing reader."""
    st.markdown('**' + title + '**')
    st.caption(description)
    if row_count is not None:
        if total_count is not None and total_count != row_count:
            st.caption(f'Showing {row_count:,} of {total_count:,} matching records. '
                       'Hover over a column heading for its definition.')
        else:
            st.caption(f'{row_count:,} row(s) shown. Hover over a column heading for its definition.')


def read_only_table(st, rows, *, title, description, overrides=None):
    """Render a readable matrix with native column help and no edit actions."""
    table_intro(st, title, description, row_count=len(rows))
    columns = list(dict.fromkeys(name for row in rows for name in row))
    return st.dataframe(rows, hide_index=True, width='stretch',
                        column_config=column_help(st, columns, overrides=overrides))
