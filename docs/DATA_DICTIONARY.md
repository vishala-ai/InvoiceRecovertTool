# Synthetic data dictionary and accounting assumptions

All supplier names, commercial terms and transactions are invented. `DEMO` contains three suppliers. `OTHER` contains a supplier/document/term canary with an overlapping supplier ID to test scope separation, but no invoice lines. Tenant selection is not authentication.

The service data is primarily Q3 2026; L010 and part of L011 deliberately contain June 30 service records to test the July 1 amendment boundary. The source snapshot is as of October 15, 2026, including October settlement/adjustment records. This is a boundary-test dataset, not a strictly quarter-filtered financial ledger.

## Files

| File | Grain and main fields | Purpose |
|---|---|---|
| manifest.json | dataset_id, as_of, synthetic, completeness flags per tenant | Explicit fixture attestation of contract/payment/adjustment completeness; not a live-system guarantee |
| suppliers.csv | tenant_id + supplier_id; name | Supplier identity within a tenant |
| documents.csv | tenant_id + doc_id; supplier, contract, role, location, kind, status, signed_date, path, sha256 | Source manifest; signed statuses are simulated metadata |
| contracts/*.md | one source document | Human-readable signed/draft/conflicting terms and untrusted note |
| reviewed_terms.csv | tenant_id + term_id; scope; effective dates; rate; supersedes; doc_id; exact quote; reviewer | Manually curated, human-reviewed fixture terms; not AI-generated extraction |
| invoice_lines.csv | tenant_id + line_id; invoice_id + line_number; supplier/contract/PO/role/location; hours; billed rate/amount; currency; invoice date | Transaction to assess |
| service_entries.csv | tenant_id + entry_id; line_id; service_date; approved_hours; approved | Dated approved quantities used to choose applicable rates |
| purchase_orders.csv | tenant_id + po_id; supplier, contract, role, currency, unit, po_rate | Context only; deliberately not the contractual rate authority |
| payments.csv | tenant_id + payment_id; line_id; amount; currency; settled_date; status | Explicit full-cash settlement allocations before subsequent pricing adjustments |
| adjustments.csv | tenant_id + adjustment_id; line_id; amount; currency; kind; status; reason; effective_date | Line-allocated pricing credits or received refunds |
| expected_results.json | test-only per-line statuses/reasons/net amounts | Acceptance oracle; excluded from operational import/calculation |
| retrieval_eval.jsonl | question; line_id; required/forbidden document IDs | Document coverage checks, not generated-answer evaluation |

## Source semantics

The standard-library schemas in `src/recovery/schemas.py` define the exact columns of all eight CSV files. Header order may vary; missing, extra or duplicate columns and malformed CSV records are rejected. Source strings are preserved without coercion. Required identifiers/text cannot be blank; dates must be valid `YYYY-MM-DD`, CSV booleans exactly `true`/`false`, currency codes three uppercase letters, and line numbers positive integers of at most nine digits. Unknown enum values are rejected; known unsupported business scopes such as EUR and OVERTIME still import for engine review.

Decimal strings must be nonnegative and finite, at most 64 characters, with at most 18 coefficient digits, at most six fractional places, stored exponent between -6 and 12, and magnitude below 10^12. Optional blanks are effective end dates, supersession IDs, unreviewed-term reviewer labels, unsigned-document signing dates and unsettled-payment settlement dates. Reviewed terms require a reviewer; signed documents and settled payments require their dates. Effective end dates cannot precede start dates.

The manifest requires `synthetic`, `dataset_id`, `as_of`, `description` and nonempty `tenants`. Every tenant requires the three completeness booleans and `attested_by`. Only literal JSON `true` enables the synthetic guard. Unknown or duplicate JSON fields are rejected. Optional `extracted_at` must include date, seconds and timezone; reports retain `null` when absent. Completeness flags may be false, causing the engine to abstain. Source and document inputs must be regular files without symlinks; documents must use `contracts/*.md` paths within the fixture directory.

`recover init --report var/import-report.json` saves schema-version-1 JSON. `IMPORTED` includes reconciled source/import row counts for every table, exact billed/payment/adjustment totals by tenant/currency/status, full-row hashes and FTS membership/body checks. Totals are Decimal strings and are never rounded or combined across currencies. `REJECTED` contains `errors` with `code`, `file`, `row`, `field` and `message`, and any available failed reconciliation. CSV row numbers refer to physical source lines (header is line 1). Rejection JSON is also emitted on stderr with exit code 2. A report-file write failure is reported separately from the actual database outcome.

Validation occurs before replacement; database insertion and reconciliation run in one transaction. Rejections preserve the previous source snapshot, FTS index, metadata and review journal. Source totals reconcile the loaded fixture to the supplied files, not to an external accounting system. Existing payment/adjustment IDs are retained; identical economic events assigned different IDs still require a future source-specific reconciliation design.

IDs are strings. Decimal fields are strings, not binary floats. Currency is an explicit ISO-style code; this MVP only evaluates USD. Dates use YYYY-MM-DD with inclusive effective start/end dates. A blank effective end date is open-ended.

Scope is exact, not fuzzy: tenant + supplier + contract + role + location. The MVP permits only HOUR units and REGULAR service. Each dated service entry is evaluated against the applicable approved term. The engine sums exact quantity × rate products, then rounds once per invoice line to two decimal places using ROUND_HALF_UP. No tax, foreign exchange, overtime multiplier, expense, minimum commitment, holiday rule, rebate, discount stacking, or interest is modeled.

Term precedence uses explicit `supersedes` chains. A future-effective amendment does not apply backward just because its file is newer. Two equally applicable amendments that do not supersede each other cause NEEDS_REVIEW. A signed document without a reviewed term also blocks the decision. Drafts and untrusted notes are excluded from governing evidence. The starter does not cryptographically verify legal signatures.

Invoice billed_amount must equal billed_hours × billed_rate under the stated rounding rule. Approved service hours must sum to invoice hours. Missing/unapproved records block the decision. A line spanning two rates needs dated service evidence; do not allocate hours evenly by calendar day.

The payment fixture models full cash payment before later pricing adjustments. Partial cash payment, overpayment relative to the exported gross invoice, or a credit used to settle the original invoice is not automatically allocated by this MVP. Such cases need a settlement-model extension and human review. Real ERP exports commonly need a dedicated adapter before they meet these input semantics.

APPLIED_CREDIT and RECEIVED_REFUND are two mutually exclusive ways an offset may be realized for a given underlying economic adjustment. Fixture IDs are unique. Only status APPLIED with reason RATE_PRICING and matching currency reduces the gross discrepancy. Pending, ambiguously classified, or excessive offsets require review. An invoice-level/unallocated credit cannot be guessed onto a line. Future-dated adjustments are excluded from an as-of assessment and flagged.

The same credit must not be imported once as a refund and again as an applied credit. Primary-key checks prevent identical ID duplication; production needs source/business-key reconciliation to catch economic duplicates with different IDs.

## Expected line results

| Line | Scenario | Expected result | Remaining candidate USD |
|---|---|---|---:|
| L001 | 8,000 hours billed at 130 after approved reduction to 120 | CANDIDATE | 80,000.00 |
| L002 | Correct amended rate | NO_DISCREPANCY | 0.00 |
| L003 | 1,000 gross discrepancy fully offset by applied credit | ALREADY_OFFSET | 0.00 |
| L004 | 1,000 gross discrepancy, 400 applied credit | CANDIDATE | 600.00 |
| L005 | Signed August increase from 100 to 110 | NO_DISCREPANCY | 0.00 |
| L006 | Two conflicting September amendments | NEEDS_REVIEW | 0.00 |
| L007 | Contract evidence missing | NEEDS_REVIEW | 0.00 |
| L008 | Invoice unpaid | UNPAID | 0.00 |
| L009 | Partial payment | NEEDS_REVIEW | 0.00 |
| L010 | June services use original 130 rate | NO_DISCREPANCY | 0.00 |
| L011 | Ten hours before and ten after effective-date boundary | CANDIDATE | 100.00 |
| L012 | EUR invoice versus USD scope | NEEDS_REVIEW | 0.00 |
| L013 | Invoice hours and service hours differ | NEEDS_REVIEW | 0.00 |
| L014 | No approved service records | NEEDS_REVIEW | 0.00 |
| L015 | July staffing service precedes August increase | NO_DISCREPANCY | 0.00 |
| L016 | Overtime outside scope | NEEDS_REVIEW | 0.00 |
| L017 | Fractional hours: 0.75 × rate difference of 10 | CANDIDATE | 7.50 |
| L018 | 1,000 gross discrepancy already refunded | ALREADY_OFFSET | 0.00 |
| L019 | Exported amount fails invoice extension check | NEEDS_REVIEW | 0.00 |
| L020 | Credit exists but has not been applied | NEEDS_REVIEW | 0.00 |

Initial candidate total: **80,707.50**. New realized recoveries: **0.00**. A zero displayed for NEEDS_REVIEW means excluded from a supported candidate total, not proof that no overpayment occurred.
