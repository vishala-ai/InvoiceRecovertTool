# Supplier/ Vendor Invoice Recovery Tool — implementation guide

## 1. Product contract

Build for one user: a finance/procurement reviewer investigating regular hourly IT-services invoices. The first decision is whether a fully paid invoice line contains a reviewable pricing discrepancy after existing offsets. The output is an evidence package and local review state, not a legal conclusion or an automatically issued claim.

Scope the pilot to three to five authorized suppliers. Use read-only exports, regular hourly services, a single currency, and a completed service period. Exclude taxes, FX, overtime, expenses, rebates, litigation/collection claims, automated payments, autonomous supplier negotiation, and ERP writes until separately designed and tested.

The value metric is incremental realized refunds/applied credits attributable to valid findings, less implementation, review and collection costs. For the lab, the result is a synthetic candidate total only. Do not call the fixture's 80,707.50 a realized saving.

## 2. Two different roles for AI tools

**Codex is the development assistant.** It reads project files, proposes and edits code, runs permitted commands, and helps debug. It is not part of the runtime recovery product. Start it in this repository, review its permissions, give it one milestone at a time, and inspect diffs/tests before accepting changes.

**The application's language model is an optional runtime dependency.** Explanation providers are explicit: a loopback-only Ollama adapter for an installed local model, or LangChain's ChatOpenAI integration for an external API model. The separate unreviewed extraction experiment still uses OpenAI. Only the OpenAI paths need API credentials. A ChatGPT/Codex login is not an API key for the Python application; API billing is separate.

**LangGraph is optional orchestration.** It coordinates assessment, retrieval, drafting and a persisted human-review pause. The financial engine remains ordinary Python. No collection of independent agents is needed for this MVP.

**Tavily is deliberately absent.** Private negotiated contract terms and supplied ledger records are the relevant evidence. Public web prices would not establish a discrepancy. Add narrowly scoped public research only when a signed contract explicitly refers to an external published schedule, with historical-version and egress controls.

## 3. Environment and first run

The user has additionally selected a synthetic Streamlit Community Cloud pilot with Groq-hosted Llama and persistent Supabase storage. Its separate entrypoint, private owner provisioning, atomic coupled-state transaction and operator steps are documented in [COMMUNITY_CLOUD.md](COMMUNITY_CLOUD.md). This does not authorize real-data onboarding or weaken the local baseline and review controls.

Recommended local setup: Mac, VS Code, Python 3.12 managed with uv, SQLite, and the Codex IDE extension. The core supports Python 3.11–3.13; the latest local verification used Python 3.12.15 and SQLite 3.53.1 on macOS. A GPU, Docker and managed database are unnecessary for these fixtures.

Follow the README setup commands from the extracted repository root. Route A runs the financial core with the Python standard library. Route B installs optional packages, creates a virtual environment, and starts the UI. Keep `.env` untracked and select a structured-output-compatible model enabled in your own API account.

The original creation environment could not download dependencies. They have since been resolved with `uv sync --all-extras`, and the generated `uv.lock` has been tested locally. Reproduce it with `uv sync --frozen --all-extras`. This extracted workspace has no Git repository; no commit was made.

### First-run gate

The initial audit must contain exactly 20 cases, four candidates, nine needs-review outcomes, four clean lines, two already-offset lines, and one unpaid line. The candidate amount must be exactly USD 80,707.50. New realized recoveries must remain zero. The operational importer must work with `expected_results.json` removed; expected outcomes are never application inputs.

## 4. Implement and verify the data foundation

This portion is implemented for the supplied fixtures in `storage.py`.

Load the manifest, source records and document bodies into SQLite. Validate identifiers, supplier and line references, source paths, document hashes and cited exact quotes. Use primary keys scoped by tenant. Import within a transaction so a failed/duplicate import does not replace the valid prior snapshot. Re-importing the same snapshot must be idempotent. Preserve the separate review journal.

The current loader is a fixture importer, not an arbitrary ERP adapter. Before accepting an actual source format, define an explicit schema/mapping and reject missing/extra/invalid fields with a machine-readable error report. Reconcile source and loaded row counts, invoice totals, payment totals, currencies and extraction timestamps. Track source system IDs and economic/business keys, not only a newly generated local ID.

Financial export semantics matter. A payment allocation is not always the original invoice amount, and a credit may be used to settle an invoice rather than paid back afterward. This starter intentionally handles full cash payment before subsequent pricing offsets. Build an explicit settlement model rather than patching partial-payment cases with guesses.

**Implemented fixture gate:** `schemas.py` defines exact manifest/CSV fields and validates source strings before database replacement. `recover init --report var/import-report.json` writes an `IMPORTED` or `REJECTED` JSON report. Rejections identify file, physical CSV row, field and error code without echoing rejected values. Source/import counts, Decimal invoice/payment/adjustment totals by tenant/currency/status, complete row digests, manifest metadata and the FTS document set are checked inside the import transaction. A mismatch rolls back the replacement and preserves the review journal. Existing source/business IDs are retained; duplicate invoice business keys are rejected.

`extracted_at` is an optional timezone-qualified manifest timestamp. Existing fixtures lack it, so reports retain `null` instead of inventing an extraction time. Format validation does not require balanced invoice extensions, full payments or complete service evidence: those business cases still reach the engine and preserve their existing assessment outcomes. See the data dictionary for schema bounds and `tests/test_import.py` for malformed-input and rollback regressions.

### Expanded synthetic portfolio and locked history

The separate `demo_portfolio.py` builder copies the earlier million-dollar database/checkpoints and preserves original import anchors, source records and review identities. It appends 35 invented vendors and 1,000 schema-validated 2026 invoice lines with exact Decimal reconciliation. Each added approved or pending case receives a real offline LangGraph draft and human-review pause; only the user-authorized 500 cases are approved. Another 250 cases receive actual open evidence requests followed by reassessment. The remaining 250 stay paused for approval. Curated term metadata states `manual_synthetic_fixture`; no model extraction or independent SME review is asserted. Existing 30 cases remain unchanged.

`legacy_archive.py` imports a strict, synthetic-only historical schema into separate supplier/contract/case tables. Exactly 2,000 invented cases (1,000 each for 2024/2025) across 50 vendors and 100 contracts are stored as `MIGRATED_APPROVED`. They are labeled **Migrated from old system — locked (synthetic)**. Immutable database triggers and operational review/assessment guards prevent these archive rows from entering the current approval process. Their historical values never contribute to current candidates, approved-for-draft value or realized recovery. Repeated identical import is idempotent; conflicting IDs, malformed inputs or mismatched relationships reject atomically.

The expanded profile is published only after schema, source/import reconciliation, original-row preservation, original approval authority, checkpoint state and zero realized recovery checks. Its marker must match database metadata at startup. Startup itself performs no import, model call or approval. Source databases and checkpoints remain untouched. See `scripts/create_scaled_demo.py` and `var/scaled-demo/verification.json`.

Invoice and contract pages provide vendor name/ID search, exact contract filtering, current/historical cohort filters and bounded 50/100-row pages. Bulk approval selects only visible eligible cases and rechecks uncached state for every decision. Historical browsing uses SQL without assessment, retrieval or workflow calls. `ui_cache.py` caches display snapshots by DB/checkpoint/WAL file identity, normalizes absent and empty WALs, rejects changes during loading and revalidates semantic-index readiness independently. Cached displays never authorize decisions. Line-scoped refreshes and SQL-scoped fingerprint reads reduce per-decision work without changing calculation semantics.

The main localhost launcher defaults to this expanded profile at port 8515, with SQLite FTS5 retrieval. An expanded semantic index is not yet built; the earlier million-dollar vector index must not be reused with the new sources. The earlier profile remains explicit through `--profile million`. This volume demonstration does not establish multi-user security, ERP compatibility, real-data accuracy or production throughput.

## 5. Implement and verify the rate engine

This portion is implemented in `engine.py`.

Determine the exact supplier, contract, role and location; check currency/unit/service scope; verify payment and approved quantities. Use service dates, not invoice dates, to determine rate applicability. Follow explicit term supersession. Include all signed amendments known in the snapshot, and block a decision when a signed amendment has not been reviewed. Do not treat the most recently uploaded file as automatically controlling.

For a line spanning rate periods, evaluate dated service entries separately, sum their exact Decimal products, and round once at invoice-line level. Do not prorate by calendar days without approved hours. Compare the billed amount with the expected amount, then subtract only matched, applied pricing credits or received refunds.

Example implemented fixture:

```
L001: 8,000 approved hours × billed rate 130 = 1,040,000
Applicable signed rate: 120
Expected amount: 8,000 × 120 = 960,000
Gross candidate: 80,000
Applied offsets: 0
Net candidate: 80,000
```

L004 demonstrates net rather than gross reporting: 1,000 gross less 400 already credited equals 600. L003 and L018 demonstrate zero remaining recovery when the discrepancy is already fully credited/refunded.

A NEEDS_REVIEW result carries zero into the candidate total because the supported amount is unknown, not because there is no possible overpayment. Required blockers include missing contracts/service records, conflicting terms, incomplete exports, unreviewed signed amendments, currency mismatch, partial payments, pending adjustments and unsupported overtime.

**Gate:** all supplied acceptance tests pass, and any change to precedence or rounding adds a regression test before code is changed.

## 6. Implement real retrieval-augmented explanation

The baseline `retrieval.py` retrieves the complete scoped signed-document set and uses SQLite FTS5 to rank it for explanation. It does not require embeddings to demonstrate useful retrieval. Full-scope inclusion prevents a top-k lexical/semantic miss from silently omitting an amendment. Drafts, untrusted notes and other-tenant records are excluded.

The optional local semantic path is now implemented for synthetic pilot use. `chunking.py` creates exact overlapping character slices with source hashes, IDs, offsets and line ranges. `embeddings.py` calls a pinned local Ollama embedding model with truncation disabled. `vector_store.py` stores validated generations in embedded Qdrant, and `semantic_retrieval.py` reconciles those generations against canonical sources before ranking. An explicit index path and local-embedding consent enable this path in the CLI, LangGraph and Streamlit; default retrieval remains unchanged. Rebuilds pin a new generation and invalidate pending reviews using the old generation. This is paragraph/line-aware character chunking, not a legal-clause parser or token-aware splitter.

Semantic ranking retains the complete scoped signed-document set independently of top-k hits. The explanation still receives complete relevant documents and withholds oversized evidence rather than trimming amendments. The fake invoice workflow runs locally without ERP, document-repository or identity-provider connections. See README commands and [production readiness](PRODUCTION_READINESS.md) for what is implemented and what remains unverified.

The offline explanation is a deterministic template and is labeled as such. It is not an LLM-generated RAG answer.

Enable local generation explicitly:

```sh
uv sync --extra ollama
uv run recover explain --line L001 --llm --provider ollama --model llama3.2:latest
```

The native HTTP adapter in `ollama_provider.py` accepts loopback endpoints only, ignores proxy environment variables, rejects redirects and cloud/remote models, and verifies an installed model's digest. It uses a 60-second socket timeout and a 262,144-byte response cap; the socket timeout is not an overall wall-clock deadline. The full scoped evidence is retained, including source hashes; a conservative prompt/schema/output budget rejects inputs exceeding the configured context instead of trimming amendments. No tool calls or automatic external fallback are available. Configure the local server with `OLLAMA_NO_CLOUD=1` separately if cloud features must be disabled server-wide; application endpoint checks are not an independent audit of server configuration.

`ai.py` shares a strict `FactDraft` schema across Ollama and OpenAI: exact fact-ID/value claims, an allowed next-action code and citations. Wrong types and extra fields are rejected. Free-text summary and limitation fields are no longer accepted from a provider. The numeric Python case is authoritative and read-only, and document text is explicitly untrusted. The existing `--llm` OpenAI provider selection remains available; offline explanations remain the default.

`explanation_facts.py` builds a required fact catalog from the assessed case, without reading expected results or evaluation labels. It preserves exact financial strings and unresolved nulls, status/reason, warnings and dated service components. The model must return every required fact exactly once; known IDs with altered values are contradicted, and unknown or malformed claims are unsupported. Missing or duplicate facts also withhold the draft. Financial consistency is reported separately from full required-fact coverage. Action codes must match the assessed status. New realized recovery is fixed at zero under the demo's existing policy.

The Ollama request grammar is narrowed to the current fact IDs/count, allowed action, governing citation count and canonical quote choices. The independent validator checks all of these again; provider grammar compliance is not treated as proof. Rejected closed claims expose a safe structured check report to the evaluator without displaying the rejected narrative. Evaluation counts case-level generation attempts, including preflight refusals, rather than claiming each attempt reached the provider.

Case component provenance must match a reviewed term/document quote, source bodies must match their hashes, and citations must include the exact canonical reviewed quotations. A genuine quote elsewhere in the same document is insufficient. The earlier source-membership validator remains available, but quote membership alone does not establish entailment. After validation, Python renders factual summary/action/limitations and returns `composition: validated_fact_template`, the closed-claim validation report, original source citations and case/source digests. The provider chooses only within the closed contract; this is not free-form model-authored narrative or a universal semantic judge. Provider failures never modify the case or approve it.

CLI `explain` and `graph-start` accept `--provider`, `--model`, `--ollama-url` and `--context-tokens` only with `--llm`. Streamlit requires provider-specific consent. LangGraph resolves configuration once before starting and persists it; replay uses that configuration and rejects changed local model identity. The new contract uses prompt version `supplier-explanation-v2-facts`; old pinned LLM threads must start a fresh review, including threads already paused at approval. Existing offline threads still resume.

Mocked provider and workflow regressions cover strict structure, invalid/omitted citations, whole-evidence transmission, local transport failures, unchanged financial cases and checkpoint replay. The opt-in live check is separate from unit discovery:

```sh
uv run python scripts/evaluate_ollama.py --run-local --model llama3.2:latest
```

It uses only bundled synthetic fixtures in a temporary database. L001 covers the superseding amendment and L011 covers a line spanning governing rate periods. Record the actual live result and model digest; unit-test success alone does not verify generation quality. The October 17 local smoke passed transport/schema/citation checks, but manual inspection found incorrect L011 claims about absent amendment and discrepancy evidence. Those observations are retained in `var/ollama-implementation/semantic-review.json`; no claim of narrative accuracy is made. Local term extraction, approval-state changes and settlement remain outside this milestone.

The subsequent closed-fact enhancement passed 151 unit/integration tests with zero skips, 21/21 offline controlled regressions and a final three-case local generation check (L001, L011, L006) on the installed `llama3.2:latest` digest. The first live trials were withheld for missing facts, an incorrect action or invalid quotations; these failed reports remain available. The final L011 template retains both dated rates and USD 100.00 candidate. See `var/faithfulness-implementation/live-evaluation.json` and `var/faithfulness-implementation/VERIFICATION.md` for scope, commands and remaining SME/held-out gates.

## 7. Add contract extraction without granting it authority

The supplied `propose-terms` command is an initial extraction adapter:

```sh
uv run recover propose-terms --doc DOC-A2 --output var/proposals/DOC-A2.json
```

It produces a structured **UNREVIEWED_PROPOSAL** with a source hash, an exact quote and unresolved questions. It does not change `reviewed_terms.csv` or the database's approved terms.

The synthetic baseline deliberately uses human-reviewed rate fixtures. Do not pitch it as an end-to-end document-understanding product yet.

Step 9 now adds a separate local Ollama adapter (`term_extraction.py`) and a durable reviewer screen showing source text and proposed fields side by side. Reviewers confirm supplier, contract, currency, unit, role, location, effective dates, coverage, supersession and source version. A typed reviewer label remains demo-only; authenticated identity is still a pilot gate. Quotes and source hashes remain linked to approved-term versions. New scoped signed evidence invalidates prior analysis before extraction; it cannot be cleared by rejecting a proposal.

For PDFs later, ingest text and tables with page coordinates; preserve originals and page references. Only use OCR when a document lacks usable text, and measure extraction confidence. Do not call an OCR/LLM result an approved contractual fact. Scanned/PDF ingestion is not included in this starter.

## 8. Review workflow and interface

The optional Streamlit app provides one review workflow: select a line, inspect the Python assessment and signed sources, explicitly start generation, review the saved explanation, choose a decision and save it locally. It is bound to localhost. Its reviewer-name field is a demonstration label, not identity verification. Its queue shows initial candidate amounts and does not yet implement a full case-management lifecycle after approval.

Both the Streamlit review controls and the CLI use LangGraph for:

```
assess → retrieve → draft → human interrupt → append local review event
```

Start with a unique thread ID and resume the same thread. SQLite checkpoints allow reconstructing the graph between commands; the optional integration test specifically exercises this. Approval recomputes and compares the displayed case digest before recording a decision. Journal writes use stable event IDs to avoid duplicates when a node is replayed. Non-candidates cannot be approved for drafting.

Streamlit creates a fresh UI thread for each started review and restores the latest saved UI workflow for the selected invoice. Discovery uses read-only checkpoint access, creates no missing files and invokes no nodes or providers. A browser/app rerun therefore keeps the original draft and pinned model configuration. The UI offers decisions only at the candidate's approval node after a valid draft; a failed draft is not a human-review pause. Consent is scoped to provider, line and case digest. A changed case or prompt contract requires a fresh review.

The accountant workspace uses persistent left-sidebar navigation: Overview, Invoice Review, Contract Term Review and Evidence Requests. Overview keeps USD portfolio values and review workload, with portfolio coverage between them. **Total approved records** uses the fresh all-approved USD forecast basis and shows its current/historical breakdown, including completed locked historical approvals even when no current forecast assumption exists. The expanded demo shows **2,520 records: 520 current / 2,000 historical**. The separate workload card is labeled **Current approved for draft** and retains only current eligible cases; archived records never enter its approval actions. Supplier and contract coverage spans current records and locked history, with amendments grouped under their contract. Rendering these counts performs no writes or model calls.

Invoice Review contains vendor/contract filters, a selectable case table and a decision area below it. Opening a case or checking one row shows its individual decisions; two or more checked rows show only the bulk form. Explicitly opening another case clears the checkbox selection. Clearing or deselecting the last checked row closes that review instead of restoring an older case. A transition back from multiple rows to one retains that row's frozen selection token, so an intervening source or workflow change still blocks the decision. Successful explicit preparation or reassessment keeps the case open and clears its checkbox selection, allowing the existing individual-review refresh to display that action's new state. It defaults to outstanding work; completed cases remain available through All cases and Overview links. Source wording and validated calculations are presented in plain language. Portfolio metrics, raw JSON, source hashes and the separate invoice dropdown are removed from this worklist. Filtering and selecting a case are read-only and do not generate an explanation.

`presentation.py` formats assessed facts without changing the financial engine or checkpoint state. Money is formatted from Decimal strings; unresolved values stay **Not established**. The selected case shows its calculation, dated rates, supporting quotations and saved explanation. Historical drafts remain tied to their saved evidence and cannot authorize a changed case. Provider preparation and any local embedding use require an explicit action and consent.

Reviewers enter a label and note, then choose **Approve for draft**, **Reject**, or **Needs review**. The latter uses the existing `request_more` / `NEEDS_MORE_EVIDENCE` outcome and typed evidence request. Candidate decisions resume the actual human-review interrupt; a failed draft, stale assessment or open evidence blocker cannot be approved. A case outside the candidate workflow directs the reviewer to evidence handling rather than inventing an approval state.

Bulk approval is shown only for two or more selected rows and is limited to eligible cases in the displayed vendor/contract scope. Select-all never includes hidden cases; a page with only one eligible selected row uses individual review. Selected identities and amounts are pinned for review and each case is checked again when saving. A known changed or unready selection blocks bulk submission before service invocation. Each decision resumes that case's existing workflow with its stable journal identity. A batch can partially succeed, so the UI reports individual successes, blocks and failures; replay does not add another journal entry. A committed journal event whose checkpoint completion was interrupted offers completion of the recorded decision instead of another choice. Saving an approval does not change the financial assessment or create a recovery.

Contract Term Review follows supplier → contracts/amendments → related cases, clearing incompatible selections as the parent changes. Existing human-curated approved rate terms are displayed without requiring AI extraction. Original source text remains available using business labels. Local extraction settings and technical source metadata are absent from this page. Saved proposal revisions can still be corrected and reviewed separately; source coverage, exact scope, dates and explicit supersession remain approval requirements. The CLI extraction adapter remains available for explicitly authorized synthetic preparation.

Evidence Requests shows only OPEN and EVIDENCE_ATTACHED requests, filtered by vendor and contract. Selecting a row binds attachment and resolution actions to that request's invoice and scope. Adding evidence does not resolve a request: reviewed contract terms and an explicit disposition are required. Synthetic text intake remains available from this selected-request context, with the synthetic-only guard and signed-simulation attestations. Resolved requests leave this operational queue, while cases needing reassessment remain in Invoice Review. Calculation evidence gaps without a structured request remain visible in Invoice Review, so an empty request list does not imply every case is resolved. Payment, credit, service and unsupported pricing gaps retain their existing limitations.

Reassessment uses approved terms and starts a new current assessment, never inheriting an older case approval. A fresh linked review is required for any new candidate. Immutable source, term, request, assessment and decision histories remain saved even though technical history tables are removed from the accountant pages. The Streamlit checkpoint sidecar remains `<database-stem>-ui-checkpoints.sqlite3`; discovery is read-only and provider retries are explicit. Reviewer labels remain demonstration labels rather than authenticated identity.

This expansion passed 169 tests with zero skips, including 13 integrated UI tests and 22 workflow tests. The UI tests use real graph/checkpoint/journal operations with simulated provider responses. A separate live Ollama L001 UI run generated once and restored the paused draft in a fresh session without another model call; no human decision was recorded for that sample. Visual browser verification could not run because no browser/native automation surface was available. Results are in `var/human-review-expansion/L001-addc070124af/`.

`APPROVED_FOR_DRAFT` is not supplier acceptance, an approved receivable, cash receipt, or a realized recovery. Sending drafts and connecting accounting systems are excluded. Future case states may include validated, draft approved, submitted, disputed, credited, refunded and closed; each transition needs permissions and evidence.

**Gate:** verify pause → process restart → resume, stale-case rejection, rejected decisions, replay idempotency, and tenant/thread isolation before demonstrating workflow reliability.

### Step 9 implementation and state model

The local implementation now includes all three increments:

| Increment | Implemented behavior |
|---|---|
| 9A | Additive, backed-up migration; immutable proposal revisions and raw output; local extraction; transactional human term approval/rejection; approved-term provenance/projection; persisted assessment revisions; source/approval freshness; fixture reinitialization protection. |
| 9B | Staged synthetic UTF-8 TXT/MD intake; explicit known scope and signed-simulation admission; immutable versions and original bytes/text; duplicate detection; incremental FTS registration; conservative invalidation; separate reviewed no-pricing-effect/out-of-scope dispositions. |
| 9C | Typed evidence requests linked atomically to new case decisions; explicit legacy request creation; attachment/disposition history; contract-only resolution; deterministic reassessment; durable new-review intents and thread lineage; crash/replay/concurrent-start handling; current-only dashboard values. |

`lifecycle_storage.py` migrates without replacing existing source or journal rows. A verified SQLite backup precedes a working-data migration. Initial scope fingerprints preserve unchanged legacy case digests; later hashes include the calculation version, source/term versions, transaction facts, completeness metadata and evidence-request state. New input with an unchanged final amount still invalidates old review authority. Proposal freshness uses source/rate context separately so attaching already-admitted evidence does not itself change a pricing proposal.

The existing `reviewed_terms` table is an engine projection. Only the human service may add a reviewed term, together with its immutable approved version and review event in one transaction. Exact existing fixture terms return their original reference. Signed intake sources require explicit complete single-rate scope coverage; finding one term is not document completeness. Both directions of the version/projection mapping are checked before calculation, including status/scope/date drift that could otherwise hide a known signed source. Historical rows have database UPDATE/DELETE guards. Mutable request/intent state is a projection over retained events.

Case-review outcomes remain `APPROVED_FOR_DRAFT`, `REJECTED`, and `NEEDS_MORE_EVIDENCE`. Proposal outcomes remain separate (`UNREVIEWED_PROPOSAL`, `TERMS_APPROVED`, `TERMS_REJECTED`, `PROPOSAL_STALE`). Assessment validity is `CURRENT` or `STALE`; evidence requests are `OPEN`, `EVIDENCE_ATTACHED`, or `RESOLVED`. New assessment revisions do not inherit earlier approvals even when their case digests/amounts match. Reviewer labels are not authenticated users, and term approval is not case approval.

`dashboard.py` counts distinct current invoice lines, not journal events. Awaiting approval requires a validated draft at the actual human interrupt. Unstarted/failed preparation and stale assessments have separate buckets. Approved-for-draft dollars are a subset of calculated candidates; rejected, held and excluded historical amounts are separate. Unknown values stay unknown, currencies are never combined, the underlying financial summary leaves expected cash unestimated and realized recovery stays zero. The separate forecast service described below supplies an explicit planning estimate without altering that financial summary. Existing counts and values are not read from expected-answer files.

Review intents are persisted before a checkpoint starts. A local process lock serializes duplicate starts without holding a database write lock during Ollama calls. Freshness is rechecked on retries and pinned into the graph's initial assessment. A crash releases the process lock; retries read existing checkpoints. A failed preparation node can run again only through an explicit preparation retry, never by supplying a human decision. Completed request-more threads remain completed; reassessment starts another linked thread. A model response arriving after source changes is withheld.

The isolated synthetic acceptance scenario verifies an invoice's current candidate changing from USD 800.00 to USD 1,300.00 after an explicitly superseding amendment, with its earlier request-more decision retained, one new review decision and no double counting or realized recovery. The original 20-case fixture dataset is exercised separately and retains USD 80,707.50.

The separate resolved-evidence demonstration at port 8513 is an explicitly hypothetical fixture variant, not an extension of the transaction-correction services. Registered invented records produce twenty current assessments with no evidence-review blockers; copied historical approvals remain stale and never transfer. Original signed documents and terms are retained, with one manually curated fictional missing agreement added. Original working data/checkpoints and the shipped fixture baseline remain unchanged. The scenario's changed service date and replacement USD/REGULAR records must not be represented as resolving the September supersession conflict or adding FX/overtime handling. See `var/demo-evidence-resolution/` for before/after records, hashes, preserved history, request-resolution events and verification.

The local Ollama extraction smoke exercised real inference and persisted an unreviewed proposal in a disposable database. It returned the correct rate but an unsupported end date and incomplete coverage; those cannot become approved inputs without correction and complete human review. General extraction correctness, independent SME acceptance, browser visual layout, live OpenAI and real-data accuracy remain unverified. See `var/step9-implementation/` for commands, backups, diff and actual verification output.

### Cash forecasting after approval and the demo auditing team

`forecast.py` adds the missing post-approval forecast assumption. `basis_from_summary` includes only current approved USD candidates and freezes each case digest, approval event, assessment identity and exact Decimal amount. `save_forecast` reloads the uncached basis under canonical/checkpoint write locks, verifies the displayed basis digest, validates a Decimal percentage in 0–100 plus reason/reviewer/as-of/horizon, and appends an immutable assumption. Idempotency rejects changed payloads; retries do not append duplicate assumptions. History cannot be updated, deleted or replaced. An unchanged total with a different approval identity still invalidates an older forecast. Reads create neither tables nor assumptions.

Overview reads the explicitly scoped `all_approved` forecast: current approved USD cases plus locked migrated approved USD records. The cash-recovery assumption editor is removed from Overview; the forecast service and existing saved assumptions remain available for explicitly reviewed maintenance. The isolated `forecast_ui.py` renderer remains tested separately. Its separate policy, complete record identities and assumption history preserve the original current-only forecast unchanged. `portfolio_basis` includes archive ID, source record and case hash, validates approved/locked/synthetic provenance and excludes other tenants and currencies. Fresh all-approved saves recheck both operational cases and archive records under the same locks. Adding another approved record invalidates the saved projection even if the old current-only basis is unchanged.

Overview retains three financial grand totals, without year/cohort sections: **Total approved value**, **estimated cash still to recover**, and **realized recovery**. Below the blue financial panel, mint **Portfolio coverage** counts distinct suppliers and contracts across current records and migrated history, grouping amendments under their contract. Light yellow **Review workload** shows seven current-case queues with colored status cards. Six queues offer paged case tables; awaiting approval is a count only on Overview and remains actionable in Invoice Review. A stale saved approval-table selection is cleared. The Deploy control is hidden only while Overview is rendered. Selected rows open the exact invoice after clearing incompatible search filters. Browsing never starts a workflow or records a decision. The main all-year synthetic portfolio has USD 3,500,000.00 in 2,520 approved records. Its original 80% scenario modeled USD 2,800,000.00. Pending/rejected/evidence-blocked/stale operational cases do not contribute. Archive records stay immutable and nonactionable; they are not new current draft approvals. The original current-only calculation, USD 1,500,000.00 approved-draft amount and USD 1,200,000.00 saved forecast remain unchanged.

The user separately authorized an invented 2024/2025 historical receipt scenario. `historical_recovery.py` adds immutable synthetic-only receipt records, linked to the exact saved all-approved forecast and original archive/source hashes. It records 100% of the **already 80%-discounted** historical estimate: USD 800,000.00 for each year, USD 1,600,000.00 total. No second probability discount is applied. Receipt dates are simulated year-end dates; creation/reviewer/rationale provenance distinguishes this scenario from actual migration or cash evidence. Existing archive, approval, forecast and checkpoint records remain untouched.

The remaining estimate excludes completed historical records and applies the current saved rate once to the still-outstanding approved basis. At 80%, Overview shows USD 1,200,000.00 estimated and USD 1,600,000.00 realized in the explicitly labeled synthetic history. Their sum equals the original USD 2,800,000.00 cash expectation. Recorded receipts remain fixed if a later forecast changes. Reads create no records; explicit writes recheck fresh canonical inputs, reject stale or changed idempotent payloads, and cannot update/delete/replace receipts. This is a historical demo overlay, not a general settlement workflow; current operational and original fixture realized recovery remain zero.

Stale or absent assumptions show “Not estimated”; explicit zero percent shows USD 0.00. The percentage represents the portion expected back **as cash** by the target date, not merely supplier acceptance or an issued credit. It is uncalibrated synthetic planning data. Forecast records remain separate from source/rate projections and the review journal, so saving them cannot change approved source amounts, source hashes, case digests, candidate totals, approval state or realized recovery.

Actual supplier acceptance, claim submission, payment-system posting and settlement reconciliation remain future work. This milestone sends no communications or claims, connects no ERP and asserts no actual cash receipts. A separately reviewed settlement ledger with independently verified cash evidence is still required before real realized recovery can be counted; the invented historical overlay does not supply that capability.

`workspace_ui.py` supplies six sidebar buttons and business descriptions. `team.py` / `team_ui.py` provide a fictional directory of two escalation managers and ten accountants, selection of a label for future invoice decisions, and read-only activity drawn from actual saved events. The directory does not reassign past reviewers, fabricate activity, authenticate users or send escalations. Local sign-in is provided separately below. Session preferences affect only future reviewer labels; all existing approval/freshness/idempotency checks still apply.

### Read-only conversational workspace and vendor contacts

`chatbot.py` routes bounded natural-language questions to allowlisted readers for vendors/POCs, contracts, cases, evidence requests and portfolio totals. IDs, named entities and conversational references are resolved inside the selected demo tenant. Ambiguous/multiple entities and unsupported or negative status filters require clarification; unknown IDs cannot widen a search. SQL is parameterized and is never supplied or generated by a model. Current case status uses the same dashboard projection, and all-year financial answers call the same historical-recovery/forecast service as Overview. Money is rendered from Decimal strings, with invented historical receipts and zero current realized recovery kept distinct.

Case explanations retain complete relevant signed sources, verify hashes/quotations and combine fresh assessments with current review/evidence history. Source projections are reconciled before signed/date/scope filtering, so a changed status or scope cannot silently hide an amendment. The chatbot UI uses offline factual answers. The service API still supports explicitly opted-in Ollama calls using the existing constrained case explanation, checked again against fresh case, review and source identities before display. A failed or stale generation is withheld. Contract-level questions quote the complete known signed rate documents for the selected contract; this is not unrestricted conversational legal interpretation. FTS5 remains the expanded profile's retrieval default; the assistant does not silently reuse another profile's Qdrant index or create a new one.

`chatbot_ui.py` provides chat messages, starter questions, result tables, evidence expanders and read-only record links. It uses the default validated factual flow without new model calls or a model-selection control. Six turns are retained in session state, scoped to the database and tenant; changes in the portfolio/checkpoint stamp invalidate prior answers. `chatbot_navigation.py` resolves links against canonical scoped records before selecting a view. Locked historical links use their saved cohort and exact case ID, including records beyond the first table page. No chat path invokes approval, reassessment, request resolution, forecast saving, supplier communication or checkpoint creation.

`vendor_contacts.py` stores fictional POC names in a separate tenant/vendor-keyed table. Reads use read-only connections and never initialize a missing schema. `scripts/seed_vendor_contacts.py` is an explicit synthetic-only seed, covering operational and migrated suppliers without modifying either source table. It rejects malformed/foreign/duplicate inputs, rolls back failed seeds, preserves existing valid names and rejects explicit conflicting replacements. Contacts remain separate from financial/case/forecast digests and cannot change approval authority. Apply the seed only after taking a verified database backup; the implementation verification retains one for the expanded demo.

`search_ui.py` provides native typeahead suggestions for every free-text search field, retaining existing keys, empty-filter behavior and custom substring searches. Suggestions come from the complete parent vendor/contract/cohort scope before table pagination. Strict existing identity/status/accountant selectors retain their closed option sets. Search values are filters only; they do not change rates, reviewer notes or approval authority.

`workspace_ui.py` adds a short business usage guide to each of the six pages. `ui_guidance.py` supplies table introductions, displayed-row/matching-record counts and native column-header help. Each table and matrix explains its scope and the next review action; Overview and team metrics have separate measure definitions. Previously static calculation, service-date, reviewed-term, proposal and historical-value matrices use read-only dataframes so column help is available consistently. Generic column configuration preserves inferred data types and original formatted money strings. Existing row keys, pagination, checkbox edits, review callbacks, filtering and decision services remain authoritative. Guidance performs no database access, calculations or provider calls.

`suggestions.py` reads the selected synthetic portfolio through read-only SQLite connections, with tenant, source-projection and historical provenance guards. Its catalog covers vendors, fictional contacts, contracts, cases, invoice IDs, source identities and open evidence requests. The chatbot composer displays supported canonical questions with friendly vendor/contact/contract/invoice labels. Formatted labels help discovery; the selected raw question keeps exact entity IDs, including when contacts share a name. Users may also type an original question. Only **Send message** submits it; typing and suggestion selection make no model or workflow calls. The form clears after submission, and ordinary rerenders do not repeat a question. The supported UI dependency is Streamlit 1.47 or later, which provides native free-text selectbox options.

### Local account access and presentation

`app.py` calls `auth_ui.require_login` before opening the portfolio, loading its manifest, rendering navigation or invoking any business reader. An unauthenticated session sees a matching first-owner setup or sign-in page. A successful setup does not automatically sign in. Bootstrap is an atomic first-account-only operation; subsequent accounts require a freshly validated owner session. There are no default credentials or post-setup public registration.

`local_auth.py` uses a separate private SQLite account store and Python's `hashlib.scrypt` with N=32768, r=8, p=3, a random 16-byte salt and a 32-byte derived hash. This is an [OWASP-listed scrypt configuration](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html). Passwords are 15–128 characters, never logged or stored as plaintext. Unknown users incur the same password-hash work as known users. Atomic persistent attempt reservations allow at most five credential checks per username and 50 per store in a 15-minute window. The global ceiling can temporarily block legitimate users after repeated attempts; it is a local rate limit, not a distributed abuse-defense service. A Python build without scrypt fails closed; the recommended uv runtime supports it.

Random opaque session tokens stay in Streamlit server session state; only token hashes are stored in SQLite. Every full app rerun revalidates account activity, security version and 30-minute idle/eight-hour absolute expiry before portfolio access. Browser reload can reset [Streamlit session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state), requiring another login. An idle screen is cleared when the next interaction detects expiry; it is not a timed remote screen lock. Password change/reset and disable revoke the affected account's sessions. Logout also clears cached browser-session selections, chat and consent. Shared display caches retain their existing scope and freshness checks; they are reachable only after the app gate.

The sidebar account menu lets the owner create, reset or disable accountant accounts and lets each user change their own password or sign out. Authentication uses its own schema, rejects a business database as an account store, and does not alter case, term, approval, checkpoint or forecast records. First-owner creation trusts the person with initial access to this loopback app; complete setup locally before sharing the computer. Forgotten owner password recovery is not provided in this milestone. Existing portfolio services/CLI remain local programmatic interfaces; the Streamlit gate is not authorization on direct database or Python access.

All accounts share the synthetic DEMO scope. Owner/accountant roles apply to account administration, without tenant/vendor/document authorization. Fictional reviewer labels remain independent from authenticated accounts, so this is not a security audit trail of individual finance decisions. Company SSO, MFA, account lifecycle integrations, authenticated decision attribution and independent penetration testing remain future work. The local access gate does not fulfill the real-data/shared-service gates in section 10.

`presentation.py` retains the original Decimal formatting and assessment helpers and adds static theme/brand rendering. The navy sidebar, teal navigation/actions, white forms and grids and responsive sign-in panels make the six-page workspace consistent. Page/table explanations, column help, native keyboard controls and all financial values and workflow guards remain unchanged. No external asset/font provider or new dependency is introduced.

## 9. Evaluation beyond the toy fixtures

The package separates these checks:

- Financial/data tests: 20 synthetic scenarios plus import, Decimal, source, scope and journal edge tests.
- Document-coverage tests: eight required/forbidden-source cases. They do not measure ranking quality, model groundedness or generalization.
- Optional integration tests: actual LangGraph pause/resume/replay and simulated Streamlit controls must execute when their extras are installed; skips are reported separately.
- Ollama generation: the separate opt-in synthetic smoke script exercises the installed local model; mocked provider tests do not establish live model behavior.
- Closed-claim explanation regressions: `scripts/evaluate_explanations.py` evaluates 21 developer-labelled synthetic cases, separately checking exact fact support, financial consistency, required coverage and canonical citations. Labels are test-only and provisional pending independent SME review. This visible set is not a held-out benchmark.

```sh
uv run python scripts/evaluate_explanations.py
uv run --all-extras python scripts/evaluate_explanations.py --run-local --model llama3.2:latest \
  --output var/faithfulness-implementation/live-evaluation.json
```

Offline evaluation makes no model calls. The opt-in local run covers L001, L011 and L006 using a temporary synthetic database. It reports failed/withheld generations, keeps full inputs intact and asserts 20 audit lines, USD 80,707.50 candidates, USD 0.00 realized recovery and zero review events. A `supported` claim means exact agreement with the closed fact catalog, not general prose entailment or independently validated legal applicability. No model judge or Ragas dependency is installed; any future semantic judge needs calibration against independent SME labels before its score can be used as an acceptance gate.

Add generated-answer tests before enabling AI for reviewers. Label correct source documents and the expected abstention/decision with a knowledgeable reviewer. Test paraphrases, contradictory evidence, omitted amendments, provider failures, fabricated citations, suspicious document instructions, different tenants, and long evidence sets. Evaluate extraction accuracy separately from retrieval and final explanation.

Maintain a held-out set that Codex is not allowed to alter. In a pilot, review both flagged and sampled unflagged invoices; only reviewing flagged cases does not measure missed discrepancies. Track precision of proposed cases, missed valid cases in the reviewed sample, evidence completeness, reviewer correction time, preparation time, token spend and API failures. Do not extrapolate the fixture's perfect results to production accuracy.

## 10. Pilot environment and release gates

Do not expose this local app publicly or load real supplier data yet. Before a real-data or shared private pilot, implement authenticated access, server-derived tenant/role permissions, row/document-level authorization, secure secrets, encrypted storage/transport, retention/deletion, input quarantine, audit retention, backups and a tested recovery procedure. Apply approved source-system mappings and end-to-end reconciliation. Obtain explicit permission before source data or traces leave the organization. The selected local synthetic demonstration does not satisfy or bypass those future gates.

Move to PostgreSQL/object storage only when a multi-user deployment and the data volume justify the additional services. The new local embedding/Qdrant path is a synthetic pilot foundation; evaluate its ranking against independently reviewed, held-out questions before adopting it for business data. Semantic similarity does not replace exact financial joins or amendment completeness. Qdrant's embedded mode uses an exclusive local directory and is not the shared server deployment needed for concurrent production access.

Keep separate development, staging and pilot datasets and credentials. Preserve human approval before supplier communications and financial-system writes. Where writes are later added, require least-privilege credentials, action allowlists, idempotency keys, reconciliation and a reliable outbox—not an unrestricted SQL or email tool given to a model.

Implement a realized-recovery ledger only after finance agrees on settlement evidence and counting rules. A credit/refund cannot be counted twice. Exclude recoveries already found by existing controls when calculating incremental product value. Deduct reviewer time, infrastructure/model cost, maintenance and supplier follow-up from measured benefit.

## 11. Official references used to check implementation direction

Original references checked October 16, 2026; Ollama integration checked October 17, 2026. Dependency installation and the installed optional integrations have since been tested locally; see README verification limits.

- Codex IDE extension: https://developers.openai.com/codex/ide/
- Codex project instructions: https://developers.openai.com/codex/guides/agents-md
- uv installation: https://docs.astral.sh/uv/getting-started/installation/
- uv Python management: https://docs.astral.sh/uv/guides/install-python/
- Python Decimal: https://docs.python.org/3/library/decimal.html
- SQLite FTS5: https://www.sqlite.org/fts5.html
- LangChain ChatOpenAI: https://docs.langchain.com/oss/python/integrations/chat/openai
- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- Streamlit run: https://docs.streamlit.io/develop/api-reference/cli/run
- Separate API billing: https://help.openai.com/en/articles/9039756-managing-billing-for-chatgpt-and-the-api-platform

Ollama references checked for this integration: [structured outputs](https://docs.ollama.com/capabilities/structured-outputs), [chat API](https://docs.ollama.com/api/chat), [model metadata](https://docs.ollama.com/api-reference/show-model-details), and [FAQ](https://docs.ollama.com/faq).
