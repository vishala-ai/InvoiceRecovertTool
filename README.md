# Supplier/ Vendor Invoice Recovery Tool

A synthetic-data-only review tool, with a local workspace and an optional serialized Cloud pilot, for a precise question:

> Which fully paid hourly-service invoice lines appear inconsistent with human-reviewed signed rate terms, after existing credits and refunds?

**Start with the calculation baseline, then enable RAG and workflow integrations.** The baseline runs without an API key, third-party runtime package, Docker, or a cloud account.

## What is included

- Schema-validated atomic SQLite import, source integrity checks and reconciliation reports; three DEMO suppliers and a separate tenant canary.
- 20 invoice-line scenarios; signed/unsigned/conflicting amendments; dated approved service records; payments; pricing credits and refunds.
- Decimal arithmetic, explicit rate supersession, source citations and review abstentions.
- Complete scoped signed-document retrieval with SQLite FTS5 ranking, plus opt-in local Qdrant ranking with source-linked chunks and Ollama embeddings.
- Optional structured explanations through local Ollama or LangChain/OpenAI, local Ollama term proposals with human review, plus the earlier separate OpenAI proposal experiment.
- Closed explanation claims with exact financial checks, reviewed-quote coverage and Python-rendered factual drafts.
- An optional LangGraph workflow with SQLite checkpoint/resume and a human-review interrupt.
- An optional localhost Streamlit workflow with saved explanations, a human-review pause and a local decision journal.
- Local password accounts, a sign-in gate and owner-managed accountant access for the synthetic workspace.
- Optional Groq-hosted Llama explanations and an atomic Supabase persistence adapter for the user-authorized Streamlit Community Cloud pilot.
- A read-only audit chatbot for vendor contacts, contracts, cases, evidence requests and shared financial summaries.
- A currency-separated Overview, immutable term/source/assessment histories, synthetic text inbox, evidence requests and linked reassessment (Step 9).
- Test-only ground truth, retrieval-coverage fixtures, Codex instructions and implementation milestones.

**It does not** ingest arbitrary PDFs, validate real signatures, provide company SSO or tenant-level authorization, connect an ERP, send supplier messages, file claims, or record real recoveries. Curated rate terms are not automated AI extraction. Read the limitations in the guide.

## Current local verification

Verified October 17, 2026 on Python 3.12.15 / SQLite 3.53.1:

- Dependencies resolved with `uv sync --all-extras`; the real `uv.lock` is included.
- 298 tests passed, zero skipped, including installed LangGraph integration, all Step 9 lifecycle paths, Overview metrics, source/term projection checks, concurrent starts and interrupted-save recovery. Ruff passed. Detailed output is in `var/step9-implementation/final-tests.txt`.
- The 20-case audit remains USD 80,707.50 candidates and USD 0.00 new realized recovery. Synthetic inputs and golden expectations are unchanged.
- Retrieval coverage is 8/8 with zero model calls. Core and import tests also run with third-party packages disabled.
- Closed-claim evaluation passed 21/21 controlled regressions without model calls, including exact financial values, governing quotes and contradictory claims. The new fact checks also execute with third-party packages disabled.
- With optional packages disabled, 156 tests executed successfully and 142 optional tests explicitly skipped (298 discovered). The separate core-only run passed 54/54. Skips are not included in the installed all-extras result above.
- Step 9's one-invoice isolated dataset passed USD 800.00 → USD 1,300.00, with one latest finding and a new linked review. The working database migration preserved all original source rows, case digests and four review events; read-only checks opened all five app sections without provider calls or new records.
- Final live `llama3.2:latest` checks passed for L001, L011 and evidence-conflict case L006. The request grammar constrains fact IDs/counts, review actions and canonical quote choices; independent Python checks still validate every result. Earlier failed trials are retained in `var/faithfulness-implementation/`.
- The integrated Streamlit L001 flow generated one local Ollama draft, paused for review and restored it in a fresh app session without regeneration. That automated verification did not record a decision. All three decisions and interrupted-save recovery were verified with temporary synthetic databases and simulated provider responses. The navigation fix was also checked against the live synthetic sample without changing its user-recorded decisions.
- **General semantic accuracy is not established:** visible drafts are now Python-rendered validated facts, not arbitrary model prose. The earlier L011 narrative defect is preserved in `var/ollama-implementation/semantic-review.json`. New labels remain provisional pending independent SME review; no held-out performance is claimed.
- Live OpenAI calls, browser rendering and real-data accuracy remain unverified. The tests use synthetic fixtures only.

The earlier creation-environment outputs remain in `examples/` as historical records. Passing fixtures does not establish accuracy on real contracts. The workspace currently has no Git repository, so no commit was made.

The October 18 semantic-retrieval milestone is a **local/private synthetic pilot**, as requested. It adds chunking, an embedded Qdrant index and local embeddings; it does not turn this application into a production service. See [production readiness and future integrations](docs/PRODUCTION_READINESS.md) for the remaining gates. No external business application is needed or connected for the simulated invoice workflow.

October 18 verification: **418 tests passed, zero skipped**, including installed LangGraph, real embedded Qdrant and Streamlit integration tests. Ruff passed; the dependency-free financial/core suite passed 54/54. A live `embeddinggemma:latest` run indexed seven synthetic signed documents into 24 overlapping chunks at the demonstrated 320/48-character settings (the default remains 1200/150). Live CLI and browser runs with `llama3.2:latest` generated validated L001 drafts and paused for human review without recording approvals. A one-hit semantic search retained both DOC-A1 and DOC-A2. The original 20-case USD 80,707.50 baseline and separate USD 1,000,000.00 approved-draft demo were verified unchanged, each with zero realized recovery. The source-projection regression also rejects a changed document classification that could otherwise hide an amendment. Commands, diff and verification artifacts are in `var/vector-retrieval-implementation/`.

## Streamlit Community Cloud — selected deployment

The selected Cloud design uses **Groq / `llama-3.3-70b-versatile`**, **Supabase PostgreSQL**, persistent accounts/reviews/checkpoints and repository **InvoiceRecovertTool**. The separate Cloud entrypoint commits the coupled workspace before showing saved results; it disables public first-owner registration and never falls back to local-only storage. Existing Python arithmetic, complete evidence and human-review guards remain authoritative.

See [Cloud setup, seed publication and verification](docs/COMMUNITY_CLOUD.md). Credentials are configured privately; local account files and runtime databases are excluded from the prepared source repository. External provisioning, actual provider calls and deployed persistence are not verified by the presence of this adapter.

## Main local demo — fixed address

The main app is **http://127.0.0.1:8515/**. Its default profile is the expanded synthetic portfolio:

```sh
uv run --all-extras --frozen python scripts/run_demo.py --check
uv run --all-extras --frozen python scripts/run_demo.py
```

The profile uses `var/scaled-demo/portfolio.sqlite3`, its matching `portfolio-ui-checkpoints.sqlite3`, and a reconciled `portfolio-profile.json` marker. It preserves the earlier 30 current cases and 20 existing approvals, then adds:

| Cohort | Vendors | Cases | Review state |
|---|---:|---:|---|
| Historical 2024 | 50 shared historical vendors | 1,000 | Approved — migrated, locked |
| Historical 2025 | Same 50 historical vendors | 1,000 | Approved — migrated, locked |
| New 2026 | 35 additional vendors | 1,000 | 500 approved, 250 awaiting approval, 250 needing evidence |

The historical archive has 100 contracts, carries the label **Migrated from old system — locked (synthetic)**, and has no approval actions. It is invented legacy data, not an actual migration. Archive records are separate immutable tables, excluded from current financial totals and review queues. Backend guards also reject assessment/review/reassessment of an archived line.

The complete current portfolio contains **1,030 cases across 38 vendors**: 520 approved, 250 awaiting approval, 250 requiring evidence, and 10 retained control cases. Current supported candidates total **USD 1,750,000.00**; approved-for-draft value is **USD 1,500,000.00**; new realized recovery is **USD 0.00**. The 50/25/25 split applies to the 1,000 newly added cases. Approvals were explicitly authorized for this synthetic demonstration, using validated offline drafts and real local workflow/journal writes. Pending cases have genuine persisted human-review pauses; evidence cases have open requests and current assessments. No model or external application was called to seed them. Rate terms are manually curated fixtures.

Search vendor names or IDs in Invoice Review or Contract Term Review, then filter contracts. Case-cohort selection separates current 2026 records from locked 2024/2025 history. Tables show 50 or 100 rows per page, and bulk selection covers only the visible page within the selected vendor/contract. Historical browsing uses SQL without generating current reviews. Display snapshots are briefly cached by database/checkpoint/WAL identity; all decisions still perform fresh, uncached validation.

The launcher verifies the published synthetic profile without contacting a provider and always binds localhost port 8515. Missing or mismatched files stop startup; it never creates data, generates explanations, builds indexes or adds approvals. The expanded profile uses canonical SQLite FTS5 retrieval. **No matching expanded vector index has been built**; the older million-dollar index is not reused or selected by the expanded launcher. Semantic retrieval remains an optional separately built, source-matched integration. Vendor search needs no embeddings.

To reproduce the new profile into a fresh directory (existing output files are never overwritten):

```sh
uv run --all-extras --frozen python scripts/create_scaled_demo.py --output var/scaled-demo-copy
```

The generator requires the existing earlier million-dollar source database and its checkpoints, copies them first, validates schema/source and Decimal reconciliation, curates synthetic terms, creates actual reviews, and publishes the marker last. Failed builds do not publish a profile. Original databases and original fixture outcomes remain unchanged. Generation/verification records are in `var/scaled-demo/`.

The earlier 30-case million-dollar profile and matching vector index remain available explicitly:

```sh
uv run --all-extras --frozen python scripts/run_demo.py --profile million --check
uv run --all-extras --frozen python scripts/run_demo.py --profile million
```

Stop an existing app on port 8515 before starting another. The launcher does not stop processes or install an automatic startup service. `.streamlit/config.toml` also sets port 8515. Direct `streamlit run app.py` selects its database through `RECOVERY_DB`. The generic CLI examples below retain the separate original 20-case fixture baseline of **USD 80,707.50 candidates**, with zero realized recovery.

October 18 expanded-portfolio verification: **495 tests passed, zero skipped**, including installed LangGraph, Streamlit and Qdrant integrations. Ruff passed; the dependency-free financial suite passed 54/54, and retrieval coverage remained 8/8. Original database/checkpoint hashes and original approvals were preserved. Expanded profile generation made zero model calls. Local measurements were about 3.5–4.0 seconds for a cold full worklist, about 60 ms for a single-case refresh, and 10–16 ms for cached display/search. These are single-process synthetic measurements, not a concurrent-user benchmark. Exact commands, diff, reconciliation and verification limits are in `var/scaled-demo/VERIFICATION.md`.

## Local sign-in

Open **http://127.0.0.1:8515/** and create the first workspace owner with your own username and a unique **15–128 character passphrase**. Then sign in. There is no default account or password. Complete this first setup yourself on the local computer before sharing access; the first account owns local account administration.

- **Account settings** in the sidebar lets the owner add accountant accounts, reset an accountant password or disable access. Accountants can change their own passwords. Share temporary credentials privately; the app sends no account emails.
- **Sign out** revokes the current session and clears this browser session's chat, selections, consent and form state. Changing a password revokes all sessions for that account; disabling an account also ends its sessions.
- Sessions expire after **30 minutes without interaction** or **8 hours total**, checked on the next interaction. Reloading the browser may require a fresh sign-in. Saved reviews remain in the journal.
- Passwords use independently salted scrypt hashes. Session tokens are hashed in the account store and retained in server-side session state. Login attempts are persistently limited for both existing and unknown usernames.
- Accounts live separately in `var/local-auth/accounts.sqlite3`, with private file permissions. `RECOVERY_AUTH_DB` may select another separate account store; it must never point to a portfolio or checkpoint file. The app fails closed if this store is unavailable. Use the uv-managed Python; a runtime without scrypt cannot create or authenticate accounts.

All signed-in users share the same **synthetic DEMO workspace**. Owner/accountant roles govern local account management only. The fictional team directory and invoice reviewer labels remain separate from sign-in identities; existing journal attribution is preserved. Company SSO, MFA, per-vendor permissions, authenticated decision attribution and owner password recovery are not implemented. Keep the app on loopback; these local accounts do not complete the production pilot gates.

The refreshed interface uses a navy navigation sidebar, teal actions, consistent white forms and grids, and matching sign-in and account panels. Existing page guidance, column tooltips and financial distinctions remain available.

Local-account verification: **705 tests passed, zero skipped**, including installed LangGraph, Streamlit and Qdrant checks; Ruff passed. The original audit remains USD 80,707.50 candidates and USD 0.00 new realized recovery. Account workflows use temporary stores in the tests; the live first-owner setup is left for the user. Commands, diff, screenshot and remaining limits are in `var/professional-login/VERIFICATION.md`.

## Accountant workspace

The left sidebar uses six icon buttons with an active-page highlight: Overview, Invoice Review, Contract Term Review, Evidence Requests, Ask the Audit Workspace and Human Approval Auditing Team. Each page includes a short business description and section guidance. The standalone document inbox is removed from navigation; synthetic supporting-evidence intake is available within a relevant evidence request.

Each page now includes **How to use this page** guidance with the intended workflow and next action. Every table and calculation matrix has a business description, a displayed-row summary and column definitions available by hovering over the header. Overview and team activity metrics also provide help. Table summaries respect filters and pagination; current case counts, historical records, journal events and financial measures retain their separate meanings. Read-only matrices use the same interactive grid for accessible column help, without adding editing or decision actions.

- **Overview:** blue USD grand totals across all approved records and years, mint portfolio coverage, and a light yellow review-workload section with colored status cards. Portfolio coverage shows **Total approved records** from the same eligible USD basis as Grand Totals, with a current/historical breakdown, plus supplier and contract counts. In the expanded demo this is **2,520 = 520 current + 2,000 historical**; completed historical records remain included. **Current approved for draft** is the separate current-workload count. **View cases** opens a queue's paged table; select a row and **Open selected invoice** to inspect it. Awaiting approval remains a summary count; its case table is available in Invoice Review. Overview hides the Deploy control. Workload counts cover current cases; locked migrated history requires no approval. Completed synthetic historical receipts appear only in realized recovery. The forecast-assumption editor is removed from this page; saved calculations remain unchanged.
- **Invoice Review:** vendor and contract filters, a selectable case table, and readable details and decisions for the selected case. Opening a case or checking one row shows only its individual review. Two or more checked rows show only bulk approval; opening a case exits bulk mode and clears the checkbox selection. Reviewers can approve eligible selected cases together within one contract. Each case still requires a current assessment and validated draft at the human-review pause; missing evidence, stale cases and unfinished preparation block approval. Switching back to one checked row retains its originally displayed review identity. The batch reports each case's outcome and is not an all-or-nothing transaction. Financial totals and technical configuration are removed from the worklist.
- **Contract Term Review:** select a supplier, inspect its contracts and amendments, then open a related case. Existing manually curated approved terms appear immediately. Original source wording remains readable; local extraction controls and technical source metadata are removed from this page. Saved proposed terms still require separate explicit human review.
- **Human Approval Auditing Team:** two fictional escalation managers, ten fictional accountants selectable for future single/bulk invoice decisions, and actual paged journal activity. Existing reviewer labels are preserved. The directory does not invent historical approvals or provide authentication or notifications.
- **Evidence Requests:** one pending-request queue and one active action panel, with side-by-side vendor/contract filters and a collapsed usage guide. Select a request, then use **Next action** to attach an existing document, add new evidence, review and resolve, or reassess. Open contract requests default to attachment; attached evidence defaults to review/resolution, with an inline document reader. Other action forms stay hidden. Resolution, approved terms, reassessment and invoice approval remain separate steps. Resolved requests leave the queue; cases requiring reassessment remain outstanding in Invoice Review. Historical records remain saved locally.
- **Ask the Audit Workspace:** ask for a vendor/POC, contract, current or locked historical case, pending evidence or portfolio totals. Answers combine existing records, fresh Python calculations and complete relevant signed evidence. Result buttons open the corresponding review page; chat never records a decision. Case explanations use the validated factual flow without a model-selection control or new model calls. Broader contract questions return source wording, without claiming general prose entailment or automatic term approval.

Example questions: `Who is the POC for SUP-A?`, `Show contracts for SUP-A`, `Explain case L001`, `Explain contract CTR-A`, `Show pending cases`, and `Show portfolio totals`. Exact IDs and specific recorded vendor/contact names give the clearest results. Ambiguous entities, unsupported status combinations and unverifiable answers request clarification or are withheld. Lists explicitly report a maximum of 50 displayed matches; narrow the vendor, contract or case to inspect more detail. Conversation state is local to the browser session and cleared when its database, tenant or source/checkpoint identity changes.

**Search suggestions:** start typing in any vendor, contract, document or invoice search field to see matching recorded values, or enter your own search text. Suggestions cover the full selected vendor/contract/year, including cases beyond the first table page. Existing Vendor, Contract and accountant selectors also support typing to narrow their choices. The chatbot question field offers supported questions with vendor, fictional POC, contract and invoice context; select a suggestion or write your own question, then choose **Send message**. Selecting a suggestion never approves a case or changes a record.

Vendor points of contact live in the separate `vendor_contacts` table and are explicitly fictional. They do not enter contract/rate inputs, calculation digests, approval state or cash forecasts. The main expanded demo has one contact for each of its 88 current/historical vendors. Contract Term Review also shows the recorded POC. For a newly prepared synthetic profile, seed names explicitly:

```sh
uv run --all-extras --frozen python scripts/seed_vendor_contacts.py \
  --db var/scaled-demo/portfolio.sqlite3 --synthetic-confirmed
```

The seed validates synthetic provenance and vendor scope, writes atomically and preserves existing names on a repeat. Startup and searches never create contacts or alter the data. Vendor contacts do not enable supplier communications.

October 18 chatbot/contact verification: **634 tests passed, zero skipped**; Ruff passed, financial/core tests passed 54/54, retrieval coverage stayed 8/8 and controlled explanation regressions passed 21/21. A real local `llama3.2:latest` chatbot call for L001 passed all 13 required fact checks and retained DOC-A1/DOC-A2. Browser checks confirmed vendor POC lookup, portfolio totals and vendor-to-contract navigation at the unchanged port 8515. The original 20-case audit remains USD 80,707.50 with zero new realized recovery. Contact seeding added 88 fictional names while preserving every pre-existing source/history/forecast/checkpoint row. Exact commands, diff, reconciliation, screenshots and verification limits are in `var/chatbot-implementation/VERIFICATION.md`.

**Expected cash forecast:** Overview reads the separately saved **all-approved** planning assumption and shows its remaining cash estimate without an editing form. The all-year basis combines current eligible approved USD drafts with locked migrated approved USD records. Pending, rejected, evidence-blocked, stale operational cases and other currencies are excluded. The saved assumption is append-only and tied to the exact approved-record basis; a changed basis withholds the projection until it is reviewed and saved again through the forecast service. Zero percent gives USD 0.00, while an absent or stale assumption remains “Not estimated”. Python Decimal calculates the projection.

The main synthetic demo's all-year approved total is **USD 3,500,000.00** across **2,520 approved records**. Its original **80%** scenario modeled **USD 2,800,000.00** in total cash. At the user's request, a separate historical demo receipt history now represents 100% collection of the **already discounted** 2024/2025 forecast: USD 1,000,000.00 approved per year × 80% = **USD 800,000.00 per year**, totaling **USD 1,600,000.00** in invented historical receipts. No second 80% haircut is applied.

Overview therefore shows **USD 1,200,000.00 estimated cash still to recover** (80% of USD 1,500,000.00 current approved drafts) and **USD 1,600,000.00 realized recovery in the synthetic history**. Realized plus remaining estimated equals the original USD 2,800,000.00 expectation; the approved value is a separate measure, not an additional cash total. The page explicitly labels the receipts as invented demo history. No actual received cash is asserted, and operational/current-year and original baseline realized recovery remain **USD 0.00**.

`historical_recovery.py` records the explicitly authorized historical scenario in separate immutable synthetic-only tables. It freezes the saved forecast/rate, archive/source identities and amounts, and year-end simulated receipt dates. Historical rows are not reapproved, reassessed or converted into current claims. Closed historical records leave the remaining forecast entirely; later percentage changes never revalue their recorded receipts. Missing or stale forecast assumptions withhold the remaining estimate. Actual collection and settlement processing remain separate future work.

The earlier current-only forecast remains in its original immutable history: 80% of USD 1,500,000.00 across 520 current approved cases was USD 1,200,000.00. The all-approved scope has a separate policy, basis digest and assumption history; expanding Overview does not silently apply that earlier saved assumption to different records. Operational calculation and current-only helpers retain their original scope and outcomes.

Selecting rows, filtering and changing pages do not call a model or record decisions. Review preparation and provider consent are explicit. Bulk approval uses each case's existing guarded workflow and journal identity; it never changes calculated amounts or records realized recovery.

October 18 earlier accountant-workspace verification: **443 tests passed, zero skipped**; Ruff passed and retrieval coverage remained 8/8 without model calls. Read-only before/after reconciliation preserved both the original USD 80,707.50 baseline and the larger demo's 20 existing USD 1,000,000.00 approved cases, including every source/journal/checkpoint row. The app was restarted at the same loopback address and the four page layouts were checked in Chrome. Bulk decisions and interrupted-save recovery were exercised in disposable synthetic databases. Commands, diff, screenshots and verification limits are in `var/ui-upgrade/VERIFICATION.md`.

October 18 approval-team/forecast verification: **528 tests passed, zero skipped**; Ruff passed, dependency-free financial/core tests passed 54/54 and retrieval coverage remained 8/8. The selected 80% assumption is the only added live financial-planning record: every pre-existing scaled-profile source/history/journal table and checkpoint fingerprint is unchanged. Original baseline and million-dollar profiles remain unchanged. Commands, diff, document snapshots and layout checks are in `var/team-forecast-update/VERIFICATION.md`.

October 18 grand-total Overview verification: **544 tests passed, zero skipped**; Ruff passed, core financial tests passed 54/54 and retrieval coverage stayed 8/8. The browser shows only the three all-year USD metrics. Existing source/approval/checkpoint records and the earlier forecast history are preserved; the all-year assumption is saved separately. See `var/grand-total-overview/VERIFICATION.md` for exact commands, diff and checks.

October 18 historical-receipt reporting verification: **563 tests passed, zero skipped**; Ruff passed, core financial tests passed 54/54 and retrieval coverage stayed 8/8. Nineteen new regressions cover historical receipt safeguards and the Overview display. The browser shows USD 3,500,000.00 approved, USD 1,200,000.00 remaining estimated cash and USD 1,600,000.00 in explicitly invented historical receipts. Every pre-existing source/archive/approval/forecast/checkpoint table is preserved. Exact commands, scoped diff, screenshot and reconciliation are in `var/historical-recovery/VERIFICATION.md`.

October 18 Overview restoration: **564 tests passed, zero skipped**; Ruff passed. Portfolio coverage and colored current review queues are restored, with paged selectable case tables and direct invoice links. The assumption editor is removed from Overview. All live financial, source, receipt, forecast and checkpoint rows are unchanged. Commands, diff and browser screenshots are in `var/overview-restoration/VERIFICATION.md`.

## Route A — run immediately with an existing Python

Use Python 3.11–3.13 with SQLite FTS5. Run all commands from the extracted project root.

```sh
python3 --version
PYTHONPATH=src python3 -m recovery init
PYTHONPATH=src python3 -m recovery audit --output var/audit.json
PYTHONPATH=src python3 -m recovery explain --line L001
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/evaluate_retrieval.py
```

Expected initial results:

| Outcome | Count |
|---|---:|
| CANDIDATE | 4 |
| NO_DISCREPANCY | 4 |
| ALREADY_OFFSET | 2 |
| NEEDS_REVIEW | 9 |
| UNPAID | 1 |

**Candidate total: USD 80,707.50. New realized recoveries: USD 0.00.**

L001 contributes 80,000.00; L004 contributes 600.00; L011 contributes 100.00; L017 contributes 7.50. These are invented test values.

To save the fixture import report:

```sh
PYTHONPATH=src python3 -m recovery init --report var/import-report.json
```

The importer uses explicit manifest and CSV schemas. Successful reports include source/import row counts, exact invoice/payment/adjustment totals separated by tenant, currency and status, and row-digest/search-index comparisons. Rejections return exit code 2 and JSON errors on stderr, with file, row, field and error code; `--report` also saves that rejection. Failed validation or reconciliation preserves the prior snapshot and review journal. Reports must use a separate path outside the source directory and database.

See `docs/DATA_DICTIONARY.md` for accepted fields and numeric bounds. Format validation preserves evidence gaps such as partial payments, conflicting rates and pending credits for the financial engine to assess. Fixture reconciliation does not independently establish source completeness or validate real signatures.

## Route B — Mac/VS Code development environment

Install uv from its official instructions. Review the downloaded installer before executing:

```sh
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
less /tmp/uv-install.sh
sh /tmp/uv-install.sh
```

Restart your terminal, navigate to this folder, then:

```sh
uv --version
uv python install 3.12
uv sync --all-extras
cp .env.example .env
uv run python -m unittest discover -s tests -v
uv run recover init
uv run recover audit --output var/audit.json
uv run python scripts/evaluate_retrieval.py
uv run streamlit run app.py --server.address 127.0.0.1
```

Open the local address printed by Streamlit (configured as http://127.0.0.1:8515/). Stop it with Control-C. The UI is not a secure public application. An internet connection is needed to install packages; baseline operation after installation does not need one. This generic setup selects the baseline database; the main demo launcher above selects the larger approved scenario.

`uv sync --all-extras` creates `.venv` and resolves dependencies. The included lockfile was resolved and tested locally; use `uv sync --frozen --all-extras` to reproduce that resolution. The optional graph test should run instead of skip after graph installation; investigate failures before considering that integration verified.

Use VS Code **File → Open Folder** to open this project. Install the official **Codex** extension published by OpenAI, sign in using an available supported method, and start with Prompt 1 in `docs/CODEX_PLAYBOOK.md`. Keep permissions limited to this workspace, review shell commands/diffs, and do not grant broad account access.

## Optional local vector retrieval and chunking

The default retrieval still needs no model or vector service. The optional `vector` extra installs Qdrant's embedded Python client; `--all-extras` includes it. Embeddings use an already installed **embedding** model, separately from the chat model used for explanations. The local check used `embeddinggemma:latest` for 768-dimensional embeddings and `llama3.2:latest` for explanations.

Create a separate synthetic pilot database and build its index explicitly:

```sh
uv sync --all-extras
ollama pull embeddinggemma:latest
uv run recover --db var/vector-pilot.sqlite3 init
uv run recover --db var/vector-pilot.sqlite3 migrate
uv run recover --db var/vector-pilot.sqlite3 index-evidence \
  --index var/vector-pilot-index --embedding-model embeddinggemma:latest \
  --allow-local-embedding
uv run recover --db var/vector-pilot.sqlite3 search-evidence \
  --index var/vector-pilot-index --line L001 \
  --query 'Which signed amendment changed the July engineer rate?' \
  --limit 1 --allow-local-embedding
uv run recover --db var/vector-pilot.sqlite3 explain --line L001 \
  --retrieval-index var/vector-pilot-index --allow-local-embedding \
  --llm --provider ollama --model llama3.2:latest
RECOVERY_DB=var/vector-pilot.sqlite3 uv run --all-extras streamlit run app.py \
  --server.address 127.0.0.1
```

Do not reinitialize an existing migrated database. Each index is tied to its synthetic source snapshot and tenant. In **Invoice review → Explanation settings**, opt into local embeddings, enter that index path and separately allow the explanation provider. Starting the workflow retrieves evidence, validates the draft and pauses for your decision. A normal reload restores its saved state without embedding or generating again.

Chunks are exact overlapping source slices, defaulting to 1,200 characters with 150-character overlap and paragraph/line boundary preferences. Each retains source IDs, hashes, offsets and line ranges. Index generations pin the embedding model digest, dimensions and chunking settings; failed builds do not replace the published generation. Changed evidence or a rebuilt index requires a fresh review. Index building and search require explicit local-embedding consent; there is no automatic download or cloud fallback in the adapter.

Semantic hits rank documents, while the **complete relevant signed contract and amendment set** still comes from canonical SQLite scope/date records. Even `--limit 1` retains all relevant documents. Vector similarity never selects financial rates, changes an amount, approves a case or establishes source completeness. Full evidence still must fit the explanation context; chunking does not silently discard amendments. Embedded Qdrant is a single-process local prototype, not a shared production database. No held-out retrieval-quality or production-scale benchmark is claimed.

## Optional local Ollama explanations

The default explanation remains an offline template. Ollama is an explicit, local explanation provider; it cannot calculate amounts, approve cases, or promote contract terms. Install the `ollama` extra (included in `--all-extras`), start your local Ollama server, and select an already installed local model:

```sh
uv sync --extra ollama
ollama list
uv run recover explain --line L001 --llm --provider ollama --model llama3.2:latest
uv run recover graph-start --line L001 --thread ollama-L001-1 \
  --llm --provider ollama --model llama3.2:latest
```

The endpoint defaults to `http://127.0.0.1:11434`; `--ollama-url` accepts only loopback HTTP endpoints. The adapter uses no proxies, follows no redirects, downloads no models, and has no cloud fallback. The model must have local weights; cloud model aliases are rejected. For defense in depth, start the Ollama server with `OLLAMA_NO_CLOUD=1` and a loopback `OLLAMA_HOST`, as described in the [Ollama FAQ](https://docs.ollama.com/faq). The application does not inspect or change your server configuration.

The complete scoped evidence, source hashes and computed case are sent together. A conservative budget covers the full prompt, schema and reserved answer within `--context-tokens` (default 16384); oversized evidence is rejected without truncation. Provider errors and invalid drafts are withheld; the calculated case remains available through `case` and offline `explain`. Model identity/digest, prompt version and local settings are saved with graph checkpoints. A changed model digest or explanation prompt version requires a new thread and review.

Explanations now use a **closed fact contract**. Ollama/OpenAI return fact IDs and exact values, an allowed review action, and governing quotations. Every required fact must occur exactly once: status, reason, currency, financial strings, nulls, warnings and each service-date component's rate. Unknown facts are `unsupported`; changed values are `contradicted`. Both withhold the draft. Canonical reviewed quotes must be retained, with source hashes and case/source digests. Python renders the visible summary, action and limitations from verified facts; providers cannot supply arbitrary summary or limitation prose. The output discloses `composition: validated_fact_template` and separate claim, financial-consistency and citation checks.

This addresses the observed L011 false "no amendment/discrepancy" narrative by constraining what can be displayed. It does not establish general natural-language entailment, independent source correctness or real-data accuracy. Streamlit uses the same checks and requires explicit provider consent. Human review remains required; rejected model output never changes the financial assessment or records an approval.

Run the reproducible offline explanation evaluation, then optionally exercise the installed local model:

```sh
uv run python scripts/evaluate_explanations.py
uv run --all-extras python scripts/evaluate_explanations.py --run-local --model llama3.2:latest \
  --output var/faithfulness-implementation/live-evaluation.json
```

The new test-only corpus contains 21 controlled regressions, including L011, financial mutations, unresolved evidence and false claims with genuine quotations. Labels are developer-authored and **pending independent SME review**. These visible regression fixtures are not a held-out benchmark; general semantic faithfulness is not scored. Default evaluation makes zero model calls. Optional local generation covers L001, L011 and L006 in a disposable synthetic database, separately reports draft acceptance and checks the unchanged audit and zero realized recovery.

Run the separate opt-in synthetic smoke check after installing a model:

```sh
uv run python scripts/evaluate_ollama.py --run-local --model llama3.2:latest
```

It generates explanations for L001 and L011 using a temporary synthetic database, checks the unchanged 20-case audit and zero realized recovery, and writes `var/ollama-implementation/live-smoke.json`. It does not record approvals or use real supplier files. Ordinary unit tests mock providers and make no generation calls.

## Optional OpenAI RAG and term proposals

The `ai` extra is included by `--all-extras`. Edit `.env` locally:

```dotenv
OPENAI_API_KEY=YOUR_LOCAL_API_KEY
OPENAI_MODEL=YOUR_ENABLED_STRUCTURED_OUTPUT_MODEL_ID
LANGSMITH_TRACING=false
```

The model value is intentionally not a guessed or hardcoded model name. Choose an API model enabled in your account that supports the structured-output integration. The starter defaults to no provider call.

```sh
uv run recover explain --line L001 --llm --provider openai
uv run recover propose-terms --doc DOC-A2 --output var/proposals/DOC-A2.json
```

The second command produces an **UNREVIEWED_PROPOSAL**, never a replacement approved rate. A reviewer must check the source and resolve applicability before any future promotion workflow. The `.env` file must stay local. The CLI database location is selected through `--db` or a shell-exported `RECOVERY_DB`; `.env` loading is currently used only by the optional AI provider adapter.

API billing is separate from ChatGPT subscription billing. Sending the case to a provider is a data-egress action. Only synthetic fixtures are approved for this starter. Keep LangSmith tracing off until trace content and retention are reviewed.

## One-screen human-review workflow

Install the graph, UI and explanation extras with `uv sync --all-extras`, initialize the synthetic snapshot, then open the local app:

```sh
uv run --all-extras streamlit run app.py --server.address 127.0.0.1
```

1. Select an invoice in **Inspect an invoice line** (start with **L001** for the initial example).
2. Select the installed Ollama model and allow this synthetic case to be sent locally.
3. Click **Start review workflow**. Python calculates the assessment, retrieves the complete signed evidence and validates the explanation before the workflow pauses.
4. Review the case, source quotations and saved explanation. Enter a reviewer label and evidence-review note.
5. Choose **Approve for draft**, **Reject**, or **Request more evidence**. The same saved workflow resumes and appends the decision to the local journal.

Invoice review opens in **Needs review** scope. Its table and **Inspect an invoice line** selector show outstanding preparation, approval, evidence and reassessment work. Approved and rejected cases leave this default list; cases with no outstanding review work are excluded too. In this scope, approval or rejection confirms the saved decision and selects the next pending candidate, without generating another explanation. Request-more cases remain outstanding evidence/reassessment work. Reassessment or a newer review returns a line to the queue even if its amount and case digest are unchanged. A committed decision with interrupted checkpoint completion remains available solely to finish the saved workflow and takes priority when reopening the page.

Use **Review next candidate** to move to another candidate awaiting a decision (for example, L001 → L004). Navigation follows the displayed filters and never generates a draft or records a decision. Give consent for the newly selected case, then click **Start review workflow**. When no invoices need review, the page shows an empty queue. Select **All cases** to inspect completed or other invoice lines; Overview's **Open selected invoice** opens this scope automatically. To intentionally repeat a completed invoice, select **Start a fresh review of L001** (or the displayed line ID) and give consent before starting.

Reloading or reopening the app restores the saved draft without regenerating it. The checkpoint file sits beside the selected database as `<database-stem>-ui-checkpoints.sqlite3`; the review journal remains in that database. A changed case or explanation contract disables the old decision controls and requires a fresh review. Invalid drafts never reach the decision form.

The review page presents a plain-language finding, a labeled calculation with currency-formatted amounts, service-date rates, exact supporting quotations and next steps. Unknown amounts remain **Not established**. Saved explanations use their saved assessment even if the current evidence has changed. Raw assessment and explanation JSON, including hashes and validation reports, is available in collapsed **Technical … details (JSON)** sections.

Invoice review also shows the current **Approved for draft** count and USD approved value. Its invoice table separates **Financial assessment**, **Review status** and **Current decision**: an approved discrepancy remains `CANDIDATE` in the financial assessment while its decision displays `APPROVED_FOR_DRAFT`. A newer assessment or review attempt removes the old decision from the current display until it is reviewed again; the historical journal remains available.

If a decision commits before checkpoint completion is interrupted, the app shows the recorded outcome and suppresses alternate choices. **Finish workflow completion** replays only that saved decision with the same stable journal key. It cannot create a second journal event or substitute a different choice.

Approval records `APPROVED_FOR_DRAFT`; rejection records `REJECTED`; requesting evidence records `NEEDS_MORE_EVIDENCE`. Requesting evidence ends this review. With Step 9 enabled, it also creates one structured evidence request in the same transaction. Use the new evidence views and a new linked review after reassessment. The app performs no supplier action and keeps realized recovery at zero. Reviewer labels are demo labels, not authenticated identities.

## Step 9 — reviewed terms, evidence and recovery overview

The running local sample at http://127.0.0.1:8512 has been backed up and migrated without changing its existing source rows, audit case digests or review decisions. Select a section using **Page** in the sidebar.

For another **existing initialized synthetic database**, apply the additive migration once:

```sh
uv run recover --db var/recovery.sqlite3 migrate
```

Migration uses SQLite's backup API, verifies integrity and journal preservation, and creates append-only histories. Backups default to a `backups/` directory beside that database; `migrate --backup-dir PATH` selects another destination. Repeating migration is a no-op. **Do not run `init` over a migrated database.** It now refuses replacement to preserve additive evidence, approved terms and history. Initialize a separate new database when you need another fixture dataset. The working database's migration and checkpoint backups are recorded in `var/step9-implementation/working-migration.json`.

- **Overview:** two sections, **USD PORTFOLIO VALUE** and **REVIEW WORKLOAD**. The USD metrics show calculated candidates, their approved-for-draft subset, expected cash recovery (**Not estimated**) and new realized recovery (zero). Workload counts cover approved drafts, rejections, evidence/review gaps, awaiting approval, preparation, reassessment and other outcomes. Each line counts once against its current assessment and review attempt. **View cases** opens that queue's case list with each amount's currency; a selected line opens Invoice review. **Close cases** restores the compact view. Other currencies are excluded from the USD headline values.
- **Contract term review:** select an existing source, explicitly consent to local Ollama, and generate an `UNREVIEWED_PROPOSAL`. The original model output and source identity remain immutable. Save corrections as a new revision with a note, then review the saved revision's source, exact quotation, dates, scope, questions and coverage. Approve or reject terms explicitly. Approval affects the engine only through the human-review service; it does not approve an invoice case. Reapproving an identical curated fixture term references its existing version and preserves its original attribution.
- **Document inbox:** stage synthetic UTF-8 `.txt`/`.md` content, at most 262,144 bytes. Bind it to an existing supplier/contract/role/location scope. Signed simulation requires a separate explicit attestation; this does not verify real signatures. Sources are stored as immutable database versions with generated projection paths. The original filename is metadata, never a write path. Signed dates after the snapshot's approved as-of date are refused. New potentially governing evidence invalidates affected findings before extraction. Unsigned drafts never control rates. Rejecting an extraction leaves signed evidence unresolved; a separate reviewed no-pricing-effect/out-of-scope disposition needs a rationale.
- **Evidence requests and history:** new request-more decisions capture a missing-evidence type and create one linked request atomically. Existing historical decisions offer **Create evidence request** explicitly; migration does not invent missing details. Attach an admitted source, review its terms, then resolve the contract/rate request with a reviewer disposition. Attachment alone does not resolve it. Payment, credit, service and other requests are tracked but cannot be resolved through this contract-only handling path.
- **Resolve through the UI:** select the request, attach the relevant evidence using the attachment form, review any unapproved contract terms, then fill the separate resolution form's reviewer label and rationale and click **Resolve reviewed contract-evidence request**. Confirm its status is **Resolved by a reviewer**. Resolved requests remain visible in history. Selecting a request also selects its invoice and prior evidence decision for reassessment below; verify the invoice, click **Reassess with Current Evidence**, then start a fresh linked review if eligible. Resolving one request does not close other requests or fix other evidence gaps.
- **Reassessment:** click **Reassess with Current Evidence**, compare historical and current results, consent to generation and start a **new linked review**. Open Invoice review for the saved explanation and decision. Earlier approvals never transfer to a new assessment, including a same-amount revision. Only the latest valid assessment contributes current value; stale history is displayed separately. A failed preparation can be retried explicitly using its pinned model configuration; an ordinary reload does not regenerate it.

Only one regular hourly USD rate and one exact role/location scope are supported per proposal. Incomplete coverage, unresolved applicability, unsupported dates/rates/scope, unsigned sources and invalid supersession block promotion. Human coverage attestation is required; model confidence and quote membership are not independent proof of contractual applicability. This remains synthetic-only, localhost-only and unauthenticated.

Run the opt-in local extraction smoke separately from reproducible mocked tests:

```sh
uv run --all-extras python scripts/evaluate_term_extraction.py --run-local --model llama3.2:latest
```

The verified local run generated one unreviewed proposal and left terms, audit and decisions unchanged. Its rate was correct, but its unsupported end date and incomplete coverage require human correction. This is transport/schema/persistence verification, not a claim of general extraction accuracy. The report is `var/step9-implementation/live-term-smoke.json`.

A separate **hypothetical resolved-evidence demo** runs at http://127.0.0.1:8513. Its clearly labeled invented correction records yield zero `NEEDS_REVIEW` cases and USD 85,307.50 candidates, with zero realized recovery. It preserves the original database/checkpoints at port 8512 and the original USD 80,707.50 fixture baseline. Earlier decisions are retained as history and do not approve the new assessments. The scenario includes a corrected August service-date assumption, replacement USD/REGULAR invoices and fictional payment/service/credit records; it does not resolve the retained September contract conflict, implement FX/overtime support or independently verify the original evidence. Exact changes and hashes are in `var/demo-evidence-resolution/correction-register.json`; results and commands are in `var/demo-evidence-resolution/VERIFICATION.md`. The app displays its dataset ID and description on every page to distinguish the scenarios.

The earlier **million-dollar synthetic demo**, still selectable with `--profile million`, adds ten invented, fully paid aggregate-team-hour invoice lines under the same approved contract terms. It contains 30 invoice lines: 20 eligible candidates explicitly approved for draft at the user's instruction, totaling exactly **USD 1,000,000.00**, and ten retained noncandidate controls. The quantities are chosen to demonstrate that target; this is not measured savings, independent SME review or cash recovery. New realized recovery remains **USD 0.00**. Original fixtures, the prior demo and its history remain unchanged; copied decisions remain historical until fresh guarded reviews approve the expanded snapshot. Generation used deterministic offline explanations for these user-directed approvals; selecting the matching vector index does not change their existing provenance or add approvals. Input hashes, import reconciliation, approvals and replay verification are retained in `var/million-dollar-demo/`. SQLite remains canonical for financial assessment and complete signed evidence; optional explicitly consented embeddings/Qdrant search use this dataset's dedicated `vector-index`, independently of the baseline indexes.

The earlier port 8514 address is a historical optional launch, not the main application address. If needed for comparison, it can still be selected explicitly:

```sh
RECOVERY_DB=var/million-dollar-demo/million.sqlite3 uv run --all-extras --frozen streamlit run app.py --server.address 127.0.0.1 --server.port 8514 --server.headless true --browser.gatherUsageStats false
```

## Optional CLI checkpointed approval workflow

No model key is needed for the offline graph workflow:

```sh
uv run recover graph-start --line L001 --thread demo-L001-1
```

Read the case and draft in the output. The workflow should pause. A separate command reconstructs the graph and loads the checkpoint:

```sh
uv run recover graph-resume --thread demo-L001-1 \
  --decision approve --reviewer Vishala \
  --note "Reviewed the synthetic amendment, service records, settlement and offsets."
```

Expected local outcome: `APPROVED_FOR_DRAFT`. It does not send anything or record money recovered. Other decisions are `reject` and `request_more`. Use the same thread to resume; use a new thread for a new investigation. A changed source/case digest blocks a stale approval. Completed threads are not reusable.

## Project map

```text
app.py                         Optional Streamlit interface
AGENTS.md                      Persistent Codex project rules
pyproject.toml                 Python/package configuration and optional extras
.env.example                   Empty local configuration template
src/recovery/storage.py        Atomic source import, metadata, scoped queries
src/recovery/engine.py         Financial checks and rate precedence
src/recovery/retrieval.py      FTS5 ranking, complete scoped evidence, citation checks
src/recovery/ai.py             Explanation routing/schema and OpenAI term proposals
src/recovery/explanation_facts.py Closed claim support, financial consistency and rendering
src/recovery/presentation.py   Plain-language review views and exact amount formatting
src/recovery/lifecycle_storage.py Additive migration, backups, input versions and immutable records
src/recovery/lifecycle.py      Transactional human promotion, intake, requests and reassessment
src/recovery/lifecycle_ui.py   Overview, term review, document inbox and evidence history views
src/recovery/term_extraction.py Local Ollama unreviewed term extraction
src/recovery/dashboard.py      Read-only case counts and currency-separated financial summaries
src/recovery/ollama_provider.py Loopback-only local explanation transport
src/recovery/review.py         Local append-only/idempotent review journal
src/recovery/workflow.py       Optional persistent LangGraph approval flow
src/recovery/cli.py            Commands
src/recovery/__main__.py       No-install python -m recovery entry point
data/                         Synthetic source data and test-only ground truth
tests/                        Core tests and optional workflow integration test
scripts/evaluate_retrieval.py  Document-coverage smoke test
scripts/evaluate_ollama.py     Opt-in synthetic local generation smoke test
scripts/evaluate_explanations.py Offline claim regressions and opt-in local evaluation
scripts/evaluate_term_extraction.py Opt-in local unreviewed proposal smoke test
examples/                     Actual creation-environment outputs
docs/                         Detailed implementation, Codex prompts and schema
var/                          Generated local DBs/output; ignored by Git
```

## Continue the build

Read `docs/IMPLEMENTATION_GUIDE.md` for phased engineering tasks, `docs/DATA_DICTIONARY.md` for source semantics, and `docs/CODEX_PLAYBOOK.md` for copy-paste Codex prompts. Start by reproducing the baseline result; do not start with unrestricted agents or real accounting-system access.
