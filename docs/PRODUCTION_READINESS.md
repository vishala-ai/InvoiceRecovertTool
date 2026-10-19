# Production readiness and private-pilot architecture

This repository started as a local, synthetic-data-only review application. The user has now explicitly selected a Streamlit Community Cloud synthetic pilot with Groq and persistent Supabase storage; see [the separate Cloud setup and limits](COMMUNITY_CLOUD.md). The semantic-search foundation adds source-preserving chunks, explicitly selected local embeddings and an optional local Qdrant index. Neither foundation makes the application a production service or authorizes real-data ingestion or other external business integrations.

The selected milestone is a **local/private synthetic pilot with a simulated invoice workflow**, matching the earlier fake-data demonstrations. No external applications are needed or connected for this milestone. Production application integrations and real-data deployment are deferred; they are not prerequisites for completing the local demonstration.

The financial question remains narrow: identify supported discrepancies on fully paid regular hourly-service invoice lines after existing applied pricing credits and received refunds. A candidate, an approved draft and realized recovery remain separate measures. New realized recovery remains zero.

## Selected simulated invoice workflow

**Synthetic invoices → canonical import → Decimal assessment → scoped vector ranking with full evidence → local explanation → human review → local journal.**

1. **Synthetic inputs:** Use invented invoice lines, contracts/amendments, manually reviewed terms, dated service records, payments, POs and pricing adjustments. Existing simulated credits/refunds are offsets, not new recoveries produced by the application.
2. **Canonical import:** Validate the synthetic manifest, schemas, source references, exact quotes and hashes; reconcile the snapshot before committing it to local SQLite. Preserve local version/review history. Use a separate database for a new demo snapshot rather than replacing a protected lifecycle database.
3. **Decimal assessment:** Compute the supported status and candidate amount from the canonical inputs with string-parsed Decimal arithmetic. Apply service-date supersession and settlement/offset checks. Incomplete or conflicting evidence remains `NEEDS_REVIEW`.
4. **Scoped semantic ranking:** On explicit local-embedding consent, use the pinned local Ollama embedding model and validated local Qdrant generation to rank passages. Independently retain the complete relevant signed-document set from SQLite. Similarity scores cannot establish the applicable rate or omit an amendment.
5. **Local explanation:** Use the deterministic offline brief or the explicitly selected local Ollama explanation path. The model cannot change financial facts, reviewed terms or approval state. Invalid output or evidence that exceeds context bounds withholds the optional draft without changing the engine's assessment.
6. **Human review:** Present the case, complete evidence and available explanation to a person who selects approve for draft, reject or request more evidence. The reviewer label is a demo input, not an authenticated identity. Changed case/index identities require a fresh review.
7. **Local journal:** Append the decision with the case/assessment identity and a stable idempotency key; checkpoint reconstruction or replay must not duplicate it. `APPROVED_FOR_DRAFT` is a local review outcome. No email, submission, ERP write, payment or new realized recovery follows.

## What exists and what remains planned

| Area | Local implementation or foundation | Required before a real-data private pilot |
| --- | --- | --- |
| Financial assessment | SQLite source records, string-parsed Decimal calculations, explicit term supersession, settlement/offset checks and evidence abstentions | Approved source-system mappings and independently reconciled exports; an agreed settlement model for any additional accounting cases |
| Contract authority | Manually curated reviewed fixture terms; local versioned evidence and review history | Authorized contract sources, authenticated reviewers, controlled promotion of reviewed terms and measured extraction quality |
| Provenance | Source IDs, original quotes, hashes, dates, case digests and stable journal event identities | Protected audit retention, access controls, backups and tested recovery; local database history alone is not a tamper-proof audit service |
| Chunking | Deterministic overlapping character windows preserving exact source slices and offsets; paragraph and line boundaries are preferred | Evaluated clause/table segmentation for each accepted document format; PDF parsing, OCR and signature validation are not established |
| Embeddings | Explicit local Ollama embedding-model selection, pinned digest, bounded requests, no silent truncation and validated vectors | Approved model/version and operating environment, measured retrieval quality, controlled upgrades and rebuild procedures |
| Vector index | Optional Qdrant local index with generation/provenance validation and scoped semantic ranking | Private server deployment, authenticated access, lifecycle controls, capacity/concurrency testing and recovery procedures |
| Evidence completeness | The full relevant signed-document set comes from canonical scoped records independently of vector top-k results | Authorized source discovery and completeness reconciliation; a complete imported snapshot does not prove that no external amendment exists |
| Model explanations | Computed case facts and approval state remain outside model authority; invalid optional output is withheld | Held-out reviewer-labeled evaluation and an agreed release gate for the supported explanation contract |
| Access | Tenant selection is a demo scope filter; reviewer labels are local demonstration inputs | Verified sessions, server-derived tenant membership/roles and authorization across SQL, objects and vectors |
| Integrations | Synthetic fixtures and local commands; external applications are not needed or connected for the selected milestone | Deferred: explicitly approved read-only adapters and credentials for any future selected applications |
| Operations | Local execution and repository checks | Deployment ownership, CI, monitoring, incident response, secrets management, retention/deletion and restore drills |

See the README for installed extras, available commands and actual verification results. This document does not assert that a dependency, embedding model, live provider or deployment has been verified merely because its adapter exists.

## Retrieval architecture and invariants

SQLite remains the canonical source for financial inputs, manually reviewed terms, document identities and review history. The Decimal engine assesses the case without consulting vector similarity or model arithmetic. Existing approved amounts and approval events are not updated by embeddings or model output.

The optional semantic path is:

1. Select authorized canonical source versions and preserve their exact bodies and hashes.
2. Produce deterministic overlapping chunks with document IDs, tenant/scope metadata, source hash, chunker version/settings, character offsets and line ranges.
3. Embed complete chunk text with an explicitly installed local embedding model. Pin the model name and digest. Use the same model identity and preprocessing for index and query vectors.
4. Build and validate an index generation. Reconcile its expected chunk identities, counts and provenance before making the generation usable.
5. Apply exact scope and generation filters to semantic queries. Revalidate returned chunk identities and source hashes against canonical records.
6. Use semantic hits to rank useful passages. Independently retrieve the complete relevant signed-document set for the case's exact scope and as-of date.
7. Assess finances deterministically and construct any optional explanation from the authoritative case and complete relevant evidence.

Current document scope includes tenant, supplier, contract, role and location, plus signed status, document kind and as-of constraints. Additional filter fields require validated source metadata; filtering a field that was never populated can silently exclude every relevant point. Scope filters are defense in depth and do not replace authenticated authorization.

The chunker prefers paragraph/line boundaries but may split a long clause at its character limit. Character offsets refer to the original Unicode string, not UTF-8 byte positions. Overlap helps retain nearby context; it does not establish legal clause completeness. Preserve the original body for citation checks and reconstruct neighboring/full-source context when needed.

An index generation must identify its source snapshot or scoped source fingerprint, source-version hashes, embedding model/digest, vector dimension/distance and chunker/preprocessing version. A source change, deletion, model change or chunking change must invalidate affected indexed evidence until a validated generation is available. A partial rebuild must not appear current. Idempotent chunk IDs and explicit reconciliation are required for replay and retry.

Ollama's `/api/embed` accepts batches and defaults to truncating overlong inputs; this adapter must use `truncate=false` and reject inputs that do not fit. Embedding coordinates are non-financial floats and never enter money/rate/hour calculations. [Ollama embedding API](https://docs.ollama.com/api/embed) and [embedding guidance](https://docs.ollama.com/capabilities/embeddings).

Chunking and vector ranking do not remove the explanation context limit. If the complete relevant signed evidence cannot fit the configured provider/context bounds, withhold the optional draft. Do not silently drop amendments or substitute top-k chunks as proof of completeness. The financial assessment retains its own supported result or evidence-review outcome.

## Current milestone and future deployment gates

### 1. Selected now: validate the local synthetic foundation

Keep synthetic-only import enforcement. Verify deterministic chunk provenance, embedding identity, index generation reconciliation, cross-tenant filtering, stale-index rejection and unchanged financial outcomes. Run provider checks only when explicitly requested and record the actual model/digest and environment.

Qdrant's Python local mode is suitable for development, prototypes and tests. Its persistent directory uses an exclusive lock; it is not a shared multi-process production database. Moving to a server changes concurrency, security and operating responsibilities and needs separate validation. [Official local-mode guidance](https://github.com/qdrant/qdrant-client#local-mode) and [local implementation](https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/local/qdrant_local.py).

### 2. Future, deferred: establish the private service boundary

Before any real records enter the system, implement:

- An identity provider with verified server-side sessions, tenant membership and reviewer/administrator roles. Enforce permissions on every case, source object, vector query and mutation; a browser-supplied tenant ID is not authority.
- Private networking and TLS for application, model, database and object-store connections. Separate query-only vector credentials from ingestion credentials; keep service/admin credentials out of browsers and model contexts.
- A migration plan for a transactional PostgreSQL ledger and workflow/checkpoint storage, with concurrency controls and durable idempotency. Preserve source versions, exact Decimal strings, case digests and stale-review checks.
- Immutable/versioned source objects with authorization, encryption, access logs and controlled retention. Use a secrets manager, approved logging/tracing and explicit data-egress policy.
- Tested database/object/index backup and restore, including reconstruction of a validated vector generation from canonical sources.

A self-hosted Qdrant server requires explicit authentication, network binding and TLS configuration. Merely starting the server does not establish a secure service. [Qdrant security](https://qdrant.tech/documentation/security/) and [filtering](https://qdrant.tech/documentation/search/filtering/).

### 3. Future, deferred: add approved read-only application adapters

The following application boundaries describe future real-data requirements. They are intentionally outside the selected fake invoice workflow; no account connection or application-specific integration is currently required.

| Application boundary | Required data and controls | Excluded authority |
| --- | --- | --- |
| ERP/AP or approved exports | Invoice lines, service quantities/approvals, payments, pricing credits/refunds and POs; explicit source/business IDs, currencies, dates, allocation/status semantics, extraction timestamps and source-total reconciliation | No ERP writes, payment instructions, automated settlement allocation or automatic claim submission |
| Contract repository | Immutable document/version IDs, original files, hashes, signing/effective-date metadata, source ACLs and amendment discovery; quarantine and format validation before parsing | A document's text cannot instruct tools; extraction proposals cannot approve themselves or replace reviewed terms |
| Identity provider | Verified identity/session, authoritative tenant/role membership, revocation and audit context | User-entered reviewer names and tenant selectors cannot grant permissions |
| Workflow/ticket application, if later selected | A reviewed case/evidence reference and explicitly authorized workflow actions with scoped credentials and idempotency | No automatic ticket creation, supplier email or external submission is authorized by the current foundation |

The synthetic-only guard remains in place until the documented pilot gates are completed. Do not label a real source as synthetic to reuse the fixture importer. Design and approve the real adapter's mapping, rejected-record report and reconciliation before connecting it.

For document extraction, show the original source and proposal side by side. An authenticated reviewer must verify applicability, rates, units, effective dates, supersession and source version before promotion. Preserve unresolved evidence as review work. PDF/OCR capability and real-signature validity require their own design and evaluation.

### 4. Future, deferred: operate and evaluate a limited real-data private pilot

Assign operational owners and define capacity, latency and recovery objectives before release. Use bounded background jobs, retry/deadline policies, idempotency, failure queues and explicit index rebuild states. Replaying a job must not duplicate review events or financial effects. Keep audit/metric logs useful without recording secrets or uncontrolled document text.

Test authorization and revocation, malformed inputs, document instructions, source/model drift, interrupted embedding batches, unavailable providers, stale checkpoints, concurrent access and backup restoration. Define deletion across canonical objects, SQL projections, vector chunks and caches according to the approved retention policy.

Apart from the separately authorized synthetic Community Cloud pilot, no other public deployment, supplier communication, payment activity, claim submission or ERP write is included in these phases. Any future business-system write capability needs a separate approved design, scoped authority, human approval, durable outbox and reconciliation. Realized-recovery accounting also remains a separate settlement feature.

## Acceptance evidence

Keep the existing baseline and expanded demo separate. Do not alter their expected values to accommodate a retrieval or provider change.

- The original 20-case baseline must retain candidate total **USD 80,707.50** and new realized recovery **USD 0.00**.
- The separate expanded resolved demo must retain its **USD 1,000,000.00 approved-for-draft total** and zero new realized recovery. That synthetic approval total is not supplier acceptance, cash received or an accuracy benchmark.
- Required signed-amendment coverage must be **100% of the relevant known signed set** in each labeled case, including lexical/semantic misses and unreviewed amendments. This is snapshot coverage, not proof of real-world source completeness.
- Tenant isolation, source/hash validation, stale-index/case rejection, interrupted-build behavior, model-digest drift and fail-closed provider behavior must pass explicit tests. Explanation/index failure must not change financial amounts or silently grant approval.
- Use held-out synthetic cases labeled by a finance/procurement subject-matter reviewer. Keep expected outcomes and evaluation files isolated from operational imports, retrieval and model prompts.
- Report ranking metrics such as recall@k, reciprocal rank or nDCG separately from governing-document completeness, explanation factual consistency and business accuracy. Measure build/query latency and resource use in the intended environment. Agree on thresholds before the release evaluation; this document supplies no invented benchmark results.

## Decisions for a future real-data pilot

The current milestone is already selected: local/private execution, invented invoices and contracts, local model/index components, human review and local history. No external application answers are needed to complete it. If a later real-data integration or production deployment is requested, resolve:

- Which ERP/AP system, export/API formats, contract repository and identity provider will be used, and who may authorize read-only access?
- Is the pilot single-tenant or multi-tenant, which reviewer roles are required, and where must source/document authorization be enforced?
- Which private hosting environment, network/data-region restrictions, model hardware and approved embedding/explanation models apply?
- What are the expected document/chunk counts, update frequency, query concurrency, latency/recovery objectives and retention/deletion requirements?
- Who owns source completeness, term approval, evaluation labels, incident response, restore drills and the final private-pilot release decision?

The selected local milestone can be completed independently of those future decisions. Describe the result as a local synthetic review workflow and semantic-search foundation; production readiness requires the deferred gates and separate verification.
