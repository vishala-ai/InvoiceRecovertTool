# Codex task sequence

Open the extracted project in VS Code/Codex. Use one prompt per task, inspect the diff, and run the stated gate. Do not paste API keys or real contracts into Codex. These tasks extend existing code; they do not ask Codex to recreate the repository.

## Prompt 1 — reproduce the baseline and verify optional integrations

```text
Read AGENTS.md, README.md, and docs/IMPLEMENTATION_GUIDE.md.
First inspect the existing code without editing it. Explain which paths are
implemented and which have not been verified with installed dependencies.
Run the no-provider financial tests and reproduce the 20-case audit.
The expected candidate total is exactly USD 80,707.50 and realized recovery is zero.
Do not change ground truth to make tests pass.

On this internet-enabled machine, resolve dependencies with uv sync --all-extras.
Run the full test suite; the optional LangGraph test should execute, not skip.
Smoke-test the Streamlit app on localhost and inspect the evidence display.
Fix only dependency/API compatibility or reproducible defects, add regression
tests, run them, and show the diff. Generate a genuine uv.lock.
Do not access .env contents, real business data, or unrelated directories.
Report exact commands and outcomes, including anything still unverified.
```

Gate: baseline totals unchanged; all installed-framework tests run; UI opens locally; lockfile is real.

## Prompt 2 — harden ingestion without expanding financial scope

```text
Add explicit schemas for every supplied CSV and manifest. Reject missing,
unexpected, malformed and duplicate columns/IDs with a clear import report.
Keep source amounts as decimal strings. Add source/import row-count and total
reconciliation, including currencies and snapshot timestamps.
Validate orphan references, unsafe paths and bad document hashes.
Use an atomic import and prove a failed import preserves the prior snapshot.
Do not add ERP credentials or relax the synthetic-only guard.
Add tests for each failure and preserve all existing acceptance outcomes.
```

Gate: import failures explain exactly which records need correction; no partial replacement or data loss.

## Prompt 3 — test the RAG adapter without trusting its prose

```text
Keep the deterministic engine unchanged. Add injectable provider interfaces
and mocked structured-output tests around ai.py. Prove provider errors,
invalid output, invented documents, fabricated quotes, omitted governing
amendments and document-embedded action instructions cannot alter case values,
approval state, source terms, or external systems.
Add an opt-in live integration test using environment variables, never hardcoded
keys. Record model ID, prompt version, latency and available token usage without
logging secrets. Do not run a paid live call without explicit approval.
Keep no-provider tests completely offline.
```

Gate: same computed amounts with AI on/off; failed or unsupported drafts withheld; live-test status explicit.

## Prompt 4 — human-reviewed term extraction

```text
Build a local review screen for UNREVIEWED_PROPOSAL output, showing exact source
text alongside rate, currency, unit, role, location, effective dates, supersession
and unresolved questions. Allow approve, reject or edit with a review note.
Do not let a model promote its own proposal into reviewed_terms.
Version approved terms and retain source hash, exact quote, reviewer and timestamp.
Invalidate affected case assessments when signed source evidence changes.
Keep reviewer identity labeled demo-only until real authentication exists.
Add tests for unsigned drafts, conflicting amendments and stale proposals.
```

Gate: original curated-fixture baseline and newly reviewed proposals are clearly distinguished; no automatic promotion.

## Prompt 5 — integrate durable review into the UI

```text
Connect the existing LangGraph checkpoint flow to the Streamlit review view
instead of replacing the financial engine. The UI must display the exact paused
case digest and documents a reviewer is deciding on.
Test app/process restart between start and resume, stale case changes, repeated
resume, rejection, evidence requests, completed thread reuse and wrong-tenant
thread access. Use stable idempotency keys for journal writes.
Keep approval as APPROVED_FOR_DRAFT. Do not send emails or claims.
Label initial candidate totals separately from current review states.
```

Gate: the actual paused case is reviewed; no duplicate or stale approval; no external action.

## Prompt 6 — evaluation harness and CI

```text
Add an evaluation harness that reports financial correctness, required-source
coverage, forbidden-source leakage, generated-answer citation support and
appropriate abstention separately. Preserve a held-out evaluation set.
Do not use model-generated labels as the sole source of truth for correctness.
Add CI that runs deterministic tests without any secrets, and installed-framework
integration tests using a verified dependency lockfile.
Document which tests require a live API and keep them opt-in.
```

Gate: a failed test identifies the failing layer; no hidden API spend or false 'all tests passed' summary.

## Prompt 7 — prepare a private-pilot design, without deploying it

```text
Read the current limitations and produce a pilot threat model and implementation
plan. Cover authenticated identities, server-enforced authorization, source
completeness, financial export semantics, secret management, tracing egress,
retention/deletion, backups, audit tamper resistance and tenant isolation.
Define a read-only adapter contract for one approved ERP export format.
Do not remove the synthetic guard, ingest real data, connect an external account
or deploy publicly. Present gates requiring business/security sign-off.
```

Gate: the team knows what must be approved and implemented before using confidential data.

## Prompt 8 — recovery ledger design after finance review

```text
Design a settlement-backed recovery ledger that separates identified, validated,
submitted, credited/refunded and closed cases. Model an economic adjustment once
so a refund and applied credit cannot double-count the same recovery.
Require verified source references, dates, currency, allocation and source-system
IDs. Report incremental recoveries separately from existing controls and preexisting
credits. Do not record new real recoveries or change accounting systems.
Use synthetic tests for partial settlements, reversals and duplicate source IDs.
```

Gate: every realized amount is backed by settlement evidence and an agreed counting rule.
