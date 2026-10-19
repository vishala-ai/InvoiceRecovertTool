# Streamlit Community Cloud — synthetic persistent pilot

The user selected **Streamlit Community Cloud**, **Groq-hosted Llama**, **Supabase PostgreSQL**, and repository **InvoiceRecovertTool**. This is a shared synthetic demonstration. Real supplier data, ERP integration, supplier communications, claim submission and real settlement remain excluded. Local execution stays at **http://127.0.0.1:8515/**.

## What is implemented

- `cloud/streamlit_app.py` runs the existing review app inside a durable workspace transaction. Root `app.py` remains the local entrypoint.
- `cloud/requirements.txt` is exported from the actual resolved `uv.lock` for UI, graph, explanation schemas and PostgreSQL. Its location beside the Cloud entrypoint avoids root lockfile precedence excluding optional extras.
- Groq uses `llama-3.3-70b-versatile`, a fixed HTTPS endpoint, JSON-object responses, a 20-second caller deadline and no automatic retries or provider fallback. Llama JSON-object mode is not strict schema enforcement: independent Python schema, exact facts and quotation checks still decide whether to display a draft. Model IDs are pinned in review checkpoints; a changed model requires re-review except for completing an already committed journal event.
- Cloud invoice preparation exposes the server-selected Groq model with explicit case/evidence consent. Local Ollama and local embedding controls are unavailable in this mode. The read-only chatbot keeps its existing validated factual flow without new model calls.
- Complete signed evidence comes from canonical SQLite/FTS5. A hosted explanation model does not require a vector database. The separate local Qdrant/EmbeddingGemma integration is not silently transferred to Cloud.

## Persistent state and transaction boundary

```mermaid
flowchart LR
    B[Authenticated browser] --> A[Streamlit Cloud interaction]
    A --> L[Supabase workspace row lock]
    L --> W[Restore validated private SQLite working copies]
    W --> F[Python assessment and complete signed evidence]
    F --> G[Optional Groq explanation]
    G --> V[Independent fact and citation validation]
    V --> H[Human decision and local checkpoint/journal]
    H --> C[Atomic PostgreSQL publication]
    C --> R[Confirm saved result or rerun]
```

The portfolio, LangGraph checkpoint database and account database are published together in one PostgreSQL transaction. The process lock and database row lock cover restoration, the entire interaction, consistent SQLite backup snapshots and commit. Changes are stored as bounded, hashed, compressed role payloads in the **non-exposed** `invoice_recovery_private` schema. Only changed role files are uploaded; no background or best-effort backup stands in for a commit.

Streamlit rerun/stop controls are honored after commit. On a connection or commit failure the session and provisional success displays are cleared, and the app asks the user to refresh the saved state. It never claims that an uncertain commit definitely failed or automatically repeats an approval. Source/version validation, freshness and journal idempotency remain authoritative.

This design deliberately serializes interactions across the shared workspace, including a bounded model call. It preserves the existing SQLite engine for a small synthetic pilot; it is not a high-throughput relational migration or proof of production capacity. PostgreSQL durability is separate from backup/restore policy. Direct local CLI/database access is outside the web authentication gate.

## 1. Prepare the source repository

After the full checks pass:

```sh
uv run --all-extras --frozen python scripts/prepare_cloud_repository.py
```

The prepared directory is `var/cloud-implementation/InvoiceRecovertTool`. It contains curated source, original synthetic fixtures, tests, documentation, Cloud configuration and the real dependency lockfile. It excludes `.env`, real secret files, account databases, runtime portfolios/checkpoints, verification archives and all other `var/` content. The staged Streamlit configuration removes only the local `[server]` binding; the original local configuration remains byte-for-byte unchanged. Review `PREPARED_FILES.json` before pushing. A private GitHub repository is suitable for source; the app's viewer visibility is configured separately in Streamlit.

## 2. Set up Supabase privately

1. Create or select the intended Supabase project. Enter database passwords yourself in its private setup UI; never send them through chat or put them in source.
2. Run `cloud/schema.sql` in the project's SQL editor. Keep `invoice_recovery_private` out of exposed Data API schemas. The script revokes `PUBLIC`, `anon` and `authenticated` access to the stored payloads.
3. Prefer a dedicated database login role granted only schema `USAGE` and `SELECT, INSERT, UPDATE` on its two tables; the SQL file includes the grant shape without any password. Use the administrator connection only for schema/role setup.
4. Copy the **Session pooler / IPv4 / port 5432** PostgreSQL URI for the selected role from Supabase Connect. Download its **public server root certificate**. The adapter requires hostname-verified TLS (`verify-full`) and a trusted root; insecure TLS modes, connection query overrides and loopback hosts are rejected. Psycopg prepared statements are disabled for pooler compatibility.

See [Supabase connections](https://supabase.com/docs/guides/database/connecting-to-postgres). Do not assume a successfully parsed certificate or a local PostgreSQL test verifies the actual Supabase TLS handshake.

## 3. Privately seed the expanded synthetic portfolio once

The source generator currently depends on earlier local profiles; the expanded dataset is not in a clean GitHub clone. Seed the existing reviewed synthetic business portfolio and its matching checkpoints directly into Supabase, rather than publishing these files to GitHub:

```sh
# Local validation only: no credentials or network required.
uv run --all-extras --frozen python scripts/seed_cloud_workspace.py \
  --source var/scaled-demo/portfolio.sqlite3 --check

# After schema creation; the connection URI is entered through a hidden prompt.
uv run --all-extras --frozen python scripts/seed_cloud_workspace.py \
  --source var/scaled-demo/portfolio.sqlite3 \
  --workspace invoice-recovery-demo \
  --ca-cert /absolute/path/to/public-supabase-root.crt \
  --prompt-database-url
```

The seed checks synthetic provenance, schema/source projections, source hashes, archive locks and matching checkpoint identities. Both source files are reserved against writers while consistent backups are made. Existing remote records cannot be overwritten. Cloud accounts start empty: the live local account database and its sessions/password hashes are **never seed inputs**. The command confirms publication only after commit; on an ambiguous connection failure inspect the remote state before retrying.

The checked expanded source has 1,030 current plus 2,000 locked historical cases; USD 1,750,000.00 calculated candidates and zero new operational realized recovery. Its original all-year approved total, forecast and invented historical-receipt overlay remain distinct. These are synthetic records, not real recovered cash.

## 4. Configure Groq and private Streamlit Secrets

Create a Groq API key in the intended account/project, with suitable usage limits. Enter it **directly in Streamlit Community Cloud Secrets**, along with the Supabase URI/certificate and workspace ID. `cloud/secrets.example.toml` documents the fields and contains placeholders only; never fill that tracked example with real credentials.

`[cloud.owner]` privately provisions the first Cloud owner when the seeded account store is empty. Use a unique 15–128 character passphrase. There is no public first-visitor owner signup. After the first successful startup, the owner account is durable in Supabase and the bootstrap owner section can be removed. It does not reset or overwrite existing accounts; normal password change/reset controls remain available. Local `johndoe` credentials are not transferred automatically.

All signed-in accounts share the synthetic DEMO workspace. Owner/accountant roles manage accounts but do not provide per-vendor permissions or authenticated attribution for the pre-existing fictional reviewer labels. Company SSO, MFA and tenant/document authorization remain future work.

See [Streamlit Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management). The Cloud path does not load local `.env` files, publish credentials in URLs, or persist API keys in model checkpoints.

## 5. Deploy and verify the actual service

1. In Streamlit Community Cloud, select the intended GitHub account/repository and branch, entrypoint **`cloud/streamlit_app.py`**, and **Python 3.12**. Review any repository-access authorization before granting it.
2. Add the private Secrets fields, then deploy. Do not enable public viewing until the configured owner login and persistence checks work.
3. Verify owner sign-in, a second accountant, provider consent, one valid Groq draft, rejection of an invalid/unavailable provider response, a human-review pause and one idempotent saved decision.
4. Reboot the app. Confirm accounts, account disabling, the decision journal and workflow checkpoints persist together. Confirm another worker observes the current approval and stale selections require fresh review.
5. Verify the actual URL, model availability/quota and Supabase pooler/TLS. Keep secrets and note/document text out of logs. Never mark these external checks complete merely because unit tests or a local PostgreSQL instance passed.

[Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) requires repository access and private configuration. [Local-file persistence is not guaranteed](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data); a local-only fallback is intentionally unavailable in this Cloud entrypoint.

## Validation and remaining limits

The original test-only 20-case expectation remains **USD 80,707.50 candidates / USD 0.00 new realized recovery**. Financial outcomes are not adjusted to fit a hosted model. Controlled Groq, Cloud transaction, account, package and seed tests run locally. A temporary real PostgreSQL 16.2 check exercised schema revocations, coupled restore, independent-process row locking, account revocation, rollback, replay and database restart.

The local test PostgreSQL was built without TLS; production TLS enforcement remains enabled, but the real Supabase handshake must still be checked with operator-provided configuration. Real Groq inference, Supabase provisioning/connection/restore policy, Cloud startup and public URL checks remain unverified until they are actually performed. Source publication is recorded separately from those service checks. This is not a completed production release.
