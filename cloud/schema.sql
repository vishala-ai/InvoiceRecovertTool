-- Run once through the Supabase SQL editor as the database administrator.
-- This private schema MUST NOT be added to Supabase's exposed Data API schemas.
-- The runtime connects directly to PostgreSQL, never with an anon/service API key.
BEGIN;
CREATE SCHEMA IF NOT EXISTS invoice_recovery_private;
REVOKE ALL ON SCHEMA invoice_recovery_private FROM PUBLIC;

CREATE TABLE IF NOT EXISTS invoice_recovery_private.workspace_heads (
    id TEXT PRIMARY KEY CHECK (id ~ '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$'),
    revision BIGINT NOT NULL DEFAULT 0 CHECK (revision >= 0)
);
CREATE TABLE IF NOT EXISTS invoice_recovery_private.workspace_files (
    workspace_id TEXT NOT NULL REFERENCES invoice_recovery_private.workspace_heads(id),
    role TEXT NOT NULL CHECK (role IN ('portfolio', 'checkpoints', 'accounts')),
    sha256 TEXT NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    raw_size BIGINT NOT NULL CHECK (raw_size BETWEEN 100 AND 134217728),
    payload BYTEA NOT NULL CHECK (octet_length(payload) BETWEEN 1 AND 67108864),
    PRIMARY KEY (workspace_id, role)
);
REVOKE ALL ON ALL TABLES IN SCHEMA invoice_recovery_private FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA invoice_recovery_private
    REVOKE ALL ON TABLES FROM PUBLIC;
DO $block$
DECLARE excluded_role TEXT;
BEGIN
    FOREACH excluded_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = excluded_role) THEN
            EXECUTE format('REVOKE ALL ON SCHEMA invoice_recovery_private FROM %I', excluded_role);
            EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA invoice_recovery_private FROM %I',
                           excluded_role);
        END IF;
    END LOOP;
END
$block$;
COMMIT;

-- Recommended: provision a separate LOGIN role privately, using a unique strong
-- password entered in the Supabase administrator console (never committed here).
-- Grant that role only USAGE on this schema and SELECT, INSERT, UPDATE on these
-- two tables. It needs no schema CREATE, DELETE, public Data API access or DDL.
-- Example grant shape, replacing invoice_recovery_runtime with the created role:
-- GRANT USAGE ON SCHEMA invoice_recovery_private TO invoice_recovery_runtime;
-- GRANT SELECT, INSERT, UPDATE ON invoice_recovery_private.workspace_heads,
--     invoice_recovery_private.workspace_files TO invoice_recovery_runtime;
