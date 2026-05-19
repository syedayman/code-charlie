-- Migration 001: code_charlie_streamlit_sessions
--
-- Sidebar metadata for the standalone Streamlit Code Charlie app. Kept
-- separate from KARR-AI's `code_charlie_sessions` table so Streamlit
-- chats don't mix with KARR dashboard chats.
--
-- Single-tenant: all rows are owned by a fixed `user_id` UUID (the
-- "gate user" — see settings.GATE_USER_ID). No FK to auth.users, no RLS.
-- The Streamlit app uses the Supabase service-role key.
--
-- LangGraph checkpoints continue to live in the shared `checkpoints` /
-- `checkpoint_writes` tables (auto-created by PostgresSaver.setup()).
-- Thread IDs are UUIDs so they never collide with KARR's checkpoints.

BEGIN;

CREATE TABLE IF NOT EXISTS code_charlie_streamlit_sessions (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         uuid NOT NULL,
  title           text,
  scope_code      text,
  scope_doc       text,
  message_count   int NOT NULL DEFAULT 0,
  created_at      timestamptz NOT NULL DEFAULT now(),
  last_message_at timestamptz NOT NULL DEFAULT now(),
  deleted_at      timestamptz
);

CREATE INDEX IF NOT EXISTS idx_ccs_streamlit_user_recent
  ON code_charlie_streamlit_sessions (user_id, last_message_at DESC)
  WHERE deleted_at IS NULL;

COMMIT;
