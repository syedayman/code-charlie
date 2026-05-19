"""
Supabase client for the Code Charlie Streamlit app.

Service-role only. No per-user JWT auth — the app is single-tenant behind
a password gate, so RLS is not used on `code_charlie_streamlit_sessions`.
The shared `compliance_embeddings` table is read-only from this app.
"""

from supabase import Client, create_client

from core.config import settings


def get_supabase_client() -> Client:
    """Service-role Supabase client. Bypasses RLS."""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
