"""
Supabase CRUD for code_charlie_streamlit_sessions.

All rows are owned by settings.GATE_USER_ID (single shared "gate user")
since the app is single-tenant behind a password gate. RLS is disabled
on this table — the service-role client is used.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

from core.config import settings
from core.supabase_client import get_supabase_client


TABLE_NAME = "code_charlie_streamlit_sessions"


def create_session() -> Dict[str, Any]:
    """Create a new session row. Returns the inserted row dict."""
    supabase = get_supabase_client()
    result = (
        supabase.table(TABLE_NAME)
        .insert({"user_id": settings.GATE_USER_ID})
        .execute()
    )
    if not result.data:
        raise RuntimeError("Failed to insert session row")
    return result.data[0]


def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the gate user's sessions, most recently active first."""
    safe_limit = max(1, min(limit, 100))
    supabase = get_supabase_client()
    result = (
        supabase.table(TABLE_NAME)
        .select("id, title, scope_code, scope_doc, message_count, created_at, last_message_at")
        .eq("user_id", settings.GATE_USER_ID)
        .is_("deleted_at", "null")
        .order("last_message_at", desc=True)
        .limit(safe_limit)
        .execute()
    )
    return result.data or []


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Return one session row or None."""
    supabase = get_supabase_client()
    result = (
        supabase.table(TABLE_NAME)
        .select("*")
        .eq("id", session_id)
        .eq("user_id", settings.GATE_USER_ID)
        .is_("deleted_at", "null")
        .maybe_single()
        .execute()
    )
    return result.data if result else None


def update_after_message(
    session_id: str,
    row: Dict[str, Any],
    scope: Optional[Dict[str, Any]],
) -> None:
    """Bump last_message_at + message_count. Set scope_code/scope_doc on
    the first turn that produced a scope (don't clobber later)."""
    supabase = get_supabase_client()

    update_payload: Dict[str, Any] = {
        "last_message_at": datetime.now(timezone.utc).isoformat(),
        "message_count": (row.get("message_count") or 0) + 1,
    }

    scope = scope or {}
    if scope.get("compliance_code") and not row.get("scope_code"):
        update_payload["scope_code"] = scope["compliance_code"]

    raw_docs = scope.get("document_names") or []
    if isinstance(raw_docs, str):
        scope_docs = [raw_docs]
    elif isinstance(raw_docs, list):
        scope_docs = [d for d in raw_docs if isinstance(d, str) and d.strip()]
    else:
        scope_docs = []
    if isinstance(scope.get("document_name"), str):
        scope_docs.append(scope["document_name"])
    scope_docs = list(dict.fromkeys(d.strip() for d in scope_docs if d.strip()))
    if scope_docs and not row.get("scope_doc"):
        update_payload["scope_doc"] = ", ".join(scope_docs)

    supabase.table(TABLE_NAME).update(update_payload).eq("id", session_id).execute()


def rename_session(session_id: str, title: str) -> None:
    supabase = get_supabase_client()
    supabase.table(TABLE_NAME).update({"title": title.strip()[:120]}).eq("id", session_id).execute()


def delete_session(session_id: str) -> None:
    """Soft-delete a session."""
    supabase = get_supabase_client()
    supabase.table(TABLE_NAME).update(
        {"deleted_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", session_id).execute()


def auto_title_session(
    session_id: str,
    first_user_message: str,
    first_assistant_reply: str,
) -> None:
    """Generate and save a concise title from the first exchange.

    Skip if a title already exists. Same prompt + model as KARR-AI's
    Code Charlie auto-titling.
    """
    supabase = get_supabase_client()

    try:
        existing = (
            supabase.table(TABLE_NAME)
            .select("title")
            .eq("id", session_id)
            .single()
            .execute()
        )
        if existing.data and existing.data.get("title"):
            return
    except Exception:
        return

    if not first_user_message.strip():
        return

    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        prompt = (
            f"User asked:\n{first_user_message[:500]}\n\n"
            f"Assistant replied (excerpt):\n{first_assistant_reply[:400]}\n\n"
            "Generate a concise 3-7 word title for this compliance research "
            "chat. Return only the title — no quotes, no trailing period."
        )
        resp = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": "You generate short, descriptive chat titles."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=30,
            temperature=0.3,
        )
        title = (resp.choices[0].message.content or "").strip().strip('"').strip("'")[:120]
        if not title:
            return
        supabase.table(TABLE_NAME).update({"title": title}).eq("id", session_id).execute()
    except Exception:
        return
