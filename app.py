"""
Code Charlie — Streamlit UI

Single-file app:
  - password gate
  - sidebar with multi-session list + new chat button
  - chat history replayed from LangGraph checkpoint
  - sources expander under each assistant message
  - clarification chips when classifier asks "which code?"
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from agent.graph import (
    build_code_charlie_graph,
    get_code_charlie_state,
    initialize_code_charlie_session_checkpoint,
    invoke_code_charlie_graph,
)
from agent.messages import add_user_message
from agent.state import (
    INTRO_MESSAGE_CONTENT,
    create_initial_code_charlie_state,
)
from core.config import settings
from lib.gate import require_password
from lib.sessions import (
    auto_title_session,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    rename_session,
    update_after_message,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(
    page_title="Code Charlie",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Gate
# =============================================================================

if not require_password():
    st.stop()


# =============================================================================
# Eager warm-up after auth (mask first-render DB/pool latency)
# =============================================================================

@st.cache_resource(show_spinner=False)
def _warm_graph() -> bool:
    """Build the LangGraph + open PostgresSaver pool once per worker."""
    build_code_charlie_graph()
    return True


if not st.session_state.get("_warmed"):
    splash = st.empty()
    with splash.container():
        st.markdown("### Code Charlie")
        st.caption("Connecting…")
        with st.spinner(""):
            _warm_graph()
    splash.empty()
    st.session_state["_warmed"] = True


# =============================================================================
# Global CSS — sidebar polish + hover-only actions
# =============================================================================

st.markdown(
    """
<style>
/* Streamlit 1.40+ adds class `st-key-{key}` on the wrapper of any keyed
   widget. We target that class — it's far more stable than data-testid
   selectors, which change across Streamlit versions. */

/* ---- Title button (one per session row) ---- */
.stApp [class*="st-key-open_"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: rgba(0, 0, 0, 0.85) !important;
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    padding: 0.55rem 0.7rem !important;
    line-height: 1.3 !important;
    width: 100% !important;
    border-radius: 8px !important;
    transition: background-color 0.1s ease !important;
}
.stApp [class*="st-key-open_"] button:hover {
    background: rgba(0, 0, 0, 0.06) !important;
}
.stApp [class*="st-key-open_"] button p,
.stApp [class*="st-key-open_"] button > div {
    text-align: left !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
}
.stApp [class*="st-key-open_"] button strong {
    font-weight: 600;
}

/* ---- Icon buttons (rename / delete) — circle, hover bg = circle, no chrome ---- */
.stApp [class*="st-key-rename_"],
.stApp [class*="st-key-delete_"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: auto !important;
}
.stApp [class*="st-key-rename_"] button,
.stApp [class*="st-key-delete_"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: rgba(0, 0, 0, 0.45) !important;
    padding: 0 !important;
    min-width: 36px !important;
    width: 36px !important;
    max-width: 36px !important;
    height: 36px !important;
    font-size: 16px !important;
    line-height: 1 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 50% !important;
    margin: 0 !important;
    flex: none !important;
    transition: background-color 0.1s ease !important;
}
.stApp [class*="st-key-rename_"] button:hover,
.stApp [class*="st-key-delete_"] button:hover {
    background: rgba(0, 0, 0, 0.08) !important;
    color: rgba(0, 0, 0, 0.9) !important;
}
.stApp [class*="st-key-rename_"] button p,
.stApp [class*="st-key-delete_"] button p {
    line-height: 1 !important;
    font-size: 16px !important;
    margin: 0 !important;
    text-align: center !important;
    justify-content: center !important;
}

/* Keep the horizontal row tidy */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    align-items: center;
    background: transparent !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Helpers
# =============================================================================

def _normalize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """LangChain message objects or dicts → plain dicts."""
    out: List[Dict[str, Any]] = []
    for msg in messages or []:
        if hasattr(msg, "content"):
            role = getattr(msg, "type", "user")
            role = "user" if role == "human" else ("assistant" if role == "ai" else role)
            content = msg.content
            metadata = getattr(msg, "metadata", None) or getattr(msg, "additional_kwargs", None) or {}
        else:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            metadata = msg.get("metadata") or {}
        out.append({"role": role, "content": content or "", "metadata": metadata})
    return out


def _fmt_time(iso_str: Optional[str]) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %H:%M")
    except Exception:
        return iso_str


def _ensure_current_session() -> str:
    """Make sure st.session_state['current_session_id'] is set. Creates one if not."""
    sid = st.session_state.get("current_session_id")
    if sid:
        return sid
    row = create_session()
    sid = row["id"]
    initialize_code_charlie_session_checkpoint(session_id=sid, user_id=settings.GATE_USER_ID)
    st.session_state["current_session_id"] = sid
    return sid


def _render_assistant_meta(metadata: Dict[str, Any]) -> None:
    """Render sources + flagged claims under an assistant message."""
    sources = metadata.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for src in sources:
                section = src.get("section") or "—"
                doc = src.get("document") or "—"
                title = src.get("title") or ""
                text = src.get("text") or ""
                similarity = src.get("similarity")
                sim_label = f" · {similarity:.0%}" if isinstance(similarity, (int, float)) else ""
                header = f"**{doc}** — Section {section}{sim_label}"
                if title:
                    header += f"  \n_{title}_"
                st.markdown(header)
                if text:
                    st.caption(text)
                st.markdown("---")

    if metadata.get("dropped_claims"):
        with st.expander("Potential issues flagged"):
            for claim in metadata["dropped_claims"]:
                st.caption(claim)


def _invoke_for_pending(session_id: str, text: str) -> None:
    """Run the graph for the pending user message and persist sidebar metadata."""
    row = get_session(session_id)
    if not row:
        st.error("Session not found.")
        return

    current_state = get_code_charlie_state(session_id)
    if current_state is None:
        current_state = create_initial_code_charlie_state(
            user_id=settings.GATE_USER_ID,
            session_id=session_id,
            initial_message=text,
        )
    else:
        current_state = {
            **current_state,
            "messages": current_state.get("messages", []) + [add_user_message(text)],
            "intent": None,
            "error": None,
        }

    was_first_user_turn = (row.get("message_count") or 0) == 0
    result_state = invoke_code_charlie_graph(current_state, thread_id=session_id)
    update_after_message(session_id, row, result_state.get("scope"))

    if was_first_user_turn and not row.get("title"):
        msgs = _normalize_messages(result_state.get("messages", []))
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        first_assistant = next(
            (
                m["content"]
                for m in msgs
                if m["role"] == "assistant" and m["content"] != INTRO_MESSAGE_CONTENT
            ),
            "",
        )
        if first_user:
            auto_title_session(session_id, first_user, first_assistant)


# =============================================================================
# Sidebar — session list
# =============================================================================

with st.sidebar:
    st.markdown("### Code Charlie")
    st.caption("Building-code compliance research")

    if st.button("＋ New chat", use_container_width=True, type="primary"):
        row = create_session()
        sid = row["id"]
        initialize_code_charlie_session_checkpoint(session_id=sid, user_id=settings.GATE_USER_ID)
        st.session_state["current_session_id"] = sid
        st.session_state.pop("pending_prompt", None)
        st.rerun()

    st.divider()
    st.caption("Recent sessions")

    sessions = list_sessions(limit=50)
    current_id = st.session_state.get("current_session_id")

    if not sessions:
        st.caption("_No sessions yet._")
    else:
        for s in sessions:
            sid = s["id"]
            is_current = sid == current_id
            title = s.get("title") or "Untitled chat"
            subtitle_bits = []
            if s.get("scope_code"):
                subtitle_bits.append(s["scope_code"])
            subtitle_bits.append(_fmt_time(s.get("last_message_at")))
            subtitle = " · ".join(b for b in subtitle_bits if b)

            row_cols = st.columns([0.74, 0.13, 0.13])
            with row_cols[0]:
                label = f"**{title}**\n\n{subtitle}" if subtitle else f"**{title}**"
                if st.button(
                    label,
                    key=f"open_{sid}",
                    use_container_width=True,
                    type="tertiary",
                ):
                    st.session_state["current_session_id"] = sid
                    st.session_state.pop("pending_prompt", None)
                    st.rerun()
            with row_cols[1]:
                if st.button("✎", key=f"rename_{sid}", help="Rename"):
                    st.session_state["_rename_target"] = {"sid": sid, "title": s.get("title") or ""}
                    st.rerun()
            with row_cols[2]:
                if st.button("🗑", key=f"delete_{sid}", help="Delete"):
                    st.session_state["_delete_target"] = {"sid": sid, "title": title}
                    st.rerun()


# Rename + delete modals (dialogs) — open when a target session is queued.
_rename_target = st.session_state.get("_rename_target")
if _rename_target:
    @st.dialog("Rename chat")
    def _rename_dialog():
        target_sid = _rename_target["sid"]
        current_title = _rename_target["title"]
        new_title = st.text_input("New title", value=current_title, key="_rename_field")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Cancel", use_container_width=True, key="_rename_cancel"):
                st.session_state.pop("_rename_target", None)
                st.rerun()
        with col_b:
            if st.button("Save", type="primary", use_container_width=True, key="_rename_save"):
                if new_title.strip():
                    rename_session(target_sid, new_title.strip())
                st.session_state.pop("_rename_target", None)
                st.rerun()
    _rename_dialog()

_delete_target = st.session_state.get("_delete_target")
if _delete_target:
    @st.dialog("Delete chat?")
    def _delete_dialog():
        target_sid = _delete_target["sid"]
        target_title = _delete_target["title"]
        st.write(f"Permanently delete **{target_title}**? This cannot be undone.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Cancel", use_container_width=True, key="_delete_cancel"):
                st.session_state.pop("_delete_target", None)
                st.rerun()
        with col_b:
            if st.button("Delete", type="primary", use_container_width=True, key="_delete_confirm"):
                delete_session(target_sid)
                if target_sid == st.session_state.get("current_session_id"):
                    st.session_state.pop("current_session_id", None)
                st.session_state.pop("_delete_target", None)
                st.rerun()
    _delete_dialog()


# =============================================================================
# Main chat area
# =============================================================================

current_session_id = _ensure_current_session()
state = get_code_charlie_state(current_session_id) or {}
messages = _normalize_messages(state.get("messages", []))

st.title("Code Charlie")
st.caption("Ask compliance questions across DBC, CIBSE, EN 81, BCO, ASME/ADA/IBC, HTM, ISO, DoH, BMU, CSI, Machinery Directive.")

# Render existing history
for m in messages:
    role = m["role"]
    content = m["content"]
    metadata = m.get("metadata") or {}

    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)
        if role == "assistant" and metadata:
            _render_assistant_meta(metadata)


# Pending message: show user bubble + spinner immediately while graph runs.
pending_prompt = st.session_state.get("pending_prompt")
if pending_prompt and pending_prompt.get("session_id") == current_session_id:
    text = pending_prompt["text"]
    # Render the user's message bubble at once so they can see what they sent.
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        with st.spinner("Researching…"):
            _invoke_for_pending(current_session_id, text)
    st.session_state.pop("pending_prompt", None)
    st.rerun()


# Pending clarification chips
pending_clar = state.get("pending_clarification") if state else None
if pending_clar and pending_clar.get("options"):
    st.markdown("**Pick one to continue:**")
    options = pending_clar["options"]
    chip_cols = st.columns(min(4, max(2, len(options))))
    for i, opt in enumerate(options):
        with chip_cols[i % len(chip_cols)]:
            if st.button(opt, key=f"chip_{current_session_id}_{i}_{opt}", use_container_width=True):
                st.session_state["pending_prompt"] = {
                    "session_id": current_session_id,
                    "text": opt,
                }
                st.rerun()


# Chat input
prompt = st.chat_input("Ask about a building code…")
if prompt:
    st.session_state["pending_prompt"] = {
        "session_id": current_session_id,
        "text": prompt,
    }
    st.rerun()
