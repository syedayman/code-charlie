"""
CodeCharlieState - LangGraph state schema for the standalone Code Charlie
chatbot.

State is the bare minimum needed for a multi-turn compliance research chat.

The LangGraph checkpointer (PostgresSaver) is the source of truth for the
message history. Supabase row code_charlie_streamlit_sessions tracks title,
scope, message_count, deleted_at — metadata that the sidebar listing needs
without replaying the checkpoint.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages

INTRO_MESSAGE_CONTENT = (
    "Hi! I'm Code Charlie. Ask me anything about building codes and compliance "
    "standards — DBC, CIBSE, EN 81, BCO, ASME, ADA, IBC, HTM, ISO, DoH, BMU, "
    "CSI, Machinery Directive, and more.\n\n"
    "Mention the code or document in your question if you know it (e.g. \"What does CIBSE "
    "Guide D say about rated speed?\"). "
)


class CodeCharlieState(TypedDict):
    """State schema for the standalone Code Charlie graph."""

    user_id: str
    session_id: str

    messages: Annotated[List[Dict[str, Any]], add_messages]

    intent: Optional[Literal[
        "compliance_question",
        "general_chat",
    ]]

    scope: Optional[Dict[str, Any]]

    pending_clarification: Optional[Dict[str, Any]]

    rag: Optional[Dict[str, Any]]

    error: Optional[str]


def create_initial_code_charlie_state(
    user_id: str,
    session_id: str,
    initial_message: str,
) -> CodeCharlieState:
    """State seed for a session that's being created with a first message."""
    return CodeCharlieState(
        user_id=user_id,
        session_id=session_id,
        messages=[{"role": "user", "content": initial_message}],
        intent=None,
        scope=None,
        pending_clarification=None,
        rag=None,
        error=None,
    )


def create_general_intro_state(
    user_id: str,
    session_id: str,
) -> CodeCharlieState:
    """State seed for a session created without a first message — the intro
    message is set so it appears the moment the user opens the chat."""
    return CodeCharlieState(
        user_id=user_id,
        session_id=session_id,
        messages=[{"role": "assistant", "content": INTRO_MESSAGE_CONTENT}],
        intent=None,
        scope=None,
        pending_clarification=None,
        rag=None,
        error=None,
    )
