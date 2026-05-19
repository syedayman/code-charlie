"""
Message helpers for the Code Charlie agent.

State is duck-typed: anything with a `.get("messages", [...])` works
(CodeCharlieState, plain dicts, LangGraph TypedDict instances).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def add_assistant_message(
    state: Mapping[str, Any],
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an assistant message dict to append to state['messages']."""
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if metadata:
        message["metadata"] = metadata
    return message


def add_user_message(
    content: str,
    selection: Optional[Dict[str, Any]] = None,  # noqa: ARG001
) -> Dict[str, Any]:
    """Build a user message dict to append to state['messages']."""
    return {
        "role": "user",
        "content": content,
    }


def get_last_user_message(state: Mapping[str, Any]) -> Optional[str]:
    """Return the content of the most recent user message, or None."""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "content"):
            msg_type = getattr(msg, "type", "")
            if msg_type == "human":
                return msg.content
        else:
            if msg.get("role") == "user":
                return msg.get("content")
    return None
