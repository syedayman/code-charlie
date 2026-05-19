"""
LangGraph builder for the standalone Code Charlie chatbot.

Graph shape:

    START
      │
      ▼
    classify_and_scope
      ├── pending_clarification → END (clarification message already on state)
      ├── intent == "compliance_question" → code_charlie_compliance_rag → END
      └── intent == "general_chat"        → END (inline reply on state)
"""

import logging
from typing import Literal, Optional

from langgraph.graph import END, StateGraph

from agent.state import CodeCharlieState
from agent.checkpointer import get_checkpointer
from agent.nodes.compliance import code_charlie_compliance_rag_node
from agent.nodes.routing import classify_and_scope_node

logger = logging.getLogger(__name__)


def _route_after_scope(state: CodeCharlieState) -> Literal["compliance", "end"]:
    """Route out of classify_and_scope based on what it set."""
    if state.get("pending_clarification"):
        return "end"
    if state.get("intent") == "compliance_question":
        return "compliance"
    return "end"


_code_charlie_compiled_graph = None


def build_code_charlie_graph():
    """Build (and cache) the Code Charlie graph."""
    global _code_charlie_compiled_graph

    if _code_charlie_compiled_graph is not None:
        return _code_charlie_compiled_graph

    logger.info("Building Code Charlie LangGraph...")

    workflow = StateGraph(CodeCharlieState)

    workflow.add_node("classify_and_scope", classify_and_scope_node)
    workflow.add_node("code_charlie_compliance_rag", code_charlie_compliance_rag_node)

    workflow.set_entry_point("classify_and_scope")
    workflow.add_conditional_edges(
        "classify_and_scope",
        _route_after_scope,
        {
            "compliance": "code_charlie_compliance_rag",
            "end": END,
        },
    )
    workflow.add_edge("code_charlie_compliance_rag", END)

    checkpointer = get_checkpointer()
    _code_charlie_compiled_graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Code Charlie graph built successfully")
    return _code_charlie_compiled_graph


def invoke_code_charlie_graph(state: CodeCharlieState, thread_id: str) -> CodeCharlieState:
    """Run the graph against the given state for one turn."""
    graph = build_code_charlie_graph()
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(state, config)


def get_code_charlie_state(thread_id: str) -> Optional[CodeCharlieState]:
    """Read the current checkpoint state for a session (None if it doesn't
    exist yet)."""
    graph = build_code_charlie_graph()
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = graph.get_state(config)
    if checkpoint and checkpoint.values:
        return checkpoint.values
    return None


def initialize_code_charlie_session_checkpoint(session_id: str, user_id: str) -> None:
    """Seed a fresh session's checkpoint with the intro message so it shows
    up the moment the user opens the chat."""
    from agent.state import create_general_intro_state

    graph = build_code_charlie_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = create_general_intro_state(user_id=user_id, session_id=session_id)
    graph.update_state(config, initial_state)
    logger.info(f"Initialized Code Charlie checkpoint for session {session_id}")
