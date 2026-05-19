"""
classify_and_scope node for the Code Charlie graph.

Responsibilities:

  1. If the prior turn left a pending_clarification, try to interpret this
     turn's message as the user's pick. On match → clear the clarification,
     set scope, replay the original question.

  2. Otherwise classify the message:
       * fast-path: deterministic alias match against the 12 known codes /
         well-known document names
       * fallback: gpt-5.4-nano JSON classifier with recent history

  3. If the classifier is confident → set scope and route to compliance RAG.

  4. If not confident → emit a clarification request with chip options.

  5. If the message isn't compliance-related → short inline general-chat reply.
"""

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from core.config import settings
from agent.state import CodeCharlieState
from agent.nodes.keyword_filter import has_compliance_keywords
from agent.messages import add_assistant_message, get_last_user_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentRef:
    """Exact ingested document target for compliance retrieval."""

    compliance_code: str
    document_name: str


_openai_client: Optional[OpenAI] = None


def _client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


CODE_OPTIONS: List[str] = [
    "DBC",
    "CIBSE",
    "EN81",
    "BCO",
    "ASME, ADA ,IBC",
    "HTM",
    "ISO",
    "DoH (Abu Dhabi)",
    "BMU",
    "CSI",
    "Machinery directive",
    "Other",
]


CODE_ALIASES: List[Tuple[str, str]] = [
    ("dubai building code", "DBC"),
    ("dbc part d", "DBC"),
    ("dbc part c", "DBC"),
    ("dbc", "DBC"),
    ("sharjah building", "DBC"),
    ("cibse guide d", "CIBSE"),
    ("cibse", "CIBSE"),
    ("en 81-20", "EN81"),
    ("en 81-70", "EN81"),
    ("en 81-72", "EN81"),
    ("en 81-41", "EN81"),
    ("en 81-73", "EN81"),
    ("en 81-58", "EN81"),
    ("en 81-76", "EN81"),
    ("en 81", "EN81"),
    ("en81", "EN81"),
    ("en115", "EN81"),
    ("en 115", "EN81"),
    ("bco", "BCO"),
    ("british council for offices", "BCO"),
    ("asme a17", "ASME, ADA ,IBC"),
    ("asme", "ASME, ADA ,IBC"),
    ("ada standards", "ASME, ADA ,IBC"),
    ("ibc 2021", "ASME, ADA ,IBC"),
    ("ibc chapter 30", "ASME, ADA ,IBC"),
    ("nfpa 5000", "ASME, ADA ,IBC"),
    ("nfpa 101", "ASME, ADA ,IBC"),
    ("htm 08-02", "HTM"),
    ("hbn 00-04", "HTM"),
    ("htm", "HTM"),
    ("iso 4190", "ISO"),
    ("iso 8100-30", "ISO"),
    ("iso 8100-32", "ISO"),
    ("iso-ts-18870", "ISO"),
    ("iso 8100", "ISO"),
    ("bs iso", "ISO"),
    ("doh abu dhabi", "DoH (Abu Dhabi)"),
    ("department of health abu dhabi", "DoH (Abu Dhabi)"),
    ("doh", "DoH (Abu Dhabi)"),
    ("bs en 1808", "BMU"),
    ("bmu", "BMU"),
    ("csi master format", "CSI"),
    ("masterformat", "CSI"),
    ("machinery directive", "Machinery directive"),
    ("machine directive", "Machinery directive"),
    ("2006/42/ec", "Machinery directive"),
    ("2023/1230", "Machinery directive"),
    ("elevator traffic handbook", "Other"),
    ("vertical transportation handbook", "Other"),
    ("strakosch", "Other"),
    ("ctbuh", "Other"),
    ("uflsc", "Other"),
    ("uae fire life safety", "Other"),
]


DOC_ALIASES: List[Tuple[str, Tuple[str, str]]] = [
    ("dbc part d", ("DBC", "DBC Part D")),
    ("dbc part c", ("DBC", "DBC Part C VT only")),
    ("cibse guide d 2025", ("CIBSE", "CIBSE Guide D 2025")),
    ("nfpa 5000", ("ASME, ADA ,IBC", "NFPA 5000-2009 Building Construction and Safety Code")),
    ("nfpa 101", ("ASME, ADA ,IBC", "NFPA 101-2024")),
    ("ibc chapter 30", ("ASME, ADA ,IBC", "IBC Chapter 30 Elevators and Conveying Systems")),
    ("ibc 2021", ("ASME, ADA ,IBC", "IBC 2021")),
    ("htm 08-02", ("HTM", "HTM 08-02 2016 Lifts")),
    ("elevator traffic handbook", ("Other", "Elevator Traffic Handbook -Theory and practice-Second edition-Gina Barney and Lutfi Al-Sharif")),
]


# Optional local PDF directory for richer alias detection. When absent (as
# on Streamlit Cloud), DOC_ALIASES alone is used. Override via env if you
# want to point it at a local KARR-AI checkout.
COMPLIANCE_SOURCES_DIR = Path(__file__).resolve().parents[2] / "compliance_sources"
_SORT_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)*_")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _clean_source_document_name(filename_stem: str) -> str:
    name = filename_stem
    while name.lower().endswith(".pdf"):
        name = name[:-4]
    name = _SORT_PREFIX_RE.sub("", name)
    name = name.replace("_", " ")
    return re.sub(r"\s+", " ", name).strip()


def _normalize_lookup_text(value: str) -> str:
    normalized = _NON_ALNUM_RE.sub(" ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


@lru_cache(maxsize=1)
def _source_document_refs() -> Tuple[DocumentRef, ...]:
    if not COMPLIANCE_SOURCES_DIR.exists():
        return ()

    refs: List[DocumentRef] = []
    seen: Set[Tuple[str, str]] = set()
    for pdf_path in sorted(COMPLIANCE_SOURCES_DIR.glob("*/*")):
        if not pdf_path.is_file() or not pdf_path.name.lower().endswith(".pdf"):
            continue
        code = pdf_path.parent.name
        document_name = _clean_source_document_name(pdf_path.stem)
        key = (code, document_name)
        if document_name and key not in seen:
            refs.append(DocumentRef(code, document_name))
            seen.add(key)
    return tuple(refs)


def _document_identifier_aliases(document_name: str) -> Set[str]:
    text = _normalize_lookup_text(document_name)
    aliases: Set[str] = {text} if text else set()

    patterns = [
        r"\bnfpa\s+\d+\b",
        r"\bibc\s+chapter\s+\d+\b",
        r"\bibc\s+\d{4}\b",
        r"\basme\s+a\d+\s+\d+\b",
        r"\b\d{4}\s+ada\s+standards\b",
        r"\bada\s+standards\b",
        r"\ben\s+81\s+\d+\b",
        r"\ben81\s+\d+\b",
        r"\ben115\b",
        r"\ben\s+115\b",
        r"\bhtm\s+\d+\s+\d+\b",
        r"\bhbn\s+\d+\s+\d+\b",
        r"\bbs\s+en\s+\d+\b",
        r"\bbs\s+iso\s+\d+\s+\d+\b",
        r"\biso\s+\d+\s+\d+\b",
        r"\biso\s+ts\s+\d+\s+\d+\b",
        r"\bcibse\s+guide\s+d(?:\s+\d{4})?\b",
        r"\bbco\s+\d{4}\b",
        r"\bcsi\s+master\s+format\s+\d{4}\b",
        r"\bcsi\s+master\s+format\b",
        r"\belevator\s+traffic\s+handbook\b",
        r"\bvertical\s+transportation\s+handbook\b",
        r"\bctbuh\s+vtprimer\s+\d{4}\b",
        r"\bctbuh\s+vtprimer\b",
        r"\buae\s+fire\s+life\s+safety\s+code\b",
        r"\bmachine\s+directive(?:\s+regulation)?(?:\s+ue)?(?:\s+\d{4}\s+\d+)?\b",
        r"\bmachinery\s+directive\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            aliases.add(match.group(0).strip())

    for alias in list(aliases):
        if alias.startswith("en81 "):
            aliases.add(alias.replace("en81 ", "en 81 ", 1))
        if alias == "en115":
            aliases.add("en 115")
        if alias.startswith("cibse guide d "):
            aliases.add("cibse guide d")
        if alias.startswith("asme ") and len(alias.split()) >= 3:
            aliases.add(" ".join(alias.split()[:2]))
        if alias.startswith("iso ") and len(alias.split()) >= 3:
            aliases.add(" ".join(alias.split()[:2]))
        if alias.startswith("bs iso ") and len(alias.split()) >= 4:
            aliases.add(" ".join(alias.split()[:3]))

    return {alias for alias in aliases if len(alias) >= 3}


@lru_cache(maxsize=1)
def _document_alias_map() -> Dict[str, Tuple[DocumentRef, ...]]:
    alias_map: Dict[str, List[DocumentRef]] = {}

    def add(alias: str, ref: DocumentRef) -> None:
        normalized = _normalize_lookup_text(alias)
        if not normalized:
            return
        refs = alias_map.setdefault(normalized, [])
        if ref not in refs:
            refs.append(ref)

    for alias, (code, document_name) in DOC_ALIASES:
        add(alias, DocumentRef(code, document_name))

    for ref in _source_document_refs():
        for alias in _document_identifier_aliases(ref.document_name):
            add(alias, ref)

    return {alias: tuple(refs) for alias, refs in alias_map.items()}


def detect_explicit_documents(message: str) -> List[DocumentRef]:
    text = _normalize_lookup_text(message or "")
    if not text:
        return []

    hits: List[Tuple[int, int, DocumentRef]] = []
    for alias, refs in _document_alias_map().items():
        if len(refs) != 1:
            continue
        match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text)
        if match:
            hits.append((match.start(), -len(alias), refs[0]))

    hits.sort(key=lambda item: (item[0], item[1]))
    selected: List[DocumentRef] = []
    seen_docs: Set[Tuple[str, str]] = set()
    for _, _, ref in hits:
        key = (ref.compliance_code, ref.document_name)
        if key not in seen_docs:
            selected.append(ref)
            seen_docs.add(key)
    return selected


def detect_ambiguous_documents(message: str) -> List[DocumentRef]:
    text = _normalize_lookup_text(message or "")
    if not text:
        return []

    exact_spans: List[Tuple[int, int]] = []
    for alias, refs in _document_alias_map().items():
        if len(refs) != 1:
            continue
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text):
            exact_spans.append((match.start(), match.end()))

    candidates: List[Tuple[int, int, DocumentRef]] = []
    for alias, refs in _document_alias_map().items():
        if len(refs) <= 1:
            continue
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text):
            if any(
                exact_start == match.start() and exact_end > match.end()
                for exact_start, exact_end in exact_spans
            ):
                continue
            for ref in refs:
                candidates.append((match.start(), -len(alias), ref))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: List[DocumentRef] = []
    seen_docs: Set[Tuple[str, str]] = set()
    for _, _, ref in candidates:
        key = (ref.compliance_code, ref.document_name)
        if key not in seen_docs:
            selected.append(ref)
            seen_docs.add(key)
    return selected


def _scope_from_documents(
    documents: List[DocumentRef],
    source: str = "explicit",
) -> Dict[str, Any]:
    document_names = [doc.document_name for doc in documents]
    codes: List[str] = []
    for doc in documents:
        if doc.compliance_code not in codes:
            codes.append(doc.compliance_code)

    return {
        "compliance_code": codes[0] if len(codes) == 1 else None,
        "document_name": document_names[0] if len(document_names) == 1 else None,
        "document_names": document_names,
        "confidence": "high",
        "source": source,
    }


def detect_explicit_scope(message: str) -> Optional[Dict[str, Any]]:
    if not message:
        return None
    text = message.lower()

    documents = detect_explicit_documents(message)
    if documents:
        return _scope_from_documents(documents)

    for alias, code in CODE_ALIASES:
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text):
            return {
                "compliance_code": code,
                "document_name": None,
                "document_names": [],
                "confidence": "high",
                "source": "explicit",
            }

    return None


_LLM_CLASSIFIER_SYSTEM = """You classify questions for a building-code research chatbot.

The chatbot has access to embeddings of these compliance codes:
- DBC (Dubai Building Code, including DBC Part C and Part D, and Sharjah Building Regulations)
- CIBSE (CIBSE Guide D 2020 / 2025)
- EN81 (BS EN 81 series for lifts + EN 115 for escalators)
- BCO (British Council for Offices guides 2009-2023)
- "ASME, ADA ,IBC" (ASME A17.1, NFPA 5000, 2010 ADA Standards, NFPA 101, IBC 2021, IBC Chapter 30)
- HTM (HTM 08-02 Lifts, HBN 00-04 Circulation — UK NHS)
- ISO (BS ISO 4190-1, ISO 8100-30, ISO 8100-32, ISO-TS-18870)
- "DoH (Abu Dhabi)" (Abu Dhabi Department of Health VT requirements)
- BMU (BS EN 1808 — Building Maintenance Units)
- CSI (CSI MasterFormat 2016)
- "Machinery directive" (EU 2006/42/EC, 2023/1230)
- Other (Elevator Traffic Handbook, Vertical Transportation Handbook, CTBUH VT Primer, UAE Fire Life Safety Code)

Decide what scope (if any) the user clearly intends, using their wording AND the recent conversation.

Return JSON exactly in this shape:
{
  "is_compliance": true | false,
  "confident": true | false,
  "compliance_code": "<one of the codes above>" | null,
  "document_name": null,
  "reason": "<short justification>"
}

Rules:
- `is_compliance` = false ONLY for greetings or completely off-topic chatter (weather, sports, etc.). Anything about lifts, elevators, escalators, building codes, fire safety, accessibility, traffic analysis, etc. is compliance.
- `confident` = true ONLY if the question clearly belongs to one of the codes above (named explicitly, OR strongly implied by domain — e.g. "Dubai" → DBC, "NHS" → HTM, "EU CE marking for lifts" → Machinery directive / EN81).
- If multiple codes plausibly apply and none is clearly primary, set `confident` = false and `compliance_code` = null. We'll ask the user.
- Never invent a code; only use the codes in the list above.
- `document_name` is always null in your output — we don't have you pick documents.
"""


def _llm_classify_scope(
    user_message: str,
    history_text: str,
) -> Dict[str, Any]:
    try:
        response = _client().chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": _LLM_CLASSIFIER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Recent conversation:\n{history_text or '(none)'}\n\n"
                        f"Current user message:\n{user_message}"
                    ),
                },
            ],
            max_completion_tokens=200,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)

        code = parsed.get("compliance_code")
        if code not in CODE_OPTIONS:
            code = None
            parsed["confident"] = False
        parsed["compliance_code"] = code
        parsed.setdefault("is_compliance", True)
        parsed.setdefault("confident", False)
        parsed.setdefault("document_name", None)
        return parsed
    except Exception as e:
        logger.warning(f"LLM scope classifier failed: {e}")
        return {
            "is_compliance": True,
            "confident": False,
            "compliance_code": None,
            "document_name": None,
            "reason": "classifier_error",
        }


def _build_clarification_message(original_question: str) -> str:
    return (
        "Quick check before I search — which compliance code is your "
        "question about?\n\n"
        "You can pick one of the chips below or just type the name. If "
        "you're not sure or want me to search everything, pick **All**."
    )


def _build_document_clarification_message(original_question: str) -> str:
    return (
        "I found more than one embedded document matching that name. "
        "Which exact document should I search?\n\n"
        "Pick one of the chips below, or choose **All matching documents** "
        "if you want me to check each matching document."
    )


def _parse_clarification_reply(message: str) -> Optional[str]:
    if not message:
        return None
    text = message.strip().lower()

    if text in {"all", "all docs", "all documents", "everything", "any"}:
        return "ALL"

    scope = detect_explicit_scope(text)
    if scope and scope.get("compliance_code"):
        return scope["compliance_code"]

    for code in CODE_OPTIONS:
        if text == code.lower():
            return code

    return None


def _document_refs_from_payload(raw_docs: Any) -> List[DocumentRef]:
    refs: List[DocumentRef] = []
    if not isinstance(raw_docs, list):
        return refs

    for item in raw_docs:
        if not isinstance(item, dict):
            continue
        code = item.get("compliance_code")
        doc = item.get("document_name")
        if isinstance(code, str) and isinstance(doc, str) and doc.strip():
            ref = DocumentRef(code, doc)
            if ref not in refs:
                refs.append(ref)
    return refs


def _document_refs_payload(documents: List[DocumentRef]) -> List[Dict[str, str]]:
    return [
        {
            "compliance_code": doc.compliance_code,
            "document_name": doc.document_name,
        }
        for doc in documents
    ]


def _document_options_from_pending(pending: Dict[str, Any]) -> List[DocumentRef]:
    return _document_refs_from_payload(pending.get("document_options") or [])


def _parse_document_clarification_reply(
    message: str,
    pending: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not message:
        return None

    text = _normalize_lookup_text(message)
    options = _document_options_from_pending(pending)
    preselected = _document_refs_from_payload(pending.get("preselected_documents") or [])

    if text in {"all", "all matching", "all matching documents", "both", "all docs"}:
        return _scope_from_documents(preselected + options, source="clarification")

    explicit_docs = detect_explicit_documents(message)
    if explicit_docs:
        return _scope_from_documents(preselected + explicit_docs, source="clarification")

    for option in options:
        if text == _normalize_lookup_text(option.document_name):
            return _scope_from_documents(preselected + [option], source="clarification")

    return None


def _build_document_pending(
    original_question: str,
    ambiguous_documents: List[DocumentRef],
    preselected_documents: Optional[List[DocumentRef]] = None,
    message_count: int = 0,
) -> Dict[str, Any]:
    options = [doc.document_name for doc in ambiguous_documents]
    if len(ambiguous_documents) > 1:
        options.append("All matching documents")

    return {
        "kind": "document",
        "options": options,
        "document_options": _document_refs_payload(ambiguous_documents),
        "preselected_documents": _document_refs_payload(preselected_documents or []),
        "original_question": original_question,
        "asked_at_message_idx": message_count - 1,
    }


def _format_recent_history(messages: List[Any], max_messages: int = 6) -> str:
    parts: List[str] = []
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    for msg in recent:
        if hasattr(msg, "content"):
            role = getattr(msg, "type", "user")
            role = "User" if role == "human" else ("Assistant" if role == "ai" else role.capitalize())
            content = msg.content
        else:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
        if content:
            if len(content) > 300:
                content = content[:300] + "..."
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


_GENERAL_CHAT_SYSTEM_PROMPT = """You are Code Charlie, an AI assistant for building codes and compliance research.

You can help users research:
- DBC (Dubai Building Code, including Part D for vertical transportation)
- CIBSE Guide D
- EN 81 / EN 115 (European lift / escalator standards)
- BCO (British Council for Offices guides)
- ASME A17.1, NFPA, IBC, ADA Standards
- HTM 08-02 (UK NHS lifts)
- ISO 4190 / ISO 8100
- Abu Dhabi DoH VT requirements
- BS EN 1808 (BMU)
- CSI MasterFormat
- EU Machinery Directive
- Reference handbooks (Elevator Traffic Handbook, Strakosch, CTBUH VT Primer, UAE Fire Life Safety Code)

If the user's message is a greeting or completely off-topic, briefly say hi and steer them back toward what you can help with. Be concise (1-3 short sentences). No emojis."""


def _generate_general_chat_reply(user_message: str, history: List[Any]) -> str:
    try:
        formatted = _format_recent_history(history, max_messages=10)
        response = _client().chat.completions.create(
            model=settings.GEN_MODEL,
            messages=[
                {"role": "system", "content": _GENERAL_CHAT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Conversation so far:\n{formatted}\n\nNew message:\n{user_message}",
                },
            ],
            max_completion_tokens=200,
            temperature=0.6,
        )
        return (response.choices[0].message.content or "").strip() or (
            "Hi! Ask me anything about building codes — DBC, CIBSE, EN 81, "
            "ASME, IBC, ISO, HTM, and more."
        )
    except Exception as e:
        logger.warning(f"general_chat generation failed: {e}")
        return (
            "Hi! Ask me anything about building codes — DBC, CIBSE, EN 81, "
            "ASME, IBC, ISO, HTM, and more."
        )


def classify_and_scope_node(state: CodeCharlieState) -> Dict[str, Any]:
    """Decide intent + scope for this turn."""
    user_message = get_last_user_message(state)
    if not user_message:
        return {
            "intent": "general_chat",
            "messages": [add_assistant_message(state, "I didn't get a message — what would you like to know about?")],
        }

    messages = state.get("messages", []) or []
    pending = state.get("pending_clarification")

    if pending:
        if pending.get("kind") == "document":
            scope = _parse_document_clarification_reply(user_message, pending)
            if scope is not None:
                original_q = (pending.get("original_question") or "").strip()
                updates: Dict[str, Any] = {
                    "pending_clarification": None,
                    "scope": scope,
                    "intent": "compliance_question",
                }
                if original_q and original_q.lower() != user_message.strip().lower():
                    updates["messages"] = [
                        {
                            "role": "user",
                            "content": original_q,
                            "metadata": {"replayed_from_clarification": True},
                        }
                    ]
                return updates

            logger.info("Document clarification reply not parseable; treating as new question")

        pending_exact_documents = detect_explicit_documents(user_message)
        pending_ambiguous_documents = detect_ambiguous_documents(user_message)
        if pending_ambiguous_documents:
            original_q = (pending.get("original_question") or user_message).strip()
            return {
                "intent": "compliance_question",
                "scope": None,
                "pending_clarification": _build_document_pending(
                    original_question=original_q,
                    ambiguous_documents=pending_ambiguous_documents,
                    preselected_documents=pending_exact_documents,
                    message_count=len(messages),
                ),
                "messages": [
                    add_assistant_message(
                        state,
                        _build_document_clarification_message(original_q),
                    )
                ],
            }
        if pending_exact_documents:
            original_q = (pending.get("original_question") or "").strip()
            scope = _scope_from_documents(pending_exact_documents, source="clarification")
            updates = {
                "pending_clarification": None,
                "scope": scope,
                "intent": "compliance_question",
            }
            if original_q and original_q.lower() != user_message.strip().lower():
                updates["messages"] = [
                    {
                        "role": "user",
                        "content": original_q,
                        "metadata": {"replayed_from_clarification": True},
                    }
                ]
            return updates

        picked = _parse_clarification_reply(user_message)
        if picked is not None:
            scope: Dict[str, Any]
            if picked == "ALL":
                scope = {
                    "compliance_code": None,
                    "document_name": None,
                    "document_names": [],
                    "confidence": "high",
                    "source": "clarification",
                }
            else:
                scope = {
                    "compliance_code": picked,
                    "document_name": None,
                    "document_names": [],
                    "confidence": "high",
                    "source": "clarification",
                }

            original_q = (pending.get("original_question") or "").strip()
            updates: Dict[str, Any] = {
                "pending_clarification": None,
                "scope": scope,
                "intent": "compliance_question",
            }
            if original_q and original_q.lower() != user_message.strip().lower():
                updates["messages"] = [
                    {
                        "role": "user",
                        "content": original_q,
                        "metadata": {"replayed_from_clarification": True},
                    }
                ]
            return updates

        logger.info("Clarification reply not parseable as a code; treating as new question")

    exact_documents = detect_explicit_documents(user_message)
    ambiguous_documents = detect_ambiguous_documents(user_message)
    if ambiguous_documents:
        return {
            "intent": "compliance_question",
            "scope": None,
            "pending_clarification": _build_document_pending(
                original_question=user_message,
                ambiguous_documents=ambiguous_documents,
                preselected_documents=exact_documents,
                message_count=len(messages),
            ),
            "messages": [
                add_assistant_message(
                    state,
                    _build_document_clarification_message(user_message),
                )
            ],
        }

    explicit = (
        _scope_from_documents(exact_documents)
        if exact_documents
        else detect_explicit_scope(user_message)
    )

    has_kw = has_compliance_keywords(user_message)

    if explicit:
        return {
            "scope": explicit,
            "pending_clarification": None,
            "intent": "compliance_question",
        }

    prior_scope = state.get("scope") or {}
    prior_code = prior_scope.get("compliance_code")
    word_count = len(user_message.strip().split())
    if prior_code and has_kw and not explicit and word_count <= 12:
        return {
            "scope": {
                "compliance_code": prior_code,
                "document_name": prior_scope.get("document_name"),
                "document_names": prior_scope.get("document_names") or [],
                "confidence": "high",
                "source": "history",
            },
            "pending_clarification": None,
            "intent": "compliance_question",
        }

    history_text = _format_recent_history(messages[:-1] if messages else [], max_messages=6)
    llm_result = _llm_classify_scope(user_message, history_text)

    if not llm_result.get("is_compliance") and not has_kw:
        reply = _generate_general_chat_reply(user_message, messages)
        return {
            "intent": "general_chat",
            "scope": None,
            "pending_clarification": None,
            "messages": [add_assistant_message(state, reply)],
        }

    if explicit:
        return {
            "scope": explicit,
            "pending_clarification": None,
            "intent": "compliance_question",
        }

    if llm_result.get("confident") and llm_result.get("compliance_code"):
        return {
            "scope": {
                "compliance_code": llm_result["compliance_code"],
                "document_name": None,
                "document_names": [],
                "confidence": "high",
                "source": "llm_inferred",
            },
            "pending_clarification": None,
            "intent": "compliance_question",
        }

    return {
        "intent": "compliance_question",
        "scope": None,
        "pending_clarification": {
            "kind": "code",
            "options": CODE_OPTIONS + ["All"],
            "original_question": user_message,
            "asked_at_message_idx": len(messages) - 1,
        },
        "messages": [add_assistant_message(state, _build_clarification_message(user_message))],
    }
