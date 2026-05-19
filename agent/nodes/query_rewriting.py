"""
Query-side rewriting hooks for the Code Charlie pipeline.

Three independently-toggleable transforms, all OpenAI-only:

  1. expand_multi_query(query) -> list[str]
        Generate 2-3 alternative phrasings via nano.

  2. is_meta_question(query) -> bool
        Regex + nano fallback. When True, scope first search to META_CHUNK_TYPES.

  3. hypothetical_doc(query) -> str
        HyDE — nano writes a paragraph that *would* answer the question.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Tuple

from openai import OpenAI

from core.config import settings

logger = logging.getLogger(__name__)


META_CHUNK_TYPES: Tuple[str, ...] = ("doc_profile", "doc_toc", "glossary")

MAX_QUERY_VARIANTS = 3
MAX_HYDE_CHARS = 1200


_openai_client: Optional[OpenAI] = None


def _client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def _is_enabled(flag_name: str, default: bool) -> bool:
    return bool(getattr(settings, flag_name, default))


_MULTI_QUERY_SYSTEM = (
    "You rephrase compliance-code search queries. Produce alternative "
    "phrasings that a building code or standard might use."
)

_MULTI_QUERY_USER = """Original question: {query}

Generate {n} alternative phrasings that a compliance document (e.g. CIBSE, DBC, EN 81) might use to describe the same topic. Use synonyms a building services engineer would recognise. Keep each variant under 18 words.

Return JSON: {{"variants": ["...", "..."]}}
Return only NEW phrasings — do not repeat the original."""


def expand_multi_query(
    query: str,
    *,
    n: int = MAX_QUERY_VARIANTS,
    enabled: Optional[bool] = None,
) -> List[str]:
    if enabled is None:
        enabled = _is_enabled("MULTI_QUERY_ENABLED", default=True)
    if not enabled or not query.strip():
        return [query]

    n = max(1, min(n, MAX_QUERY_VARIANTS))
    try:
        resp = _client().chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": _MULTI_QUERY_SYSTEM},
                {"role": "user", "content": _MULTI_QUERY_USER.format(query=query, n=n)},
            ],
            temperature=0,
            max_completion_tokens=240,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        variants = parsed.get("variants") or []
        cleaned: List[str] = [query]
        for v in variants:
            if not isinstance(v, str):
                continue
            v_clean = v.strip()
            if not v_clean or v_clean.lower() == query.strip().lower():
                continue
            if v_clean in cleaned:
                continue
            cleaned.append(v_clean)
            if len(cleaned) >= n + 1:
                break
        return cleaned
    except Exception as e:
        logger.warning("multi-query expansion failed: %s", e)
        return [query]


_META_REGEX = re.compile(
    r"^\s*(?:"
    r"what\s+(?:does|do(?:es)?)\s+(?:\w+\s+){1,6}cover\b|"
    r"what(?:'?s|\s+is)\s+(?:this\s+(?:doc|document|guide)|the\s+(?:doc|document|guide))\s+(?:about|for|cover)|"
    r"what(?:'?s|\s+is)\s+in\s+(?:this\s+)?(?:doc|document|guide|cibse|dbc|en\s*81)|"
    r"list\s+(?:the\s+)?(?:chapters?|sections?|contents?)|"
    r"overview\s+of\s+(?:this\s+|the\s+)?(?:doc|document|guide|cibse|dbc|chapter)|"
    r"summary\s+of\s+(?:this\s+|the\s+)?(?:doc|document|guide|cibse|dbc|chapter)|"
    r"table\s+of\s+contents|"
    r"what\s+(?:topics|sections|chapters)\s+(?:does|are|exist)"
    r")",
    re.IGNORECASE,
)


_META_CLASSIFIER_SYSTEM = (
    "You classify whether a user query is a META question about a compliance "
    "document (its scope, chapters, glossary, purpose) versus a specific "
    "technical question about its content."
)

_META_CLASSIFIER_USER = """Query: {query}

Is this a META question about the document itself (its scope, structure, chapter list, glossary, definitions, what it covers in general) — NOT a question about a specific value, rule, calculation, or section?

Return JSON: {{"is_meta": true|false}}"""


def is_meta_question(
    query: str,
    *,
    enabled: Optional[bool] = None,
) -> bool:
    if enabled is None:
        enabled = _is_enabled("META_QUESTION_DETECTION_ENABLED", default=True)
    if not enabled or not query.strip():
        return False

    if _META_REGEX.search(query):
        return True

    if re.search(
        r"\b(section|table|figure|clause|appendix|chapter)\s+[A-Z\d]",
        query,
        re.IGNORECASE,
    ):
        return False
    if re.search(r"\d", query):
        return False

    try:
        resp = _client().chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": _META_CLASSIFIER_SYSTEM},
                {"role": "user", "content": _META_CLASSIFIER_USER.format(query=query)},
            ],
            temperature=0,
            max_completion_tokens=20,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        return bool(parsed.get("is_meta"))
    except Exception as e:
        logger.warning("meta-question classifier failed: %s", e)
        return False


_HYDE_SYSTEM = (
    "You write a short paragraph as if it were copied from a building-code "
    "compliance document, answering a user's question. The paragraph will "
    "be embedded and used as a retrieval query; it does NOT need to be "
    "factually accurate, just realistic in style and vocabulary."
)

_HYDE_USER = """Question: {query}

Write a single paragraph (4-6 sentences) that *would* answer this question if it appeared in a compliance document for vertical transportation in buildings (CIBSE / DBC / EN 81 style). Use the vocabulary, units, and tone such a document would use. Do not preface or annotate — return ONLY the paragraph."""


def hypothetical_doc(
    query: str,
    *,
    enabled: Optional[bool] = None,
) -> Optional[str]:
    if enabled is None:
        enabled = _is_enabled("HYDE_ENABLED", default=False)
    if not enabled or not query.strip():
        return None

    try:
        resp = _client().chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": _HYDE_USER.format(query=query)},
            ],
            temperature=0.3,
            max_completion_tokens=350,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text[:MAX_HYDE_CHARS] if text else None
    except Exception as e:
        logger.warning("HyDE generation failed: %s", e)
        return None


def should_apply_hyde(query: str) -> bool:
    q = query.strip()
    if not q:
        return False
    if len(q.split()) > 25:
        return True
    if q.count("?") >= 2:
        return True
    if " and " in q.lower() or " versus " in q.lower() or " vs " in q.lower():
        return True
    return False
