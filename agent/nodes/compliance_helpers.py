"""
Retrieval helpers shared by the Code Charlie compliance RAG node.

Ported from KARR-AI's worker/agent/nodes/compliance.py with the per-report
`compliance_rag_node` and `run_react_agent` stripped out — Code Charlie
has its own ReAct loop in agent/nodes/compliance.py and only needs the
pure search / lookup / formatting helpers from this module.

Includes:
- Query resolution with chat history (resolve_query_with_context)
- Query decomposition for complex multi-part questions (decompose_complex_query)
- Hybrid search RPC + fallback vector-only search (hybrid_search_compliance)
- Section / figure / table direct lookup (get_compliance_section)
- Deterministic chart-grid + curve lookups (lookup_chart)
- Cross-reference extraction (extract_cross_references)
- Search-result formatters (format_search_results, format_section_results)
- Source dedupe + frontend formatting (deduplicate_sources, format_sources_for_frontend)
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

from openai import OpenAI

from core.config import settings
from core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.65
TOP_K = 8
MAX_AGENT_ITERATIONS = 6

_openai_client = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


# =============================================================================
# Query Resolution with Chat History
# =============================================================================

def format_history_for_context(messages: List[Dict[str, Any]], max_messages: int = 6) -> str:
    history_parts = []
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

    for msg in recent_messages:
        if hasattr(msg, 'content'):
            role = getattr(msg, 'type', 'user')
            if role == 'human':
                role = 'User'
            elif role == 'ai':
                role = 'Assistant'
            content = msg.content
        else:
            role = msg.get("role", "user").capitalize()
            if role == "User":
                role = "User"
            elif role == "Assistant":
                role = "Assistant"
            content = msg.get("content", "")

        if content:
            if len(content) > 300:
                content = content[:300] + "..."
            history_parts.append(f"{role}: {content}")

    return "\n".join(history_parts)


def resolve_query_with_context(current_question: str, history: List[Dict[str, Any]]) -> str:
    if not history:
        return current_question

    ambiguous_patterns = [
        r'\b(it|they|them|this|that|these|those)\b',
        r'\b(what about|how about|and|also)\b',
        r'^(how many|how much|what|which|where)\b(?!.*\b(elevator|lift|escalator|building|floor|code|dbc|requirement)\b)',
    ]

    is_ambiguous = any(re.search(p, current_question.lower()) for p in ambiguous_patterns)

    if not is_ambiguous and len(current_question.split()) > 5:
        return current_question

    try:
        client = get_openai_client()
        formatted_history = format_history_for_context(history)

        response = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[{
                "role": "system",
                "content": """Rewrite the question to be fully self-contained using context from the conversation history.
Include any implicit context (building type, specifications, numbers) from the history.

Examples:
- History: "User: What are elevator specs for 3-star hotels?"
- Current: "What about service elevators?"
- Rewritten: "What are the service elevator specifications for a 3-star hotel?"

- History: "User: How many elevators for a 25-floor residential?"
- Current: "What if there are 3 boarding floors?"
- Rewritten: "How many elevators are required for a 25-floor residential building with 3 boarding floors?"

If the question is already self-contained, return it unchanged.
Return ONLY the rewritten question, nothing else."""
            }, {
                "role": "user",
                "content": f"Conversation history:\n{formatted_history}\n\nCurrent question: {current_question}"
            }],
            max_completion_tokens=200,
            temperature=0
        )
        resolved = response.choices[0].message.content.strip()
        logger.info(f"Query resolved: '{current_question}' → '{resolved}'")
        return resolved

    except Exception as e:
        logger.warning(f"Query resolution failed: {e}, using original")
        return current_question


# =============================================================================
# Query Decomposition
# =============================================================================

def decompose_complex_query(question: str, project_context: Dict[str, Any]) -> List[str]:
    complexity_indicators = [
        ("floors" in question.lower() and ("population" in question.lower() or "people" in question.lower())),
        ("boarding" in question.lower() and "floor" in question.lower()),
        bool(re.search(r'(figure|table).*and.*(figure|table)', question, re.I)),
        ("how many" in question.lower() and "elevator" in question.lower()),
        ("sum" in question.lower() or "total" in question.lower()),
    ]

    if not any(complexity_indicators):
        return [question]

    try:
        client = get_openai_client()
        context_str = json.dumps(project_context) if project_context else "{}"

        response = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[{
                "role": "system",
                "content": """Decompose complex compliance questions into sub-queries for a building code search.

For elevator count calculations in DBC, typical sub-queries are:
1. Population/capacity chart lookup (Figure D.13, D.15, D.17, D.20)
2. Boarding floor adjustment (Figure D.14, D.14a, D.21)
3. Specification requirements (Table D.6, D.14, D.15)

Return JSON: {"queries": ["query1", "query2"]}
If the question is simple, return {"queries": ["original question"]}
Maximum 3 sub-queries."""
            }, {
                "role": "user",
                "content": f"Context: {context_str}\n\nQuestion: {question}"
            }],
            max_completion_tokens=300,
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        queries = result.get("queries", [question])

        if queries:
            logger.info(f"Query decomposed into {len(queries)} sub-queries")
            return queries[:3]

    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}")

    return [question]


# =============================================================================
# Hybrid Search
# =============================================================================

def embed_text(text: str) -> List[float]:
    """
    Generate embedding vector using settings.COMPLIANCE_EMBED_MODEL
    (text-embedding-3-large, 3072 dims) — the model the compliance_embeddings
    table is built against.
    """
    client = get_openai_client()
    response = client.embeddings.create(
        model=settings.COMPLIANCE_EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def hybrid_search_compliance(
    query: str,
    top_k: int = TOP_K,
    compliance_code: Optional[str] = None,
    document_name: Optional[str] = None,
    chunk_types: Optional[List[str]] = None,
    query_embedding_override: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Hybrid (vector + full-text) search via Supabase RPC, with vector-only fallback."""
    try:
        query_embedding = (
            query_embedding_override
            if query_embedding_override is not None
            else embed_text(query)
        )
        supabase = get_supabase_client()

        try:
            params: Dict[str, Any] = {
                "query_text": query,
                "query_embedding": query_embedding,
                "match_count": top_k,
            }
            if compliance_code:
                params["filter_compliance_code"] = compliance_code
            if document_name:
                params["filter_document_name"] = document_name
            if chunk_types:
                params["filter_chunk_types"] = chunk_types

            result = supabase.rpc("hybrid_search_compliance", params).execute()

            if result.data:
                sorted_results = sorted(
                    result.data,
                    key=lambda x: x.get("combined_score", 0),
                    reverse=True
                )
                return sorted_results[:top_k]

        except Exception as rpc_error:
            logger.warning(f"Hybrid search RPC failed, falling back to vector search: {rpc_error}")

        fallback_params: Dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_threshold": 0.4,
            "match_count": top_k,
        }
        if compliance_code:
            fallback_params["filter_compliance_code"] = compliance_code
        if document_name:
            fallback_params["filter_document_name"] = document_name

        result = supabase.rpc("match_compliance_embeddings", fallback_params).execute()

        return result.data if result.data else []

    except Exception as e:
        logger.error(f"Error in hybrid search: {e}", exc_info=True)
        return []


def get_compliance_section(section_num: str, doc_name: str = None) -> List[Dict[str, Any]]:
    """Retrieve a specific section by number ('D.8.8.2', 'Figure D.13', 'Table D.5')."""
    try:
        supabase = get_supabase_client()

        try:
            params = {"section_num": section_num}
            if doc_name:
                params["doc_name"] = doc_name

            result = supabase.rpc("get_compliance_section", params).execute()

            if result.data:
                return result.data

        except Exception as rpc_error:
            logger.warning(f"Section lookup RPC failed: {rpc_error}")

        query = supabase.table("compliance_embeddings").select(
            "id, section_number, section_title, clause_text, document_name, metadata"
        )

        clean_num = section_num.strip()
        if clean_num.lower().startswith("figure "):
            clean_num = clean_num[7:].strip()
        elif clean_num.lower().startswith("table "):
            clean_num = clean_num[6:].strip()

        query = query.or_(
            f"section_number.eq.{clean_num},"
            f"section_number.ilike.{clean_num}.%,"
            f"section_title.ilike.%{clean_num}%"
        )

        if doc_name:
            query = query.eq("document_name", doc_name)

        result = query.limit(5).execute()
        return result.data if result.data else []

    except Exception as e:
        logger.error(f"Error in section lookup: {e}", exc_info=True)
        return []


# =============================================================================
# Lookup-chart (deterministic chart-grid + curve lookups)
# =============================================================================

_BUCKET_LE_RE = re.compile(r"^\s*(?:≤|<=)\s*([\d,\.]+)\s*$")
_BUCKET_LT_RE = re.compile(r"^\s*<\s*([\d,\.]+)\s*$")
_BUCKET_GE_RE = re.compile(r"^\s*(?:≥|>=)\s*([\d,\.]+)\s*$")
_BUCKET_GT_RE = re.compile(r"^\s*>\s*([\d,\.]+)\s*$")
_BUCKET_RANGE_RE = re.compile(r"^\s*([\d,\.]+)\s*[-–—]\s*([\d,\.]+)\s*$")


def _parse_number(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").strip())
    except Exception:
        return None


def _bucket_match(value: float, bucket_label: str) -> bool:
    s = (bucket_label or "").strip()
    if not s:
        return False

    m = _BUCKET_LE_RE.match(s)
    if m:
        hi = _parse_number(m.group(1))
        return hi is not None and value <= hi

    m = _BUCKET_LT_RE.match(s)
    if m:
        hi = _parse_number(m.group(1))
        return hi is not None and value < hi

    m = _BUCKET_GE_RE.match(s)
    if m:
        lo = _parse_number(m.group(1))
        return lo is not None and value >= lo

    m = _BUCKET_GT_RE.match(s)
    if m:
        lo = _parse_number(m.group(1))
        return lo is not None and value > lo

    m = _BUCKET_RANGE_RE.match(s)
    if m:
        lo = _parse_number(m.group(1))
        hi = _parse_number(m.group(2))
        return lo is not None and hi is not None and lo <= value <= hi

    n = _parse_number(s)
    return n is not None and value == n


def _resolve_bucket_index(value: float, buckets: List[str]) -> Optional[int]:
    for i, label in enumerate(buckets):
        if _bucket_match(value, label):
            return i
    return None


def _interpolate_curve(points: List[List[float]], x: float) -> Optional[float]:
    if not points:
        return None
    pts = sorted([(float(p[0]), float(p[1])) for p in points], key=lambda p: p[0])
    if x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


def lookup_chart(
    figure_id: str,
    inputs: Dict[str, Any],
    compliance_code: Optional[str] = None,
    document_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministically resolve a figure-grid / curve question."""
    try:
        supabase = get_supabase_client()

        clean_id = (figure_id or "").strip()
        if clean_id.lower().startswith("figure "):
            clean_id = clean_id[7:].strip()

        query = supabase.table("compliance_embeddings").select(
            "figure_id, structured_data, document_name, section_number, chunk_type"
        ).eq("figure_id", clean_id)

        if compliance_code:
            query = query.eq("compliance_code", compliance_code)
        if document_name:
            query = query.eq("document_name", document_name)

        rows = query.limit(5).execute().data or []
        rows = [r for r in rows if r.get("structured_data")]

        if not rows:
            return {
                "ok": False,
                "figure_id": clean_id,
                "error": f"No structured data found for figure_id={clean_id}",
            }

        rows.sort(key=lambda r: 0 if r.get("chunk_type") in ("lookup_chart", "line_chart") else 1)
        row = rows[0]
        data = row["structured_data"] or {}
        kind = data.get("type")

        if kind == "lookup_chart":
            x_axis = data.get("x_axis") or {}
            y_axis = data.get("y_axis") or {}
            values = data.get("values") or []

            x_buckets = x_axis.get("buckets") or []
            y_buckets = y_axis.get("buckets") or []

            x_label = (x_axis.get("label") or "").strip().lower()
            y_label = (y_axis.get("label") or "").strip().lower()

            x_value = _pick_input_for_axis(x_label, inputs)
            y_value = _pick_input_for_axis(y_label, inputs)

            if x_value is None or y_value is None:
                return {
                    "ok": False,
                    "figure_id": clean_id,
                    "error": (
                        f"Could not map inputs {list(inputs.keys())} to axes "
                        f"x='{x_axis.get('label')}', y='{y_axis.get('label')}'."
                    ),
                }

            x_idx = _resolve_bucket_index(float(x_value), x_buckets)
            y_idx = _resolve_bucket_index(float(y_value), y_buckets)
            if x_idx is None or y_idx is None:
                return {
                    "ok": False,
                    "figure_id": clean_id,
                    "error": (
                        f"Input value(s) outside chart range: "
                        f"x={x_value} (buckets={x_buckets[:3]}...), "
                        f"y={y_value} (buckets={y_buckets[:3]}...)."
                    ),
                }

            try:
                cell = values[y_idx][x_idx]
            except Exception:
                cell = None

            return {
                "ok": cell is not None,
                "figure_id": clean_id,
                "value": cell,
                "unit": data.get("unit"),
                "matched_axes": {
                    x_axis.get("label"): x_buckets[x_idx],
                    y_axis.get("label"): y_buckets[y_idx],
                },
                "source": {
                    "document_name": row.get("document_name"),
                    "section_number": row.get("section_number"),
                },
                "error": None if cell is not None else "Empty cell at matched buckets",
            }

        if kind == "line_chart":
            curves = data.get("curves") or {}
            curve_name = inputs.get("curve")
            if not curve_name or curve_name not in curves:
                if len(curves) == 1:
                    curve_name = next(iter(curves.keys()))
                else:
                    return {
                        "ok": False,
                        "figure_id": clean_id,
                        "error": (
                            f"Specify a curve name in `inputs.curve`. "
                            f"Available: {list(curves.keys())}"
                        ),
                    }

            x_axis = data.get("x_axis") or {}
            x_value = _pick_input_for_axis((x_axis.get("label") or "").lower(), inputs)
            if x_value is None:
                return {
                    "ok": False,
                    "figure_id": clean_id,
                    "error": f"Missing input for x-axis '{x_axis.get('label')}'.",
                }

            y = _interpolate_curve(curves[curve_name], float(x_value))
            return {
                "ok": y is not None,
                "figure_id": clean_id,
                "value": y,
                "unit": (data.get("y_axis") or {}).get("unit"),
                "matched_axes": {x_axis.get("label"): x_value, "curve": curve_name},
                "source": {
                    "document_name": row.get("document_name"),
                    "section_number": row.get("section_number"),
                },
                "error": None if y is not None else "Input x is outside the curve range.",
            }

        return {
            "ok": False,
            "figure_id": clean_id,
            "error": f"Unsupported structured_data.type='{kind}' for lookup_chart tool.",
        }

    except Exception as e:
        logger.error("lookup_chart failed: %s", e, exc_info=True)
        return {"ok": False, "figure_id": figure_id, "error": str(e)}


def _pick_input_for_axis(axis_label_lower: str, inputs: Dict[str, Any]) -> Optional[float]:
    if not axis_label_lower:
        return None

    for key, val in inputs.items():
        norm_key = key.replace("_", " ").lower()
        if val is None:
            continue
        if norm_key == axis_label_lower:
            return float(val)
        if norm_key in axis_label_lower or axis_label_lower in norm_key:
            return float(val)

    axis_tokens = set(axis_label_lower.split())
    for key, val in inputs.items():
        if val is None:
            continue
        key_tokens = set(key.replace("_", " ").lower().split())
        if axis_tokens & key_tokens:
            try:
                return float(val)
            except Exception:
                return None
    return None


# =============================================================================
# Cross-Reference Detection
# =============================================================================

def extract_cross_references(chunk_text: str) -> List[str]:
    patterns = [
        r'see\s+(?:the\s+)?(?:Table|Figure|Section)\s+([A-Z]\.\d+(?:\.\d+)*[a-z]?)',
        r'sum\s+of\s+(?:the\s+)?(?:numbers?\s+)?(?:obtained\s+from\s+)?(?:Figure|Table)\s+([A-Z]\.\d+[a-z]?)\s+and\s+(?:Figure|Table)\s+([A-Z]\.\d+[a-z]?)',
        r'in\s+addition\s+to\s+(?:Section\s+)?([A-Z]\.\d+(?:\.\d+)*)',
        r'refer\s+to\s+(?:Table|Figure|Section)\s+([A-Z]\.\d+(?:\.\d+)*[a-z]?)',
        r'(?:Figure|Table)\s+([A-Z]\.\d+[a-z]?)',
        r'Section\s+([A-Z]\.\d+(?:\.\d+)+)',
    ]

    refs = set()
    for pattern in patterns:
        matches = re.findall(pattern, chunk_text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                for item in match:
                    if item and re.match(r'^[A-Z]\.\d+', item, re.I):
                        refs.add(item.upper())
            elif match and re.match(r'^[A-Z]\.\d+', match, re.I):
                refs.add(match.upper())

    return list(refs)


# =============================================================================
# Result formatters
# =============================================================================

_ROLE_LABEL = {
    "primary":          "PRIMARY HIT",
    "adjacent_before":  "context (preceding chunk)",
    "adjacent_after":   "context (following chunk)",
    "sibling":          "context (sibling section)",
    "parent":           "context (parent section)",
}


def format_search_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No results found."

    primaries = [r for r in results if r.get("_context_role", "primary") == "primary"]
    contexts  = [r for r in results if r.get("_context_role", "primary") != "primary"]

    primaries = primaries[:5]

    parts: List[str] = []

    PRIMARY_TRUNC = 25000
    for i, r in enumerate(primaries, 1):
        section = r.get("section_number", "Unknown")
        title = r.get("section_title", "")
        text = r.get("clause_text", "")
        doc = r.get("document_name", "Unknown")
        similarity = r.get("similarity", r.get("combined_score", 0))

        parts.append(f"[{i}] PRIMARY HIT — {doc} - Section {section}")
        if title:
            parts.append(f"    Title: {title}")
        parts.append(f"    Similarity: {similarity:.2%}")
        if len(text) > PRIMARY_TRUNC:
            text = text[:PRIMARY_TRUNC] + "..."
        parts.append(f"    Content: {text}")
        parts.append("")

    if contexts:
        parts.append("--- Surrounding context (use to inform your answer; cite the PRIMARY HITS) ---")
        for r in contexts:
            role = r.get("_context_role", "context")
            section = r.get("section_number", "Unknown")
            title = r.get("section_title", "")
            text = r.get("clause_text", "")
            doc = r.get("document_name", "Unknown")
            label = _ROLE_LABEL.get(role, role)
            header = f"[{label}] {doc} - Section {section}"
            if title:
                header += f" — {title}"
            parts.append(header)
            if len(text) > 350:
                text = text[:350] + "..."
            parts.append(f"    {text}")
            parts.append("")

    return "\n".join(parts)


def format_section_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "Section not found."

    parts = []
    for r in results:
        section = r.get("section_number", "Unknown")
        title = r.get("section_title", "")
        text = r.get("clause_text", "")
        doc = r.get("document_name", "Unknown")

        parts.append(f"[{doc} - Section {section}]")
        if title:
            parts.append(f"Title: {title}")
        parts.append(f"Content:\n{text}")
        parts.append("")

    return "\n".join(parts)


def deduplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_ids = set()
    seen_sections = set()
    unique = []

    for s in sources:
        sid = s.get("id")
        section = s.get("section_number", "")

        if sid and sid in seen_ids:
            continue

        if section and section in seen_sections:
            continue

        if sid:
            seen_ids.add(sid)
        if section:
            seen_sections.add(section)

        unique.append(s)

    return unique


def format_sources_for_frontend(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for s in sources[:8]:
        text = s.get("clause_text", "")
        formatted.append({
            "section": s.get("section_number", ""),
            "title": s.get("section_title", ""),
            "text": text[:200] + "..." if len(text) > 200 else text,
            "document": s.get("document_name", ""),
            "similarity": s.get("similarity", s.get("combined_score", 0))
        })
    return formatted
