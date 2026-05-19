"""
code_charlie_compliance_rag node — the workhorse of Code Charlie.

Takes scope from state.scope (compliance_code AND document_name), runs a
ReAct loop with these tools:
  - search(query, scope="primary"|"all")
  - get_section(section_number)
  - lookup_chart(figure_id, inputs)
  - widen_scope(reason)

LLM rerank pass over top-30 hits keeps the 5 best. Citation verification
flags hard hallucinations. Confidence guard kicks in when nothing came back.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from core.config import settings
from agent.state import CodeCharlieState
from agent.nodes.compliance_helpers import (
    MAX_AGENT_ITERATIONS,
    decompose_complex_query,
    deduplicate_sources,
    extract_cross_references,
    format_search_results,
    format_section_results,
    format_sources_for_frontend,
    get_compliance_section,
    hybrid_search_compliance,
    lookup_chart,
    resolve_query_with_context,
)
from agent.nodes.context_expansion import expand_with_context
from agent.nodes.query_rewriting import (
    META_CHUNK_TYPES,
    expand_multi_query,
    hypothetical_doc,
    is_meta_question,
    should_apply_hyde,
)
from core.supabase_client import get_supabase_client
from agent.messages import add_assistant_message, get_last_user_message

logger = logging.getLogger(__name__)


RERANK_FETCH_K = 30
RERANK_KEEP_K = 5
RERANK_PREFILTER_SIMILARITY = 0.45
LOW_CONFIDENCE_THRESHOLD = 0.55
MAX_TOTAL_LLM_CALLS = 12
RECOVERY_MIN_TOP_SCORE = 0.48
RECOVERY_MAX_QUERIES = 12

_openai_client: Optional[OpenAI] = None


def _client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


@dataclass
class GeneralAgentResult:
    answer: str
    sources: List[Dict[str, Any]]
    iterations: int
    incomplete: bool = False
    scope_widened: bool = False
    widen_reason: Optional[str] = None
    dropped_claims: List[str] = field(default_factory=list)


def _apply_section_diversity(
    chunks: List[Dict[str, Any]],
    *,
    max_per_section: int = 2,
) -> List[Dict[str, Any]]:
    if not chunks:
        return chunks
    seen: Dict[Any, int] = {}
    head: List[Dict[str, Any]] = []
    tail: List[Dict[str, Any]] = []
    for c in chunks:
        key = (c.get("document_name"), c.get("section_number"))
        if seen.get(key, 0) < max_per_section:
            head.append(c)
            seen[key] = seen.get(key, 0) + 1
        else:
            tail.append(c)
    return head + tail


def _candidate_similarity(c: Dict[str, Any]) -> float:
    for key in ("combined_score", "vector_score", "similarity"):
        val = c.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


_RERANK_SYSTEM = (
    "You are a strict relevance judge for compliance-code retrieval. "
    "Score each candidate on how directly it answers the question."
)

_RERANK_USER_TEMPLATE = """Question: {query}

Score every candidate 0-100 by how directly it answers the question. Use the rubric:
  90-100: Directly contains the answer (specific number, rule, or definition the user asked for)
  70-89:  Contains the relevant section, answer must be inferred
  40-69:  Adjacent / supporting context (defines a term used; refers to the relevant section)
  10-39:  Same topic, different sub-question
  0-9:    Irrelevant

Candidates:
{candidates}

Return JSON: {{"scores": [{{"index": 0, "score": 95}}, ...]}}
Score every candidate. Use only the indices shown."""


def llm_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = RERANK_KEEP_K,
) -> List[Dict[str, Any]]:
    if len(candidates) <= top_n:
        return candidates

    survivors: List[Dict[str, Any]] = [
        c for c in candidates
        if _candidate_similarity(c) >= RERANK_PREFILTER_SIMILARITY
    ]
    if len(survivors) < top_n:
        survivors = candidates[: max(top_n, len(survivors))]

    if len(survivors) <= top_n:
        return survivors[:top_n]

    try:
        lines = []
        for idx, c in enumerate(survivors):
            doc = c.get("document_name", "?")
            section = c.get("section_number") or "?"
            title = c.get("section_title") or ""
            text = (c.get("clause_text") or "")[:400]
            lines.append(
                f"[{idx}] {doc} — Section {section}"
                + (f" ({title})" if title else "")
                + f"\n    {text}"
            )

        prompt = _RERANK_USER_TEMPLATE.format(
            query=query,
            candidates="\n".join(lines),
        )

        response = _client().chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=400,
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        scores = parsed.get("scores", []) or []

        score_by_idx: Dict[int, int] = {}
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            sc = entry.get("score")
            if not isinstance(idx, int) or not (0 <= idx < len(survivors)):
                continue
            try:
                score_by_idx[idx] = int(sc)
            except (TypeError, ValueError):
                continue

        if not score_by_idx:
            return survivors[:top_n]

        ranked_idxs = sorted(
            range(len(survivors)),
            key=lambda i: (
                -score_by_idx.get(i, -1),
                -_candidate_similarity(survivors[i]),
            ),
        )
        return [survivors[i] for i in ranked_idxs[:top_n]]

    except Exception as e:
        logger.warning(f"LLM rerank failed: {e}; using similarity order")
        return survivors[:top_n]


_CITATION_VERIFY_SYSTEM = """You are a strict but conservative citation auditor for a compliance answer.

You will receive:
  - the draft answer (markdown) with parenthetical citations like
    "(DBC Part D, Section D.8.8.2)" or "(CIBSE Guide D 2025, Table 4.6)"
  - the source chunks the answer was built from

Your job is to flag ONLY clear hallucinations. Default to NOT flagging.

A claim is "unsupported" ONLY IF ALL of these are true:
  1. The claim cites a specific document + section/table/figure.
  2. The cited document does NOT appear anywhere in the source chunks.
  3. There is no plausible paraphrase of the claim in any source chunk.

DO NOT flag:
  - Paraphrases of source content (even loose ones)
  - Claims that mention a table/figure number in passing without citing it
    as the sole source (e.g. "specified in Table D.35" is fine as long as
    the cited section actually appears in the sources)
  - Specific numbers, even if you can't verify them word-for-word —
    excerpts are truncated and may omit numeric values that appear elsewhere
    in the same section
  - Claims about the cited document's general subject matter
  - Cross-references to other sections / tables / figures

Return JSON: {
  "unsupported_claims": [
     { "snippet":  "<short quote from draft, <=160 chars>",
       "citation": "<citation as written>",
       "reason":   "<one-sentence reason>" }
  ]
}

If in doubt, return {"unsupported_claims": []}. False positives are worse
than false negatives here.
"""


def verify_citations(
    answer: str,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not answer or not sources:
        return {"verified_answer": answer, "dropped_claims": []}

    cited_docs = set()
    for m in re.finditer(r"\(([^)]+?),\s*[^)]+\)", answer):
        cited_docs.add(m.group(1).strip().lower())
    source_docs = {(s.get("document_name") or "").strip().lower() for s in sources}
    if cited_docs and source_docs and (cited_docs & source_docs):
        return {"verified_answer": answer, "dropped_claims": []}

    try:
        sources_blob = []
        for i, s in enumerate(sources[:12]):
            sources_blob.append(
                f"[{i}] {s.get('document_name','?')} — Section "
                f"{s.get('section_number') or '?'}"
                + (f" ({s.get('section_title')})" if s.get('section_title') else "")
                + f"\n    {(s.get('clause_text') or '')[:500]}"
            )

        response = _client().chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": _CITATION_VERIFY_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Draft answer:\n{answer}\n\n"
                        f"Source chunks:\n" + "\n".join(sources_blob)
                    ),
                },
            ],
            max_completion_tokens=400,
            temperature=0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        unsupported = parsed.get("unsupported_claims", []) or []

        dropped: List[str] = []
        for c in unsupported:
            snippet = (c.get("snippet") or "").strip()
            if not snippet:
                continue
            dropped.append(
                f"{snippet} ({c.get('citation','?')}) — {c.get('reason','')}"
            )
        return {"verified_answer": answer, "dropped_claims": dropped}

    except Exception as e:
        logger.warning(f"Citation verification failed: {e}")
        return {"verified_answer": answer, "dropped_claims": []}


_CODE_CHARLIE_REACT_SYSTEM_PROMPT = """You are Code Charlie, a research assistant for building codes and compliance standards.

The user is researching one or more of these codes:
DBC, CIBSE, EN81, BCO, ASME/ADA/IBC, HTM, ISO, DoH (Abu Dhabi), BMU, CSI,
Machinery directive, plus reference handbooks under "Other".

## Scope for this turn
Primary scope: {scope_summary}

## Tools
1. **search(query, scope="primary")** — hybrid (semantic + keyword) search.
   Pass `scope="primary"` to stay inside the locked-in code (default).
   Pass `scope="all"` ONLY after calling `widen_scope` first.
2. **get_section(section_number)** — fetch a specific section / figure /
   table by number (e.g. "D.8.8.2", "Figure D.13", "Table 4.6").
3. **lookup_chart(figure_id, inputs)** — deterministic value lookup against
   a chart's structured data. Prefer this over reading a chart in prose.
4. **widen_scope(reason)** — drop the primary-scope filter for the rest of
   this turn. ONLY call this when:
     (a) the user explicitly asks to compare codes ("compare DBC and CIBSE");
     (b) the user explicitly says "any code" / "all docs";
     (c) your first primary-scope search returned no useful hits.
   After widen_scope, your next `search` call should use `scope="all"`.

## Citation rules — non-negotiable
- Every factual claim MUST end with one citation in the form
  `(<Document>, <Section/Figure/Table>)`. Examples:
    `(DBC Part D, Section D.8.8.2)`
    `(CIBSE Guide D 2025, Table 4.6)`
    `(EN 81-20, Clause 5.3.4)`
- ONE citation per claim. Do not stitch two sources into the same claim.
- If two sources disagree, report both — name each source, do not average.
- If you cannot find a fact in the retrieved chunks, say so plainly. Never
  invent section numbers, figure IDs, or numeric values.
- When you finish the answer, briefly list the documents you used.

## Multiple values from different sources
Before giving a numeric value, compare the retrieved source titles, section
hierarchy, table captions, and row labels such as "Application", "Usage",
"Class", "Building type", or similar fields.
- If two values come from different sections/tables or rows, clearly say
  which value comes from which source and what case that source applies to.
- If the sources appear to cover different applications, do not choose one
  silently and do not merge their numbers into one configuration. Give the
  likely general case first only when it is clear, then list the special-case
  values with their source labels.
- If the user specified one application, answer that application and explain
  that other retrieved values are for different cases if they might confuse
  the answer.
- If you cannot tell whether two values apply to the same case, say the
  sources give different values and ask for the missing context. Never present
  uncertain values as one definitive requirement.

## When you don't know
If you can't find a confident answer in the primary scope, either:
  - call widen_scope("primary scope had no relevant hits") and try once
    more across all documents, OR
  - tell the user clearly that you didn't find it, and suggest they
    rephrase or ask about a different code.

Stay focused. Don't write essays. Aim for tight, well-cited answers."""


def _scope_summary(scope: Optional[Dict[str, Any]]) -> str:
    if not scope:
        return "none locked in — searching across every ingested compliance document."
    code = scope.get("compliance_code")
    docs = _scope_document_names(scope)
    if len(docs) == 1:
        return f"document = {docs[0]} (code: {code or '?'})."
    if len(docs) > 1:
        return (
            "documents = "
            + "; ".join(docs)
            + f" (code: {code or 'multiple / mixed'})."
        )
    if code:
        return f"compliance code = {code} (any document in this code)."
    return "none locked in — searching across every ingested compliance document."


def _scope_document_names(scope: Optional[Dict[str, Any]]) -> List[str]:
    if not scope:
        return []

    raw = scope.get("document_names") or []
    if isinstance(raw, str):
        raw_docs = [raw]
    elif isinstance(raw, list):
        raw_docs = [d for d in raw if isinstance(d, str)]
    else:
        raw_docs = []

    single = scope.get("document_name")
    if isinstance(single, str) and single:
        raw_docs.append(single)

    docs: List[str] = []
    for doc in raw_docs:
        cleaned = doc.strip()
        if cleaned and cleaned not in docs:
            docs.append(cleaned)
    return docs


def _result_score(row: Dict[str, Any]) -> float:
    score = row.get("combined_score", row.get("similarity", 0)) or 0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _dedupe_search_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("document_name"),
            row.get("section_number"),
            row.get("chunk_index"),
            row.get("clause_text"),
        )
        existing = deduped.get(key)
        if existing is None or _result_score(row) > _result_score(existing):
            deduped[key] = row
    return sorted(deduped.values(), key=_result_score, reverse=True)


_RECOVERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "code",
    "codes", "come", "comes", "do", "does", "for", "from", "give",
    "guide", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "say", "says", "the", "there", "to", "under", "what", "when",
    "which", "with",
}

_RECOVERY_TERM_EXPANSIONS = {
    "requirement": ["guidance", "criteria", "provision"],
    "requirements": ["guidance", "criteria", "provisions"],
    "rule": ["requirement", "guidance", "criteria"],
    "rules": ["requirements", "guidance", "criteria"],
    "standard": ["recommended", "typical", "minimum"],
    "standards": ["recommended", "typical", "minimum"],
    "elevator": ["lift"],
    "elevators": ["lifts"],
    "lift": ["elevator"],
    "lifts": ["elevators"],
    "car": ["lift car", "cabin", "platform"],
    "cabin": ["car", "lift car"],
    "size": ["dimensions", "space requirements", "area"],
    "sizes": ["dimensions", "space requirements"],
    "door": ["entrance", "opening", "clear opening"],
    "doors": ["entrances", "openings", "clear openings"],
    "opening": ["entrance", "clear opening", "door width"],
    "openings": ["entrances", "clear openings"],
    "width": ["clear width", "opening width", "entrance width"],
    "distance": ["travel distance", "walking distance", "route distance"],
    "distances": ["travel distances", "walking distances", "route distances"],
    "center": ["centre"],
    "centers": ["centres"],
    "mall": ["shopping centre", "shopping center", "retail development"],
    "malls": ["shopping centres", "shopping centers", "retail developments"],
}


def _dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value).strip(" ?.,;:")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _strip_query_boilerplate(query: str) -> str:
    q = re.sub(r"\s+", " ", query).strip(" ?")
    q = re.sub(
        r"(?i)^\s*(?:what(?:'s|\s+is)?|what\s+does|what\s+do|tell\s+me|"
        r"can\s+you\s+tell\s+me|please\s+tell\s+me)\b",
        "",
        q,
    ).strip()
    q = re.sub(
        r"(?i)\b(?:when\s+it\s+comes\s+to|with\s+regard\s+to|in\s+relation\s+to|"
        r"regarding|about)\b",
        " ",
        q,
    )
    q = re.sub(r"(?i)\b(?:cibse|dbc|en\s*81|asme|ada|ibc|iso|htm|bco)\b", " ", q)
    q = re.sub(r"(?i)\bguide\s+[a-z]\b", " ", q)
    return re.sub(r"\s+", " ", q).strip(" ?")


def _keyword_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:,\d{3})*(?:\.\d+)?", query)
    kept = [
        token for token in tokens
        if token.lower() not in _RECOVERY_STOPWORDS and len(token) > 1
    ]
    return " ".join(kept[:12])


def _number_normalized_variants(query: str) -> List[str]:
    variants = []
    no_commas = re.sub(r"(?<=\d),(?=\d{3}\b)", "", query)
    if no_commas != query:
        variants.append(no_commas)

    def _add_commas(match: re.Match[str]) -> str:
        raw = match.group(0)
        if len(raw) < 4:
            return raw
        return f"{int(raw):,}"

    with_commas = re.sub(r"\b\d{4,}\b", _add_commas, query)
    if with_commas != query:
        variants.append(with_commas)
    return variants


def _term_expansion_variants(query: str) -> List[str]:
    variants: List[str] = []
    for term, replacements in _RECOVERY_TERM_EXPANSIONS.items():
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        if not pattern.search(query):
            continue
        for replacement in replacements[:2]:
            variants.append(pattern.sub(replacement, query, count=1))
    return variants


def _recovery_query_variants(query: str) -> List[str]:
    stripped = _strip_query_boilerplate(query)
    keyword = _keyword_query(stripped or query)
    base = stripped or keyword or query

    variants = [base, keyword]
    variants.extend(_number_normalized_variants(base))
    if keyword:
        variants.extend([
            f"{keyword} dimensions",
            f"{keyword} requirements",
            f"{keyword} guidance",
            f"{keyword} criteria",
            f"{keyword} table",
        ])
    variants.extend(_term_expansion_variants(base))

    original = query.strip().lower()
    return [
        v for v in _dedupe_strings(variants)
        if v.lower() != original
    ][:RECOVERY_MAX_QUERIES]


def _should_run_recovery(rows: List[Dict[str, Any]], top_k: int) -> bool:
    if not rows:
        return True
    return _result_score(rows[0]) < RECOVERY_MIN_TOP_SCORE


def _search_compliance_scope(
    query: str,
    top_k: int,
    compliance_code: Optional[str],
    document_names: List[str],
    *,
    chunk_types: Optional[List[str]] = None,
    apply_multi_query: bool = True,
    hyde_embedding: Optional[List[float]] = None,
    recovery_enabled: bool = True,
) -> List[Dict[str, Any]]:
    queries = [query]
    if apply_multi_query:
        queries = expand_multi_query(query)

    def _one_search(
        q: str,
        embed_override: Optional[List[float]],
    ) -> List[Dict[str, Any]]:
        if not document_names:
            return hybrid_search_compliance(
                q,
                top_k=top_k,
                compliance_code=compliance_code,
                document_name=None,
                chunk_types=chunk_types,
                query_embedding_override=embed_override,
            )
        per_doc_k = top_k if len(document_names) == 1 else max(
            RERANK_KEEP_K,
            (top_k + len(document_names) - 1) // len(document_names),
        )
        rows: List[Dict[str, Any]] = []
        for doc in document_names:
            rows.extend(
                hybrid_search_compliance(
                    q,
                    top_k=per_doc_k,
                    compliance_code=compliance_code,
                    document_name=doc,
                    chunk_types=chunk_types,
                    query_embedding_override=embed_override,
                )
            )
        return rows

    all_rows: List[Dict[str, Any]] = []

    if hyde_embedding is not None and queries:
        all_rows.extend(_one_search(queries[0], hyde_embedding))

    for q in queries:
        all_rows.extend(_one_search(q, None))

    primary_rows = _dedupe_search_results(all_rows)
    if (
        recovery_enabled
        and chunk_types is None
        and _should_run_recovery(primary_rows, top_k)
    ):
        existing_queries = {q.lower() for q in queries}
        recovery_queries = [
            q for q in _recovery_query_variants(query)
            if q.lower() not in existing_queries
        ]
        logger.info(
            "Primary search weak for %r; running %d recovery queries",
            query,
            len(recovery_queries),
        )
        for recovery_query in recovery_queries:
            recovered = _one_search(recovery_query, None)
            for row in recovered:
                metadata = row.get("metadata")
                if isinstance(metadata, dict):
                    metadata["retrieval_recovery_query"] = recovery_query
            all_rows.extend(recovered)

    return _dedupe_search_results(all_rows)


def _rerank_keep_count(document_names: List[str], candidate_count: int) -> int:
    if not document_names:
        return min(RERANK_KEEP_K, candidate_count)
    return min(max(RERANK_KEEP_K, len(document_names) * 3), 12, candidate_count)


def _ensure_document_coverage(
    ranked: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    document_names: List[str],
    max_results: int,
) -> List[Dict[str, Any]]:
    if len(document_names) <= 1:
        return ranked[:max_results]

    out = list(ranked)
    present = {r.get("document_name") for r in out}
    keys = {
        (
            r.get("document_name"),
            r.get("section_number"),
            r.get("chunk_index"),
            r.get("clause_text"),
        )
        for r in out
    }

    for doc in document_names:
        if doc in present:
            continue
        for candidate in candidates:
            key = (
                candidate.get("document_name"),
                candidate.get("section_number"),
                candidate.get("chunk_index"),
                candidate.get("clause_text"),
            )
            if candidate.get("document_name") == doc and key not in keys:
                out.append(candidate)
                present.add(doc)
                keys.add(key)
                break
    return out[:max_results]


def _get_sections_for_scope(
    section_number: str,
    document_names: List[str],
) -> List[Dict[str, Any]]:
    if not document_names:
        return get_compliance_section(section_number)

    rows: List[Dict[str, Any]] = []
    for document_name in document_names:
        rows.extend(get_compliance_section(section_number, doc_name=document_name))
    return _dedupe_search_results(rows)


def _lookup_chart_for_scope(
    figure_id: str,
    inputs: Dict[str, Any],
    compliance_code: Optional[str],
    document_names: List[str],
) -> Dict[str, Any]:
    if not document_names:
        return lookup_chart(
            figure_id=figure_id,
            inputs=inputs,
            compliance_code=compliance_code,
        )

    results = [
        lookup_chart(
            figure_id=figure_id,
            inputs=inputs,
            compliance_code=compliance_code,
            document_name=document_name,
        )
        for document_name in document_names
    ]
    return {
        "ok": any(result.get("ok") for result in results),
        "figure_id": figure_id,
        "results": results,
    }


def _react_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Hybrid semantic + keyword search across compliance documents. "
                    "Pass scope='primary' (default) to stay in the locked-in code/document(s), "
                    "or scope='all' to search every document — but only after you "
                    "have called widen_scope() with a reason."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "scope": {
                            "type": "string",
                            "enum": ["primary", "all"],
                            "description": "Primary keeps the locked scope; 'all' widens.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_section",
                "description": "Retrieve a specific section by number (e.g. 'D.8.8.2', 'Figure D.13', 'Table 4.6').",
                "parameters": {
                    "type": "object",
                    "properties": {"section_number": {"type": "string"}},
                    "required": ["section_number"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_chart",
                "description": (
                    "Deterministic lookup against a chart's structured data. "
                    "Returns {ok, value, matched_axes, source} on success."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "figure_id": {"type": "string"},
                        "inputs": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["figure_id", "inputs"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "widen_scope",
                "description": (
                    "Drop the primary-scope filter for the rest of this turn. "
                    "Call before any search(scope='all'). Provide a short reason."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                },
            },
        },
    ]


def _run_react_agent(
    question: str,
    chat_history: List[Any],
    scope: Optional[Dict[str, Any]],
    max_iterations: int = MAX_AGENT_ITERATIONS,
    *,
    meta_intent: bool = False,
) -> GeneralAgentResult:
    client = _client()

    primary_code = scope.get("compliance_code") if scope else None
    primary_docs = _scope_document_names(scope)

    widened = {"value": False, "reason": None}
    llm_call_count = {"n": 0}
    agent_search_count = {"n": 0}

    hyde_eligible = should_apply_hyde(question)

    system_prompt = _CODE_CHARLIE_REACT_SYSTEM_PROMPT.format(
        scope_summary=_scope_summary(scope),
    )
    messages: List[Any] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    all_sources: List[Dict[str, Any]] = []
    followed_refs: set = set()

    tools = _react_tools()

    for iteration in range(max_iterations):
        if llm_call_count["n"] >= MAX_TOTAL_LLM_CALLS:
            logger.warning(
                "Hit MAX_TOTAL_LLM_CALLS=%d before agent terminated", MAX_TOTAL_LLM_CALLS
            )
            break

        try:
            response = client.chat.completions.create(
                model=settings.GEN_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_completion_tokens=1500,
                temperature=0.2,
            )
            llm_call_count["n"] += 1
            msg = response.choices[0].message

            if not msg.tool_calls:
                logger.info("Code Charlie agent finished in %d iterations", iteration + 1)
                return GeneralAgentResult(
                    answer=msg.content or "",
                    sources=deduplicate_sources(all_sources),
                    iterations=iteration + 1,
                    scope_widened=widened["value"],
                    widen_reason=widened["reason"],
                )

            messages.append(msg)
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    tool_args = {}

                if tool_name == "search":
                    query = tool_args.get("query", "")
                    requested_scope = tool_args.get("scope", "primary")
                    effective_all = requested_scope == "all" and widened["value"]
                    eff_code = None if effective_all else primary_code
                    eff_docs = [] if effective_all else primary_docs

                    is_first_search = agent_search_count["n"] == 0
                    agent_search_count["n"] += 1

                    chunk_types_filter: Optional[List[str]] = None
                    hyde_embedding: Optional[List[float]] = None
                    if is_first_search:
                        if meta_intent:
                            chunk_types_filter = list(META_CHUNK_TYPES)
                        if hyde_eligible:
                            hypo = hypothetical_doc(question, enabled=True)
                            if hypo:
                                try:
                                    from agent.nodes.compliance_helpers import (
                                        embed_text,
                                    )
                                    hyde_embedding = embed_text(hypo)
                                except Exception as he:
                                    logger.warning("HyDE embed failed: %s", he)

                    logger.info(
                        "Agent searching: %r (req_scope=%s, widened=%s, code=%s, "
                        "docs=%s, first=%s, meta=%s, hyde=%s)",
                        query, requested_scope, widened["value"], eff_code, eff_docs,
                        is_first_search, bool(chunk_types_filter), bool(hyde_embedding),
                    )
                    raw = _search_compliance_scope(
                        query,
                        top_k=RERANK_FETCH_K,
                        compliance_code=eff_code,
                        document_names=eff_docs,
                        chunk_types=chunk_types_filter,
                        apply_multi_query=True,
                        hyde_embedding=hyde_embedding,
                    )
                    keep_n = _rerank_keep_count(eff_docs, len(raw))
                    rerank_pool = min(len(raw), max(keep_n * 3, 12))
                    if rerank_pool > keep_n and llm_call_count["n"] < MAX_TOTAL_LLM_CALLS:
                        ranked = llm_rerank(query, raw[:rerank_pool], top_n=rerank_pool)
                        llm_call_count["n"] += 1
                    else:
                        ranked = raw[:rerank_pool]
                    ranked = _apply_section_diversity(ranked, max_per_section=2)[:keep_n]
                    ranked = _ensure_document_coverage(ranked, raw, eff_docs, keep_n)

                    try:
                        ranked_with_context = expand_with_context(
                            ranked,
                            get_supabase_client(),
                        )
                    except Exception as ctx_err:
                        logger.warning("context expansion failed: %s", ctx_err)
                        ranked_with_context = ranked

                    observation = format_search_results(ranked_with_context)
                    for r in ranked:
                        all_sources.append(r)
                    for r in ranked:
                        for ref in extract_cross_references(r.get("clause_text", "") or ""):
                            if ref not in followed_refs:
                                observation += (
                                    f"\n[Note: content mentions {ref} — "
                                    f"consider get_section('{ref}') if relevant]"
                                )

                elif tool_name == "get_section":
                    section_num = tool_args.get("section_number", "")
                    logger.info("Agent get_section: %s", section_num)
                    if section_num:
                        followed_refs.add(section_num.upper())
                    section_docs = [] if widened["value"] else primary_docs
                    results = _get_sections_for_scope(section_num, section_docs)
                    observation = format_section_results(results)
                    for r in results:
                        all_sources.append(r)

                elif tool_name == "lookup_chart":
                    figure_id = tool_args.get("figure_id", "")
                    chart_inputs = tool_args.get("inputs", {}) or {}
                    logger.info(
                        "Agent lookup_chart: figure=%s inputs=%s code=%s",
                        figure_id, chart_inputs, primary_code,
                    )
                    chart_docs = [] if widened["value"] else primary_docs
                    result_dict = _lookup_chart_for_scope(
                        figure_id=figure_id,
                        inputs=chart_inputs,
                        compliance_code=None if widened["value"] else primary_code,
                        document_names=chart_docs,
                    )
                    observation = json.dumps(result_dict, ensure_ascii=False)

                elif tool_name == "widen_scope":
                    reason = (tool_args.get("reason") or "").strip() or "(unspecified)"
                    widened["value"] = True
                    widened["reason"] = reason
                    logger.info("Agent widen_scope invoked: %s", reason)
                    observation = json.dumps({
                        "ok": True,
                        "note": "Scope widened. Subsequent search() calls may use scope='all'.",
                    })

                else:
                    observation = f"Unknown tool: {tool_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

        except Exception as e:
            logger.error("Code Charlie agent iteration %d failed: %s", iteration + 1, e, exc_info=True)
            break

    return GeneralAgentResult(
        answer=(
            "I wasn't able to gather enough information to answer this "
            "confidently. Try rephrasing or narrowing the question to a "
            "specific section."
        ),
        sources=deduplicate_sources(all_sources),
        iterations=max_iterations,
        incomplete=True,
        scope_widened=widened["value"],
        widen_reason=widened["reason"],
    )


def code_charlie_compliance_rag_node(state: CodeCharlieState) -> Dict[str, Any]:
    user_message = get_last_user_message(state)
    if not user_message:
        return {
            "messages": [add_assistant_message(
                state,
                "I didn't catch a question — what would you like to look up?",
            )],
            "rag": None,
        }

    recent_history = state.get("messages", [])[-6:]
    scope = state.get("scope") or {}

    resolved_query = resolve_query_with_context(user_message, recent_history)

    sub_queries = decompose_complex_query(resolved_query, {})

    meta_intent = is_meta_question(resolved_query)
    if meta_intent:
        logger.info("Meta-question detected; first search will scope to META_CHUNK_TYPES")

    if len(sub_queries) == 1:
        result = _run_react_agent(
            resolved_query, recent_history, scope, meta_intent=meta_intent,
        )
    else:
        combined = (
            f"Answer this question: {resolved_query}\n\n"
            "Aspects to cover:\n"
            + "\n".join(f"- {q}" for q in sub_queries)
            + "\nSearch for each aspect and combine into one coherent answer."
        )
        result = _run_react_agent(
            combined, recent_history, scope, meta_intent=meta_intent,
        )

    formatted_sources = format_sources_for_frontend(result.sources)

    confidence = 0.0
    if result.sources:
        sims = [
            s.get("similarity", s.get("combined_score", 0)) or 0
            for s in result.sources
        ]
        confidence = max(sims) if sims else 0.0

    final_answer = result.answer

    if (not result.sources or not final_answer.strip()) and not result.scope_widened:
        scope_docs = _scope_document_names(scope)
        scope_label = (
            ", ".join(scope_docs)
            if scope_docs
            else scope.get("compliance_code") or "the selected scope"
        )
        final_answer = (
            f"I couldn't find anything in **{scope_label}** for that query. "
            "Want me to search across all documents, or could you rephrase "
            "with a specific section / topic in mind?"
        )

    dropped_claims: List[str] = []
    if result.sources and final_answer == result.answer:
        verified = verify_citations(final_answer, result.sources)
        final_answer = verified["verified_answer"]
        dropped_claims = verified["dropped_claims"]

    metadata: Dict[str, Any] = {
        "sources": formatted_sources,
        "confidence": confidence,
        "scope_widened": result.scope_widened,
    }
    if result.widen_reason:
        metadata["widen_reason"] = result.widen_reason
    if dropped_claims:
        metadata["dropped_claims"] = dropped_claims
    if result.incomplete:
        metadata["incomplete"] = True

    rag_payload: Dict[str, Any] = {
        "query": user_message,
        "resolved_query": resolved_query,
        "answer": final_answer,
        "sources": formatted_sources,
        "confidence": confidence,
        "scope_used": "all" if result.scope_widened else "primary",
        "scope_widened": result.scope_widened,
        "widen_reason": result.widen_reason,
    }

    return {
        "messages": [add_assistant_message(state, final_answer, metadata=metadata)],
        "rag": rag_payload,
    }
