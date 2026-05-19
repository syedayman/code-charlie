"""
Context expansion for compliance retrieval.

After the reranker picks the top-K hit chunks, this module fetches the
surrounding context — chunks that are likely to complete or clarify the
hit but didn't rank highly on their own:

  - Adjacent: chunk_index ± window in the same document.
  - Siblings: chunks under the same parent section.
  - Parent: the parent section's own chunk.

All fetches use the same `compliance_embeddings` table. No new LLM calls.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_DOTTED_SECTION_RE = re.compile(r"^([A-Z]?\d+(?:\.\d+)+)$")


_CONTEXT_SELECT = (
    "id,document_name,document_version,compliance_code,section_number,"
    "section_title,section_hierarchy,clause_text,chunk_index,chunk_type,"
    "page_range,figure_id,structured_data,context_summary,metadata"
)


def _nearest_primary_chunk_index(
    chunk_index: int,
    primary_indices_by_doc: Dict[str, List[int]],
    doc: str,
) -> Optional[int]:
    indices = primary_indices_by_doc.get(doc) or []
    if not indices:
        return None
    return min(indices, key=lambda i: abs(i - chunk_index))


def _parent_path(hierarchy: List[str]) -> List[str]:
    if not hierarchy or len(hierarchy) < 2:
        return []
    return list(hierarchy[:-1])


def _derive_parent_section_number(section_number: Optional[str]) -> Optional[str]:
    if not section_number:
        return None
    s = section_number.strip()
    if not _DOTTED_SECTION_RE.match(s):
        return None
    head, _, _ = s.rpartition(".")
    return head or None


def _is_immediate_child_section(child: str, parent: str) -> bool:
    if not (child and parent):
        return False
    if not child.startswith(parent + "."):
        return False
    tail = child[len(parent) + 1 :]
    return "." not in tail


def expand_with_context(
    hits: List[Dict[str, Any]],
    supabase: Any,
    *,
    adjacent_window: int = 1,
    include_siblings: bool = True,
    include_parent: bool = True,
    max_siblings_per_hit: int = 2,
    max_context: int = 6,
) -> List[Dict[str, Any]]:
    """Expand hits with adjacent + sibling + parent chunks."""
    if not hits:
        return []

    primary_ids: Set[str] = set()
    primary_indices_by_doc: Dict[str, List[int]] = {}
    for h in hits:
        h["_context_role"] = "primary"
        if h.get("id"):
            primary_ids.add(h["id"])
        doc = h.get("document_name")
        ci = h.get("chunk_index")
        if doc and isinstance(ci, int):
            primary_indices_by_doc.setdefault(doc, []).append(ci)

    want_adjacent: Dict[str, Set[int]] = {}
    want_parent_paths: Dict[str, Set[Tuple[str, ...]]] = {}
    want_parent_section_numbers: Dict[str, Set[str]] = {}

    for h in hits:
        doc = h.get("document_name")
        if not doc:
            continue
        ci = h.get("chunk_index")
        if isinstance(ci, int) and adjacent_window > 0:
            adj = want_adjacent.setdefault(doc, set())
            for delta in range(1, adjacent_window + 1):
                if ci - delta >= 0:
                    adj.add(ci - delta)
                adj.add(ci + delta)

        if include_siblings or include_parent:
            hierarchy = h.get("section_hierarchy") or []
            parent = _parent_path(hierarchy)
            if parent:
                paths = want_parent_paths.setdefault(doc, set())
                paths.add(tuple(parent))
            else:
                parent_sec = _derive_parent_section_number(h.get("section_number"))
                if parent_sec:
                    nums = want_parent_section_numbers.setdefault(doc, set())
                    nums.add(parent_sec)

    seen_ids: Set[str] = set(primary_ids)
    context_buckets: Dict[str, List[Dict[str, Any]]] = {
        "adjacent_before": [],
        "adjacent_after": [],
        "sibling": [],
        "parent": [],
    }

    for doc, indices in want_adjacent.items():
        if not indices:
            continue
        idx_list = sorted(indices)
        try:
            res = (
                supabase.table("compliance_embeddings")
                .select(_CONTEXT_SELECT)
                .eq("document_name", doc)
                .in_("chunk_index", idx_list)
                .execute()
            )
            for row in res.data or []:
                if row.get("id") in seen_ids:
                    continue
                seen_ids.add(row["id"])
                ci = row.get("chunk_index") or 0
                near = _nearest_primary_chunk_index(
                    ci, primary_indices_by_doc, doc
                )
                role = "adjacent_before" if (near is not None and ci < near) else "adjacent_after"
                row["_context_role"] = role
                context_buckets[role].append(row)
        except Exception as e:
            logger.warning("adjacent fetch failed for %s: %s", doc, e)

    if include_siblings or include_parent:
        for doc, parent_paths in want_parent_paths.items():
            for parent_path in parent_paths:
                if not parent_path:
                    continue
                parent_path_list = list(parent_path)
                try:
                    res = (
                        supabase.table("compliance_embeddings")
                        .select(_CONTEXT_SELECT)
                        .eq("document_name", doc)
                        .contains("section_hierarchy", parent_path_list)
                        .limit(max_siblings_per_hit + 4)
                        .execute()
                    )
                except Exception as e:
                    logger.warning(
                        "sibling fetch failed for %s @ %s: %s",
                        doc, parent_path_list, e,
                    )
                    continue

                siblings_added_for_this_path = 0
                for row in res.data or []:
                    if row.get("id") in seen_ids:
                        continue
                    row_hier = row.get("section_hierarchy") or []
                    plen = len(parent_path_list)
                    if len(row_hier) == plen and include_parent:
                        seen_ids.add(row["id"])
                        row["_context_role"] = "parent"
                        context_buckets["parent"].append(row)
                    elif len(row_hier) == plen + 1 and include_siblings:
                        if siblings_added_for_this_path >= max_siblings_per_hit:
                            continue
                        seen_ids.add(row["id"])
                        row["_context_role"] = "sibling"
                        context_buckets["sibling"].append(row)
                        siblings_added_for_this_path += 1

    if include_siblings or include_parent:
        for doc, parent_sections in want_parent_section_numbers.items():
            for parent_sec in parent_sections:
                try:
                    res = (
                        supabase.table("compliance_embeddings")
                        .select(_CONTEXT_SELECT)
                        .eq("document_name", doc)
                        .like("section_number", f"{parent_sec}%")
                        .limit(max_siblings_per_hit * 4 + 4)
                        .execute()
                    )
                except Exception as e:
                    logger.warning(
                        "section_number fallback fetch failed for %s @ %s: %s",
                        doc, parent_sec, e,
                    )
                    continue

                siblings_added = 0
                for row in res.data or []:
                    if row.get("id") in seen_ids:
                        continue
                    sec = (row.get("section_number") or "").strip()
                    if sec == parent_sec:
                        if include_parent:
                            seen_ids.add(row["id"])
                            row["_context_role"] = "parent"
                            context_buckets["parent"].append(row)
                    elif (
                        include_siblings
                        and _is_immediate_child_section(sec, parent_sec)
                    ):
                        if siblings_added >= max_siblings_per_hit:
                            continue
                        seen_ids.add(row["id"])
                        row["_context_role"] = "sibling"
                        context_buckets["sibling"].append(row)
                        siblings_added += 1

    ordered_context: List[Dict[str, Any]] = []
    for role in ("adjacent_before", "adjacent_after", "sibling", "parent"):
        ordered_context.extend(context_buckets[role])

    ordered_context = ordered_context[:max_context]

    return list(hits) + ordered_context
