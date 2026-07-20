"""Context validation & packing (Feature: Context Validation & Packing),
parent-section attachment (Feature: Parent-Child Retrieval), and optional
clause-graph expansion (Feature: Clause Relationship Graph).

Runs on the final selected chunks *after* ranking and *before* prompt
formatting. The engine keeps the original chunk list for citation
verification and the evidence panel; packing produces the cleaned view the
LLM sees:

- exact and contained duplicates removed
- adjacent overlapping chunks (chunker overlap tails) merged back together
- repeated heading lines stripped
- logical document order restored (document, page, chunk index) with clause
  numbering and section hierarchy intact

Pure functions over RetrievedChunk copies — originals are never mutated.
"""

from __future__ import annotations

import re
from dataclasses import replace

from . import graph as graph_mod
from . import vector_store
from .config import settings
from .vector_store import RetrievedChunk


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _overlap_len(a: str, b: str, limit: int) -> int:
    """Longest suffix of `a` that is a prefix of `b`, up to `limit` chars."""
    max_k = min(len(a), len(b), limit)
    for k in range(max_k, 19, -1):  # below ~20 chars it's coincidence, not overlap
        if a[-k:] == b[:k]:
            return k
    return 0


def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop exact duplicates and chunks wholly contained in another chunk."""
    kept: list[RetrievedChunk] = []
    norms: list[str] = []
    for c in chunks:
        n = _norm(c.text)
        if any(n == other or n in other for other in norms):
            continue
        # a previously kept chunk may be contained in this larger one
        kept = [k for k, kn in zip(kept, norms) if kn not in n]
        norms = [_norm(k.text) for k in kept]
        kept.append(c)
        norms.append(n)
    return kept


def _merge_adjacent(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Merge document-adjacent chunks, stitching out the chunker's overlap.

    Only merges within the same document and page so every merged block's
    citation header stays truthful."""
    if not chunks:
        return chunks
    merged: list[RetrievedChunk] = [replace(chunks[0])]
    for c in chunks[1:]:
        prev = merged[-1]
        adjacent = (
            c.document == prev.document
            and c.page == prev.page
            and prev.index >= 0
            and c.index == prev.index + 1
        )
        if adjacent:
            k = _overlap_len(prev.text, c.text, settings.chunk_overlap + 8)
            stitched = prev.text + ("\n\n" if not k else "") + c.text[k:]
            merged[-1] = replace(
                prev,
                text=stitched,
                index=c.index,  # so a third consecutive chunk can chain-merge
                matched_on=sorted(set(prev.matched_on) | set(c.matched_on)),
            )
        else:
            merged.append(replace(c))
    return merged


def _strip_repeated_headings(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for c in chunks:
        text = c.text
        first_line = text.split("\n", 1)[0].strip()
        if c.heading and _norm(first_line) == _norm(c.heading):
            if c.heading in seen and "\n" in text:
                text = text.split("\n", 1)[1].lstrip("\n")
            seen.add(c.heading)
        out.append(replace(c, text=text) if text is not c.text else c)
    return out


def pack(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Clean, non-redundant, document-ordered context for the LLM."""
    if not settings.context_packing or not chunks:
        return chunks
    ordered = sorted(chunks, key=lambda c: (c.document, c.page, c.index))
    return _strip_repeated_headings(_merge_adjacent(_dedupe(ordered)))


# --- Parent-child attachment (Feature: Parent-Child Retrieval) --------------

def attach_parents(chunks: list[RetrievedChunk]) -> None:
    """Attach one level of parent-section context to child chunks (in place,
    on the final selection only — never widens the candidate set).

    Skipped when the parent's section already appears in the selection, so
    no context is duplicated and nothing irrelevant is added."""
    if not settings.parent_context_enabled:
        return
    present = {(c.document, (c.section or "").lower()) for c in chunks}
    for c in chunks:
        if not c.parent_section:
            continue
        parent_key = (c.document, c.parent_section.lower())
        covered = any(
            doc == c.document and (sec == c.parent_section.lower()
                                   or sec.startswith(c.parent_section.lower() + "."))
            for doc, sec in present
        )
        if covered or parent_key in present:
            continue
        parent_text = vector_store.fetch_section(
            c.document, c.parent_section, settings.parent_context_max_chars
        )
        if parent_text and _norm(parent_text) not in _norm(c.text):
            c.parent_context = parent_text
            if "parent-context" not in c.matched_on:
                c.matched_on.append("parent-context")


# --- Optional graph expansion (Feature: Clause Relationship Graph) ----------

def expand_references(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Cross-reference resolution (Feature: Cross-Reference Resolution).

    When a retrieved section points at another ("incorporated by reference
    into Item 1.01", "subject to Clause 5", "see Exhibit A"), the referenced
    sections are fetched and added to the context. Chained references are
    followed breadth-first up to settings.graph_max_depth hops; a visited
    set prevents reference cycles (A→B→A) from looping, and the total
    addition is capped at settings.graph_max_expansion chunks.

    Each added chunk's matched_on records the resolution path
    ("graph-reference:Item 3.03->Item 1.01") for debug-mode explainability.
    """
    if not settings.graph_enabled or not chunks:
        return chunks
    visited: set[tuple[str, str]] = {(c.document, (c.section or "").lower()) for c in chunks}
    extras: list[RetrievedChunk] = []
    frontier: list[tuple[str, str | None]] = [
        (c.document, c.section) for c in chunks if c.section
    ]
    for _hop in range(max(1, settings.graph_max_depth)):
        next_frontier: list[tuple[str, str | None]] = []
        for document, section in frontier:
            for ref in graph_mod.referenced_sections(document, section):
                key = (document, ref.lower())
                if key in visited or len(extras) >= settings.graph_max_expansion:
                    continue
                visited.add(key)
                fetched = vector_store.fetch_section_chunk(document, ref)
                if fetched is not None:
                    fetched.matched_on = [f"graph-reference:{section}->{ref}"]
                    extras.append(fetched)
                    next_frontier.append((document, ref))
        if not next_frontier or len(extras) >= settings.graph_max_expansion:
            break
        frontier = next_frontier
    return chunks + extras
