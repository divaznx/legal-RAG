"""Clause relationship graph (Feature: Clause Relationship Graph).

At ingest, every chunk's cross-references ("Clause 5", "Section 2.1", ...)
are collected into a per-document adjacency map persisted at
data/clause_graph.json:

    {"MSA_Acme_v2.1.txt": {"Clause 6": ["Clause 5"], ...}}

Retrieval-side expansion is OPTIONAL (settings.graph_enabled, default off):
when enabled, sections referenced by the final retrieved chunks are pulled
in as additional context, bounded by settings.graph_max_expansion.
"""

from __future__ import annotations

import json
from pathlib import Path

from .chunking import Chunk
from .config import settings


def _graph_path() -> Path:
    return settings.data_path / "clause_graph.json"


def load_graph() -> dict[str, dict[str, list[str]]]:
    path = _graph_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_graph(graph: dict[str, dict[str, list[str]]]) -> None:
    _graph_path().write_text(json.dumps(graph, indent=2), encoding="utf-8")


def build_document_graph(chunks: list[Chunk]) -> dict[str, list[str]]:
    """section -> referenced sections, merged across the document's chunks."""
    edges: dict[str, list[str]] = {}
    for chunk in chunks:
        if not chunk.section or not chunk.references:
            continue
        targets = edges.setdefault(chunk.section, [])
        for ref in chunk.references:
            if ref not in targets:
                targets.append(ref)
    return edges


def update_graph(document: str, chunks: list[Chunk]) -> None:
    graph = load_graph()
    edges = build_document_graph(chunks)
    if edges:
        graph[document] = edges
    else:
        graph.pop(document, None)
    _save_graph(graph)


def remove_document(document: str) -> None:
    graph = load_graph()
    if document in graph:
        del graph[document]
        _save_graph(graph)


def referenced_sections(document: str, section: str | None) -> list[str]:
    """Sections that `section` of `document` references (one hop)."""
    if not section:
        return []
    return load_graph().get(document, {}).get(section, [])
