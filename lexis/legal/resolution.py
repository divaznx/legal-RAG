"""Document Resolution Layer — runs before any retrieval.

The failure this prevents is the one that actually ends careers: answering
from the wrong agreement. A corpus holding an MSA v1.0 (30-day termination
notice, Delaware law) and its amended-and-restated v2.1 (60 days, New York
law) will, under plain semantic retrieval, return chunks from both — and the
model will fluently blend a 30-day Delaware answer out of a contract that no
longer governs.

So the target document is resolved FIRST, from document-level profiles:

    explicit filename > doc-type + party > clause inventory > version lineage

and evidence is then filtered to that document. When the question is genuinely
ambiguous across unrelated agreements, the honest move is to ask rather than
guess, so this layer can return a clarification request instead of an answer.

Version awareness is handled here too: within one lineage the latest version
wins by default and the superseded ones are reported, never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import entities, ontology
from .intent import IntentResult
from .profile import DocumentProfile, version_key

_LATEST_RE = re.compile(
    r"(?i)\b(?:latest|current|newest|most recent|in force|effective|operative|"
    r"amended and restated|restated|as amended)\b"
)
_OLDEST_RE = re.compile(
    r"(?i)\b(?:original|first|earliest|initial|previous|prior|earlier|superseded|old)\b"
)
_EXPLICIT_VERSION_RE = re.compile(r"(?i)\b(?:v|version|rev|revision)\.?\s*(\d+(?:\.\d+)*)\b")
_ALL_DOCS_RE = re.compile(
    r"(?i)\b(?:all|every|each|any) (?:of )?(?:my |our |the )?"
    r"(?:agreements?|contracts?|documents?|files?)\b|\backross (?:all|my|our|the)\b"
)


@dataclass
class DocumentResolution:
    documents: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: str = "Low"          # High | Medium | Low
    needs_clarification: bool = False
    clarification: str = ""
    options: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    missing_clause: str | None = None
    signals: dict = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return bool(self.documents) and not self.needs_clarification


def _describe(p: DocumentProfile, latest_in_family: bool) -> str:
    bits = [f"{p.doc_type_label} v{p.version}"]
    if p.organizations:
        bits.append(p.organizations[0])
    if p.is_amendment:
        bits.append("amended & restated")
    if latest_in_family:
        bits.append("latest")
    # ASCII separator: this string is printed to Windows terminals by the CLI,
    # where a non-cp1252 dash renders as a replacement character.
    return f"{p.document} - {', '.join(bits)}"


def _doc_type_mentions(question: str, profiles: list[DocumentProfile]) -> set[str]:
    """Doc-type codes named in the question, via label, code, and synonyms."""
    lowered = question.lower()
    hits: set[str] = set()
    expansions = set(ontology.synonyms_for(question)) | {lowered}
    for p in profiles:
        forms = {p.doc_type.lower(), p.doc_type_label.lower()}
        forms |= {f for f in ontology.synonyms_for(p.doc_type_label) }
        forms |= {f for f in ontology.synonyms_for(p.doc_type)}
        forms.discard("")
        for form in forms:
            if len(form) < 3:
                continue
            pattern = rf"(?<!\w){re.escape(form)}(?!\w)"
            if re.search(pattern, lowered) or any(re.search(pattern, e) for e in expansions):
                hits.add(p.doc_type)
                break
    return hits


def _org_mentions(question: str, profiles: list[DocumentProfile]) -> set[str]:
    """Party/organisation names named in the question.

    Matched on the distinctive leading word ("Acme" out of "Acme Consulting
    LLC") because nobody types the registered entity suffix in a question.
    """
    lowered = question.lower()
    hits: set[str] = set()
    for p in profiles:
        for org in p.organizations:
            head = re.split(r"\s+", org.strip())[0].strip(".,").lower()
            if len(head) >= 4 and re.search(rf"(?<!\w){re.escape(head)}(?!\w)", lowered):
                hits.add(p.document)
    return hits


def _filename_mentions(question: str, profiles: list[DocumentProfile]) -> set[str]:
    lowered = question.lower()
    hits: set[str] = set()
    for p in profiles:
        stem = Path(p.document).stem.lower()
        if p.document.lower() in lowered or (len(stem) >= 5 and stem in lowered):
            hits.add(p.document)
    return hits


def _latest_per_family(profiles: list[DocumentProfile]) -> dict[str, DocumentProfile]:
    latest: dict[str, DocumentProfile] = {}
    for p in profiles:
        current = latest.get(p.family)
        if current is None or version_key(p.version) > version_key(current.version):
            latest[p.family] = p
    return latest


def resolve(
    question: str,
    profiles: list[DocumentProfile],
    intent: IntentResult | None = None,
    allow_clarification: bool = True,
) -> DocumentResolution:
    """Pick the agreement(s) this question is about."""
    if not profiles:
        return DocumentResolution(reason="No documents have been ingested.")

    policy = intent.policy if intent else None
    signals: dict = {}
    pool = list(profiles)

    # 1. Explicit filename — the strongest possible signal.
    by_name = _filename_mentions(question, pool)
    if by_name:
        signals["filename"] = sorted(by_name)
        pool = [p for p in pool if p.document in by_name]
        if len(pool) == 1:
            return DocumentResolution(
                documents=[pool[0].document],
                reason=f"Question names {pool[0].document} explicitly.",
                confidence="High", signals=signals,
            )

    # 2. Document type ("the NDA", "master services agreement").
    types = _doc_type_mentions(question, pool)
    if types:
        signals["doc_type"] = sorted(types)
        narrowed = [p for p in pool if p.doc_type in types]
        if narrowed:
            pool = narrowed

    # 3. Party / organisation.
    orgs = _org_mentions(question, pool)
    if orgs:
        signals["organization"] = sorted(orgs)
        narrowed = [p for p in pool if p.document in orgs]
        if narrowed:
            pool = narrowed

    # 4. Clause inventory — "Clause 8" only exists in some agreements.
    clause_refs = entities.extract_clause_refs(question)
    missing_clause: str | None = None
    if clause_refs:
        wanted = {ref.number for ref in clause_refs}
        signals["clause_refs"] = sorted(wanted)
        with_inventory = [p for p in pool if p.clause_numbers]
        holders = [p for p in with_inventory if wanted & set(p.clause_numbers)]
        if holders:
            pool = holders
        elif with_inventory:
            # Every candidate has a clause inventory and none contains it.
            missing_clause = clause_refs[0].label

    # 5. Version lineage.
    explicit_version = _EXPLICIT_VERSION_RE.search(question)
    if explicit_version:
        wanted_version = explicit_version.group(1)
        signals["version"] = wanted_version
        exact = [p for p in pool if p.version == wanted_version]
        if exact:
            pool = exact
    wants_latest = bool(_LATEST_RE.search(question))
    wants_oldest = bool(_OLDEST_RE.search(question))

    if missing_clause:
        return DocumentResolution(
            documents=[p.document for p in pool],
            reason=f"No ingested agreement contains {missing_clause}.",
            confidence="High",
            missing_clause=missing_clause,
            options=[_describe(p, False) for p in pool],
            signals=signals,
        )

    # 6. Corpus-wide questions and comparisons legitimately span documents.
    multi_ok = bool(policy and policy.multi_document)
    if _ALL_DOCS_RE.search(question):
        signals["scope"] = "all documents"
        return DocumentResolution(
            documents=[p.document for p in pool],
            reason="Question explicitly spans the whole corpus.",
            confidence="Medium", signals=signals,
        )

    if len(pool) == 1:
        return DocumentResolution(
            documents=[pool[0].document],
            reason=f"Only {pool[0].document} matches the question's signals.",
            confidence="High" if signals else "Medium",
            signals=signals,
        )

    families = {p.family for p in pool}

    # 7. Single lineage, several versions -> version awareness decides.
    if len(families) == 1:
        ordered = sorted(pool, key=lambda p: version_key(p.version), reverse=True)
        if multi_ok or (wants_oldest and wants_latest):
            return DocumentResolution(
                documents=[p.document for p in ordered],
                reason="Comparison across versions of the same agreement.",
                confidence="High", signals=signals,
            )
        chosen = ordered[-1] if wants_oldest else ordered[0]
        rest = [p for p in ordered if p.document != chosen.document]
        return DocumentResolution(
            documents=[chosen.document],
            reason=(f"{'Earliest' if wants_oldest else 'Latest'} version of this "
                    f"agreement (v{chosen.version}); "
                    f"{len(rest)} other version(s) excluded."),
            confidence="High",
            superseded=[f"{p.document} (v{p.version})" for p in rest],
            signals=signals,
        )

    # 8. Genuinely different agreements -> compare only if asked, else ask.
    if multi_ok:
        latest = _latest_per_family(pool)
        return DocumentResolution(
            documents=[p.document for p in latest.values()],
            reason="Comparison across distinct agreements.",
            confidence="Medium", signals=signals,
        )

    latest = _latest_per_family(pool)
    options = [_describe(p, True) for p in latest.values()]
    if allow_clarification:
        return DocumentResolution(
            documents=[],
            reason=f"{len(latest)} unrelated agreements match this question.",
            confidence="Low",
            needs_clarification=True,
            clarification=_clarification_text(question, options),
            options=options,
            signals=signals,
        )

    return DocumentResolution(
        documents=[p.document for p in latest.values()],
        reason="Ambiguous across agreements; clarification disabled.",
        confidence="Low", options=options, signals=signals,
    )


def _clarification_text(question: str, options: list[str]) -> str:
    listing = "\n".join(f"  {i}. {o}" for i, o in enumerate(options, start=1))
    return (
        "Which agreement should I answer from? The question matches more than "
        "one, and mixing clauses across agreements would produce an unreliable "
        f"answer.\n\n{listing}\n\n"
        "Re-ask naming the agreement (for example: "
        f"\"{question.rstrip('?')} in the {options[0].split(' - ')[0]}?\")."
    )
