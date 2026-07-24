"""Document Resolution Layer — runs before any retrieval.

The failure this prevents is the one that actually ends careers: answering
from the wrong agreement. A corpus holding an MSA v1.0 (30-day termination
notice, Delaware law) and its amended-and-restated v2.1 (60 days, New York
law) will, under plain semantic retrieval, return chunks from both — and the
model will fluently blend a 30-day Delaware answer out of a contract that no
longer governs.

So the target document is resolved FIRST, from document-level profiles:

    explicit filename > doc-type + party > defined terms > clause inventory
    > version lineage > legal-concept affinity

and evidence is then filtered to that document. When one agreement is merely
*substantially* more likely than the rest (unique defined terms, the legal
issue is a primary subject of one agreement only), resolution proceeds at
Medium confidence and the stated assumption travels with the answer. Only
when candidates are genuinely level does the honest move become asking, so
this layer can return a clarification request instead of an answer.

Every decision keeps its negative evidence: each candidate that was ruled
out is recorded with the reason (missing clause, different parties, wrong
agreement type, superseded version), so the selection is auditable and the
confidence rests on why the others were rejected, not only on why the winner
matched. Matching is deliberately forgiving — aliases, abbreviations, and
minor misspellings resolve to the same document — and when the current
question names no agreement at all, the agreement the conversation has been
about stays active rather than re-asking the user every turn.

Version awareness is handled here too: within one lineage the latest version
wins by default and the superseded ones are reported, never silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
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
    r"(?:agreements?|contracts?|documents?|files?|ndas?|msas?|sows?|dpas?|slas?)\b"
    r"|\bacross (?:all|my|our|the)\b"
    r"|\b(?:which|what) (?:agreement|contract|document)\b"
    r"|\bcross-?referenc\w*\b"
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
    assumption: str = ""             # user-visible, set only at Medium confidence
    signals: dict = field(default_factory=dict)
    rejected: dict = field(default_factory=dict)   # document -> why it was ruled out

    @property
    def resolved(self) -> bool:
        return bool(self.documents) and not self.needs_clarification


def _describe(p: DocumentProfile, latest_in_family: bool) -> str:
    """Human-recognisable identity for a candidate agreement.

    A lawyer choosing between candidates recognises parties, governing law,
    and subject matter, not filenames — so those lead, and the filename rides
    along in parentheses only because re-asking by filename must keep working.
    """
    bits = []
    if p.organizations:
        bits.append("parties: " + ", ".join(p.organizations[:2]))
    if p.governing_law:
        bits.append(f"governing law: {p.governing_law}")
    if p.concepts:
        subjects = ", ".join(c.replace("_", " ") for c in p.concepts[:3])
        bits.append(f"deals with: {subjects}")
    if p.is_amendment:
        bits.append("amended & restated")
    if latest_in_family:
        bits.append("latest version")
    detail = f" - {'; '.join(bits)}" if bits else ""
    # ASCII separators: this string is printed to Windows terminals by the CLI,
    # where a non-cp1252 dash renders as a replacement character.
    return f"{p.doc_type_label} v{p.version} ({p.document}){detail}"


def _question_words(question: str) -> list[str]:
    return re.findall(r"[a-z0-9][\w&'-]*", question.lower())


def _fuzzy_contains(form: str, question_words: list[str]) -> bool:
    """True when the question contains `form` up to a minor misspelling.

    Users type "master servces agreement" and "Nothwind"; exact word-boundary
    regexes miss both. Only distinctive forms get the fuzzy treatment — short
    abbreviations ("msa", "nda") stay exact-match, otherwise every three-letter
    token in the question would land on one of them.
    """
    if len(form) < 6:
        return False
    span = len(form.split())
    for i in range(len(question_words) - span + 1):
        window = " ".join(question_words[i:i + span])
        if abs(len(window) - len(form)) <= 3 and \
                SequenceMatcher(None, form, window).ratio() >= 0.85:
            return True
    return False


def _doc_type_mentions(question: str, profiles: list[DocumentProfile]) -> set[str]:
    """Doc-type codes named in the question — via label, code, synonyms
    (aliases like "confidentiality agreement" for an NDA), or a fuzzy match
    that tolerates minor misspellings."""
    lowered = question.lower()
    words = _question_words(question)
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
            if (re.search(pattern, lowered)
                    or any(re.search(pattern, e) for e in expansions)
                    or _fuzzy_contains(form, words)):
                hits.add(p.doc_type)
                break
    return hits


def _org_mentions(question: str, profiles: list[DocumentProfile]) -> set[str]:
    """Party/organisation names named in the question.

    Matched on the distinctive leading word ("Acme" out of "Acme Consulting
    LLC") because nobody types the registered entity suffix in a question.
    Distinctive names also match through minor misspellings ("Nothwind").
    """
    lowered = question.lower()
    words = _question_words(question)
    hits: set[str] = set()
    for p in profiles:
        for org in p.organizations:
            head = re.split(r"\s+", org.strip())[0].strip(".,").lower()
            if len(head) < 4:
                continue
            if re.search(rf"(?<!\w){re.escape(head)}(?!\w)", lowered) or (
                len(head) >= 5 and any(
                    SequenceMatcher(None, head, w).ratio() >= 0.86
                    for w in words if len(w) >= 4
                )
            ):
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


def _defined_term_matches(
    question: str, pool: list[DocumentProfile]
) -> dict[str, list[str]]:
    """document -> defined terms from that document's inventory that the
    question actually uses.

    Defined terms are often the fingerprint of one specific agreement —
    "Subscription Fees" exists only in the SaaS Agreement, "Purchase Order"
    only in the Supply Agreement. Terms that every candidate defines
    ("Agreement", "Services") identify nothing and are dropped, so only
    discriminating terms are returned.
    """
    lowered = question.lower()
    term_docs: dict[str, set[str]] = {}
    for p in pool:
        for term in p.defined_terms:
            t = term.strip().lower()
            # Single short words ("Term", "Fee") collide with ordinary English
            # far too often to identify a document.
            if not t or (" " not in t and len(t) < 5):
                continue
            if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", lowered):
                term_docs.setdefault(t, set()).add(p.document)
    discriminating = {t: d for t, d in term_docs.items() if len(d) < len(pool)}
    out: dict[str, list[str]] = {}
    for term, docs in discriminating.items():
        for doc in docs:
            out.setdefault(doc, []).append(term)
    return out


def _concept_affinity(question: str, pool: list[DocumentProfile]) -> dict[str, float]:
    """How central the question's legal concepts are to each candidate.

    profile.concepts is frequency-ordered at ingest, so an early position
    means the concept is a primary subject of that agreement — a
    confidentiality question points at the NDA, not at the MSA that mentions
    confidentiality once in passing.
    """
    q_concepts = ontology.detect_concepts(question)
    scores: dict[str, float] = {}
    for p in pool:
        score = 0.0
        for concept in q_concepts:
            if concept in p.concepts:
                score += 1.0 if p.concepts.index(concept) < 5 else 0.5
        if score:
            scores[p.document] = score
    return scores


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
    history: list[str] | None = None,
) -> DocumentResolution:
    """Pick the agreement(s) this question is about.

    `history` is the user's prior questions in this conversation, oldest
    first. It is the weakest signal: consulted only when the current question
    itself identifies no agreement, so an explicit mention always overrides
    the active agreement.
    """
    if not profiles:
        return DocumentResolution(reason="No documents have been ingested.")

    policy = intent.policy if intent else None
    signals: dict = {}
    pool = list(profiles)

    # Negative evidence: every candidate ruled out is recorded with the
    # reason, so the final selection is auditable and its confidence rests on
    # why the others were rejected, not only on why the winner matched.
    rejected: dict[str, str] = {}

    def narrow(keep: list[DocumentProfile], why) -> None:
        nonlocal pool
        if keep and len(keep) < len(pool):
            for p in pool:
                if p not in keep:
                    rejected.setdefault(p.document, why(p))
            pool = keep

    # 1. Explicit filename — the strongest possible signal.
    by_name = _filename_mentions(question, pool)
    if by_name:
        signals["filename"] = sorted(by_name)
        narrow([p for p in pool if p.document in by_name],
               lambda p: "the question explicitly names a different document")
        if len(pool) == 1:
            return DocumentResolution(
                documents=[pool[0].document],
                reason=f"Question names {pool[0].document} explicitly.",
                confidence="High", signals=signals, rejected=rejected,
            )

    # 2. Document type ("the NDA", "master services agreement") — aliases,
    # abbreviations, and near-misspellings all resolve here.
    types = _doc_type_mentions(question, pool)
    if types:
        signals["doc_type"] = sorted(types)
        labels = ", ".join(sorted(types))
        narrow([p for p in pool if p.doc_type in types],
               lambda p: f"different agreement type ({p.doc_type_label}); "
                         f"the question refers to: {labels}")

    # 3. Party / organisation.
    orgs = _org_mentions(question, pool)
    if orgs:
        signals["organization"] = sorted(orgs)
        narrow([p for p in pool if p.document in orgs],
               lambda p: "none of its parties are named in the question")

    # 4. Defined terms — "Subscription Fees" often names exactly one agreement.
    term_hits = _defined_term_matches(question, pool)
    if term_hits:
        best = max(len(terms) for terms in term_hits.values())
        keep = {doc for doc, terms in term_hits.items() if len(terms) == best}
        matched_terms = sorted({t for doc in keep for t in term_hits[doc]})
        signals["defined_terms"] = matched_terms
        quoted = ", ".join(f'"{t}"' for t in matched_terms[:3])
        narrow([p for p in pool if p.document in keep],
               lambda p: f"does not define the term(s) the question uses ({quoted})")

    # 5. Clause inventory — "Clause 8" only exists in some agreements.
    clause_refs = entities.extract_clause_refs(question)
    missing_clause: str | None = None
    if clause_refs:
        wanted = {ref.number for ref in clause_refs}
        signals["clause_refs"] = sorted(wanted)
        label = clause_refs[0].label
        with_inventory = [p for p in pool if p.clause_numbers]
        holders = [p for p in with_inventory if wanted & set(p.clause_numbers)]
        if holders:
            narrow(holders, lambda p: f"does not contain {label}")
        elif with_inventory:
            # Every candidate has a clause inventory and none contains it.
            missing_clause = label

    # 6. Version lineage.
    explicit_version = _EXPLICIT_VERSION_RE.search(question)
    if explicit_version:
        wanted_version = explicit_version.group(1)
        signals["version"] = wanted_version
        exact = [p for p in pool if p.version == wanted_version]
        narrow(exact, lambda p: f"different version (v{p.version}); "
                                f"the question asks about v{wanted_version}")
    wants_latest = bool(_LATEST_RE.search(question))
    wants_oldest = bool(_OLDEST_RE.search(question))

    if missing_clause:
        return DocumentResolution(
            documents=[p.document for p in pool],
            reason=f"No ingested agreement contains {missing_clause}.",
            confidence="High",
            missing_clause=missing_clause,
            options=[_describe(p, False) for p in pool],
            signals=signals, rejected=rejected,
        )

    # 7. Corpus-wide questions and comparisons legitimately span documents —
    # but "all my agreements" means the operative ones, so each lineage still
    # contributes only its governing version.
    multi_ok = bool(policy and policy.multi_document)
    if _ALL_DOCS_RE.search(question):
        signals["scope"] = "all documents"
        latest = _latest_per_family(pool)
        older = [p for p in pool if p not in latest.values()]
        for p in older:
            rejected.setdefault(
                p.document, f"superseded version (v{p.version}) within its lineage")
        return DocumentResolution(
            documents=[p.document for p in latest.values()],
            reason="Question explicitly spans the whole corpus.",
            confidence="Medium",
            superseded=[f"{p.document} (v{p.version})" for p in older],
            signals=signals, rejected=rejected,
        )

    # 8. Active agreement — prior conversation context. Consulted only when
    # the current question identifies nothing itself, so "now the NDA" always
    # beats stickiness, and a lawyer asking five follow-ups about one
    # agreement is never re-asked which agreement they mean.
    context_note = ""
    doc_signal_keys = {"filename", "doc_type", "organization", "defined_terms",
                       "clause_refs"}
    if (history and not multi_ok
            and len({p.family for p in pool}) > 1
            and not (doc_signal_keys & signals.keys())):
        for prior in reversed(history):
            hits = set(_filename_mentions(prior, pool)) | _org_mentions(prior, pool)
            prior_types = _doc_type_mentions(prior, pool)
            hits |= {p.document for p in pool if p.doc_type in prior_types}
            if hits:
                signals["conversation"] = sorted(hits)
                narrow([p for p in pool if p.document in hits],
                       lambda p: "not the agreement the conversation has been about")
                context_note = (
                    f"Assumed the question continues about the "
                    f"{pool[0].doc_type_label} ({pool[0].document}) discussed "
                    "earlier in this conversation. Name another agreement to "
                    "switch."
                )
                break

    if len(pool) == 1:
        return DocumentResolution(
            documents=[pool[0].document],
            reason=f"Only {pool[0].document} matches the question's signals.",
            confidence="Medium" if context_note else ("High" if signals else "Medium"),
            assumption=context_note,
            signals=signals, rejected=rejected,
        )

    families = {p.family for p in pool}

    # 9. Single lineage, several versions -> version awareness decides.
    if len(families) == 1:
        ordered = sorted(pool, key=lambda p: version_key(p.version), reverse=True)
        if multi_ok or (wants_oldest and wants_latest):
            return DocumentResolution(
                documents=[p.document for p in ordered],
                reason="Comparison across versions of the same agreement.",
                confidence="High", signals=signals, rejected=rejected,
            )
        chosen = ordered[-1] if wants_oldest else ordered[0]
        rest = [p for p in ordered if p.document != chosen.document]
        for p in rest:
            rejected.setdefault(
                p.document,
                f"superseded version (v{p.version}); v{chosen.version} governs"
                if not wants_oldest else
                f"later version (v{p.version}); the question asks about the earliest",
            )
        return DocumentResolution(
            documents=[chosen.document],
            reason=(f"{'Earliest' if wants_oldest else 'Latest'} version of this "
                    f"agreement (v{chosen.version}); "
                    f"{len(rest)} other version(s) excluded."),
            confidence="Medium" if context_note else "High",
            assumption=context_note,
            superseded=[f"{p.document} (v{p.version})" for p in rest],
            signals=signals, rejected=rejected,
        )

    # 10. Genuinely different agreements -> compare only if asked, else rank.
    if multi_ok:
        latest = _latest_per_family(pool)
        return DocumentResolution(
            documents=[p.document for p in latest.values()],
            reason="Comparison across distinct agreements.",
            confidence="Medium", signals=signals, rejected=rejected,
        )

    # 11. Legal-concept affinity + defined-term coverage. One agreement being
    # substantially more likely than the rest is Medium confidence: proceed,
    # but say the assumption out loud so the user can override it. Only a
    # genuine tie is worth interrupting the user for.
    latest = _latest_per_family(pool)
    candidates = list(latest.values())
    for p in pool:
        if p not in candidates:
            rejected.setdefault(
                p.document, f"superseded version (v{p.version}) within its lineage")
    affinity = _concept_affinity(question, candidates)
    cand_terms = _defined_term_matches(question, candidates)
    scores = {
        p.document: affinity.get(p.document, 0.0) + len(cand_terms.get(p.document, ()))
        for p in candidates
    }
    candidates.sort(key=lambda p: scores[p.document], reverse=True)
    top = scores[candidates[0].document]
    runner_up = scores[candidates[1].document] if len(candidates) > 1 else 0.0

    if top > 0 and (runner_up == 0 or top >= 2 * runner_up):
        chosen = candidates[0]
        basis = []
        if cand_terms.get(chosen.document):
            quoted = ", ".join(f'"{t}"' for t in sorted(cand_terms[chosen.document])[:3])
            basis.append(f"it is the agreement that defines {quoted}")
        if affinity.get(chosen.document):
            basis.append("the legal issue in the question is a primary subject "
                         "of this agreement")
        signals["ranking"] = {p.document: scores[p.document] for p in candidates}
        for p in candidates[1:]:
            rejected.setdefault(
                p.document,
                "weaker match on the question's legal concepts and defined terms "
                f"than {chosen.doc_type_label} v{chosen.version}",
            )
        return DocumentResolution(
            documents=[chosen.document],
            reason=(f"Most likely target among {len(candidates)} candidate "
                    f"agreements ({' and '.join(basis)})."),
            confidence="Medium",
            assumption=(f"Assumed the question is about the {chosen.doc_type_label} "
                        f"v{chosen.version} ({chosen.document}) because "
                        f"{' and '.join(basis)}. Name the agreement to override "
                        "this assumption."),
            options=[_describe(p, True) for p in candidates],
            signals=signals, rejected=rejected,
        )

    # Ranked most-likely-first even on a tie, so the clarification list leads
    # with the best guesses.
    options = [_describe(p, True) for p in candidates]
    if allow_clarification:
        return DocumentResolution(
            documents=[],
            reason=f"{len(latest)} unrelated agreements match this question equally.",
            confidence="Low",
            needs_clarification=True,
            clarification=_clarification_text(question, candidates),
            options=options,
            signals=signals, rejected=rejected,
        )

    return DocumentResolution(
        documents=[p.document for p in candidates],
        reason="Ambiguous across agreements; clarification disabled.",
        confidence="Low", options=options, signals=signals, rejected=rejected,
    )


def _clarification_text(question: str, candidates: list[DocumentProfile]) -> str:
    listing = "\n".join(
        f"  {i}. {_describe(p, True)}" for i, p in enumerate(candidates, start=1)
    )
    example = f"{candidates[0].doc_type_label}" if candidates else "the agreement"
    return (
        "Which agreement should I answer from? The question matches more than "
        "one equally well, and mixing clauses across agreements would produce "
        f"an unreliable answer. Candidates, most likely first:\n\n{listing}\n\n"
        "Re-ask naming the agreement or a party to it (for example: "
        f"\"{question.rstrip('?')} in the {example}?\")."
    )
