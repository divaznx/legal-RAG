"""Tests for the platform layer: object model, extraction, playbook review.

Run:  python -m pytest tests/ -q     (or: python tests/test_platform.py)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexis.extraction import dates, money, obligations
from lexis.extraction.dates import resolve
from lexis.extraction.text import normalize, parse_count, sentences
from lexis.playbook import model as playbook_model


# --- text normalisation ---------------------------------------------------

def test_normalization_preserves_original_offsets():
    """Contract text is hard-wrapped, so patterns must run over collapsed text
    while spans still index the original — a fact that cannot be highlighted
    in the source is one a lawyer has to re-verify by hand."""
    raw = "gives not\nless than ninety (90) days'\nwritten notice"
    norm = normalize(raw)
    assert "not less than ninety (90) days' written notice" in norm.text
    start = norm.text.index("not less than")
    assert raw[norm.origin(start):].startswith("not\nless than")


def test_parse_count_prefers_parenthesised_digits():
    assert parse_count("thirty (30) days") == 30
    assert parse_count("seventy-two (72) hours") == 72
    assert parse_count("ninety days") == 90
    assert parse_count("thirty-six months") == 36
    assert parse_count("no number here") is None


# --- obligations ----------------------------------------------------------

def test_obligation_modality_is_never_collapsed():
    """"may suspend" and "shall suspend" differ by one word and by the entire
    allocation of risk. A register that renders rights as duties is worse
    than no register."""
    text = ("The Supplier may suspend access to the Platform. "
            "The Customer shall pay all invoices. "
            "The Customer shall not resell the Platform.")
    found = obligations.extract(text, "d", "c")
    by_action = {o.action[:12]: o for o in found}
    assert by_action["suspend acce"].modality == "right"
    assert by_action["pay all invo"].modality == "obligation"
    assert by_action["resell the P"].modality == "prohibition"


def test_obligation_captures_party_condition_and_deadline():
    text = ("4.3 The Supplier may suspend access if any undisputed invoice "
            "remains unpaid for more than forty-five (45) days.")
    found = obligations.extract(text, "d", "c")
    assert len(found) == 1
    o = found[0]
    assert o.obligor == "Supplier"
    assert o.modality == "right"
    assert "undisputed invoice" in (o.condition or "")


def test_obligation_ignores_non_actor_subjects():
    """"This Agreement shall commence on the Effective Date" is not a duty
    borne by anybody."""
    text = "This Agreement shall commence on the Effective Date."
    assert obligations.extract(text, "d", "c") == []


def test_obligation_survives_line_wrapping():
    wrapped = "The Customer shall pay\nall Subscription Fees within thirty (30)\ndays."
    found = obligations.extract(wrapped, "d", "c")
    assert found and found[0].obligor == "Customer"
    assert found[0].deadline_days == 30


# --- dates ----------------------------------------------------------------

def test_relative_deadline_captures_anchor_and_direction():
    text = ("unless either party gives not less than ninety (90) days' written "
            "notice before the end of the then-current term.")
    found = [d for d in dates.extract(text, "d", "c") if d.rule_type == "relative"]
    assert found
    rule = found[0]
    assert rule.days == 90 and rule.direction == "before" and rule.anchor == "term_end"


def test_thereafter_does_not_match_as_after():
    """Without a left word boundary, "thereafter" matches "after" and its
    trailing group swallows the renewal deadline that follows it."""
    text = ("continues for an initial term of thirty-six (36) months, and thereafter "
            "renews for successive twelve (12) month periods unless either party gives "
            "not less than ninety (90) days' written notice before the end of the "
            "then-current term.")
    found = dates.extract(text, "d", "c")
    assert any(d.days == 90 and d.anchor == "term_end" for d in found)
    assert any(d.rule_type == "duration" and d.days == 36 for d in found)


def test_only_the_agreement_term_is_anchored_to_the_effective_date():
    """A force majeure event "continuing for more than sixty (60) days" is a
    duration too, but it runs from an event that may never occur. Anchoring it
    to signature would put a confident, meaningless date in the calendar."""
    term = dates.extract("continues for an initial term of twelve (12) months.", "d", "c")
    assert any(d.anchor == "effective_date" for d in term)

    fm = dates.extract("If a force majeure event continues for more than sixty (60) "
                       "days, either party may terminate.", "d", "c")
    assert all(d.anchor != "effective_date" for d in fm if d.rule_type == "duration")


def test_month_arithmetic_is_calendar_correct():
    """A 30-day approximation puts a 36-month term two weeks early, and the
    renewal notice hanging off it two weeks earlier still."""
    anchor = date(2025, 3, 14)
    term_end = resolve({"rule_type": "relative", "days": 36, "unit": "month",
                        "direction": "after"}, anchor)
    assert term_end == date(2028, 3, 14)

    notice = resolve({"rule_type": "relative", "days": 90, "unit": "day",
                      "direction": "before"}, term_end)
    assert notice == date(2027, 12, 15)


def test_month_arithmetic_clamps_short_months():
    assert resolve({"rule_type": "relative", "days": 1, "unit": "month",
                    "direction": "after"}, date(2025, 1, 31)) == date(2025, 2, 28)
    assert resolve({"rule_type": "relative", "days": 1, "unit": "month",
                    "direction": "after"}, date(2024, 1, 31)) == date(2024, 2, 29)


# --- money ----------------------------------------------------------------

def test_cap_with_no_figure_is_still_a_cap():
    """The most common cap in commercial drafting contains no digit at all,
    and the negation sits in the subject: "NEITHER PARTY'S aggregate liability
    shall exceed the fees paid". Missing it reports a capped contract as
    uncapped — a blocker finding that is simply false."""
    text = ("Neither party's aggregate liability under this Agreement shall exceed "
            "the fees paid or payable in the twelve (12) months preceding the claim.")
    caps = [m for m in money.extract(text, "d", "c") if m.kind == "cap"]
    assert caps and caps[0].multiplier == 1.0


def test_relative_and_percentage_caps():
    two_x = money.extract("liability shall not exceed two times (2x) the fees paid.", "d", "c")
    assert any(m.kind == "cap" and m.multiplier == 2.0 for m in two_x)

    pct = money.extract("total aggregate liability is limited to one hundred and fifty "
                        "percent (150%) of the Subscription Fees.", "d", "c")
    assert any(m.kind == "cap" and m.multiplier == 1.5 for m in pct)


def test_money_is_classified_by_legal_function():
    fees = money.extract("The Customer shall pay Subscription Fees of GBP 96,000 per annum.",
                         "d", "c")
    assert any(m.kind == "fee" and m.amount == 96000 and m.currency == "GBP" for m in fees)

    interest = money.extract("Overdue amounts accrue interest at four percent (4%) per annum.",
                             "d", "c")
    assert any(m.kind == "interest" for m in interest)


# --- playbook -------------------------------------------------------------

def test_builtin_playbook_loads_and_validates():
    book = playbook_model.builtin("saas_customer")
    assert book.position == "customer" and book.rules
    for rule in book.rules:
        assert rule.rationale, f"{rule.id} has no rationale"
        assert rule.check.type in playbook_model.CHECK_TYPES
        # A finding a lawyer cannot argue from is a finding they will ignore.
        assert rule.suggested, f"{rule.id} suggests no alternative language"


def test_playbook_rejects_invalid_definitions():
    for bad in (
        {"id": "p", "name": "n", "rules": [
            {"id": "r", "title": "t", "concept": "c", "check": {"type": "nonsense"}}]},
        {"id": "p", "name": "n", "rules": [
            {"id": "r", "title": "t", "concept": "c", "severity": "catastrophic"}]},
    ):
        try:
            playbook_model.from_dict(bad)
        except ValueError:
            continue
        raise AssertionError(f"invalid playbook accepted: {bad}")


# --- reproducibility ------------------------------------------------------

def test_concept_expansion_is_order_stable_and_interleaved():
    """Two properties, both load-bearing for reproducibility.

    STABLE: `list(set(...))` iterates in hash order, which Python randomises
    per process. The expansion order decides which concepts get a retrieval
    probe within the probe budget, so a set here made the same question return
    different evidence on different runs — two lawyers reviewing the same
    contract getting different answers.

    INTERLEAVED: the budget is small, so draining one concept's neighbourhood
    before touching the next spends every probe near the first seed.
    """
    from lexis.legal.ontology import expand_related

    seeds = ["services", "survival", "termination"]
    first = expand_related(seeds, depth=1)
    assert all(expand_related(seeds, depth=1) == first for _ in range(5))

    # each seed's neighbours must appear, not just the first seed's
    tail = [c for c in first if c not in seeds]
    from lexis.legal.ontology import CONCEPTS
    reached = {seed for seed in seeds
               if any(n in tail for n in CONCEPTS[seed].related)}
    assert len(reached) >= 2, f"expansion starved some seeds: {tail}"


def test_definition_targets_are_order_stable():
    from lexis.legal.definitions import definition_targets

    known = {"confidential information", "personal data", "platform",
             "subscription fees", "authorised user"}
    question = "How is Confidential Information handled under the Platform terms?"
    first = definition_targets(question, known)
    assert all(definition_targets(question, known) == first for _ in range(5))


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok    {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"{'ALL PASS' if not failures else f'{failures} FAILURE(S)'}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
