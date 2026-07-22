"""Ingest-time extraction of structured legal facts from clause text.

Deterministic by construction. These outputs populate registers and calendars
that a lawyer will filter, export, and act on, so they must be reproducible:
the same contract yields the same rows on every run. An LLM pass would drift,
and "the obligation register changed but the contract didn't" is a trust
failure the product does not recover from.

Where the rules cannot reach, they extract nothing rather than guessing —
visible absence, not invisible invention.
"""

from . import dates, money, obligations, pipeline
from .pipeline import EXTRACTOR_VERSION, ExtractionReport, extract_document

__all__ = ["dates", "money", "obligations", "pipeline",
           "extract_document", "ExtractionReport", "EXTRACTOR_VERSION"]
