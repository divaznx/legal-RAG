"""Structured store for the legal object model.

Retrieval answers questions. This answers "show me every renewal in the next
90 days", "which contracts put uncapped indemnity on us", "what does this
vendor owe us and when" — reads that no vector store can serve, and which are
the actual product.
"""

from . import db, models, repository
from .models import Finding, KeyDate, MoneyTerm, Obligation

__all__ = ["db", "models", "repository", "Finding", "KeyDate", "MoneyTerm", "Obligation"]
