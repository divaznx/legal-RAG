"""Playbook-driven contract review.

Risk is not a property of a clause; it is a property of a clause relative to a
position. This package holds the position (`model.Playbook`) and the engine
that measures a document against it (`engine.review`).
"""

from . import engine, model
from .engine import review
from .model import Playbook, Rule, builtin, list_builtin, load

__all__ = ["engine", "model", "review", "Playbook", "Rule",
           "builtin", "list_builtin", "load"]
