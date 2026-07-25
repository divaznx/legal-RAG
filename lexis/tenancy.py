"""Tenant scoping — the boundary that lets one deployment hold more than one
client's documents.

Isolation is **collection-per-tenant**, not a payload filter. Qdrant supports
both, and payload partitioning scales to more tenants, but it makes isolation
a property of every query being written correctly: one `fetch_*` helper that
forgets the tenant condition leaks another client's contract into an answer,
and nothing fails loudly when it happens. This module makes the tenant part
of the collection *name*, so a missing filter cannot leak — there is no
shared collection to leak from. For a deployment holding a handful of
clients or practice groups, that trade is the right way round.

The tenant travels in a ContextVar rather than a parameter threaded through
every call, because the storage seams are only three (`vector_store`
collection, `ingest` manifest, `cache` file) while the call sites between
them — planner, retrieval, engine — are many and have no business knowing
about tenants.

ContextVars and threads: a value set in an async context propagates *into*
worker threads (anyio copies the context), but a value set inside a worker
thread does not propagate back out. So the API binds the tenant in an
`async def` dependency, never a sync one — see api.py.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from .config import settings

# A tenant id becomes both a Qdrant collection suffix and a directory name,
# so it is validated as a strict slug rather than trusted. Without this,
# a tenant of "../../etc" is a path traversal in `data_path()`.
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

_current: ContextVar[str] = ContextVar("lexis_tenant")


class InvalidTenant(ValueError):
    """Raised when a tenant id is not a safe slug."""


def normalize(tenant: str) -> str:
    """Validate and canonicalise a tenant id, or raise InvalidTenant."""
    slug = (tenant or "").strip().lower()
    if not _SLUG.match(slug):
        raise InvalidTenant(
            f"Invalid tenant id {tenant!r}: use 1-63 characters of a-z, 0-9, "
            "'-' or '_', starting with a letter or digit."
        )
    return slug


def current() -> str:
    return _current.get(settings.default_tenant)


def set_current(tenant: str) -> str:
    """Bind the tenant for the rest of this context. Returns the normalized id."""
    slug = normalize(tenant)
    _current.set(slug)
    return slug


@contextmanager
def using(tenant: str) -> Iterator[str]:
    """Bind a tenant for the duration of the block, then restore the previous one.

    For ordinary straight-line code: the CLI, the UI, tests. To scope a
    generator that a server iterates, use `scoped()` instead — see below.
    """
    slug = normalize(tenant)
    token = _current.set(slug)
    try:
        yield slug
    finally:
        try:
            _current.reset(token)
        except ValueError:
            # The block exited in a different Context from the one it
            # entered, so the token is not resettable here. That means the
            # entering context was discarded anyway and there is nothing to
            # restore — raising would turn a no-op into a crash.
            pass


def scoped(tenant: str, iterator):
    """Yield from `iterator` with `tenant` bound around every step.

    Streaming responses cannot use `using()`. Starlette pulls a sync
    generator through `iterate_in_threadpool`, which runs each `__next__` in
    a *fresh copy* of the context: a `set()` performed on one step is gone by
    the next one, and the token from step one cannot be reset on step five
    ("was created in a different Context"). Wrapping the generator body in a
    context manager therefore binds the tenant for the first chunk only, and
    then raises on close.

    Re-binding before each step is the shape that actually holds: there is no
    token to reset, and every resumption starts with the tenant set.

    Note that it deliberately does not restore the previous binding. Under a
    server each step's context is thrown away regardless, so there is nothing
    to restore; called from ordinary straight-line code it will leave the
    tenant bound, which is why `using()` remains the right tool there.
    """
    slug = normalize(tenant)
    while True:
        _current.set(slug)
        try:
            item = next(iterator)
        except StopIteration:
            return
        yield item


def collection(tenant: str | None = None) -> str:
    """Qdrant collection holding this tenant's chunks."""
    return f"{settings.qdrant_collection}__{normalize(tenant or current())}"


def data_path(tenant: str | None = None) -> Path:
    """Per-tenant state directory (manifest, answer cache, uploads)."""
    path = settings.data_path / "tenants" / normalize(tenant or current())
    path.mkdir(parents=True, exist_ok=True)
    return path


def known_tenants() -> list[str]:
    """Tenants with state on disk. Advisory only — the key store is the
    authority on who may reach one."""
    root = settings.data_path / "tenants"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())
