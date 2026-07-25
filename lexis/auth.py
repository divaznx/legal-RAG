"""API-key authentication and the tenant/role binding that comes with it.

A key is the only thing that names a tenant on the HTTP API. The tenant is
never taken from a header, a query parameter, or a request body, because any
of those would let a caller with a valid key for one client read another's
documents by editing a string — the whole isolation boundary would reduce to
a field the attacker controls.

Keys are stored as SHA-256 digests, never in plaintext: a stolen
`api_keys.json` should not be a stolen deployment. SHA-256 (rather than
bcrypt/argon2) is deliberate and sufficient here — the input is 32 bytes of
`secrets` entropy, not a human-chosen password, so there is no dictionary to
run and the slow-KDF argument does not apply. It also keeps verification off
the latency budget of every request.

Two roles:

  analyst  ask questions, list documents
  admin    everything, plus ingest, delete, and key management

The split exists because ingestion and deletion are the operations that
change what every future answer is grounded in. A reviewer who can ask
questions should not silently be able to remove the clause that makes an
answer inconvenient.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import tenancy
from .config import settings

KEY_PREFIX = "lxs_"

# admin is a superset of analyst; compared by rank, never by equality.
ROLES: dict[str, int] = {"analyst": 1, "admin": 2}

_lock = threading.Lock()
_cache: tuple[float, int, list[dict]] | None = None  # (mtime, size, records)


class AuthError(Exception):
    """Key store misuse (unknown role, bad tenant, duplicate label)."""


@dataclass(frozen=True)
class Principal:
    key_id: str
    label: str
    tenant: str
    role: str

    def can(self, required: str) -> bool:
        return ROLES.get(self.role, 0) >= ROLES[required]

    def as_dict(self) -> dict:
        return {"key_id": self.key_id, "label": self.label,
                "tenant": self.tenant, "role": self.role}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _load() -> list[dict]:
    """Records from disk, re-read whenever the file changes.

    Re-reading on change (rather than caching for the process lifetime) is
    what makes `keys revoke` take effect on the next request instead of the
    next restart — a revocation that needs a restart is not a revocation.
    """
    global _cache
    path = settings.api_keys_path
    if not path.exists():
        _cache = None
        return []
    stat = path.stat()
    if _cache is not None and _cache[0] == stat.st_mtime and _cache[1] == stat.st_size:
        return _cache[2]
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuthError(f"Key store at {path} is not valid JSON: {exc}") from exc
    _cache = (stat.st_mtime, stat.st_size, records)
    return records


def _save(records: list[dict]) -> None:
    global _cache
    path = settings.api_keys_path
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    _restrict(tmp)
    os.replace(tmp, path)
    _cache = None


def _restrict(path: Path) -> None:
    """Best-effort owner-only permissions.

    Effective on POSIX. On Windows the mode bits are largely ignored, so the
    key store's protection there is the ACL of the directory it sits in —
    documented rather than pretended away.
    """
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def issue(label: str, tenant: str, role: str = "analyst") -> tuple[Principal, str]:
    """Mint a key. The plaintext is returned once and never stored."""
    if role not in ROLES:
        raise AuthError(f"Unknown role {role!r}; expected one of {', '.join(ROLES)}.")
    slug = tenancy.normalize(tenant)
    label = label.strip()
    if not label:
        raise AuthError("A key needs a label — it is what the audit log attributes actions to.")

    secret = KEY_PREFIX + secrets.token_urlsafe(32)
    record = {
        "id": secrets.token_hex(4),
        "hash": _digest(secret),
        "label": label,
        "tenant": slug,
        "role": role,
        "created_at": _now(),
        "revoked_at": None,
    }
    with _lock:
        records = list(_load())
        records.append(record)
        _save(records)
    return _principal(record), secret


def verify(presented: str | None) -> Principal | None:
    """Resolve a presented key to a Principal, or None."""
    if not presented:
        return None
    candidate = _digest(presented.strip())
    for record in _load():
        if record.get("revoked_at"):
            continue
        if hmac.compare_digest(record.get("hash", ""), candidate):
            return _principal(record)
    return None


def _principal(record: dict) -> Principal:
    return Principal(
        key_id=record["id"],
        label=record["label"],
        tenant=record["tenant"],
        role=record.get("role", "analyst"),
    )


def revoke(key_id: str) -> bool:
    with _lock:
        records = list(_load())
        for record in records:
            if record["id"] == key_id and not record.get("revoked_at"):
                record["revoked_at"] = _now()
                _save(records)
                return True
    return False


def list_keys(include_revoked: bool = False) -> list[dict]:
    """Key metadata without the digests — safe to print or return over HTTP."""
    out = []
    for record in _load():
        if record.get("revoked_at") and not include_revoked:
            continue
        out.append({k: v for k, v in record.items() if k != "hash"})
    return sorted(out, key=lambda r: (r["tenant"], r["created_at"]))


def has_keys() -> bool:
    return any(not r.get("revoked_at") for r in _load())


def bootstrap() -> tuple[Principal, str] | None:
    """Mint the first admin key if the store is empty.

    An appliance that boots with authentication enabled and no way in is a
    support call; one that boots with a blank password is a breach. Minting a
    random key and printing it once is the middle path, and it is why
    `auth_enabled` can default to true without making first-run painful.
    """
    with _lock:
        if any(not r.get("revoked_at") for r in _load()):
            return None
    return issue("bootstrap admin", settings.default_tenant, "admin")
