"""Content-hash identifiers.

Every entity id is derived from the facts that make it, so consuming the same
fact twice yields the same id and idempotence comes for free. This mirrors
`dsh-superagent-prototype/src/core/ids.ts`; the reasoning is in
`docs/pah-dev-doc-v1.md` §3.1.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["key"]

_SEP = "\x1f"


def _canonical(value: Any) -> str:
    """Stable text for a hash input: json for containers, repr otherwise."""
    if isinstance(value, (dict, list, tuple, bool, int, float, str)) or value is None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return repr(value)


def key(*values: Any, length: int = 32) -> str:
    """A 32-hex-character id derived from `values`.

    Equal inputs give equal ids; reordering arguments gives a different id,
    because the order is part of the fact being named.
    """
    payload = _SEP.join(_canonical(v) for v in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
