from __future__ import annotations

import re

_NAME_CLEAN_RE = re.compile(r"[\.\'\-]")
_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", re.IGNORECASE)


def norm_name(s: str) -> str:
    """
    Normalize player names for cross-source joins.
    Rules are intentionally simple + deterministic.
    """
    s = (s or "").lower()
    s = _NAME_CLEAN_RE.sub("", s)
    s = _SUFFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


