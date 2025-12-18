from __future__ import annotations

import hashlib
import json
from typing import Iterable, List, Mapping, Sequence


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_lineup_string(cpt_name_norm: str, util_name_norms: Sequence[str]) -> str:
    # stable, explicit, and platform-independent
    parts: List[str] = [f"CPT:{cpt_name_norm}"]
    for i, n in enumerate(util_name_norms, start=1):
        parts.append(f"UTIL{i}:{n}")
    return "|".join(parts)


def lineup_hash(cpt_name_norm: str, util_name_norms: Sequence[str]) -> str:
    s = canonical_lineup_string(cpt_name_norm, util_name_norms)
    return sha256_hex(s.encode("utf-8"))


def schema_fingerprint(columns_and_dtypes: Iterable[tuple[str, str]]) -> str:
    """
    Hash of canonicalized (column_name, dtype) pairs.
    Dtypes should be stable strings (e.g. pandas dtype str()).
    """
    payload = [{"col": c, "dtype": d} for (c, d) in columns_and_dtypes]
    b = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_hex(b)


def data_fingerprint(preview_rows: Sequence[Mapping[str, object]]) -> str:
    """
    Hash of preview rows (already truncated to <=200) to allow quick regression checks.
    """
    b = json.dumps(list(preview_rows), sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return sha256_hex(b)


