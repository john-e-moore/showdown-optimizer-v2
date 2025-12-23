from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


@dataclass(frozen=True)
class ContestMeta:
    contest_id: str
    contest_size: int
    entry_fee: float
    max_entries: int
    payout_table: List[float]  # 1-indexed conceptually; stored as list where idx 0 is rank 1
    raw_source: str


def _coerce_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(str(x).strip().replace("$", "").replace(",", ""))
    except Exception:
        return None


def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(str(x).strip().replace("$", "").replace(",", ""))
    except Exception:
        return None


def _expand_payouts(payouts: List[Dict[str, Any]], *, contest_size: int) -> List[float]:
    """
    Expand payout ranges to a per-rank payout table length = contest_size.

    Accepts DK-like structures such as:
      {\"minRank\": 1, \"maxRank\": 1, \"amount\": 5000}
    """
    table = [0.0 for _ in range(int(contest_size))]
    for p in payouts:
        lo = _coerce_int(p.get("minRank") or p.get("min_rank") or p.get("rank_min")) or 0
        hi = _coerce_int(p.get("maxRank") or p.get("max_rank") or p.get("rank_max")) or lo
        amt = _coerce_float(p.get("amount") or p.get("payout") or p.get("value")) or 0.0
        lo = max(1, lo)
        hi = max(lo, hi)
        for r in range(lo, min(hi, contest_size) + 1):
            table[r - 1] = float(amt)
    return table


def _extract_meta(payload: Dict[str, Any], *, contest_id: str, source: str) -> ContestMeta:
    # Try a few common shapes: direct contest object or wrapped
    obj = payload
    for k in ["contest", "data", "result"]:
        if isinstance(obj.get(k), dict):
            obj = obj[k]
            break

    contest_size = (
        _coerce_int(obj.get("contestSize"))
        or _coerce_int(obj.get("contest_size"))
        or _coerce_int(obj.get("entries"))
        or _coerce_int(obj.get("size"))
    )
    entry_fee = _coerce_float(obj.get("entryFee")) or _coerce_float(obj.get("entry_fee"))
    max_entries = _coerce_int(obj.get("maxEntries")) or _coerce_int(obj.get("max_entries"))

    payouts_raw = None
    for k in ["payouts", "payout", "payoutStructure", "payout_structure"]:
        v = obj.get(k)
        if isinstance(v, list):
            payouts_raw = v
            break
        if isinstance(v, dict) and isinstance(v.get("payouts"), list):
            payouts_raw = v["payouts"]
            break

    if contest_size is None or entry_fee is None or max_entries is None or payouts_raw is None:
        raise ValueError(
            "Unable to parse contest metadata from DK payload. "
            f"contest_id={contest_id} source={source} top_keys={sorted(payload.keys())} "
            f"contest_keys={sorted(obj.keys())}"
        )

    payout_table = _expand_payouts(list(payouts_raw), contest_size=int(contest_size))
    if len(payout_table) != int(contest_size):
        raise ValueError(f"Invalid payout table length={len(payout_table)} contest_size={contest_size}")

    return ContestMeta(
        contest_id=str(contest_id),
        contest_size=int(contest_size),
        entry_fee=float(entry_fee),
        max_entries=int(max_entries),
        payout_table=payout_table,
        raw_source=str(source),
    )


class DkApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        cache_dir: Path,
        timeout_s: float = 20.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout_s = float(timeout_s)
        self.headers = dict(headers or {})
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, contest_id: str) -> Path:
        return self.cache_dir / f"contest_{contest_id}.json"

    def fetch_contest_json(self, contest_id: str) -> Dict[str, Any]:
        """
        Fetch contest JSON. Uses disk cache if present.

        Note: DraftKings has multiple internal/public endpoints; this client tries a small
        set of candidates and expects one to return JSON.
        """
        cid = str(contest_id)
        cache_path = self._cache_path(cid)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        # Candidate endpoints (best-effort); override base_url if needed.
        candidates = [
            f"{self.base_url}/contest/v1/contest/{cid}",
            f"{self.base_url}/contest/v1/contests/{cid}",
            f"{self.base_url}/contest/detail/v1/contests/{cid}",
            f"{self.base_url}/contests/v1/contests/{cid}",
        ]

        last_err: Optional[Exception] = None
        for url in candidates:
            try:
                with httpx.Client(timeout=self.timeout_s, headers=self.headers, follow_redirects=True) as client:
                    resp = client.get(url)
                if resp.status_code != 200:
                    continue
                ctype = resp.headers.get("content-type", "")
                if "json" not in ctype.lower():
                    # Some endpoints return HTML; skip those
                    continue
                payload = resp.json()
                cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                return payload
            except Exception as e:  # pragma: no cover (network-dependent)
                last_err = e
                continue

        msg = f"Failed to fetch DK contest JSON for contest_id={cid} via candidates under base_url={self.base_url}"
        if last_err is not None:
            msg += f" last_err={type(last_err).__name__}:{last_err}"
        raise RuntimeError(msg)

    def fetch_contest_meta(self, contest_id: str) -> ContestMeta:
        payload = self.fetch_contest_json(contest_id)
        return _extract_meta(payload, contest_id=str(contest_id), source=self.base_url)


