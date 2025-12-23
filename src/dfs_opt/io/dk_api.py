from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

_DEBUG_LOG_PATH = "/home/john/showdown-optimizer-v2/.cursor/debug.log"


def _dbg_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    """Append one NDJSON debug line. Best-effort; never raises."""
    try:
        payload = {
            "sessionId": "debug-session",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(__import__("time").time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


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
    for k in ["contestDetail", "contest", "data", "result"]:
        if isinstance(obj.get(k), dict):
            obj = obj[k]
            break

    contest_size = (
        _coerce_int(obj.get("maximumEntries"))
        or _coerce_int(obj.get("contestSize"))
        or _coerce_int(obj.get("contest_size"))
        or _coerce_int(obj.get("entries"))
        or _coerce_int(obj.get("size"))
    )
    entry_fee = _coerce_float(obj.get("entryFee")) or _coerce_float(obj.get("entry_fee"))
    max_entries = (
        _coerce_int(obj.get("maximumEntriesPerUser"))
        or _coerce_int(obj.get("maxEntries"))
        or _coerce_int(obj.get("max_entries"))
    )

    payouts_raw = None
    for k in ["payouts", "payout", "payoutStructure", "payout_structure", "payoutSummary"]:
        v = obj.get(k)
        if isinstance(v, list):
            payouts_raw = v
            break
        if isinstance(v, dict) and isinstance(v.get("payouts"), list):
            payouts_raw = v["payouts"]
            break

    # DK contestDetail payloads often expose payouts as payoutSummary tiers:
    # [{minPosition, maxPosition, payoutDescriptions:[{value:...}, ...]}, ...]
    if payouts_raw is not None and isinstance(payouts_raw, list) and payouts_raw and isinstance(payouts_raw[0], dict):
        if ("minPosition" in payouts_raw[0]) or ("maxPosition" in payouts_raw[0]) or ("payoutDescriptions" in payouts_raw[0]):
            tiers = payouts_raw
            normalized: List[Dict[str, Any]] = []
            for t in tiers:
                if not isinstance(t, dict):
                    continue
                descs = t.get("payoutDescriptions")
                amount = None
                if isinstance(descs, list) and descs and isinstance(descs[0], dict):
                    amount = descs[0].get("value") or descs[0].get("amount") or descs[0].get("payout")
                normalized.append(
                    {
                        "minRank": t.get("minRank") or t.get("minPosition") or t.get("min_rank") or t.get("rank_min"),
                        "maxRank": t.get("maxRank") or t.get("maxPosition") or t.get("max_rank") or t.get("rank_max"),
                        "amount": t.get("amount") or t.get("payout") or t.get("value") or amount,
                    }
                )
            payouts_raw = normalized

    if contest_size is None or entry_fee is None or max_entries is None or payouts_raw is None:
        # #region agent log
        _dbg_log(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="src/dfs_opt/io/dk_api.py:_extract_meta",
            message="meta_parse_failed",
            data={
                "contest_id": str(contest_id),
                "source": str(source),
                "top_keys": sorted(list(payload.keys()))[:50],
                "contest_keys": sorted(list(obj.keys()))[:50],
                "contest_size": contest_size,
                "entry_fee": entry_fee,
                "max_entries": max_entries,
                "has_payouts_raw": payouts_raw is not None,
                "errorStatus_type": type(payload.get("errorStatus")).__name__,
                "contestDetail_type": type(payload.get("contestDetail")).__name__,
            },
        )
        # #endregion agent log
        raise ValueError(
            "Unable to parse contest metadata from DK payload. "
            f"contest_id={contest_id} source={source} top_keys={sorted(payload.keys())} "
            f"contest_keys={sorted(obj.keys())}"
        )

    payout_table = _expand_payouts(list(payouts_raw), contest_size=int(contest_size))
    if len(payout_table) != int(contest_size):
        raise ValueError(f"Invalid payout table length={len(payout_table)} contest_size={contest_size}")

    # #region agent log
    _dbg_log(
        run_id="pre-fix",
        hypothesis_id="H2",
        location="src/dfs_opt/io/dk_api.py:_extract_meta",
        message="meta_parse_ok",
        data={
            "contest_id": str(contest_id),
            "contest_size": int(contest_size),
            "entry_fee": float(entry_fee),
            "max_entries": int(max_entries),
            "payout_table_len": int(len(payout_table)),
            "payout_rank1": float(payout_table[0]) if payout_table else None,
        },
    )
    # #endregion agent log

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
        # #region agent log
        _dbg_log(
            run_id="pre-fix",
            hypothesis_id="H3",
            location="src/dfs_opt/io/dk_api.py:DkApiClient.fetch_contest_json",
            message="fetch_contest_json_start",
            data={
                "contest_id": cid,
                "base_url": self.base_url,
                "cache_exists": bool(cache_path.exists()),
                "cache_path": str(cache_path),
            },
        )
        # #endregion agent log
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            # #region agent log
            _dbg_log(
                run_id="pre-fix",
                hypothesis_id="H3",
                location="src/dfs_opt/io/dk_api.py:DkApiClient.fetch_contest_json",
                message="fetch_contest_json_cache_hit",
                data={
                    "contest_id": cid,
                    "payload_type": type(payload).__name__,
                    "top_keys": (sorted(list(payload.keys()))[:50] if isinstance(payload, dict) else ["<non-dict>"]),
                    "has_errorStatus": bool(isinstance(payload, dict) and payload.get("errorStatus") is not None),
                },
            )
            # #endregion agent log
            return payload

        # Candidate endpoints (best-effort); override base_url if needed.
        candidates = [
            # Preferred public endpoint format (explicit JSON)
            f"{self.base_url}/contests/v1/contests/{cid}?format=json",
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
                # #region agent log
                _dbg_log(
                    run_id="pre-fix",
                    hypothesis_id="H1",
                    location="src/dfs_opt/io/dk_api.py:DkApiClient.fetch_contest_json",
                    message="candidate_response",
                    data={
                        "contest_id": cid,
                        "url": url,
                        "status_code": int(resp.status_code),
                        "content_type": str(resp.headers.get("content-type", ""))[:200],
                    },
                )
                # #endregion agent log
                if resp.status_code != 200:
                    continue
                ctype = resp.headers.get("content-type", "")
                if "json" not in ctype.lower():
                    # Some endpoints return HTML; skip those
                    continue
                payload = resp.json()
                # #region agent log
                _dbg_log(
                    run_id="pre-fix",
                    hypothesis_id="H1",
                    location="src/dfs_opt/io/dk_api.py:DkApiClient.fetch_contest_json",
                    message="candidate_payload_shape",
                    data={
                        "contest_id": cid,
                        "url": url,
                        "payload_type": type(payload).__name__,
                        "top_keys": (sorted(list(payload.keys()))[:50] if isinstance(payload, dict) else ["<non-dict>"]),
                        "has_errorStatus": bool(isinstance(payload, dict) and payload.get("errorStatus") is not None),
                    },
                )
                # #endregion agent log
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


