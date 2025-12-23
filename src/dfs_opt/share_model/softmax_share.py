from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class FeatureSchema:
    """
    Parameterization that matches prompts/train-test.md but avoids huge dense one-hot matrices.

    We use an intercept + continuous features + per-category coefficients (reference level has coef 0).
    """

    intercept: bool = True
    continuous: Tuple[str, ...] = ("own_score_logprod", "own_max_log", "own_min_log", "avg_corr")

    # Canonical categorical levels (reference is levels[0])
    cpt_archetype_levels: Tuple[str, ...] = ("stud_1_2", "stud_3_5", "mid_6_10", "value_11_plus")
    stack_pattern_levels: Tuple[str, ...] = ("3-3", "4-2", "5-1")
    salary_left_bin_levels: Tuple[str, ...] = (
        "0_200",
        "200_500",
        "500_1000",
        "1000_2000",
        "2000_4000",
        "4000_8000",
        "8000_plus",
    )
    pct_gap_bin_levels: Tuple[str, ...] = (
        "0_0.01",
        "0.01_0.02",
        "0.02_0.04",
        "0.04_0.07",
        "0.07_0.15",
        "0.15_0.30",
        "0.30_plus",
    )

    def num_params(self) -> int:
        n = 0
        if self.intercept:
            n += 1
        n += len(self.continuous)
        # reference level dropped for each categorical
        n += (len(self.cpt_archetype_levels) - 1)
        n += (len(self.stack_pattern_levels) - 1)
        n += (len(self.salary_left_bin_levels) - 1)
        n += (len(self.pct_gap_bin_levels) - 1)
        return int(n)

    def param_names(self) -> List[str]:
        out: List[str] = []
        if self.intercept:
            out.append("intercept")
        out.extend([f"cont:{c}" for c in self.continuous])
        out.extend([f"cpt_archetype:{lvl}" for lvl in self.cpt_archetype_levels[1:]])
        out.extend([f"stack_pattern:{lvl}" for lvl in self.stack_pattern_levels[1:]])
        out.extend([f"salary_left_bin:{lvl}" for lvl in self.salary_left_bin_levels[1:]])
        out.extend([f"pct_gap_bin:{lvl}" for lvl in self.pct_gap_bin_levels[1:]])
        return out

    def _cat_maps(self) -> Dict[str, Dict[str, int]]:
        return {
            "cpt_archetype": {k: i for i, k in enumerate(self.cpt_archetype_levels)},
            "stack_pattern": {k: i for i, k in enumerate(self.stack_pattern_levels)},
            "salary_left_bin": {k: i for i, k in enumerate(self.salary_left_bin_levels)},
            "pct_proj_gap_to_optimal_bin": {k: i for i, k in enumerate(self.pct_gap_bin_levels)},
        }


@dataclass(frozen=True)
class FitConfig:
    l2_lambda: float = 1e-3
    max_iter: int = 200
    val_slate_frac: float = 0.2
    seed: int = 1337


@dataclass(frozen=True)
class UniverseSlateData:
    slate_id: str
    sport: str
    # continuous arrays (float32)
    cont: np.ndarray  # shape [n, 4]
    # categorical codes (uint8)
    cpt_arch_code: np.ndarray  # [n]
    stack_code: np.ndarray  # [n] in {0,1,2} corresponding to schema.stack_pattern_levels
    salary_left_bin_code: np.ndarray  # [n]
    pct_gap_bin_code: np.ndarray  # [n]

    @property
    def n_lineups(self) -> int:
        return int(self.cont.shape[0])


@dataclass(frozen=True)
class SlateSufficientStats:
    slate_id: str
    N_total: int
    sum_y: float  # should equal N_total
    sum_y_cont: np.ndarray  # [4]
    sum_y_cpt_arch: np.ndarray  # [levels]
    sum_y_stack: np.ndarray  # [levels]
    sum_y_salary_bin: np.ndarray  # [levels]
    sum_y_pct_gap_bin: np.ndarray  # [levels]
    warnings: Dict[str, Any]

    def to_feature_vector(self, schema: FeatureSchema) -> np.ndarray:
        parts: List[np.ndarray] = []
        if schema.intercept:
            parts.append(np.array([float(self.sum_y)], dtype=np.float64))
        parts.append(self.sum_y_cont.astype(np.float64, copy=False))
        parts.append(self.sum_y_cpt_arch[1:].astype(np.float64, copy=False))
        parts.append(self.sum_y_stack[1:].astype(np.float64, copy=False))
        parts.append(self.sum_y_salary_bin[1:].astype(np.float64, copy=False))
        parts.append(self.sum_y_pct_gap_bin[1:].astype(np.float64, copy=False))
        return np.concatenate(parts, axis=0)


def _safe_float_array(s: pa.ChunkedArray | pa.Array) -> np.ndarray:
    arr = s.combine_chunks() if isinstance(s, pa.ChunkedArray) else s
    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.float32)


def _dict_to_codes(
    col: pa.ChunkedArray | pa.Array,
    *,
    canonical_levels: Tuple[str, ...],
    unknown_to_ref: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert a dictionary-encoded string column into canonical uint8 codes aligned to canonical_levels.
    Any unknown label maps to 0 (reference) if unknown_to_ref else raises.
    """
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if not pa.types.is_dictionary(arr.type):
        # allow plain strings too
        vals = np.asarray(arr.to_numpy(zero_copy_only=False), dtype=object)
        mapping = {k: i for i, k in enumerate(canonical_levels)}
        codes = np.zeros(vals.shape[0], dtype=np.uint8)
        unknown = 0
        for i, v in enumerate(vals.tolist()):
            if v is None:
                unknown += 1
                continue
            idx = mapping.get(str(v))
            if idx is None:
                if not unknown_to_ref:
                    raise ValueError(f"Unknown categorical value '{v}' (levels={canonical_levels})")
                unknown += 1
                idx = 0
            codes[i] = np.uint8(idx)
        return codes, {"unknown_mapped_to_ref": int(unknown)}

    dict_arr = pa.DictionaryArray.from_arrays(arr.indices, arr.dictionary)
    dict_vals = [str(x) for x in dict_arr.dictionary.to_pylist()]
    mapping = {k: i for i, k in enumerate(canonical_levels)}

    trans = np.zeros(len(dict_vals), dtype=np.uint8)
    unknown = 0
    for old_idx, v in enumerate(dict_vals):
        new_idx = mapping.get(str(v))
        if new_idx is None:
            if not unknown_to_ref:
                raise ValueError(f"Unknown categorical value '{v}' (levels={canonical_levels})")
            unknown += 1
            new_idx = 0
        trans[old_idx] = np.uint8(new_idx)

    idx = np.asarray(dict_arr.indices.to_numpy(zero_copy_only=False), dtype=np.int64)
    # -1 indicates null; map to ref
    nulls = int((idx < 0).sum())
    idx = np.where(idx < 0, 0, idx)
    codes = trans[idx].astype(np.uint8, copy=False)
    return codes, {"unknown_dict_values_mapped_to_ref": int(unknown), "nulls_mapped_to_ref": int(nulls)}


def load_universe_slate(
    *,
    universe_root: Path,
    sport: str,
    slate_id: str,
    schema: FeatureSchema,
) -> Tuple[UniverseSlateData, Dict[str, Any]]:
    """
    Load the minimal per-lineup feature arrays from a precomputed universe directory.
    """
    slate_dir = universe_root / "dk-results" / "showdown" / sport.lower() / str(slate_id)
    lineups_path = slate_dir / "lineups_enriched.parquet"
    metadata_path = slate_dir / "metadata.json"
    if not lineups_path.exists():
        raise FileNotFoundError(f"Missing universe parquet for slate_id={slate_id}: {lineups_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing universe metadata.json for slate_id={slate_id}: {metadata_path}")

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    # stack_code map is stored as strings -> pattern
    stack_code_map = meta.get("stack_code_map") or {"0": "3-3", "1": "4-2", "2": "5-1"}
    inv_map = {int(k): str(v) for k, v in stack_code_map.items()}
    canonical_stack = {lvl: i for i, lvl in enumerate(schema.stack_pattern_levels)}
    # translate stack_code -> canonical stack_code in {0,1,2} according to schema order
    stack_trans = np.zeros(256, dtype=np.uint8)
    unknown_stack = 0
    for k, v in inv_map.items():
        idx = canonical_stack.get(str(v))
        if idx is None:
            unknown_stack += 1
            idx = 0
        stack_trans[int(k)] = np.uint8(idx)

    # Load columns with pyarrow
    cols = [
        *schema.continuous,
        "stack_code",
        "cpt_archetype",
        "salary_left_bin",
        "pct_proj_gap_to_optimal_bin",
    ]
    tbl = pq.read_table(lineups_path, columns=cols, memory_map=True)
    n = int(tbl.num_rows)
    if n <= 0:
        raise ValueError(f"Universe parquet has 0 rows for slate_id={slate_id}: {lineups_path}")

    cont = np.column_stack([_safe_float_array(tbl[c]) for c in schema.continuous]).astype(np.float32, copy=False)
    stack_code_raw = np.asarray(tbl["stack_code"].to_numpy(zero_copy_only=False), dtype=np.uint8)
    stack_code = stack_trans[stack_code_raw].astype(np.uint8, copy=False)

    cpt_arch_code, m1 = _dict_to_codes(tbl["cpt_archetype"], canonical_levels=schema.cpt_archetype_levels)
    salary_bin_code, m2 = _dict_to_codes(tbl["salary_left_bin"], canonical_levels=schema.salary_left_bin_levels)
    pct_gap_bin_code, m3 = _dict_to_codes(
        tbl["pct_proj_gap_to_optimal_bin"], canonical_levels=schema.pct_gap_bin_levels
    )

    metrics = {
        "slate_dir": str(slate_dir),
        "n_lineups": int(n),
        "stack_code_unknown_mapped_to_ref": int(unknown_stack),
        **{f"cpt_archetype.{k}": v for k, v in m1.items()},
        **{f"salary_left_bin.{k}": v for k, v in m2.items()},
        **{f"pct_gap_bin.{k}": v for k, v in m3.items()},
    }

    return (
        UniverseSlateData(
            slate_id=str(slate_id),
            sport=str(sport).lower(),
            cont=cont,
            cpt_arch_code=cpt_arch_code,
            stack_code=stack_code,
            salary_left_bin_code=salary_bin_code,
            pct_gap_bin_code=pct_gap_bin_code,
        ),
        metrics,
    )


def build_slate_sufficient_stats(
    entries: pd.DataFrame,
    *,
    slate_id: str,
    schema: FeatureSchema,
) -> SlateSufficientStats:
    """
    Compute per-slate sufficient stats from entry-level rows.

    We aggregate over contests:
      - y_c(L) = dup_count for lineup L within contest
      - objective sums over contests, but logZ is slate-dependent -> aggregate N and y across contests.
    """
    required = [
        "contest_id",
        "slate_id",
        "lineup_hash",
        "dup_count",
        "contest_size",
        "cpt_archetype",
        "stack_pattern",
        "salary_left_bin",
        "pct_proj_gap_to_optimal_bin",
        *schema.continuous,
    ]
    missing = [c for c in required if c not in entries.columns]
    if missing:
        raise ValueError(f"entries missing required columns for share model: {missing}")

    df = entries[entries["slate_id"] == slate_id].copy()
    if len(df) == 0:
        raise ValueError(f"No entry rows for slate_id={slate_id}")

    # contest sizes: sum unique contests
    contests = df[["contest_id", "contest_size"]].drop_duplicates()
    N_total = int(pd.to_numeric(contests["contest_size"], errors="coerce").fillna(0).sum())

    # y(L) aggregated over contests: for each contest,lineup_hash take max dup_count then sum across contests
    y_by_contest = (
        df.groupby(["contest_id", "lineup_hash"], dropna=False)["dup_count"].max().reset_index()
    )
    y_total = y_by_contest.groupby("lineup_hash", dropna=False)["dup_count"].sum().reset_index()
    y_total = y_total.rename(columns={"dup_count": "y_total"})

    # One representative feature row per lineup_hash within the slate (features should be slate-dependent)
    feat_cols = [
        "lineup_hash",
        "cpt_archetype",
        "stack_pattern",
        "salary_left_bin",
        "pct_proj_gap_to_optimal_bin",
        *schema.continuous,
    ]
    feat = df[feat_cols].drop_duplicates(subset=["lineup_hash"]).copy()
    obs = feat.merge(y_total, how="inner", on="lineup_hash", validate="one_to_one")

    sum_y = float(pd.to_numeric(obs["y_total"], errors="coerce").fillna(0.0).sum())

    cont = np.column_stack(
        [pd.to_numeric(obs[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) for c in schema.continuous]
    )
    y = pd.to_numeric(obs["y_total"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    sum_y_cont = (y[:, None] * cont).sum(axis=0)

    maps = schema._cat_maps()
    warnings: Dict[str, Any] = {}

    def weighted_counts(series: pd.Series, *, levels: Tuple[str, ...], key: str) -> np.ndarray:
        mapping = maps[key]
        codes = np.zeros(len(series), dtype=np.uint8)
        unknown = 0
        vals = series.astype("object").tolist()
        for i, v in enumerate(vals):
            idx = mapping.get(str(v))
            if idx is None:
                unknown += 1
                idx = 0
            codes[i] = np.uint8(idx)
        counts = np.zeros(len(levels), dtype=np.float64)
        for code, w in zip(codes.tolist(), y.tolist(), strict=True):
            counts[int(code)] += float(w)
        if unknown:
            warnings[f"{key}.unknown_mapped_to_ref"] = int(unknown)
        return counts

    sum_y_cpt_arch = weighted_counts(obs["cpt_archetype"], levels=schema.cpt_archetype_levels, key="cpt_archetype")
    sum_y_stack = weighted_counts(obs["stack_pattern"], levels=schema.stack_pattern_levels, key="stack_pattern")
    sum_y_salary = weighted_counts(obs["salary_left_bin"], levels=schema.salary_left_bin_levels, key="salary_left_bin")
    sum_y_gap = weighted_counts(
        obs["pct_proj_gap_to_optimal_bin"],
        levels=schema.pct_gap_bin_levels,
        key="pct_proj_gap_to_optimal_bin",
    )

    return SlateSufficientStats(
        slate_id=str(slate_id),
        N_total=int(N_total),
        sum_y=float(sum_y),
        sum_y_cont=sum_y_cont.astype(np.float64, copy=False),
        sum_y_cpt_arch=sum_y_cpt_arch.astype(np.float64, copy=False),
        sum_y_stack=sum_y_stack.astype(np.float64, copy=False),
        sum_y_salary_bin=sum_y_salary.astype(np.float64, copy=False),
        sum_y_pct_gap_bin=sum_y_gap.astype(np.float64, copy=False),
        warnings=warnings,
    )


def _unpack_theta(theta: np.ndarray, schema: FeatureSchema) -> Dict[str, Any]:
    """
    Convert flat theta vector to structured arrays for fast scoring.
    Reference level coef = 0 for each categorical group.
    """
    t = theta.astype(np.float64, copy=False)
    k = 0
    intercept = 0.0
    if schema.intercept:
        intercept = float(t[0])
        k = 1
    cont = t[k : k + len(schema.continuous)]
    k += len(schema.continuous)

    def cat_coefs(levels: Tuple[str, ...]) -> np.ndarray:
        nonlocal k
        out = np.zeros(len(levels), dtype=np.float64)
        # fill levels[1:]
        n = len(levels) - 1
        out[1:] = t[k : k + n]
        k += n
        return out

    cpt = cat_coefs(schema.cpt_archetype_levels)
    stack = cat_coefs(schema.stack_pattern_levels)
    salary = cat_coefs(schema.salary_left_bin_levels)
    gap = cat_coefs(schema.pct_gap_bin_levels)
    if k != len(t):
        raise ValueError(f"theta length mismatch: consumed={k} len={len(t)}")
    return {"intercept": intercept, "cont": cont, "cpt": cpt, "stack": stack, "salary": salary, "gap": gap}


def universe_logZ_and_exp_features(
    slate: UniverseSlateData, *, theta: np.ndarray, schema: FeatureSchema
) -> Tuple[float, np.ndarray]:
    """
    Compute:
      - logZ = log(sum_{L in U} exp(u(L)))
      - E_p[feature_vector] for the parameterization induced by schema.param_names()

    This runs in O(|U|) with stable logsumexp.
    """
    tt = _unpack_theta(theta, schema)

    u = (
        float(tt["intercept"])
        + slate.cont.astype(np.float64) @ tt["cont"].astype(np.float64)
        + tt["cpt"][slate.cpt_arch_code.astype(np.int64)]
        + tt["stack"][slate.stack_code.astype(np.int64)]
        + tt["salary"][slate.salary_left_bin_code.astype(np.int64)]
        + tt["gap"][slate.pct_gap_bin_code.astype(np.int64)]
    )

    # logZ (use SciPy's stable logsumexp)
    logZ = float(logsumexp(u))
    w = np.exp(u - logZ)  # p(L)

    # expected feature vector in the same order as SlateSufficientStats.to_feature_vector()
    parts: List[np.ndarray] = []
    if schema.intercept:
        parts.append(np.array([1.0], dtype=np.float64))

    # continuous means
    exp_cont = (w[:, None] * slate.cont.astype(np.float64)).sum(axis=0)
    parts.append(exp_cont.astype(np.float64, copy=False))

    # category probabilities per level, excluding reference later
    def cat_probs(codes: np.ndarray, n_levels: int) -> np.ndarray:
        counts = np.bincount(codes.astype(np.int64), weights=w, minlength=n_levels).astype(np.float64)
        # ensure sums to 1 within numeric tolerance
        return counts

    parts.append(cat_probs(slate.cpt_arch_code, len(schema.cpt_archetype_levels))[1:])
    parts.append(cat_probs(slate.stack_code, len(schema.stack_pattern_levels))[1:])
    parts.append(cat_probs(slate.salary_left_bin_code, len(schema.salary_left_bin_levels))[1:])
    parts.append(cat_probs(slate.pct_gap_bin_code, len(schema.pct_gap_bin_levels))[1:])
    exp_feat = np.concatenate(parts, axis=0)
    return logZ, exp_feat


def _nll_and_grad(
    theta: np.ndarray,
    *,
    schema: FeatureSchema,
    stats_by_slate: Dict[str, SlateSufficientStats],
    universe_by_slate: Dict[str, UniverseSlateData],
    l2_lambda: float,
    slate_ids: Iterable[str],
) -> Tuple[float, np.ndarray]:
    nll = 0.0
    grad = np.zeros_like(theta, dtype=np.float64)

    for sid in slate_ids:
        st = stats_by_slate[sid]
        univ = universe_by_slate[sid]

        sum_yx = st.to_feature_vector(schema)
        logZ, exp_feat = universe_logZ_and_exp_features(univ, theta=theta, schema=schema)
        N = float(st.N_total)

        nll += -(float(np.dot(theta, sum_yx)) - N * logZ)
        grad += -(sum_yx) + N * exp_feat

    # L2 regularization
    lam = float(l2_lambda)
    if lam > 0.0:
        nll += lam * float(np.dot(theta, theta))
        grad += 2.0 * lam * theta

    return float(nll), grad


def _split_slates(slate_ids: List[str], *, val_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    slates = list(dict.fromkeys([str(s) for s in slate_ids]))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(slates)
    n = len(slates)
    n_val = int(math.floor(float(val_frac) * n))
    n_val = max(1, n_val) if n >= 2 else 0
    val = slates[:n_val]
    train = slates[n_val:]
    if not train and val:
        train = val[:1]
        val = val[1:]
    return train, val


def _marginal_from_counts(counts: np.ndarray, *, levels: Tuple[str, ...]) -> Dict[str, float]:
    total = float(np.sum(counts))
    if total <= 0:
        return {lvl: 0.0 for lvl in levels}
    return {lvl: float(counts[i] / total) for i, lvl in enumerate(levels)}


def fit_softmax_share_model_bucket(
    *,
    bucket_entries: pd.DataFrame,
    gpp_category: str,
    universe_root: Path,
    artifacts_dir: Path,
    fit: FitConfig,
    schema: Optional[FeatureSchema] = None,
) -> Dict[str, Any]:
    """
    Fit the softmax share model for a single gpp_category bucket.

    Writes artifacts under:
      <artifacts_dir>/share_models/<gpp_category>/
    """
    if schema is None:
        schema = FeatureSchema()

    if "slate_id" not in bucket_entries.columns:
        raise ValueError("bucket_entries missing slate_id")

    sport = str(bucket_entries["sport"].iloc[0]).lower() if "sport" in bucket_entries.columns else gpp_category.split("-")[0]

    slates = sorted({str(x) for x in bucket_entries["slate_id"].dropna().astype(str).tolist()})
    if len(slates) == 0:
        raise ValueError(f"No slates found for gpp_category={gpp_category}")

    train_slates, val_slates = _split_slates(slates, val_frac=fit.val_slate_frac, seed=fit.seed)

    # Precompute sufficient stats + load universes once (fast iterations later)
    stats_by_slate: Dict[str, SlateSufficientStats] = {}
    universe_by_slate: Dict[str, UniverseSlateData] = {}
    load_metrics: Dict[str, Any] = {"universe_load": {}, "stats": {}}
    missing_universe: List[str] = []

    for sid in slates:
        st = build_slate_sufficient_stats(bucket_entries, slate_id=sid, schema=schema)
        stats_by_slate[sid] = st
        load_metrics["stats"][sid] = {
            "N_total": int(st.N_total),
            "sum_y": float(st.sum_y),
            "warnings": st.warnings,
        }
        try:
            univ, m = load_universe_slate(universe_root=universe_root, sport=sport, slate_id=sid, schema=schema)
            universe_by_slate[sid] = univ
            load_metrics["universe_load"][sid] = m
        except FileNotFoundError:
            missing_universe.append(str(sid))

    # Filter out slates without universes
    slates_ok = [s for s in slates if s in universe_by_slate]
    train_slates = [s for s in train_slates if s in universe_by_slate]
    val_slates = [s for s in val_slates if s in universe_by_slate]
    if len(slates_ok) == 0:
        raise ValueError(
            f"No slates with available universes for gpp_category={gpp_category} under universe_root={universe_root}"
        )
    if len(train_slates) == 0:
        # fallback: use all available
        train_slates = slates_ok
        val_slates = []

    theta0 = np.zeros(schema.num_params(), dtype=np.float64)

    t0 = time.perf_counter()

    def fun(th: np.ndarray) -> float:
        v, _ = _nll_and_grad(
            th,
            schema=schema,
            stats_by_slate=stats_by_slate,
            universe_by_slate=universe_by_slate,
            l2_lambda=fit.l2_lambda,
            slate_ids=train_slates,
        )
        return float(v)

    def jac(th: np.ndarray) -> np.ndarray:
        _, g = _nll_and_grad(
            th,
            schema=schema,
            stats_by_slate=stats_by_slate,
            universe_by_slate=universe_by_slate,
            l2_lambda=fit.l2_lambda,
            slate_ids=train_slates,
        )
        return g.astype(np.float64, copy=False)

    opt = minimize(
        fun=fun,
        x0=theta0,
        jac=jac,
        method="L-BFGS-B",
        options={"maxiter": int(fit.max_iter)},
    )

    theta_hat = opt.x.astype(np.float64, copy=False)
    train_nll, _ = _nll_and_grad(
        theta_hat,
        schema=schema,
        stats_by_slate=stats_by_slate,
        universe_by_slate=universe_by_slate,
        l2_lambda=0.0,
        slate_ids=train_slates,
    )
    val_nll = None
    if val_slates:
        val_nll, _ = _nll_and_grad(
            theta_hat,
            schema=schema,
            stats_by_slate=stats_by_slate,
            universe_by_slate=universe_by_slate,
            l2_lambda=0.0,
            slate_ids=val_slates,
        )

    # diagnostics on validation: predicted vs actual marginals
    def aggregate_actual(slates_sel: List[str]) -> Dict[str, Dict[str, float]]:
        if not slates_sel:
            return {}
        a_cpt = np.zeros(len(schema.cpt_archetype_levels), dtype=np.float64)
        a_stack = np.zeros(len(schema.stack_pattern_levels), dtype=np.float64)
        a_sal = np.zeros(len(schema.salary_left_bin_levels), dtype=np.float64)
        a_gap = np.zeros(len(schema.pct_gap_bin_levels), dtype=np.float64)
        for sid in slates_sel:
            st = stats_by_slate[sid]
            a_cpt += st.sum_y_cpt_arch
            a_stack += st.sum_y_stack
            a_sal += st.sum_y_salary_bin
            a_gap += st.sum_y_pct_gap_bin
        return {
            "cpt_archetype": _marginal_from_counts(a_cpt, levels=schema.cpt_archetype_levels),
            "stack_pattern": _marginal_from_counts(a_stack, levels=schema.stack_pattern_levels),
            "salary_left_bin": _marginal_from_counts(a_sal, levels=schema.salary_left_bin_levels),
            "pct_proj_gap_to_optimal_bin": _marginal_from_counts(a_gap, levels=schema.pct_gap_bin_levels),
        }

    def aggregate_predicted(slates_sel: List[str]) -> Dict[str, Dict[str, float]]:
        if not slates_sel:
            return {}
        # weight each slate by N_total to match the likelihood weighting
        p_cpt = np.zeros(len(schema.cpt_archetype_levels), dtype=np.float64)
        p_stack = np.zeros(len(schema.stack_pattern_levels), dtype=np.float64)
        p_sal = np.zeros(len(schema.salary_left_bin_levels), dtype=np.float64)
        p_gap = np.zeros(len(schema.pct_gap_bin_levels), dtype=np.float64)
        for sid in slates_sel:
            univ = universe_by_slate[sid]
            st = stats_by_slate[sid]
            _, exp_feat = universe_logZ_and_exp_features(univ, theta=theta_hat, schema=schema)
            # exp_feat packs: [1] + cont(4) + cat_nonref...; recover full probs per group
            k = 0
            if schema.intercept:
                k += 1
            k += len(schema.continuous)
            # cpt
            cpt_nonref = exp_feat[k : k + (len(schema.cpt_archetype_levels) - 1)]
            k += len(schema.cpt_archetype_levels) - 1
            cpt_full = np.zeros(len(schema.cpt_archetype_levels), dtype=np.float64)
            cpt_full[1:] = cpt_nonref
            cpt_full[0] = max(0.0, 1.0 - float(cpt_nonref.sum()))
            # stack
            stack_nonref = exp_feat[k : k + (len(schema.stack_pattern_levels) - 1)]
            k += len(schema.stack_pattern_levels) - 1
            stack_full = np.zeros(len(schema.stack_pattern_levels), dtype=np.float64)
            stack_full[1:] = stack_nonref
            stack_full[0] = max(0.0, 1.0 - float(stack_nonref.sum()))
            # salary
            sal_nonref = exp_feat[k : k + (len(schema.salary_left_bin_levels) - 1)]
            k += len(schema.salary_left_bin_levels) - 1
            sal_full = np.zeros(len(schema.salary_left_bin_levels), dtype=np.float64)
            sal_full[1:] = sal_nonref
            sal_full[0] = max(0.0, 1.0 - float(sal_nonref.sum()))
            # gap
            gap_nonref = exp_feat[k : k + (len(schema.pct_gap_bin_levels) - 1)]
            gap_full = np.zeros(len(schema.pct_gap_bin_levels), dtype=np.float64)
            gap_full[1:] = gap_nonref
            gap_full[0] = max(0.0, 1.0 - float(gap_nonref.sum()))

            w = float(st.N_total)
            p_cpt += w * cpt_full
            p_stack += w * stack_full
            p_sal += w * sal_full
            p_gap += w * gap_full

        return {
            "cpt_archetype": _marginal_from_counts(p_cpt, levels=schema.cpt_archetype_levels),
            "stack_pattern": _marginal_from_counts(p_stack, levels=schema.stack_pattern_levels),
            "salary_left_bin": _marginal_from_counts(p_sal, levels=schema.salary_left_bin_levels),
            "pct_proj_gap_to_optimal_bin": _marginal_from_counts(p_gap, levels=schema.pct_gap_bin_levels),
        }

    diag = {}
    if val_slates:
        diag = {
            "val_slates": val_slates,
            "actual": aggregate_actual(val_slates),
            "predicted": aggregate_predicted(val_slates),
        }

    elapsed = time.perf_counter() - t0

    out_dir = artifacts_dir / "share_models" / str(gpp_category)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    theta_payload = {
        "schema_version": 1,
        "gpp_category": str(gpp_category),
        "sport": str(sport),
        "feature_schema": {
            "intercept": bool(schema.intercept),
            "continuous": list(schema.continuous),
            "cpt_archetype_levels": list(schema.cpt_archetype_levels),
            "stack_pattern_levels": list(schema.stack_pattern_levels),
            "salary_left_bin_levels": list(schema.salary_left_bin_levels),
            "pct_gap_bin_levels": list(schema.pct_gap_bin_levels),
            "param_names": schema.param_names(),
        },
        "theta": {name: float(val) for name, val in zip(schema.param_names(), theta_hat.tolist(), strict=True)},
    }
    theta_path = out_dir / "theta.json"
    theta_path.write_text(json.dumps(theta_payload, indent=2, sort_keys=True), encoding="utf-8")

    fit_metrics = {
        "schema_version": 1,
        "gpp_category": str(gpp_category),
        "sport": str(sport),
        "optimizer": {
            "method": "L-BFGS-B",
            "success": bool(opt.success),
            "status": int(opt.status),
            "message": str(opt.message),
            "nit": int(opt.nit),
            "nfev": int(getattr(opt, "nfev", -1)),
        },
        "fit_config": asdict(fit),
        "timing": {"fit_time_s": float(elapsed)},
        "data": {
            "num_slates_total": int(len(slates)),
            "num_slates_loaded": int(len(slates_ok)),
            "train_slates": list(train_slates),
            "val_slates": list(val_slates),
            "missing_universe_slates": list(missing_universe),
        },
        "metrics": {
            "train_nll": float(train_nll),
            "train_nll_per_entry": float(
                train_nll
                / max(1.0, float(sum(stats_by_slate[s].N_total for s in train_slates)))
            ),
            "val_nll": (None if val_nll is None else float(val_nll)),
            "val_nll_per_entry": (
                None
                if val_nll is None
                else float(
                    val_nll
                    / max(1.0, float(sum(stats_by_slate[s].N_total for s in val_slates)))
                )
            ),
        },
        "load_metrics": load_metrics,
    }
    metrics_path = out_dir / "fit_metrics.json"
    metrics_path.write_text(json.dumps(fit_metrics, indent=2, sort_keys=True), encoding="utf-8")

    if diag:
        (diag_dir / "val_marginals.json").write_text(json.dumps(diag, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "gpp_category": str(gpp_category),
        "sport": str(sport),
        "theta_path": str(theta_path),
        "fit_metrics_path": str(metrics_path),
        "diagnostics_dir": str(diag_dir),
        "train_nll": float(train_nll),
        "val_nll": (None if val_nll is None else float(val_nll)),
        "opt_success": bool(opt.success),
        "missing_universe_slates": list(missing_universe),
    }


def dense_logZ_and_expX(X: np.ndarray, theta: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Helper for tests: given dense X [n,k] and theta [k], return (logZ, E_p[X]).
    """
    u = X @ theta
    logZ = float(logsumexp(u))
    w = np.exp(u - logZ)
    expX = (w[:, None] * X).sum(axis=0)
    return logZ, expX


def _load_theta_json(theta_json_path: Path, *, schema: FeatureSchema) -> np.ndarray:
    payload = json.loads(Path(theta_json_path).read_text(encoding="utf-8"))
    th = payload.get("theta")
    if not isinstance(th, dict):
        raise ValueError(f"{theta_json_path}: invalid theta payload (missing 'theta' mapping)")
    # Accept payload param order if present, else use schema.param_names()
    param_names = payload.get("feature_schema", {}).get("param_names")
    if not isinstance(param_names, list) or not param_names:
        param_names = schema.param_names()

    theta = np.zeros(schema.num_params(), dtype=np.float64)
    name_to_idx = {n: i for i, n in enumerate(schema.param_names())}
    missing = 0
    for n in param_names:
        if n not in name_to_idx:
            continue
        v = th.get(n)
        if v is None:
            missing += 1
            continue
        theta[name_to_idx[n]] = float(v)
    if missing:
        # ok: missing params treated as 0
        pass
    return theta


def score_universe_with_theta(
    *,
    lineups_enriched_parquet: Path,
    theta_json_path: Path,
    out_parquet_path: Path,
    schema: Optional[FeatureSchema] = None,
) -> Dict[str, Any]:
    """
    Apply a fitted softmax share model to a slate lineup universe parquet.

    Writes `out_parquet_path` with:
      - lineup_id (row index)
      - u (utility)
      - logp
      - p
    """
    if schema is None:
        schema = FeatureSchema()

    theta = _load_theta_json(Path(theta_json_path), schema=schema)
    tt = _unpack_theta(theta, schema)

    cols = [
        *schema.continuous,
        "stack_code",
        "cpt_archetype",
        "salary_left_bin",
        "pct_proj_gap_to_optimal_bin",
    ]
    tbl = pq.read_table(Path(lineups_enriched_parquet), columns=cols, memory_map=True)
    n = int(tbl.num_rows)
    if n <= 0:
        raise ValueError(f"{lineups_enriched_parquet}: empty universe parquet")

    cont = np.column_stack([_safe_float_array(tbl[c]) for c in schema.continuous]).astype(np.float64, copy=False)
    stack_code = np.asarray(tbl["stack_code"].to_numpy(zero_copy_only=False), dtype=np.int64)
    cpt_code, m1 = _dict_to_codes(tbl["cpt_archetype"], canonical_levels=schema.cpt_archetype_levels)
    salary_code, m2 = _dict_to_codes(tbl["salary_left_bin"], canonical_levels=schema.salary_left_bin_levels)
    gap_code, m3 = _dict_to_codes(
        tbl["pct_proj_gap_to_optimal_bin"], canonical_levels=schema.pct_gap_bin_levels
    )

    u = (
        float(tt["intercept"])
        + cont @ tt["cont"].astype(np.float64)
        + tt["cpt"][cpt_code.astype(np.int64)]
        + tt["stack"][stack_code.astype(np.int64)]
        + tt["salary"][salary_code.astype(np.int64)]
        + tt["gap"][gap_code.astype(np.int64)]
    ).astype(np.float64, copy=False)

    logZ = float(logsumexp(u))
    logp = (u - logZ).astype(np.float64, copy=False)
    p = np.exp(logp).astype(np.float64, copy=False)

    out_tbl = pa.table(
        {
            "lineup_id": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
            "u": pa.array(u, type=pa.float64()),
            "logp": pa.array(logp, type=pa.float64()),
            "p": pa.array(p, type=pa.float64()),
        }
    )
    pq.write_table(out_tbl, Path(out_parquet_path), compression="zstd")

    # Metrics: entropy and top-k mass
    p64 = p.astype(np.float64, copy=False)
    entropy = float(-np.sum(np.where(p64 > 0, p64 * np.log(p64), 0.0)))
    order = np.argsort(p64)[::-1]
    topk = [10, 100, 1000]
    topk_mass = {f"top_{k}_mass": float(np.sum(p64[order[: min(k, n)]])) for k in topk}

    return {
        "n_lineups": int(n),
        "logZ": float(logZ),
        "entropy": float(entropy),
        **topk_mass,
        **{f"cpt_archetype.{k}": v for k, v in m1.items()},
        **{f"salary_left_bin.{k}": v for k, v in m2.items()},
        **{f"pct_gap_bin.{k}": v for k, v in m3.items()},
        # lightweight preview of p for step preview
        "preview_p": p64[order[: min(200, n)]].tolist(),
    }


