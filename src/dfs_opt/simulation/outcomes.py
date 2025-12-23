from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from dfs_opt.features.correlation import load_corr_matrix_csv
from dfs_opt.parsing.names import norm_name


@dataclass(frozen=True)
class OutcomeSimSpec:
    mean: np.ndarray  # float64[n]
    std: np.ndarray  # float64[n]
    corr: np.ndarray  # float64[n,n]


def _nearest_psd(corr: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """
    Ensure correlation matrix is PSD by eigenvalue clipping.
    """
    w, v = np.linalg.eigh(corr)
    w = np.maximum(w, float(eps))
    out = (v * w) @ v.T
    # re-normalize diagonal to 1
    d = np.sqrt(np.clip(np.diag(out), 1e-12, 1e12))
    out = out / d[:, None] / d[None, :]
    return out


def build_outcome_spec(
    *,
    players_df: pd.DataFrame,
    corr_matrix_csv: Path,
    std_mode: str = "dk_std_or_fallback",
    std_scale: float = 1.0,
) -> Tuple[OutcomeSimSpec, dict]:
    """
    Build outcome sim spec aligned to `players_df` row order (name_norm must exist).

    std_mode:
      - \"dk_std_or_fallback\": use `dk_std` if present else `std = std_scale*sqrt(max(proj,eps))`
    """
    if "name_norm" not in players_df.columns:
        raise ValueError("players_df missing name_norm")
    if "proj_points" not in players_df.columns:
        raise ValueError("players_df missing proj_points")

    n = int(len(players_df))
    proj = pd.to_numeric(players_df["proj_points"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    mean = proj.astype(np.float64, copy=False)
    eps = 1e-6

    std = None
    used = "fallback"
    if std_mode == "dk_std_or_fallback" and "dk_std" in players_df.columns:
        s = pd.to_numeric(players_df["dk_std"], errors="coerce")
        if s.notna().any():
            std = s.fillna(0.0).to_numpy(dtype=np.float64)
            used = "dk_std"
    if std is None:
        std = float(std_scale) * np.sqrt(np.clip(mean, eps, 1e12))
        used = "fallback_sqrt_proj"

    # Build dense corr aligned to players_df via name normalization lookup
    lookup = load_corr_matrix_csv(Path(corr_matrix_csv))
    names = players_df["name_norm"].astype(str).tolist()
    corr = np.eye(n, dtype=np.float64)
    missing_pairs = 0
    total_pairs = 0
    for i in range(n):
        a = norm_name(names[i])
        for j in range(i + 1, n):
            total_pairs += 1
            b = norm_name(names[j])
            if (a, b) not in lookup.corr and (b, a) not in lookup.corr:
                missing_pairs += 1
            v = float(lookup.get_pair(a, b))
            corr[i, j] = v
            corr[j, i] = v

    corr = _nearest_psd(corr)
    miss_rate = 0.0 if total_pairs == 0 else float(missing_pairs / total_pairs)

    metrics = {"std_mode_used": used, "corr_missing_pairs_rate": miss_rate}
    return OutcomeSimSpec(mean=mean, std=std, corr=corr), metrics


def simulate_correlated_normals(
    spec: OutcomeSimSpec,
    *,
    num_sims: int,
    seed: int,
) -> np.ndarray:
    """
    Return draws [num_sims, n_players] of correlated normal outcomes.
    """
    rng = np.random.default_rng(int(seed))
    n = int(spec.mean.shape[0])
    L = np.linalg.cholesky(spec.corr.astype(np.float64, copy=False))
    z = rng.standard_normal(size=(int(num_sims), n), dtype=np.float64)
    x = z @ L.T
    return spec.mean[None, :] + x * spec.std[None, :]


