from __future__ import annotations

import numpy as np

try:
    import numba as nb
except Exception as e:  # pragma: no cover
    raise ImportError("numba is required for lineup-universe feature kernels. Install with `pip install numba`.") from e


@nb.njit(parallel=True, cache=True)
def avg_corr_for_lineups(
    cpt: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    u4: np.ndarray,
    u5: np.ndarray,
    corr_mat: np.ndarray,
) -> np.ndarray:
    """
    Compute average pairwise correlation over each 6-player showdown lineup.

    corr_mat is a dense float32[n_players, n_players] matrix aligned to the same
    player indexing used by the lineup slot arrays.
    Denominator is fixed at 15 (= C(6,2)).
    """
    n = int(cpt.shape[0])
    out = np.empty(n, dtype=np.float32)
    for i in nb.prange(n):
        a0 = int(cpt[i])
        a1 = int(u1[i])
        a2 = int(u2[i])
        a3 = int(u3[i])
        a4 = int(u4[i])
        a5 = int(u5[i])

        s = 0.0
        # 15 pairs
        s += float(corr_mat[a0, a1])
        s += float(corr_mat[a0, a2])
        s += float(corr_mat[a0, a3])
        s += float(corr_mat[a0, a4])
        s += float(corr_mat[a0, a5])

        s += float(corr_mat[a1, a2])
        s += float(corr_mat[a1, a3])
        s += float(corr_mat[a1, a4])
        s += float(corr_mat[a1, a5])

        s += float(corr_mat[a2, a3])
        s += float(corr_mat[a2, a4])
        s += float(corr_mat[a2, a5])

        s += float(corr_mat[a3, a4])
        s += float(corr_mat[a3, a5])

        s += float(corr_mat[a4, a5])

        out[i] = np.float32(s / 15.0)
    return out


