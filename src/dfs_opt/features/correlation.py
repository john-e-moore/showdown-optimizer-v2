from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from dfs_opt.parsing.names import norm_name


@dataclass(frozen=True)
class CorrLookup:
    """
    Correlation lookup keyed by normalized (name_i, name_j) pairs.
    Missing pairs are treated as 0 by callers.
    """

    corr: Dict[Tuple[str, str], float]

    def get_pair(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        v = self.corr.get((a, b))
        if v is not None:
            return float(v)
        v = self.corr.get((b, a))
        if v is not None:
            return float(v)
        return 0.0


def load_corr_matrix_csv(path: Path) -> CorrLookup:
    """
    Load Sabersim-style correlation matrix CSV (square matrix with row-name column).

    The file typically looks like:
      Column1,<playerA>,<playerB>,...
      <playerA>,1.0,0.12,...
      <playerB>,0.12,1.0,...
    """
    df = pd.read_csv(path)
    if df.shape[0] == 0 or df.shape[1] < 2:
        raise ValueError(f"{path}: invalid correlation matrix shape {df.shape}")

    row_name_col = str(df.columns[0])
    col_names_raw = [str(c) for c in df.columns[1:]]

    row_names_raw = df[row_name_col].astype(str).tolist()
    if len(row_names_raw) != len(df):
        raise ValueError(f"{path}: invalid correlation matrix; cannot read row names")

    row_names = [norm_name(n) for n in row_names_raw]
    col_names = [norm_name(n) for n in col_names_raw]

    # Build a sparse-ish lookup dict; we store all pairs present in the file.
    # Values are coerced to float, invalid cells become NaN -> skipped.
    corr: Dict[Tuple[str, str], float] = {}
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    for i, rn in enumerate(row_names):
        row = values.iloc[i].to_numpy()
        for j, cn in enumerate(col_names):
            v = row[j]
            if pd.isna(v):
                continue
            corr[(rn, cn)] = float(v)

    return CorrLookup(corr=corr)


def avg_corr_for_lineup(names_norm: Iterable[str], lookup: CorrLookup) -> float:
    """
    Average pairwise correlation over 6-player showdown lineup.
    Fixed denominator = 15 (C(6,2)).
    """
    names = [str(n) for n in names_norm]
    if len(names) != 6:
        raise ValueError(f"avg_corr expects 6 players, got {len(names)}")

    s = 0.0
    # 15 pairs for 6 players
    for i in range(6):
        a = names[i]
        for j in range(i + 1, 6):
            b = names[j]
            s += lookup.get_pair(a, b)
    return s / 15.0


