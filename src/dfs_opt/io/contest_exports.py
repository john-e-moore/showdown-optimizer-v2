from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ContestCsvExportResult:
    contest_id: str
    field_csv: Path
    grades_top_csv: Path
    n_field_unique: int
    n_top: int


def _build_nameid_array(players_df: pd.DataFrame) -> np.ndarray:
    """
    Build an array where index i corresponds to player i in the enumeration table,
    and value is formatted as: "Name (DFS ID)".
    """
    # Try common column names; Pipeline B uses parse_sabersim_showdown_csv -> player_name + dfs_id.
    name_col = None
    for c in ["player_name", "Name", "name", "player", "display_name", "name_norm"]:
        if c in players_df.columns:
            name_col = c
            break
    if name_col is None:
        raise ValueError(f"players.parquet missing a usable name column; got {list(players_df.columns)}")

    id_col = None
    for c in ["dfs_id", "DFS ID", "dk_id", "id"]:
        if c in players_df.columns:
            id_col = c
            break

    names = players_df[name_col].astype(str).fillna("").tolist()
    if id_col is None:
        return np.asarray([n if n else str(i) for i, n in enumerate(names)], dtype=object)

    ids_raw = players_df[id_col]
    # Pandas nullable ints produce <NA>; normalize to "".
    ids_str = ids_raw.astype("string").fillna("").astype(str).tolist()

    out: List[str] = []
    for i, (nm, pid) in enumerate(zip(names, ids_str, strict=True)):
        nm = (nm or "").strip()
        pid = (pid or "").strip()
        if not nm:
            out.append(str(i))
        elif pid and pid.lower() != "nan":
            out.append(f"{nm} ({pid})")
        else:
            out.append(nm)
    return np.asarray(out, dtype=object)


def _enrich_lineups_for_ids(
    *,
    lineup_ids: Sequence[int],
    lineups_enriched_df: pd.DataFrame,
    nameid: np.ndarray,
) -> pd.DataFrame:
    if not lineup_ids:
        return pd.DataFrame()

    ids = pd.Series(list(lineup_ids)).astype(int)
    max_id = int(ids.max())
    if max_id >= len(lineups_enriched_df):
        raise ValueError(f"lineup_id out of range: max={max_id} n_lineups={len(lineups_enriched_df)}")

    slots = lineups_enriched_df.iloc[ids.to_numpy()][
        ["cpt", "u1", "u2", "u3", "u4", "u5", "proj_points", "salary_used", "avg_corr"]
    ].copy()

    def fmt_slot(col: str) -> pd.Series:
        idx = slots[col].astype(int).to_numpy()
        return pd.Series(nameid[idx])

    out = pd.DataFrame(
        {
            "lineup_id": ids.to_numpy(),
            "cpt": fmt_slot("cpt"),
            "util1": fmt_slot("u1"),
            "util2": fmt_slot("u2"),
            "util3": fmt_slot("u3"),
            "util4": fmt_slot("u4"),
            "util5": fmt_slot("u5"),
            "proj_points": pd.to_numeric(slots["proj_points"], errors="coerce"),
            "salary_used": pd.to_numeric(slots["salary_used"], errors="coerce").astype("Int64"),
            "avg_corr": pd.to_numeric(slots["avg_corr"], errors="coerce"),
        }
    )
    return out


def export_contest_csvs(
    *,
    contest_id: str,
    contest_dir: Path,
    players_parquet: Path,
    lineups_enriched_parquet: Path,
    top_n: int = 1000,
) -> ContestCsvExportResult:
    """
    Write human-readable contest outputs next to per-contest parquet artifacts:
    - field_sample.parquet -> field_sample.csv
    - lineup_grades.parquet -> lineup_grades_top1000.csv
    """
    contest_dir = Path(contest_dir)

    players_df = pq.read_table(players_parquet, memory_map=True).to_pandas()
    nameid = _build_nameid_array(players_df)

    lineups_enriched_df = pq.read_table(
        lineups_enriched_parquet,
        columns=["cpt", "u1", "u2", "u3", "u4", "u5", "proj_points", "salary_used", "avg_corr"],
        memory_map=True,
    ).to_pandas()

    # -------------------------
    # Field sample CSV
    # -------------------------
    field_parquet = contest_dir / "field_sample.parquet"
    field_csv = contest_dir / "field_sample.csv"
    field_df = pq.read_table(field_parquet, memory_map=True).to_pandas()
    field_df["lineup_id"] = pd.to_numeric(field_df["lineup_id"], errors="coerce").astype("Int64")
    field_df["dup_count"] = pd.to_numeric(field_df["dup_count"], errors="coerce").astype("Int64")
    field_df = field_df.dropna(subset=["lineup_id"]).copy()
    field_df["lineup_id"] = field_df["lineup_id"].astype(int)

    field_enriched = _enrich_lineups_for_ids(
        lineup_ids=field_df["lineup_id"].astype(int).tolist(),
        lineups_enriched_df=lineups_enriched_df,
        nameid=nameid,
    )
    field_out = field_df.merge(field_enriched, on="lineup_id", how="left")
    # Keep a stable, readable order.
    field_out = field_out[
        ["lineup_id", "dup_count", "cpt", "util1", "util2", "util3", "util4", "util5", "proj_points", "salary_used", "avg_corr"]
    ]
    field_out.to_csv(field_csv, index=False)

    # -------------------------
    # Grades top-N CSV
    # -------------------------
    grades_parquet = contest_dir / "lineup_grades.parquet"
    grades_csv = contest_dir / "lineup_grades_top1000.csv"
    grades_df = pq.read_table(grades_parquet, memory_map=True).to_pandas()
    grades_df["lineup_id"] = pd.to_numeric(grades_df["lineup_id"], errors="coerce").astype("Int64")
    grades_df = grades_df.dropna(subset=["lineup_id"]).copy()
    grades_df["lineup_id"] = grades_df["lineup_id"].astype(int)

    grades_df = grades_df.sort_values(["roi", "lineup_id"], ascending=[False, True]).reset_index(drop=True)
    top_df = grades_df.head(int(top_n)).copy()

    top_enriched = _enrich_lineups_for_ids(
        lineup_ids=top_df["lineup_id"].astype(int).tolist(),
        lineups_enriched_df=lineups_enriched_df,
        nameid=nameid,
    )
    top_out = top_df.merge(top_enriched, on="lineup_id", how="left")

    # Keep the core grading metrics, plus readable slots and requested features.
    cols: List[str] = [
        "lineup_id",
        "roi",
        "exp_winnings",
        "top_0_1_pct",
        "top_1_pct",
        "top_5_pct",
        "top_20_pct",
        "cpt",
        "util1",
        "util2",
        "util3",
        "util4",
        "util5",
        "proj_points",
        "salary_used",
        "avg_corr",
    ]
    # Some older runs may omit some columns; keep whatever exists.
    cols_present = [c for c in cols if c in top_out.columns]
    top_out = top_out[cols_present]
    top_out.to_csv(grades_csv, index=False)

    return ContestCsvExportResult(
        contest_id=str(contest_id),
        field_csv=field_csv,
        grades_top_csv=grades_csv,
        n_field_unique=int(len(field_df)),
        n_top=int(len(top_df)),
    )


