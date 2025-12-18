from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from dfs_opt.utils.hashing import sha256_file


@dataclass(frozen=True)
class SlateInputs:
    sport: str
    slate_id: str
    projection_csv: Path
    standings_files: List[Path]  # extracted CSVs
    source_archives: List[Path]  # zip files (optional)


def find_showdown_slates(data_root: Path) -> List[Tuple[str, str, Path]]:
    """
    Returns list of (sport, slate_id, slate_dir) under:
      <data_root>/dk-results/showdown/<sport>/<slate_id>/
    """
    base = data_root / "dk-results" / "showdown"
    if not base.exists():
        raise FileNotFoundError(f"Expected showdown folder at {base}")

    slates: List[Tuple[str, str, Path]] = []
    for sport_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        for slate_dir in sorted([p for p in sport_dir.iterdir() if p.is_dir()]):
            slates.append((sport_dir.name, slate_dir.name, slate_dir))
    return slates


def extract_standings_zips(zips: Iterable[Path], *, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    for z in zips:
        with zipfile.ZipFile(z, "r") as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                target = out_dir / Path(member).name
                zf.extract(member, path=out_dir)
                # zipfile keeps original nested paths; normalize by copying to root if needed
                nested = out_dir / member
                if nested != target and nested.exists():
                    nested.replace(target)
                extracted.append(target)
    return extracted


def load_slate_inputs(
    *,
    slate_dir: Path,
    sport: str,
    slate_id: str,
    extract_dir: Path,
) -> SlateInputs:
    # projection: choose first csv in slate_dir that is not under contests/
    proj_candidates = [p for p in slate_dir.glob("*.csv") if p.is_file()]
    if not proj_candidates:
        raise FileNotFoundError(f"No projection CSV found in {slate_dir}")
    projection_csv = sorted(proj_candidates)[0]

    contests_dir = slate_dir / "contests"
    standings_files: List[Path] = []
    source_archives: List[Path] = []
    if contests_dir.exists():
        # extracted csv folders
        standings_files.extend([p for p in contests_dir.rglob("*.csv") if p.is_file() and "contest-standings" in p.name])
        # zips
        zips = [p for p in contests_dir.glob("*.zip") if p.is_file()]
        source_archives.extend(zips)
        # Only extract zips if we didn't already find any extracted standings CSVs.
        # Many repos keep both extracted CSVs and the original zips; extracting everything
        # each run would be unnecessarily slow and produce massive artifacts.
        if zips and not standings_files:
            standings_files.extend(extract_standings_zips(zips, out_dir=extract_dir))

    standings_files = sorted({p.resolve() for p in standings_files})
    return SlateInputs(
        sport=sport,
        slate_id=slate_id,
        projection_csv=projection_csv,
        standings_files=standings_files,
        source_archives=source_archives,
    )


def build_input_inventory(inputs: List[SlateInputs]) -> pd.DataFrame:
    rows = []
    for s in inputs:
        rows.append(
            {
                "kind": "projection",
                "sport": s.sport,
                "slate_id": s.slate_id,
                "path": str(s.projection_csv),
                "sha256": sha256_file(str(s.projection_csv)),
                "size_bytes": s.projection_csv.stat().st_size,
            }
        )
        for p in s.standings_files:
            rows.append(
                {
                    "kind": "standings_csv",
                    "sport": s.sport,
                    "slate_id": s.slate_id,
                    "path": str(p),
                    "sha256": sha256_file(str(p)),
                    "size_bytes": p.stat().st_size,
                }
            )
        for z in s.source_archives:
            rows.append(
                {
                    "kind": "standings_zip",
                    "sport": s.sport,
                    "slate_id": s.slate_id,
                    "path": str(z),
                    "sha256": sha256_file(str(z)),
                    "size_bytes": z.stat().st_size,
                }
            )

    return pd.DataFrame(rows)


