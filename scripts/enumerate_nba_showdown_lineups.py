#!/usr/bin/env python3
"""
Enumerate NBA DK showdown lineup universes for all historical slates.

Scans:
  data/historical/raw/dk-results/showdown/nba/<slate_id>/

Writes (mirrored under enriched_root):
  data/historical/enriched/dk-results/showdown/nba/<slate_id>/
    - players.parquet
    - lineups.parquet
    - lineups_enriched.parquet
    - metadata.json
    - steps/<NN_step_name>/{preview.csv,schema.json,step_manifest.json}  (optional)

Skips a slate if:
  <out_dir>/lineups_enriched.parquet already exists
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Allow running as a repo script without requiring an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dfs_opt.config.settings import ContestConfig  # noqa: E402
from dfs_opt.features.optimal import compute_optimal_showdown_proj  # noqa: E402
from dfs_opt.io.artifacts import new_run_id  # noqa: E402
from dfs_opt.lineup_pool.enumerate_universe_showdown import (  # noqa: E402
    enumerate_showdown_universe,
    prepare_player_arrays,
    sample_lineups_df,
)
from dfs_opt.lineup_pool.enrich_universe_showdown import enrich_lineup_universe_showdown  # noqa: E402
from dfs_opt.models.manifests import ManifestIO, StepManifest  # noqa: E402
from dfs_opt.parsing.sabersim import parse_sabersim_showdown_csv  # noqa: E402
from dfs_opt.utils.hashing import data_fingerprint, schema_fingerprint, sha256_file  # noqa: E402


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _schema_json(df: pd.DataFrame) -> Dict[str, Any]:
    cols: List[Dict[str, Any]] = []
    for c in df.columns:
        s = df[c]
        nulls = int(s.isna().sum())
        cols.append(
            {
                "name": str(c),
                "dtype": str(s.dtype),
                "null_count": nulls,
                "null_rate": float(nulls / max(1, len(df))),
            }
        )
    return {"num_rows": int(len(df)), "columns": cols}


def _write_step(
    *,
    steps_dir: Path,
    step_idx: int,
    step_name: str,
    run_id: str,
    pipeline: str,
    df_in: Optional[pd.DataFrame],
    df_out: pd.DataFrame,
    inputs: Sequence[Tuple[str, str, str]],
    outputs: Sequence[Tuple[str, str, str]],
    metrics: Dict[str, Any],
) -> None:
    """
    Write pipeline-style step sidecars matching dfs_opt.io.artifacts.ArtifactWriter semantics.
    """
    step_dir = steps_dir / f"{step_idx:02d}_{step_name}"
    if step_dir.exists():
        shutil.rmtree(step_dir)
    step_dir.mkdir(parents=True, exist_ok=False)

    started_at = _utc_now()
    t0 = time.perf_counter()

    preview = df_out.head(200)
    preview_path = step_dir / "preview.csv"
    preview.to_csv(preview_path, index=False)

    schema_info = _schema_json(df_out)
    schema_path = step_dir / "schema.json"
    schema_path.write_text(json.dumps(schema_info, indent=2, sort_keys=True), encoding="utf-8")

    finished_at = _utc_now()
    duration_s = time.perf_counter() - t0

    cols_and_dtypes: Iterable[tuple[str, str]] = ((c, str(df_out[c].dtype)) for c in df_out.columns)
    sfp = schema_fingerprint(cols_and_dtypes)
    dfp = data_fingerprint(preview.to_dict(orient="records"))

    manifest = StepManifest(
        run_id=run_id,
        pipeline=pipeline,  # type: ignore[arg-type]
        step_name=step_name,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=float(duration_s),
        inputs=[ManifestIO(path=p, checksum_sha256=chk, logical_name=ln) for (p, chk, ln) in inputs],
        outputs=[ManifestIO(path=p, checksum_sha256=chk, logical_name=ln) for (p, chk, ln) in outputs]
        + [
            ManifestIO(
                path=str(preview_path),
                checksum_sha256=sha256_file(str(preview_path)),
                logical_name="preview.csv",
            ),
            ManifestIO(
                path=str(schema_path),
                checksum_sha256=sha256_file(str(schema_path)),
                logical_name="schema.json",
            ),
        ],
        row_count_in=0 if df_in is None else int(len(df_in)),
        row_count_out=int(len(df_out)),
        schema_fingerprint=sfp,
        data_fingerprint=dfp,
        metrics=metrics,
        warnings=[],
        errors=[],
    )
    (step_dir / "step_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _is_windows_zone_identifier(path: Path) -> bool:
    # Some slates contain files like "...csv:Zone.Identifier"
    return path.name.endswith(":Zone.Identifier") or ":Zone.Identifier" in path.name


def _discover_projection_and_corr(*, slate_dir: Path) -> Tuple[Path, Path]:
    csvs = [p for p in slate_dir.glob("*.csv") if p.is_file() and not _is_windows_zone_identifier(p)]
    corr = sorted([p for p in csvs if p.name.endswith("_corr_matrix.csv")])
    if not corr:
        raise FileNotFoundError(f"{slate_dir}: missing *_corr_matrix.csv")
    corr_csv = corr[0]

    proj_candidates = sorted([p for p in csvs if p != corr_csv and (not p.name.endswith("_corr_matrix.csv"))])
    if not proj_candidates:
        raise FileNotFoundError(f"{slate_dir}: missing projection CSV (non-corr *.csv)")
    projection_csv = proj_candidates[0]
    return projection_csv, corr_csv


def _iter_slate_dirs(*, raw_root: Path) -> List[Path]:
    base = raw_root / "dk-results" / "showdown" / "nba"
    if not base.exists():
        raise FileNotFoundError(f"Expected NBA showdown folder at {base}")
    return sorted([p for p in base.iterdir() if p.is_dir()])


def _out_dir_for_slate(*, enriched_root: Path, raw_root: Path, slate_dir: Path) -> Path:
    return enriched_root / slate_dir.relative_to(raw_root)


def _write_outputs(
    *,
    out_dir: Path,
    run_id: str,
    config: ContestConfig,
    players_enum_df: pd.DataFrame,
    lineup_res: Any,
    prep_meta: Dict[str, Any],
    optimal_proj_points: float,
    enriched_tbl: pa.Table,
    persist_steps: bool,
    steps_payloads: Dict[str, Dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Core stable artifacts
    players_path = out_dir / "players.parquet"
    players_enum_df.to_parquet(players_path, index=False)

    lineups_path = out_dir / "lineups.parquet"
    lineups_tbl = pa.table(
        {
            "cpt": pa.array(lineup_res.cpt, type=pa.uint16()),
            "u1": pa.array(lineup_res.u1, type=pa.uint16()),
            "u2": pa.array(lineup_res.u2, type=pa.uint16()),
            "u3": pa.array(lineup_res.u3, type=pa.uint16()),
            "u4": pa.array(lineup_res.u4, type=pa.uint16()),
            "u5": pa.array(lineup_res.u5, type=pa.uint16()),
            "salary_used": pa.array(lineup_res.salary_used, type=pa.int32()),
            "salary_left": pa.array(lineup_res.salary_left, type=pa.int32()),
            "proj_points": pa.array(lineup_res.proj_points, type=pa.float32()),
            "stack_code": pa.array(lineup_res.stack_code, type=pa.uint8()),
        }
    )
    pq.write_table(lineups_tbl, lineups_path, compression="zstd")

    lineups_enriched_path = out_dir / "lineups_enriched.parquet"
    pq.write_table(enriched_tbl, lineups_enriched_path, compression="zstd")

    metadata_path = out_dir / "metadata.json"
    metadata = {
        "slate_id": str(config.slate_id),
        "sport": str(config.sport).lower(),
        "created_at_utc": _utc_now().isoformat(),
        "run_id": str(run_id),
        "salary_cap": int(config.salary_cap),
        "num_players": int(lineup_res.num_players),
        "num_lineups": int(lineup_res.num_lineups),
        "team_mapping": prep_meta["team_mapping"],
        "filters": {
            "min_proj_points": float(config.min_proj_points),
            "max_players": (None if config.max_players is None else int(config.max_players)),
        },
        "schema": {name: str(lineups_tbl.schema.field(name).type) for name in lineups_tbl.schema.names},
        "stack_code_map": {"0": "3-3", "1": "4-2", "2": "5-1"},
        "enriched_file": "lineups_enriched.parquet",
        "enriched_schema": {name: str(enriched_tbl.schema.field(name).type) for name in enriched_tbl.schema.names},
        "optimal_proj_points": float(optimal_proj_points),
        "timings": lineup_res.metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    if persist_steps:
        steps_dir = out_dir / "steps"
        if steps_dir.exists():
            shutil.rmtree(steps_dir)
        steps_dir.mkdir(parents=True, exist_ok=False)

        # Populate canonical outputs (checksums are only available after files exist).
        enum_key = "02_enumerate_lineup_universe"
        if enum_key in steps_payloads:
            steps_payloads[enum_key]["outputs"] = [
                (str(players_path), sha256_file(str(players_path)), "players.parquet"),
                (str(lineups_path), sha256_file(str(lineups_path)), "lineups.parquet"),
                (str(metadata_path), sha256_file(str(metadata_path)), "metadata.json"),
            ]
        enrich_key = "03_enrich_lineup_universe_features"
        if enrich_key in steps_payloads:
            steps_payloads[enrich_key]["outputs"] = [
                (str(lineups_enriched_path), sha256_file(str(lineups_enriched_path)), "lineups_enriched.parquet"),
            ]

        for step_key, p in steps_payloads.items():
            _write_step(
                steps_dir=steps_dir,
                step_idx=int(p["step_idx"]),
                step_name=str(p["step_name"]),
                run_id=run_id,
                pipeline="contest",
                df_in=p.get("df_in"),
                df_out=p["df_out"],
                inputs=p.get("inputs", []),
                outputs=p.get("outputs", []),
                metrics=p.get("metrics", {}),
            )


def process_slate(
    *,
    slate_dir: Path,
    raw_root: Path,
    enriched_root: Path,
    salary_cap: int,
    min_proj_points: float,
    max_players: Optional[int],
    own_log_eps: float,
    seed: int,
    persist_steps: bool,
) -> Dict[str, Any]:
    slate_id = slate_dir.name
    out_dir = _out_dir_for_slate(enriched_root=enriched_root, raw_root=raw_root, slate_dir=slate_dir)

    if (out_dir / "lineups_enriched.parquet").exists():
        return {"status": "skipped", "slate_id": slate_id, "out_dir": str(out_dir)}

    projection_csv, corr_csv = _discover_projection_and_corr(slate_dir=slate_dir)
    run_id = new_run_id()

    cfg = ContestConfig(
        projection_csv=projection_csv,
        corr_matrix_csv=corr_csv,
        slate_id=slate_id,
        sport="nba",
        artifacts_root=Path("unused"),
        seed=int(seed),
        persist_step_outputs=False,
        salary_cap=int(salary_cap),
        min_proj_points=float(min_proj_points),
        max_players=max_players,
        own_log_eps=float(own_log_eps),
    )

    # -------------------------
    # 00. ingest
    # -------------------------
    inv_df = pd.DataFrame(
        [
            {
                "kind": "projection",
                "sport": "nba",
                "slate_id": slate_id,
                "path": str(projection_csv),
                "sha256": sha256_file(str(projection_csv)),
            },
            {
                "kind": "corr_matrix",
                "sport": "nba",
                "slate_id": slate_id,
                "path": str(corr_csv),
                "sha256": sha256_file(str(corr_csv)),
            },
        ]
    )
    step00_inputs = [
        (str(projection_csv), sha256_file(str(projection_csv)), "sabersim_projection"),
        (str(corr_csv), sha256_file(str(corr_csv)), "corr_matrix_csv"),
    ]

    # -------------------------
    # 01. parse_projections
    # -------------------------
    players_flex, m_parse = parse_sabersim_showdown_csv(projection_csv)
    players_flex["sport"] = "nba"
    players_flex["slate_id"] = slate_id

    # -------------------------
    # 02. enumerate_lineup_universe
    # -------------------------
    players_enum_df, arrays, prep_meta = prepare_player_arrays(
        players_flex,
        min_proj_points=float(min_proj_points),
        max_players=max_players,
    )
    lineup_res = enumerate_showdown_universe(arrays, salary_cap=int(salary_cap))

    # Compute optimal for pct-gap feature + metadata
    opt_res = compute_optimal_showdown_proj(
        players_enum_df.rename(columns={"salary": "salary", "proj_points": "proj_points"}),
        salary_cap=int(salary_cap),
        min_proj_points=float(min_proj_points),
    )
    optimal_proj_points = float(opt_res.optimal_proj_points)
    if not (optimal_proj_points > 0.0):
        raise ValueError(f"Invalid optimal_proj_points computed: {optimal_proj_points}")

    # -------------------------
    # 03. enrich_lineup_universe_features
    # -------------------------
    enrich_cols, enrich_metrics = enrich_lineup_universe_showdown(
        players_enum_df=players_enum_df,
        cpt=lineup_res.cpt,
        u1=lineup_res.u1,
        u2=lineup_res.u2,
        u3=lineup_res.u3,
        u4=lineup_res.u4,
        u5=lineup_res.u5,
        salary_left=lineup_res.salary_left,
        proj_points=lineup_res.proj_points,
        corr_matrix_csv=corr_csv,
        captain_tiers=cfg.captain_tiers,
        optimal_proj_points=optimal_proj_points,
        own_log_eps=float(own_log_eps),
        salary_cap=int(salary_cap),
        min_proj_points=float(min_proj_points),
    )

    salary_left_bin_labels = [
        "0_200",
        "200_500",
        "500_1000",
        "1000_2000",
        "2000_4000",
        "4000_8000",
        "8000_plus",
    ]
    gap_bin_labels = [
        "0_0.01",
        "0.01_0.02",
        "0.02_0.04",
        "0.04_0.07",
        "0.07_0.15",
        "0.15_0.30",
        "0.30_plus",
    ]
    cpt_arch_labels = list(enrich_cols["cpt_archetype_labels"])

    salary_left_bin = pa.DictionaryArray.from_arrays(
        pa.array(enrich_cols["salary_left_bin_code"], type=pa.uint8()),
        pa.array(salary_left_bin_labels, type=pa.string()),
    )
    pct_gap_bin = pa.DictionaryArray.from_arrays(
        pa.array(enrich_cols["pct_proj_gap_to_optimal_bin_code"], type=pa.uint8()),
        pa.array(gap_bin_labels, type=pa.string()),
    )
    cpt_arch = pa.DictionaryArray.from_arrays(
        pa.array(enrich_cols["cpt_archetype_code"], type=pa.uint8()),
        pa.array(cpt_arch_labels, type=pa.string()),
    )

    enriched_tbl = pa.table(
        {
            "cpt": pa.array(lineup_res.cpt, type=pa.uint16()),
            "u1": pa.array(lineup_res.u1, type=pa.uint16()),
            "u2": pa.array(lineup_res.u2, type=pa.uint16()),
            "u3": pa.array(lineup_res.u3, type=pa.uint16()),
            "u4": pa.array(lineup_res.u4, type=pa.uint16()),
            "u5": pa.array(lineup_res.u5, type=pa.uint16()),
            "salary_used": pa.array(lineup_res.salary_used, type=pa.int32()),
            "salary_left": pa.array(lineup_res.salary_left, type=pa.int32()),
            "proj_points": pa.array(lineup_res.proj_points, type=pa.float32()),
            "stack_code": pa.array(lineup_res.stack_code, type=pa.uint8()),
            "own_score_logprod": pa.array(enrich_cols["own_score_logprod"], type=pa.float32()),
            "own_max_log": pa.array(enrich_cols["own_max_log"], type=pa.float32()),
            "own_min_log": pa.array(enrich_cols["own_min_log"], type=pa.float32()),
            "avg_corr": pa.array(enrich_cols["avg_corr"], type=pa.float32()),
            "cpt_archetype": cpt_arch,
            "salary_left_bin": salary_left_bin,
            "pct_proj_gap_to_optimal": pa.array(enrich_cols["pct_proj_gap_to_optimal"], type=pa.float32()),
            "pct_proj_gap_to_optimal_bin": pct_gap_bin,
        }
    )

    # Steps payloads (for optional sidecars)
    sample_df = sample_lineups_df(lineup_res, n=200)
    k = min(200, int(lineup_res.num_lineups))
    preview_df = pd.DataFrame(
        {
            "cpt": lineup_res.cpt[:k],
            "u1": lineup_res.u1[:k],
            "u2": lineup_res.u2[:k],
            "u3": lineup_res.u3[:k],
            "u4": lineup_res.u4[:k],
            "u5": lineup_res.u5[:k],
            "salary_used": lineup_res.salary_used[:k],
            "salary_left": lineup_res.salary_left[:k],
            "proj_points": lineup_res.proj_points[:k],
            "stack_code": lineup_res.stack_code[:k],
            "own_score_logprod": np.asarray(enrich_cols["own_score_logprod"][:k]),
            "own_max_log": np.asarray(enrich_cols["own_max_log"][:k]),
            "own_min_log": np.asarray(enrich_cols["own_min_log"][:k]),
            "avg_corr": np.asarray(enrich_cols["avg_corr"][:k]),
            "cpt_archetype": [cpt_arch_labels[int(x)] for x in np.asarray(enrich_cols["cpt_archetype_code"][:k])],
            "salary_left_bin": [
                salary_left_bin_labels[int(x)] for x in np.asarray(enrich_cols["salary_left_bin_code"][:k])
            ],
            "pct_proj_gap_to_optimal": np.asarray(enrich_cols["pct_proj_gap_to_optimal"][:k]),
            "pct_proj_gap_to_optimal_bin": [
                gap_bin_labels[int(x)] for x in np.asarray(enrich_cols["pct_proj_gap_to_optimal_bin_code"][:k])
            ],
        }
    )

    steps_payloads = {
        "00_ingest": {
            "step_idx": 0,
            "step_name": "ingest",
            "df_in": None,
            "df_out": inv_df,
            "inputs": step00_inputs,
            "outputs": [],
            "metrics": {"num_files": int(len(inv_df))},
        },
        "01_parse_projections": {
            "step_idx": 1,
            "step_name": "parse_projections",
            "df_in": None,
            "df_out": players_flex,
            "inputs": [(str(projection_csv), sha256_file(str(projection_csv)), "sabersim_projection")],
            "outputs": [],
            "metrics": {
                "num_players": int(m_parse["num_players"]),
                "num_rows_raw": int(m_parse["num_rows_raw"]),
                "dropped_zero_proj": int(m_parse["dropped_zero_proj"]),
            },
        },
        "02_enumerate_lineup_universe": {
            "step_idx": 2,
            "step_name": "enumerate_lineup_universe",
            "df_in": players_enum_df,
            "df_out": sample_df,
            "inputs": [],
            "outputs": [],
            "metrics": {**prep_meta, **lineup_res.metrics},
        },
        "03_enrich_lineup_universe_features": {
            "step_idx": 3,
            "step_name": "enrich_lineup_universe_features",
            "df_in": None,
            "df_out": preview_df,
            "inputs": [(str(corr_csv), sha256_file(str(corr_csv)), "corr_matrix_csv")],
            "outputs": [],
            "metrics": {
                "optimal_proj_points": float(enrich_metrics.optimal_proj_points),
                "corr_missing_pairs_rate": float(enrich_metrics.corr_missing_pairs_rate),
                "own_log_eps": float(enrich_metrics.own_log_eps),
            },
        },
    }

    _write_outputs(
        out_dir=out_dir,
        run_id=run_id,
        config=cfg,
        players_enum_df=players_enum_df,
        lineup_res=lineup_res,
        prep_meta=prep_meta,
        optimal_proj_points=optimal_proj_points,
        enriched_tbl=enriched_tbl,
        persist_steps=persist_steps,
        steps_payloads=steps_payloads,
    )

    return {
        "status": "processed",
        "slate_id": slate_id,
        "out_dir": str(out_dir),
        "num_players": int(lineup_res.num_players),
        "num_lineups": int(lineup_res.num_lineups),
        "run_id": run_id,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Enumerate NBA showdown lineup universes for historical slates.")
    p.add_argument("--raw-root", type=str, default="data/historical/raw")
    p.add_argument("--enriched-root", type=str, default="data/historical/enriched")

    p.add_argument("--salary-cap", type=int, default=50000)
    p.add_argument("--min-proj-points", type=float, default=0.0)
    p.add_argument("--max-players", type=int, default=None)
    p.add_argument("--own-log-eps", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--persist-steps", dest="persist_steps", action="store_true")
    p.add_argument("--no-persist-steps", dest="persist_steps", action="store_false")
    p.set_defaults(persist_steps=True)

    p.add_argument("--fail-fast", dest="fail_fast", action="store_true")
    p.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    p.set_defaults(fail_fast=True)

    args = p.parse_args()

    raw_root = Path(args.raw_root).resolve()
    enriched_root = Path(args.enriched_root).resolve()

    slate_dirs = _iter_slate_dirs(raw_root=raw_root)
    total = len(slate_dirs)
    processed = 0
    skipped = 0
    failed = 0
    failures: List[Tuple[str, str]] = []

    t_all = time.perf_counter()
    for i, slate_dir in enumerate(slate_dirs, start=1):
        slate_id = slate_dir.name
        t0 = time.perf_counter()
        try:
            res = process_slate(
                slate_dir=slate_dir,
                raw_root=raw_root,
                enriched_root=enriched_root,
                salary_cap=int(args.salary_cap),
                min_proj_points=float(args.min_proj_points),
                max_players=args.max_players,
                own_log_eps=float(args.own_log_eps),
                seed=int(args.seed),
                persist_steps=bool(args.persist_steps),
            )
            if res["status"] == "skipped":
                skipped += 1
                print(f"[{i}/{total}] SKIP  slate_id={slate_id} out_dir={res['out_dir']}")
            else:
                processed += 1
                print(
                    f"[{i}/{total}] OK    slate_id={slate_id} num_players={res['num_players']} "
                    f"num_lineups={res['num_lineups']} out_dir={res['out_dir']} "
                    f"duration_s={time.perf_counter() - t0:.2f}"
                )
        except Exception as e:
            failed += 1
            failures.append((slate_id, repr(e)))
            print(f"[{i}/{total}] FAIL  slate_id={slate_id} err={e!r} duration_s={time.perf_counter() - t0:.2f}")
            if args.fail_fast:
                break

    duration_s = time.perf_counter() - t_all
    print(
        f"\nSummary: total={total} processed={processed} skipped={skipped} failed={failed} duration_s={duration_s:.2f}"
    )
    if failures:
        print("Failures:")
        for slate_id, err in failures[:20]:
            print(f"  - {slate_id}: {err}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


