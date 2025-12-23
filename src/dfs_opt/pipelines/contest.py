from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dfs_opt.config.settings import ContestConfig
from dfs_opt.features.optimal import compute_optimal_showdown_proj
from dfs_opt.io.artifacts import ArtifactWriter, new_run_id, utc_now
from dfs_opt.models.manifests import ManifestIO, RunManifest
from dfs_opt.parsing.sabersim import parse_sabersim_showdown_csv
from dfs_opt.lineup_pool.enumerate_universe_showdown import (
    enumerate_showdown_universe,
    prepare_player_arrays,
    sample_lineups_df,
)
from dfs_opt.lineup_pool.enrich_universe_showdown import enrich_lineup_universe_showdown
from dfs_opt.utils.git import try_get_git_sha
from dfs_opt.utils.hashing import sha256_file
from dfs_opt.utils.logging import configure_run_logger, get_step_logger


def _init_contest_run(config: ContestConfig) -> Tuple[str, Any, ArtifactWriter]:
    started_at = utc_now()
    run_id = new_run_id()
    writer = ArtifactWriter(artifacts_root=config.artifacts_root, pipeline="contest", run_id=run_id)
    writer.init_run_dirs()

    logger = configure_run_logger(
        logs_dir=writer.logs_dir,
        run_id=run_id,
        pipeline="contest",
        level=getattr(config, "log_level", "INFO"),
    )
    logger.info(
        "Run started: slate_id=%s sport=%s artifacts_root=%s seed=%s persist_step_outputs=%s git_sha=%s",
        str(config.slate_id),
        str(config.sport).lower(),
        str(config.artifacts_root),
        int(config.seed),
        bool(config.persist_step_outputs),
        try_get_git_sha(),
    )
    return run_id, logger, writer


def run_contest_lineup_gen(config: ContestConfig) -> Dict[str, Any]:
    """
    Minimal Pipeline B runner for showdown lineup universe enumeration.

    Writes canonical artifacts under:
      <artifacts_root>/contest/<run_id>/...
    """
    started_at = utc_now()
    run_id, logger, writer = _init_contest_run(config)

    run_inputs: List[ManifestIO] = []
    run_outputs: List[ManifestIO] = []
    row_counts_by_step: Dict[str, Dict[str, int]] = {}

    try:
        # -------------------------
        # 00. ingest
        # -------------------------
        step = "00_ingest"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        inv_df = pd.DataFrame(
            [
                {
                    "kind": "projection",
                    "sport": str(config.sport).lower(),
                    "slate_id": str(config.slate_id),
                    "path": str(config.projection_csv),
                    "sha256": sha256_file(str(config.projection_csv)),
                },
                {
                    "kind": "corr_matrix",
                    "sport": str(config.sport).lower(),
                    "slate_id": str(config.slate_id),
                    "path": str(config.corr_matrix_csv),
                    "sha256": sha256_file(str(config.corr_matrix_csv)),
                },
            ]
        )
        step00_inputs = [
            (str(config.projection_csv), sha256_file(str(config.projection_csv)), "sabersim_projection"),
            (str(config.corr_matrix_csv), sha256_file(str(config.corr_matrix_csv)), "corr_matrix_csv"),
        ]
        run_inputs.append(ManifestIO(path=step00_inputs[0][0], checksum_sha256=step00_inputs[0][1], logical_name=step00_inputs[0][2]))
        run_inputs.append(ManifestIO(path=step00_inputs[1][0], checksum_sha256=step00_inputs[1][1], logical_name=step00_inputs[1][2]))

        writer.write_step(
            step_idx=0,
            step_name="ingest",
            df_in=None,
            df_out=inv_df,
            inputs=step00_inputs,
            outputs=[],
            metrics={"num_files": int(len(inv_df))},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": 0, "row_count_out": int(len(inv_df))}
        step_logger.info("Finished step: duration_s=%.3f row_count_out=%s", time.perf_counter() - t0, int(len(inv_df)))

        # -------------------------
        # 01. parse_projections
        # -------------------------
        step = "01_parse_projections"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        players_flex, m = parse_sabersim_showdown_csv(config.projection_csv)
        players_flex["sport"] = str(config.sport).lower()
        players_flex["slate_id"] = str(config.slate_id)

        writer.write_step(
            step_idx=1,
            step_name="parse_projections",
            df_in=None,
            df_out=players_flex,
            inputs=[(str(config.projection_csv), sha256_file(str(config.projection_csv)), "sabersim_projection")],
            outputs=[],
            metrics={
                "num_players": int(m["num_players"]),
                "num_rows_raw": int(m["num_rows_raw"]),
                "dropped_zero_proj": int(m["dropped_zero_proj"]),
            },
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": 0, "row_count_out": int(len(players_flex))}
        step_logger.info("Finished step: duration_s=%.3f row_count_out=%s", time.perf_counter() - t0, int(len(players_flex)))

        # -------------------------
        # 02. enumerate_lineup_universe
        # -------------------------
        step = "02_enumerate_lineup_universe"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        players_enum_df, arrays, prep_meta = prepare_player_arrays(
            players_flex,
            min_proj_points=float(config.min_proj_points),
            max_players=config.max_players,
        )
        res = enumerate_showdown_universe(arrays, salary_cap=int(config.salary_cap))

        # Write players + lineups at run root so other steps can reuse them.
        players_path = writer.run_dir / "players.parquet"
        players_enum_df.to_parquet(players_path, index=False)
        run_outputs.append(
            ManifestIO(path=str(players_path), checksum_sha256=sha256_file(str(players_path)), logical_name="players.parquet")
        )

        lineups_path = writer.run_dir / "lineups.parquet"
        tbl = pa.table(
            {
                "cpt": pa.array(res.cpt, type=pa.uint16()),
                "u1": pa.array(res.u1, type=pa.uint16()),
                "u2": pa.array(res.u2, type=pa.uint16()),
                "u3": pa.array(res.u3, type=pa.uint16()),
                "u4": pa.array(res.u4, type=pa.uint16()),
                "u5": pa.array(res.u5, type=pa.uint16()),
                "salary_used": pa.array(res.salary_used, type=pa.int32()),
                "salary_left": pa.array(res.salary_left, type=pa.int32()),
                "proj_points": pa.array(res.proj_points, type=pa.float32()),
                "stack_code": pa.array(res.stack_code, type=pa.uint8()),
            }
        )
        pq.write_table(tbl, lineups_path, compression="zstd")
        run_outputs.append(
            ManifestIO(path=str(lineups_path), checksum_sha256=sha256_file(str(lineups_path)), logical_name="lineups.parquet")
        )

        # Metadata for downstream use.
        opt_res = compute_optimal_showdown_proj(
            players_enum_df.rename(columns={"salary": "salary", "proj_points": "proj_points"}),
            salary_cap=int(config.salary_cap),
            min_proj_points=float(config.min_proj_points),
        )
        optimal_proj_points = float(opt_res.optimal_proj_points)
        if not (optimal_proj_points > 0.0):
            raise ValueError(f"Invalid optimal_proj_points computed: {optimal_proj_points}")

        metadata = {
            "slate_id": str(config.slate_id),
            "sport": str(config.sport).lower(),
            "created_at_utc": utc_now().isoformat(),
            "salary_cap": int(config.salary_cap),
            "num_players": int(res.num_players),
            "num_lineups": int(res.num_lineups),
            "team_mapping": prep_meta["team_mapping"],
            "filters": {
                "min_proj_points": float(config.min_proj_points),
                "max_players": (None if config.max_players is None else int(config.max_players)),
            },
            "schema": {name: str(tbl.schema.field(name).type) for name in tbl.schema.names},
            "stack_code_map": {"0": "3-3", "1": "4-2", "2": "5-1"},
            "enriched_file": "lineups_enriched.parquet",
            "enriched_schema": {
                "cpt": "uint16",
                "u1": "uint16",
                "u2": "uint16",
                "u3": "uint16",
                "u4": "uint16",
                "u5": "uint16",
                "salary_used": "int32",
                "salary_left": "int32",
                "proj_points": "float",
                "stack_code": "uint8",
                "own_score_logprod": "float",
                "own_max_log": "float",
                "own_min_log": "float",
                "avg_corr": "float",
                "cpt_archetype": "dictionary<values=string, indices=int32, ordered=0>",
                "salary_left_bin": "dictionary<values=string, indices=int32, ordered=0>",
                "pct_proj_gap_to_optimal": "float",
                "pct_proj_gap_to_optimal_bin": "dictionary<values=string, indices=int32, ordered=0>",
            },
            "feature_maps": {
                "salary_left_bin_labels": [
                    "0_200",
                    "200_500",
                    "500_1000",
                    "1000_2000",
                    "2000_4000",
                    "4000_8000",
                    "8000_plus",
                ],
                "pct_proj_gap_to_optimal_bin_labels": [
                    "0_0.01",
                    "0.01_0.02",
                    "0.02_0.04",
                    "0.04_0.07",
                    "0.07_0.15",
                    "0.15_0.30",
                    "0.30_plus",
                ],
                "cpt_archetype_labels": [lbl for _, lbl in config.captain_tiers],
            },
            "optimal_proj_points": optimal_proj_points,
            "timings": res.metrics,
        }
        metadata_path = writer.run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        run_outputs.append(
            ManifestIO(path=str(metadata_path), checksum_sha256=sha256_file(str(metadata_path)), logical_name="metadata.json")
        )

        sample_df = sample_lineups_df(res, n=200)
        outputs = [
            (str(players_path), sha256_file(str(players_path)), "players.parquet"),
            (str(lineups_path), sha256_file(str(lineups_path)), "lineups.parquet"),
            (str(metadata_path), sha256_file(str(metadata_path)), "metadata.json"),
        ]
        writer.write_step(
            step_idx=2,
            step_name="enumerate_lineup_universe",
            df_in=players_enum_df,
            df_out=sample_df,
            inputs=[],
            outputs=outputs,
            metrics={
                **prep_meta,
                **res.metrics,
            },
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": int(len(players_enum_df)), "row_count_out": int(res.num_lineups)}
        step_logger.info(
            "Finished step: duration_s=%.3f num_players=%s num_lineups=%s",
            time.perf_counter() - t0,
            int(res.num_players),
            int(res.num_lineups),
        )

        # -------------------------
        # 03. enrich_lineup_universe_features
        # -------------------------
        step = "03_enrich_lineup_universe_features"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        enrich_cols, enrich_metrics = enrich_lineup_universe_showdown(
            players_enum_df=players_enum_df,
            cpt=res.cpt,
            u1=res.u1,
            u2=res.u2,
            u3=res.u3,
            u4=res.u4,
            u5=res.u5,
            salary_left=res.salary_left,
            proj_points=res.proj_points,
            corr_matrix_csv=config.corr_matrix_csv,
            captain_tiers=config.captain_tiers,
            optimal_proj_points=optimal_proj_points,
            own_log_eps=float(config.own_log_eps),
            salary_cap=int(config.salary_cap),
            min_proj_points=float(config.min_proj_points),
        )

        # Build dictionary-encoded categorical columns
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

        # Use signed dictionary indices for better pandas compatibility (pyarrow limitation).
        salary_left_bin = pa.DictionaryArray.from_arrays(
            pa.array(enrich_cols["salary_left_bin_code"], type=pa.int32()),
            pa.array(salary_left_bin_labels, type=pa.string()),
        )
        pct_gap_bin = pa.DictionaryArray.from_arrays(
            pa.array(enrich_cols["pct_proj_gap_to_optimal_bin_code"], type=pa.int32()),
            pa.array(gap_bin_labels, type=pa.string()),
        )
        cpt_arch = pa.DictionaryArray.from_arrays(
            pa.array(enrich_cols["cpt_archetype_code"], type=pa.int32()),
            pa.array(cpt_arch_labels, type=pa.string()),
        )

        enriched_tbl = pa.table(
            {
                "cpt": pa.array(res.cpt, type=pa.uint16()),
                "u1": pa.array(res.u1, type=pa.uint16()),
                "u2": pa.array(res.u2, type=pa.uint16()),
                "u3": pa.array(res.u3, type=pa.uint16()),
                "u4": pa.array(res.u4, type=pa.uint16()),
                "u5": pa.array(res.u5, type=pa.uint16()),
                "salary_used": pa.array(res.salary_used, type=pa.int32()),
                "salary_left": pa.array(res.salary_left, type=pa.int32()),
                "proj_points": pa.array(res.proj_points, type=pa.float32()),
                "stack_code": pa.array(res.stack_code, type=pa.uint8()),
                # requested features
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

        lineups_enriched_path = writer.run_dir / "lineups_enriched.parquet"
        pq.write_table(enriched_tbl, lineups_enriched_path, compression="zstd")
        run_outputs.append(
            ManifestIO(
                path=str(lineups_enriched_path),
                checksum_sha256=sha256_file(str(lineups_enriched_path)),
                logical_name="lineups_enriched.parquet",
            )
        )

        # Preview/sample for step artifacts
        k = min(200, int(res.num_lineups))
        preview_df = pd.DataFrame(
            {
                "cpt": res.cpt[:k],
                "u1": res.u1[:k],
                "u2": res.u2[:k],
                "u3": res.u3[:k],
                "u4": res.u4[:k],
                "u5": res.u5[:k],
                "salary_used": res.salary_used[:k],
                "salary_left": res.salary_left[:k],
                "proj_points": res.proj_points[:k],
                "stack_code": res.stack_code[:k],
                "own_score_logprod": np.asarray(enrich_cols["own_score_logprod"][:k]),
                "own_max_log": np.asarray(enrich_cols["own_max_log"][:k]),
                "own_min_log": np.asarray(enrich_cols["own_min_log"][:k]),
                "avg_corr": np.asarray(enrich_cols["avg_corr"][:k]),
                "cpt_archetype": [cpt_arch_labels[int(x)] for x in np.asarray(enrich_cols["cpt_archetype_code"][:k])],
                "salary_left_bin": [salary_left_bin_labels[int(x)] for x in np.asarray(enrich_cols["salary_left_bin_code"][:k])],
                "pct_proj_gap_to_optimal": np.asarray(enrich_cols["pct_proj_gap_to_optimal"][:k]),
                "pct_proj_gap_to_optimal_bin": [gap_bin_labels[int(x)] for x in np.asarray(enrich_cols["pct_proj_gap_to_optimal_bin_code"][:k])],
            }
        )

        outputs = [
            (str(lineups_enriched_path), sha256_file(str(lineups_enriched_path)), "lineups_enriched.parquet"),
        ]
        writer.write_step(
            step_idx=3,
            step_name="enrich_lineup_universe_features",
            df_in=None,
            df_out=preview_df,
            inputs=[(str(config.corr_matrix_csv), sha256_file(str(config.corr_matrix_csv)), "corr_matrix_csv")],
            outputs=outputs,
            metrics={
                "optimal_proj_points": float(enrich_metrics.optimal_proj_points),
                "corr_missing_pairs_rate": float(enrich_metrics.corr_missing_pairs_rate),
                "own_log_eps": float(enrich_metrics.own_log_eps),
            },
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": int(res.num_lineups), "row_count_out": int(res.num_lineups)}
        step_logger.info(
            "Finished step: duration_s=%.3f row_count_out=%s optimal_proj_points=%.3f",
            time.perf_counter() - t0,
            int(res.num_lineups),
            float(enrich_metrics.optimal_proj_points),
        )

        finished_at = utc_now()
        run_manifest = RunManifest(
            run_id=run_id,
            pipeline="contest",
            started_at=started_at,
            finished_at=finished_at,
            git_sha=try_get_git_sha(),
            config=asdict(config),
            inputs=run_inputs,
            outputs=run_outputs,
            row_counts_by_step=row_counts_by_step,
            warnings=[],
        )
        (writer.run_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        logger.info(
            "Run finished: duration_s=%.3f artifacts_dir=%s",
            (finished_at - started_at).total_seconds(),
            str(writer.run_dir),
        )
        return {"run_id": run_id, "artifacts_dir": str(writer.run_dir)}
    except Exception:
        logger.exception("Contest pipeline (lineup-gen) failed")
        raise


def run_contest_pipeline(config: ContestConfig) -> Dict[str, Any]:
    """
    Full Pipeline B runner:
      projections + correlations + DKEntries + theta -> shares -> field sims -> prune -> grade -> fill DKEntries.

    This orchestrates steps 00–08 from `agent/PIPELINES.md`.
    """
    started_at = utc_now()
    run_id, logger, writer = _init_contest_run(config)

    run_inputs: List[ManifestIO] = []
    run_outputs: List[ManifestIO] = []
    row_counts_by_step: Dict[str, Dict[str, int]] = {}

    try:
        if config.dkentries_csv is None:
            raise ValueError("ContestConfig.dkentries_csv is required for run_contest_pipeline")
        if config.theta_json is None:
            raise ValueError("ContestConfig.theta_json is required for run_contest_pipeline (per-contest bucketing added later)")

        # -------------------------
        # 00. ingest
        # -------------------------
        step = "00_ingest"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        # defer import; added by later tasks
        from dfs_opt.config.gpp_bins import load_gpp_bins_registry  # noqa: WPS433

        inv_rows = [
            {
                "kind": "projection",
                "sport": str(config.sport).lower(),
                "slate_id": str(config.slate_id),
                "path": str(config.projection_csv),
                "sha256": sha256_file(str(config.projection_csv)),
            },
            {
                "kind": "corr_matrix",
                "sport": str(config.sport).lower(),
                "slate_id": str(config.slate_id),
                "path": str(config.corr_matrix_csv),
                "sha256": sha256_file(str(config.corr_matrix_csv)),
            },
            {
                "kind": "dkentries",
                "sport": str(config.sport).lower(),
                "slate_id": str(config.slate_id),
                "path": str(config.dkentries_csv),
                "sha256": sha256_file(str(config.dkentries_csv)),
            },
        ]
        inv_df = pd.DataFrame(inv_rows)

        step00_inputs = [
            (str(config.projection_csv), sha256_file(str(config.projection_csv)), "sabersim_projection"),
            (str(config.corr_matrix_csv), sha256_file(str(config.corr_matrix_csv)), "corr_matrix_csv"),
            (str(config.dkentries_csv), sha256_file(str(config.dkentries_csv)), "dkentries_csv"),
        ]
        run_inputs.extend(
            [ManifestIO(path=p, checksum_sha256=chk, logical_name=ln) for (p, chk, ln) in step00_inputs]
        )

        # Snapshot registry key-set for debug/repro
        reg = load_gpp_bins_registry(path=getattr(config, "gpp_bins_yaml", None))
        inv_df["gpp_bins_registry_source"] = reg.source
        inv_df["gpp_bins_registry_count"] = int(len(reg.allowed_keys))

        writer.write_step(
            step_idx=0,
            step_name="ingest",
            df_in=None,
            df_out=inv_df,
            inputs=step00_inputs,
            outputs=[],
            metrics={"num_files": int(len(inv_df))},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": 0, "row_count_out": int(len(inv_df))}
        step_logger.info("Finished step: duration_s=%.3f row_count_out=%s", time.perf_counter() - t0, int(len(inv_df)))

        # -------------------------
        # 01. parse_projections
        # -------------------------
        step = "01_parse_projections"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        players_flex, m = parse_sabersim_showdown_csv(config.projection_csv)
        players_flex["sport"] = str(config.sport).lower()
        players_flex["slate_id"] = str(config.slate_id)

        writer.write_step(
            step_idx=1,
            step_name="parse_projections",
            df_in=None,
            df_out=players_flex,
            inputs=[(str(config.projection_csv), sha256_file(str(config.projection_csv)), "sabersim_projection")],
            outputs=[],
            metrics={
                "num_players": int(m["num_players"]),
                "num_rows_raw": int(m["num_rows_raw"]),
                "dropped_zero_proj": int(m["dropped_zero_proj"]),
            },
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": 0, "row_count_out": int(len(players_flex))}
        step_logger.info("Finished step: duration_s=%.3f row_count_out=%s", time.perf_counter() - t0, int(len(players_flex)))

        # -------------------------
        # 02. enumerate_lineup_universe
        # -------------------------
        step = "02_enumerate_lineup_universe"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        players_enum_df, arrays, prep_meta = prepare_player_arrays(
            players_flex,
            min_proj_points=float(config.min_proj_points),
            max_players=config.max_players,
        )
        res = enumerate_showdown_universe(arrays, salary_cap=int(config.salary_cap))

        # Write players + lineups at run root so other steps can reuse them.
        players_path = writer.run_dir / "players.parquet"
        players_enum_df.to_parquet(players_path, index=False)
        run_outputs.append(
            ManifestIO(path=str(players_path), checksum_sha256=sha256_file(str(players_path)), logical_name="players.parquet")
        )

        lineups_path = writer.run_dir / "lineups.parquet"
        tbl = pa.table(
            {
                "cpt": pa.array(res.cpt, type=pa.uint16()),
                "u1": pa.array(res.u1, type=pa.uint16()),
                "u2": pa.array(res.u2, type=pa.uint16()),
                "u3": pa.array(res.u3, type=pa.uint16()),
                "u4": pa.array(res.u4, type=pa.uint16()),
                "u5": pa.array(res.u5, type=pa.uint16()),
                "salary_used": pa.array(res.salary_used, type=pa.int32()),
                "salary_left": pa.array(res.salary_left, type=pa.int32()),
                "proj_points": pa.array(res.proj_points, type=pa.float32()),
                "stack_code": pa.array(res.stack_code, type=pa.uint8()),
            }
        )
        pq.write_table(tbl, lineups_path, compression="zstd")
        run_outputs.append(
            ManifestIO(path=str(lineups_path), checksum_sha256=sha256_file(str(lineups_path)), logical_name="lineups.parquet")
        )

        opt_res = compute_optimal_showdown_proj(
            players_enum_df.rename(columns={"salary": "salary", "proj_points": "proj_points"}),
            salary_cap=int(config.salary_cap),
            min_proj_points=float(config.min_proj_points),
        )
        optimal_proj_points = float(opt_res.optimal_proj_points)
        if not (optimal_proj_points > 0.0):
            raise ValueError(f"Invalid optimal_proj_points computed: {optimal_proj_points}")

        metadata = {
            "slate_id": str(config.slate_id),
            "sport": str(config.sport).lower(),
            "created_at_utc": utc_now().isoformat(),
            "salary_cap": int(config.salary_cap),
            "num_players": int(res.num_players),
            "num_lineups": int(res.num_lineups),
            "team_mapping": prep_meta["team_mapping"],
            "filters": {
                "min_proj_points": float(config.min_proj_points),
                "max_players": (None if config.max_players is None else int(config.max_players)),
            },
            "schema": {name: str(tbl.schema.field(name).type) for name in tbl.schema.names},
            "stack_code_map": {"0": "3-3", "1": "4-2", "2": "5-1"},
            "optimal_proj_points": optimal_proj_points,
            "timings": res.metrics,
        }
        metadata_path = writer.run_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        run_outputs.append(
            ManifestIO(path=str(metadata_path), checksum_sha256=sha256_file(str(metadata_path)), logical_name="metadata.json")
        )

        sample_df = sample_lineups_df(res, n=200)
        outputs = [
            (str(players_path), sha256_file(str(players_path)), "players.parquet"),
            (str(lineups_path), sha256_file(str(lineups_path)), "lineups.parquet"),
            (str(metadata_path), sha256_file(str(metadata_path)), "metadata.json"),
        ]
        writer.write_step(
            step_idx=2,
            step_name="enumerate_lineup_universe",
            df_in=players_enum_df,
            df_out=sample_df,
            inputs=[],
            outputs=outputs,
            metrics={**prep_meta, **res.metrics},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": int(len(players_enum_df)), "row_count_out": int(res.num_lineups)}
        step_logger.info(
            "Finished step: duration_s=%.3f num_players=%s num_lineups=%s",
            time.perf_counter() - t0,
            int(res.num_players),
            int(res.num_lineups),
        )

        # -------------------------
        # 03. enrich_lineup_universe_features
        # -------------------------
        step = "03_enrich_lineup_universe_features"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        enrich_cols, enrich_metrics = enrich_lineup_universe_showdown(
            players_enum_df=players_enum_df,
            cpt=res.cpt,
            u1=res.u1,
            u2=res.u2,
            u3=res.u3,
            u4=res.u4,
            u5=res.u5,
            salary_left=res.salary_left,
            proj_points=res.proj_points,
            corr_matrix_csv=config.corr_matrix_csv,
            captain_tiers=config.captain_tiers,
            optimal_proj_points=optimal_proj_points,
            own_log_eps=float(config.own_log_eps),
            salary_cap=int(config.salary_cap),
            min_proj_points=float(config.min_proj_points),
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
            pa.array(enrich_cols["salary_left_bin_code"], type=pa.int32()),
            pa.array(salary_left_bin_labels, type=pa.string()),
        )
        pct_gap_bin = pa.DictionaryArray.from_arrays(
            pa.array(enrich_cols["pct_proj_gap_to_optimal_bin_code"], type=pa.int32()),
            pa.array(gap_bin_labels, type=pa.string()),
        )
        cpt_arch = pa.DictionaryArray.from_arrays(
            pa.array(enrich_cols["cpt_archetype_code"], type=pa.int32()),
            pa.array(cpt_arch_labels, type=pa.string()),
        )

        enriched_tbl = pa.table(
            {
                "cpt": pa.array(res.cpt, type=pa.uint16()),
                "u1": pa.array(res.u1, type=pa.uint16()),
                "u2": pa.array(res.u2, type=pa.uint16()),
                "u3": pa.array(res.u3, type=pa.uint16()),
                "u4": pa.array(res.u4, type=pa.uint16()),
                "u5": pa.array(res.u5, type=pa.uint16()),
                "salary_used": pa.array(res.salary_used, type=pa.int32()),
                "salary_left": pa.array(res.salary_left, type=pa.int32()),
                "proj_points": pa.array(res.proj_points, type=pa.float32()),
                "stack_code": pa.array(res.stack_code, type=pa.uint8()),
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

        lineups_enriched_path = writer.run_dir / "lineups_enriched.parquet"
        pq.write_table(enriched_tbl, lineups_enriched_path, compression="zstd")
        run_outputs.append(
            ManifestIO(
                path=str(lineups_enriched_path),
                checksum_sha256=sha256_file(str(lineups_enriched_path)),
                logical_name="lineups_enriched.parquet",
            )
        )

        k = min(200, int(res.num_lineups))
        preview_df = pd.DataFrame(
            {
                "cpt": res.cpt[:k],
                "u1": res.u1[:k],
                "u2": res.u2[:k],
                "u3": res.u3[:k],
                "u4": res.u4[:k],
                "u5": res.u5[:k],
                "salary_used": res.salary_used[:k],
                "salary_left": res.salary_left[:k],
                "proj_points": res.proj_points[:k],
                "stack_code": res.stack_code[:k],
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

        outputs = [(str(lineups_enriched_path), sha256_file(str(lineups_enriched_path)), "lineups_enriched.parquet")]
        writer.write_step(
            step_idx=3,
            step_name="enrich_lineup_universe_features",
            df_in=None,
            df_out=preview_df,
            inputs=[(str(config.corr_matrix_csv), sha256_file(str(config.corr_matrix_csv)), "corr_matrix_csv")],
            outputs=outputs,
            metrics={
                "optimal_proj_points": float(enrich_metrics.optimal_proj_points),
                "corr_missing_pairs_rate": float(enrich_metrics.corr_missing_pairs_rate),
                "own_log_eps": float(enrich_metrics.own_log_eps),
            },
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": int(res.num_lineups), "row_count_out": int(res.num_lineups)}
        step_logger.info(
            "Finished step: duration_s=%.3f row_count_out=%s optimal_proj_points=%.3f",
            time.perf_counter() - t0,
            int(res.num_lineups),
            float(enrich_metrics.optimal_proj_points),
        )

        # -------------------------
        # 04. compute_softmax_shares
        # -------------------------
        step = "04_compute_softmax_shares"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        from dfs_opt.share_model.softmax_share import score_universe_with_theta  # noqa: WPS433

        lineup_utilities_path = writer.run_dir / "lineup_utilities.parquet"
        share_metrics = score_universe_with_theta(
            lineups_enriched_parquet=lineups_enriched_path,
            theta_json_path=Path(config.theta_json),
            out_parquet_path=lineup_utilities_path,
        )
        run_outputs.append(
            ManifestIO(
                path=str(lineup_utilities_path),
                checksum_sha256=sha256_file(str(lineup_utilities_path)),
                logical_name="lineup_utilities.parquet",
            )
        )

        preview_util = pd.DataFrame(
            {
                "lineup_id": np.arange(min(200, int(res.num_lineups))),
                "p": np.asarray(share_metrics.get("preview_p", []), dtype=float)[: min(200, len(share_metrics.get("preview_p", [])))],
            }
        )
        writer.write_step(
            step_idx=4,
            step_name="compute_softmax_shares",
            df_in=None,
            df_out=preview_util if len(preview_util) else pd.DataFrame({"note": ["see lineup_utilities.parquet"]}),
            inputs=[(str(config.theta_json), sha256_file(str(config.theta_json)), "theta.json")],
            outputs=[(str(lineup_utilities_path), sha256_file(str(lineup_utilities_path)), "lineup_utilities.parquet")],
            metrics=share_metrics,
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": int(res.num_lineups), "row_count_out": int(res.num_lineups)}
        step_logger.info("Finished step: duration_s=%.3f", time.perf_counter() - t0)

        # -------------------------
        # 05. simulate_contests_from_dkentries (+ prune)
        # -------------------------
        step = "05_simulate_contests_from_dkentries"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        from dfs_opt.io.dkentries import read_dkentries  # noqa: WPS433
        from dfs_opt.io.dk_api import DkApiClient  # noqa: WPS433
        from dfs_opt.simulation.contest_sim import (  # noqa: WPS433
            build_pruned_universe,
            sample_field_counts,
        )

        dkentries = read_dkentries(Path(config.dkentries_csv))
        contests = dkentries.entries[["contest_id"]].drop_duplicates().reset_index(drop=True)

        # DK API fetch + caching
        dk_api = DkApiClient(
            base_url=str(config.dk_api_base_url),
            cache_dir=Path(writer.run_dir) / "dk_api_cache",
            timeout_s=float(config.dk_api_timeout_s),
            headers=dict(config.dk_api_headers or {}),
        )
        per_contest_rows: List[Dict[str, Any]] = []
        contests_dir = writer.run_dir / "contests"
        contests_dir.mkdir(parents=True, exist_ok=True)

        # Load p(L) (full universe) once for pruning/sampling
        util_tbl = pq.read_table(lineup_utilities_path, columns=["lineup_id", "p"], memory_map=True)
        p = np.asarray(util_tbl["p"].to_numpy(zero_copy_only=False), dtype=np.float64)
        if not np.isfinite(p).all():
            raise ValueError("lineup_utilities contains non-finite probabilities")

        for contest_id in contests["contest_id"].astype(str).tolist():
            meta = dk_api.fetch_contest_meta(contest_id)
            n_user = int((dkentries.entries["contest_id"].astype(str) == str(contest_id)).sum())
            n_field = int(meta.contest_size) - n_user
            if n_field <= 0:
                raise ValueError(f"Contest {contest_id}: contest_size={meta.contest_size} < num_user_entries={n_user}")

            # prune and sample within the pruned set
            pruned = build_pruned_universe(p, mass_threshold=float(config.prune_mass_threshold))
            p_pruned = pruned.p_renorm
            field_counts = sample_field_counts(
                p=p_pruned,
                lineup_ids=pruned.lineup_ids,
                n_entries=n_field,
                seed=int(config.seed),
                dirichlet_alpha=(None if config.dirichlet_alpha is None else float(config.dirichlet_alpha)),
            )

            contest_dir = contests_dir / str(contest_id)
            contest_dir.mkdir(parents=True, exist_ok=True)
            field_path = contest_dir / "field_sample.parquet"
            pq.write_table(
                pa.table(
                    {"lineup_id": pa.array(field_counts["lineup_id"], type=pa.int64()), "dup_count": pa.array(field_counts["dup_count"], type=pa.int32())}
                ),
                field_path,
                compression="zstd",
            )
            pruned_path = contest_dir / "pruned_universe.parquet"
            pq.write_table(
                pa.table({"lineup_id": pa.array(pruned.lineup_ids, type=pa.int64()), "p": pa.array(p_pruned, type=pa.float64())}),
                pruned_path,
                compression="zstd",
            )

            per_contest_rows.append(
                {
                    "contest_id": str(contest_id),
                    "contest_size": int(meta.contest_size),
                    "entry_fee": float(meta.entry_fee),
                    "max_entries": int(meta.max_entries),
                    "num_user_entries": int(n_user),
                    "num_field_entries": int(n_field),
                    "pruned_size": int(len(pruned.lineup_ids)),
                    "pruned_mass_threshold": float(config.prune_mass_threshold),
                }
            )

        step05_df = pd.DataFrame(per_contest_rows)
        outputs = []
        for r in per_contest_rows:
            cid = str(r["contest_id"])
            outputs.append((str((contests_dir / cid / "field_sample.parquet")), sha256_file(str((contests_dir / cid / "field_sample.parquet"))), f"contests/{cid}/field_sample.parquet"))
            outputs.append((str((contests_dir / cid / "pruned_universe.parquet")), sha256_file(str((contests_dir / cid / "pruned_universe.parquet"))), f"contests/{cid}/pruned_universe.parquet"))

        writer.write_step(
            step_idx=5,
            step_name="simulate_contests_from_dkentries",
            df_in=dkentries.entries,
            df_out=step05_df.head(200) if len(step05_df) else pd.DataFrame({"note": ["no contests found"]}),
            inputs=[(str(config.dkentries_csv), sha256_file(str(config.dkentries_csv)), "dkentries_csv")],
            outputs=outputs,
            metrics={"num_contests": int(len(step05_df))},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": int(len(dkentries.entries)), "row_count_out": int(len(step05_df))}
        step_logger.info("Finished step: duration_s=%.3f num_contests=%s", time.perf_counter() - t0, int(len(step05_df)))

        # -------------------------
        # 06. grade_lineups
        # -------------------------
        step = "06_grade_lineups"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        from dfs_opt.simulation.contest_sim import grade_lineups_for_contest  # noqa: WPS433

        grade_rows: List[Dict[str, Any]] = []
        for r in per_contest_rows:
            cid = str(r["contest_id"])
            contest_dir = contests_dir / cid
            grades_path = contest_dir / "lineup_grades.parquet"

            meta = dk_api.fetch_contest_meta(cid)
            pruned_tbl = pq.read_table(contest_dir / "pruned_universe.parquet", memory_map=True)
            pruned_ids = np.asarray(pruned_tbl["lineup_id"].to_numpy(zero_copy_only=False), dtype=np.int64)
            pruned_p = np.asarray(pruned_tbl["p"].to_numpy(zero_copy_only=False), dtype=np.float64)
            field_tbl = pq.read_table(contest_dir / "field_sample.parquet", memory_map=True)
            field_counts = {
                "lineup_id": np.asarray(field_tbl["lineup_id"].to_numpy(zero_copy_only=False), dtype=np.int64),
                "dup_count": np.asarray(field_tbl["dup_count"].to_numpy(zero_copy_only=False), dtype=np.int32),
            }

            grade_res = grade_lineups_for_contest(
                players_parquet=players_path,
                lineups_parquet=lineups_path,
                field_counts=field_counts,
                candidate_lineup_ids=pruned_ids,
                payout_table=meta.payout_table,
                entry_fee=float(meta.entry_fee),
                num_sims=int(config.num_sims),
                seed=int(config.seed),
                corr_matrix_csv=Path(config.corr_matrix_csv),
                std_mode=str(config.std_mode),
                std_scale=float(config.std_scale),
                tie_break=str(config.tie_break),
            )
            pq.write_table(grade_res.table, grades_path, compression="zstd")
            grade_rows.append({"contest_id": cid, **grade_res.metrics})

        step06_df = pd.DataFrame(grade_rows)
        outputs = [
            (
                str(contests_dir / str(r["contest_id"]) / "lineup_grades.parquet"),
                sha256_file(str(contests_dir / str(r["contest_id"]) / "lineup_grades.parquet")),
                f"contests/{str(r['contest_id'])}/lineup_grades.parquet",
            )
            for r in per_contest_rows
        ]
        writer.write_step(
            step_idx=6,
            step_name="grade_lineups",
            df_in=None,
            df_out=step06_df.head(200) if len(step06_df) else pd.DataFrame({"note": ["no contests graded"]}),
            inputs=[],
            outputs=outputs,
            metrics={"num_contests": int(len(step06_df)), "num_sims": int(config.num_sims)},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": int(len(per_contest_rows)), "row_count_out": int(len(step06_df))}
        step_logger.info("Finished step: duration_s=%.3f", time.perf_counter() - t0)

        # -------------------------
        # 07. assign_best_lineups_to_entries
        # -------------------------
        step = "07_assign_best_lineups_to_entries"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        from dfs_opt.io.dkentries import fill_dkentries_with_assignments  # noqa: WPS433

        assignments: Dict[str, List[int]] = {}
        for r in per_contest_rows:
            cid = str(r["contest_id"])
            contest_dir = contests_dir / cid
            grades_tbl = pq.read_table(contest_dir / "lineup_grades.parquet", columns=["lineup_id", "roi"], memory_map=True)
            df = grades_tbl.to_pandas()
            df = df.sort_values(["roi", "lineup_id"], ascending=[False, True]).reset_index(drop=True)
            n_user = int(r["num_user_entries"])
            chosen = df["lineup_id"].astype(int).head(n_user).tolist()
            # enforce uniqueness (should already be unique)
            chosen = list(dict.fromkeys(chosen))
            if len(chosen) < n_user:
                raise ValueError(f"Contest {cid}: not enough unique lineups after grading to fill {n_user} entries")
            assignments[cid] = chosen[:n_user]

        filled = fill_dkentries_with_assignments(
            dkentries=dkentries,
            players_df=players_enum_df,
            lineups_parquet=lineups_path,
            assignments=assignments,
            output_format=str(config.dkentries_output_format),
        )
        out_csv = writer.run_dir / "DKEntries_filled.csv"
        filled.to_csv(out_csv, index=False)
        run_outputs.append(
            ManifestIO(path=str(out_csv), checksum_sha256=sha256_file(str(out_csv)), logical_name="DKEntries_filled.csv")
        )

        writer.write_step(
            step_idx=7,
            step_name="assign_best_lineups_to_entries",
            df_in=dkentries.entries,
            df_out=filled.head(200),
            inputs=[(str(config.dkentries_csv), sha256_file(str(config.dkentries_csv)), "dkentries_csv")],
            outputs=[(str(out_csv), sha256_file(str(out_csv)), "DKEntries_filled.csv")],
            metrics={"num_contests": int(len(assignments)), "num_rows": int(len(filled))},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": int(len(dkentries.entries)), "row_count_out": int(len(filled))}
        step_logger.info("Finished step: duration_s=%.3f", time.perf_counter() - t0)

        # -------------------------
        # 08. export_contest_csvs (human-readable CSVs next to parquet)
        # -------------------------
        step = "08_export_contest_csvs"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        from dfs_opt.io.contest_exports import export_contest_csvs  # noqa: WPS433

        lineups_enriched_path = writer.run_dir / "lineups_enriched.parquet"
        export_rows: List[Dict[str, Any]] = []
        export_outputs: List[Tuple[str, str, str]] = []

        for r in per_contest_rows:
            cid = str(r["contest_id"])
            contest_dir = contests_dir / cid

            export_res = export_contest_csvs(
                contest_id=cid,
                contest_dir=contest_dir,
                players_parquet=players_path,
                lineups_enriched_parquet=lineups_enriched_path,
                top_n=1000,
            )

            # Add to run manifest outputs for easier discovery / download.
            field_chk = sha256_file(str(export_res.field_csv))
            grades_chk = sha256_file(str(export_res.grades_top_csv))

            run_outputs.append(
                ManifestIO(
                    path=str(export_res.field_csv),
                    checksum_sha256=field_chk,
                    logical_name=f"contests/{cid}/field_sample.csv",
                )
            )
            run_outputs.append(
                ManifestIO(
                    path=str(export_res.grades_top_csv),
                    checksum_sha256=grades_chk,
                    logical_name=f"contests/{cid}/lineup_grades_top1000.csv",
                )
            )

            export_outputs.append((str(export_res.field_csv), field_chk, f"contests/{cid}/field_sample.csv"))
            export_outputs.append(
                (str(export_res.grades_top_csv), grades_chk, f"contests/{cid}/lineup_grades_top1000.csv")
            )

            export_rows.append(
                {
                    "contest_id": cid,
                    "n_field_unique": int(export_res.n_field_unique),
                    "n_top": int(export_res.n_top),
                    "field_csv": str(export_res.field_csv),
                    "grades_top_csv": str(export_res.grades_top_csv),
                }
            )

        step08_df = pd.DataFrame(export_rows)
        writer.write_step(
            step_idx=8,
            step_name="export_contest_csvs",
            df_in=None,
            df_out=step08_df.head(200) if len(step08_df) else pd.DataFrame({"note": ["no contests exported"]}),
            inputs=[],
            outputs=export_outputs,
            metrics={"num_contests": int(len(step08_df))},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_in": int(len(per_contest_rows)), "row_count_out": int(len(step08_df))}
        step_logger.info("Finished step: duration_s=%.3f", time.perf_counter() - t0)

        # -------------------------
        # 09. write_outputs (run manifest)
        # -------------------------
        finished_at = utc_now()
        run_manifest = RunManifest(
            run_id=run_id,
            pipeline="contest",
            started_at=started_at,
            finished_at=finished_at,
            git_sha=try_get_git_sha(),
            config=asdict(config),
            inputs=run_inputs,
            outputs=run_outputs,
            row_counts_by_step=row_counts_by_step,
            warnings=[],
        )
        (writer.run_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info(
            "Run finished: duration_s=%.3f artifacts_dir=%s",
            (finished_at - started_at).total_seconds(),
            str(writer.run_dir),
        )
        return {"run_id": run_id, "artifacts_dir": str(writer.run_dir)}
    except Exception:
        logger.exception("Contest pipeline failed")
        raise


