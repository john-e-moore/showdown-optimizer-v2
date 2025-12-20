from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dfs_opt.config.settings import ContestConfig
from dfs_opt.io.artifacts import ArtifactWriter, new_run_id, utc_now
from dfs_opt.models.manifests import ManifestIO, RunManifest
from dfs_opt.parsing.sabersim import parse_sabersim_showdown_csv
from dfs_opt.lineup_pool.enumerate_universe_showdown import (
    enumerate_showdown_universe,
    prepare_player_arrays,
    sample_lineups_df,
)
from dfs_opt.utils.git import try_get_git_sha
from dfs_opt.utils.hashing import sha256_file
from dfs_opt.utils.logging import configure_run_logger, get_step_logger


def run_contest_lineup_gen(config: ContestConfig) -> Dict[str, Any]:
    """
    Minimal Pipeline B runner for showdown lineup universe enumeration.

    Writes canonical artifacts under:
      <artifacts_root>/contest/<run_id>/...
    """
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
        "Run started: slate_id=%s sport=%s projection_csv=%s artifacts_root=%s seed=%s persist_step_outputs=%s git_sha=%s",
        str(config.slate_id),
        str(config.sport).lower(),
        str(config.projection_csv),
        str(config.artifacts_root),
        int(config.seed),
        bool(config.persist_step_outputs),
        try_get_git_sha(),
    )

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
                }
            ]
        )
        step00_inputs = [(str(config.projection_csv), sha256_file(str(config.projection_csv)), "sabersim_projection")]
        run_inputs.append(ManifestIO(path=step00_inputs[0][0], checksum_sha256=step00_inputs[0][1], logical_name=step00_inputs[0][2]))

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


