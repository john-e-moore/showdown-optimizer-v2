from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from dfs_opt.config.settings import TrainingConfig
from dfs_opt.config.gpp_bins import load_gpp_bins_registry, validate_gpp_category
from dfs_opt.distributions.fit import fit_target_distributions
from dfs_opt.features.enrich_showdown import enrich_showdown_entries
from dfs_opt.features.optimal import OptimalShowdownCache, add_optimal_and_gap, compute_optimal_showdown_proj, players_fingerprint
from dfs_opt.io.artifacts import ArtifactWriter, new_run_id, utc_now
from dfs_opt.io.inventory import SlateInputs, build_input_inventory, find_showdown_slates, load_slate_inputs
from dfs_opt.models.manifests import ManifestIO, RunManifest
from dfs_opt.parsing.dk_standings import parse_dk_showdown_entries
from dfs_opt.parsing.sabersim import parse_sabersim_showdown_csv
from dfs_opt.utils.git import try_get_git_sha
from dfs_opt.utils.hashing import sha256_file
from dfs_opt.utils.logging import configure_run_logger, get_step_logger


def run_training_pipeline(config: TrainingConfig) -> Dict[str, Any]:
    """
    Pipeline A entrypoint.

    Implemented incrementally per agent/PIPELINES.md. This function orchestrates reading/writing
    and uses pure transforms for parsing/features/distributions.
    """
    random.seed(config.seed)

    started_at = utc_now()
    run_id = new_run_id()
    writer = ArtifactWriter(artifacts_root=config.artifacts_root, pipeline="training", run_id=run_id)
    writer.init_run_dirs()

    # --- agent debug logging (NDJSON) ---
    # Writes to the provisioned log path for this Cursor debug session.
    # Keep this tiny + safe; never log secrets/PII.
    def _agent_log(*, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
        try:
            payload = {
                "sessionId": "debug-session",
                "runId": str(run_id),
                "hypothesisId": str(hypothesis_id),
                "location": str(location),
                "message": str(message),
                "data": data,
                "timestamp": int(time.time() * 1000),
            }
            with open("/home/john/showdown-optimizer-v2/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Never fail the pipeline due to debug logging.
            return

    logger = configure_run_logger(
        logs_dir=writer.logs_dir,
        run_id=run_id,
        pipeline="training",
        level=getattr(config, "log_level", "INFO"),
    )
    logger.info(
        "Run started: data_root=%s artifacts_root=%s seed=%s persist_step_outputs=%s gpp_category=%s git_sha=%s",
        str(config.data_root),
        str(config.artifacts_root),
        config.seed,
        config.persist_step_outputs,
        config.gpp_category,
        try_get_git_sha(),
    )

    gpp_registry = load_gpp_bins_registry()
    allowed_gpp_categories = gpp_registry.allowed_keys
    if config.gpp_category:
        validate_gpp_category(
            config.gpp_category,
            allowed=allowed_gpp_categories,
            context="training config/CLI",
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

        slates_meta = find_showdown_slates(config.data_root)
        slate_inputs: List[SlateInputs] = []
        extract_base = writer.run_dir / "tmp" / "extracted"
        for sport, slate_id, slate_dir in slates_meta:
            slate_inputs.append(
                load_slate_inputs(
                    slate_dir=slate_dir,
                    sport=sport,
                    slate_id=slate_id,
                    extract_dir=extract_base / sport / slate_id,
                )
            )

        inv_df = build_input_inventory(slate_inputs)
        step00_inputs = [(r["path"], r["sha256"], f'{r["kind"]}') for r in inv_df.to_dict(orient="records")]
        for p, chk, ln in step00_inputs:
            run_inputs.append(ManifestIO(path=p, checksum_sha256=chk, logical_name=ln))

        _step00 = writer.write_step(
            step_idx=0,
            step_name="ingest",
            df_in=None,
            df_out=inv_df,
            inputs=step00_inputs,
            outputs=[],
            metrics={"num_files": int(len(inv_df)), "num_slates": int(len(slate_inputs))},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_out": int(len(inv_df)), "row_count_in": 0}
        step_logger.info(
            "Finished step: duration_s=%.3f row_count_out=%s num_files=%s num_slates=%s",
            time.perf_counter() - t0,
            int(len(inv_df)),
            int(len(inv_df)),
            int(len(slate_inputs)),
        )

        # -------------------------
        # 01. parse_projections
        # -------------------------
        step = "01_parse_projections"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        players_tables = []
        metrics_01 = {"num_players_total": 0, "dropped_zero_proj_total": 0}
        step01_inputs: List[Tuple[str, str, str]] = []
        for s in slate_inputs:
            df_p, m = parse_sabersim_showdown_csv(s.projection_csv)
            df_p["sport"] = s.sport.lower()
            df_p["slate_id"] = s.slate_id
            players_tables.append(df_p)
            metrics_01["num_players_total"] += int(m["num_players"])
            metrics_01["dropped_zero_proj_total"] += int(m["dropped_zero_proj"])
            step01_inputs.append((str(s.projection_csv), sha256_file(str(s.projection_csv)), "sabersim_projection"))
            step_logger.info(
                "Parsed projections: slate_id=%s num_players=%s dropped_zero_proj=%s path=%s",
                s.slate_id,
                int(m["num_players"]),
                int(m["dropped_zero_proj"]),
                str(s.projection_csv),
            )

        players_all = _concat_or_empty(players_tables)
        _step01 = writer.write_step(
            step_idx=1,
            step_name="parse_projections",
            df_in=None,
            df_out=players_all,
            inputs=step01_inputs,
            outputs=[],
            metrics=metrics_01,
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_out": int(len(players_all)), "row_count_in": 0}
        step_logger.info(
            "Finished step: duration_s=%.3f row_count_out=%s num_players_total=%s dropped_zero_proj_total=%s",
            time.perf_counter() - t0,
            int(len(players_all)),
            int(metrics_01["num_players_total"]),
            int(metrics_01["dropped_zero_proj_total"]),
        )

        # -------------------------
        # 02. parse_contest_entries
        # -------------------------
        step = "02_parse_contest_entries"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        parsed_entries_tables = []
        metrics_02 = {"num_contests": 0, "invalid_lineup_rows": 0, "parse_success_rate_mean": 0.0}
        step02_inputs: List[Tuple[str, str, str]] = []

        # Counters for debugging why no contests pass filter
        _dbg_seen_files = 0
        _dbg_kept_files = 0
        _dbg_skipped_mismatch = 0
        _dbg_gpp_counts: Counter[str] = Counter()
        _dbg_sport_standings_counts: Counter[str] = Counter()
        _dbg_sampled = 0
        _dbg_sampled_skips = 0

        # #region agent log
        total_slates = int(len(slate_inputs))
        total_standings = int(sum(len(s.standings_files) for s in slate_inputs))
        for s in slate_inputs:
            _dbg_sport_standings_counts[str(s.sport).lower()] += int(len(s.standings_files))
        _agent_log(
            hypothesis_id="H1",
            location="src/dfs_opt/pipelines/training.py:02_parse_contest_entries:init",
            message="Step02 starting; summarize standings discovery and filter config",
            data={
                "config_gpp_category": config.gpp_category,
                "num_slates": total_slates,
                "total_standings_files": total_standings,
                "standings_files_by_sport": dict(_dbg_sport_standings_counts),
            },
        )
        # #endregion

        for s in slate_inputs:
            for standings_path in s.standings_files:
                _dbg_seen_files += 1
                # quick size bin inference from file (use row count, fallback to 0)
                try:
                    contest_size = int(len(pd.read_csv(standings_path)))
                except Exception:
                    contest_size = 0
                size_bin = _size_bin_label(contest_size, config)

                if _dbg_sampled < 8:
                    # #region agent log
                    _agent_log(
                        hypothesis_id="H3",
                        location="src/dfs_opt/pipelines/training.py:02_parse_contest_entries:pre_parse",
                        message="About to parse standings CSV; record inferred size bin inputs",
                        data={
                            "sport": str(s.sport).lower(),
                            "slate_id": str(s.slate_id),
                            "path": str(standings_path),
                            "contest_size_len_csv": int(contest_size),
                            "size_bin": str(size_bin),
                            "filter_gpp_category": config.gpp_category,
                        },
                    )
                    # #endregion
                    _dbg_sampled += 1

                df_e, m = parse_dk_showdown_entries(
                    standings_path, sport=s.sport, slate_id=s.slate_id, size_bin=size_bin
                )

                validate_gpp_category(
                    str(m["gpp_category"]),
                    allowed=allowed_gpp_categories,
                    context=f"DK standings parse ({standings_path})",
                )

                _dbg_gpp_counts[str(m.get("gpp_category"))] += 1
                if _dbg_sampled < 8:
                    # #region agent log
                    _agent_log(
                        hypothesis_id="H2",
                        location="src/dfs_opt/pipelines/training.py:02_parse_contest_entries:post_parse",
                        message="Parsed standings; record inferred gpp_category and key metrics",
                        data={
                            "sport": str(s.sport).lower(),
                            "slate_id": str(s.slate_id),
                            "path": str(standings_path),
                            "parsed_gpp_category": str(m.get("gpp_category")),
                            "parsed_contest_size": int(m.get("contest_size") or 0),
                            "parsed_max_entries_per_user": m.get("max_entries_per_user"),
                            "parse_success_rate": float(m.get("parse_success_rate") or 0.0),
                            "invalid_lineup_rows": int(m.get("invalid_lineup_rows") or 0),
                            "filter_gpp_category": config.gpp_category,
                        },
                    )
                    # #endregion

                if config.gpp_category and m["gpp_category"] != config.gpp_category:
                    _dbg_skipped_mismatch += 1
                    if _dbg_sampled_skips < 8:
                        # #region agent log
                        _agent_log(
                            hypothesis_id="H1",
                            location="src/dfs_opt/pipelines/training.py:02_parse_contest_entries:filter",
                            message="Skipping contest due to gpp_category mismatch",
                            data={
                                "path": str(standings_path),
                                "parsed_gpp_category": str(m.get("gpp_category")),
                                "filter_gpp_category": config.gpp_category,
                            },
                        )
                        # #endregion
                        _dbg_sampled_skips += 1
                    continue

                parsed_entries_tables.append(df_e)
                _dbg_kept_files += 1
                metrics_02["num_contests"] += 1
                metrics_02["invalid_lineup_rows"] += int(m["invalid_lineup_rows"])
                metrics_02["parse_success_rate_mean"] += float(m["parse_success_rate"])
                step02_inputs.append((str(standings_path), sha256_file(str(standings_path)), "dk_standings"))
                step_logger.info(
                    "Parsed contest: slate_id=%s contest_size=%s size_bin=%s gpp_category=%s parse_success_rate=%.3f invalid_lineup_rows=%s path=%s",
                    s.slate_id,
                    contest_size,
                    size_bin,
                    str(m.get("gpp_category")),
                    float(m["parse_success_rate"]),
                    int(m["invalid_lineup_rows"]),
                    str(standings_path),
                )

        parsed_entries = _concat_or_empty(parsed_entries_tables)
        if metrics_02["num_contests"] > 0:
            metrics_02["parse_success_rate_mean"] /= metrics_02["num_contests"]

        # #region agent log
        _agent_log(
            hypothesis_id="H1",
            location="src/dfs_opt/pipelines/training.py:02_parse_contest_entries:summary",
            message="Step02 summary; how many contests were seen vs kept and which categories were produced",
            data={
                "seen_files": int(_dbg_seen_files),
                "kept_files": int(_dbg_kept_files),
                "skipped_mismatch": int(_dbg_skipped_mismatch),
                "distinct_parsed_gpp_categories": int(len(_dbg_gpp_counts)),
                "parsed_gpp_category_counts_top": dict(_dbg_gpp_counts.most_common(12)),
                "parsed_entries_shape": [int(parsed_entries.shape[0]), int(parsed_entries.shape[1])],
                "parsed_entries_columns": [str(c) for c in list(parsed_entries.columns)[:50]],
            },
        )
        # #endregion

        _step02 = writer.write_step(
            step_idx=2,
            step_name="parse_contest_entries",
            df_in=None,
            df_out=parsed_entries,
            inputs=step02_inputs,
            outputs=[],
            metrics=metrics_02,
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {"row_count_out": int(len(parsed_entries)), "row_count_in": 0}
        step_logger.info(
            "Finished step: duration_s=%.3f row_count_out=%s num_contests=%s invalid_lineup_rows=%s parse_success_rate_mean=%.3f",
            time.perf_counter() - t0,
            int(len(parsed_entries)),
            int(metrics_02["num_contests"]),
            int(metrics_02["invalid_lineup_rows"]),
            float(metrics_02["parse_success_rate_mean"]),
        )

        # -------------------------
        # 03. join_and_enrich
        # -------------------------
        step = "03_join_and_enrich"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        enriched_tables = []
        unmatched_total = 0

        # #region agent log
        _agent_log(
            hypothesis_id="H4",
            location="src/dfs_opt/pipelines/training.py:03_join_and_enrich:pre_groupby",
            message="About to group parsed_entries by slate_id; record schema to explain KeyError",
            data={
                "parsed_entries_shape": [int(parsed_entries.shape[0]), int(parsed_entries.shape[1])],
                "parsed_entries_columns": [str(c) for c in list(parsed_entries.columns)[:50]],
                "has_slate_id": bool("slate_id" in parsed_entries.columns),
            },
        )
        # #endregion

        for slate_id, grp in parsed_entries.groupby("slate_id", dropna=False):
            ply = players_all[players_all["slate_id"] == slate_id].copy()
            if len(ply) == 0:
                continue
            enriched, m = enrich_showdown_entries(
                grp,
                ply.rename(
                    columns={
                        "name_norm": "name_norm",
                        "team": "team",
                        "salary": "salary",
                        "proj_points": "proj_points",
                    }
                ),
                captain_tiers=config.segment_definitions.captain_tiers,
            )
            unmatched_total += int(m["unmatched_player_names"])
            enriched_tables.append(enriched)

        enriched_entries = _concat_or_empty(enriched_tables)
        join_coverage = 0.0 if len(parsed_entries) == 0 else float(len(enriched_entries) / len(parsed_entries))
        _step03 = writer.write_step(
            step_idx=3,
            step_name="join_and_enrich",
            df_in=parsed_entries,
            df_out=enriched_entries,
            inputs=[],
            outputs=[],
            metrics={"unmatched_player_names": unmatched_total, "join_coverage": join_coverage},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {
            "row_count_in": int(len(parsed_entries)),
            "row_count_out": int(len(enriched_entries)),
        }
        step_logger.info(
            "Finished step: duration_s=%.3f row_count_in=%s row_count_out=%s join_coverage=%.3f unmatched_player_names=%s",
            time.perf_counter() - t0,
            int(len(parsed_entries)),
            int(len(enriched_entries)),
            join_coverage,
            unmatched_total,
        )

        # -------------------------
        # 04. optimal_and_gap
        # -------------------------
        step = "04_optimal_and_gap"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        cache = OptimalShowdownCache()
        per_slate_metrics: Dict[str, Any] = {}
        enriched_with_gap_tables = []
        for slate_id, grp in enriched_entries.groupby("slate_id", dropna=False):
            ply = players_all[players_all["slate_id"] == slate_id].copy()
            key = f"{slate_id}:{players_fingerprint(ply)}"
            cached = cache.get(key)
            if cached is None:
                res = compute_optimal_showdown_proj(ply.rename(columns={"salary": "salary", "proj_points": "proj_points"}))
                cache.set(key, res)
            else:
                res = cached
            per_slate_metrics[str(slate_id)] = {
                "optimal_proj_points": res.optimal_proj_points,
                "compute_time_s": res.compute_time_s,
                "num_players_considered": res.num_players_considered,
            }
            step_logger.info(
                "Optimal computed: slate_id=%s optimal_proj_points=%.3f compute_time_s=%.3f num_players_considered=%s",
                str(slate_id),
                float(res.optimal_proj_points),
                float(res.compute_time_s),
                int(res.num_players_considered),
            )
            enriched_with_gap_tables.append(add_optimal_and_gap(grp, optimal_proj_points=res.optimal_proj_points))

        enriched_with_gap = _concat_or_empty(enriched_with_gap_tables)
        _step04 = writer.write_step(
            step_idx=4,
            step_name="optimal_and_gap",
            df_in=enriched_entries,
            df_out=enriched_with_gap,
            inputs=[],
            outputs=[],
            metrics={"per_slate": per_slate_metrics},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {
            "row_count_in": int(len(enriched_entries)),
            "row_count_out": int(len(enriched_with_gap)),
        }
        step_logger.info(
            "Finished step: duration_s=%.3f row_count_in=%s row_count_out=%s",
            time.perf_counter() - t0,
            int(len(enriched_entries)),
            int(len(enriched_with_gap)),
        )

        # -------------------------
        # 05. fit_target_distributions
        # -------------------------
        step = "05_fit_target_distributions"
        step_logger = get_step_logger(logger, step=step)
        step_logger.info("Starting step")
        t0 = time.perf_counter()

        td_dir = writer.run_dir / "target_distributions"
        td_dir.mkdir(parents=True, exist_ok=False)
        dist_rows = []
        for gpp_category, grp in enriched_with_gap.groupby("gpp_category", dropna=False):
            cat = str(gpp_category)
            dist = fit_target_distributions(grp, gpp_category=cat)
            out_path = td_dir / f"{cat}.json"
            out_path.write_text(
                json.dumps(dist.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            run_outputs.append(
                ManifestIO(
                    path=str(out_path),
                    checksum_sha256=sha256_file(str(out_path)),
                    logical_name=f"target_distributions:{cat}",
                )
            )
            dist_rows.append({"gpp_category": cat, "n_rows": int(len(grp)), "path": str(out_path)})
            step_logger.info("Wrote target distribution: gpp_category=%s n_rows=%s path=%s", cat, int(len(grp)), str(out_path))

        dist_df = pd.DataFrame(dist_rows)
        _step05 = writer.write_step(
            step_idx=5,
            step_name="fit_target_distributions",
            df_in=enriched_with_gap,
            df_out=dist_df,
            inputs=[],
            outputs=[(r["path"], sha256_file(r["path"]), "target_distribution") for r in dist_rows],
            metrics={"num_categories": int(len(dist_rows))},
            persist_parquet=config.persist_step_outputs,
        )
        row_counts_by_step[step] = {
            "row_count_in": int(len(enriched_with_gap)),
            "row_count_out": int(len(dist_df)),
        }
        step_logger.info(
            "Finished step: duration_s=%.3f row_count_in=%s row_count_out=%s num_categories=%s",
            time.perf_counter() - t0,
            int(len(enriched_with_gap)),
            int(len(dist_df)),
            int(len(dist_rows)),
        )

        # Write final parquet output
        entries_path = writer.run_dir / "entries_enriched.parquet"
        enriched_with_gap.to_parquet(entries_path, index=False)
        run_outputs.append(
            ManifestIO(
                path=str(entries_path),
                checksum_sha256=sha256_file(str(entries_path)),
                logical_name="entries_enriched.parquet",
            )
        )
        logger.info("Wrote output: %s", str(entries_path))

        finished_at = utc_now()
        run_manifest = RunManifest(
            run_id=run_id,
            pipeline="training",
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

        logger.info("Run finished: duration_s=%.3f artifacts_dir=%s", (finished_at - started_at).total_seconds(), str(writer.run_dir))
        return {"run_id": run_id, "artifacts_dir": str(writer.run_dir)}
    except Exception:
        logger.exception("Training pipeline failed")
        raise


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _concat_or_empty(tables: List[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _size_bin_label(contest_size: int, config: TrainingConfig) -> str:
    for b in config.segment_definitions.size_bins:
        if b.contains(contest_size):
            return b.label
    # should never happen if bins include a catch-all
    return config.segment_definitions.size_bins[-1].label


