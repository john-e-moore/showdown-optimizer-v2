from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from dfs_opt.config.settings import ContestConfig, SegmentDefinitions, SegmentSizeBin, TrainingConfig


def load_training_config(path: Path) -> TrainingConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return training_config_from_dict(data)


def load_contest_config(path: Path) -> ContestConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return contest_config_from_dict(data)


def training_config_from_dict(data: Dict[str, Any]) -> TrainingConfig:
    seg = data.get("segment_definitions") or {}
    size_bins_raw = seg.get("size_bins")
    size_bins = None
    if isinstance(size_bins_raw, list):
        size_bins = [
            SegmentSizeBin(
                label=str(b["label"]),
                min_size=int(b["min_size"]),
                max_size_exclusive=(None if b.get("max_size_exclusive") is None else int(b["max_size_exclusive"])),
            )
            for b in size_bins_raw
        ]

    captain_tiers_raw = seg.get("captain_tiers")
    captain_tiers = None
    if isinstance(captain_tiers_raw, list):
        captain_tiers = [(int(t[0]), str(t[1])) for t in captain_tiers_raw]

    seg_defs = SegmentDefinitions(
        size_bins=size_bins if size_bins is not None else SegmentDefinitions().size_bins,
        captain_tiers=captain_tiers if captain_tiers is not None else SegmentDefinitions().captain_tiers,
    )

    return TrainingConfig(
        data_root=Path(data["data_root"]),
        artifacts_root=Path(data.get("artifacts_root", "artifacts")),
        seed=int(data.get("seed", 1337)),
        persist_step_outputs=bool(data.get("persist_step_outputs", False)),
        log_level=str(data.get("log_level", "INFO")),
        gpp_category=data.get("gpp_category"),
        segment_definitions=seg_defs,
        universe_root=Path(data.get("universe_root", TrainingConfig.universe_root)),
        share_model_enabled=bool(data.get("share_model_enabled", False)),
        share_model_lambda=float(data.get("share_model_lambda", 1e-3)),
        share_model_max_iter=int(data.get("share_model_max_iter", 200)),
        share_model_val_slate_frac=float(data.get("share_model_val_slate_frac", 0.2)),
        share_model_seed=int(data.get("share_model_seed", 1337)),
    )


def apply_cli_overrides(
    cfg: TrainingConfig,
    *,
    data_root: Optional[Path] = None,
    artifacts_root: Optional[Path] = None,
    seed: Optional[int] = None,
    persist_step_outputs: Optional[bool] = None,
    log_level: Optional[str] = None,
    gpp_category: Optional[str] = None,
    universe_root: Optional[Path] = None,
    share_model_enabled: Optional[bool] = None,
    share_model_lambda: Optional[float] = None,
    share_model_max_iter: Optional[int] = None,
    share_model_val_slate_frac: Optional[float] = None,
    share_model_seed: Optional[int] = None,
) -> TrainingConfig:
    return replace(
        cfg,
        data_root=data_root if data_root is not None else cfg.data_root,
        artifacts_root=artifacts_root if artifacts_root is not None else cfg.artifacts_root,
        seed=seed if seed is not None else cfg.seed,
        persist_step_outputs=persist_step_outputs
        if persist_step_outputs is not None
        else cfg.persist_step_outputs,
        log_level=log_level if log_level is not None else cfg.log_level,
        gpp_category=gpp_category if gpp_category is not None else cfg.gpp_category,
        universe_root=universe_root if universe_root is not None else cfg.universe_root,
        share_model_enabled=share_model_enabled if share_model_enabled is not None else cfg.share_model_enabled,
        share_model_lambda=share_model_lambda if share_model_lambda is not None else cfg.share_model_lambda,
        share_model_max_iter=share_model_max_iter if share_model_max_iter is not None else cfg.share_model_max_iter,
        share_model_val_slate_frac=share_model_val_slate_frac
        if share_model_val_slate_frac is not None
        else cfg.share_model_val_slate_frac,
        share_model_seed=share_model_seed if share_model_seed is not None else cfg.share_model_seed,
    )


def contest_config_from_dict(data: Dict[str, Any]) -> ContestConfig:
    return ContestConfig(
        projection_csv=Path(data["projection_csv"]),
        corr_matrix_csv=Path(data["corr_matrix_csv"]),
        slate_id=str(data["slate_id"]),
        sport=str(data.get("sport", "nba")),
        artifacts_root=Path(data.get("artifacts_root", "artifacts")),
        seed=int(data.get("seed", 1337)),
        persist_step_outputs=bool(data.get("persist_step_outputs", False)),
        log_level=str(data.get("log_level", "INFO")),
        salary_cap=int(data.get("salary_cap", 50000)),
        min_proj_points=float(data.get("min_proj_points", 0.0)),
        max_players=(None if data.get("max_players") is None else int(data.get("max_players"))),
        captain_tiers=[(int(t[0]), str(t[1])) for t in (data.get("captain_tiers") or SegmentDefinitions().captain_tiers)],
        own_log_eps=float(data.get("own_log_eps", 1e-6)),
    )


def apply_contest_cli_overrides(
    cfg: ContestConfig,
    *,
    projection_csv: Optional[Path] = None,
    corr_matrix_csv: Optional[Path] = None,
    slate_id: Optional[str] = None,
    sport: Optional[str] = None,
    artifacts_root: Optional[Path] = None,
    seed: Optional[int] = None,
    persist_step_outputs: Optional[bool] = None,
    log_level: Optional[str] = None,
    salary_cap: Optional[int] = None,
    min_proj_points: Optional[float] = None,
    max_players: Optional[int] = None,
) -> ContestConfig:
    return replace(
        cfg,
        projection_csv=projection_csv if projection_csv is not None else cfg.projection_csv,
        corr_matrix_csv=corr_matrix_csv if corr_matrix_csv is not None else cfg.corr_matrix_csv,
        slate_id=slate_id if slate_id is not None else cfg.slate_id,
        sport=sport if sport is not None else cfg.sport,
        artifacts_root=artifacts_root if artifacts_root is not None else cfg.artifacts_root,
        seed=seed if seed is not None else cfg.seed,
        persist_step_outputs=persist_step_outputs
        if persist_step_outputs is not None
        else cfg.persist_step_outputs,
        log_level=log_level if log_level is not None else cfg.log_level,
        salary_cap=salary_cap if salary_cap is not None else cfg.salary_cap,
        min_proj_points=min_proj_points if min_proj_points is not None else cfg.min_proj_points,
        max_players=max_players if max_players is not None else cfg.max_players,
    )


