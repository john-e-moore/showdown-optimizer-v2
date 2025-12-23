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
        dkentries_csv=(None if data.get("dkentries_csv") is None else Path(data["dkentries_csv"])),
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
        theta_json=(None if data.get("theta_json") is None else Path(data["theta_json"])),
        share_models_root=(None if data.get("share_models_root") is None else Path(data["share_models_root"])),
        gpp_bins_yaml=(None if data.get("gpp_bins_yaml") is None else Path(data["gpp_bins_yaml"])),
        dk_api_base_url=str(data.get("dk_api_base_url", ContestConfig.dk_api_base_url)),
        dk_api_timeout_s=float(data.get("dk_api_timeout_s", 20.0)),
        dk_api_headers=(None if data.get("dk_api_headers") is None else dict(data["dk_api_headers"])),
        prune_mass_threshold=float(data.get("prune_mass_threshold", 0.9995)),
        dirichlet_alpha=(None if data.get("dirichlet_alpha") is None else float(data["dirichlet_alpha"])),
        num_sims=int(data.get("num_sims", 2000)),
        std_mode=str(data.get("std_mode", "dk_std_or_fallback")),
        std_scale=float(data.get("std_scale", 1.0)),
        tie_break=str(data.get("tie_break", "lineup_id")),
        dkentries_output_format=str(data.get("dkentries_output_format", "name_id")),
    )


def apply_contest_cli_overrides(
    cfg: ContestConfig,
    *,
    projection_csv: Optional[Path] = None,
    corr_matrix_csv: Optional[Path] = None,
    dkentries_csv: Optional[Path] = None,
    slate_id: Optional[str] = None,
    sport: Optional[str] = None,
    artifacts_root: Optional[Path] = None,
    seed: Optional[int] = None,
    persist_step_outputs: Optional[bool] = None,
    log_level: Optional[str] = None,
    salary_cap: Optional[int] = None,
    min_proj_points: Optional[float] = None,
    max_players: Optional[int] = None,
    theta_json: Optional[Path] = None,
    share_models_root: Optional[Path] = None,
    gpp_bins_yaml: Optional[Path] = None,
    dk_api_base_url: Optional[str] = None,
    dk_api_timeout_s: Optional[float] = None,
    prune_mass_threshold: Optional[float] = None,
    dirichlet_alpha: Optional[float] = None,
    num_sims: Optional[int] = None,
    std_mode: Optional[str] = None,
    std_scale: Optional[float] = None,
    tie_break: Optional[str] = None,
    dkentries_output_format: Optional[str] = None,
) -> ContestConfig:
    return replace(
        cfg,
        projection_csv=projection_csv if projection_csv is not None else cfg.projection_csv,
        corr_matrix_csv=corr_matrix_csv if corr_matrix_csv is not None else cfg.corr_matrix_csv,
        dkentries_csv=dkentries_csv if dkentries_csv is not None else cfg.dkentries_csv,
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
        theta_json=theta_json if theta_json is not None else cfg.theta_json,
        share_models_root=share_models_root if share_models_root is not None else cfg.share_models_root,
        gpp_bins_yaml=gpp_bins_yaml if gpp_bins_yaml is not None else cfg.gpp_bins_yaml,
        dk_api_base_url=dk_api_base_url if dk_api_base_url is not None else cfg.dk_api_base_url,
        dk_api_timeout_s=dk_api_timeout_s if dk_api_timeout_s is not None else cfg.dk_api_timeout_s,
        prune_mass_threshold=prune_mass_threshold
        if prune_mass_threshold is not None
        else cfg.prune_mass_threshold,
        dirichlet_alpha=dirichlet_alpha if dirichlet_alpha is not None else cfg.dirichlet_alpha,
        num_sims=num_sims if num_sims is not None else cfg.num_sims,
        std_mode=std_mode if std_mode is not None else cfg.std_mode,
        std_scale=std_scale if std_scale is not None else cfg.std_scale,
        tie_break=tie_break if tie_break is not None else cfg.tie_break,
        dkentries_output_format=dkentries_output_format
        if dkentries_output_format is not None
        else cfg.dkentries_output_format,
    )


