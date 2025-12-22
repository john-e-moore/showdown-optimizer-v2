from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from dfs_opt.config.gpp_bins import load_gpp_bins_registry, validate_gpp_category
from dfs_opt.config.load import apply_cli_overrides, load_training_config
from dfs_opt.config.settings import TrainingConfig
from dfs_opt.pipelines.training import run_training_pipeline

app = typer.Typer(add_completion=False, help="Pipeline A: training/target distributions")


@app.command("run")
def run(
    config: Optional[Path] = typer.Option(
        None, help="Optional YAML config file; CLI options override values from config"
    ),
    data_root: Path = typer.Option(..., help="Root containing raw data (e.g. data/historical/raw)"),
    artifacts_root: Path = typer.Option(Path("artifacts"), help="Where to write run artifacts"),
    seed: int = typer.Option(1337, help="Global seed recorded in manifests"),
    persist_step_outputs: bool = typer.Option(
        False, help="If set, write step outputs as parquet alongside preview/schema/manifests"
    ),
    log_level: str = typer.Option("INFO", help="Python logging level (INFO, DEBUG, WARNING, ...)"),
    gpp_category: Optional[str] = typer.Option(
        None, help="If set, run only this segment bucket (e.g. nba-showdown-mme-1k-10k)"
    ),
    universe_root: Path = typer.Option(
        Path("data/historical/enriched"),
        help="Root containing precomputed lineup universes (expects dk-results/showdown/<sport>/<slate_id>/lineups_enriched.parquet)",
    ),
    share_model_enabled: bool = typer.Option(False, help="If set, fit the softmax lineup share model (step 06)"),
    share_model_lambda: float = typer.Option(1e-3, help="L2 regularization strength for share model"),
    share_model_max_iter: int = typer.Option(200, help="Max optimizer iterations for share model"),
    share_model_val_slate_frac: float = typer.Option(0.2, help="Validation slate fraction for share model (hold out full slates)"),
    share_model_seed: int = typer.Option(1337, help="RNG seed for share-model train/val split"),
) -> None:
    if gpp_category:
        reg = load_gpp_bins_registry()
        try:
            validate_gpp_category(gpp_category, allowed=reg.allowed_keys, context="--gpp-category")
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e

    if config is None:
        cfg = TrainingConfig(
            data_root=data_root,
            artifacts_root=artifacts_root,
            seed=seed,
            persist_step_outputs=persist_step_outputs,
            log_level=log_level,
            gpp_category=gpp_category,
            universe_root=universe_root,
            share_model_enabled=share_model_enabled,
            share_model_lambda=share_model_lambda,
            share_model_max_iter=share_model_max_iter,
            share_model_val_slate_frac=share_model_val_slate_frac,
            share_model_seed=share_model_seed,
        )
    else:
        cfg = load_training_config(config)
        cfg = apply_cli_overrides(
            cfg,
            data_root=data_root,
            artifacts_root=artifacts_root,
            seed=seed,
            persist_step_outputs=persist_step_outputs,
            log_level=log_level,
            gpp_category=gpp_category,
            universe_root=universe_root,
            share_model_enabled=share_model_enabled,
            share_model_lambda=share_model_lambda,
            share_model_max_iter=share_model_max_iter,
            share_model_val_slate_frac=share_model_val_slate_frac,
            share_model_seed=share_model_seed,
        )
    run_training_pipeline(cfg)


