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
        )
    run_training_pipeline(cfg)


