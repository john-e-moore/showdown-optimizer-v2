from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from dfs_opt.config.settings import SegmentDefinitions, SegmentSizeBin, TrainingConfig


def load_training_config(path: Path) -> TrainingConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return training_config_from_dict(data)


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
        gpp_category=data.get("gpp_category"),
        segment_definitions=seg_defs,
    )


def apply_cli_overrides(
    cfg: TrainingConfig,
    *,
    data_root: Optional[Path] = None,
    artifacts_root: Optional[Path] = None,
    seed: Optional[int] = None,
    persist_step_outputs: Optional[bool] = None,
    gpp_category: Optional[str] = None,
) -> TrainingConfig:
    return replace(
        cfg,
        data_root=data_root if data_root is not None else cfg.data_root,
        artifacts_root=artifacts_root if artifacts_root is not None else cfg.artifacts_root,
        seed=seed if seed is not None else cfg.seed,
        persist_step_outputs=persist_step_outputs
        if persist_step_outputs is not None
        else cfg.persist_step_outputs,
        gpp_category=gpp_category if gpp_category is not None else cfg.gpp_category,
    )


