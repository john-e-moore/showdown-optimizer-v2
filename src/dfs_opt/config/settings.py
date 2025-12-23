from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class SegmentSizeBin:
    """Contest size bin (inclusive lower bound, exclusive upper bound unless upper is None)."""

    label: str
    min_size: int
    max_size_exclusive: Optional[int]

    def contains(self, size: int) -> bool:
        if size < self.min_size:
            return False
        if self.max_size_exclusive is None:
            return True
        return size < self.max_size_exclusive


@dataclass(frozen=True)
class SegmentDefinitions:
    """
    Defaults match the plan: size bins configurable and segment name like
    '<sport>-showdown-<mme|single-entry>-<size_bin_label>'.
    """

    size_bins: List[SegmentSizeBin] = field(
        default_factory=lambda: [
            SegmentSizeBin(label="0-1k", min_size=0, max_size_exclusive=1000),
            SegmentSizeBin(label="1k-10k", min_size=1000, max_size_exclusive=10000),
            SegmentSizeBin(label="10k+", min_size=10000, max_size_exclusive=None),
        ]
    )

    # captain salary-rank tiers: (max_rank_inclusive, label)
    captain_tiers: List[Tuple[int, str]] = field(
        default_factory=lambda: [
            (2, "stud_1_2"),
            (5, "stud_3_5"),
            (10, "mid_6_10"),
            (9999, "value_11_plus"),
        ]
    )


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    artifacts_root: Path = Path("artifacts")
    seed: int = 1337
    persist_step_outputs: bool = False
    log_level: str = "INFO"

    # optional filter: run only a single segment bucket
    gpp_category: Optional[str] = None

    # segmentation + feature config
    segment_definitions: SegmentDefinitions = field(default_factory=SegmentDefinitions)

    # Path to precomputed per-slate lineup universes (produced by Pipeline B / offline backfill).
    # Expected layout:
    #   <universe_root>/dk-results/showdown/<sport>/<slate_id>/lineups_enriched.parquet
    universe_root: Path = Path("data/historical/enriched")

    # --- softmax lineup share model (optional step) ---
    share_model_enabled: bool = False
    share_model_lambda: float = 1e-3
    share_model_max_iter: int = 200
    share_model_val_slate_frac: float = 0.2
    share_model_seed: int = 1337


@dataclass(frozen=True)
class ContestConfig:
    """
    Minimal Pipeline B config for lineup universe enumeration (lineup-gen slice).

    This will be extended later as Pipeline B grows (candidate pool, raking, sims, etc).
    """

    projection_csv: Path
    # Required for lineup-universe feature enrichment (avg_corr).
    corr_matrix_csv: Path
    slate_id: str
    # DKEntries CSV to fill (Pipeline B full run).
    dkentries_csv: Path | None = None
    sport: str = "nba"

    artifacts_root: Path = Path("artifacts")
    seed: int = 1337
    persist_step_outputs: bool = False
    log_level: str = "INFO"

    # enumeration knobs
    salary_cap: int = 50000
    min_proj_points: float = 0.0
    max_players: Optional[int] = None

    # feature knobs (match TrainingConfig defaults)
    captain_tiers: List[Tuple[int, str]] = field(default_factory=lambda: SegmentDefinitions().captain_tiers)
    own_log_eps: float = 1e-6

    # --- share model / theta ---
    # For now, allow passing a direct theta.json path (may be overridden per contest later).
    theta_json: Path | None = None
    share_models_root: Path | None = None
    gpp_bins_yaml: Path | None = None

    # --- DK API ---
    dk_api_base_url: str = "https://api.draftkings.com"
    dk_api_timeout_s: float = 20.0
    dk_api_headers: Optional[dict[str, str]] = None

    # --- pruning / field sampling ---
    prune_mass_threshold: float = 0.9995
    dirichlet_alpha: Optional[float] = None

    # --- grading ---
    num_sims: int = 2000
    # std_mode: \"dk_std\" uses projection std column when available, else fallback rule.
    std_mode: str = "dk_std_or_fallback"
    std_scale: float = 1.0
    tie_break: str = "lineup_id"

    # --- dkentries output ---
    # \"name_id\" writes \"Name (ID)\" using the projections' DFS ID.
    dkentries_output_format: str = "name_id"


