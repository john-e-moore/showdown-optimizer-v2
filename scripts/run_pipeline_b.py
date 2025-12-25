from __future__ import annotations

"""
Run Pipeline B (contest execution) with parameters set in this file.

Usage (recommended):
  /home/john/showdown-optimizer-v2/.venv/bin/python scripts/run_pipeline_b.py
"""

from pathlib import Path

from dfs_opt.config.settings import ContestConfig
from dfs_opt.pipelines.contest import run_contest_pipeline

# -----------------------------------------------------------------------------
# CONFIG: edit these values
# -----------------------------------------------------------------------------

# Inputs
PROJECTION_CSV = Path("data/inputs/NBA_2025-12-25-230pm_DK_SHOWDOWN_SAS-@-OKC.csv")
CORR_MATRIX_CSV = Path("data/inputs/NBA_2025-12-25-230pm_DK_SHOWDOWN_SAS-@-OKC_corr_matrix.csv")
DKENTRIES_CSV = Path("data/inputs/DKEntries.csv")

# Slate identifiers (purely for logging/manifests)
SLATE_ID = "NBA_2025-12-25_SAS-OKC"
SPORT = "nba"

# Share model
THETA_JSON = Path(
    "artifacts/training/2025-12-22T232934Z_6c20ec/share_models/nba-showdown-mme-1k-10k/theta.json"
)

# Outputs
ARTIFACTS_ROOT = Path("artifacts")
PERSIST_STEP_OUTPUTS = True
LOG_LEVEL = "INFO"

# Randomness
SEED = 1337

# Enumeration knobs
SALARY_CAP = 50000
MIN_PROJ_POINTS = 0.0
MAX_PLAYERS = None  # set an int for debugging smaller slates

# Pruning / field sampling
#PRUNE_MASS_THRESHOLD = 0.9995
PRUNE_MASS_THRESHOLD = 0.98
DIRICHLET_ALPHA = None  # set e.g. 50.0 for heavier-tail duplication; None disables

# Grading
NUM_SIMS = 3000
STD_MODE = "dk_std_or_fallback"
STD_SCALE = 1.0
TIE_BREAK = "lineup_id"

# DK API
DK_API_BASE_URL = "https://api.draftkings.com"
DK_API_TIMEOUT_S = 20.0
DK_API_HEADERS = None  # optionally set dict like {"Cookie": "..."} if required

# DKEntries output formatting
DKENTRIES_OUTPUT_FORMAT = "name_id"  # currently supported: "name_id"


def main() -> None:
    cfg = ContestConfig(
        projection_csv=PROJECTION_CSV,
        corr_matrix_csv=CORR_MATRIX_CSV,
        dkentries_csv=DKENTRIES_CSV,
        slate_id=SLATE_ID,
        sport=SPORT,
        artifacts_root=ARTIFACTS_ROOT,
        seed=SEED,
        persist_step_outputs=PERSIST_STEP_OUTPUTS,
        log_level=LOG_LEVEL,
        salary_cap=SALARY_CAP,
        min_proj_points=MIN_PROJ_POINTS,
        max_players=MAX_PLAYERS,
        theta_json=THETA_JSON,
        prune_mass_threshold=PRUNE_MASS_THRESHOLD,
        dirichlet_alpha=DIRICHLET_ALPHA,
        num_sims=NUM_SIMS,
        std_mode=STD_MODE,
        std_scale=STD_SCALE,
        tie_break=TIE_BREAK,
        dk_api_base_url=DK_API_BASE_URL,
        dk_api_timeout_s=DK_API_TIMEOUT_S,
        dk_api_headers=DK_API_HEADERS,
        dkentries_output_format=DKENTRIES_OUTPUT_FORMAT,
    )

    res = run_contest_pipeline(cfg)
    print("Pipeline B finished.")
    print(res)


if __name__ == "__main__":
    main()


