# Showdown-optimizer-v2

This repo implements two composable pipelines:
- **Pipeline A (training)**: raw projections + DK contest exports → enriched entries → target distributions.
- **Pipeline B (contest execution)**: (later) build/weight/sample lineups and fill `DKEntries`.

Canonical specs live in:
- `/home/john/showdown-optimizer-v2/agent/ARCHITECTURE.md`
- `/home/john/showdown-optimizer-v2/agent/PIPELINES.md`
- `/home/john/showdown-optimizer-v2/agent/DATA_CONTRACTS.md`

## Setup

Create a venv and install:

```bash
python -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

## Run Pipeline A (training)

Example (use your local data under `data/`):

```bash
dfs-opt training run --data-root data/historical/raw --artifacts-root artifacts
```

Filter to a single segment bucket:

```bash
dfs-opt training run --data-root data/historical/raw --artifacts-root artifacts --gpp-category nba-showdown-mme-1k-10k
```

## Tests

```bash
pytest -q
```