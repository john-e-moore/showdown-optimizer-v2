**Objective**
Draft a plan to implement "Pipeline A" specified in `agent/PIPELINES.md`.

**Details**
- The purpose of pipeline A is to enrich lineup data for various buckets of DFS slates and then fit distributions for various features: salary remaining, optimal lineup projection and gap to optimal, stack pattern + heavy team, captain archetype, and duplication count. 
- We will bucket this analysis by contest sport (e.g. "NBA", "NFL"), type (e.g. "Classic", "Showdown Captain Mode"), size (e.g. 0-1k, 1k-10k, 10k+), and type (e.g. "MME", "Single Entry"). The pipeline should by default automatically bin contests into different types and compute and output distributions for each one. There should be optional flags letting the user run the pipeline for a specific bin, e.g. "--gpp-category nba-showdown-mme-1k-10k" or similar. We should store these category definitions in config; I may want to change their definitions at some point.
- Input data is located in `data/raw/dk-results/showdown/{sport}/{slate}/`. In that directory is the Sabersim projections .csv as well as a `contests/` folder, which contains one or multiple zip files with draftkings contest results.
- Pipeline A first ingests, parses, and joins these data and then enriches it with the new features. An example of the enriched data is located at `data/historical/entriched/enriched_entries_example.csv`.
- A large language model gave me a script to generate that CSV; you can find it at `scripts/build_showdown_training_dataset.py`. You can ignore the file structure and such in that sheet, but use it as a reference for how to generate features. Our code should not be in one script like this, but separated out logically according to code specs in the `agent/` folder.
- Ignore the `inputs/` directory and its contents for now; it will be used for pipeline B.

