# Contest Simulation (Field + Pruning + Grading Metrics)

This document describes how we will:

- **Simulate a contest field** (i.e., what lineups other entrants play) using the softmax lineup-share model
- **Prune the lineup universe** for efficient simulation
- **Grade lineups** with contest metrics like ROI and top-percent finish rates

This is intentionally separate from `prompts/train-test.md`, which focuses on training/testing the lineup-share model.

---

## 1) Inputs

### 1.1 Slate-level lineup universe + features
Per slate, we assume an enumerated lineup universe \(\mathcal{U}\) and its per-lineup feature table:

- `lineups_enriched.parquet` (one row per lineup \(L\in\mathcal{U}\))
- Includes the features used by the share model (e.g., `own_*`, `avg_corr`, `salary_left_bin`, `pct_proj_gap_to_optimal_bin`, etc.)

Important: these features are **slate-dependent**, not contest-dependent.

### 1.2 Trained model parameters
For the applicable contest bucket (MME size bin, etc.), we assume:

- `theta.json` (coefficients + feature schema)

### 1.3 Contest metadata
For the contest we want to simulate, we need:

- \(N\): number of entries (contest size)
- Entry fee
- Payout structure (rank → payout)

---

## 2) Simulating the field (lineup selection + duplication)

### 2.1 Compute lineup utilities and shares
For each lineup \(L\in\mathcal{U}\), compute the model utility \(u(L)\) from `theta.json`, then compute:

\[
p(L) = \frac{\exp(u(L))}{\sum_{L'\in\mathcal{U}} \exp(u(L'))}
\]

Use stable softmax via \(\log Z = \log\sum\exp(u)\) and \(\log p(L) = u(L) - \log Z\).

### 2.2 Sample a field of size \(N\)
Sample \(N\) entries **with replacement** from \(\mathcal{U}\) according to \(p(L)\). Duplicates emerge naturally. Store the resulting:

- Entry-level field (list of \(N\) lineup ids), or
- A compact representation: `{lineup_id -> dup_count}`

### 2.3 Optional: heavier-tail duplication (Dirichlet–multinomial)
Real contests can show heavier duplication tails than a pure multinomial. Optional extension:

1. Sample \(\tilde{p} \sim \text{Dirichlet}(\alpha p)\)
2. Sample counts \(y \sim \text{Multinomial}(N, \tilde{p})\)

Lower \(\alpha\) ⇒ more “clumping” and higher duplication variance. Calibrate \(\alpha\) by matching historical duplication histograms.

---

## 3) Pruning the lineup universe for simulation

Full universes can be millions of lineups. For simulation runtime, we prune **only for sampling/scoring**, not for training.

### 3.1 Probability-mass pruning (recommended)
After computing \(p(L)\):

- Sort lineups by \(p(L)\) descending
- Keep the smallest prefix whose cumulative mass ≥ a threshold (e.g. 0.999 or 0.9995)
- Renormalize \(p(L)\) over the kept set

### 3.2 Safety rules

- Never prune by a fixed absolute cutoff like `p < 1e-12` unless it’s calibrated per-slate.
- If you are grading a specific user lineup set \(S\), always force-include all \(L\in S\) in the pruned universe (even if low-probability), then renormalize for field sampling.

---

## 4) Grading lineups (ROI + top-% finish rates)

Field simulation gives us *who we’re playing against*. To grade ROI / finish rates we also need *game outcomes*.

### 4.1 Outcome simulation (conceptual)
For each simulation draw \(t=1..T\):

- Simulate correlated player outcomes for the slate (or use historical outcomes in backtests)
- Compute each lineup’s fantasy points
- Rank the simulated contest field by points (ties handled consistently)
- Apply the contest payout table to ranks
- Split payouts across duplicate lineups (each duplicate gets \( \text{payout(rank)} / \text{dup\_count(lineup)} \))

This produces per-lineup distributions of winnings for the lineups we care about (e.g., the pruned
universe or a user-selected set).

### 4.2 Metrics
Let `entry_fee` be the contest entry fee. For a lineup \(L\), across simulations \(t=1..T\):

- **ROI**
  \[
  \text{ROI}(L) = \frac{\mathbb{E}[\text{winnings}(L)] - \text{entry\_fee}}{\text{entry\_fee}}
  \]

- **Top 0.1% finish rate**
  \[
  P\left(\text{rank}(L) \le \lceil 0.001\cdot N\rceil\right)
  \]

- **Top 1% finish rate**
  \[
  P\left(\text{rank}(L) \le \lceil 0.01\cdot N\rceil\right)
  \]

- **Top 5% finish rate**
  \[
  P\left(\text{rank}(L) \le \lceil 0.05\cdot N\rceil\right)
  \]

- **Top 20% finish rate**
  \[
  P\left(\text{rank}(L) \le \lceil 0.20\cdot N\rceil\right)
  \]

Notes:
- Finish rates should be computed on the **duplication-aware payout-splitting contest** (i.e., ranks reflect the full field; winnings reflect split payouts).
- Decide and document tie-breaking (e.g., average rank for ties, or deterministic tie-break using lineup hash).

---

## 5) Key performance note

Because the share-model features are **slate-dependent**, \(p(L)\) is the same for every contest on the slate for a given bucket model. Contest-specific differences in grading come from:

- contest size \(N\)
- payout table / entry fee
- sampling variability


