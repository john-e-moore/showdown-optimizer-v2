# Softmax Lineup Share Model (NBA Showdown)

This document specifies the **v1 softmax / discrete-choice model** used to predict each lineup’s **share of the field** (and therefore duplication frequency) for a given **contest bucket** (e.g., NBA Showdown, DK, MME, 10k+ entrants).

The implementation goal is:

- Given a slate’s **enumerated universe of valid lineups** \(\mathcal{U}\) and per-lineup features \(x(L)\),
- Fit a model that produces a **probability distribution over lineups**:
  \[
  p(L) = P(\text{a random contest entry is lineup } L)
  \]
- Use \(p(L)\) to (a) **explain observed duplication patterns** in historical contests, and (b) **simulate a contest field** (see `prompts/contest-sim.md`).

---

## 1) Definitions

### 1.1 Contest bucket
A contest bucket is defined by:
- `sport` (NBA)
- `format` (DK Showdown)
- `contest_type` (MME, SE, etc.)
- `entry_count_bin` (0–1k, 1k–10k, 10k+)

Model parameters are learned **per bucket** (or per bucket + season segment, if desired).

### 1.2 Lineup universe
For a slate, define a lineup \(L\) as:
- 1 CPT + 5 UTILs (no duplicates)
- Salary cap = 50,000
- Any additional hard validity constraints (site rules)

Let \(\mathcal{U}\) be the set of **all valid lineups** for that slate.
For NBA showdown, \(|\mathcal{U}|\) is typically ~0.4M–3.5M and can be enumerated.

### 1.3 Observed contest data
For each historical contest (same bucket) on a slate, we observe **counts** per lineup:
- \(y(L)\) = number of entrants who played lineup \(L\)
- \(N\) = total entries in the contest

These counts are derived from the DraftKings contest export:
- Parse the `Lineup` string into slots
- Compute `dup_count` per unique lineup, i.e. \(y(L)\)

### 1.4 Data layout (critical for training performance)
We store **slate-dependent** and **contest-dependent** data separately:

- **Per slate**: a parquet of the **full lineup universe** with features (one row per lineup in \(\mathcal{U}\)).  
  Canonical artifact: `lineups_enriched.parquet` (see `agent/DATA_CONTRACTS.md`).
- **Per contest**: a compact table of **observed lineups only** (unique `lineup_hash` + `dup_count`, plus contest metadata like `contest_id`, `slate_id`, and `contest_size`).

Training loops over contests and:
- Looks up utilities for the observed lineups on that slate
- Computes \(\log Z\) from the slate’s full lineup utilities

So we **never materialize** a literal `contest × universe` dataset. Since our features are **slate-dependent** (not contest-dependent), this is fast.

---

## 2) Feature set (v1)

All features are computed **per lineup** \(L\). Below are the v1 features.

### 2.1 Ownership features
Ownership values are **slot-specific** probabilities:
- \(o^{CPT}_p\) = projected CPT ownership for player \(p\) (probability in \([0,1]\))
- \(o^{FLEX}_p\) = projected FLEX ownership for player \(p\)

> If any ownership is 0, clamp with epsilon, e.g. `eps=1e-6`.

#### (1) `own_score_logprod`
Log-product ownership score:
\[
\text{own\_score\_logprod}(L)=\log(o^{CPT}_{cpt}) + \sum_{u\in utils}\log(o^{FLEX}_u)
\]

#### (2) `own_max_log`
\[
\text{own\_max\_log}(L)=\max\left(\log(o^{CPT}_{cpt}),\;\{\log(o^{FLEX}_u)\}_{u\in utils}\right)
\]

#### (3) `own_min_log`
\[
\text{own\_min\_log}(L)=\min\left(\log(o^{CPT}_{cpt}),\;\{\log(o^{FLEX}_u)\}_{u\in utils}\right)
\]

### 2.2 Correlation feature
Given a player–player Pearson correlation matrix \(\rho_{ij}\) (FLEX player identities, **no CPT-specific rows**):

#### (4) `avg_corr`
Compute pairwise average correlation across the 6 players:
\[
\text{avg\_corr}(L) = \frac{1}{\binom{6}{2}}\sum_{i<j\in L}\rho_{ij} = \frac{1}{15}\sum_{i<j\in L}\rho_{ij}
\]

> (Optional later) Add extra penalty for negative correlations, but v1 uses only `avg_corr`.

### 2.3 Optimality / projection feature
Let:
- \(\text{proj}(L)\) be the lineup’s projected points (CPT = 1.5× FLEX projection + sum FLEX projections).
- \(\text{proj}^*\) be the **optimal projected lineup** under the cap for that slate (computed by brute force enumeration or solver).

#### (5a) `pct_proj_gap_to_optimal` (computed, not used directly in v1 model)
\[
\text{pct\_proj\_gap\_to\_optimal}(L) = \frac{\text{proj}^* - \text{proj}(L)}{\text{proj}^*}
\]

We compute this as a continuous diagnostic feature, but the v1 model uses the **binned** version below.

#### (5b) `pct_proj_gap_to_optimal_bin` (used in v1 model)
Categorical bin of `pct_proj_gap_to_optimal` using these exact bins/labels:

- `0_0.01` for \([0.00, 0.01)\)
- `0.01_0.02` for \([0.01, 0.02)\)
- `0.02_0.04` for \([0.02, 0.04)\)
- `0.04_0.07` for \([0.04, 0.07)\)
- `0.07_0.15` for \([0.07, 0.15)\)
- `0.15_0.30` for \([0.15, 0.30)\)
- `0.30_plus` for \([0.30, \infty)\)

### 2.4 CPT archetype (categorical)
Rank players by **FLEX salary** on the slate (descending). Define CPT archetype using the CPT player’s salary rank:

#### (6) `cpt_archetype`
One of:
- `stud_1_2` (top 2 salaries)
- `stud_3_5` (ranks 3–5)
- `mid_6_10` (ranks 6–10)
- `value_11+` (rank 11 or lower)

Encoded as one-hot with a reference category (or full one-hot if using no intercept).

### 2.5 Stack pattern (categorical)
For NBA showdown with exactly two teams, count players from each team among the 6 roster slots.

#### (7) `stack_pattern`
One of:
- `5-1`
- `4-2`
- `3-3`

(Optionally include `6-0` if it appears meaningfully in your historical bucket.)

Encoded as one-hot.

### 2.6 Salary left
Salary cap = 50,000.

#### (8) `salary_left`
\[
\text{salary\_left}(L) = 50000 - \text{salary\_used}(L)
\]

Recommended: include a binned one-hot version `salary_left_bin` in addition to raw `salary_left` (even in v1), e.g.:
- 0–200
- 200–500
- 500–1000
- 1000–2000
- 2000–4000
- 4000–8000
- 8000+

---

## 3) Utility model and softmax probability

### 3.1 Utility function (v1)
Define a scalar utility \(u(L)\):

\[
u(L) = \theta_0
+ \theta_1 \cdot \text{own\_score\_logprod}(L)
+ \theta_2 \cdot \text{own\_max\_log}(L)
+ \theta_3 \cdot \text{own\_min\_log}(L)
+ \theta_4 \cdot \text{avg\_corr}(L)
+ \text{onehot}(\text{cpt\_archetype}(L))
+ \text{onehot}(\text{stack\_pattern}(L))
+ \text{onehot}(\text{salary\_left\_bin}(L))
+ \text{onehot}(\text{pct\_proj\_gap\_to\_optimal\_bin}(L))
\]

> Notes:
- If using an intercept \(\theta_0\), drop one reference category per one-hot group.
- Standardize continuous features per slate or bucket if helpful (optional).

### 3.2 Softmax probability over the slate universe
For a given slate universe \(\mathcal{U}\):

\[
p(L) = \frac{\exp(u(L))}{\sum_{L'\in\mathcal{U}} \exp(u(L'))}
\]

This produces:
- A valid probability for every lineup \(L\in\mathcal{U}\)
- \(\sum_{L\in\mathcal{U}} p(L) = 1\)

Interpretation:
- \(p(L)\) is the predicted **share of entries** that will be lineup \(L\) in contests from this bucket.

---

## 4) Training objective (maximum likelihood on counts)

### 4.1 Likelihood per contest (multinomial)
Given an observed contest with total entries \(N\) and lineup counts \(y(L)\), assume:

\[
(y(L))_{L\in\mathcal{U}} \sim \text{Multinomial}(N,\; (p(L))_{L\in\mathcal{U}})
\]

### 4.2 Log-likelihood
Ignoring the constant multinomial coefficient, the log-likelihood for one contest is:

\[
\log \mathcal{L}(\theta) = \sum_{L\in\mathcal{U}} y(L)\cdot u(L) - N\cdot \log\left(\sum_{L'\in\mathcal{U}} \exp(u(L'))\right)
\]

Train by **maximizing** \(\log \mathcal{L}\) across all contests in the bucket, i.e. sum the above over contests.

### 4.3 Training loop (no contest × universe materialization)
For performance we structure computation around the fact that \(u(L)\) depends only on **slate** features:

- Precompute (or load) per-slate feature matrices \(X_s\) for all \(L\in\mathcal{U}_s\) from `lineups_enriched.parquet`.
- For each contest \(c\) on slate \(s\), keep a compact table of observed lineups:
  - `lineup_hash`
  - `dup_count` (= \(y(L)\))
  - `contest_size` (= \(N\))

Then each optimization step does:
- For each slate \(s\): compute utilities \(u_s\) for all lineups in \(\mathcal{U}_s\), and compute \(\log Z_s = \log\sum_{L\in\mathcal{U}_s}\exp(u(L))\).
- For each contest \(c\) on slate \(s\): look up the observed lineups’ utilities and accumulate:
  \[
  \sum_{L\in \text{observed}(c)} y(L)\cdot u(L)\;-\;N\cdot \log Z_s
  \]

This is equivalent to the multinomial objective, but avoids ever constructing a giant sparse tensor.

### 4.4 Regularization
Add L2 regularization to prevent ownership features from dominating:

\[
\text{Objective} = -\sum_{contests} \log \mathcal{L}(\theta) + \lambda \lVert\theta\rVert^2
\]

Pick \(\lambda\) via validation.

---

## 6) Practical implementation notes

### 6.1 Efficiency: stable softmax
Compute `logsumexp` to avoid overflow:
- \(\log Z = \log \sum \exp(u)\) via `scipy.special.logsumexp`
- \(\log p(L) = u(L) - \log Z\)

### 6.3 Artifact requirements
For each slate + bucket model fit, write artifacts:
- `theta.json` (coefficients, feature schema)
- `fit_metrics.json` (train/val NLL, regularization, convergence)
- `predicted_distributions/`:
  - salary_left histogram
  - pct_gap histogram
  - stack pattern distribution
  - CPT archetype distribution
  - duplication histogram (predicted vs actual) on validation contests

---

## 7) Deliverables for the agent (implementation checklist)

1. **Feature computation** on full lineup universe:
   - Compute features 1–8 above for every lineup.
2. **Training**:
   - Input: per-slate `lineups_enriched.parquet` + per-contest compact observed-lineup counts.
   - Optimize negative log-likelihood + L2.
3. **Inference**:
   - For a new slate, compute \(p(L)\) over \(\mathcal{U}\).
4. **Validation**:
   - Compare predicted vs actual histograms for key distributions.
   - Compare duplication histograms and top duplicated lineup identities.

> Field simulation + universe pruning + ROI/top-% grading live in `prompts/contest-sim.md`.
