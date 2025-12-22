**Objective**
Introduce additional features into pipeline a, which enriches historical projection and contest data in preparation for tabular machine learning / regression. Adhere to the guidelines in the `agent/` directory. 

**Details**
1. own_score_logprod
 - Defined as the sum of the log of ownership for each player in the lineup. Treat CPT the same as UTIL here; no 1.5x multiplier, just the sum of log ownership.
2. own_max_log
 - Defined as the log of ownership of the highest owned player in the lineup.
3. own_min_log
 - Defined as the log of ownership of the lowest owned player in the lineup.
4. avg_corr
 - Defined as the sum of the pairwise pearson correlations of players in the lineup divided by 15 (division by 15 to make it more stable cross-slate).
 - Since there are 6 players in a showdown lineup, there will be 15 pairs.
 - In the same folder as the sabersim projections for a given game, there is a .csv file with the correlation matrix for the players in that game. The filename will be similar but with a `_corr_matrix.csv` suffix.
5. pct_contest_lineups
 - Defined as the percentage of lineups in the entire contest that this lineup and any duplicates makes up.
 - For example, if the lineup was duplicated 5 times in a contest with 100 entrants, pct_contest_lineups would be 0.05 (5%).
6. pct_proj_gap_to_optimal
 - Defined as proj_gap_to_optimal / optimal_proj_points
7. pct_proj_gap_to_optimal_bin
 - Bins for the previous feature: 0-1%, 1-2%, 2-4%, 4-7%, 7-15%, 15-30%, 30%+ (use decimals instead of % in the data).
7. salary_left_bin 
 - Bins for salary remaining: 0-200, 200-500, 500-1000, 1000-2000, 2000-4000, 4000-8000, 8000+

