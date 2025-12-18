from __future__ import annotations

import pandas as pd

from dfs_opt.features.enrich_showdown import enrich_showdown_entries


def test_enrich_showdown_computes_salary_proj_stack_dup_and_archetype() -> None:
    players = pd.DataFrame(
        [
            {"name_norm": "a", "team": "AAA", "salary": 10000, "proj_points": 20.0},
            {"name_norm": "b", "team": "AAA", "salary": 8000, "proj_points": 16.0},
            {"name_norm": "c", "team": "AAA", "salary": 6000, "proj_points": 12.0},
            {"name_norm": "d", "team": "AAA", "salary": 4000, "proj_points": 8.0},
            {"name_norm": "e", "team": "BBB", "salary": 7000, "proj_points": 14.0},
            {"name_norm": "f", "team": "BBB", "salary": 5000, "proj_points": 10.0},
        ]
    )

    # two identical lineups in same contest -> dup_count = 2
    entries = pd.DataFrame(
        [
            {
                "contest_id": "123",
                "slate_id": "slate",
                "lineup_hash": "h1",
                "cpt_name": "A",
                "cpt_name_norm": "a",
                "util1_name": "B",
                "util2_name": "C",
                "util3_name": "D",
                "util4_name": "E",
                "util5_name": "F",
                "util1_name_norm": "b",
                "util2_name_norm": "c",
                "util3_name_norm": "d",
                "util4_name_norm": "e",
                "util5_name_norm": "f",
            },
            {
                "contest_id": "123",
                "slate_id": "slate",
                "lineup_hash": "h1",
                "cpt_name": "A",
                "cpt_name_norm": "a",
                "util1_name": "B",
                "util2_name": "C",
                "util3_name": "D",
                "util4_name": "E",
                "util5_name": "F",
                "util1_name_norm": "b",
                "util2_name_norm": "c",
                "util3_name_norm": "d",
                "util4_name_norm": "e",
                "util5_name_norm": "f",
            },
        ]
    )

    tiers = [(1, "stud_1"), (999, "other")]
    enriched, metrics = enrich_showdown_entries(entries, players, captain_tiers=tiers)

    assert len(enriched) == 2
    assert metrics["unmatched_player_names"] == 0
    assert enriched["dup_count"].tolist() == [2, 2]

    # salary: 1.5*10000 + (8000+6000+4000+7000+5000) = 45000
    assert enriched["salary_used"].tolist() == [45000, 45000]
    assert enriched["salary_left"].tolist() == [5000, 5000]

    # proj: 1.5*20 + (16+12+8+14+10) = 90
    assert enriched["proj_points"].tolist() == [90.0, 90.0]

    # teams: AAA x4, BBB x2
    assert enriched["stack_pattern"].tolist() == ["4-2", "4-2"]
    assert enriched["heavy_team"].tolist() == ["AAA", "AAA"]

    # salary rank: A is rank 1 -> stud_1
    assert enriched["cpt_archetype"].tolist() == ["stud_1", "stud_1"]


