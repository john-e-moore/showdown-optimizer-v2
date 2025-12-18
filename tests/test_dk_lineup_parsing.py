from __future__ import annotations

from dfs_opt.parsing.dk_lineup import parse_dk_showdown_lineup


def test_parse_dk_showdown_lineup_success() -> None:
    s = "CPT A UTIL B UTIL C UTIL D UTIL E UTIL F"
    parsed = parse_dk_showdown_lineup(s)
    assert parsed is not None
    assert parsed.cpt_name == "A"
    assert parsed.util_names == ["B", "C", "D", "E", "F"]


def test_parse_dk_showdown_lineup_rejects_non_cpt() -> None:
    assert parse_dk_showdown_lineup("UTIL A UTIL B") is None


def test_parse_dk_showdown_lineup_rejects_wrong_slot_count() -> None:
    s = "CPT A UTIL B UTIL C UTIL D UTIL E"
    assert parse_dk_showdown_lineup(s) is None


