from __future__ import annotations

from dfs_opt.parsing.dk_standings import infer_max_entries_per_user


def test_infer_max_entries_per_user_parses_suffix() -> None:
    assert infer_max_entries_per_user("foo (1/1)") == 1
    assert infer_max_entries_per_user("bar (60/150)") == 150


def test_infer_max_entries_per_user_none_when_missing() -> None:
    assert infer_max_entries_per_user("foo") is None
    assert infer_max_entries_per_user(None) is None


