from __future__ import annotations

from pathlib import Path

from dfs_opt.io.dkentries import read_dkentries


def test_read_dkentries_skips_player_list_section_and_keeps_all_entry_rows(tmp_path: Path) -> None:
    """
    DKEntries exports often contain:
    - an entries upload template section (consistent column width)
    - a trailing "player list" appendix with a different width (extra columns)

    We must keep all entry rows and ignore the mismatched-width appendix so Pipeline B
    counts the correct number of entries to fill.
    """

    header = "Entry ID,Contest Name,Contest ID,Entry Fee,CPT,UTIL,UTIL,UTIL,UTIL,UTIL,,Instructions\n"
    contest_id = "186428978"

    # 28 entry rows (match header width)
    entry_rows = []
    for i in range(28):
        entry_id = str(5000000000 + i)
        entry_rows.append(
            f'{entry_id},Contest,{contest_id},$10,Player A (1),Player B (2),Player C (3),Player D (4),Player E (5),Player F (6),,""\n'
        )

    # Player list appendix: mismatched width (extra columns starting at/after the blank column)
    player_list_header = ",,,,,,,,,,,Position,Name + ID,Name,ID,Roster Position,Salary\n"
    player_list_row = ",,,,,,,,,,,PG,Player A (1),Player A,1,CPT,15000\n"

    p = tmp_path / "DKEntries.csv"
    p.write_text(header + "".join(entry_rows) + player_list_header + player_list_row, encoding="utf-8")

    dk = read_dkentries(p)
    assert len(dk.entries) == 28
    assert set(dk.entries["contest_id"].astype(str).tolist()) == {contest_id}


def test_read_dkentries_keeps_all_entry_rows_when_appendix_begins_after_200_lines(tmp_path: Path) -> None:
    """
    Regression: large MME exports can have the player-list appendix start after many entry rows.
    The ParserError fallback must not depend on a short prescan window.
    """

    header = "Entry ID,Contest Name,Contest ID,Entry Fee,CPT,UTIL,UTIL,UTIL,UTIL,UTIL,,Instructions\n"
    contest_id = "186428978"

    # 250 entry rows (appendix begins after >200 lines)
    entry_rows = []
    for i in range(250):
        entry_id = str(5000000000 + i)
        entry_rows.append(
            f'{entry_id},Contest,{contest_id},$10,Player A (1),Player B (2),Player C (3),Player D (4),Player E (5),Player F (6),,""\n'
        )

    # Player list appendix: mismatched width (extra columns)
    player_list_header = ",,,,,,,,,,,Position,Name + ID,Name,ID,Roster Position,Salary\n"
    player_list_row = ",,,,,,,,,,,PG,Player A (1),Player A,1,CPT,15000\n"

    p = tmp_path / "DKEntries.csv"
    p.write_text(header + "".join(entry_rows) + player_list_header + player_list_row, encoding="utf-8")

    dk = read_dkentries(p)
    assert len(dk.entries) == 250
    assert set(dk.entries["contest_id"].astype(str).tolist()) == {contest_id}


def test_read_dkentries_keeps_entry_rows_when_extra_columns_are_appended_to_entry_lines(tmp_path: Path) -> None:
    """
    Real-world DK format drift: the "player list" columns can be appended onto the SAME CSV lines
    as the entries, creating wide rows that must be truncated to the entries header width.
    """

    header = "Entry ID,Contest Name,Contest ID,Entry Fee,CPT,UTIL,UTIL,UTIL,UTIL,UTIL,,Instructions\n"
    contest_id = "186465227"

    # Normal-width entries
    entry_rows_normal = []
    for i in range(10):
        entry_id = str(6000000000 + i)
        entry_rows_normal.append(
            f'{entry_id},Contest,{contest_id},$10,Player A (1),Player B (2),Player C (3),Player D (4),Player E (5),Player F (6),,""\n'
        )

    # Entry rows with extra appended columns (should still count as entries)
    entry_rows_wide = []
    for i in range(5):
        entry_id = str(6000000100 + i)
        entry_rows_wide.append(
            f'{entry_id},Contest,{contest_id},$10,Player A (1),Player B (2),Player C (3),Player D (4),Player E (5),Player F (6),,"",Position,Name + ID,Salary\n'
        )

    # Player-list-only rows (blank Entry ID) with extra columns (must be ignored by entry_mask)
    player_only_rows = [
        ',,,,,,,,,,,PG,Player A (1),15000\n',
        ',,,,,,,,,,,C,Player B (2),14000\n',
    ]

    p = tmp_path / "DKEntries.csv"
    p.write_text(
        header + "".join(entry_rows_normal) + "".join(entry_rows_wide) + "".join(player_only_rows),
        encoding="utf-8",
    )

    dk = read_dkentries(p)
    assert len(dk.entries) == 15
    assert set(dk.entries["contest_id"].astype(str).tolist()) == {contest_id}


