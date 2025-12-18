from __future__ import annotations

import math
from typing import Optional

from dfs_opt.models.lineup import ParsedLineup


def parse_dk_showdown_lineup(lineup_str: object) -> Optional[ParsedLineup]:
    """
    Parse DK showdown lineup strings like:
      'CPT <name> UTIL <name> UTIL <name> UTIL <name> UTIL <name> UTIL <name>'

    Returns None if parsing fails.
    """
    if lineup_str is None:
        return None
    if isinstance(lineup_str, float) and math.isnan(lineup_str):
        return None

    lineup = str(lineup_str).strip()
    if not lineup.startswith("CPT "):
        return None

    util_sep = " UTIL "
    first_util_pos = lineup.find(util_sep)
    if first_util_pos == -1:
        return None

    cpt = lineup[4:first_util_pos].strip()
    rest = lineup[first_util_pos + len(util_sep) :]
    utils = [x.strip() for x in rest.split(util_sep)]
    if len(utils) != 5:
        return None

    return ParsedLineup(cpt_name=cpt, util_names=utils)


