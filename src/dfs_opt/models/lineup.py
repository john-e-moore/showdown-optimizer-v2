from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ParsedLineup:
    cpt_name: str
    util_names: List[str]  # length 5

    @property
    def players_all(self) -> List[str]:
        return [self.cpt_name] + list(self.util_names)


