from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class SegmentSizeBin:
    """Contest size bin (inclusive lower bound, exclusive upper bound unless upper is None)."""

    label: str
    min_size: int
    max_size_exclusive: Optional[int]

    def contains(self, size: int) -> bool:
        if size < self.min_size:
            return False
        if self.max_size_exclusive is None:
            return True
        return size < self.max_size_exclusive


@dataclass(frozen=True)
class SegmentDefinitions:
    """
    Defaults match the plan: size bins configurable and segment name like
    '<sport>-showdown-<mme|single-entry>-<size_bin_label>'.
    """

    size_bins: List[SegmentSizeBin] = field(
        default_factory=lambda: [
            SegmentSizeBin(label="0-1k", min_size=0, max_size_exclusive=1000),
            SegmentSizeBin(label="1k-10k", min_size=1000, max_size_exclusive=10000),
            SegmentSizeBin(label="10k+", min_size=10000, max_size_exclusive=None),
        ]
    )

    # captain salary-rank tiers: (max_rank_inclusive, label)
    captain_tiers: List[Tuple[int, str]] = field(
        default_factory=lambda: [
            (2, "stud_1_2"),
            (5, "stud_3_5"),
            (10, "mid_6_10"),
            (9999, "value_11_plus"),
        ]
    )


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    artifacts_root: Path = Path("artifacts")
    seed: int = 1337
    persist_step_outputs: bool = False

    # optional filter: run only a single segment bucket
    gpp_category: Optional[str] = None

    # segmentation + feature config
    segment_definitions: SegmentDefinitions = field(default_factory=SegmentDefinitions)


