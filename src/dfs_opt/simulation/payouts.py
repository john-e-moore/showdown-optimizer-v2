from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class PayoutResult:
    winnings: float
    rank_start: int
    rank_end: int


def payout_for_block(payout_table: List[float], *, rank_start: int, block_size: int) -> float:
    """
    DK-style payout splitting within a duplicate lineup block:
      payout_per_entry = sum(payout(rank_start..rank_start+block_size-1)) / block_size

    `payout_table` is 1-indexed conceptually, stored as list where idx=0 is rank 1.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if rank_start <= 0:
        raise ValueError(f"rank_start must be >= 1, got {rank_start}")
    lo = rank_start - 1
    hi = min(len(payout_table), lo + int(block_size))
    if lo >= len(payout_table):
        return 0.0
    s = float(np.sum(np.asarray(payout_table[lo:hi], dtype=np.float64)))
    return s / float(block_size)


