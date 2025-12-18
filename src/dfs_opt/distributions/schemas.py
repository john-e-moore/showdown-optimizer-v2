from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Histogram(BaseModel):
    bin_edges: Optional[List[float]] = None
    counts: List[int]
    rates: List[float]
    stats: Dict[str, float] = Field(default_factory=dict)


class CategoricalRates(BaseModel):
    counts: Dict[str, int]
    rates: Dict[str, float]


class TargetDistributions(BaseModel):
    schema_version: str = "v1"
    generated_at: datetime
    gpp_category: str
    source_contests: List[str]

    salary_left: Histogram
    proj_gap_to_optimal: Histogram
    stack_pattern: CategoricalRates
    cpt_archetype: CategoricalRates
    dup_count: Histogram

    validation: Dict[str, Any] = Field(default_factory=dict)


