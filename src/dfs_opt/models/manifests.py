from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ManifestIO(BaseModel):
    path: str
    checksum_sha256: str
    logical_name: str


class StructuredWarning(BaseModel):
    code: str
    message: str
    sample_rows: Optional[List[Dict[str, Any]]] = None


class StepManifest(BaseModel):
    run_id: str
    pipeline: Literal["training", "contest"]
    step_name: str

    started_at: datetime
    finished_at: datetime
    duration_s: float

    inputs: List[ManifestIO] = Field(default_factory=list)
    outputs: List[ManifestIO] = Field(default_factory=list)

    row_count_in: int
    row_count_out: int

    schema_fingerprint: str
    data_fingerprint: str

    metrics: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[StructuredWarning] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class RunManifest(BaseModel):
    run_id: str
    pipeline: Literal["training", "contest"]
    started_at: datetime
    finished_at: datetime
    git_sha: Optional[str] = None

    config: Dict[str, Any]
    inputs: List[ManifestIO] = Field(default_factory=list)
    outputs: List[ManifestIO] = Field(default_factory=list)

    row_counts_by_step: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    warnings: List[StructuredWarning] = Field(default_factory=list)


