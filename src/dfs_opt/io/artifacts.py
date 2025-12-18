from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

from dfs_opt.models.manifests import ManifestIO, StepManifest
from dfs_opt.utils.hashing import data_fingerprint, schema_fingerprint, sha256_file, sha256_hex


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    # timestamp + short random suffix for uniqueness
    ts = utc_now().strftime("%Y-%m-%dT%H%M%SZ")
    suffix = sha256_hex(f"{time.time_ns()}".encode("utf-8"))[:6]
    return f"{ts}_{suffix}"


class ArtifactWriter:
    def __init__(self, *, artifacts_root: Path, pipeline: str, run_id: str) -> None:
        self.artifacts_root = artifacts_root
        self.pipeline = pipeline
        self.run_id = run_id

        self.run_dir = artifacts_root / pipeline / run_id
        self.steps_dir = self.run_dir / "steps"
        self.logs_dir = self.run_dir / "logs"

    def init_run_dirs(self) -> None:
        self.steps_dir.mkdir(parents=True, exist_ok=False)
        self.logs_dir.mkdir(parents=True, exist_ok=False)

    def step_dir(self, step_idx: int, step_name: str) -> Path:
        return self.steps_dir / f"{step_idx:02d}_{step_name}"

    def write_step(
        self,
        *,
        step_idx: int,
        step_name: str,
        df_in: Optional[pd.DataFrame],
        df_out: pd.DataFrame,
        inputs: Sequence[Tuple[str, str, str]],
        outputs: Sequence[Tuple[str, str, str]],
        metrics: Dict[str, Any],
        warnings: Sequence[Dict[str, Any]] = (),
        persist_parquet: bool = False,
    ) -> StepManifest:
        """
        Write required sidecar artifacts for a step folder:
        - preview.csv (<=200 rows)
        - schema.json
        - step_manifest.json
        - optional: outputs.parquet

        inputs/outputs: tuples of (path, checksum_sha256, logical_name)
        """
        step_path = self.step_dir(step_idx, step_name)
        step_path.mkdir(parents=True, exist_ok=False)

        started_at = utc_now()
        t0 = time.perf_counter()

        preview = df_out.head(200)
        preview_path = step_path / "preview.csv"
        preview.to_csv(preview_path, index=False)

        schema_info = _schema_json(df_out)
        schema_path = step_path / "schema.json"
        schema_path.write_text(json.dumps(schema_info, indent=2, sort_keys=True), encoding="utf-8")

        parquet_path = None
        if persist_parquet:
            parquet_path = step_path / "outputs.parquet"
            df_out.to_parquet(parquet_path, index=False)

        finished_at = utc_now()
        duration_s = time.perf_counter() - t0

        cols_and_dtypes: Iterable[tuple[str, str]] = ((c, str(df_out[c].dtype)) for c in df_out.columns)
        sfp = schema_fingerprint(cols_and_dtypes)
        dfp = data_fingerprint(preview.to_dict(orient="records"))

        manifest = StepManifest(
            run_id=self.run_id,
            pipeline=self.pipeline,  # type: ignore[arg-type]
            step_name=step_name,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration_s,
            inputs=[ManifestIO(path=p, checksum_sha256=chk, logical_name=ln) for (p, chk, ln) in inputs],
            outputs=[ManifestIO(path=p, checksum_sha256=chk, logical_name=ln) for (p, chk, ln) in outputs]
            + [
                ManifestIO(
                    path=str(preview_path),
                    checksum_sha256=sha256_file(str(preview_path)),
                    logical_name="preview.csv",
                ),
                ManifestIO(
                    path=str(schema_path),
                    checksum_sha256=sha256_file(str(schema_path)),
                    logical_name="schema.json",
                ),
            ]
            + (
                [
                    ManifestIO(
                        path=str(parquet_path),
                        checksum_sha256=sha256_file(str(parquet_path)),
                        logical_name="outputs.parquet",
                    )
                ]
                if parquet_path
                else []
            ),
            row_count_in=0 if df_in is None else int(len(df_in)),
            row_count_out=int(len(df_out)),
            schema_fingerprint=sfp,
            data_fingerprint=dfp,
            metrics=metrics,
            warnings=list(warnings),
            errors=[],
        )

        manifest_path = step_path / "step_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return manifest


def _schema_json(df: pd.DataFrame) -> Dict[str, Any]:
    cols = []
    for c in df.columns:
        s = df[c]
        nulls = int(s.isna().sum())
        cols.append(
            {
                "name": c,
                "dtype": str(s.dtype),
                "null_count": nulls,
                "null_rate": float(nulls / max(1, len(df))),
            }
        )
    return {"num_rows": int(len(df)), "columns": cols}


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


