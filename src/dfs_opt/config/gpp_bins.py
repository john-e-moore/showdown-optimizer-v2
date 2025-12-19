from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Set

import yaml

try:
    # Python 3.10+
    from importlib import resources as importlib_resources
except Exception:  # pragma: no cover
    import importlib_resources  # type: ignore[no-redef]


@dataclass(frozen=True)
class GppBinsRegistry:
    allowed_keys: Set[str]
    source: str


def _parse_bins_yaml(text: str, *, source: str) -> GppBinsRegistry:
    data = yaml.safe_load(text) or {}
    bins = data.get("bins")
    if not isinstance(bins, list):
        raise ValueError(f"Invalid GPP bins YAML ({source}): expected 'bins' list")

    keys: Set[str] = set()
    for idx, b in enumerate(bins):
        if not isinstance(b, dict):
            raise ValueError(f"Invalid GPP bins YAML ({source}): bins[{idx}] must be a mapping")
        k = b.get("key")
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Invalid GPP bins YAML ({source}): bins[{idx}].key must be a string")
        if k in keys:
            raise ValueError(f"Invalid GPP bins YAML ({source}): duplicate key '{k}'")
        keys.add(k)

    return GppBinsRegistry(allowed_keys=keys, source=source)


def load_gpp_bins_registry(*, path: Optional[Path] = None) -> GppBinsRegistry:
    """
    Load the static GPP bins registry.

    - If `path` is provided, load YAML from that filesystem path.
    - Otherwise, load the packaged default from `dfs_opt.resources/gpp_contests.yaml`.
    """
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
        return _parse_bins_yaml(text, source=str(path))

    resource = importlib_resources.files("dfs_opt.resources").joinpath("gpp_contests.yaml")
    text = resource.read_text(encoding="utf-8")
    return _parse_bins_yaml(text, source="package:dfs_opt.resources/gpp_contests.yaml")


def validate_gpp_category(category: str, *, allowed: Iterable[str], context: str) -> None:
    allowed_set = set(allowed)
    if category not in allowed_set:
        raise ValueError(
            f"Unknown gpp_category '{category}' in {context}. "
            "This project uses a static registry; add the bin to "
            "'dfs_opt/resources/gpp_contests.yaml' (or fix your inputs)."
        )


