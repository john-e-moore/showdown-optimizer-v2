from __future__ import annotations

import subprocess
from typing import Optional


def try_get_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        sha = out.decode("utf-8").strip()
        return sha if sha else None
    except Exception:
        return None


