from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional


class _RunContextFilter(logging.Filter):
    def __init__(self, *, run_id: str, pipeline: str) -> None:
        super().__init__()
        self._run_id = run_id
        self._pipeline = pipeline

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 (filter)
        # Ensure formatter fields always exist.
        if not hasattr(record, "run_id"):
            record.run_id = self._run_id
        if not hasattr(record, "pipeline"):
            record.pipeline = self._pipeline
        if not hasattr(record, "step"):
            record.step = "-"
        return True


def _find_file_handler(logger: logging.Logger, *, log_path: Path) -> Optional[logging.FileHandler]:
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                existing = Path(h.baseFilename)
            except Exception:
                continue
            if existing == log_path:
                return h
    return None


def _replace_run_context_filter(handler: logging.Handler, ctx_filter: logging.Filter) -> None:
    handler.filters = [f for f in handler.filters if not isinstance(f, _RunContextFilter)]
    handler.addFilter(ctx_filter)


def configure_run_logger(
    *,
    logs_dir: Path,
    run_id: str,
    pipeline: str,
    level: str = "INFO",
) -> logging.Logger:
    """
    Configure a run-scoped logger that writes to:
      - <logs_dir>/run.log (plain text)
      - stderr (for CLI visibility)

    Idempotent: calling this multiple times in-process will not duplicate handlers
    for the same run.log path.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run.log"

    logger = logging.getLogger("dfs_opt")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    fmt = "%(asctime)s %(levelname)s [%(pipeline)s %(run_id)s %(step)s] %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%SZ"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    formatter.converter = time.gmtime  # ensure UTC timestamps

    ctx_filter = _RunContextFilter(run_id=run_id, pipeline=pipeline)

    # File handler (dedupe by absolute path)
    log_path_abs = log_path.resolve()
    fh = _find_file_handler(logger, log_path=log_path_abs)
    if fh is None:
        fh = logging.FileHandler(log_path_abs, encoding="utf-8")
        fh.setLevel(logger.level)
        fh.setFormatter(formatter)
        _replace_run_context_filter(fh, ctx_filter)
        logger.addHandler(fh)
    else:
        # Keep existing handler but ensure it has the right formatter/filter.
        fh.setLevel(logger.level)
        fh.setFormatter(formatter)
        _replace_run_context_filter(fh, ctx_filter)

    # Stderr handler (dedupe by stream identity)
    stderr_handler: Optional[logging.StreamHandler] = None
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr:
            stderr_handler = h
            break

    if stderr_handler is None:
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setLevel(logger.level)
        sh.setFormatter(formatter)
        _replace_run_context_filter(sh, ctx_filter)
        logger.addHandler(sh)
    else:
        stderr_handler.setLevel(logger.level)
        stderr_handler.setFormatter(formatter)
        _replace_run_context_filter(stderr_handler, ctx_filter)

    return logger


def get_step_logger(base_logger: logging.Logger, *, step: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(base_logger, extra={"step": step})

