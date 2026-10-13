"""Logging setup shared by all pipeline stages."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if any(getattr(h, "_huskyquiet", False) for h in root.handlers):
        return
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        handler._huskyquiet = True  # type: ignore[attr-defined]
        root.addHandler(handler)
