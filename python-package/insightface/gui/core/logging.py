"""Logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str | Path) -> Path:
    path = Path(log_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / "app.log"
    logger = logging.getLogger("insightface.gui")
    logger.setLevel(logging.INFO)
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file) for handler in logger.handlers):
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return log_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"insightface.gui.{name}")
