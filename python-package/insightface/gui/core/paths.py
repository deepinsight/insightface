"""Workspace and filesystem helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def default_workspace() -> Path:
    return Path.home() / ".insightface" / "gui"


def default_config_path() -> Path:
    return default_workspace() / "config.json"


def expand_path(value: str | os.PathLike[str] | None) -> Path:
    if value is None or str(value).strip() == "":
        return default_workspace()
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def workspace_paths(workspace: str | os.PathLike[str] | None = None) -> Dict[str, Path]:
    root = expand_path(workspace)
    return {
        "workspace": root,
        "database": root / "insightface_gui.db",
        "crops": root / "crops",
        "exports": root / "exports",
        "reports": root / "reports",
        "logs": root / "logs",
        "cache": root / "cache",
    }


def ensure_workspace(workspace: str | os.PathLike[str] | None = None) -> Dict[str, Path]:
    paths = workspace_paths(workspace)
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    return paths
