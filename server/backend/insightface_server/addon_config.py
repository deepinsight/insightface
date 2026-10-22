"""Shared configuration checks and atomic saves for explicit addon enablement."""

from __future__ import annotations

import fcntl
import os
import stat
import tempfile
from collections.abc import Callable, Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path

import tomlkit
from tomlkit.items import Array

from .config import SUPPORTED_ADDONS, load_server_config


class AddonConfigError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def writable(path: Path, *, directory: bool = False) -> bool:
    try:
        mode = path.stat().st_mode
        # Root must still respect a deliberately read-only configuration.
        return bool(mode & 0o222) and os.access(path, os.W_OK | (os.X_OK if directory else 0))
    except OSError:
        return False


def file_mount(path: Path) -> bool:
    if os.path.ismount(path):
        return True
    # os.path.ismount does not detect same-filesystem Linux bind mounts.
    try:
        target = str(path.resolve())
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            mount_point = line.split()[4]
            for escaped, literal in ((r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")):
                mount_point = mount_point.replace(escaped, literal)
            if mount_point == target:
                return True
    except (OSError, IndexError):
        pass
    return False


def editable_config_error(path: Path | None) -> tuple[str, str] | None:
    if path is None:
        return "config_file_missing", "Set INSIGHTFACE_CONFIG_FILE to an editable server.toml to enable liveness (or use the model tool's --config-file option)."
    if not path.is_file() or path.is_symlink():
        return "config_file_not_regular", "The configured server.toml must be an existing regular file, not a symbolic link."
    if file_mount(path):
        return "config_file_mount", "Mount the configuration directory writable instead of bind-mounting the single server.toml file, then recreate the container."
    if not writable(path) or not writable(path.parent, directory=True):
        return "config_not_writable", "The configuration file and its directory must allow writes by the Server process. Use a writable configuration directory mount, check host directory/file permissions, and recreate the container."
    return None


def require_editable_config(path: Path | None) -> Path:
    error = editable_config_error(path)
    if error:
        raise AddonConfigError(*error)
    assert path is not None
    load_server_config(path)
    return path


@contextmanager
def liveness_config_lock(path: Path) -> Iterator[None]:
    """Share one stable lock between Web and CLI across download and config save."""

    lock_fd = os.open(path.parent / ".liveness-management.lock", os.O_CREAT | os.O_RDONLY, 0o644)
    with os.fdopen(lock_fd, "r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AddonConfigError(
                "addon_job_in_progress",
                "Another Server or model installer is preparing liveness; wait and retry.",
            ) from None
        yield


def write_enabled_addons(
    path: Path,
    addons: tuple[str, ...],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Append enabled addons atomically; caller holds the shared configuration lock."""

    if any(addon not in SUPPORTED_ADDONS for addon in addons):
        raise ValueError("Unsupported addon requested for configuration")
    require_editable_config(path)
    # Re-read after download so unrelated intervening edits survive.
    original = path.read_text(encoding="utf-8")
    document = tomlkit.parse(original)
    for section, key in (("inference", "addons"), ("addons", "auto_download")):
        if section not in document:
            document[section] = tomlkit.table()
        table = document[section]
        if not isinstance(table, MutableMapping):
            raise ValueError(f"[{section}] must be a TOML table")
        if key not in table:
            table[key] = tomlkit.array()
        values = table[key]
        if not isinstance(values, Array):
            raise ValueError(f"{section}.{key} must be an array")
        for addon in addons:
            if addon not in values:
                values.append(addon)
    updated = tomlkit.dumps(document)
    if cancelled is not None and cancelled():
        raise RuntimeError("Server shutdown interrupted addon preparation")
    if original == updated:
        if path.read_text(encoding="utf-8") != original:
            raise RuntimeError("Configuration changed while saving; retry")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".server-config-", suffix=".toml", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        load_server_config(temporary)
        if cancelled is not None and cancelled():
            raise RuntimeError("Server shutdown interrupted addon preparation")
        # Detect non-cooperating manual edits before publishing our snapshot.
        require_editable_config(path)
        if path.read_text(encoding="utf-8") != original:
            raise RuntimeError("Configuration changed while saving; retry")
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
