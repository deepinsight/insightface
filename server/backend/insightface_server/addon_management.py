"""Explicit Web addon preparation; running inference never changes here."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Request

from .addon_config import (
    AddonConfigError,
    editable_config_error,
    liveness_config_lock,
    writable,
    write_enabled_addons,
)
from .addons import install_addon
from .config import Settings, load_server_config
from .errors import ApiError


def require_management_request(request: Request, cors_origins: tuple[str, ...]) -> None:
    """Require a non-simple JSON request and reject unrelated browser origins."""

    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise ApiError("json_required", "Use Content-Type: application/json with an empty object.", 415)
    origin = request.headers.get("origin")
    if origin is None:
        return  # CLI/API clients do not send Origin.
    try:
        source = urlsplit(origin)
    except ValueError:
        raise ApiError("origin_not_allowed", "This origin may not change Server configuration.", 403) from None
    valid_origin = (
        origin != "null"
        and source.scheme in ("http", "https")
        and bool(source.netloc)
        and not source.path
        and not source.query
        and not source.fragment
        and source.username is None
    )
    same_origin = source.netloc == request.url.netloc and source.scheme == request.url.scheme
    if not valid_origin or (not same_origin and origin not in cors_origins and "*" not in cors_origins):
        raise ApiError("origin_not_allowed", "This origin may not change Server configuration.", 403)


class LivenessManager:
    def __init__(self, settings: Settings, *, enabled: bool):
        self.settings = settings
        self.enabled = enabled
        self.model_path = settings.models_dir / "addons" / "liveness.onnx"
        self._mutex = threading.Lock()
        self._state = "idle"
        self._error: dict[str, str] | None = None
        self._task: asyncio.Task | None = None
        self._stopping = threading.Event()

    def _installed(self) -> tuple[bool, dict[str, str] | None]:
        from insightface.addons import ensure_addon

        try:
            ensure_addon("liveness", root=self.settings.models_dir, download=False)
            return True, None
        except FileNotFoundError:
            return False, None
        except (OSError, RuntimeError):
            return False, {
                "code": "addon_model_invalid",
                "message": (
                    f"The existing addon at {self.model_path} is unreadable or failed SHA256 "
                    "verification. Restore the official model, or remove the invalid file "
                    "and retry. It will not be overwritten automatically."
                ),
            }

    def _capability(self) -> tuple[str, str] | None:
        config_error = editable_config_error(self.settings.config_file)
        if config_error:
            return config_error
        addon_dir = self.model_path.parent
        parent = addon_dir if addon_dir.exists() else self.settings.models_dir
        if not parent.is_dir() or not writable(parent, directory=True):
            return "addon_directory_not_writable", "Mount the model directory writable at /models and allow the Server process to create or write its addons subdirectory. Check host directory permissions and recreate the container after changing mounts."
        return None

    def status(self) -> dict[str, Any]:
        installed, artifact_error = self._installed()
        capability = self._capability()
        unavailable_code, reason = capability if capability else (None, None)
        configured_enabled = self.enabled
        config_error = None
        try:
            configured_enabled = "liveness" in load_server_config(self.settings.config_file).addons
        except ValueError:
            reason = "The current server.toml is unreadable or invalid. Correct it before enabling liveness."
            unavailable_code = "addon_config_invalid"
            config_error = {"code": "addon_config_invalid", "message": reason}
        with self._mutex:
            state, error = self._state, self._error
        error = error or config_error or artifact_error
        if reason is None and artifact_error:
            unavailable_code, reason = artifact_error["code"], artifact_error["message"]
        if reason is None and self._stopping.is_set():
            unavailable_code, reason = "server_stopping", "The Server is shutting down."
        if state != "downloading":
            if error:
                state = "error"
            else:
                state = "ready" if installed else "idle"
        return {
            "enabled": self.enabled,
            "installed": installed,
            "configured_enabled": configured_enabled,
            "restart_required": configured_enabled != self.enabled,
            "can_enable": reason is None and artifact_error is None and not self._stopping.is_set(),
            "unavailable_code": unavailable_code,
            "unavailable_reason": reason,
            "state": state,
            "error": error,
            "model_path": str(self.model_path),
            "config_file": str(self.settings.config_file) if self.settings.config_file else None,
        }

    async def enable(self) -> dict[str, Any]:
        snapshot = await asyncio.to_thread(self.status)
        if not snapshot["can_enable"]:
            raise ApiError(
                "addon_management_unavailable",
                snapshot["unavailable_reason"] or "The Server is shutting down.",
                409,
            )
        # No await between checking and publishing the owned task: duplicate
        # requests on this event loop join the same job.
        if self._task is None or self._task.done():
            with self._mutex:
                self._state, self._error = "downloading", None
            self._task = asyncio.create_task(asyncio.to_thread(self._prepare))
        return await asyncio.to_thread(self.status)

    def _save_config(self, path: Path) -> None:
        write_enabled_addons(path, ("liveness",), cancelled=self._stopping.is_set)

    def _prepare(self) -> None:
        stage = "config"
        try:
            path = self.settings.config_file
            assert path is not None
            with liveness_config_lock(path):
                capability = self._capability()
                if capability:
                    raise ApiError("addon_management_unavailable", capability[1], 409)
                load_server_config(path)
                stage = "download"
                installed, artifact_error = self._installed()
                if artifact_error:
                    raise ApiError(artifact_error["code"], artifact_error["message"], 409)
                if not installed:
                    install_addon("liveness", self.settings.models_dir)
                installed, artifact_error = self._installed()
                if not installed:
                    raise RuntimeError("Addon verification failed after installation")
                stage = "config"
                self._save_config(path)
            with self._mutex:
                self._state, self._error = "ready", None
        except Exception as exc:
            # Requests exceptions can contain proxy user/passwords. Never copy
            # their text to the public status or logs.
            error = (
                {"code": exc.code, "message": exc.message}
                if isinstance(exc, ApiError) or (isinstance(exc, AddonConfigError) and exc.code == "addon_job_in_progress")
                else {
                    "code": "addon_download_failed" if stage == "download" else "addon_config_save_failed",
                    "message": (
                        "Could not download or verify the official liveness model. Check the Server network/proxy settings and retry. Configuration was not changed."
                        if stage == "download"
                        else "Could not save the liveness configuration. Check server.toml, directory mount permissions, and concurrent edits, then retry. A downloaded model can be reused."
                    ),
                }
            )
            with self._mutex:
                self._state, self._error = "error", error

    async def close(self) -> None:
        self._stopping.set()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10)
            except TimeoutError:
                # The downloader has bounded connect/read timeouts. Its worker
                # may finish caching a file but must not save config afterward.
                pass
