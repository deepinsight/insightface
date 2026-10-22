"""Server addon installation and read-only startup verification."""

import fcntl
import os
from pathlib import Path


class LivenessUnavailable(RuntimeError):
    """The configured liveness model could not complete inference."""


def addon_summary(name: str) -> dict[str, object]:
    from insightface.addons import ADDON_CATALOG

    artifact = ADDON_CATALOG[name]
    return {
        "model_id": name,
        "task": name,
        "file": f"addons/{artifact.filename}",
        "sha256": artifact.sha256,
    }


def install_addon(name: str, models_dir: Path) -> Path:
    from insightface.addons import ADDON_CATALOG, ensure_addon

    artifact = ADDON_CATALOG[name]
    directory = models_dir / "addons"
    directory.mkdir(parents=True, exist_ok=True)
    # Use the same advisory lock for CLI and Web installs, including custom
    # deployments with different UIDs. Keep existing directory permissions.
    lock_fd = os.open(directory / f".{name}-install.lock", os.O_CREAT | os.O_RDONLY, 0o644)
    with os.fdopen(lock_fd, "r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = (directory / artifact.filename).exists()
        path = ensure_addon(name, root=models_dir)
        if not cached or path.stat().st_mode & 0o444 != 0o444:
            # Downloads start with tempfile's private mode. These are public
            # model artifacts, also readable outside the installing process.
            path.chmod(0o644)
        return path


def require_installed_addon(name: str, models_dir: Path) -> Path:
    from insightface.addons import ADDON_CATALOG, ensure_addon

    path = models_dir / "addons" / ADDON_CATALOG[name].filename
    hint = (
        f"Configured addon '{name}' is required at {path}. "
        "The Server does not download models at startup. "
        f"Install it using the writable model tool: models addons install {name} "
        "(Compose: docker compose -f server/deploy/compose.cpu.yml run --rm "
        f"models addons install {name}; use compose.cuda12.yml for CUDA). "
        f'Or set [addons].auto_download = ["{name}"] in server.toml and rerun '
        "models install <your-model-package>."
    )
    try:
        return ensure_addon(name, root=models_dir, download=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f"addon_model_missing: {hint}") from exc
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"addon_model_invalid: {exc}. Restore the published model file. {hint}"
        ) from exc
