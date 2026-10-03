"""Application bootstrap for InsightFace Evaluation Studio."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable

from pathlib import Path

from .core.config import AppConfig, PersonConfigError, load_config, save_config
from .core.face_engine import FaceEngine, is_cuda_provider_available, providers_from_choice
from .core.logging import setup_logging
from .core.model_packages import GUI_MODEL_PACKAGES
from .core.storage import Storage


@dataclass
class StudioContext:
    config: AppConfig
    config_exists: bool
    storage: Storage
    engine: FaceEngine
    log_file: str
    runtime_safe_mode: bool = False
    model_downloads_in_progress: int = 0
    privateframe_jobs_in_progress: int = 0
    person_analysis_jobs_in_progress: int = 0


def context_activity_count(context, name: str) -> int:
    """Read a non-negative GUI activity count from a shared context."""

    try:
        return max(0, int(getattr(context, name, 0)))
    except (TypeError, ValueError):
        return 0


def begin_context_activity(context, name: str) -> int:
    count = context_activity_count(context, name) + 1
    setattr(context, name, count)
    return count


def end_context_activity(context, name: str) -> int:
    count = max(0, context_activity_count(context, name) - 1)
    setattr(context, name, count)
    return count


def create_face_engine(config: AppConfig) -> FaceEngine:
    """Create an unloaded GUI engine from one global configuration snapshot."""

    custom_model_dir = (
        "" if config.model_name in GUI_MODEL_PACKAGES else config.custom_model_dir
    )
    return FaceEngine(
        model_name=config.model_name,
        providers=providers_from_choice(config.provider),
        det_size=config.det_size_tuple,
        root=config.model_root,
        custom_model_dir=custom_model_dir,
    )


def engine_matches_config(engine: FaceEngine, config: AppConfig) -> bool:
    """Return whether an engine represents the current global model settings."""

    engine_custom_dir = getattr(engine, "custom_model_dir", None)
    configured_custom_value = (
        "" if config.model_name in GUI_MODEL_PACKAGES else config.custom_model_dir
    )
    configured_custom_dir = (
        Path(configured_custom_value).expanduser()
        if configured_custom_value
        else None
    )
    return (
        str(getattr(engine, "model_name", "")) == str(config.model_name)
        and Path(getattr(engine, "root", "")).expanduser()
        == Path(config.model_root).expanduser()
        and engine_custom_dir == configured_custom_dir
        and tuple(getattr(engine, "requested_providers", ()))
        == tuple(providers_from_choice(config.provider))
        and tuple(getattr(engine, "det_size", ())) == config.det_size_tuple
    )


def reconfigure_context_engine(
    context: StudioContext,
    *,
    force: bool = False,
) -> bool:
    """Invalidate the shared GUI engine when global model settings changed."""

    if not force and engine_matches_config(context.engine, context.config):
        return False
    context.engine = create_face_engine(context.config)
    return True


def _unique_existing_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    unique_paths = []
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            unique_paths.append(path)
    return unique_paths


def qt_plugin_root_candidates() -> list[Path]:
    """Return plausible Qt plugin roots for PySide6 wheel and conda layouts."""

    try:
        import PySide6
    except ImportError:
        return []

    candidates = []
    for package_dir_raw in getattr(PySide6, "__path__", []):
        package_dir = Path(package_dir_raw)
        candidates.extend(
            [
                package_dir / "Qt" / "plugins",
                package_dir / "plugins",
            ]
        )

    try:
        from PySide6.QtCore import QLibraryInfo

        candidates.append(Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)))
    except Exception:
        pass

    candidates.extend(
        [
            Path(sys.prefix) / "plugins",
            Path(sys.prefix) / "Library" / "plugins",
            Path(sys.prefix) / "lib" / "qt6" / "plugins",
        ]
    )
    return _unique_existing_paths(candidates)


def configure_qt_plugin_paths() -> None:
    """Point Qt at PySide6-Essentials plugins when the PySide6 meta package is absent."""

    for plugins in qt_plugin_root_candidates():
        platforms = plugins / "platforms"
        if platforms.exists():
            if not os.environ.get("QT_PLUGIN_PATH"):
                os.environ["QT_PLUGIN_PATH"] = str(plugins)
            if not os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
            return


def create_context(args=None) -> StudioContext:
    config_path = None
    runtime_safe_mode = bool(args is not None and getattr(args, "safe_mode", False))
    if args is not None and getattr(args, "workspace", None):
        config_path = Path(args.workspace).expanduser() / "config.json"
    config, exists = load_config(config_path)
    if args is not None:
        if getattr(args, "workspace", None):
            config.workspace_path = args.workspace
            config.database_path = ""
            config.crop_dir = ""
            config.export_dir = ""
            config.report_dir = ""
            config.log_dir = ""
            config.cache_dir = ""
            config.apply_workspace_defaults()
        if getattr(args, "model", None):
            selected_model = str(args.model).strip()
            config.model_name = selected_model
            config.custom_model_dir = (
                "" if selected_model in GUI_MODEL_PACKAGES else selected_model
            )
        if getattr(args, "provider", None):
            value = str(args.provider).upper()
            if value == "CPU":
                config.provider = "CPU"
            elif value == "CUDA":
                config.provider = "CUDA" if is_cuda_provider_available() else "Auto"
            else:
                config.provider = "Auto"
    if config.safe_mode:
        # safe-mode is a startup troubleshooting flag. Older builds persisted it
        # into config.json, which made normal launches silently skip model load.
        config.safe_mode = False
        if not config.auto_load_model:
            config.auto_load_model = True
    if str(config.provider).strip().lower() == "cuda" and not is_cuda_provider_available():
        config.provider = "Auto"
    config.apply_workspace_defaults()
    save_config(config)
    log_file = setup_logging(config.log_dir)
    storage = Storage(config.database_path)
    engine = create_face_engine(config)
    return StudioContext(
        config=config,
        config_exists=exists,
        storage=storage,
        engine=engine,
        log_file=str(log_file),
        runtime_safe_mode=runtime_safe_mode,
    )


def run_app(args=None) -> int:
    configure_qt_plugin_paths()
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("InsightFace GUI requires PySide6.")
        print("Please install with: pip install insightface[gui]")
        return 1

    from .main_window import MainWindow
    from .resources import configure_application_metadata

    app = QApplication.instance() or QApplication(sys.argv[:1])
    configure_application_metadata(app)
    try:
        context = create_context(args)
    except PersonConfigError as error:
        from PySide6.QtWidgets import QMessageBox
        from .core.i18n import tr

        QMessageBox.critical(None, tr('Invalid person configuration', 'system'),
            tr('Could not load person settings:\n{error}', 'system').format(error=str(error)))
        return 2
    window = MainWindow(context)
    window.resize(1320, 860)
    window.show()
    return int(app.exec())
