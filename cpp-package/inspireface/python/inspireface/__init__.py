from pathlib import Path as _Path

from . import modules as _modules
from . import param as _param
from .modules import *
from .param import *


def _package_version() -> str:
    source_version = _Path(__file__).resolve().parent.parent / "version.txt"
    try:
        if source_version.is_file():
            return source_version.read_text(encoding="utf-8").strip()
    except OSError:
        pass

    try:
        from importlib.metadata import PackageNotFoundError, version as distribution_version
    except ImportError:  # Python 3.7
        from importlib_metadata import PackageNotFoundError, version as distribution_version

    try:
        return distribution_version("inspireface")
    except PackageNotFoundError:
        # Source trees copied without packaging metadata still have a usable
        # native version, so retain a deterministic fallback.
        return version()


__version__ = _package_version()
__native_version__ = version()
native_version = version

# Compatibility aliases for code that reached through the historical wildcard
# imports. They remain directly addressable but are not part of the stable star
# export manifest.
core = _modules.core
exception = _modules.exception
herror = _modules.herror
inspireface = _modules.inspireface
utils = _modules.utils

__all__ = tuple(
    dict.fromkeys(
        _modules.__all__
        + _param.__all__
        + ("__version__", "__native_version__", "native_version")
    )
)

del _Path
