"""Resolve the packaged InspireFace native library without loading it."""

import os
import platform
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union


_LIBRARY_NAMES = {
    "darwin": "libInspireFace.dylib",
    "linux": "libInspireFace.so",
    "windows": "libInspireFace.dll",
}

_ARCHITECTURES = {
    "amd64": "x64",
    "x86_64": "x64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


def platform_library_spec(
    system: Optional[str] = None,
    machine: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Return ``(platform_directory, architecture, library_name)``.

    The architecture follows the running Python process. For example, an
    x86_64 Python running through Rosetta must load an x86_64 library.
    """
    normalized_system = (system or platform.system()).lower()
    normalized_machine = (machine or platform.machine()).lower()
    library_name = _LIBRARY_NAMES.get(normalized_system)
    architecture = _ARCHITECTURES.get(normalized_machine)
    if library_name is None or architecture is None:
        raise RuntimeError(
            "Unsupported platform: system={}, machine={}".format(
                normalized_system,
                normalized_machine,
            )
        )
    return normalized_system, architecture, library_name


def get_lib_path(
    package_dir: Optional[Union[str, os.PathLike]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the validated path to the InspireFace native library.

    ``INSPIREFACE_LIBRARY_PATH`` is an opt-in override for local development
    and diagnostics. Normal wheel users continue loading the bundled library.
    """
    environment = os.environ if environ is None else environ
    override = environment.get("INSPIREFACE_LIBRARY_PATH")
    if override:
        library_path = Path(override).expanduser().resolve()
    else:
        platform_dir, architecture, library_name = platform_library_spec()
        root = Path(package_dir) if package_dir is not None else Path(__file__).parent
        library_path = root / "libs" / platform_dir / architecture / library_name

    if not library_path.is_file():
        raise RuntimeError("InspireFace native library not found: {}".format(library_path))
    return str(library_path)
