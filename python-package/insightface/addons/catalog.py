"""Verified addon downloads, stored flat under ``<root>/addons``."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import requests


@dataclass(frozen=True)
class AddonArtifact:
    filename: str
    url: str
    sha256: str
    size_bytes: int


ADDON_CATALOG = MappingProxyType(
    {
        "liveness": AddonArtifact(
            filename="liveness.onnx",
            url=(
                "https://github.com/deepinsight/insightface-model-addons/"
                "releases/download/addons/liveness.onnx"
            ),
            sha256="87a9ac1dbb16a61eec212957e5095e62a8769c1e188af9b0198f253302c4afdb",
            size_bytes=1484006,
        ),
    }
)


def _verify(path: Path, artifact: AddonArtifact) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != artifact.sha256:
        raise RuntimeError(f"Addon SHA256 mismatch: {path}")


def ensure_addon(name: str, root="~/.insightface", *, download=True) -> Path:
    """Reuse a verified file or atomically install a verified download.

    An existing file with an unexpected digest is an error. For offline use,
    copy the published artifact to ``<root>/addons/<filename>`` beforehand.
    """

    if name not in ADDON_CATALOG:
        raise ValueError(f"Unknown addon {name!r}; available: {list(ADDON_CATALOG)}")
    artifact = ADDON_CATALOG[name]
    destination = Path(root).expanduser() / "addons" / artifact.filename
    if destination.exists():
        _verify(destination, artifact)
        return destination
    if not download:
        raise FileNotFoundError(f"Addon model is not installed: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{name}-", suffix=".download", delete=False
        ) as stream:
            temporary = Path(stream.name)
            with requests.get(artifact.url, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                size = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    size += len(chunk)
                    if size > artifact.size_bytes:
                        raise RuntimeError(
                            f"Addon download exceeds expected size: {name}"
                        )
                    stream.write(chunk)
        _verify(temporary, artifact)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
