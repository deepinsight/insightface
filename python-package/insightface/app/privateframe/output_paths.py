"""Stable public output names shared by the PrivateFrame API, CLI, and GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrivateFrameOutputPaths:
    """Default artifact paths for one source video."""

    source: Path
    output_dir: Path
    result_json: Path
    result_video: Path
    debug_video: Path
    workdir: Path


def default_output_paths(
    input_path: str | Path,
    output_dir: str | Path | None = None,
) -> PrivateFrameOutputPaths:
    """Return stable sibling names based on the original video stem.

    The analysis JSON and normal rendered video deliberately share one stem so
    external tools can move, discover, and edit them as a pair.  ``workdir`` is
    runtime/audit storage, not a public result location.
    """

    source = Path(input_path).expanduser().resolve()
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source.parent
    )
    source_stem = source.stem or "video"
    result_stem = f"{source_stem}_privateframe"
    return PrivateFrameOutputPaths(
        source=source,
        output_dir=destination,
        result_json=destination / f"{result_stem}.json",
        result_video=destination / f"{result_stem}.mp4",
        debug_video=destination / f"{result_stem}_debug.mp4",
        workdir=destination / f".{source_stem}_privateframe_work",
    )


__all__ = ["PrivateFrameOutputPaths", "default_output_paths"]
