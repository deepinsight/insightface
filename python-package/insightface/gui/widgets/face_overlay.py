"""Overlay formatting helpers."""

from __future__ import annotations


def face_label(name: str = "Unknown", similarity: float | None = None) -> str:
    if similarity is None:
        return name or "Unknown"
    return f"{name or 'Unknown'} {similarity:.2f}"
