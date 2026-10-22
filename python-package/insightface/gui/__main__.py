"""Command line entry point for InsightFace Evaluation Studio."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InsightFace Evaluation Studio")
    parser.add_argument("--workspace", help="Use a specific local workspace directory")
    parser.add_argument("--model", help="Model pack name or local model directory")
    parser.add_argument("--provider", choices=["auto", "cpu", "cuda", "Auto", "CPU", "CUDA"], help="Execution provider")
    parser.add_argument("--safe-mode", action="store_true", help="Open the GUI without loading a model")
    parser.add_argument("--version", action="store_true", help="Print version information and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        try:
            import insightface

            insightface_version = getattr(insightface, "__version__", __version__)
        except Exception:
            insightface_version = __version__
        print(f"InsightFace Evaluation Studio {__version__}")
        print(f"insightface {insightface_version}")
        return 0
    try:
        from .app import run_app
    except ImportError as exc:
        if "PySide6" in str(exc):
            print("InsightFace GUI requires PySide6.")
            print("Please install with: pip install insightface[gui]")
            return 1
        raise
    return run_app(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
