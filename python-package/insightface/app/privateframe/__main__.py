"""Module entry point for ``python -m insightface.app.privateframe``."""

from __future__ import annotations

from .cli_bootstrap import main

if __name__ == "__main__":
    raise SystemExit(main())
