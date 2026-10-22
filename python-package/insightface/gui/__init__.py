"""InsightFace Evaluation Studio.

The GUI package is intentionally import-light. Importing ``insightface.gui``
does not require PySide6; GUI dependencies are loaded by the entry point.
"""

from .. import __version__

APP_NAME = "InsightFace Evaluation Studio"
APP_DISPLAY_NAME = f"InsightFace Evaluation Studio v{__version__}"

__all__ = ["__version__", "APP_NAME", "APP_DISPLAY_NAME"]
