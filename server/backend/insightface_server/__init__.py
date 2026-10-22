"""InsightFace Server."""

import os

# Server startup can import ORT before the InsightFace Python package.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

__version__ = "0.3.1"
