"""Shared constants and product copy."""

from __future__ import annotations

from .. import __version__

APP_NAME = "InsightFace Evaluation Studio"
APP_VERSION = __version__
APP_ORGANIZATION = "InsightFace"
APP_DOMAIN = "insightface.ai"
APP_ID = "ai.insightface.evaluationstudio"
APP_PROCESS_NAME = APP_NAME
WINDOW_TITLE = f"InsightFace Evaluation Studio v{APP_VERSION}"
SUBTITLE = (
    "Local face recognition, face search, photo organization, model evaluation, "
    "and face swap studio."
)

LOCAL_PROCESSING_NOTICE = (
    "All processing is local by default. No images, videos, embeddings, or reports "
    "are uploaded automatically."
)

LICENSE_NOTICE = (
    "Code and model files may have different licenses. Commercial deployment "
    "requires appropriate model authorization."
)

RESPONSIBLE_USE_NOTICE = (
    "This tool does not provide legal advice. Users are responsible for consent, "
    "privacy, retention, and compliance with applicable biometric regulations."
)

COMMERCIAL_NOTICE = (
    "This evaluation may use research or non-commercial model files. Production "
    "or commercial deployment requires an appropriate commercial model license. "
    "Please contact InsightFace for commercial model licensing, private model "
    "evaluation, SDK/API access, SLA, or custom training."
)

DEFAULT_MODEL_NAME = "cheetah_s"
DEFAULT_PROVIDER = "Auto"
DEFAULT_DET_SIZE = (0, 0)
AUTO_DET_SIZES = ((128, 128), (640, 640))
DEFAULT_THRESHOLD = 0.4
DEFAULT_TOP_K = 5
DEFAULT_LICENSE_STATUS = "Research / Non-commercial"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
