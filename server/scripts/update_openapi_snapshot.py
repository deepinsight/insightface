#!/usr/bin/env python3
"""Regenerate the reviewed public OpenAPI contract snapshot.

Run this only together with the matching API implementation, all localized
``docs/api*.md`` files, SDK/UI changes, tests, and migration guidance when
needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = SERVER_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from insightface_server.app import create_app  # noqa: E402
from insightface_server.config import Settings  # noqa: E402


def main() -> None:
    root = SERVER_DIR / "build" / "openapi-snapshot"
    settings = Settings(
        data_dir=root / "data",
        models_dir=root / "models",
        inference_mode="mock",
        execution_provider="CPUExecutionProvider",
        auth_enabled=False,
        startup_api_key=None,
        max_image_bytes=2 * 1024 * 1024,
        max_image_pixels=2_000_000,
        max_request_bytes=8 * 1024 * 1024,
        max_registration_images=20,
        request_timeout_seconds=10,
        detector_threshold=0.5,
        registration_min_score=0.6,
        registration_min_quality=0.35,
        registration_min_face_size=40,
        default_threshold=0.4,
        cors_origins=(),
        log_level="WARNING",
        save_face_crops=False,
        device_id=0,
    )
    document = json.dumps(
        create_app(settings).openapi(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    output = SERVER_DIR / "docs" / "openapi.snapshot.json"
    output.write_text(f"{document}\n", encoding="utf-8")
    print(f"updated {output}")


if __name__ == "__main__":
    main()
