from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

SERVER_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = SERVER_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from insightface_server.app import create_app  # noqa: E402
from insightface_server.config import Settings  # noqa: E402


def synthetic_image(
    seed: int = 1,
    *,
    width: int = 128,
    height: int = 128,
    blank: bool = False,
) -> bytes:
    """Create a deterministic synthetic PNG; it does not depict a real person."""

    if blank:
        pixels = np.full((height, width, 3), 127, dtype=np.uint8)
    else:
        generator = np.random.default_rng(seed)
        pixels = generator.integers(24, 232, (height, width, 3), dtype=np.uint8)
        # Add stable high-contrast geometry so the mock quality heuristics are exercised.
        pixels[::8, :, :] = 245
        pixels[:, ::8, :] = 12
    stream = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def image_bytes() -> Callable[..., bytes]:
    return synthetic_image


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    counter = 0

    def factory(**overrides: object) -> Settings:
        nonlocal counter
        counter += 1
        root = tmp_path / f"instance-{counter}"
        defaults = Settings(
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
        return replace(defaults, **overrides)

    return factory


@pytest.fixture
def client(make_settings: Callable[..., Settings]):
    settings = make_settings()
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def create_collection():
    def factory(
        client: TestClient,
        collection_id: str = "employees",
        *,
        threshold: float = 0.4,
    ) -> dict[str, object]:
        response = client.post(
            "/v1/collections",
            json={
                "id": collection_id,
                "name": collection_id.title(),
                "description": "Synthetic test collection",
                "threshold": threshold,
                "metadata": {"fixture": True},
            },
        )
        assert response.status_code == 201, response.text
        return response.json()["collection"]

    return factory


@pytest.fixture
def create_person(image_bytes: Callable[..., bytes]):
    def factory(
        client: TestClient,
        collection_id: str,
        person_id: str,
        *,
        seed: int = 1,
        name: str | None = None,
    ) -> dict[str, object]:
        response = client.post(
            f"/v1/collections/{collection_id}/persons",
            data={
                "id": person_id,
                "name": name or person_id.title(),
                "external_id": f"external-{person_id}",
                "metadata": '{"synthetic":true}',
            },
            files={"images": (f"{person_id}.png", image_bytes(seed), "image/png")},
        )
        assert response.status_code == 201, response.text
        return response.json()

    return factory
