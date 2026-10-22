from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from insightface_server.api.responses import LivenessManagementResponse

SERVER_DIR = Path(__file__).resolve().parents[2]
DOC_SUFFIXES = ("", ".zh-CN", ".ja", ".de", ".es", ".fr", ".ru", ".pt", ".ko")
DOCUMENTED_OPERATION = re.compile(
    r"^### `(?P<method>GET|POST|PATCH|DELETE) (?P<path>/v1/[^`]+)`$",
    re.MULTILINE,
)


def _public_operations(client: TestClient) -> set[str]:
    operations: set[str] = set()
    for route in client.app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/v1/"):
            continue
        for method in route.methods & {"GET", "POST", "PATCH", "DELETE"}:
            operations.add(f"{method} {route.path}")
    return operations


def _documented_sections(markdown: str) -> dict[str, str]:
    matches = list(DOCUMENTED_OPERATION.finditer(markdown))
    return {
        f"{match.group('method')} {match.group('path')}": markdown[
            match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        ]
        for index, match in enumerate(matches)
    }


def test_every_locale_documents_exactly_every_public_operation(client: TestClient) -> None:
    expected = _public_operations(client)
    assert len(expected) == 31
    for suffix in DOC_SUFFIXES:
        path = SERVER_DIR / "docs" / f"api{suffix}.md"
        sections = _documented_sections(path.read_text(encoding="utf-8"))
        assert set(sections) == expected, path.name
        for operation, content in sections.items():
            # CJK conveys the same guidance with fewer code points than Latin
            # scripts, so route coverage and labeled guidance are the primary
            # contract; this floor only catches empty placeholder sections.
            assert len(content.strip()) >= 70, f"{path.name}: {operation} is too brief"
            assert content.count("**") >= 4, (
                f"{path.name}: {operation} lacks usage/result guidance"
            )


def test_reviewed_openapi_snapshot_matches_runtime_contract(client: TestClient) -> None:
    expected = json.loads(
        (SERVER_DIR / "docs" / "openapi.snapshot.json").read_text(encoding="utf-8")
    )
    assert client.app.openapi() == expected


def test_every_locale_explains_management_states_and_has_a_valid_response_example(
    client: TestClient,
) -> None:
    schema = client.app.openapi()["components"]["schemas"]["LivenessManagementResponse"]
    fields = set(schema["properties"])
    state_values = set(schema["properties"]["state"]["enum"])
    for suffix in DOC_SUFFIXES:
        path = SERVER_DIR / "docs" / f"api{suffix}.md"
        markdown = path.read_text(encoding="utf-8")
        sections = _documented_sections(markdown)
        status = sections["GET /v1/addons/liveness"]
        prepare = sections["POST /v1/addons/liveness/enable"]
        # Check the prose, not only a copied JSON block: clients need to know
        # how installed, active and next-startup settings differ.
        prose = re.sub(r"```[\s\S]*?```", "", status)
        for field in fields:
            assert f"`{field}`" in prose, f"{path.name}: {field} is unexplained"
        for value in state_values:
            assert f"`{value}`" in prose, f"{path.name}: {value} is unexplained"
        examples = re.findall(r"```json\s*([\s\S]*?)```", status)
        assert examples, path.name
        pending = json.loads(examples[0])
        LivenessManagementResponse.model_validate(pending)
        assert set(pending) == fields, path.name
        assert pending["installed"] and pending["configured_enabled"]
        assert not pending["enabled"] and pending["restart_required"]
        for contract in (
            "application/json", "{}", "202", "GET /v1/addons/liveness",
            "invalid_addon_request", "unauthorized", "origin_not_allowed",
            "addon_management_unavailable", "json_required", "addon_download_failed",
            "addon_config_save_failed", "addon_config_invalid", "addon_model_invalid",
            "addon_job_in_progress", "error.code", "restart_required=false",
        ):
            assert contract in prepare, f"{path.name}: missing {contract}"


def test_translated_liveness_guidance_has_no_english_placeholder_labels() -> None:
    for suffix in DOC_SUFFIXES[1:]:
        path = SERVER_DIR / "docs" / f"api{suffix}.md"
        markdown = path.read_text(encoding="utf-8")
        for placeholder in (
            "| live |", "| fake |", "| input rejected |", "Startup errors:",
            "Compose host:", "`both` (default)",
        ):
            assert placeholder not in markdown, f"{path.name}: {placeholder}"


def test_live_openapi_explains_image_formats_and_liveness_policy(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    introduction = schema["info"]["description"]
    for guidance in ("JPEG, PNG, WebP, and BMP", "server.toml", "normal", "observe"):
        assert guidance in introduction
    for guidance in (
        "Liveness is disabled by default",
        'inference.addons = ["liveness"]',
        'addons.auto_download = ["liveness"]',
        "models install <package>",
        "base package is cached",
        "inference.addons = []",
        "Registration skips liveness by default",
        "startup does not download models",
    ):
        assert guidance in introduction

    image_paths = (
        "/v1/detect",
        "/v1/embeddings",
        "/v1/compare",
        "/v1/collections/{collection_id}/search",
        "/v1/collections/{collection_id}/persons",
        "/v1/collections/{collection_id}/persons/{person_id}/faces",
    )
    for path in image_paths:
        description = schema["paths"][path]["post"]["description"]
        assert "liveness" in description, path
        assert "liveness_unavailable" in description, path
        assert "input_rejected" in description, path
    assert "HTTP 200" in schema["paths"]["/v1/detect"]["post"]["description"]
    for path in image_paths[-2:]:
        assert "liveness_on_registration=false" in schema["paths"][path]["post"]["description"]

    result = schema["components"]["schemas"]["LivenessResult"]
    assert set(result["properties"]) == {"status", "is_live", "live_score", "reason"}
    assert set(result["required"]) == {"status", "is_live", "live_score"}
    assert '"status":"input_rejected","is_live":null,"live_score":null' in result["description"]
    assert "English" in result["properties"]["reason"]["description"]
    assert "Older results may omit" in result["properties"]["reason"]["description"]
    for field in result["properties"].values():
        assert field["description"]
    observation = schema["components"]["schemas"]["FaceObservation"]
    assert "Omitted" in observation["properties"]["liveness"]["description"]
    monitor = schema["paths"]["/v1/monitors/{monitor_id}/state"]["get"]["description"]
    assert "liveness_blocked_faces" in monitor
    assert "unknown_faces" in monitor
