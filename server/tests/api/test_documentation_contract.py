from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

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
    assert len(expected) == 29
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
