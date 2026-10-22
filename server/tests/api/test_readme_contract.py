"""GitHub overview claims must match the same runtime served to Web clients."""

import re
from pathlib import Path

from fastapi.routing import APIRoute

SERVER_DIR = Path(__file__).resolve().parents[2]


def test_localized_readme_operation_counts_match_running_api(client):
    operations = {
        (method, route.path)
        for route in client.app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/v1/")
        for method in route.methods & {"GET", "POST", "PATCH", "DELETE"}
    }
    for path in SERVER_DIR.glob("README*.md"):
        text = path.read_text(encoding="utf-8")
        statements = [line for line in text.splitlines() if "snake_case" in line]
        assert len(statements) == 1, path.name
        count = re.search(r"(\d+)", statements[0].replace("/v1", ""))
        assert count and int(count.group(1)) == len(operations), path.name


def test_all_github_overviews_include_current_addon_defaults_and_image_support():
    readmes = sorted(SERVER_DIR.glob("README*.md"))
    assert len(readmes) == 9
    for path in readmes:
        text = path.read_text(encoding="utf-8")
        for marker in (
            "inference.addons = []", "addons.auto_download = []",
            "liveness_on_registration = false", "server/config/server.toml",
            "`status`", "`is_live`", "`live_score`", "BMP",
            "raccoon_s", "raccoon_l", "docs/user-guide",
        ):
            assert marker in text, (path.name, marker)
