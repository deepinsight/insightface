from __future__ import annotations

import pytest

from insightface.app.privateframe.artifact_render import _identity_should_blur
from insightface.app.privateframe.pipeline import _export_observations


@pytest.mark.parametrize(
    ("mode", "action", "expected"),
    [("all", "auto", True), ("blur_only", "auto", False),
     ("exempt", "auto", True), ("blur_only", "blur", True),
     ("exempt", "keep", False)],
)
def test_unconfirmed_endpoints_follow_policy_before_and_after_json_export(
    mode: str, action: str, expected: bool,
) -> None:
    observation = {
        "track_id": "t1", "frame_idx": 100, "box": [1, 2, 31, 42],
        "endpoint_repair_reason": "interpolate_unanchored_endpoint",
        "reduced_assurance": True,
    }
    policy = {"mode": mode, "unknown_action": action}
    recognition = {
        "enabled": True, "references": {"files": [{"file": "photo.jpg"}]},
        "tracks": {"t1": {"status": "CONFIRMED",
                           "matched_reference_files": ["photo.jpg"]}},
    }
    exported = _export_observations([observation])[0]
    assert exported["identity_unconfirmed"] is True
    assert "force_blur" not in exported
    for item in (observation, exported):
        should_blur, _reason = _identity_should_blur(item, policy, recognition)
        assert should_blur is expected
