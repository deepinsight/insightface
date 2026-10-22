from __future__ import annotations

import json
import logging
from copy import deepcopy

import pytest
import yaml

from insightface.app.privateframe import base_config, cli, pipeline
from insightface.app.privateframe.artifact_render import _identity_should_blur


@pytest.mark.parametrize("mode", ["blur_only", "exempt"])
def test_photo_config_needs_only_flat_directory_and_mode(tmp_path, mode):
    value = {"recognition": {"mode": mode, "reference_dir": "."}}
    base_config.validate_recognition(value, tmp_path)
    assert value["recognition"]["reference_dir"] == str(tmp_path)
    assert value["recognition"]["unknown_action"] == "auto"
    assert "target_persons" not in value["recognition"]


@pytest.mark.parametrize("field", ["gallery_dir", "target_persons"])
def test_old_person_library_fields_are_rejected(field, tmp_path):
    with pytest.raises(ValueError, match="unknown recognition settings"):
        base_config.validate_recognition({"recognition": {field: "old"}}, tmp_path)


@pytest.mark.parametrize("value", [None, [], {}, "ignore", True])
def test_unknown_action_rejects_invalid_values(value, tmp_path):
    with pytest.raises(ValueError, match="unknown_action"):
        base_config.validate_recognition(
            {"recognition": {"unknown_action": value}}, tmp_path,
        )


@pytest.mark.parametrize(
    ("mode", "fallback", "matched", "unmatched"),
    [("all", "blur", True, True), ("blur_only", "keep", True, False),
     ("exempt", "blur", False, True)],
)
def test_saved_result_reuses_mode_specific_policy(mode, fallback, matched, unmatched):
    defaults = deepcopy(yaml.safe_load(base_config.DEFAULT_CONFIG_PATH.read_text())["render"])
    defaults["recognition_policy"] = {"mode": mode, "unknown_action": "auto"}
    recognition = {
        "enabled": True,
        "references": {"files": [{"file": "photo.jpg"}]},
        "tracks": {
            "target": {"status": "CONFIRMED", "matched_reference_files": ["photo.jpg"]},
            "stranger": {"status": "UNKNOWN", "matched_reference_files": [],
                         "reason": "below_similarity_threshold"},
        },
    }
    result = json.loads(json.dumps({"render_defaults": defaults, "recognition": recognition}))
    settings, _digest = pipeline._render_settings(result, None, None)
    assert settings["recognition_policy"]["unknown_action"] == fallback
    for track, expected in (("target", matched), ("stranger", unmatched)):
        actual, _reason = _identity_should_blur(
            {"track_id": track}, settings["recognition_policy"], result["recognition"],
        )
        assert actual is expected


@pytest.mark.parametrize("problem", ["missing_track", "invalid_status", "foreign_reference"])
def test_broken_result_is_not_treated_as_an_unmatched_person(problem):
    record = {"status": "CONFIRMED", "matched_reference_files": ["photo.jpg"]}
    recognition = {"enabled": True, "references": {"files": [{"file": "photo.jpg"}]},
                   "tracks": {"t1": record}}
    if problem == "missing_track":
        recognition["tracks"].clear()
    elif problem == "invalid_status":
        record["status"] = "ERROR"
    else:
        record["matched_reference_files"] = ["absent.jpg"]
    with pytest.raises(ValueError):
        _identity_should_blur({"track_id": "t1"}, {"mode": "blur_only"}, recognition)


def test_unicode_photo_names_match_after_json_normalization():
    recognition = {
        "enabled": True, "references": {"files": [{"file": "é.jpg"}]},
        "tracks": {"t1": {"status": "CONFIRMED", "matched_reference_files": ["e\u0301.jpg"]}},
    }
    assert _identity_should_blur({"track_id": "t1"}, {"mode": "exempt"}, recognition)[0] is False


@pytest.mark.parametrize("problem", ["missing_tracks", "missing_decision", "bad_status"])
def test_render_preflight_rejects_broken_reference_decisions(problem):
    defaults = deepcopy(yaml.safe_load(base_config.DEFAULT_CONFIG_PATH.read_text())["render"])
    defaults["recognition_policy"] = {"mode": "blur_only"}
    recognition = {
        "enabled": True, "references": {"files": [{"file": "photo.jpg"}]},
        "tracks": {"t1": {"status": "UNKNOWN", "matched_reference_files": []}},
    }
    if problem == "missing_tracks":
        del recognition["tracks"]
    elif problem == "missing_decision":
        recognition["tracks"].clear()
    else:
        recognition["tracks"]["t1"]["status"] = "ERROR"
    with pytest.raises(ValueError):
        pipeline._render_settings(
            {"render_defaults": defaults, "recognition": recognition,
             "observations": [{"track_id": "t1"}]}, None, None,
        )


@pytest.mark.parametrize("progress", ["jsonl", "none"])
def test_reference_logs_leave_stdout_clean_and_restore_logger(capsys, progress):
    logger = logging.getLogger("insightface.app.privateframe.recognition")
    before = (logger.level, logger.propagate, list(logger.handlers))
    with cli._recognition_logs(progress):
        logger.info("group.jpg: detected 3 faces; selected largest")
        logger.warning("blank.png: no_face; skipped")
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 2
    if progress == "jsonl":
        events = [json.loads(line) for line in lines]
        assert [value["level"] for value in events] == ["info", "warning"]
        assert all(value["event"] == "log" for value in events)
    else:
        assert "group.jpg" in lines[0] and "blank.png" in lines[1]
    assert (logger.level, logger.propagate, logger.handlers) == before
