"""Check advertised configuration/document rules against real validators."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from insightface.app.privateframe import cli
from insightface.app.privateframe.base_config import (
    DEFAULT_CONFIG_PATH,
    validate_recognition,
    validate_revalidation_passes,
    validate_scan_passes,
)
from insightface.app.privateframe.cli_contract import (
    _full_config_contract,
    build_describe_payload,
)
from insightface.app.privateframe.pipeline import validate_result_document


@pytest.fixture(scope="module")
def contract():
    return build_describe_payload(cli.command_parser())


@pytest.fixture(scope="module")
def full_config():
    return _full_config_contract()


def _document():
    return {
        "format": "privateframe-result",
        "schema_version": 1,
        "source_video": {
            "file_name": "example.mp4",
            "metadata": {"width": 16, "height": 16, "fps": 30, "frame_count": 1},
            "coordinate_system": "pixel_xyxy",
            "frame_index_origin": 0,
            "timing_contract": "cfr_frame_index",
        },
        "render_defaults": {},
        "recognition": {"enabled": False},
        "observations": [
            {
                "frame_idx": 0,
                "track_id": "face",
                "box": [0, 0, 8, 8],
                "source": "manual",
            }
        ],
    }


def test_reference_photo_options_replace_named_gallery_contract(contract, full_config, tmp_path):
    options = contract["config"]["dotted_options"]
    for obsolete in ("recognition.gallery_dir", "recognition.target_persons"):
        assert obsolete not in options
        assert obsolete not in full_config["dotted_options"]
    assert options["recognition.reference_dir"]["type"] == ["string", "null"]
    assert options["recognition.reference_dir"]["default"] is None
    policy = options["recognition.unknown_action"]
    assert policy["default"] == "auto"
    assert policy["enum"] == ["auto", "blur", "keep"]
    for mode in options["recognition.mode"]["enum"]:
        for action in policy["enum"]:
            config = {"recognition": {
                "mode": mode, "reference_dir": str(tmp_path), "unknown_action": action,
            }}
            validate_recognition(config, tmp_path)
    with pytest.raises(ValueError, match="unknown_action"):
        validate_recognition({"recognition": {"unknown_action": "ignore"}}, tmp_path)


@pytest.mark.parametrize("recognition", [{}, {"enabled": None}, {"enabled": "false"}])
def test_advertised_recognition_requirement_is_enforced(contract, recognition):
    schema = contract["artifacts"]["result_json"]["stable_render_input_schema"][
        "recognition"
    ]
    assert schema["required"] == ["enabled"]
    assert schema["enabled"]["type"] == "boolean"
    document = _document()
    document["recognition"] = recognition
    with pytest.raises(TypeError, match="recognition.enabled must be boolean"):
        validate_result_document(document)


def test_document_compatibility_exceptions_remain_accepted(contract):
    artifact = contract["artifacts"]["result_json"]
    compatibility = artifact["accepted_input_compatibility"]
    assert "emitted" in artifact["schema_semantics"]["stable_render_input_schema"]
    assert (
        "accepted_input_compatibility"
        in artifact["schema_semantics"]["stable_render_input_schema"]
    )
    document = _document()
    for field in compatibility["source_video_optional_fields"]:
        document["source_video"].pop(field)
    assert validate_result_document(document) is document

    assert compatibility["source_video_file_name_nullable"] is True
    document["source_video"]["file_name"] = None
    assert validate_result_document(document) is document
    assert compatibility["observation_source"]["accepts_unlisted_values"] is True
    document["observations"][0]["source"] = "legacy-editor-custom-source"
    assert validate_result_document(document) is document
    document["observations"][0]["source"] = ""
    with pytest.raises(TypeError, match="source must be a non-empty string"):
        validate_result_document(document)


def test_advertised_frame_index_bound_is_enforced(contract):
    schema = contract["artifacts"]["result_json"]["stable_render_input_schema"]
    assert schema["observations"]["items"]["frame_idx"]["constraint"] == (
        "frame_idx < source_video.metadata.frame_count"
    )
    document = _document()
    document["observations"][0]["frame_idx"] = document["source_video"]["metadata"][
        "frame_count"
    ]
    with pytest.raises(ValueError, match="exceeds the source frame count"):
        validate_result_document(document)


@pytest.mark.parametrize(
    "passes", [None, [{"name": "local", "input_size": 128, "crop_expansion": 1.3}]]
)
def test_optional_pass_types_match_valid_configuration(contract, full_config, passes):
    assert "revalidation.passes" not in contract["config"]["dotted_options"]
    spec = full_config["dotted_options"]["revalidation.passes"]
    assert spec["type"] == ["array", "null"]
    assert spec["nullable"] is True
    assert spec["has_default"] is False
    assert spec["items"]["required"] == ["name", "input_size", "crop_expansion"]
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    config["revalidation"]["passes"] = passes
    validate_revalidation_passes(config)


@pytest.mark.parametrize("passes", [[], "local", 1, True])
def test_optional_pass_types_reject_invalid_configuration(full_config, passes):
    spec = full_config["dotted_options"]["revalidation.passes"]
    assert spec["min_items"] == 1
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    config["revalidation"]["passes"] = passes
    with pytest.raises(TypeError, match="must be a non-empty list"):
        validate_revalidation_passes(config)


@pytest.mark.parametrize(
    "field,value", [("input_size", 31), ("input_size", True), ("crop_expansion", 0)]
)
def test_optional_pass_item_constraints_match_validator(full_config, field, value):
    items = full_config["dotted_options"]["revalidation.passes"]["items"]
    assert items["input_size"]["multiple_of"] == 32
    assert items["crop_expansion"]["exclusive_minimum"] == 0
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    config["revalidation"]["passes"] = [
        {"name": "local", "input_size": 128, "crop_expansion": 1.3}
    ]
    config["revalidation"]["passes"][0][field] = value
    with pytest.raises((TypeError, ValueError)):
        validate_revalidation_passes(config)


def test_optional_filter_types_match_enabled_filter_validator(contract, full_config):
    options = full_config["dotted_options"]
    prefix = "scan.passes.0.candidate_filter."
    assert not any(
        path.startswith(prefix) for path in contract["config"]["dotted_options"]
    )
    assert options[prefix + "enabled"]["type"] == "boolean"
    assert options[prefix + "min_box_area_fraction"]["type"] == "number"
    assert options[prefix + "max_height_width_ratio"]["type"] == ["number", "null"]
    assert options[prefix + "max_height_width_ratio"]["nullable"] is True
    assert options[prefix + "max_height_width_ratio"]["has_default"] is False

    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    config["scan"]["passes"][0]["candidate_filter"] = {
        "enabled": True,
        "min_box_area_fraction": 0.1,
        "min_height_width_ratio": 0.8,
        "max_height_width_ratio": None,
        "aspect_ratio_exempt_min_area_fraction": 0.5,
    }
    validate_scan_passes(config)
    for field, value in (
        ("enabled", "true"),
        ("min_box_area_fraction", 2),
        ("max_height_width_ratio", 0.1),
    ):
        invalid = deepcopy(config)
        invalid["scan"]["passes"][0]["candidate_filter"][field] = value
        with pytest.raises((TypeError, ValueError)):
            validate_scan_passes(invalid)
