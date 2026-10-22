"""Endpoint evidence controls validate without model or media work."""

from copy import deepcopy

import pytest

from insightface.app.privateframe import cli
from insightface.app.privateframe.base_config import (
    DEFAULT_CONFIG_PATH,
    apply_config_overrides,
    read_default_config,
    validate_config_keys,
    validate_endpoint_conflicts,
)
from insightface.app.privateframe.cli_contract import (
    _full_config_contract,
    build_describe_payload,
)
from insightface.app.privateframe.config import load_config


PREFIX = "tracking.endpoint_conflicts"


def _config(**changes):
    config = read_default_config()
    config["tracking"]["endpoint_conflicts"].update(changes)
    return config


def test_base_endpoint_controls_support_ab_and_reuse_only_overrides():
    config = _config()
    original = deepcopy(config["tracking"]["endpoint_conflicts"])
    validate_endpoint_conflicts(config)
    apply_config_overrides(config, {f"{PREFIX}.enabled": False})
    validate_config_keys(config)
    validate_endpoint_conflicts(config)
    assert config["tracking"]["endpoint_conflicts"] == {**original, "enabled": False}

    for budget in ("max_calls_per_frame", "max_calls_total", "max_calls_per_video_second", "cache_entries"):
        config = _config(**{budget: 0})
        validate_endpoint_conflicts(config)
        assert config["tracking"]["endpoint_conflicts"][budget] == 0


def test_endpoint_controls_are_advanced_and_absent_from_curated_discovery():
    payload = build_describe_payload(cli.command_parser())
    assert not any(path.startswith(PREFIX + ".") for path in payload["config"]["dotted_options"])
    full = _full_config_contract()["dotted_options"]
    for key, value in _config()["tracking"]["endpoint_conflicts"].items():
        if isinstance(value, list):
            for index, item in enumerate(value):
                assert full[f"{PREFIX}.{key}.{index}"]["default"] == item
        else:
            assert full[f"{PREFIX}.{key}"]["default"] == value


def test_duplicate_review_defaults_are_independent_of_tracking_review():
    config = _config()
    assert config["tracking"]["endpoint_conflicts"]["angles"] == [0]
    assert config["tracking"]["endpoint_conflicts"]["recheck_min_frame_gap"] == 1
    assert config["revalidation"]["angles"] == [0, 90, -90]


@pytest.mark.parametrize("angles", [[0], [0, 90, -90], [90], [-90, 0]])
def test_public_loader_accepts_independent_duplicate_review_angles(angles):
    config = load_config(
        DEFAULT_CONFIG_PATH,
        config_overrides={f"{PREFIX}.angles": angles},
        materialize_models=False,
    )
    assert config["tracking"]["endpoint_conflicts"]["angles"] == angles
    assert config["revalidation"]["angles"] == [0, 90, -90]


@pytest.mark.parametrize("angles", [None, [], 0, [True], [0.0], ["0"], [45], [0, 0]])
def test_invalid_duplicate_review_angles_are_rejected(angles):
    with pytest.raises((TypeError, ValueError), match="angles"):
        validate_endpoint_conflicts(_config(angles=angles))


@pytest.mark.parametrize("value", [None, [], False, "enabled"])
def test_endpoint_control_section_must_be_a_mapping(value):
    config = _config()
    config["tracking"]["endpoint_conflicts"] = value
    with pytest.raises(TypeError, match="endpoint_conflicts must be a mapping"):
        validate_endpoint_conflicts(config)


@pytest.mark.parametrize("field,value,error", [
    ("enabled", 1, TypeError),
    ("enabled", "false", TypeError),
    ("nearby_iou", True, TypeError),
    ("nearby_iou", -0.01, ValueError),
    ("nearby_iou", float("nan"), ValueError),
    ("nearby_center_distance", 0, ValueError),
    ("nearby_center_distance", float("inf"), ValueError),
    ("match_min_iou", 1.01, ValueError),
    ("match_max_center_distance", False, TypeError),
    ("match_max_center_distance", -0.1, ValueError),
    ("match_max_area_ratio", 0.99, ValueError),
    ("match_max_area_ratio", float("inf"), ValueError),
    ("match_min_confidence", "0.35", TypeError),
    ("match_min_confidence", -0.1, ValueError),
    ("match_min_margin", 1.01, ValueError),
    ("match_min_margin", float("nan"), ValueError),
    ("equivalence_iou", 0, ValueError),
    ("equivalence_iou", 1.01, ValueError),
    ("equivalence_iou", float("nan"), ValueError),
    ("recheck_min_frame_gap", 0, ValueError),
    ("recheck_min_frame_gap", 1.5, TypeError),
    ("max_calls_per_frame", True, TypeError),
    ("max_calls_per_frame", -1, ValueError),
    ("max_calls_total", -1, ValueError),
    ("max_calls_total", 3.5, TypeError),
    ("max_calls_per_video_second", -0.01, ValueError),
    ("max_calls_per_video_second", False, TypeError),
    ("max_calls_per_video_second", float("inf"), ValueError),
    ("cache_entries", -1, ValueError),
    ("cache_entries", 1.5, TypeError),
])
def test_invalid_endpoint_controls_fail_with_the_field_name(field, value, error):
    with pytest.raises(error, match=field):
        validate_endpoint_conflicts(_config(**{field: value}))


def test_disabled_endpoint_controls_still_reject_invalid_values():
    with pytest.raises(ValueError, match="match_min_margin"):
        validate_endpoint_conflicts(_config(enabled=False, match_min_margin=-1))


def test_endpoint_controls_reject_unknown_and_missing_fields():
    with pytest.raises(ValueError, match="unknown.*endpoint_conflicts.*mispelled"):
        validate_endpoint_conflicts(_config(mispelled=1))
    config = _config()
    del config["tracking"]["endpoint_conflicts"]["equivalence_iou"]
    with pytest.raises(ValueError, match="missing.*equivalence_iou"):
        validate_endpoint_conflicts(config)


@pytest.mark.parametrize("mode", ["all", "blur_only", "exempt"])
def test_endpoint_configuration_does_not_rewrite_the_recognition_policy(mode):
    config = _config()
    config["recognition"]["mode"] = mode
    before = deepcopy(config["recognition"])
    validate_endpoint_conflicts(config)
    assert config["recognition"] == before


def test_public_loader_enforces_endpoint_budget_validation():
    with pytest.raises(ValueError, match="max_calls_total"):
        load_config(
            DEFAULT_CONFIG_PATH,
            config_overrides={f"{PREFIX}.max_calls_total": -1},
            materialize_models=False,
        )
