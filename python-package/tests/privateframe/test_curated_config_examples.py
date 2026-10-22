"""Advertised configuration examples must work against the real Base overlay."""

import pytest

from insightface.app.privateframe import cli
from insightface.app.privateframe.base_config import DEFAULT_CONFIG_PATH
from insightface.app.privateframe.cli_contract import build_describe_payload
from insightface.app.privateframe.config import load_config


@pytest.mark.parametrize("index,mode", [(0, "vbr"), (1, "crf")])
def test_rate_control_mode_examples_replace_incompatible_defaults(index, mode):
    description = build_describe_payload(cli.command_parser())
    option = description["config"]["dotted_options"]["render.video_output.rate_control.mode"]
    args = option["examples"][index]["argv"]
    clean, overrides = cli._parse_dotted_config_overrides(args)
    assert clean == []
    config = load_config(DEFAULT_CONFIG_PATH, config_overrides=overrides, materialize_models=False)
    rate = config["render"]["video_output"]["rate_control"]
    assert rate["mode"] == mode
    assert set(rate) == ({"mode", "bitrate"} if mode == "vbr" else {"mode", "quality"})


def test_quality_leaf_override_keeps_its_existing_mode():
    _, overrides = cli._parse_dotted_config_overrides([
        "--render.video_output.rate_control.quality", "24",
    ])
    config = load_config(DEFAULT_CONFIG_PATH, config_overrides=overrides, materialize_models=False)
    assert config["render"]["video_output"]["rate_control"] == {"mode": "crf", "quality": 24}


@pytest.mark.parametrize("mode", ["blur_only", "exempt"])
def test_photo_selection_cli_example_needs_only_mode_and_flat_folder(tmp_path, mode):
    _, overrides = cli._parse_dotted_config_overrides([
        "--recognition.mode", mode,
        "--recognition.reference_dir", str(tmp_path),
    ])
    config = load_config(DEFAULT_CONFIG_PATH, config_overrides=overrides, materialize_models=False)
    recognition = config["recognition"]
    assert recognition["mode"] == mode
    assert recognition["reference_dir"] == str(tmp_path)
    assert recognition["unknown_action"] == "auto"
    assert "gallery_dir" not in recognition
    assert "target_persons" not in recognition
