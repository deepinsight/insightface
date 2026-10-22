from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from insightface.app.privateframe import base_config, pipeline

CONFIG_PATH = (
    Path(base_config.__file__).with_name("configs")
    / "base.yaml"
)


def _analysis_result():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    defaults = deepcopy(raw["render"])
    defaults["recognition_policy"] = {
        "mode": "all",
        "unknown_action": "blur",
    }
    return {
        "render_defaults": defaults,
        "recognition": {"enabled": False, "reason": "policy_all"},
    }


def test_default_video_encoding_balances_quality_and_file_size():
    output = _analysis_result()["render_defaults"]["video_output"]
    assert output["preset"] == "medium"
    assert output["rate_control"] == {"mode": "crf", "quality": 23}


def test_render_dotted_override_is_above_render_config_yaml(tmp_path):
    overlay = tmp_path / "render.yaml"
    overlay.write_text(
        yaml.safe_dump({"render": {"redaction": {"box_scale": 1.5}}}),
        encoding="utf-8",
    )

    settings, _digest = pipeline._render_settings(
        _analysis_result(),
        overlay,
        {"render.redaction.box_scale": 1.25},
    )

    assert settings["redaction"]["box_scale"] == 1.25


def test_single_method_override_uses_mosaic_defaults_from_base():
    settings, _digest = pipeline._render_settings(
        _analysis_result(),
        None,
        {"render.redaction.method": "mosaic"},
    )

    assert settings["redaction"]["method"] == "mosaic"
    assert settings["redaction"]["mosaic"] == {
        "block_size_ratio": 0.12,
        "min_block_size": 8,
    }


def test_old_result_can_add_pyramid_settings_with_dotted_overrides():
    result = _analysis_result()
    result["render_defaults"]["redaction"]["gaussian"].pop("algorithm")
    result["render_defaults"]["redaction"]["gaussian"].pop("max_side")

    settings, _digest = pipeline._render_settings(
        result,
        None,
        {
            "render.redaction.gaussian.algorithm": "pyramid",
            "render.redaction.gaussian.max_side": 64,
        },
    )

    assert settings["redaction"]["gaussian"]["algorithm"] == "pyramid"
    assert settings["redaction"]["gaussian"]["max_side"] == 64


def test_render_dotted_rate_control_replaces_the_whole_mapping():
    settings, _digest = pipeline._render_settings(
        _analysis_result(),
        None,
        {
            "render.video_output.rate_control": {
                "mode": "vbr",
                "bitrate": "8M",
                "max_bitrate": "10M",
            }
        },
    )

    assert settings["video_output"]["rate_control"] == {
        "mode": "vbr",
        "bitrate": "8M",
        "max_bitrate": "10M",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"scan.workers": 8},
        {"recognition.mode": "all"},
        {"render": {"redaction": {"method": "mosaic"}}},
        {"render": {"recognition_policy": {"mode": "all"}}},
    ],
)
def test_render_api_rejects_non_render_config_overrides(overrides):
    with pytest.raises(ValueError, match=r"render\.\*"):
        pipeline._render_settings(_analysis_result(), None, overrides)


def test_render_api_rejects_unknown_render_config_path():
    with pytest.raises(ValueError, match="configuration override path"):
        pipeline._render_settings(
            _analysis_result(),
            None,
            {"render.unknown": True},
        )


def test_process_reapplies_only_render_subset_after_render_config(monkeypatch, tmp_path):
    captured = {}

    def analyze(**kwargs):
        captured["analysis"] = kwargs
        return {"phase": "analysis"}

    def render(**kwargs):
        captured["render"] = kwargs
        return {"phase": "render"}

    monkeypatch.setattr(pipeline, "analyze_streaming_pipeline", analyze)
    monkeypatch.setattr(pipeline, "render_streaming_artifacts", render)

    result = pipeline.run_streaming_pipeline(
        config_path=tmp_path / "config.yaml",
        input_path=tmp_path / "input.mp4",
        workdir=tmp_path / "work",
        debug_path=None,
        redacted_path=tmp_path / "redacted.mp4",
        render_config=tmp_path / "render.yaml",
        config_overrides={
            "scan.workers": 8,
            "render.redaction.box_scale": 1.25,
        },
        config_override_root=tmp_path,
    )

    assert result == {
        "analysis": {"phase": "analysis"},
        "render": {"phase": "render"},
    }
    assert captured["analysis"]["config_overrides"] == {
        "scan.workers": 8,
        "render.redaction.box_scale": 1.25,
    }
    assert captured["render"]["config_overrides"] == {
        "render.redaction.box_scale": 1.25
    }
    assert captured["render"]["render_config"] == tmp_path / "render.yaml"
