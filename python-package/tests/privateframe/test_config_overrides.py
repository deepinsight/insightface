from __future__ import annotations

from copy import deepcopy
from inspect import signature
from pathlib import Path

import pytest
import yaml
from insightface.app.privateframe import base_config
from insightface.app.privateframe.base_config import (
    apply_config_overrides,
    validate_config_keys,
    validate_scan_passes,
)
from insightface.app.privateframe.config import load_config
from insightface.app.privateframe.pipeline import (
    analyze_streaming_pipeline,
    run_streaming_pipeline,
)

CONFIG_PATH = (
    Path(base_config.__file__).with_name("configs")
    / "base.yaml"
)


def _raw_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_distribution_contains_only_one_base_yaml():
    assert sorted(path.name for path in CONFIG_PATH.parent.glob("*.yaml")) == [
        "base.yaml"
    ]


def _without_model_materialization(monkeypatch):
    monkeypatch.setattr(base_config, "materialize_model_package", lambda _config: None)
    monkeypatch.setattr(
        base_config,
        "validate_model_package_contracts",
        lambda _config: None,
    )


def test_dotted_overrides_update_scalars_list_indices_and_optional_leaf():
    config = _raw_config()

    apply_config_overrides(
        config,
        {
            "scan.workers": 8,
            "scan.passes.0.input_size": 608,
            "recognition.max_frames_per_track": 4,
        },
    )

    assert config["scan"]["workers"] == 8
    assert config["scan"]["passes"][0]["input_size"] == 608
    assert config["recognition"]["max_frames_per_track"] == 4
    validate_config_keys(config)


def test_dotted_override_creates_schema_approved_missing_mapping():
    config = _raw_config()
    del config["render"]["redaction"]["feather"]

    apply_config_overrides(
        config,
        {"render.redaction.feather.ratio": 0.1},
    )

    assert config["render"]["redaction"]["feather"] == {"ratio": 0.1}
    validate_config_keys(config)


def test_dict_override_deep_merges_but_rate_control_is_atomic():
    config = _raw_config()
    original_gaussian = deepcopy(config["render"]["redaction"]["gaussian"])

    apply_config_overrides(
        config,
        {
            "render.redaction.gaussian": {"sigma": 1.5},
            "render.video_output.rate_control": {
                "mode": "vbr",
                "bitrate": "8M",
                "max_bitrate": "10M",
            },
        },
    )

    assert config["render"]["redaction"]["gaussian"] == {
        **original_gaussian,
        "sigma": 1.5,
    }
    assert config["render"]["video_output"]["rate_control"] == {
        "mode": "vbr",
        "bitrate": "8M",
        "max_bitrate": "10M",
    }
    validate_config_keys(config)


def test_list_terminal_override_replaces_the_list():
    config = _raw_config()

    apply_config_overrides(config, {"scan.passes.0.angles": [0]})

    assert config["scan"]["passes"][0]["angles"] == [0]


@pytest.mark.parametrize(
    "path",
    [
        "scan.unknown",
        "runtime.scrfd_static_shape_session",
        "runtime.providers",
        "runtime.resolved_provider",
        "models.manifest_path",
        "models.detection.path",
        "scan",
        "runtime",
        "render",
        "recognition",
        "schema_version",
        "base_config",
    ],
)
def test_dotted_override_rejects_unknown_generated_or_internal_path(path):
    with pytest.raises(ValueError, match="configuration override path"):
        apply_config_overrides(_raw_config(), {path: 1})


@pytest.mark.parametrize(
    "path",
    [
        "scan.passes.-1.input_size",
        "scan.passes.00.input_size",
        "scan.passes.first.input_size",
    ],
)
def test_dotted_override_rejects_invalid_list_index(path):
    with pytest.raises(ValueError, match="list index"):
        apply_config_overrides(_raw_config(), {path: 640})


def test_dotted_override_rejects_out_of_range_list_index():
    with pytest.raises(IndexError, match="out of range"):
        apply_config_overrides(
            _raw_config(),
            {"scan.passes.99.input_size": 640},
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"scan.passes": [], "scan.passes.0.input_size": 640},
        {"render.redaction": {}, "render.redaction.method": "mosaic"},
    ],
)
def test_python_api_rejects_parent_child_override_paths(overrides):
    with pytest.raises(ValueError, match="paths overlap"):
        apply_config_overrides(_raw_config(), overrides)


@pytest.mark.parametrize("value", [float("nan"), {1: "value"}, {"value"}])
def test_python_api_rejects_non_json_override_value(value):
    with pytest.raises((TypeError, ValueError), match="configuration override"):
        apply_config_overrides(_raw_config(), {"scan.workers": value})


def test_wrong_override_value_type_reaches_normal_strict_validator():
    config = _raw_config()
    apply_config_overrides(config, {"scan.passes.0.input_size": "640"})

    with pytest.raises(TypeError, match="input_size must be an integer"):
        validate_scan_passes(config)


def test_override_layer_order_is_base_then_derived_then_dotted(
    monkeypatch,
    tmp_path,
):
    _without_model_materialization(monkeypatch)
    derived = tmp_path / "derived.yaml"
    derived.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "base_config": str(CONFIG_PATH),
                "scan": {"max_analysis_fps": 15},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(
        derived,
        config_overrides={"scan.max_analysis_fps": 24},
        config_override_root=tmp_path,
    )

    assert config["scan"]["max_analysis_fps"] == 24.0


def test_custom_yaml_implicitly_inherits_packaged_base(monkeypatch, tmp_path):
    _without_model_materialization(monkeypatch)
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "scan": {"max_analysis_fps": 15},
                "render": {"redaction": {"method": "mosaic"}},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(custom)

    base = _raw_config()
    assert config["scan"]["max_analysis_fps"] == 15.0
    assert config["scan"]["workers"] == base["scan"]["workers"]
    assert config["render"]["redaction"]["method"] == "mosaic"
    assert config["models"]["name"] == base["models"]["name"]


def test_implicit_base_layer_order_is_base_then_custom_then_dotted(
    monkeypatch,
    tmp_path,
):
    _without_model_materialization(monkeypatch)
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "scan": {"max_analysis_fps": 15},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(
        custom,
        config_overrides={"scan.max_analysis_fps": 24},
        config_override_root=tmp_path,
    )

    assert config["scan"]["max_analysis_fps"] == 24.0


def test_audio_bitrate_override_accepts_integer_form():
    config = load_config(
        CONFIG_PATH,
        config_overrides={"render.video_output.audio.bitrate": 192_000},
        materialize_models=False,
    )

    assert config["render"]["video_output"]["audio"]["bitrate"] == 192_000


def test_models_root_is_a_public_dotted_override(tmp_path):
    root = tmp_path / "shared-insightface-root"

    config = load_config(
        CONFIG_PATH,
        config_overrides={"models.root": str(root)},
        config_override_root=tmp_path,
        materialize_models=False,
    )

    assert config["models"]["root"] == str(root)


@pytest.mark.parametrize("value", [None, 1, True, [], {}])
def test_models_root_dotted_override_requires_a_string(tmp_path, value):
    with pytest.raises(TypeError, match="models.root must be a string"):
        load_config(
            CONFIG_PATH,
            config_overrides={"models.root": value},
            config_override_root=tmp_path,
            materialize_models=False,
        )


@pytest.mark.parametrize("value", ["", "   "])
def test_models_root_dotted_override_requires_a_non_empty_path(tmp_path, value):
    with pytest.raises(ValueError, match="models.root must be a non-empty path"):
        load_config(
            CONFIG_PATH,
            config_overrides={"models.root": value},
            config_override_root=tmp_path,
            materialize_models=False,
        )


def test_explicit_relative_base_config_takes_precedence_over_packaged_base(
    monkeypatch,
    tmp_path,
):
    _without_model_materialization(monkeypatch)
    explicit_base = _raw_config()
    explicit_base["scan"]["workers"] = 7
    (tmp_path / "parent.yaml").write_text(
        yaml.safe_dump(explicit_base),
        encoding="utf-8",
    )
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "base_config": "parent.yaml",
                "scan": {"max_analysis_fps": 15},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(custom)

    assert config["scan"]["workers"] == 7
    assert config["scan"]["max_analysis_fps"] == 15.0


def test_complete_custom_yaml_remains_supported(monkeypatch, tmp_path):
    _without_model_materialization(monkeypatch)
    complete = _raw_config()
    complete["scan"]["workers"] = 9
    custom = tmp_path / "complete.yaml"
    custom.write_text(yaml.safe_dump(complete), encoding="utf-8")

    config = load_config(custom)

    assert config["scan"]["workers"] == 9
    assert config["render"]["video_output"]["preset"] == "medium"


@pytest.mark.parametrize("document", [None, [], "invalid", {"scan": {}}])
def test_implicit_custom_yaml_requires_current_schema_document(
    monkeypatch,
    tmp_path,
    document,
):
    _without_model_materialization(monkeypatch)
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="custom ONNX config.*schema_version: 1"):
        load_config(custom)


def test_implicit_custom_yaml_rejects_unknown_keys_before_merging(
    monkeypatch,
    tmp_path,
):
    _without_model_materialization(monkeypatch)
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "scan": {"unknown_setting": 1},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown configuration setting"):
        load_config(custom)


@pytest.mark.parametrize("base_value", [None, "", "   ", 1, True, []])
def test_explicit_base_config_must_be_a_non_empty_path(
    monkeypatch,
    tmp_path,
    base_value,
):
    _without_model_materialization(monkeypatch)
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "base_config": base_value,
                "scan": {"max_analysis_fps": 15},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="base_config must be a non-empty path"):
        load_config(custom)


@pytest.mark.parametrize("legacy_key", ["performance_mode", "frame_stride"])
def test_removed_scan_sampling_key_is_rejected_as_unknown(
    monkeypatch,
    tmp_path,
    legacy_key,
):
    _without_model_materialization(monkeypatch)

    with pytest.raises(
        ValueError,
        match=rf"unknown configuration override path: scan\.{legacy_key}",
    ):
        load_config(
            CONFIG_PATH,
            config_overrides={f"scan.{legacy_key}": 3},
            config_override_root=tmp_path,
        )


def test_provider_and_artifact_level_use_uniform_dotted_fields(
    monkeypatch,
    tmp_path,
):
    _without_model_materialization(monkeypatch)

    config = load_config(
        CONFIG_PATH,
        config_overrides={
            "runtime.provider": "CPUExecutionProvider",
            "runtime.scrfd_static_shape_sessions": False,
            "output.artifacts_level": "audit",
        },
        config_override_root=tmp_path,
    )

    assert config["runtime"]["resolved_provider"] == "CPUExecutionProvider"
    assert config["runtime"]["scrfd_static_shape_sessions"] is False
    assert config["output"]["artifacts_level"] == "audit"


def test_model_package_is_selected_by_override_on_the_single_base(
    monkeypatch,
    tmp_path,
):
    _without_model_materialization(monkeypatch)

    config = load_config(
        CONFIG_PATH,
        config_overrides={"models.name": "raccoon_l"},
        config_override_root=tmp_path,
    )

    assert config["models"]["name"] == "raccoon_l"


def test_scrfd_static_shape_sessions_defaults_to_true(
    monkeypatch,
):
    _without_model_materialization(monkeypatch)

    config = load_config(
        CONFIG_PATH,
    )

    assert config["runtime"]["scrfd_static_shape_sessions"] is True


def test_runtime_provider_dotted_value_is_strictly_a_string(
    monkeypatch,
    tmp_path,
):
    _without_model_materialization(monkeypatch)

    with pytest.raises(TypeError, match="runtime.provider must be a string"):
        load_config(
            CONFIG_PATH,
            config_overrides={"runtime.provider": 1},
            config_override_root=tmp_path,
        )


@pytest.mark.parametrize("value", [0, 1, "false", None, [], {}])
def test_scrfd_static_shape_sessions_is_strictly_boolean(
    monkeypatch,
    tmp_path,
    value,
):
    _without_model_materialization(monkeypatch)

    with pytest.raises(
        TypeError,
        match="runtime.scrfd_static_shape_sessions must be boolean",
    ):
        load_config(
            CONFIG_PATH,
            config_overrides={
                "runtime.scrfd_static_shape_sessions": value,
            },
            config_override_root=tmp_path,
        )


def test_high_level_python_apis_only_expose_uniform_config_overrides():
    removed = {
        "provider",
        "artifacts_level",
        "frame_stride",
        "max_analysis_fps",
        "performance_mode",
    }

    assert not removed & set(signature(load_config).parameters)
    assert not removed & set(signature(analyze_streaming_pipeline).parameters)
    assert not removed & set(signature(run_streaming_pipeline).parameters)
    assert not hasattr(base_config, "load_config")


def test_generic_reference_path_uses_explicit_override_root(monkeypatch, tmp_path):
    _without_model_materialization(monkeypatch)
    reference = tmp_path / "reference"
    reference.mkdir()

    config = load_config(
        CONFIG_PATH,
        config_overrides={
            "recognition.mode": "exempt",
            "recognition.reference_dir": "reference",
        },
        config_override_root=tmp_path,
    )

    assert config["recognition"]["reference_dir"] == str(reference.resolve())


def test_implicit_custom_yaml_reference_root_is_unchanged_by_other_cli_override(
    monkeypatch,
    tmp_path,
):
    _without_model_materialization(monkeypatch)
    derived_root = tmp_path / "derived"
    cli_root = tmp_path / "cli"
    derived_root.mkdir()
    cli_root.mkdir()
    reference = derived_root / "reference"
    reference.mkdir()
    derived = derived_root / "config.yaml"
    derived.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "recognition": {
                    "mode": "exempt",
                    "reference_dir": "reference",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(
        derived,
        config_overrides={"scan.workers": 8},
        config_override_root=cli_root,
    )

    assert config["recognition"]["reference_dir"] == str(reference.resolve())


def test_cli_reference_override_takes_its_own_root_over_derived_yaml(
    monkeypatch,
    tmp_path,
):
    _without_model_materialization(monkeypatch)
    derived_root = tmp_path / "derived"
    cli_root = tmp_path / "cli"
    derived_root.mkdir()
    cli_root.mkdir()
    (derived_root / "yaml-reference").mkdir()
    cli_reference = cli_root / "cli-reference"
    cli_reference.mkdir()
    derived = derived_root / "config.yaml"
    derived.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "base_config": str(CONFIG_PATH),
                "recognition": {
                    "mode": "exempt",
                    "reference_dir": "yaml-reference",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(
        derived,
        config_overrides={"recognition.reference_dir": "cli-reference"},
        config_override_root=cli_root,
    )

    assert config["recognition"]["reference_dir"] == str(cli_reference.resolve())
