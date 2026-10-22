"""Keep the offline reference complete and synchronized with supported config."""

from __future__ import annotations

import html
import re
from copy import deepcopy

import pytest
import yaml

from insightface.app.privateframe import cli
from insightface.app.privateframe.base_config import (
    DEFAULT_CONFIG_PATH,
    apply_config_overrides,
)
from insightface.app.privateframe.cli_contract import _full_config_contract
from insightface.app.privateframe.config import validate_video_output
from insightface.app.privateframe.config_reference import (
    DEFAULT_REFERENCE_PATH,
    contract_fingerprint,
    default_text,
    description_for,
    generate_reference,
    main,
)


@pytest.fixture(scope="module")
def contract():
    return _full_config_contract()


def test_bundled_reference_is_current_and_lists_every_public_path(contract):
    text = DEFAULT_REFERENCE_PATH.read_text(encoding="utf-8")
    assert text == generate_reference(
        contract
    ), "Regenerate with python -m insightface.app.privateframe.config_reference"
    rows = {}
    for line in text.splitlines():
        match = re.match(r"^\| `([^`]+)` \|", line)
        if match:
            assert match[1] not in rows
            rows[match[1]] = line
    for path, spec in contract["dotted_options"].items():
        assert path in rows
        assert default_text(spec) in rows[path]
        assert (
            html.escape(description_for(path, spec), quote=False).replace("|", "\\|")
            in rows[path]
        )
    assert len(set(rows) & set(contract["dotted_options"])) == len(
        contract["dotted_options"]
    )


def test_reference_explains_optional_fields_arrays_and_compatibility(contract):
    text = generate_reference(contract)
    for phrase in (
        "schema_version: 1",
        "base_config",
        "Arrays supplied in YAML replace the entire array",
        "indexed CLI overrides modify an existing element",
        "revalidation.passes.0.input_size",
        "Not set (optional)",
        "Compatibility-only diagnostics",
        "render.debug_line_thickness",
        "render.video_output.audio.debug",
        "--debug PATH",
        "PyAV aac accepts existing AAC only",
        "bitrate",
    ):
        assert phrase in text
    for path in (
        "revalidation.passes",
        "render.redaction.feather.ratio",
        "render.video_output.rate_control.bitrate",
    ):
        assert contract["dotted_options"][path]["has_default"] is False
    assert "Runtime-computed fields" in text


def test_reference_generation_is_independent_of_install_location(contract):
    installed = deepcopy(contract)
    installed["default_path"] = "/unrelated/install/site-packages/base.yaml"
    assert contract_fingerprint(installed) == contract_fingerprint(contract)
    assert generate_reference(installed) == generate_reference(contract)
    assert installed["default_path"] not in generate_reference(installed)


def test_reference_explains_endpoint_evidence_scope_and_budget_semantics(contract):
    text = generate_reference(contract)
    for phrase in (
        "tracking.endpoint_conflicts.enabled",
        "only in recognition.mode=all",
        "independently measured local face with a shared same-frame detection",
        "Predicted-box to target-local matching uses revalidation.match_* geometry",
        "including IoU OR containment",
        "both target-local selection and shared-instance matching",
        "not a cutoff for ignoring other returned faces",
        "weaker second-face evidence can still block removal",
        "An unmatched candidate is not automatically a duplicate",
        "incomplete review cannot establish a duplicate",
        "min(max_calls_total, ceil(source duration in seconds * max_calls_per_video_second))",
        "not by wall-clock speed",
        "Entries cannot substitute for another frame's evidence",
    ):
        assert phrase in text


def test_reference_explains_flat_photo_selection_and_unknown_policy(contract):
    text = generate_reference(contract)
    for phrase in (
        "recognition.reference_dir",
        "recognition.unknown_action",
        "largest detected face",
        "no fallback to smaller faces",
        "no usable reference faces is an error",
        "uncertain or unmatched people remain visible in blur_only",
        "stored in analysis JSON",
        "inference errors stop processing",
    ):
        assert phrase in text
    assert "recognition.gallery_dir" not in text
    assert "recognition.target_persons" not in text


def test_changed_default_changes_reference_and_fingerprint(contract):
    changed = deepcopy(contract)
    changed["defaults"]["scan"]["max_analysis_fps"] = 30
    changed["dotted_options"]["scan.max_analysis_fps"]["default"] = 30
    assert contract_fingerprint(changed) != contract_fingerprint(contract)
    assert generate_reference(changed) != generate_reference(contract)


def test_undocumented_new_parameter_requires_a_meaningful_description():
    with pytest.raises(ValueError, match="Missing configuration reference description"):
        description_for("new_section.unknown_field", {"default": 1})
    assert (
        description_for(
            "new_section.new_field",
            {"description": "Limit retained history by source frames."},
        )
        == "Limit retained history by source frames."
    )


def test_rate_control_example_replaces_incompatible_siblings(contract):
    assert "--render.video_output.rate_control '{mode: vbr, bitrate: 4M}'" in (
        generate_reference(contract)
    )
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    _clean, overrides = cli._parse_dotted_config_overrides(
        ["--render.video_output.rate_control", "{mode: vbr, bitrate: 4M}"]
    )
    apply_config_overrides(config, overrides)
    assert "quality" not in config["render"]["video_output"]["rate_control"]
    validate_video_output(config["render"]["video_output"])

    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    apply_config_overrides(
        config,
        {
            "render.video_output.rate_control.mode": "vbr",
            "render.video_output.rate_control.bitrate": "4M",
        },
    )
    with pytest.raises(ValueError, match="vbr rate control has invalid settings"):
        validate_video_output(config["render"]["video_output"])


def test_array_parent_and_index_cannot_be_overridden_in_the_same_command(contract):
    assert "overlapping parent/child CLI paths are rejected" in generate_reference(
        contract
    )
    with pytest.raises(ValueError, match="configuration override paths overlap"):
        cli._parse_dotted_config_overrides(
            ["--scan.passes", "[]", "--scan.passes.0.input_size", "128"]
        )
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    apply_config_overrides(config, {"scan.passes.0.input_size": 128})
    assert config["scan"]["passes"][0]["input_size"] == 128


def test_check_never_rewrites_current_stale_or_missing_reference(tmp_path, capsys):
    path = tmp_path / "reference.md"
    assert main(["--output", str(path)]) == 0
    original = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    assert main(["--check", "--output", str(path)]) == 0
    assert path.stat().st_mtime_ns == mtime
    assert path.read_bytes() == original
    path.write_text("stale reference\n", encoding="utf-8")
    stale_mtime = path.stat().st_mtime_ns
    assert main(["--check", "--output", str(path)]) == 1
    assert path.read_text() == "stale reference\n"
    assert path.stat().st_mtime_ns == stale_mtime
    missing = tmp_path / "missing.md"
    assert main(["--check", "--output", str(missing)]) == 1
    assert not missing.exists()
    assert "stale" in capsys.readouterr().err
