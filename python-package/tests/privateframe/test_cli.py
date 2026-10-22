from __future__ import annotations

import json
from pathlib import Path

import pytest
from insightface.app.privateframe import cli


def _status(capsys, command: str) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.out.endswith("\n")
    assert captured.out.count("\n") == 1
    payload = json.loads(captured.out)
    assert payload["status_schema_version"] == 1
    assert payload["command"] == command
    return payload


def _assert_argument_error(exit_code: int, capsys, command: str) -> None:
    assert exit_code == 2
    payload = _status(capsys, command)
    assert payload["ok"] is False
    assert payload["error"]["stage"] == "arguments"


def _analyze_args(*extra: str) -> list[str]:
    return [
        "analyze",
        "--config",
        "config.yaml",
        "--input",
        "input.mp4",
        "--workdir",
        "work",
        *extra,
    ]


def _process_args(*extra: str) -> list[str]:
    return [
        "process",
        "--config",
        "config.yaml",
        "--input",
        "input.mp4",
        "--workdir",
        "work",
        *extra,
    ]


@pytest.mark.parametrize(
    "args_factory",
    [_analyze_args, _process_args],
    ids=["analyze", "process"],
)
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--provider", "CPUExecutionProvider"),
        ("--artifacts-level", "audit"),
        ("--performance-mode", "fast"),
        ("--stride", "2"),
    ],
)
def test_cli_rejects_removed_config_shortcut_flags(
    args_factory,
    option,
    value,
    capsys,
):
    exit_code = cli.main(args_factory(option, value, "--dry-run"))

    _assert_argument_error(exit_code, capsys, args_factory()[0])


def test_analysis_help_mentions_dotted_config_overrides(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.command_parser().parse_args(["analyze", "--help"])

    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--section.field VALUE" in output
    assert "--scan.max_analysis_fps 15" in output
    assert "configs/base.yaml" in output
    assert "custom files inherit it" in output


@pytest.mark.parametrize("command", ["analyze", "process"])
def test_analysis_commands_default_to_packaged_base_config(command):
    args = cli.command_parser().parse_args(
        [command, "--input", "input.mp4", "--output-dir", "exports"]
    )

    config_path = Path(args.config)
    assert config_path == cli._DEFAULT_CONFIG_PATH
    assert config_path.is_file()


@pytest.mark.parametrize(
    ("command", "pipeline_name", "result"),
    [
        ("analyze", "analyze_streaming_pipeline", {"phase": "analysis"}),
        ("process", "run_streaming_pipeline", {"analysis": {}, "render": {}}),
    ],
)
def test_analysis_commands_forward_packaged_default_config(
    command,
    pipeline_name,
    result,
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def pipeline(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(cli, pipeline_name, pipeline)

    exit_code = cli.main(
        [
            command,
            "--input",
            "input.mp4",
            "--output-dir",
            str(tmp_path / "exports"),
        ]
    )

    assert exit_code == 0
    assert Path(captured["config_path"]) == cli._DEFAULT_CONFIG_PATH
    payload = _status(capsys, command)
    assert payload["ok"] is True
    assert payload["summary"] == {}


@pytest.mark.parametrize(
    ("command", "pipeline_name", "result", "expected_timings", "expected_summary"),
    [
        (
            "analyze",
            "analyze_streaming_pipeline",
            {
                "provider": "CoreMLExecutionProvider",
                "frame_count": 120,
                "detections": 18,
                "tracks": 5,
                "accepted_tracks": 4,
                "observations": 96,
                "timings": {
                    "analysis_seconds": 1.5,
                    "artifact_seconds": 0.4,
                    "total_seconds": 1.9,
                },
                "recognition": {"tracks": {"private": "diagnostics"}},
                "cache": {"peak_frames": 300},
                "input": "/private/source.mp4",
                "development_report": "/private/result.dev.json",
            },
            {"total_seconds": 1.9},
            {
                "frame_count": 120,
                "face_tracks": 4,
                "face_regions": 96,
            },
        ),
        (
            "render",
            "render_streaming_artifacts",
            {
                "frame_count": 120,
                "observations": 96,
                "blurred_observations": 80,
                "kept_observations": 16,
                "fail_safe_observations": 3,
                "seconds": 2.5,
                "outputs": [
                    {
                        "path": "/private/output.mp4",
                        "sha256": "not-public",
                        "metadata": {"codec": "h264"},
                    }
                ],
                "render_settings_sha256": "not-public",
                "recognition_policy": {"private": "diagnostics"},
            },
            {"total_seconds": 2.5},
            {
                "frame_count": 120,
                "face_regions": 96,
                "redacted_face_regions": 80,
                "kept_face_regions": 16,
            },
        ),
        (
            "process",
            "run_streaming_pipeline",
            {
                "analysis": {
                    "provider": "CUDAExecutionProvider",
                    "frame_count": 120,
                    "detections": 18,
                    "tracks": 5,
                    "accepted_tracks": 4,
                    "observations": 96,
                    "timings": {
                        "analysis_seconds": 1.5,
                        "artifact_seconds": 0.4,
                        "total_seconds": 1.9,
                    },
                    "profile": {"private": "diagnostics"},
                },
                "render": {
                    "frame_count": 120,
                    "observations": 96,
                    "blurred_observations": 80,
                    "kept_observations": 16,
                    "fail_safe_observations": 3,
                    "seconds": 2.5,
                    "outputs": [{"sha256": "not-public"}],
                },
            },
            {"total_seconds": 4.4},
            {
                "frame_count": 120,
                "face_tracks": 4,
                "face_regions": 96,
                "redacted_face_regions": 80,
                "kept_face_regions": 16,
            },
        ),
    ],
)
def test_success_stdout_exposes_only_stable_metrics(
    command,
    pipeline_name,
    result,
    expected_timings,
    expected_summary,
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(cli, pipeline_name, lambda **_kwargs: result)

    exit_code = cli.main(
        [
            command,
            "--input",
            "input.mp4",
            "--output-dir",
            str(tmp_path / "exports"),
            "--progress",
            "none",
        ]
    )

    assert exit_code == 0
    payload = _status(capsys, command)
    assert set(payload) == {
        "status_schema_version",
        "ok",
        "command",
        "artifacts",
        "runtime",
        "timings",
        "summary",
    }
    assert payload["timings"] == expected_timings
    assert payload["summary"] == expected_summary
    assert payload["runtime"] == {
        "provider": (
            "CoreMLExecutionProvider"
            if command == "analyze"
            else "CUDAExecutionProvider" if command == "process" else None
        )
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    for private_value in (
        "sha256",
        "artifact_seconds",
        "development_report",
        "recognition_policy",
    ):
        assert private_value not in serialized
    assert not {
        "recognition",
        "cache",
        "profile",
        "outputs",
        "fail_safe_observations",
        "detections",
        "tracks",
    }.intersection(payload["summary"])


@pytest.mark.parametrize("command", ["analyze", "process"])
def test_analysis_commands_preserve_explicit_config(command):
    args = cli.command_parser().parse_args(
        [
            command,
            "--config",
            "custom.yaml",
            "--input",
            "input.mp4",
            "--output-dir",
            "exports",
        ]
    )

    assert args.config == "custom.yaml"


@pytest.mark.parametrize("command", ["render", "process"])
def test_user_help_hides_internal_debug_video_output(command, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.command_parser().parse_args([command, "--help"])

    assert raised.value.code == 0
    assert "--debug" not in capsys.readouterr().out


def test_cli_forwards_config_fields_without_specialized_shortcuts(
    monkeypatch,
    capsys,
):
    captured = {}

    def analyze(**kwargs):
        captured.update(kwargs)
        return {"phase": "analysis"}

    monkeypatch.setattr(cli, "analyze_streaming_pipeline", analyze)

    exit_code = cli.main(
        _analyze_args(
            "--models.name",
            "raccoon_l",
            "--models.root",
            "/srv/insightface",
            "--runtime.provider",
            "CPUExecutionProvider",
            "--runtime.scrfd_static_shape_sessions",
            "false",
            "--output.artifacts_level=audit",
            "--scan.max_analysis_fps=15",
        )
    )

    assert exit_code == 0
    assert captured["config_overrides"] == {
        "models.name": "raccoon_l",
        "models.root": "/srv/insightface",
        "runtime.provider": "CPUExecutionProvider",
        "runtime.scrfd_static_shape_sessions": False,
        "output.artifacts_level": "audit",
        "scan.max_analysis_fps": 15,
    }
    assert _status(capsys, "analyze")["ok"] is True


def test_cli_extracts_yaml_typed_dotted_overrides_in_both_forms():
    clean, overrides = cli._parse_dotted_config_overrides(
        _analyze_args(
            "--scan.workers",
            "8",
            "--scan.max_analysis_fps=29.97",
            "--scan.passes.0.angles=[0, 90]",
            "--render.redaction.feather={enabled: true, ratio: 0.1}",
        )
    )

    assert clean == _analyze_args()
    assert overrides == {
        "scan.workers": 8,
        "scan.max_analysis_fps": 29.97,
        "scan.passes.0.angles": [0, 90],
        "render.redaction.feather": {"enabled": True, "ratio": 0.1},
    }


def test_cli_forwards_dotted_overrides_and_invocation_root(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def analyze(**kwargs):
        captured.update(kwargs)
        return {"phase": "analysis"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "analyze_streaming_pipeline", analyze)

    exit_code = cli.main(
        _analyze_args(
            "--scan.workers",
            "8",
            "--recognition.unknown_action=keep",
        )
    )

    assert exit_code == 0
    assert captured["config_overrides"] == {
        "scan.workers": 8,
        "recognition.unknown_action": "keep",
    }
    assert captured["config_override_root"] == Path(tmp_path)
    assert _status(capsys, "analyze")["ok"] is True


def test_process_forwards_dotted_overrides(monkeypatch, tmp_path, capsys):
    captured = {}

    def process(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "render": {}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_streaming_pipeline", process)

    exit_code = cli.main(
        [
            "process",
            "--config",
            "config.yaml",
            "--input",
            "input.mp4",
            "--workdir",
            "work",
            "--redacted",
            "output.mp4",
            "--scan.pipeline_depth=4",
        ]
    )

    assert exit_code == 0
    assert captured["config_overrides"] == {"scan.pipeline_depth": 4}
    assert captured["config_override_root"] == Path(tmp_path)
    assert _status(capsys, "process")["ok"] is True


def test_render_rejects_non_render_dotted_config_override(capsys):
    exit_code = cli.main(
        [
            "render",
            "--input",
            "input.mp4",
            "--result",
            "input_privateframe.json",
            "--redacted",
            "output.mp4",
            "--scan.workers",
            "8",
        ]
    )

    _assert_argument_error(exit_code, capsys, "render")


def test_render_forwards_render_dotted_override(monkeypatch, tmp_path, capsys):
    captured = {}

    def render(**kwargs):
        captured.update(kwargs)
        return {"phase": "render"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "render_streaming_artifacts", render)

    exit_code = cli.main(
        [
            "render",
            "--input",
            "input.mp4",
            "--result",
            "input_privateframe.json",
            "--redacted",
            "output.mp4",
            "--render.redaction.method=mosaic",
        ]
    )

    assert exit_code == 0
    assert captured["config_overrides"] == {
        "render.redaction.method": "mosaic"
    }
    assert _status(capsys, "render")["ok"] is True


@pytest.mark.parametrize(
    "args",
    [
        ("--scan.workers", "8", "--scan.workers=9"),
        (
            "--render.redaction",
            "{}",
            "--render.redaction.method=gaussian",
        ),
        ("--scan.passes", "[]", "--scan.passes.0.input_size=640"),
    ],
)
def test_cli_rejects_duplicate_or_parent_child_config_paths(args, capsys):
    exit_code = cli.main(_analyze_args(*args, "--dry-run"))

    _assert_argument_error(exit_code, capsys, "analyze")


@pytest.mark.parametrize(
    "option",
    [
        "--scan.unknown=8",
        "--runtime.providers=[]",
        "--models.manifest_path=x",
        "--schema_version=2",
        "--scan.passes.-1.input_size=640",
        "--scan.passes.00.input_size=640",
    ],
)
def test_cli_rejects_unknown_internal_or_invalid_dotted_path_in_dry_run(
    option,
    capsys,
):
    exit_code = cli.main(_analyze_args(option, "--dry-run"))

    _assert_argument_error(exit_code, capsys, "analyze")


@pytest.mark.parametrize(
    "raw_value",
    [
        ".nan",
        ".inf",
        "2026-08-29",
        "!!set {one: null}",
        "!!binary YWJj",
        "{1: value}",
    ],
)
def test_cli_rejects_non_json_yaml_override_values(raw_value, capsys):
    exit_code = cli.main(
        _analyze_args("--scan.workers", raw_value, "--dry-run")
    )

    _assert_argument_error(exit_code, capsys, "analyze")


def test_cli_rejects_dotted_override_without_value(capsys):
    exit_code = cli.main(_analyze_args("--scan.workers"))

    _assert_argument_error(exit_code, capsys, "analyze")


def test_cli_does_not_abbreviate_ordinary_flags(capsys):
    exit_code = cli.main(_analyze_args("--str", "4", "--dry-run"))

    _assert_argument_error(exit_code, capsys, "analyze")


def test_analyze_output_dir_defaults_to_public_json_only(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def analyze(**kwargs):
        captured.update(kwargs)
        return {"phase": "analysis"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "analyze_streaming_pipeline", analyze)

    exit_code = cli.main(
        [
            "analyze",
            "--config",
            "config.yaml",
            "--input",
            "camera.clip.mp4",
            "--output-dir",
            "exports",
        ]
    )

    output_dir = tmp_path / "exports"
    assert exit_code == 0
    assert captured["workdir"] == str(output_dir / ".camera.clip_privateframe_work")
    assert captured["result_path"] == str(output_dir / "camera.clip_privateframe.json")
    assert "redacted_path" not in captured
    assert _status(capsys, "analyze")["ok"] is True


def test_process_output_dir_defaults_to_paired_json_and_video(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def process(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "render": {}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_streaming_pipeline", process)

    exit_code = cli.main(
        [
            "process",
            "--config",
            "config.yaml",
            "--input",
            "camera.clip.mp4",
            "--output-dir",
            "exports",
        ]
    )

    output_dir = tmp_path / "exports"
    assert exit_code == 0
    assert captured["workdir"] == str(output_dir / ".camera.clip_privateframe_work")
    assert captured["result_path"] == str(output_dir / "camera.clip_privateframe.json")
    assert captured["redacted_path"] == str(output_dir / "camera.clip_privateframe.mp4")
    assert captured["debug_path"] is None
    assert _status(capsys, "process")["ok"] is True


def test_render_output_dir_reuses_the_paired_json_and_video(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def render(**kwargs):
        captured.update(kwargs)
        return {"phase": "render"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "render_streaming_artifacts", render)

    exit_code = cli.main(
        [
            "render",
            "--input",
            "camera.clip.mp4",
            "--output-dir",
            "exports",
        ]
    )

    output_dir = tmp_path / "exports"
    assert exit_code == 0
    assert captured["workdir"] == str(
        output_dir / ".camera.clip_privateframe_work"
    )
    assert captured["result_path"] == str(
        output_dir / "camera.clip_privateframe.json"
    )
    assert captured["redacted_path"] == str(
        output_dir / "camera.clip_privateframe.mp4"
    )
    assert captured["debug_path"] is None
    assert _status(capsys, "render")["ok"] is True


def test_output_dir_keeps_explicit_legacy_paths_and_aliases(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    def process(**kwargs):
        captured.update(kwargs)
        return {"analysis": {}, "render": {}}

    monkeypatch.setattr(cli, "run_streaming_pipeline", process)

    exit_code = cli.main(
        [
            "process",
            "--config",
            "config.yaml",
            "--input",
            "input.mp4",
            "--output-dir",
            str(tmp_path / "exports"),
            "--workdir",
            "custom-work",
            "--json-output",
            "custom.json",
            "--video-output",
            "custom.mp4",
        ]
    )

    assert exit_code == 0
    assert captured["workdir"] == "custom-work"
    assert captured["result_path"] == "custom.json"
    assert captured["redacted_path"] == "custom.mp4"
    assert _status(capsys, "process")["ok"] is True


@pytest.mark.parametrize("command", ["analyze", "process"])
def test_analysis_commands_still_require_workdir_without_output_dir(
    command,
    capsys,
):
    exit_code = cli.main(
        [
            command,
            "--config",
            "config.yaml",
            "--input",
            "input.mp4",
            "--dry-run",
        ]
    )

    _assert_argument_error(exit_code, capsys, command)


def test_output_dir_dry_run_reports_resolved_default_paths(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(
        [
            "process",
            "--config",
            "config.yaml",
            "--input",
            "input.mp4",
            "--output-dir",
            "exports",
            "--dry-run",
        ]
    )

    payload = _status(capsys, "process")
    plan = payload["plan"]
    output_dir = tmp_path / "exports"
    assert exit_code == 0
    assert plan["output_dir"] == str(output_dir)
    assert plan["workdir"] == str(output_dir / ".input_privateframe_work")
    assert plan["artifacts"]["result_json"] == str(
        output_dir / "input_privateframe.json"
    )
    assert plan["artifacts"]["result_video"] == str(
        output_dir / "input_privateframe.mp4"
    )
