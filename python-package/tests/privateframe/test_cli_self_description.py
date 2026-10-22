from __future__ import annotations

import json
from pathlib import Path

import pytest

import insightface
from insightface.app.privateframe import cli


def _run_main(argv: list[str]) -> int:
    """Normalize argparse's version/help exit into the console exit code."""

    try:
        result = cli.main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    return int(result or 0)


def _single_compact_json(text: str) -> dict[str, object]:
    """Parse the one-record stdout contract and reject pretty/multi-line JSON."""

    assert text.endswith("\n")
    assert text.count("\n") == 1
    payload = json.loads(text)
    assert isinstance(payload, dict)
    assert text[:-1] == json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return payload


def _assert_status_envelope(payload: dict[str, object], command: str) -> None:
    assert payload["status_schema_version"] == 1
    assert payload["command"] == command
    assert isinstance(payload["ok"], bool)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"test fixture; inference is stubbed")
    return source


def test_describe_is_a_machine_readable_public_contract(capsys):
    exit_code = _run_main(["describe"])

    captured = capsys.readouterr()
    payload = _single_compact_json(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    _assert_status_envelope(payload, "describe")
    assert payload["ok"] is True
    assert isinstance(payload["tool"], dict)
    assert payload["tool"]["name"] == "insightface-privateframe"
    assert payload["tool"]["version"] == insightface.__version__
    assert set(
        (
            "commands",
            "config",
            "artifacts",
            "status_output",
            "exit_codes",
            "examples",
        )
    ).issubset(payload)
    assert {"analyze", "render", "process", "describe", "doctor"}.issubset(
        payload["commands"]
    )

    # Hidden implementation/debug controls must not become part of the public
    # agent-discoverable surface, even if compatibility parsing remains.
    public_contract = json.dumps(payload, ensure_ascii=False)
    assert "--debug" not in public_contract
    public_options = set()
    parameter_groups = [payload["tool"].get("global_parameters", [])]
    parameter_groups.extend(
        specification.get("parameters", [])
        for specification in payload["commands"].values()
    )
    for parameters in parameter_groups:
        for parameter in parameters:
            public_options.add(parameter["name"])
            public_options.update(parameter.get("aliases", []))
    # ``--json-output`` remains a result-path alias; only the obsolete status
    # formatting switch named exactly ``--json`` is removed.
    assert "--json" not in public_options
    assert "--no-verify-source" not in public_options
    dotted_options = payload["config"]["dotted_options"]
    sampling = dotted_options["scan.max_analysis_fps"]
    assert sampling["default"] == 15
    assert "Fast mode" in sampling["description"]
    assert sampling["type"] == "number"
    assert sampling["unit"] == "frames_per_second_of_input_video"
    assert "wall-clock processing speed" in sampling["description"]
    assert "5%" in sampling["description"]
    assert "briefly visible faces" in sampling["tuning_guidance"]["increase"]
    assert "risk of missing faces" in sampling["tuning_guidance"]["decrease"]
    model_root = dotted_options["models.root"]
    assert model_root["option"] == "--models.root"
    assert model_root["type"] == "string"
    assert model_root["default"] == "~/.insightface"
    assert model_root["format"] == "local_directory_path"
    assert "authoritative" in model_root["description"]
    assert payload["config"]["validation"][
        "semantic_and_cross_field_constraints"
    ] == "authoritative command --dry-run"
    assert "render.debug_line_thickness" not in dotted_options
    assert "render.video_output.audio.debug" not in dotted_options
    assert "scan.session_sharing" not in dotted_options
    assert "recognition.max_frames_per_track" not in dotted_options
    analyze_parameters = {
        item["name"]: item for item in payload["commands"]["analyze"]["parameters"]
    }
    doctor_parameters = {
        item["name"]: item for item in payload["commands"]["doctor"]["parameters"]
    }
    assert "SQLite cache" in analyze_parameters["--workdir"]["help"]
    assert "never modified" in analyze_parameters["--input"]["help"]
    assert "without creating" in doctor_parameters["--output-dir"]["help"]
    result_schema = payload["artifacts"]["result_json"]["stable_render_input_schema"]
    assert result_schema["source_video"]["file_name"] == {
        "type": "string",
        "min_length": 1,
    }
    assert result_schema["source_video"]["required"] == [
        "file_name",
        "metadata",
        "coordinate_system",
        "frame_index_origin",
        "timing_contract",
    ]
    assert result_schema["source_video"]["metadata"]["frame_count"] == {
        "type": "integer",
        "minimum": 1,
    }
    assert result_schema["observations"]["items"]["required"] == [
        "frame_idx",
        "track_id",
        "box",
        "source",
    ]
    assert result_schema["observations"]["items"]["source"]["enum"] == [
        "detector",
        "tracked",
        "interpolated",
        "repaired",
        "manual",
    ]
    assert result_schema["observations"]["items"]["box"]["length"] == 4
    recognition_schema = result_schema["recognition"]
    assert recognition_schema["type"] == "object"
    assert recognition_schema["required"] == ["enabled"]
    assert recognition_schema["enabled"] == {"type": "boolean"}
    selective = recognition_schema["fields_when_enabled"]
    assert selective["references"]["accepted_images"]["minimum"] == 1
    assert selective["tracks"]["values"]["status"]["enum"] == [
        "CONFIRMED", "UNKNOWN", "CONFLICT",
    ]
    policy = result_schema["render_defaults"]["recognition_policy"]
    assert policy["unknown_action"]["enum"] == ["blur", "keep"]
    assert "identity_unconfirmed" in result_schema["observations"]["items"]
    assert "force_blur" not in result_schema["observations"]["items"]
    assert payload["artifacts"]["result_json"]["source_compatibility"] == {
        "checks": ["width", "height", "fps", "decoded_frame_count"],
        "content_hash_required": False,
    }
    artifacts = payload["artifacts"]
    assert artifacts["output_dir_defaults"]["result_json"] == (
        "<input_stem>_privateframe.json"
    )
    assert artifacts["workdir_fallbacks"]["result_json"] == (
        "<workdir>/result.privateframe.json"
    )
    assert artifacts["workdir_fallbacks"]["result_video"] is None


def test_describe_teaches_an_unfamiliar_automation_client_how_to_use_the_tool(
    capsys,
):
    exit_code = _run_main(["describe"])

    payload = _single_compact_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["contract_schema_version"] == 2

    tool = payload["tool"]
    assert tool["purpose_id"] == "video_face_privacy_redaction"
    assert tool["summary"] == cli.command_parser().description
    assert {"faces", "video", "blur", "mosaic"}.issubset(
        set(tool["summary"].lower().replace(".", "").split())
    )
    assert tool["primary_input"] == "source_video"
    assert tool["primary_file_outputs"] == ["result_json", "result_video"]
    assert tool["installation"]["pip_argv"] == [
        "python",
        "-m",
        "pip",
        "install",
        "insightface[privateframe]",
    ]

    discovery = payload["discovery"]
    assert "vendor_neutral_ai_coding_agents" in discovery["audience"]
    assert discovery["machine_discovery_argv"] == [
        "insightface-privateframe",
        "describe",
    ]
    assert discovery["default_command_for_face_redaction"] == "process"
    assert discovery["command_selection"] == {
        "redact_video_now": "process",
        "analysis_json_only": "analyze",
        "render_existing_or_edited_result_json": "render",
        "inspect_environment_readiness": "doctor",
        "inspect_machine_contract": "describe",
    }
    assert discovery["defaults"]["redaction_style"] == "gaussian"
    assert discovery["defaults"]["redaction_style"] == (
        payload["config"]["dotted_options"]["render.redaction.method"]["default"]
    )
    assert discovery["defaults"]["mosaic_override"] == [
        "--render.redaction.method",
        "mosaic",
    ]
    assert discovery["safe_automation"]["continue_when"] == {
        "stdout.ok": True,
        "stdout.ready": True,
    }
    assert "Codex" not in json.dumps(discovery)
    assert "Claude" not in json.dumps(discovery)

    commands = payload["commands"]
    assert commands["analyze"]["operation"] == "analyze"
    assert commands["analyze"]["reads"] == ["source_video"]
    assert commands["analyze"]["outputs"] == ["result_json"]
    assert commands["render"]["operation"] == "render"
    assert commands["render"]["reads"] == ["source_video", "result_json"]
    assert commands["render"]["outputs"] == ["result_video"]
    assert commands["render"]["writes"] == ["result_video", "work_directory"]
    assert "work_directory" in commands["render"]["conditional_writes"]
    assert commands["process"]["operation"] == "analyze_and_render"
    assert commands["process"]["reads"] == ["source_video"]
    assert commands["process"]["outputs"] == ["result_json", "result_video"]
    assert commands["doctor"]["reads"] == []
    assert "source_video" in commands["doctor"]["optional_reads"]
    assert commands["describe"]["writes"] == []
    assert all(commands[name]["when_to_use"] for name in commands)

    artifacts = payload["artifacts"]
    artifact_ids = {
        "source_video",
        "analysis_config",
        "render_config",
        "output_directory",
        "result_json",
        "result_video",
        "work_directory",
    }
    assert artifacts["source_video"]["option"] == "--input"
    assert artifacts["source_video"]["locator"] == "local_path"
    assert artifacts["source_video"]["modified"] is False
    assert artifacts["source_video"]["uploaded_by_privateframe"] is False
    assert artifacts["source_video"]["read_by"] == [
        "analyze",
        "render",
        "process",
    ]
    assert artifacts["source_video"]["optionally_read_by"] == ["doctor"]
    assert artifacts["output_directory"]["used_by"] == [
        "analyze",
        "render",
        "process",
    ]
    assert artifacts["output_directory"]["optionally_used_by"] == ["doctor"]
    assert artifacts["result_json"]["distinct_from_stdout_status"] is True
    assert artifacts["result_json"]["produced_by"] == ["analyze", "process"]
    assert artifacts["result_json"]["consumed_by"] == ["render"]
    assert artifacts["result_video"]["produced_by"] == ["render", "process"]
    assert artifacts["result_video"]["derived_from"] == [
        "source_video",
        "result_json",
    ]
    for command in commands.values():
        for field in ("reads", "optional_reads", "outputs", "writes"):
            assert set(command.get(field, ())) <= artifact_ids
    for artifact in (artifacts[name] for name in artifact_ids):
        for field in (
            "read_by",
            "optionally_read_by",
            "used_by",
            "optionally_used_by",
            "produced_by",
            "consumed_by",
            "written_by",
            "conditionally_written_by",
        ):
            assert set(artifact.get(field, ())) <= set(commands)

    primary_io = payload["primary_io"]
    assert primary_io["primary_input"] == "source_video"
    assert primary_io["command_outputs_field_semantics"] == "file_artifacts_only"
    assert primary_io["status_output"]["channel"] == "stdout"
    assert primary_io["status_output"]["artifact"] is False
    assert primary_io["status_output"]["not_the_analysis_result_file"] is True

    workflows = {item["id"]: item for item in payload["recommended_workflows"]}
    default = workflows["redact_video_now"]
    assert default["default"] is True
    assert default["commands"] == ["process"]
    assert default["inputs"] == ["source_video", "output_directory"]
    assert default["outputs"] == ["result_json", "result_video"]
    assert "--dry-run" in default["preflight_argv"]
    assert "--dry-run" in workflows["analysis_json_only"]["preflight_argv"]
    editable = workflows["analyze_edit_then_render"]
    assert editable["commands"] == ["analyze", "render"]
    assert editable["intermediate"] == "result_json"
    assert editable["intermediate_may_be_edited"] is True
    render_step = next(
        step for step in editable["steps"] if step.get("command") == "render"
    )
    assert "--dry-run" in render_step["preflight_argv"]
    for workflow in workflows.values():
        assert set(workflow["commands"]) <= set(commands)
        assert set(workflow["inputs"]) <= artifact_ids
        assert set(workflow["outputs"]) <= artifact_ids

    serialized = json.dumps(payload, ensure_ascii=False)
    assert serialized.index('"summary"') < serialized.index('"dotted_options"')

    status_output = payload["status_output"]
    assert status_output["execution_success"]["artifact_paths_field"] == "artifacts"
    assert status_output["execution_success"]["artifact_path_keys"] == [
        "result_json",
        "result_video",
    ]
    success = status_output["execution_success"]
    assert success["required_fields"] == [
        "artifacts",
        "runtime",
        "timings",
        "summary",
    ]
    assert success["summary_fields_by_command"] == {
        "analyze": [
            "frame_count",
            "face_tracks",
            "face_regions",
        ],
        "render": [
            "frame_count",
            "face_regions",
            "redacted_face_regions",
            "kept_face_regions",
        ],
        "process": [
            "frame_count",
            "face_tracks",
            "face_regions",
            "redacted_face_regions",
            "kept_face_regions",
        ],
    }
    assert success["timing_fields_by_command"] == {
        "analyze": ["total_seconds"],
        "render": ["total_seconds"],
        "process": ["total_seconds"],
    }
    assert "diagnostics are excluded" in success["summary_semantics"]
    assert "counted separately" in success["summary_field_semantics"]["face_regions"]
    assert "null for render" in success["runtime_provider_semantics"]
    assert "not the complete CLI process wall time" in success["timing_semantics"]
    assert status_output["dry_run"]["artifact_paths_field"] == "plan.artifacts"

    config = payload["config"]
    assert config["scope"] == "common_and_intermediate"
    assert config["unlisted_options_supported"] is True
    assert "common_options" not in payload
    assert "schema" not in config
    assert "defaults" not in config
    groups = config["groups"]
    assert groups
    assert all(group["id"] and group["description"] for group in groups)
    paths = [path for group in groups for path in group["fields"]]
    assert len(paths) == len(set(paths))
    assert {
        "models.name",
        "runtime.provider",
        "recognition.mode",
        "recognition.reference_dir",
        "recognition.unknown_action",
        "scan.max_analysis_fps",
        "render.redaction.method",
        "render.redaction.box_scale",
        "render.video_output.rate_control.quality",
        "render.video_output.preset",
        "render.video_output.audio.redacted",
    } <= set(paths)
    assert set(paths) == set(config["dotted_options"])
    assert len(paths) == 30
    for path, specification in config["dotted_options"].items():
        assert specification["option"] == f"--{path}"
        assert specification["type"]
        assert "default" in specification
        assert specification["description"]
        assert specification["when_to_use"]
        assert specification["tradeoff"]
        assert "debug" not in path
        assert path != "output.artifacts_level"
    # Advanced settings must not leak back through another configuration tree.
    serialized_config = json.dumps(config)
    for internal_name in (
        "kalman_optical_flow", "bidirectional_fusion", "candidate_filter",
        "revalidation", "session_sharing", "progress_every_frames",
        "debug_line_thickness", "artifacts_level",
    ):
        assert internal_name not in serialized_config
    reference = config["full_reference"]
    assert reference["format"] == "markdown"
    assert reference["scope"] == "all_supported_configuration"
    reference_path = Path(reference["path"])
    assert reference_path.is_absolute()
    assert reference_path.is_file()
    assert reference_path.name == "configuration.md"
    assert "tracking.kalman_optical_flow.roi_size" in reference_path.read_text(
        encoding="utf-8"
    )


def test_recommended_workflow_argv_templates_are_accepted_by_the_real_parser(
    capsys,
    tmp_path,
):
    assert _run_main(["describe"]) == 0
    payload = _single_compact_json(capsys.readouterr().out)
    parser = cli.command_parser(machine_errors=True)
    substitutions = {
        "<source_video_path>": str(tmp_path / "source.mp4"),
        "<output_directory_path>": str(tmp_path / "output"),
    }
    templates: list[list[str]] = []
    for workflow in payload["recommended_workflows"]:
        for field in ("preflight_argv", "execute_argv"):
            if field in workflow:
                templates.append(workflow[field])
        for step in workflow.get("steps", []):
            for field in ("preflight_argv", "argv"):
                if field in step:
                    templates.append(step[field])

    assert templates
    for template in templates:
        argv = [substitutions.get(token, token) for token in template][1:]
        clean, overrides = cli._parse_dotted_config_overrides(argv)
        args = parser.parse_args(clean)
        cli._apply_output_defaults(args, parser)
        assert args.command in {"analyze", "render", "process"}
        assert overrides == {}


def test_doctor_returns_structured_readiness_even_when_not_ready(capsys):
    exit_code = _run_main(["doctor"])

    captured = capsys.readouterr()
    payload = _single_compact_json(captured.out)
    _assert_status_envelope(payload, "doctor")
    assert exit_code >= 0
    assert captured.err == ""
    assert isinstance(payload["ready"], bool)
    assert isinstance(payload["checks"], (list, dict))
    assert set(("runtime", "models", "media", "output", "safety")).issubset(payload)


def test_version_is_available_without_constructing_a_subcommand(capsys):
    exit_code = _run_main(["--version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert insightface.__version__ in captured.out


@pytest.mark.parametrize(
    ("command", "pipeline_name", "result"),
    [
        ("analyze", "analyze_streaming_pipeline", {"phase": "analysis"}),
        (
            "process",
            "run_streaming_pipeline",
            {"analysis": {"phase": "analysis"}, "render": {"phase": "render"}},
        ),
        ("render", "render_streaming_artifacts", {"phase": "render"}),
    ],
)
def test_execution_commands_write_one_compact_status_json_to_stdout(
    command,
    pipeline_name,
    result,
    monkeypatch,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)
    output_dir = tmp_path / "exports"
    argv = [
        command,
        "--input",
        str(source),
        "--output-dir",
        str(output_dir),
        "--progress",
        "none",
    ]
    if command == "render":
        result_path = output_dir / "input_privateframe.json"
        result_path.parent.mkdir(parents=True)
        result_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli, pipeline_name, lambda **_kwargs: result)

    exit_code = _run_main(argv)

    captured = capsys.readouterr()
    payload = _single_compact_json(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    _assert_status_envelope(payload, command)
    assert payload["ok"] is True
    expected_result = str((output_dir / "input_privateframe.json").resolve())
    expected_video = (
        str((output_dir / "input_privateframe.mp4").resolve())
        if command in {"render", "process"}
        else None
    )
    assert payload["artifacts"] == {
        "result_json": expected_result,
        "result_video": expected_video,
    }


def test_execution_error_uses_the_stdout_json_envelope(
    monkeypatch,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)

    def fail(**_kwargs):
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(cli, "analyze_streaming_pipeline", fail)

    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "exports"),
            "--progress",
            "none",
        ]
    )

    captured = capsys.readouterr()
    payload = _single_compact_json(captured.out)
    assert exit_code != 0
    assert captured.err == ""
    _assert_status_envelope(payload, "analyze")
    assert payload["ok"] is False
    error = payload["error"]
    assert set(("code", "stage", "type", "message", "retryable", "hints")).issubset(
        error
    )
    assert error["type"] == "RuntimeError"
    assert error["message"] == "synthetic inference failure"
    assert isinstance(error["retryable"], bool)
    assert isinstance(error["hints"], list)


def test_keyboard_interrupt_uses_the_cancelled_status_envelope(
    monkeypatch,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)

    def interrupt(**_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "analyze_streaming_pipeline", interrupt)

    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "exports"),
            "--progress",
            "none",
        ]
    )

    payload = _single_compact_json(capsys.readouterr().out)
    assert exit_code == 130
    assert payload["ok"] is False
    assert payload["error"]["code"] == "cancelled"
    assert payload["error"]["type"] == "KeyboardInterrupt"
    assert payload["error"]["message"] == "operation interrupted by user"
    assert payload["error"]["retryable"] is True


def test_keyboard_interrupt_during_argument_parsing_is_also_structured(
    monkeypatch,
    capsys,
):
    def interrupt(_argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_parse_dotted_config_overrides", interrupt)

    exit_code = _run_main(["analyze"])

    payload = _single_compact_json(capsys.readouterr().out)
    assert exit_code == 130
    assert payload["command"] == "analyze"
    assert payload["error"]["code"] == "cancelled"
    assert payload["error"]["stage"] == "arguments"


def test_status_output_remains_strict_json_when_a_runtime_value_is_nonfinite(
    monkeypatch,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)
    monkeypatch.setattr(
        cli,
        "analyze_streaming_pipeline",
        lambda **_kwargs: {"timings": {"seconds": float("nan")}},
    )

    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "exports"),
            "--progress",
            "none",
        ]
    )

    payload = _single_compact_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["timings"] == {"total_seconds": None}
    assert payload["summary"] == {}


@pytest.mark.parametrize("progress", ["auto", "text", "jsonl", "none"])
def test_progress_modes_are_public_parser_choices(progress, tmp_path):
    args = cli.command_parser().parse_args(
        [
            "analyze",
            "--input",
            str(tmp_path / "input.mp4"),
            "--output-dir",
            str(tmp_path / "exports"),
            "--progress",
            progress,
        ]
    )

    assert args.progress == progress


def test_invalid_progress_is_a_structured_argument_error(tmp_path, capsys):
    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(tmp_path / "input.mp4"),
            "--output-dir",
            str(tmp_path / "exports"),
            "--progress",
            "verbose",
        ]
    )

    captured = capsys.readouterr()
    payload = _single_compact_json(captured.out)
    assert exit_code == 2
    assert captured.err == ""
    _assert_status_envelope(payload, "analyze")
    assert payload["ok"] is False
    assert payload["error"]["stage"] == "arguments"


def test_jsonl_progress_uses_stderr_without_polluting_stdout(
    monkeypatch,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)

    def analyze(**kwargs):
        kwargs["progress"](2, 10, "analysis")
        return {"phase": "analysis"}

    monkeypatch.setattr(cli, "analyze_streaming_pipeline", analyze)

    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "exports"),
            "--progress",
            "jsonl",
        ]
    )

    captured = capsys.readouterr()
    status = _single_compact_json(captured.out)
    progress_lines = captured.err.splitlines()
    assert exit_code == 0
    _assert_status_envelope(status, "analyze")
    assert len(progress_lines) == 1
    progress = json.loads(progress_lines[0])
    assert progress_lines[0] == json.dumps(
        progress,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert progress["phase"] == "analysis"
    assert progress["current"] == 2
    assert progress["total"] == 10


@pytest.mark.parametrize(
    ("override", "value"),
    [("scan.max_analysis_fps", 30), ("tracking.kalman_optical_flow.roi_size", 256)],
)
def test_dry_run_returns_a_resolved_plan_without_calling_the_pipeline(
    monkeypatch,
    tmp_path,
    capsys,
    override,
    value,
):
    source = _source(tmp_path)

    def unexpected(**_kwargs):  # pragma: no cover - failure explains itself
        raise AssertionError("dry-run must not execute inference or rendering")

    monkeypatch.setattr(cli, "run_streaming_pipeline", unexpected)

    exit_code = _run_main(
        [
            "process",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "exports"),
            f"--{override}",
            str(value),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    payload = _single_compact_json(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    _assert_status_envelope(payload, "process")
    assert payload["dry_run"] is True
    assert isinstance(payload["ready"], bool)
    assert set(
        ("config", "input", "workdir", "artifacts", "config_overrides")
    ).issubset(payload["plan"])
    assert payload["plan"]["config_overrides"] == {override: value}


def test_existing_artifact_is_protected_unless_overwrite_is_explicit(
    monkeypatch,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    existing = output_dir / "input_privateframe.json"
    existing.write_text('{"owned_by":"user"}', encoding="utf-8")
    calls = []

    def analyze(**kwargs):
        calls.append(kwargs)
        return {"phase": "analysis"}

    monkeypatch.setattr(cli, "analyze_streaming_pipeline", analyze)
    base = [
        "analyze",
        "--input",
        str(source),
        "--output-dir",
        str(output_dir),
        "--progress",
        "none",
    ]

    protected_exit = _run_main(base)
    protected_capture = capsys.readouterr()
    protected = _single_compact_json(protected_capture.out)
    assert protected_exit != 0
    assert protected_capture.err == ""
    assert calls == []
    assert protected["ok"] is False
    assert protected["error"]["stage"] == "preflight"
    assert str(existing) in protected["error"]["message"]

    overwrite_exit = _run_main([*base, "--overwrite"])
    overwrite_capture = capsys.readouterr()
    overwrite = _single_compact_json(overwrite_capture.out)
    assert overwrite_exit == 0
    assert overwrite_capture.err == ""
    assert len(calls) == 1
    assert overwrite["ok"] is True


def test_concurrent_output_claim_returns_a_retryable_busy_error(
    monkeypatch,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    result_path = output_dir / "input_privateframe.json"
    lock_path = cli._output_lock_path(result_path)
    lock_path.write_text(f"pid={cli.os.getpid()}\n", encoding="ascii")
    calls = []

    def analyze(**kwargs):
        calls.append(kwargs)
        return {"phase": "analysis"}

    monkeypatch.setattr(cli, "analyze_streaming_pipeline", analyze)

    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--progress",
            "none",
        ]
    )

    payload = _single_compact_json(capsys.readouterr().out)
    assert exit_code == 1
    assert calls == []
    assert payload["error"]["code"] == "output_busy"
    assert payload["error"]["stage"] == "preflight"
    assert payload["error"]["retryable"] is True
    assert lock_path.exists()


def test_workdir_is_part_of_the_concurrent_resource_claim(
    monkeypatch,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)
    workdir = tmp_path / "shared-work"
    lock_path = cli._workdir_lock_path(workdir)
    lock_path.write_text(f"pid={cli.os.getpid()}\n", encoding="ascii")
    calls = []
    monkeypatch.setattr(
        cli,
        "analyze_streaming_pipeline",
        lambda **kwargs: calls.append(kwargs) or {"phase": "analysis"},
    )

    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(source),
            "--workdir",
            str(workdir),
            "--result",
            str(tmp_path / "different-result.json"),
            "--progress",
            "none",
        ]
    )

    payload = _single_compact_json(capsys.readouterr().out)
    assert exit_code == 1
    assert calls == []
    assert payload["error"]["code"] == "output_busy"
    assert str(workdir) in payload["error"]["message"]


def test_dry_run_reports_an_active_resource_lock_without_writing(
    tmp_path,
    capsys,
):
    source = _source(tmp_path)
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    lock_path = cli._output_lock_path(output_dir / "input_privateframe.json")
    lock_path.write_text(f"pid={cli.os.getpid()}\n", encoding="ascii")

    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    payload = _single_compact_json(capsys.readouterr().out)
    resource_check = next(
        check for check in payload["checks"] if check["name"] == "resource_locks"
    )
    assert exit_code == 0
    assert payload["ready"] is False
    assert resource_check["ok"] is False
    assert resource_check["details"]["active"][0]["owner_pid"] == cli.os.getpid()


def test_execution_reclaims_a_lock_owned_by_a_dead_process(
    monkeypatch,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)
    output_dir = tmp_path / "exports"
    output_dir.mkdir()
    lock_path = cli._output_lock_path(output_dir / "input_privateframe.json")
    lock_path.write_text("pid=99999999\n", encoding="ascii")
    calls = []
    monkeypatch.setattr(cli, "_pid_is_running", lambda _pid: False)
    monkeypatch.setattr(
        cli,
        "analyze_streaming_pipeline",
        lambda **kwargs: calls.append(kwargs) or {"phase": "analysis"},
    )

    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--progress",
            "none",
        ]
    )

    payload = _single_compact_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert len(calls) == 1
    assert not lock_path.exists()


def test_process_dry_run_validates_the_render_config_layer(tmp_path, capsys):
    source = _source(tmp_path)
    render_config = tmp_path / "invalid-render.yaml"
    render_config.write_text(
        "video_output:\n  unsupported_setting: value\n",
        encoding="utf-8",
    )

    exit_code = _run_main(
        [
            "process",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "exports"),
            "--render-config",
            str(render_config),
            "--progress",
            "none",
            "--dry-run",
        ]
    )

    payload = _single_compact_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["ready"] is False
    render_check = next(
        check for check in payload["checks"] if check["name"] == "render_settings"
    )
    assert render_check["ok"] is False
    assert "unsupported_setting" in render_check["message"]
    assert payload["diagnostics"]["summary"]["total"] == len(payload["checks"])
    assert payload["diagnostics"]["summary"]["errors"] >= 1


@pytest.mark.parametrize(
    "option",
    [
        "scan.session_sharing",
        "scan.workers",
        "scan.pipeline_depth",
        "streaming.progress_every_frames",
    ],
)
def test_dry_run_rejects_invalid_runtime_leaf_values_before_inference(
    option,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)

    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "exports"),
            f"--{option}",
            "bogus",
            "--dry-run",
        ]
    )

    payload = _single_compact_json(capsys.readouterr().out)
    config_check = next(
        check for check in payload["checks"] if check["name"] == "config.load"
    )
    assert exit_code == 0
    assert payload["ready"] is False
    assert config_check["ok"] is False
    assert option in config_check["details"]["error"]


def test_dry_run_checks_explicit_paths_not_only_output_dir_defaults(
    tmp_path,
    capsys,
):
    source = _source(tmp_path)

    exit_code = _run_main(
        [
            "analyze",
            "--input",
            str(source),
            "--workdir",
            str(tmp_path / "work"),
            "--result",
            str(source),
            "--overwrite",
            "--progress",
            "none",
            "--dry-run",
        ]
    )

    payload = _single_compact_json(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ready"] is False
    distinct = next(
        check for check in payload["checks"] if check["name"] == "paths.distinct"
    )
    assert distinct["ok"] is False


def test_render_dry_run_compares_result_metadata_with_the_decoded_input(
    monkeypatch,
    tmp_path,
    capsys,
):
    source = _source(tmp_path)
    result_path = tmp_path / "input_privateframe.json"
    result_path.write_text(
        json.dumps(
            {
                "format": "privateframe-result",
                "schema_version": 1,
                "observations": [],
                "source_video": {
                    "file_name": "input.mp4",
                    "metadata": {
                        "width": 1,
                        "height": 360,
                        "fps": 30.0,
                        "frame_count": 10,
                    },
                },
                "render_defaults": {},
                "recognition": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "insightface.app.privateframe.doctor.run_doctor",
        lambda **_kwargs: {
            "ok": True,
            "ready": True,
            "checks": [],
            "runtime": {},
            "models": {},
            "media": {
                "first_frame_decoded": True,
                "privateframe_metadata": {
                    "width": 640,
                    "height": 360,
                    "fps": 30.0,
                    "frame_count": 10,
                },
            },
            "output": {},
            "safety": {},
        },
    )

    exit_code = _run_main(
        [
            "render",
            "--input",
            str(source),
            "--result",
            str(result_path),
            "--redacted",
            str(tmp_path / "output.mp4"),
            "--dry-run",
        ]
    )

    payload = _single_compact_json(capsys.readouterr().out)
    result_check = next(
        check for check in payload["checks"] if check["name"] == "render_result"
    )
    assert exit_code == 0
    assert payload["ready"] is False
    assert "metadata.width does not match" in result_check["message"]
