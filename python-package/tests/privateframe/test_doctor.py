from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from insightface.app.privateframe import doctor, model_catalog
from insightface.app.privateframe.base_config import DEFAULT_CONFIG_PATH


def _write_manifest_package(
    root: Path,
    *,
    detection_bytes: bytes | None = b"detection",
    verification_bytes: bytes | None = b"verification",
    recognition_bytes: bytes | None = b"recognition",
    declare_sha256: bool = False,
    sha256_overrides: dict[str, str] | None = None,
) -> Path:
    package = root / "models" / "raccoon_s"
    package.mkdir(parents=True)
    actual = {
        "detection": detection_bytes,
        "verification": verification_bytes,
        "recognition": recognition_bytes,
    }
    filenames = {
        "detection": "detector.onnx",
        "verification": "verifier.onnx",
        "recognition": "recognizer.onnx",
    }
    for task, contents in actual.items():
        if contents is not None:
            (package / filenames[task]).write_bytes(contents)
    manifest = {
        "manifest_version": 2,
        "model_id": "raccoon_s",
        "tasks": {
            "detection": {
                "file": filenames["detection"],
                "preprocessing": "embedded",
            },
            "verification": {
                "file": filenames["verification"],
                "expansion": 1.2,
                "preprocessing": "embedded",
            },
            "recognition": {
                "file": filenames["recognition"],
                "preprocessing": {"mean": 127.5, "std": 127.5},
            },
        },
        "license": "MODEL.LICENSE",
    }
    if declare_sha256:
        overrides = sha256_overrides or {}
        for task, contents in actual.items():
            if contents is not None:
                manifest["tasks"][task]["sha256"] = overrides.get(
                    task,
                    hashlib.sha256(contents).hexdigest(),
                )
    (package / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return package


@pytest.fixture
def stable_pyav(monkeypatch: pytest.MonkeyPatch) -> None:
    def report(
        checks: list[dict[str, Any]],
        _config: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
        doctor._add_check(
            checks,
            "runtime.pyav",
            True,
            "test PyAV capability is available",
            details={"version": "12.0.0"},
        )
        return {"installed": True, "version": "12.0.0"}, None

    monkeypatch.setattr(doctor, "_pyav_report", report)


def _checks_by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in report["checks"]}


def test_doctor_offline_config_never_downloads_or_constructs_a_session(
    monkeypatch: pytest.MonkeyPatch,
    stable_pyav: None,
) -> None:
    calls = {"ensure_available": 0, "session": 0}

    def forbidden_download(*_args: Any, **_kwargs: Any) -> None:
        calls["ensure_available"] += 1
        raise AssertionError("doctor must not invoke ModelZoo materialization")

    def forbidden_session(*_args: Any, **_kwargs: Any) -> None:
        calls["session"] += 1
        raise AssertionError("doctor must not construct an inference Session")

    monkeypatch.setattr(model_catalog, "ensure_available", forbidden_download)
    monkeypatch.setattr(doctor.ort, "InferenceSession", forbidden_session)

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        check_models=False,
    )

    assert report["ok"] is True
    assert report["ready"] is True
    assert report["config"]["materialize_models"] is False
    assert calls == {"ensure_available": 0, "session": 0}
    assert report["safety"] == {
        "read_only": True,
        "scope": "doctor checks after Python dependency imports",
        "model_downloads": False,
        "onnx_sessions_created": False,
        "coreml_compilation": False,
        "warmup": False,
        "files_or_directories_created": False,
        "dependency_import_side_effects_controlled": False,
        "input_frames_decoded": False,
        "input_probe": "skipped",
        "output_writability_probe": "skipped",
    }


def test_doctor_without_optional_inputs_marks_checks_skipped_without_losing_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    monkeypatch.setattr(
        doctor, "DEFAULT_INSIGHTFACE_ROOT", str(tmp_path / "models-root")
    )

    report = doctor.run_doctor()

    checks = _checks_by_name(report)
    assert report["ok"] is True
    assert report["ready"] is True
    for name in (
        "config.load",
        "runtime.configured_provider",
        "models.selected_package",
        "media.input",
        "output.directory",
        "output.targets",
    ):
        assert checks[name]["ok"] is True
        assert checks[name]["details"]["skipped"] is True


def test_doctor_reports_a_missing_selected_model_package(
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    root = tmp_path / "insightface-root"

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        config_overrides={"models.root": str(root)},
    )

    check = _checks_by_name(report)["models.selected_directory"]
    assert report["ok"] is True
    assert report["ready"] is False
    assert check["ok"] is False
    assert check["severity"] == "error"
    selected = report["models"]["packages"]["raccoon_s"]
    assert selected["directory_exists"] is False
    assert selected["manifest"]["exists"] is False
    assert report["models"]["insightface_root"] == str(root.resolve())
    assert report["models"]["root"] == str((root / "models").resolve())


def test_doctor_never_falls_back_from_the_effective_model_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    fallback_root = tmp_path / "fallback-root"
    selected_root = tmp_path / "selected-root"
    _write_manifest_package(fallback_root, declare_sha256=True)
    monkeypatch.setattr(doctor, "DEFAULT_INSIGHTFACE_ROOT", str(fallback_root))

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        config_overrides={"models.root": str(selected_root)},
    )

    check = _checks_by_name(report)["models.selected_directory"]
    assert report["ready"] is False
    assert check["ok"] is False
    assert report["models"]["insightface_root"] == str(selected_root.resolve())
    assert report["models"]["packages"]["raccoon_s"]["path"] == str(
        selected_root / "models" / "raccoon_s"
    )


def test_doctor_reports_an_invalid_model_root_without_scanning_the_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    fallback_root = tmp_path / "fallback-root"
    _write_manifest_package(fallback_root, declare_sha256=True)
    monkeypatch.setattr(doctor, "DEFAULT_INSIGHTFACE_ROOT", str(fallback_root))

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        config_overrides={"models.root": []},
    )

    checks = _checks_by_name(report)
    assert report["ok"] is True
    assert report["ready"] is False
    assert checks["config.load"]["ok"] is False
    assert checks["models.root"]["ok"] is False
    assert report["models"]["configured_root"] == []
    assert report["models"]["packages"] == {}
    assert report["models"]["root"] is None


def test_doctor_reports_a_missing_selected_manifest(
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    root = tmp_path / "insightface-root"
    (root / "models" / "raccoon_s").mkdir(parents=True)

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        config_overrides={"models.root": str(root)},
    )

    check = _checks_by_name(report)["models.selected_manifest"]
    manifest = report["models"]["packages"]["raccoon_s"]["manifest"]
    assert report["ok"] is True
    assert report["ready"] is False
    assert check["ok"] is False
    assert check["severity"] == "error"
    assert manifest["exists"] is False
    assert manifest["valid"] is False
    assert manifest["manifest_version"] is None


def test_doctor_structures_missing_model_files_without_hashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    root = tmp_path / "insightface-root"
    _write_manifest_package(
        root,
        detection_bytes=None,
        verification_bytes=None,
    )
    monkeypatch.setattr(
        doctor,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("a task without a declared digest must not be hashed")
        ),
    )

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        config_overrides={"models.root": str(root)},
    )

    checks = _checks_by_name(report)
    tasks = report["models"]["packages"]["raccoon_s"]["tasks"]
    assert report["ok"] is True
    assert report["ready"] is False
    assert tasks["detection"]["exists"] is False
    assert tasks["verification"]["exists"] is False
    assert tasks["recognition"]["exists"] is True
    assert all("hash_status" not in value for value in tasks.values())
    assert all("expected_sha256" not in value for value in tasks.values())
    assert all(
        value["sha256"]
        == {"declared": None, "actual": None, "status": "not_declared"}
        for value in tasks.values()
    )
    assert checks["models.selected.detection"]["severity"] == "error"
    assert checks["models.selected.verification"]["severity"] == "error"
    assert checks["models.selected.recognition"]["ok"] is True
    json.dumps(report, allow_nan=False)


def test_doctor_reports_verified_declared_model_sha256(
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    root = tmp_path / "insightface-root"
    _write_manifest_package(root, declare_sha256=True)

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        config_overrides={"models.root": str(root)},
    )

    checks = _checks_by_name(report)
    tasks = report["models"]["packages"]["raccoon_s"]["tasks"]
    assert report["ready"] is True
    for task in ("detection", "verification", "recognition"):
        digest = tasks[task]["sha256"]
        assert digest["status"] == "verified"
        assert digest["actual"] == digest["declared"]
        assert checks[f"models.selected.{task}"]["ok"] is True


def test_doctor_fails_a_selected_required_task_with_mismatched_sha256(
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    root = tmp_path / "insightface-root"
    _write_manifest_package(
        root,
        declare_sha256=True,
        sha256_overrides={"detection": "0" * 64},
    )

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        config_overrides={"models.root": str(root)},
    )

    check = _checks_by_name(report)["models.selected.detection"]
    digest = report["models"]["packages"]["raccoon_s"]["tasks"]["detection"][
        "sha256"
    ]
    assert report["ready"] is False
    assert check["ok"] is False
    assert check["severity"] == "error"
    assert digest["declared"] == "0" * 64
    assert digest["actual"] == hashlib.sha256(b"detection").hexdigest()
    assert digest["status"] == "mismatch"


class _FakeCodec:
    def __init__(self, name: str, mode: str) -> None:
        self.name = name
        self.long_name = f"fake {name}"
        self.type = "video"
        self.is_decoder = mode == "r"
        self.is_encoder = mode == "w"


class _FakeStreams:
    def __init__(self, video: Any) -> None:
        self.video = [video]
        self.audio: list[Any] = []

    def __len__(self) -> int:
        return len(self.video) + len(self.audio)


class _FakeContainer:
    def __init__(self, fps: float) -> None:
        rate = Fraction(str(fps))
        frame_count = max(1, round(fps * 2.0))
        time_base = 1 / rate
        codec_context = SimpleNamespace(
            name="h264",
            profile="High",
            width=640,
            height=360,
            pix_fmt="yuv420p",
        )
        stream = SimpleNamespace(
            index=0,
            average_rate=rate,
            guessed_rate=None,
            base_rate=None,
            frames=frame_count,
            duration=frame_count,
            time_base=time_base,
            codec_context=codec_context,
        )
        self.streams = _FakeStreams(stream)
        self.duration = round((frame_count / fps) * 1_000_000)
        self.format = SimpleNamespace(name="mp4", long_name="fake MP4")
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def decode(self, _stream: Any):
        yield SimpleNamespace(
            width=640,
            height=360,
            format=SimpleNamespace(name="yuv420p"),
        )


class _FakeAV:
    time_base = 1_000_000

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.containers: list[_FakeContainer] = []

    @staticmethod
    def Codec(name: str, mode: str) -> _FakeCodec:
        return _FakeCodec(name, mode)

    def open(self, _path: str, *, mode: str) -> _FakeContainer:
        assert mode == "r"
        value = _FakeContainer(self.fps)
        self.containers.append(value)
        return value


@pytest.mark.parametrize(
    ("source_fps", "expected_stride"),
    [
        (30.02, 1),
        (60.0, 2),
        (120.0, 4),
    ],
)
def test_doctor_derives_analysis_stride_from_header_fps_with_tolerance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_fps: float,
    expected_stride: int,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"container headers are supplied by the fake")
    fake_av = _FakeAV(source_fps)

    def pyav_report(
        checks: list[dict[str, Any]],
        _config: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], _FakeAV]:
        doctor._add_check(
            checks,
            "runtime.pyav",
            True,
            "fake PyAV is available",
        )
        return {"installed": True, "version": "12.0.0"}, fake_av

    monkeypatch.setattr(doctor, "_pyav_report", pyav_report)
    monkeypatch.setattr(
        doctor,
        "probe_video",
        lambda _path: SimpleNamespace(
            to_dict=lambda: {
                "path": str(source),
                "width": 640,
                "height": 360,
                "fps": source_fps,
                "frame_count": max(1, round(source_fps * 2.0)),
                "duration": 2.0,
            }
        ),
    )

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        config_overrides={"scan.max_analysis_fps": 30},
        input_path=source,
        check_models=False,
    )

    assert report["ready"] is True
    assert report["media"]["video_stream"]["fps"] == pytest.approx(source_fps)
    assert report["media"]["analysis"]["effective_frame_stride"] == expected_stride
    assert report["media"]["first_frame_decoded"] is True
    assert report["safety"]["input_frames_decoded"] is True
    assert all(container.closed for container in fake_av.containers)


def test_doctor_rejects_an_input_that_cannot_decode_a_first_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"header-only fake")
    fake_av = _FakeAV(30.0)
    monkeypatch.setattr(_FakeContainer, "decode", lambda self, stream: iter(()))

    def pyav_report(
        checks: list[dict[str, Any]],
        _config: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], _FakeAV]:
        doctor._add_check(checks, "runtime.pyav", True, "fake PyAV is available")
        return {"installed": True, "version": "12.0.0"}, fake_av

    monkeypatch.setattr(doctor, "_pyav_report", pyav_report)

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        input_path=source,
        check_models=False,
    )

    check = _checks_by_name(report)["media.input"]
    assert report["ready"] is False
    assert check["ok"] is False
    assert "first video frame" in check["message"]
    assert report["media"]["first_frame_decoded"] is False
    assert report["safety"]["input_frames_decoded"] is False
    assert all(container.closed for container in fake_av.containers)


def test_analysis_only_doctor_skips_render_capability_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[bool] = []

    def pyav_report(
        checks: list[dict[str, Any]],
        _config: dict[str, Any] | None,
        *,
        check_render_capabilities: bool = True,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
        received.append(check_render_capabilities)
        doctor._add_check(checks, "runtime.pyav", True, "fake PyAV is available")
        return {"installed": True, "version": "12.0.0"}, None

    monkeypatch.setattr(doctor, "_pyav_report", pyav_report)

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        check_models=False,
        check_render_capabilities=False,
    )

    assert received == [False]
    assert report["ready"] is True


def test_effective_pyav_aac_mode_is_checked_against_source_audio(
    stable_pyav: None,
) -> None:
    settings = {
        "video_output": {
            "backend": "pyav",
            "audio": {"redacted": "aac"},
        }
    }

    incompatible = doctor.diagnose_render_settings(
        settings,
        input_audio_codec="opus",
        input_audio_present=True,
    )
    compatible = doctor.diagnose_render_settings(
        settings,
        input_audio_codec="aac",
        input_audio_present=True,
    )

    assert _checks_by_name(incompatible)["runtime.audio_input"]["ok"] is False
    assert incompatible["ready"] is False
    assert _checks_by_name(compatible)["runtime.audio_input"]["ok"] is True
    assert compatible["ready"] is True


def test_ffmpeg_help_capability_rejects_an_unknown_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="Codec 'definitely_missing' is not recognized by FFmpeg.",
    )
    monkeypatch.setattr(doctor.subprocess, "run", lambda *_args, **_kwargs: completed)

    capability = doctor._ffmpeg_help_capability(
        "/usr/bin/ffmpeg",
        kind="encoder",
        name="definitely_missing",
    )

    assert capability["available"] is False
    assert capability["returncode"] == 1


def test_pyav_capability_rejects_an_encoder_pixel_format_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = SimpleNamespace(
        name="libx264",
        long_name="fake libx264",
        type="video",
        is_decoder=False,
        is_encoder=True,
        video_formats=[SimpleNamespace(name="yuv420p")],
    )
    fake_av = SimpleNamespace(
        __version__="12.0.0",
        Codec=lambda _name, _mode: codec,
        VideoFormat=lambda name: SimpleNamespace(name=name),
        format=SimpleNamespace(
            ContainerFormat=lambda name: SimpleNamespace(name=name),
        ),
    )
    original_import = doctor.importlib.import_module

    def import_module(name: str):
        return fake_av if name == "av" else original_import(name)

    monkeypatch.setattr(doctor.importlib, "import_module", import_module)
    checks: list[dict[str, Any]] = []

    doctor._pyav_report(
        checks,
        {
            "render": {
                "video_output": {
                    "backend": "pyav",
                    "encoder": "libx264",
                    "pixel_format": "rgb24",
                }
            }
        },
    )

    by_name = {item["name"]: item for item in checks}
    assert by_name["runtime.video_encoder"]["ok"] is True
    assert by_name["runtime.pixel_format"]["ok"] is False
    assert by_name["runtime.pixel_format"]["details"]["supported_pixel_formats"] == [
        "yuv420p"
    ]


def test_pyav_capability_rejects_an_invalid_encoder_preset() -> None:
    settings = {
        "video_output": {
            "backend": "pyav",
            "encoder": "libx264",
            "pixel_format": "yuv420p",
            "preset": "definitely-not-a-preset",
            "rate_control": {"mode": "crf", "quality": 18},
            "keyframe_interval": 60,
            "audio": {"redacted": "none"},
        }
    }

    report = doctor.diagnose_render_settings(
        settings,
        input_audio_present=False,
    )

    check = _checks_by_name(report)["runtime.encoder_options"]
    assert report["ready"] is False
    assert check["ok"] is False
    assert check["details"]["probe"] == "in_memory_codec_context_open"


@pytest.mark.skipif(doctor.shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_ffmpeg_capability_rejects_an_invalid_encoder_preset() -> None:
    report = doctor.diagnose_render_settings(
        {
            "video_output": {
                "backend": "ffmpeg",
                "encoder": "libx264",
                "pixel_format": "yuv420p",
                "preset": "definitely-not-a-preset",
                "rate_control": {"mode": "crf", "quality": 18},
                "keyframe_interval": 60,
                "audio": {"redacted": "none"},
            }
        },
        input_audio_present=False,
    )

    check = _checks_by_name(report)["runtime.encoder_options"]
    assert report["ready"] is False
    assert check["ok"] is False
    assert check["details"]["probe"] == "one_in_memory_synthetic_frame"


def test_ffmpeg_capability_probes_the_final_keyframe_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def run(command, **_kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(doctor.subprocess, "run", run)
    details = doctor._ffmpeg_encoder_settings_capability(
        "ffmpeg",
        {
            "encoder": "libx264",
            "pixel_format": "yuv420p",
            "preset": "medium",
            "rate_control": {"mode": "crf", "quality": 18},
            "keyframe_interval": 73,
        },
    )

    command = captured["command"]
    assert details["available"] is True
    assert command[command.index("-g") + 1] == "73"


def test_ffmpeg_aac_capability_probes_the_configured_bitrate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(doctor.subprocess, "run", run)
    details = doctor._ffmpeg_aac_settings_capability("ffmpeg", "224k")

    command = captured["command"]
    assert details["available"] is True
    assert details["probe"] == "one_in_memory_synthetic_audio_frame"
    assert command[command.index("-b:a") + 1] == "224k"
    assert captured["input"] == b"\0" * (1024 * 2 * 2)


@pytest.mark.skipif(doctor.shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_ffmpeg_capability_rejects_an_unencodable_audio_bitrate() -> None:
    report = doctor.diagnose_render_settings(
        {
            "video_output": {
                "backend": "ffmpeg",
                "encoder": "libx264",
                "pixel_format": "yuv420p",
                "preset": "medium",
                "rate_control": {"mode": "crf", "quality": 18},
                "keyframe_interval": 60,
                "audio": {"redacted": "aac", "bitrate": 10**200},
            }
        },
        input_audio_present=True,
        input_audio_codec="aac",
    )

    check = _checks_by_name(report)["runtime.audio_encoder"]
    assert report["ready"] is False
    assert check["ok"] is False
    assert check["details"]["configured_settings"]["returncode"] != 0


@pytest.mark.skipif(doctor.shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_ffmpeg_skips_aac_probe_when_source_has_no_audio() -> None:
    report = doctor.diagnose_render_settings(
        {
            "video_output": {
                "backend": "ffmpeg",
                "encoder": "libx264",
                "pixel_format": "yuv420p",
                "preset": "medium",
                "rate_control": {"mode": "crf", "quality": 18},
                "keyframe_interval": 60,
                "audio": {"redacted": "aac", "bitrate": 1},
            }
        },
        input_audio_present=False,
        target_modes=("redacted",),
    )

    check = _checks_by_name(report)["runtime.audio_encoder"]
    assert report["ready"] is True
    assert check["ok"] is True
    assert check["details"]["skipped"] is True
    assert check["details"]["input_audio_present"] is False


def test_audio_copy_uses_an_mp4_muxer_compatibility_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeAV:
        pass

    def pyav_report(
        checks: list[dict[str, Any]],
        _config: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], FakeAV]:
        doctor._add_check(checks, "runtime.pyav", True, "fake PyAV is available")
        return {"installed": True, "version": "12.0.0"}, FakeAV()

    monkeypatch.setattr(doctor, "_pyav_report", pyav_report)
    monkeypatch.setattr(
        doctor,
        "_mp4_audio_copy_capability",
        lambda _av, _path: {
            "available": False,
            "input_codec": "wavpack",
            "error": "codec not supported in MP4",
        },
    )
    source = tmp_path / "source.mkv"
    source.write_bytes(b"probe is mocked")

    report = doctor.diagnose_render_settings(
        {
            "video_output": {
                "backend": "pyav",
                "audio": {"redacted": "copy"},
            }
        },
        input_audio_codec="wavpack",
        input_audio_present=True,
        input_path=source,
    )

    check = _checks_by_name(report)["runtime.audio_input"]
    assert report["ready"] is False
    assert check["ok"] is False
    assert check["details"]["copy_capability"]["input_codec"] == "wavpack"


def test_output_diagnostics_do_not_create_the_requested_directory(
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    destination = tmp_path / "not-created" / "nested"
    assert not destination.exists()

    report = doctor.run_doctor(
        output_dir=destination,
        check_models=False,
    )

    assert report["ready"] is True
    assert report["output"]["exists"] is False
    assert report["output"]["creatable"] is True
    assert report["output"]["writability_probe"] == ("os.access_without_file_creation")
    assert not destination.exists()


def test_check_models_false_skips_a_missing_selected_package(
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    root = tmp_path / "empty-insightface-root"

    report = doctor.run_doctor(
        config_path=DEFAULT_CONFIG_PATH,
        config_overrides={"models.root": str(root)},
        check_models=False,
    )

    check = _checks_by_name(report)["models.selected_package"]
    assert report["ready"] is True
    assert report["models"]["checked"] is False
    assert report["models"]["skipped"] is True
    assert check["ok"] is True
    assert check["details"]["skipped"] is True
    assert not root.exists()


def test_doctor_payload_is_strict_json_compatible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stable_pyav: None,
) -> None:
    monkeypatch.setattr(
        doctor, "DEFAULT_INSIGHTFACE_ROOT", str(tmp_path / "empty-root")
    )

    report = doctor.run_doctor(
        config_path=tmp_path / "missing.yaml",
        input_path=tmp_path / "missing.mp4",
        output_dir=tmp_path / "future-output",
    )

    assert report["ok"] is True
    assert report["ready"] is False
    encoded = json.dumps(report, allow_nan=False)
    assert json.loads(encoded) == report
