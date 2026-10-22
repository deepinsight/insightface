from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from insightface.app.privateframe import artifact_render, pipeline
from insightface.app.privateframe.artifact_render import RenderTarget
from insightface.app.privateframe.artifacts import write_json
from insightface.app.privateframe.base_config import (
    DEFAULT_CONFIG_PATH,
    read_default_config,
)


def test_user_observations_keep_only_render_and_editor_fields() -> None:
    values = [
        {
            "frame_idx": 7,
            "track_id": "t00003",
            "box": [10.0, 20.0, 30.0, 40.0],
            "source": "kalman_optical_flow",
            "reduced_assurance": True,
            "endpoint_repair_reason": "interpolate_unanchored_endpoint",
            "motion_box": [10.0, 20.0, 30.0, 40.0],
            "source_aabb": [10.0, 20.0, 30.0, 40.0],
            "raw_output_box": [9.0, 19.0, 29.0, 39.0],
            "detector_landmarks": [[1.0, 2.0]] * 5,
            "verifier_face_probability": 0.99,
        }
    ]

    assert pipeline._export_observations(values) == [
        {
            "frame_idx": 7,
            "track_id": "t00003",
            "box": [10.0, 20.0, 30.0, 40.0],
            "source": "repaired",
            "reduced_assurance": True,
            "identity_unconfirmed": True,
        }
    ]


def test_internal_tracking_sources_are_normalized_for_users() -> None:
    assert pipeline._public_observation_source({"source": "detector"}) == "detector"
    assert (
        pipeline._public_observation_source(
            {
                "source": "kalman_optical_flow",
                "geometry_source": "detector_anchor_interpolation",
            }
        )
        == "interpolated"
    )
    assert (
        pipeline._public_observation_source({"source": "kalman_optical_flow"})
        == "tracked"
    )


def test_user_recognition_keeps_reference_audit_and_render_decisions() -> None:
    references = {
        "accepted_images": 1, "skipped_images": 1,
        "files": [{"file": "photo.jpg", "detected_face_count": 2,
                   "selected_box": [1, 2, 31, 42]}],
        "skipped": [{"file": "empty.png", "reason": "no_face"}],
        "fingerprint": "abc",
    }
    record = {"status": "CONFIRMED", "matched_reference_files": ["photo.jpg"],
              "similarity": 0.83, "reason": "confirmed_reference_set"}
    value = {
        "enabled": True, "references": references,
        "statistics": {"recognizer_calls": 8},
        "tracks": {"t00001": {**record, "frame_indices": [1, 20, 40]}},
    }
    exported = pipeline._export_recognition(value)
    assert exported == {"enabled": True, "references": references,
                        "tracks": {"t00001": record}}
    assert exported["references"] is not references
    assert pipeline._export_recognition(
        {"enabled": False, "reason": "policy_all"}
    ) == {"enabled": False}


@pytest.fixture(params=["output_directory", "workdir"])
def analysis_job(request, monkeypatch, tmp_path):
    """Exercise real artifact writes with a model-free, one-frame scan result."""

    source = tmp_path / "input.mp4"
    source.write_bytes(b"source video fixture")
    workdir = tmp_path / "work"
    workdir.mkdir()
    result_path = (
        workdir / pipeline.RESULT_FILENAME
        if request.param == "workdir"
        else tmp_path / "input_privateframe.json"
    )
    report_path = result_path.with_name(f"{result_path.stem}.dev.json")
    config = read_default_config()
    config["runtime"]["resolved_provider"] = "CPUExecutionProvider"
    scan_result = {
        key: 0
        for key in (
            "reverse_jobs", "reverse_frames", "long_gap_reanchors",
            "discarded_unanchored_tail_frames", "endpoint_affine_jobs",
            "endpoint_affine_frames", "endpoint_affine_published_frames",
            "interpolate_endpoint_jobs", "interpolate_endpoint_frames",
            "interpolate_endpoint_published_frames", "interpolate_endpoint_seconds",
            "fragment_stitches", "bidirectional_gap_jobs", "bidirectional_gap_frames",
            "bidirectional_accepted_frames", "bidirectional_rejected_frames",
            "bidirectional_review_resolutions", "bidirectional_skipped_jobs",
            "bidirectional_association_attempts", "bidirectional_association_rescues",
        )
    }
    scan_result.update(
        {
            "scan": {
                "metadata": {
                    "width": 320, "height": 240, "fps": 30.0,
                    "frame_count": 1, "duration": 1 / 30,
                },
                "frame_count": 1,
                "frames": [{"frame_idx": 0, "time_seconds": 0.0}],
                "detections": [],
            },
            "tracks": [],
            "tracking": {"observations": []},
            "review": {"observations": [], "evidence": [], "accepted_tracks": 0},
            "bidirectional_audits": [],
            "interpolate_endpoint_reason_counts": {},
            "detector_sampling": {
                "analyzed_frames": 1, "regular_scan_frames": 1,
                "forced_scan_frames": 0, "skipped_scan_frames": 0,
                "max_analysis_fps": 30, "effective_frame_stride": 1,
            },
            "local_review_sampling": {
                "attempts": 0, "sampled_out": 0, "forced_attempts": 0,
                "verifier_calls": 0, "verifier_cache_hits": 0,
            },
            "cache": {"recent_frame_target_frames": 0},
        }
    )
    monkeypatch.setattr(pipeline, "load_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        pipeline, "run_stream", lambda *_args, **_kwargs: deepcopy(scan_result)
    )
    monkeypatch.setattr(pipeline, "_model_fingerprints", lambda _config: {})
    monkeypatch.setattr(pipeline, "_model_package_fingerprint", lambda _config: {})
    monkeypatch.setattr(
        pipeline, "active_face_detector", lambda _config: ("detector", {})
    )
    monkeypatch.setattr(pipeline, "git_version", lambda _root: {})

    def run(artifacts_level):
        config["output"]["artifacts_level"] = artifacts_level
        return pipeline.analyze_streaming_pipeline(
            config_path=DEFAULT_CONFIG_PATH,
            input_path=source,
            workdir=workdir,
            result_path=None if request.param == "workdir" else result_path,
        )

    return SimpleNamespace(result=result_path, report=report_path, run=run)


@pytest.mark.parametrize("artifacts_level", ["final", "audit", "debug"])
@pytest.mark.parametrize(
    "report_case",
    ["user_json", "other_result", "bad_json", "invalid_utf8", "unreadable"],
)
def test_analysis_preserves_unowned_development_report(
    analysis_job, monkeypatch, artifacts_level, report_case
) -> None:
    job = analysis_job
    previous_result = b'{"previous_result": true}'
    job.result.write_bytes(previous_result)
    contents = {
        "user_json": b'{"owned_by": "user"}',
        "other_result": (
            b'{"format": "privateframe-development-report", "result_file": "other.json"}'
        ),
        "bad_json": b"not JSON",
        "invalid_utf8": b"\xff\xfe",
        "unreadable": b"unreadable user file",
    }[report_case]
    job.report.write_bytes(contents)
    if report_case == "unreadable":
        read_text = Path.read_text

        def read(path, *args, **kwargs):
            if path == job.report:
                raise PermissionError("report is not readable")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)

    if artifacts_level == "final":
        summary = job.run(artifacts_level)
        document = json.loads(job.result.read_text(encoding="utf-8"))
        assert pipeline.validate_result_document(document) is document
        assert document["source_video"]["file_name"] == "input.mp4"
        assert "development_report" not in summary
    else:
        with pytest.raises(FileExistsError, match="unrecognized development report"):
            job.run(artifacts_level)
        assert job.result.read_bytes() == previous_result
    assert job.report.read_bytes() == contents


@pytest.mark.parametrize("artifacts_level", ["final", "audit", "debug"])
@pytest.mark.parametrize("previous_report", [False, True])
def test_analysis_manages_only_its_own_development_report(
    analysis_job, artifacts_level, previous_report
) -> None:
    job = analysis_job
    if previous_report:
        job.report.write_text(
            json.dumps({
                "format": "privateframe-development-report",
                "schema_version": 1,
                "result_file": job.result.name,
                "previous_report": True,
            }),
            encoding="utf-8",
        )

    job.run(artifacts_level)

    document = json.loads(job.result.read_text(encoding="utf-8"))
    assert document["format"] == "privateframe-result"
    if artifacts_level == "final":
        assert not job.report.exists()
    else:
        report = json.loads(job.report.read_text(encoding="utf-8"))
        assert report["format"] == "privateframe-development-report"
        assert report["result_file"] == job.result.name
        assert report["artifacts_level"] == artifacts_level
        assert "previous_report" not in report


@pytest.mark.parametrize("artifacts_level", ["final", "audit", "debug"])
def test_analysis_keeps_previous_result_and_report_when_result_write_fails(
    analysis_job, monkeypatch, artifacts_level
) -> None:
    job = analysis_job
    previous_result = b'{"previous_result": true}'
    previous_report = json.dumps({
        "format": "privateframe-development-report",
        "schema_version": 1,
        "result_file": job.result.name,
    }).encode()
    job.result.write_bytes(previous_result)
    job.report.write_bytes(previous_report)

    def write(path, value, **kwargs):
        if Path(path) == job.result:
            raise OSError("result write failed")
        return write_json(path, value, **kwargs)

    monkeypatch.setattr(pipeline, "write_json", write)
    with pytest.raises(OSError, match="result write failed"):
        job.run(artifacts_level)

    assert job.result.read_bytes() == previous_result
    assert job.report.read_bytes() == previous_report


def test_compact_json_writer_uses_no_formatting_whitespace(tmp_path) -> None:
    path = tmp_path / "result.json"

    write_json(path, {"b": [1, 2], "a": {"value": True}}, indent=None)

    assert path.read_text(encoding="utf-8") == (
        '{"a":{"value":true},"b":[1,2]}\n'
    )


def test_renderer_uses_lightweight_source_geometry_checks(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        artifact_render,
        "probe_video",
        lambda _path: SimpleNamespace(width=320, height=240, fps=30.0),
    )

    with pytest.raises(ValueError, match="source video dimensions do not match"):
        artifact_render.render_artifacts(
            source=tmp_path / "input.mp4",
            targets=[RenderTarget("redacted", tmp_path / "output.mp4")],
            settings={},
            analysis_result={
                "source_video": {
                    "metadata": {
                        "width": 640,
                        "height": 360,
                        "fps": 30.0,
                        "frame_count": 1,
                    }
                },
                "observations": [],
            },
        )
