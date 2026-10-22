from __future__ import annotations

import json
import re
from copy import deepcopy

import pytest

from insightface.app.privateframe import cli, doctor, pipeline


def _valid_result() -> dict[str, object]:
    return {
        "format": "privateframe-result",
        "schema_version": 1,
        "observations": [
            {
                "frame_idx": 0,
                "box": [1.0, 2.0, 11.0, 12.0],
                "track_id": "t00000",
                "source": "detector",
                "extra_forward_compatible_field": {"kept": True},
            }
        ],
        "source_video": {
            "file_name": "input.mp4",
            "coordinate_system": "pixel_xyxy",
            "timing_contract": "cfr_frame_index",
            "frame_index_origin": 0,
            "metadata": {
                "width": 1280,
                "height": 720,
                "fps": 30.0,
                "frame_count": 1,
                "duration": 1.0 / 30.0,
            },
        },
        "render_defaults": {},
        "recognition": {"enabled": False},
    }


def _invalid_result(case: str) -> dict[str, object] | list[object]:
    value = deepcopy(_valid_result())
    if case == "root":
        return []
    if case == "format":
        value["format"] = "other-result"
    elif case == "schema_type":
        value["schema_version"] = True
    elif case == "schema_value":
        value["schema_version"] = 2
    elif case == "source_video":
        value["source_video"] = []
    elif case == "source_file_name":
        value["source_video"]["file_name"] = ""
    elif case == "source_coordinate_system":
        value["source_video"]["coordinate_system"] = "normalized_xyxy"
    elif case == "source_timing_contract":
        value["source_video"]["timing_contract"] = "pts"
    elif case == "source_frame_origin":
        value["source_video"]["frame_index_origin"] = 1
    elif case == "source_metadata":
        value["source_video"]["metadata"] = []
    elif case == "source_width_float":
        value["source_video"]["metadata"]["width"] = 640.5
    elif case == "source_frame_count_zero":
        value["source_video"]["metadata"]["frame_count"] = 0
    elif case == "source_fps_nonfinite":
        value["source_video"]["metadata"]["fps"] = float("nan")
    elif case == "source_duration_zero":
        value["source_video"]["metadata"]["duration"] = 0.0
    elif case == "render_defaults":
        value["render_defaults"] = []
    elif case == "recognition":
        value["recognition"] = []
    elif case == "recognition_enabled":
        value["recognition"]["enabled"] = "false"
    elif case == "observations":
        value["observations"] = {}
    elif case == "observation":
        value["observations"] = [None]
    elif case == "frame_negative":
        value["observations"][0]["frame_idx"] = -1
    elif case == "frame_float":
        value["observations"][0]["frame_idx"] = 0.0
    elif case == "frame_bool":
        value["observations"][0]["frame_idx"] = True
    elif case == "frame_out_of_range":
        value["observations"][0]["frame_idx"] = 1
    elif case == "track_id":
        value["observations"][0]["track_id"] = ""
    elif case == "source":
        value["observations"][0]["source"] = None
    elif case == "identity_unconfirmed":
        value["observations"][0]["identity_unconfirmed"] = 1
    elif case == "box_length":
        value["observations"][0]["box"] = [1.0, 2.0, 11.0]
    elif case == "box_type":
        value["observations"][0]["box"] = [1.0, 2.0, "11", 12.0]
    elif case == "box_nonfinite":
        value["observations"][0]["box"] = [1.0, 2.0, float("inf"), 12.0]
    elif case == "box_x_order":
        value["observations"][0]["box"] = [11.0, 2.0, 11.0, 12.0]
    elif case == "box_y_order":
        value["observations"][0]["box"] = [1.0, 12.0, 11.0, 2.0]
    else:  # pragma: no cover - keeps the case table exhaustive
        raise AssertionError(f"unknown invalid result case: {case}")
    return value


INVALID_CASES = [
    pytest.param("root", "root must be an object", id="root-object"),
    pytest.param("format", "format must be privateframe-result", id="format"),
    pytest.param("schema_type", "schema_version must be an integer", id="schema-type"),
    pytest.param("schema_value", "schema_version must be 1", id="schema-value"),
    pytest.param("source_video", "source_video must be an object", id="source-video"),
    pytest.param(
        "source_file_name",
        "source_video.file_name must be a string",
        id="source-file-name",
    ),
    pytest.param(
        "source_coordinate_system",
        "source_video.coordinate_system must be 'pixel_xyxy'",
        id="source-coordinate-system",
    ),
    pytest.param(
        "source_timing_contract",
        "source_video.timing_contract must be 'cfr_frame_index'",
        id="source-timing-contract",
    ),
    pytest.param(
        "source_frame_origin",
        "source_video.frame_index_origin must be 0",
        id="source-frame-origin",
    ),
    pytest.param(
        "source_metadata",
        "source_video.metadata must be an object",
        id="source-metadata",
    ),
    pytest.param(
        "source_width_float",
        "source_video.metadata.width must be a positive integer",
        id="source-width-float",
    ),
    pytest.param(
        "source_frame_count_zero",
        "source_video.metadata.frame_count must be positive",
        id="source-frame-count-zero",
    ),
    pytest.param(
        "source_fps_nonfinite",
        "source_video.metadata.fps must be a finite number",
        id="source-fps-nonfinite",
    ),
    pytest.param(
        "source_duration_zero",
        "source_video.metadata.duration must be positive and finite",
        id="source-duration-zero",
    ),
    pytest.param(
        "render_defaults",
        "render_defaults must be an object",
        id="render-defaults",
    ),
    pytest.param("recognition", "recognition must be an object", id="recognition"),
    pytest.param(
        "recognition_enabled",
        "recognition.enabled must be boolean",
        id="recognition-enabled",
    ),
    pytest.param(
        "observations", "observations must be an array", id="observations-array"
    ),
    pytest.param(
        "observation", "observations[0] must be an object", id="observation-object"
    ),
    pytest.param(
        "frame_negative",
        "frame_idx must be a non-negative integer",
        id="frame-negative",
    ),
    pytest.param(
        "frame_float",
        "frame_idx must be a non-negative integer",
        id="frame-float",
    ),
    pytest.param(
        "frame_bool",
        "frame_idx must be a non-negative integer",
        id="frame-bool",
    ),
    pytest.param(
        "frame_out_of_range",
        "frame_idx exceeds the source frame count",
        id="frame-out-of-range",
    ),
    pytest.param("track_id", "track_id must be a non-empty string", id="track-id"),
    pytest.param("source", "source must be a non-empty string", id="source"),
    pytest.param("identity_unconfirmed", "identity_unconfirmed must be boolean", id="identity-unconfirmed"),
    pytest.param(
        "box_length", "box must be an array of four finite numbers", id="box-length"
    ),
    pytest.param("box_type", "box must contain four finite numbers", id="box-type"),
    pytest.param(
        "box_nonfinite", "box must contain four finite numbers", id="box-nonfinite"
    ),
    pytest.param(
        "box_x_order", "box must satisfy x2 > x1 and y2 > y1", id="box-x-order"
    ),
    pytest.param(
        "box_y_order", "box must satisfy x2 > x1 and y2 > y1", id="box-y-order"
    ),
]


def test_validate_result_document_preserves_existing_valid_documents():
    value = _valid_result()

    validated = pipeline.validate_result_document(value)

    assert validated is value
    assert validated["observations"][0]["extra_forward_compatible_field"] == {
        "kept": True
    }


def test_validate_result_document_accepts_legacy_v1_provenance_fields():
    value = _valid_result()
    value["source_video"].update(
        {"path": "/old/location/input.mp4", "sha256": "0" * 64, "bytes": 123}
    )
    value["analysis"] = {
        "artifacts_level": "debug",
        "provider": "CPUExecutionProvider",
        "git": {"commit": "legacy", "dirty": True},
    }
    value["observations"][0]["endpoint_repair_reason"] = (
        "interpolate_unanchored_endpoint"
    )

    assert pipeline.validate_result_document(value) is value


@pytest.mark.parametrize(("case", "message"), INVALID_CASES)
def test_validate_result_document_rejects_malformed_structure(case, message):
    with pytest.raises((TypeError, ValueError), match=re.escape(message)):
        pipeline.validate_result_document(_invalid_result(case))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        pytest.param("format", "format must be privateframe-result", id="format"),
        pytest.param("schema_value", "schema_version must be 1", id="schema"),
        pytest.param(
            "source_width_float",
            "source_video.metadata.width must be a positive integer",
            id="source-width",
        ),
        pytest.param(
            "source_fps_nonfinite",
            "source_video.metadata.fps must be a finite number",
            id="source-fps",
        ),
        pytest.param(
            "observations", "observations must be an array", id="observations"
        ),
        pytest.param(
            "frame_negative",
            "frame_idx must be a non-negative integer",
            id="frame-index",
        ),
        pytest.param(
            "box_nonfinite",
            "box must contain four finite numbers",
            id="box-finite",
        ),
        pytest.param(
            "box_x_order",
            "box must satisfy x2 > x1 and y2 > y1",
            id="box-order",
        ),
    ],
)
def test_render_and_render_dry_run_share_result_validation(
    case,
    message,
    monkeypatch,
    tmp_path,
    capsys,
):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"dry-run source fixture")
    result_path = tmp_path / "input_privateframe.json"
    result_path.write_text(json.dumps(_invalid_result(case)), encoding="utf-8")
    output_path = tmp_path / "input_privateframe.mp4"

    with pytest.raises((TypeError, ValueError), match=re.escape(message)):
        pipeline.render_streaming_artifacts(
            input_path=source,
            result_path=result_path,
            redacted_path=output_path,
        )

    monkeypatch.setattr(
        doctor,
        "run_doctor",
        lambda **_kwargs: {
            "ok": True,
            "ready": True,
            "checks": [],
            "runtime": {},
            "models": {},
            "media": {},
            "output": {},
            "safety": {},
        },
    )
    exit_code = cli.main(
        [
            "render",
            "--input",
            str(source),
            "--result",
            str(result_path),
            "--redacted",
            str(output_path),
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    result_check = next(
        item for item in payload["checks"] if item["name"] == "render_result"
    )
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["ready"] is False
    assert result_check["ok"] is False
    assert message in result_check["message"]
