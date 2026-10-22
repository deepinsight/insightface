import logging
from types import SimpleNamespace

import pytest
from insightface.app.privateframe import revalidation
from insightface.app.privateframe.revalidation import LocalReviewer, _finalize


def _region_reviewer(angles=(0, 90, -90), containment_threshold=0.8):
    return LocalReviewer(
        {
            "models": {"detection": {"max_detections": 300}},
            "scan": {"global_nms_iou": 0.45, "containment_threshold": containment_threshold},
            "revalidation": {"input_size": 160, "angles": list(angles), "confidence_threshold": 0.18},
        },
        face_analysis=SimpleNamespace(models={"detection": object()}),
        verifier=object(),
    )


def test_region_detection_restores_global_coordinates_and_keeps_two_faces(monkeypatch):
    import numpy as np

    reviewer = _region_reviewer()
    calls = []
    boxes = [(10, 20, 40, 70), (80, 20, 120, 70)]

    def detect(host, image, **kwargs):
        angle = reviewer.settings["angles"][len(calls)]
        calls.append((host, image.shape, kwargs))
        result = []
        for x1, y1, x2, y2 in boxes:
            point = [x1 + 5, y1 + 5]
            if angle == 90:
                box = [y1, 160 - x2, y2, 160 - x1]
                point = [point[1], 159 - point[0]]
            elif angle == -90:
                box = [100 - y2, x1, 100 - y1, x2]
                point = [99 - point[1], point[0]]
            else:
                box = [x1, y1, x2, y2]
            result.append({"box": box, "confidence": 0.9, "landmarks": [point] * 5})
        return result

    monkeypatch.setattr(revalidation, "detect_faces", detect)
    result = reviewer.detect_region(np.zeros((100, 160, 3), dtype=np.uint8), origin=(-30, 400))
    assert result["complete"] is True
    assert result["detector_calls"] == 3
    assert result["roi"] == [-30, 400, 130, 500]
    assert len(result["detections"]) == 2
    assert result["detections"][0]["box"] == [-20, 420, 10, 470]
    assert result["detections"][0]["landmarks"] == [[-15, 425]] * 5
    assert result["detections"][1]["box"] == [50, 420, 90, 470]
    assert all(host is reviewer.face_analysis for host, _shape, _kwargs in calls)
    assert all(kwargs["input_sizes"] == [160] for _host, _shape, kwargs in calls)


def test_region_detection_call_budget_reports_incomplete_evidence(monkeypatch):
    import numpy as np

    reviewer = _region_reviewer()
    calls = []

    def detect(*_args, **_kwargs):
        calls.append(1)
        return [{"box": [0, 0, 10, 10], "confidence": 0.9}]

    monkeypatch.setattr(revalidation, "detect_faces", detect)
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    empty = reviewer.detect_region(image, max_calls=0)
    assert empty["detector_calls"] == 0 and empty["detections"] == []
    assert empty["complete"] is False and calls == []
    partial = reviewer.detect_region(image, max_calls=1)
    assert partial["detector_calls"] == 1 and len(calls) == 1
    assert partial["complete"] is False
    assert partial["angles"] == [0]
    with pytest.raises(ValueError, match="max_calls"):
        reviewer.detect_region(image, max_calls=-1)


def test_region_single_angle_is_complete_and_does_not_change_tracking_review(monkeypatch):
    import numpy as np

    reviewer = _region_reviewer()
    reviewer.settings.update({
        "crop_expansion": 2.0, "match_max_area_ratio": 6.0,
        "match_max_center_distance": 0.85, "match_min_iou": 0.08,
        "match_min_containment": 0.35,
    })
    configured_angles = reviewer.settings["angles"]
    rotations = []
    rotate = revalidation.rotate_image

    def record_rotation(image, angle):
        rotations.append(angle)
        return rotate(image, angle)

    def detect(_host, image, **_kwargs):
        height, width = image.shape[:2]
        return [
            {"box": [width * 0.1, height * 0.2, width * 0.3, height * 0.7], "confidence": 0.9},
            {"box": [width * 0.6, height * 0.2, width * 0.9, height * 0.7], "confidence": 0.8},
        ]

    monkeypatch.setattr(revalidation, "rotate_image", record_rotation)
    monkeypatch.setattr(revalidation, "detect_faces", detect)
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    selected = [0]
    single = reviewer.detect_region(image, origin=(-30, 400), angles=selected, max_calls=1)
    assert single["complete"] is True
    assert single["detector_calls"] == 1 and single["angles"] == [0]
    assert [face["box"] for face in single["detections"]] == [
        [-14.0, 420.0, 18.0, 470.0], [66.0, 420.0, 114.0, 470.0],
    ]
    assert selected == [0]
    assert reviewer.settings["angles"] is configured_angles
    assert configured_angles == [0, 90, -90]

    default = reviewer.detect_region(image)
    assert default["complete"] is True
    assert default["detector_calls"] == 3 and default["angles"] == [0, 90, -90]
    reviewer.local_match(image, [20.0, 20.0, 40.0, 40.0])
    assert rotations == [0, 0, 90, -90, 0, 90, -90]
    assert reviewer.settings["angles"] is configured_angles


def test_region_explicit_angle_order_and_budget_are_independent(monkeypatch):
    import numpy as np

    calls = []

    def detect(_host, image, **_kwargs):
        calls.append(image.shape)
        return [{"box": [10.0, 20.0, 30.0, 40.0], "confidence": 0.9}]

    monkeypatch.setattr(revalidation, "detect_faces", detect)
    reviewer = _region_reviewer()
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    empty = reviewer.detect_region(image, angles=(0,), max_calls=0)
    assert empty["complete"] is False and empty["angles"] == []
    assert empty["detector_calls"] == 0 and empty["detections"] == [] and calls == []
    partial = reviewer.detect_region(image, origin=(5, 6), angles=(90, 0), max_calls=1)
    assert partial["complete"] is False
    assert partial["angles"] == [90] and partial["detector_calls"] == 1
    assert calls == [(160, 100, 3)]
    assert partial["detections"][0]["box"] == [125.0, 16.0, 145.0, 36.0]


@pytest.mark.parametrize("angles", [[], (), [True], [False], [0.0], [90.5], ["0"], [None], [180], [0, 0], [0, 90, 90], 0, "0"])
def test_region_rejects_invalid_angle_overrides_before_inference(monkeypatch, angles):
    import numpy as np

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("invalid angles reached the detector")

    monkeypatch.setattr(revalidation, "detect_faces", fail_if_called)
    with pytest.raises(ValueError, match="angles"):
        _region_reviewer().detect_region(np.zeros((30, 30, 3), dtype=np.uint8), angles=angles)


@pytest.mark.parametrize("angles", [None, [0]])
def test_region_detection_errors_propagate_instead_of_becoming_no_face(monkeypatch, angles):
    import numpy as np

    def fail(*_args, **_kwargs):
        raise RuntimeError("detector execution failed")

    monkeypatch.setattr(revalidation, "detect_faces", fail)
    with pytest.raises(RuntimeError, match="detector execution failed"):
        _region_reviewer().detect_region(np.zeros((30, 30, 3), dtype=np.uint8), angles=angles)


def test_region_multiview_nms_uses_scan_threshold_not_local_equivalence(monkeypatch):
    import numpy as np

    reviewer = _region_reviewer(angles=(0, 90), containment_threshold=1.0)
    calls = []

    def detect(*_args, **_kwargs):
        angle_index = len(calls) % 2
        calls.append(1)
        # After inverse rotation the second box is [20, 0, 120, 100].
        return [{"box": [0, 0, 100, 100] if angle_index == 0 else [0, 80, 100, 180],
                 "confidence": 0.9 - angle_index * 0.1}]

    monkeypatch.setattr(revalidation, "detect_faces", detect)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    assert len(reviewer.detect_region(image)["detections"]) == 1
    assert len(reviewer.detect_region(image, nms_iou=0.95)["detections"]) == 2


def test_local_reviewer_accepts_detection_only_host_and_explicit_verifier() -> None:
    detector = object()
    verifier = object()
    detection_host = SimpleNamespace(models={"detection": detector})
    config = {
        "models": {"detection": {"max_detections": 300}},
        "revalidation": {},
    }

    reviewer = LocalReviewer(
        config,
        detector=detector,
        verifier=verifier,
        face_analysis=detection_host,
    )

    assert reviewer.face_analysis is detection_host
    assert reviewer.detector is detector
    assert reviewer.verifier is verifier


def test_finalize_keeps_one_best_geometry_per_accepted_track_frame() -> None:
    track = {
        "track_id": "face",
        "accepted": True,
        "accepted_intervals": [[5, 5]],
    }
    motion_box = [10.0, 10.0, 30.0, 30.0]
    detector_box = [11.0, 11.0, 31.0, 31.0]

    output = _finalize(
        [track],
        {
            "detections": [
                {
                    "track_id": "face",
                    "frame_idx": 5,
                    "source": "detector",
                    "box": detector_box,
                    "confidence": 0.9,
                }
            ]
        },
        {
            "observations": [
                {
                    "track_id": "face",
                    "frame_idx": 5,
                    "source": "kalman_optical_flow",
                    "box": motion_box,
                    "motion_box": motion_box,
                }
            ]
        },
        [],
    )

    assert len(output) == 1
    assert output[0]["track_id"] == "face"
    assert output[0]["source"] == "detector"
    assert output[0]["box"] == detector_box


def test_finalize_keeps_distinct_tracks_until_output_boxes_are_stabilized() -> None:
    boxes = {
        "reference": [0.0, 0.0, 100.0, 100.0],
        "overlapping": [15.0, 0.0, 115.0, 100.0],
    }
    tracks = [
        {
            "track_id": track_id,
            "accepted": True,
            "accepted_intervals": [[5, 5]],
        }
        for track_id in boxes
    ]
    detections = [
        {
            "track_id": track_id,
            "frame_idx": 5,
            "source": "detector",
            "box": box,
            "confidence": 0.9,
        }
        for track_id, box in boxes.items()
    ]

    output = _finalize(
        tracks,
        {"detections": detections},
        {"observations": []},
        [],
    )

    assert {item["track_id"] for item in output} == set(boxes)


def test_stitched_accepted_intervals_do_not_bridge_frames_without_geometry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fragment identity does not manufacture boxes in an unobserved gap."""

    caplog.set_level(logging.DEBUG, logger=revalidation.__name__)
    summary = {
        "detector_source_frames": 4,
        "local_match_fraction": 1.0,
        "confidence_p50": 0.9,
        "verifier_p50": 0.9,
    }
    monkeypatch.setattr(
        revalidation,
        "_summary",
        lambda *_args, **_kwargs: dict(summary),
    )
    monkeypatch.setattr(
        revalidation,
        "_admission_decision",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "admission_path": "standard",
        },
    )
    track = {
        "track_id": "stitched",
        "stitched_track_ids": ["left-fragment", "right-fragment"],
    }
    evidence = [
        {
            "track_id": "stitched",
            "frame_idx": frame_idx,
            "source": "tracking",
            "box": [10.0, 10.0, 30.0, 30.0],
        }
        for frame_idx in (10, 11, 14, 15)
    ]
    config = {
        "revalidation": {
            "policy": {
                "continuity": {
                    "segment_max_center_jump": 1.0,
                    "segment_max_area_ratio": 2.0,
                }
            }
        },
        "tracking": {
            "fragment_stitching": {
                # The old interval merger bridged this two-frame hole merely
                # because it was below this threshold.
                "max_interval_gap_frames": 8,
            }
        },
        "recognition": {"mode": "all"},
        "scan": {
            "global_nms_iou": 0.45,
            "containment_threshold": 0.80,
        },
    }

    revalidation.finalize_precomputed(
        {"frames": [], "detections": []},
        [track],
        {"observations": [], "shadows": []},
        evidence,
        config,
        detector_frame_stride=1,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    diagnostics = [record for record in caplog.records if record.name == revalidation.__name__]
    assert len(diagnostics) == 1
    assert diagnostics[0].levelno == logging.DEBUG
    assert "[admission/onnx] stitched" in diagnostics[0].getMessage()

    assert track["accepted"] is True
    assert track["accepted_intervals"] == [[10, 11], [14, 15]]


def test_overall_admission_falls_back_to_each_actual_continuity_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole-track admission must not invent geometry or publish no interval."""

    def summary(
        values: list[dict[str, object]], *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        return {
            "scope": "whole" if len(values) == 4 else "segment",
            "detector_source_frames": len(values),
            "local_match_fraction": 1.0,
            "confidence_p50": 0.9,
            "verifier_p50": 0.9,
        }

    monkeypatch.setattr(revalidation, "_summary", summary)
    monkeypatch.setattr(
        revalidation,
        "_admission_decision",
        lambda value, *_args, **_kwargs: {
            "accepted": value["scope"] == "whole",
            "admission_path": "standard",
        },
    )
    track = {"track_id": "accepted-as-a-whole"}
    evidence = [
        {
            "track_id": track["track_id"],
            "frame_idx": frame_idx,
            "source": "tracking",
            "box": [10.0, 10.0, 30.0, 30.0],
        }
        for frame_idx in (10, 11, 14, 15)
    ]
    config = {
        "revalidation": {
            "policy": {
                "continuity": {
                    "segment_max_center_jump": 1.0,
                    "segment_max_area_ratio": 2.0,
                }
            }
        },
        "tracking": {"fragment_stitching": {}},
    }

    revalidation.finalize_precomputed(
        {"frames": [], "detections": []},
        [track],
        {"observations": []},
        evidence,
        config,
        detector_frame_stride=1,
    )

    assert track["accepted"] is True
    assert track["accepted_intervals"] == [[10, 11], [14, 15]]
