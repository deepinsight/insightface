from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import numpy as np

from insightface.model_zoo.scrfd import SCRFD


def _bare_scrfd(*, input_sizes=((64, 64),)):
    detector = SCRFD.__new__(SCRFD)
    detector.det_thresh = 0.5
    detector.nms_thresh = 0.4
    detector.static_input_size = None
    detector.input_size = input_sizes[-1]
    detector.input_sizes = list(input_sizes)
    detector.use_kps = False
    return detector


def _candidate(score):
    return np.asarray([[1.0, 2.0, 5.0, 6.0, score]], dtype=np.float32)


def test_scrfd_embedded_preprocessing_builds_uint8_blob():
    detector = SCRFD.__new__(SCRFD)
    detector.input_mean = 0.0
    detector.input_std = 1.0
    detector.input_dtype = np.uint8
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30

    blob = detector._prepare_input_blob(image, (2, 2))

    assert blob.dtype == np.uint8
    assert blob.flags.c_contiguous
    assert np.array_equal(blob[0, :, 0, 0], [30, 20, 10])


def test_scrfd_mean_std_preprocessing_keeps_float32_blob():
    detector = SCRFD.__new__(SCRFD)
    detector.input_mean = 10.0
    detector.input_std = 10.0
    detector.input_dtype = np.float32
    image = np.full((2, 2, 3), 30, dtype=np.uint8)

    blob = detector._prepare_input_blob(image, (2, 2))

    assert blob.dtype == np.float32
    assert np.allclose(blob, 2.0)


def test_detect_uses_default_threshold_for_legacy_calls():
    detector = _bare_scrfd()
    thresholds = []

    def candidates(_image, _input_size, threshold):
        thresholds.append(threshold)
        return _candidate(0.8), None

    detector._detect_candidates = candidates
    detector.nms = lambda _detections: [0]

    detections, landmarks = detector.detect(
        np.zeros((8, 8, 3), dtype=np.uint8),
        (64, 64),
        0,
        "default",
    )

    assert thresholds == [0.5]
    assert np.allclose(detections[:, 4], [0.8])
    assert landmarks is None


def test_detect_call_threshold_does_not_mutate_shared_default():
    detector = _bare_scrfd()
    thresholds = []

    def candidates(_image, _input_size, threshold):
        thresholds.append(threshold)
        return _candidate(threshold), None

    detector._detect_candidates = candidates
    detector.nms = lambda _detections: [0]

    detections, _landmarks = detector.detect(
        np.zeros((8, 8, 3), dtype=np.uint8),
        input_size=(64, 64),
        det_thresh=0.25,
    )

    assert thresholds == [0.25]
    assert np.allclose(detections[:, 4], [0.25])
    assert detector.det_thresh == 0.5


def test_multiscale_candidates_are_concatenated_before_one_global_nms():
    detector = _bare_scrfd(input_sizes=((64, 64), (128, 128)))
    nms_inputs = []

    def candidates(_image, input_size, _threshold):
        score = 0.6 if input_size == (64, 64) else 0.9
        return _candidate(score), None

    def nms(detections):
        nms_inputs.append(detections.copy())
        return list(range(len(detections)))

    detector._detect_candidates = candidates
    detector.nms = nms

    detections, _landmarks = detector.detect(
        np.zeros((8, 8, 3), dtype=np.uint8),
        input_size=[(64, 64), (128, 128)],
        det_thresh=0.2,
    )

    assert len(nms_inputs) == 1
    assert np.allclose(nms_inputs[0][:, 4], [0.9, 0.6])
    assert np.allclose(detections[:, 4], [0.9, 0.6])


def test_global_sort_preserves_input_order_for_equal_scores():
    detector = _bare_scrfd(input_sizes=((64, 64), (128, 128)))
    nms_inputs = []

    def candidates(_image, input_size, _threshold):
        left = 1.0 if input_size == (64, 64) else 20.0
        return np.asarray(
            [[left, 2.0, left + 4.0, 6.0, 0.8]],
            dtype=np.float32,
        ), None

    def nms(detections):
        nms_inputs.append(detections.copy())
        return list(range(len(detections)))

    detector._detect_candidates = candidates
    detector.nms = nms

    detector.detect(
        np.zeros((8, 8, 3), dtype=np.uint8),
        input_size=[(64, 64), (128, 128)],
    )

    assert nms_inputs[0][:, 0].tolist() == [1.0, 20.0]


def test_nms_keeps_the_first_equal_score_candidate_deterministically():
    detector = _bare_scrfd()
    detections = np.asarray(
        [
            [1.0, 1.0, 10.0, 10.0, 0.8],
            [1.0, 1.0, 10.0, 10.0, 0.8],
        ],
        dtype=np.float32,
    )

    assert detector.nms(detections) == [0]


def test_concurrent_calls_keep_their_own_thresholds():
    detector = _bare_scrfd()
    barrier = Barrier(2)
    lock = Lock()
    observed = []

    def candidates(_image, _input_size, threshold):
        barrier.wait()
        with lock:
            observed.append(threshold)
        return _candidate(threshold), None

    detector._detect_candidates = candidates
    detector.nms = lambda _detections: [0]
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(detector.detect, image, (64, 64), 0, "default", value)
            for value in (0.2, 0.8)
        ]
        results = [future.result()[0][0, 4] for future in futures]

    assert sorted(observed) == [0.2, 0.8]
    assert np.allclose(results, [0.2, 0.8])
    assert detector.det_thresh == 0.5
