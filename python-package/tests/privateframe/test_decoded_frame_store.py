from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from insightface.app.privateframe.packet_cache import DecodedFrameStore


def _frame(frame_index: int, *, height: int = 4, width: int = 5) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = frame_index
    image[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
    image[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
    return image


def test_remember_is_bounded_by_bytes_and_frame_target_with_lru_refresh() -> None:
    frame_bytes = int(_frame(0, height=2, width=2).nbytes)
    store = DecodedFrameStore(frame_target=2, byte_capacity=2 * frame_bytes)
    store.remember(0, _frame(0, height=2, width=2))
    store.remember(1, _frame(1, height=2, width=2))

    def forbidden(_first: int, _last: int) -> Mapping[int, np.ndarray]:
        raise AssertionError("cached frame was decoded again")

    assert set(store.read_range(0, 0, loader=forbidden)) == {0}
    store.remember(2, _frame(2, height=2, width=2))

    calls: list[tuple[int, int]] = []

    def load(first: int, last: int) -> Mapping[int, np.ndarray]:
        calls.append((first, last))
        return {
            index: _frame(index, height=2, width=2)
            for index in range(first, last + 1)
        }

    store.read_range(1, 1, loader=load)

    assert calls == [(1, 1)]
    assert store.hits == 1
    assert store.frame_count == 2
    assert store.live_bytes == 2 * frame_bytes
    assert store.peak_bytes == 2 * frame_bytes


def test_overlapping_crop_requests_reuse_complete_decoded_frames() -> None:
    store = DecodedFrameStore(
        frame_target=3,
        byte_capacity=3 * int(_frame(0).nbytes),
    )
    calls: list[tuple[int, int]] = []

    def load(first: int, last: int) -> Mapping[int, np.ndarray]:
        calls.append((first, last))
        return {index: _frame(index) for index in range(first, last + 1)}

    left = store.read_range(0, 2, crop=(0, 0, 2, 2), loader=load)
    calls_after_first_read = list(calls)
    right = store.read_range(0, 2, crop=(3, 2, 5, 4), loader=load)

    assert calls == calls_after_first_read
    assert store.hits == 3
    assert store.frame_count == 3
    for index in range(3):
        np.testing.assert_array_equal(left[index], _frame(index)[0:2, 0:2])
        np.testing.assert_array_equal(right[index], _frame(index)[2:4, 3:5])


def test_missing_ranges_are_loaded_in_byte_bounded_chunks() -> None:
    frame = _frame(0, height=4, width=4)
    frame_bytes = int(frame.nbytes)
    store = DecodedFrameStore(frame_target=10, byte_capacity=2 * frame_bytes)
    # A live frame provides the fixed-resolution byte estimate before history
    # is requested, as it does in the streaming pipeline.
    store.remember(99, frame)
    calls: list[tuple[int, int]] = []
    materialized_bytes: list[int] = []

    def load(first: int, last: int) -> Mapping[int, np.ndarray]:
        calls.append((first, last))
        result = {
            index: _frame(index, height=4, width=4)
            for index in range(first, last + 1)
        }
        materialized_bytes.append(sum(int(value.nbytes) for value in result.values()))
        return result

    result = store.read_range(0, 4, crop=(0, 0, 1, 1), loader=load)

    assert calls == [(0, 1), (2, 3), (4, 4)]
    assert max(materialized_bytes) <= store.byte_capacity
    assert list(result) == [0, 1, 2, 3, 4]
    assert store.frame_count <= 2
    assert store.live_bytes <= store.byte_capacity
    assert store.peak_bytes <= store.byte_capacity


def test_decode_block_reserves_lru_bytes_before_loader_materializes_frames() -> None:
    frame = _frame(0, height=4, width=4)
    frame_bytes = int(frame.nbytes)
    store = DecodedFrameStore(frame_target=2, byte_capacity=2 * frame_bytes)
    store.remember(90, _frame(90, height=4, width=4))
    store.remember(91, _frame(91, height=4, width=4))
    combined_bytes_during_load: list[int] = []

    def load(first: int, last: int) -> Mapping[int, np.ndarray]:
        loaded = {
            index: _frame(index, height=4, width=4)
            for index in range(first, last + 1)
        }
        combined_bytes_during_load.append(
            store.live_bytes
            + sum(int(image.nbytes) for image in loaded.values())
        )
        return loaded

    store.read_range(0, 3, crop=(0, 0, 1, 1), loader=load)

    assert combined_bytes_during_load == [store.byte_capacity, store.byte_capacity]
    assert max(combined_bytes_during_load) <= store.byte_capacity


def test_per_frame_crops_are_applied_after_full_frames_are_cached() -> None:
    store = DecodedFrameStore(
        frame_target=2,
        byte_capacity=2 * int(_frame(0).nbytes),
    )
    calls = 0

    def load(first: int, last: int) -> Mapping[int, np.ndarray]:
        nonlocal calls
        calls += 1
        return {index: _frame(index) for index in range(first, last + 1)}

    first = store.read_range(
        0,
        1,
        crops={0: (0, 0, 2, 2), 1: (1, 1, 4, 3)},
        loader=load,
    )
    first_call_count = calls
    second = store.read_range(
        0,
        1,
        crops={0: (2, 2, 5, 4), 1: (-1, -1, 1, 1)},
        loader=load,
    )

    assert calls == first_call_count
    np.testing.assert_array_equal(first[0], _frame(0)[0:2, 0:2])
    np.testing.assert_array_equal(first[1], _frame(1)[1:3, 1:4])
    np.testing.assert_array_equal(second[0], _frame(0)[2:4, 2:5])
    assert second[1].shape == (2, 2, 3)
    np.testing.assert_array_equal(second[1][1, 1], _frame(1)[0, 0])


def test_oversized_frame_is_returned_but_not_retained() -> None:
    image = _frame(7)
    store = DecodedFrameStore(
        frame_target=10,
        byte_capacity=int(image.nbytes) - 1,
    )
    calls = 0

    def load(first: int, last: int) -> Mapping[int, np.ndarray]:
        nonlocal calls
        calls += 1
        return {index: image for index in range(first, last + 1)}

    assert store.read_range(7, 7, loader=load)[7] is image
    assert store.read_range(7, 7, loader=load)[7] is image
    assert calls == 2
    assert store.frame_count == 0
    assert store.live_bytes == 0
    assert store.peak_bytes == 0


def test_loader_must_return_every_requested_frame() -> None:
    store = DecodedFrameStore(frame_target=2, byte_capacity=1024)

    with pytest.raises(RuntimeError, match=r"omitted frames: \[1\]"):
        store.read_range(0, 1, loader=lambda _first, _last: {0: _frame(0)})


@pytest.mark.parametrize(
    ("frame_target", "byte_capacity", "message"),
    [(-1, 10, "frame_target"), (1, -10, "byte_capacity")],
)
def test_capacity_cannot_be_negative(
    frame_target: int,
    byte_capacity: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DecodedFrameStore(frame_target, byte_capacity)
