from __future__ import annotations

import numpy as np
import pytest
from insightface_server.search import (
    IndexRecord,
    NativeCapabilities,
    ReferenceSearchIndex,
    SearchIndexCapacityError,
)
from insightface_server.search.reference import (
    profile_similarity,
    quantize_bf16_to_fp32,
    quantize_int8_x736,
    quantize_int8_x1000,
)


def unit(value: list[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (1 << 9, "device_exact"),
        ((1 << 8) | (1 << 9), "device_exact"),
        (1 << 8, "host_reference"),
        (0, "unavailable"),
    ],
)
def test_native_capability_reports_grouped_topk_location(
    flags: int, expected: str
) -> None:
    capabilities = NativeCapabilities(
        backend="native_cuda",
        profiles=("fp32_v1",),
        flags=flags,
        device=0,
        compute_capability="12.0",
        cuda_runtime_version=12090,
        cuda_driver_version=13000,
        build_info="test",
    )

    assert capabilities.grouped_person_topk_mode == expected


@pytest.mark.parametrize(
    "profile",
    ["fp32_v1", "fp16_v1", "bf16_v1", "int8_x1000_v1", "int8_x736_v1"],
)
def test_reference_profiles_return_raw_cosine_and_group_by_person(profile: str) -> None:
    query = unit([1.0] * 512)
    alternate = unit([1.0] * 384 + [-1.0] * 128)
    orthogonal = unit([1.0] * 256 + [-1.0] * 256)
    index = ReferenceSearchIndex(profile=profile, dimension=512, capacity_rows=4)
    index.add_batch(
        [
            IndexRecord(1, 10, query),
            IndexRecord(2, 10, alternate),
            IndexRecord(3, 20, orthogonal),
        ]
    )

    hits = index.search_persons(query, 10)

    assert [hit.person_numeric_id for hit in hits] == [10, 20]
    assert hits[0].vector_id == 1
    assert hits[0].cosine == pytest.approx(1.0, abs=2e-2)
    assert hits[1].cosine == pytest.approx(0.0, abs=2e-2)


@pytest.mark.parametrize(
    "profile",
    ["fp32_v1", "fp16_v1", "bf16_v1", "int8_x1000_v1", "int8_x736_v1"],
)
def test_reference_profiles_keep_negative_raw_cosine(profile: str) -> None:
    positive = unit([1.0] * 512)
    negative = -positive
    index = ReferenceSearchIndex(profile=profile, dimension=512, capacity_rows=1)
    index.add_batch([IndexRecord(1, 1, negative)])

    hit = index.search_persons(positive, 1)[0]

    assert hit.cosine == pytest.approx(-1.0, abs=3e-2)


def test_int8_x1000_quantization_rounds_and_saturates_to_int8() -> None:
    values = np.asarray(
        [
            -1.0,
            -0.1276,
            -0.1274,
            -0.12449999898672104,
            -0.0005,
            0.0005,
            0.12449999898672104,
            0.1266,
            0.1276,
            1.0,
        ],
        dtype=np.float32,
    )

    assert quantize_int8_x1000(values).tolist() == [
        -128,
        -128,
        -127,
        -125,
        -1,
        1,
        125,
        127,
        127,
        127,
    ]


def test_int8_x736_quantization_rounds_and_saturates_to_int8() -> None:
    scale = np.float32(736.0)
    values = np.asarray(
        [
            -1.0,
            -128.0 / scale,
            -127.6 / scale,
            -127.4 / scale,
            -0.5 / scale,
            0.5 / scale,
            124.49999898672104 / scale,
            126.6 / scale,
            127.6 / scale,
            1.0,
        ],
        dtype=np.float32,
    )

    assert quantize_int8_x736(values).tolist() == [
        -128,
        -128,
        -128,
        -127,
        -1,
        1,
        125,
        127,
        127,
        127,
    ]


@pytest.mark.parametrize(
    "profile",
    ["fp32_v1", "fp16_v1", "bf16_v1", "int8_x1000_v1", "int8_x736_v1"],
)
def test_profile_similarity_matches_profile_encoding(profile: str) -> None:
    left = unit([1.0] * 384 + [-1.0] * 128)
    right = unit([1.0] * 256 + [-1.0] * 256)

    if profile == "fp16_v1":
        expected = float(
            np.dot(
                left.astype(np.float16).astype(np.float32),
                right.astype(np.float16).astype(np.float32),
            )
        )
    elif profile == "bf16_v1":
        expected = float(
            np.dot(quantize_bf16_to_fp32(left), quantize_bf16_to_fp32(right))
        )
    elif profile.startswith("int8_"):
        scale = 1000 if "x1000" in profile else 736
        quantize = quantize_int8_x1000 if scale == 1000 else quantize_int8_x736
        expected = float(
            np.dot(
                quantize(left).astype(np.int32),
                quantize(right).astype(np.int32),
            )
        ) / float(scale * scale)
    else:
        expected = float(np.dot(left, right))

    assert profile_similarity(profile, left, right) == pytest.approx(expected)


def test_profile_similarity_returns_unclipped_int8_raw_score() -> None:
    value = unit([1.0] * 512)

    score = profile_similarity("int8_x736_v1", value, value)

    assert score == pytest.approx(557_568 / 541_696)
    assert score > 1.0


def test_profile_similarity_validates_profile_shape_and_normalization() -> None:
    with pytest.raises(ValueError, match="unsupported search profile"):
        profile_similarity("int8_v0", unit([1.0, 0.0]), unit([1.0, 0.0]))
    with pytest.raises(ValueError, match="same non-empty shape"):
        profile_similarity("fp32_v1", unit([1.0, 0.0]), unit([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="L2-normalized"):
        profile_similarity(
            "fp32_v1",
            np.asarray([2.0, 0.0], dtype=np.float32),
            unit([1.0, 0.0]),
        )


def test_int8_group_selection_uses_unclipped_accumulator_before_public_score() -> None:
    first = np.concatenate(
        [
            np.full(99, 0.045, dtype=np.float32),
            np.full(
                413,
                np.sqrt((1.0 - 99 * 0.045**2) / 413),
                dtype=np.float32,
            ),
        ]
    )
    second = np.concatenate(
        [
            np.full(100, 0.045, dtype=np.float32),
            np.full(
                412,
                np.sqrt((1.0 - 100 * 0.045**2) / 412),
                dtype=np.float32,
            ),
        ]
    )
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    index = ReferenceSearchIndex(
        profile="int8_x1000_v1", dimension=512, capacity_rows=2
    )
    index.add_batch([IndexRecord(10, 1, first), IndexRecord(20, 1, second)])

    hit = index.search_persons(first, 1)[0]

    # Quantized second·first (1.000087) beats first·first (1.000043). Selection
    # therefore happens before the public raw-cosine range is clamped.
    assert hit.vector_id == 20
    assert hit.cosine == 1.0


def test_int8_x736_uses_541696_divisor_and_unclipped_accumulator_order() -> None:
    first = np.concatenate(
        [
            np.full(99, 0.045, dtype=np.float32),
            np.full(
                413,
                np.sqrt((1.0 - 99 * 0.045**2) / 413),
                dtype=np.float32,
            ),
        ]
    )
    second = np.concatenate(
        [
            np.full(100, 0.045, dtype=np.float32),
            np.full(
                412,
                np.sqrt((1.0 - 100 * 0.045**2) / 412),
                dtype=np.float32,
            ),
        ]
    )
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    index = ReferenceSearchIndex(
        profile="int8_x736_v1", dimension=512, capacity_rows=2
    )
    index.add_batch([IndexRecord(10, 1, first), IndexRecord(20, 1, second)])

    hit = index.search_persons(first, 1)[0]

    assert hit.vector_id == 20
    assert hit.cosine == pytest.approx(530_755 / 541_696)


def test_reference_delete_reuses_reserved_slot_and_capacity_is_hard() -> None:
    index = ReferenceSearchIndex(profile="fp32_v1", dimension=2, capacity_rows=2)
    index.add_batch(
        [IndexRecord(1, 1, unit([1, 0])), IndexRecord(2, 2, unit([0, 1]))]
    )
    with pytest.raises(SearchIndexCapacityError):
        index.add_batch([IndexRecord(3, 3, unit([-1, 0]))])

    assert index.remove_batch([1, 99]) == {1}
    index.add_batch([IndexRecord(3, 3, unit([-1, 0]))])

    stats = index.stats()
    assert stats.live_rows == 2
    assert stats.physical_rows == 2
    assert stats.reallocations == 0
    assert [hit.vector_id for hit in index.search_persons(unit([-1, 0]), 2)] == [3, 2]


def test_reference_rejects_non_normalized_vectors() -> None:
    index = ReferenceSearchIndex(profile="fp32_v1", dimension=2, capacity_rows=1)
    with pytest.raises(ValueError, match="L2-normalized"):
        index.add_batch([IndexRecord(1, 1, np.array([2.0, 0.0], dtype=np.float32))])
