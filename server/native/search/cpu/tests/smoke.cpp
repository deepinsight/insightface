#include "ifs_search.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using Vector = std::array<float, IFS_SEARCH_DIMENSION>;

Vector constant(float sign = 1.0f) {
    Vector result{};
    const float value = sign / std::sqrt(static_cast<float>(result.size()));
    result.fill(value);
    return result;
}

Vector split() {
    Vector result = constant();
    for (size_t i = result.size() / 2; i < result.size(); ++i) {
        result[i] = -result[i];
    }
    return result;
}

float tolerance(uint32_t profile) {
    if (profile == IFS_SEARCH_PROFILE_FP32_V1) return 1.0e-5f;
    if (profile == IFS_SEARCH_PROFILE_BF16_V1) return 2.0e-2f;
    return 3.0e-2f;
}

void normalize(Vector *value) {
    double squared = 0.0;
    for (float item : *value) squared += static_cast<double>(item) * item;
    const float inverse = 1.0f / std::sqrt(static_cast<float>(squared));
    for (float &item : *value) item *= inverse;
}

void test_int8_quantization_float32_contract(
    uint32_t profile, float expected_score, float expected_group_score) {
    ifs_search_create_options_t options{};
    options.struct_size = sizeof(options);
    options.profile = profile;
    options.reserve_rows = 1;
    options.max_rows = 1;
    options.device = -1;

    ifs_search_index_t index = nullptr;
    assert(ifs_search_create(&options, &index) == IFS_SEARCH_OK);
    Vector stored{};
    Vector query{};
    stored[0] = query[0] = 0.12449999898672104f;
    const float tail = std::sqrt(
        (1.0f - stored[0] * stored[0]) /
        static_cast<float>(stored.size() - 1));
    for (size_t i = 1; i < stored.size(); ++i) {
        stored[i] = tail;
        query[i] = i <= 255 ? tail : -tail;
    }
    normalize(&stored);
    normalize(&query);
    const uint64_t id = 1;
    const uint64_t group = 1;
    assert(ifs_search_add_batch(index, &id, &group, stored.data(), 1) ==
           IFS_SEARCH_OK);
    uint64_t out_id = 0;
    float score = 0.0f;
    uint64_t count = 0;
    assert(ifs_search_topk(index, query.data(), 1, &out_id, &score, &count,
                           nullptr) == IFS_SEARCH_OK);
    assert(count == 1 && out_id == id);
    // Multiplication remains FP32 and rounding is half away from zero on both
    // supported scales, matching CUDA and the NumPy reference backend.
    assert(std::abs(score - expected_score) < 1.0e-6f);
    ifs_search_destroy(index);

    options.reserve_rows = 2;
    options.max_rows = 2;
    assert(ifs_search_create(&options, &index) == IFS_SEARCH_OK);
    Vector first{};
    Vector second{};
    for (size_t i = 0; i < first.size(); ++i) {
        first[i] = i < 99
            ? 0.045f
            : std::sqrt((1.0f - 99.0f * 0.045f * 0.045f) / 413.0f);
        second[i] = i < 100
            ? 0.045f
            : std::sqrt((1.0f - 100.0f * 0.045f * 0.045f) / 412.0f);
    }
    normalize(&first);
    normalize(&second);
    std::vector<float> rows;
    rows.insert(rows.end(), first.begin(), first.end());
    rows.insert(rows.end(), second.begin(), second.end());
    const std::array<uint64_t, 2> ids{10, 20};
    const std::array<uint64_t, 2> groups{1, 1};
    assert(ifs_search_add_batch(index, ids.data(), groups.data(), rows.data(), 2) ==
           IFS_SEARCH_OK);
    uint64_t out_group = 0;
    assert(ifs_search_grouped_topk(index, first.data(), 1, &out_group, &out_id,
                                   &score, &count, nullptr) == IFS_SEARCH_OK);
    // Group selection uses the unbounded accumulator before the public ABI
    // clamps the returned raw-cosine approximation.
    assert(count == 1 && out_group == 1 && out_id == 20);
    assert(std::abs(score - expected_group_score) < 1.0e-6f);
    ifs_search_destroy(index);
}

void test_int8_scales_are_per_index() {
    ifs_search_create_options_t options{};
    options.struct_size = sizeof(options);
    options.reserve_rows = 1;
    options.max_rows = 1;
    options.device = -1;
    ifs_search_index_t x1000 = nullptr;
    ifs_search_index_t x736 = nullptr;
    options.profile = IFS_SEARCH_PROFILE_INT8_X1000_V1;
    assert(ifs_search_create(&options, &x1000) == IFS_SEARCH_OK);
    options.profile = IFS_SEARCH_PROFILE_INT8_X736_V1;
    assert(ifs_search_create(&options, &x736) == IFS_SEARCH_OK);

    Vector stored{};
    Vector query{};
    stored[0] = query[0] = 0.12449999898672104f;
    const float tail = std::sqrt(
        (1.0f - stored[0] * stored[0]) /
        static_cast<float>(stored.size() - 1));
    for (size_t i = 1; i < stored.size(); ++i) {
        stored[i] = tail;
        query[i] = i <= 255 ? tail : -tail;
    }
    normalize(&stored);
    normalize(&query);
    const uint64_t id = 1;
    const uint64_t group = 1;
    assert(ifs_search_add_batch(x1000, &id, &group, stored.data(), 1) ==
           IFS_SEARCH_OK);
    assert(ifs_search_add_batch(x736, &id, &group, stored.data(), 1) ==
           IFS_SEARCH_OK);

    uint64_t out_id = 0;
    uint64_t count = 0;
    float score = 0.0f;
    assert(ifs_search_topk(x1000, query.data(), 1, &out_id, &score, &count,
                           nullptr) == IFS_SEARCH_OK);
    assert(count == 1 && out_id == id);
    assert(std::abs(score - 0.013689f) < 1.0e-6f);
    assert(ifs_search_topk(x736, query.data(), 1, &out_id, &score, &count,
                           nullptr) == IFS_SEARCH_OK);
    assert(count == 1 && out_id == id);
    assert(std::abs(score - 7440.0f / 541696.0f) < 1.0e-6f);

    ifs_search_destroy(x736);
    ifs_search_destroy(x1000);
}

void test_profile(uint32_t profile) {
    ifs_search_create_options_t options{};
    options.struct_size = sizeof(options);
    options.profile = profile;
    options.reserve_rows = 3;
    options.max_rows = 3;
    options.device = -1;
    options.topk_mode = IFS_SEARCH_TOPK_AUTO;

    ifs_search_index_t index = nullptr;
    assert(ifs_search_create(&options, &index) == IFS_SEARCH_OK);
    assert(index != nullptr);

    const Vector positive = constant();
    const Vector orthogonal = split();
    const Vector duplicate = positive;
    std::vector<float> rows;
    rows.insert(rows.end(), positive.begin(), positive.end());
    rows.insert(rows.end(), orthogonal.begin(), orthogonal.end());
    rows.insert(rows.end(), duplicate.begin(), duplicate.end());
    const std::array<uint64_t, 3> vector_ids{10, 20, 30};
    const std::array<uint64_t, 3> group_ids{100, 200, 300};
    assert(ifs_search_add_batch(index, vector_ids.data(), group_ids.data(),
                                rows.data(), 3) == IFS_SEARCH_OK);

    std::array<uint64_t, 3> ids{};
    std::array<float, 3> scores{};
    uint64_t count = 0;
    ifs_search_timings_t timings{};
    timings.struct_size = sizeof(timings);
    assert(ifs_search_topk(index, positive.data(), 3, ids.data(), scores.data(),
                           &count, &timings) == IFS_SEARCH_OK);
    assert(count == 3);
    assert(ids[0] == 10 && ids[1] == 30 && ids[2] == 20);
    assert(std::abs(scores[0] - 1.0f) < tolerance(profile));
    assert(std::abs(scores[2]) < tolerance(profile));

    std::array<uint64_t, 3> out_groups{};
    std::array<uint64_t, 3> out_faces{};
    assert(ifs_search_grouped_topk(
               index, positive.data(), 3, out_groups.data(), out_faces.data(),
               scores.data(), &count, &timings) == IFS_SEARCH_OK);
    assert(count == 3);
    assert(out_groups[0] == 100 && out_faces[0] == 10);
    assert(out_groups[1] == 300 && out_faces[1] == 30);
    assert(out_groups[2] == 200 && out_faces[2] == 20);

    uint64_t removed = 0;
    const uint64_t deleted = 10;
    assert(ifs_search_delete_batch(index, &deleted, 1, &removed) == IFS_SEARCH_OK);
    assert(removed == 1);
    const uint64_t replacement_id = 40;
    const uint64_t replacement_group = 400;
    assert(ifs_search_add_batch(index, &replacement_id, &replacement_group,
                                positive.data(), 1) == IFS_SEARCH_OK);

    const uint64_t overflow_id = 50;
    const uint64_t overflow_group = 500;
    assert(ifs_search_add_batch(index, &overflow_id, &overflow_group,
                                positive.data(), 1) ==
           IFS_SEARCH_CAPACITY_EXCEEDED);

    ifs_search_stats_t stats{};
    stats.struct_size = sizeof(stats);
    assert(ifs_search_get_stats(index, &stats) == IFS_SEARCH_OK);
    assert(stats.live_rows == 3);
    assert(stats.physical_rows == 3);
    assert(stats.tombstone_rows == 0);
    assert(stats.capacity_rows == 3);
    assert(stats.max_rows == 3);
    ifs_search_destroy(index);
}

}  // namespace

int main() {
    static_assert(IFS_SEARCH_ABI_VERSION == 2u, "unexpected search ABI");
    static_assert(IFS_SEARCH_DIMENSION == 512u, "fixed dimension changed");
    static_assert(IFS_SEARCH_PROFILE_INT8_X1000_V1 == 3,
                  "persisted x1000 profile code changed");
    static_assert(IFS_SEARCH_PROFILE_INT8_X736_V1 == 4,
                  "x736 profile must append after existing codes");
    assert(ifs_search_abi_version() == 2u);
    assert(ifs_search_dimension() == 512u);

    ifs_search_capabilities_t capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    assert(ifs_search_get_capabilities(-1, &capabilities) == IFS_SEARCH_OK);
    assert(capabilities.backend == IFS_SEARCH_BACKEND_CPU);
    assert(capabilities.compute_capability_major == -1);
    assert(capabilities.compute_capability_minor == -1);
    assert((capabilities.flags & IFS_SEARCH_CAP_GROUPED_PERSON_TOPK) != 0);
    assert((capabilities.flags & IFS_SEARCH_CAP_GROUPED_HOST_REFERENCE) != 0);
    assert((capabilities.profile_mask &
            (UINT64_C(1) << IFS_SEARCH_PROFILE_FP16_V1)) == 0);
    assert((capabilities.profile_mask &
            (UINT64_C(1) << IFS_SEARCH_PROFILE_INT8_X736_V1)) != 0);
    assert(std::string(ifs_search_build_info()).find(
               "grouped_topk=host-reference") != std::string::npos);

    test_profile(IFS_SEARCH_PROFILE_FP32_V1);
    test_profile(IFS_SEARCH_PROFILE_BF16_V1);
    test_profile(IFS_SEARCH_PROFILE_INT8_X1000_V1);
    test_profile(IFS_SEARCH_PROFILE_INT8_X736_V1);
    test_int8_quantization_float32_contract(
        IFS_SEARCH_PROFILE_INT8_X1000_V1, 0.013689f, 1.0f);
    test_int8_quantization_float32_contract(
        IFS_SEARCH_PROFILE_INT8_X736_V1, 7440.0f / 541696.0f,
        530755.0f / 541696.0f);
    test_int8_scales_are_per_index();

    ifs_search_create_options_t fp16{};
    fp16.struct_size = sizeof(fp16);
    fp16.profile = IFS_SEARCH_PROFILE_FP16_V1;
    fp16.device = -1;
    ifs_search_index_t index = nullptr;
    assert(ifs_search_create(&fp16, &index) == IFS_SEARCH_UNSUPPORTED);
    assert(index == nullptr);

    ifs_search_create_options_t options{};
    options.struct_size = sizeof(options);
    options.profile = IFS_SEARCH_PROFILE_FP32_V1;
    options.reserve_rows = 1;
    options.max_rows = 1;
    options.device = -1;
    assert(ifs_search_create(&options, &index) == IFS_SEARCH_OK);
    Vector invalid{};
    invalid[0] = 0.5f;
    const uint64_t id = 1;
    const uint64_t group = 1;
    assert(ifs_search_add_batch(index, &id, &group, invalid.data(), 1) ==
           IFS_SEARCH_INVALID_ARGUMENT);
    ifs_search_destroy(index);
    return 0;
}
