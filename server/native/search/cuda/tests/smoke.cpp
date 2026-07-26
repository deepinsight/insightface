#include "ifs_search.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Vector = std::array<float, IFS_SEARCH_DIMENSION>;

Vector constant() {
    Vector result{};
    result.fill(1.0f / std::sqrt(static_cast<float>(result.size())));
    return result;
}

Vector split() {
    Vector result = constant();
    for (size_t i = result.size() / 2; i < result.size(); ++i) {
        result[i] = -result[i];
    }
    return result;
}

Vector wave(uint64_t seed) {
    Vector result{};
    double squared_norm = 0.0;
    for (size_t i = 0; i < result.size(); ++i) {
        const double value =
            std::sin((static_cast<double>(seed) + 1.0) *
                     (static_cast<double>(i) + 1.0) * 0.0017) +
            0.5 * std::cos((static_cast<double>(seed) + 3.0) *
                           (static_cast<double>(i) + 1.0) * 0.0009);
        result[i] = static_cast<float>(value);
        squared_norm += value * value;
    }
    const float inverse_norm =
        static_cast<float>(1.0 / std::sqrt(squared_norm));
    for (float &value : result) value *= inverse_norm;
    return result;
}

Vector sparse(uint64_t seed) {
    Vector result{};
    const size_t first = static_cast<size_t>(seed % result.size());
    size_t second = static_cast<size_t>((seed * 131 + 17) % result.size());
    if (second == first) second = (second + 1) % result.size();
    result[first] = std::sqrt(0.75f);
    result[second] = seed % 2 == 0 ? 0.5f : -0.5f;
    return result;
}

float tolerance(uint32_t profile) {
    if (profile == IFS_SEARCH_PROFILE_FP32_V1) return 1.0e-5f;
    if (profile == IFS_SEARCH_PROFILE_FP16_V1) return 5.0e-3f;
    if (profile == IFS_SEARCH_PROFILE_BF16_V1) return 2.0e-2f;
    return 3.0e-2f;
}

void normalize(Vector *value) {
    double squared = 0.0;
    for (float item : *value) squared += static_cast<double>(item) * item;
    const float inverse = 1.0f / std::sqrt(static_cast<float>(squared));
    for (float &item : *value) item *= inverse;
}

void test_int8_numeric_contract(
    uint32_t profile, float expected_score, float expected_group_score) {
    ifs_search_create_options_t options{};
    options.struct_size = sizeof(options);
    options.profile = profile;
    options.reserve_rows = 1;
    options.max_rows = 1;
    options.device = 0;
    options.topk_mode = IFS_SEARCH_TOPK_DEVICE;

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
    assert(count == 1 && out_group == 1 && out_id == 20);
    assert(std::abs(score - expected_group_score) < 1.0e-6f);
    ifs_search_destroy(index);
}

void test_int8_scales_are_per_index() {
    ifs_search_create_options_t options{};
    options.struct_size = sizeof(options);
    options.reserve_rows = 1;
    options.max_rows = 1;
    options.device = 0;
    options.topk_mode = IFS_SEARCH_TOPK_DEVICE;
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
    options.reserve_rows = 6;
    options.max_rows = 6;
    options.device = 0;
    options.topk_mode = IFS_SEARCH_TOPK_DEVICE;
    options.growth_factor = 1.5;

    ifs_search_index_t index = nullptr;
    assert(ifs_search_create(&options, &index) == IFS_SEARCH_OK);
    const Vector positive = constant();
    const Vector orthogonal = split();
    std::vector<float> rows;
    rows.insert(rows.end(), positive.begin(), positive.end());
    rows.insert(rows.end(), orthogonal.begin(), orthogonal.end());
    rows.insert(rows.end(), positive.begin(), positive.end());
    rows.insert(rows.end(), positive.begin(), positive.end());
    const std::array<uint64_t, 4> ids{10, 20, 30, 5};
    const std::array<uint64_t, 4> groups{100, 200, 300, 100};
    assert(ifs_search_add_batch(index, ids.data(), groups.data(), rows.data(), 4) ==
           IFS_SEARCH_OK);

    std::array<uint64_t, 6> out_ids{};
    std::array<float, 6> scores{};
    uint64_t count = 0;
    ifs_search_timings_t timings{};
    timings.struct_size = sizeof(timings);
    assert(ifs_search_topk(index, positive.data(), 4, out_ids.data(),
                           scores.data(), &count, &timings) == IFS_SEARCH_OK);
    assert(count == 4);
    assert(out_ids[0] == 5 && out_ids[1] == 10 && out_ids[2] == 30 &&
           out_ids[3] == 20);
    assert(std::abs(scores[0] - 1.0f) < tolerance(profile));
    assert(std::abs(scores[3]) < tolerance(profile));

    std::array<uint64_t, 6> out_groups{};
    assert(ifs_search_grouped_topk(
               index, positive.data(), 6, out_groups.data(), out_ids.data(),
               scores.data(), &count, &timings) == IFS_SEARCH_OK);
    assert(count == 3);
    assert(out_groups[0] == 100 && out_ids[0] == 5);
    assert(out_groups[1] == 300 && out_ids[1] == 30);
    assert(out_groups[2] == 200 && out_ids[2] == 20);

    uint64_t removed = 0;
    const uint64_t deleted = 5;
    assert(ifs_search_delete_batch(index, &deleted, 1, &removed) == IFS_SEARCH_OK);
    assert(removed == 1);
    assert(ifs_search_grouped_topk(
               index, positive.data(), 6, out_groups.data(), out_ids.data(),
               scores.data(), &count, &timings) == IFS_SEARCH_OK);
    assert(count == 3);
    assert(out_groups[0] == 100 && out_ids[0] == 10);

    const uint64_t deleted_last_in_group = 10;
    assert(ifs_search_delete_batch(index, &deleted_last_in_group, 1, &removed) ==
           IFS_SEARCH_OK);
    assert(removed == 1);
    const uint64_t replacement_id = 40;
    const uint64_t replacement_group = 400;
    assert(ifs_search_add_batch(index, &replacement_id, &replacement_group,
                                positive.data(), 1) == IFS_SEARCH_OK);
    const uint64_t reactivated_id = 50;
    const uint64_t reactivated_group = 100;
    assert(ifs_search_add_batch(index, &reactivated_id, &reactivated_group,
                                positive.data(), 1) == IFS_SEARCH_OK);
    assert(ifs_search_grouped_topk(
               index, positive.data(), 6, out_groups.data(), out_ids.data(),
               scores.data(), &count, &timings) == IFS_SEARCH_OK);
    assert(count == 4);
    assert(out_groups[0] == 100 && out_ids[0] == 50);
    assert(out_groups[1] == 300 && out_ids[1] == 30);
    assert(out_groups[2] == 400 && out_ids[2] == 40);
    assert(out_groups[3] == 200 && out_ids[3] == 20);

    const uint64_t overflow_id = 60;
    const uint64_t overflow_group = 600;
    assert(ifs_search_add_batch(index, &overflow_id, &overflow_group,
                                positive.data(), 1) ==
           IFS_SEARCH_CAPACITY_EXCEEDED);

    ifs_search_stats_t stats{};
    stats.struct_size = sizeof(stats);
    assert(ifs_search_get_stats(index, &stats) == IFS_SEARCH_OK);
    assert(stats.physical_rows == 6);
    assert(stats.live_rows == 4);
    assert(stats.tombstone_rows == 2);
    assert(stats.capacity_rows == 6);
    ifs_search_destroy(index);
}

void test_group_metadata_survives_growth() {
    ifs_search_create_options_t options{};
    options.struct_size = sizeof(options);
    options.profile = IFS_SEARCH_PROFILE_FP32_V1;
    options.reserve_rows = 1;
    options.max_rows = 8;
    options.device = 0;
    options.topk_mode = IFS_SEARCH_TOPK_DEVICE;
    options.growth_factor = 1.5;

    ifs_search_index_t index = nullptr;
    assert(ifs_search_create(&options, &index) == IFS_SEARCH_OK);
    const Vector positive = constant();
    const Vector orthogonal = split();
    std::vector<float> rows;
    rows.insert(rows.end(), positive.begin(), positive.end());
    rows.insert(rows.end(), orthogonal.begin(), orthogonal.end());
    rows.insert(rows.end(), positive.begin(), positive.end());
    rows.insert(rows.end(), positive.begin(), positive.end());
    const std::array<uint64_t, 4> ids{8, 7, 6, 5};
    const std::array<uint64_t, 4> groups{900, 800, 900, 700};
    assert(ifs_search_add_batch(index, ids.data(), groups.data(), rows.data(), 4) ==
           IFS_SEARCH_OK);

    std::array<uint64_t, 4> out_groups{};
    std::array<uint64_t, 4> out_ids{};
    std::array<float, 4> scores{};
    uint64_t count = 0;
    assert(ifs_search_grouped_topk(
               index, positive.data(), 4, out_groups.data(), out_ids.data(),
               scores.data(), &count, nullptr) == IFS_SEARCH_OK);
    assert(count == 3);
    assert(out_groups[0] == 700 && out_ids[0] == 5);
    assert(out_groups[1] == 900 && out_ids[1] == 6);
    assert(out_groups[2] == 800 && out_ids[2] == 7);

    ifs_search_stats_t stats{};
    stats.struct_size = sizeof(stats);
    assert(ifs_search_get_stats(index, &stats) == IFS_SEARCH_OK);
    assert(stats.reallocations > 0);
    ifs_search_destroy(index);
}

void test_grouped_matches_exact_face_reference(uint32_t profile) {
    constexpr size_t initial_rows = 513;
    constexpr size_t added_rows = 23;
    ifs_search_create_options_t options{};
    options.struct_size = sizeof(options);
    options.profile = profile;
    options.reserve_rows = 3;
    options.max_rows = 700;
    options.device = 0;
    options.topk_mode = IFS_SEARCH_TOPK_HOST;
    options.growth_factor = 1.25;

    ifs_search_index_t index = nullptr;
    assert(ifs_search_create(&options, &index) == IFS_SEARCH_OK);
    std::vector<uint64_t> ids(initial_rows);
    std::vector<uint64_t> groups(initial_rows);
    std::vector<float> vectors;
    vectors.reserve(initial_rows * IFS_SEARCH_DIMENSION);
    std::unordered_map<uint64_t, uint64_t> group_by_id;
    for (size_t i = 0; i < initial_rows; ++i) {
        ids[i] = 1000 + i;
        groups[i] = 100 + (i * 37) % 137;
        group_by_id.emplace(ids[i], groups[i]);
        const Vector value = wave(i + 11);
        vectors.insert(vectors.end(), value.begin(), value.end());
    }
    assert(ifs_search_add_batch(index, ids.data(), groups.data(),
                                vectors.data(), initial_rows) == IFS_SEARCH_OK);

    std::vector<uint64_t> deleted;
    for (size_t i = 0; i < initial_rows; i += 17) deleted.push_back(ids[i]);
    uint64_t removed = 0;
    assert(ifs_search_delete_batch(index, deleted.data(), deleted.size(),
                                   &removed) == IFS_SEARCH_OK);
    assert(removed == deleted.size());

    std::vector<uint64_t> added_ids(added_rows);
    std::vector<uint64_t> added_groups(added_rows);
    std::vector<float> added_vectors;
    added_vectors.reserve(added_rows * IFS_SEARCH_DIMENSION);
    for (size_t i = 0; i < added_rows; ++i) {
        added_ids[i] = 5000 + i;
        added_groups[i] = i % 2 == 0 ? 100 + (i * 19) % 137 : 1000 + i;
        group_by_id.emplace(added_ids[i], added_groups[i]);
        const Vector value = wave(i + 3000);
        added_vectors.insert(added_vectors.end(), value.begin(), value.end());
    }
    assert(ifs_search_add_batch(index, added_ids.data(), added_groups.data(),
                                added_vectors.data(), added_rows) ==
           IFS_SEARCH_OK);

    ifs_search_stats_t stats{};
    stats.struct_size = sizeof(stats);
    assert(ifs_search_get_stats(index, &stats) == IFS_SEARCH_OK);
    assert(stats.reallocations > 0);
    const size_t live_rows = static_cast<size_t>(stats.live_rows);
    const Vector query = wave(9999);
    std::vector<uint64_t> face_ids(live_rows);
    std::vector<float> face_scores(live_rows);
    uint64_t face_count = 0;
    assert(ifs_search_topk(index, query.data(), live_rows, face_ids.data(),
                           face_scores.data(), &face_count, nullptr) ==
           IFS_SEARCH_OK);
    assert(face_count == live_rows);

    struct Expected {
        uint64_t group_id;
        uint64_t vector_id;
        float score;
    };
    std::unordered_map<uint64_t, Expected> best_by_group;
    for (size_t i = 0; i < live_rows; ++i) {
        const uint64_t group_id = group_by_id.at(face_ids[i]);
        const Expected candidate{group_id, face_ids[i], face_scores[i]};
        const auto found = best_by_group.find(group_id);
        if (found == best_by_group.end() ||
            candidate.score > found->second.score ||
            (candidate.score == found->second.score &&
             candidate.vector_id < found->second.vector_id)) {
            best_by_group[group_id] = candidate;
        }
    }
    std::vector<Expected> expected;
    expected.reserve(best_by_group.size());
    for (const auto &entry : best_by_group) expected.push_back(entry.second);
    std::sort(expected.begin(), expected.end(),
              [](const Expected &left, const Expected &right) {
                  if (left.score != right.score) {
                      return left.score > right.score;
                  }
                  if (left.group_id != right.group_id) {
                      return left.group_id < right.group_id;
                  }
                  return left.vector_id < right.vector_id;
              });

    constexpr size_t grouped_k = 100;
    std::array<uint64_t, grouped_k> out_groups{};
    std::array<uint64_t, grouped_k> out_ids{};
    std::array<float, grouped_k> out_scores{};
    uint64_t grouped_count = 0;
    assert(ifs_search_grouped_topk(
               index, query.data(), grouped_k, out_groups.data(),
               out_ids.data(), out_scores.data(), &grouped_count, nullptr) ==
           IFS_SEARCH_OK);
    assert(grouped_count == std::min(grouped_k, expected.size()));
    for (size_t i = 0; i < static_cast<size_t>(grouped_count); ++i) {
        assert(out_groups[i] == expected[i].group_id);
        assert(out_ids[i] == expected[i].vector_id);
        assert(std::abs(out_scores[i] - expected[i].score) < 1.0e-6f);
    }
    ifs_search_destroy(index);
}

void test_grouped_multistage_and_k_boundaries() {
    constexpr size_t row_count = 9001;
    ifs_search_create_options_t options{};
    options.struct_size = sizeof(options);
    options.profile = IFS_SEARCH_PROFILE_FP32_V1;
    options.reserve_rows = row_count;
    options.max_rows = row_count;
    options.device = 0;
    options.topk_mode = IFS_SEARCH_TOPK_HOST;
    options.growth_factor = 1.5;

    ifs_search_index_t index = nullptr;
    assert(ifs_search_create(&options, &index) == IFS_SEARCH_OK);
    const Vector query = constant();
    const uint64_t reserved_id = UINT64_MAX;
    const uint64_t reserved_group = 1;
    assert(ifs_search_add_batch(index, &reserved_id, &reserved_group,
                                query.data(), 1) ==
           IFS_SEARCH_INVALID_ARGUMENT);
    std::vector<uint64_t> ids(row_count);
    std::vector<uint64_t> groups(row_count);
    std::vector<float> vectors;
    vectors.reserve(row_count * IFS_SEARCH_DIMENSION);
    for (size_t i = 0; i < row_count; ++i) {
        ids[i] = 10000 + i;
        groups[i] = ids[i];
        const Vector value = sparse(i);
        vectors.insert(vectors.end(), value.begin(), value.end());
    }
    assert(ifs_search_add_batch(index, ids.data(), groups.data(),
                                vectors.data(), row_count) == IFS_SEARCH_OK);

    std::array<uint64_t, 101> face_ids{};
    std::array<float, 101> face_scores{};
    uint64_t face_count = 0;
    assert(ifs_search_topk(index, query.data(), 100, face_ids.data(),
                           face_scores.data(), &face_count, nullptr) ==
           IFS_SEARCH_OK);
    assert(face_count == 100);

    std::array<uint64_t, 101> out_groups{};
    std::array<uint64_t, 101> out_ids{};
    std::array<float, 101> out_scores{};
    uint64_t grouped_count = 0;
    assert(ifs_search_grouped_topk(
               index, query.data(), 100, out_groups.data(), out_ids.data(),
               out_scores.data(), &grouped_count, nullptr) == IFS_SEARCH_OK);
    assert(grouped_count == 100);
    for (size_t i = 0; i < 100; ++i) {
        assert(out_groups[i] == face_ids[i]);
        assert(out_ids[i] == face_ids[i]);
        assert(out_scores[i] == face_scores[i]);
    }

    assert(ifs_search_grouped_topk(
               index, query.data(), 1, out_groups.data(), out_ids.data(),
               out_scores.data(), &grouped_count, nullptr) == IFS_SEARCH_OK);
    assert(grouped_count == 1 && out_ids[0] == face_ids[0]);
    assert(ifs_search_grouped_topk(index, query.data(), 0, nullptr, nullptr,
                                   nullptr, &grouped_count, nullptr) ==
           IFS_SEARCH_OK);
    assert(grouped_count == 0);
    assert(ifs_search_grouped_topk(
               index, query.data(), 101, out_groups.data(), out_ids.data(),
               out_scores.data(), &grouped_count, nullptr) ==
           IFS_SEARCH_UNSUPPORTED);

    ifs_search_destroy(index);
}

}  // namespace

int main() {
    static_assert(IFS_SEARCH_PROFILE_INT8_X1000_V1 == 3,
                  "persisted x1000 profile code changed");
    static_assert(IFS_SEARCH_PROFILE_INT8_X736_V1 == 4,
                  "x736 profile must append after existing codes");
    ifs_search_capabilities_t capabilities{};
    capabilities.struct_size = sizeof(capabilities);
    const ifs_search_status_t capability_status =
        ifs_search_get_capabilities(0, &capabilities);
    if (capability_status != IFS_SEARCH_OK) {
        std::fprintf(stderr,
                     "SKIP: CUDA device 0 is unavailable in this build environment: %s\n",
                     ifs_search_last_error());
        return 77;
    }
    assert(capabilities.backend == IFS_SEARCH_BACKEND_CUDA);
    assert((capabilities.flags & IFS_SEARCH_CAP_DEVICE_TOPK) != 0);
    assert((capabilities.flags & IFS_SEARCH_CAP_GROUPED_PERSON_TOPK) != 0);
    assert((capabilities.flags & IFS_SEARCH_CAP_GROUPED_DEVICE_RESIDENT) != 0);
    assert((capabilities.flags & IFS_SEARCH_CAP_GROUPED_HOST_REFERENCE) == 0);
    assert(capabilities.device_topk_limit == 100);
    assert((capabilities.profile_mask &
            (UINT64_C(1) << IFS_SEARCH_PROFILE_INT8_X736_V1)) != 0);
    assert(std::string(ifs_search_build_info()).find(
               "grouped_topk=device-resident-exact") != std::string::npos);

    for (uint32_t profile = IFS_SEARCH_PROFILE_FP32_V1;
         profile <= IFS_SEARCH_PROFILE_INT8_X736_V1; ++profile) {
        if ((capabilities.profile_mask & (UINT64_C(1) << profile)) != 0) {
            test_profile(profile);
            test_grouped_matches_exact_face_reference(profile);
        }
    }
    test_group_metadata_survives_growth();
    test_grouped_multistage_and_k_boundaries();
    test_int8_numeric_contract(
        IFS_SEARCH_PROFILE_INT8_X1000_V1, 0.013689f, 1.0f);
    test_int8_numeric_contract(
        IFS_SEARCH_PROFILE_INT8_X736_V1, 7440.0f / 541696.0f,
        530755.0f / 541696.0f);
    test_int8_scales_are_per_index();
    return 0;
}
