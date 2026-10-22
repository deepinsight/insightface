#include "ifs_search.h"

#include "ifs_cpu_legacy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr double kDefaultGrowthFactor = 2.0;
constexpr double kNormalizedSquaredTolerance = 1.0e-3;

thread_local std::string g_last_error;

struct CpuIndex {
    ifs_cpu_index_t core = nullptr;
    ifs_search_profile_t profile = IFS_SEARCH_PROFILE_FP32_V1;
    uint64_t max_rows = 0;
    uint64_t physical_rows = 0;
    uint64_t live_rows = 0;
    uint64_t tombstone_rows = 0;
    uint64_t reallocations = 0;
    double growth_factor = kDefaultGrowthFactor;
    std::unordered_map<uint64_t, uint64_t> vector_to_group;
    mutable std::shared_mutex state_mutex;

    ~CpuIndex() {
        ifs_cpu_index_destroy(core);
    }
};

ifs_search_status_t fail(ifs_search_status_t status, const std::string &message) {
    g_last_error = message;
    return status;
}

void clear_error() {
    g_last_error.clear();
}

float public_cosine(float score) {
    return std::clamp(score, -1.0f, 1.0f);
}

CpuIndex *as_index(ifs_search_index_t index) {
    return static_cast<CpuIndex *>(index);
}

bool valid_profile(uint32_t profile) {
    return profile == IFS_SEARCH_PROFILE_FP32_V1 ||
           profile == IFS_SEARCH_PROFILE_BF16_V1 ||
           profile == IFS_SEARCH_PROFILE_INT8_X1000_V1 ||
           profile == IFS_SEARCH_PROFILE_INT8_X736_V1;
}

uint32_t int8_scale_for(uint32_t profile) {
    switch (profile) {
        case IFS_SEARCH_PROFILE_INT8_X1000_V1:
            return IFS_SEARCH_INT8_X1000_SCALE;
        case IFS_SEARCH_PROFILE_INT8_X736_V1:
            return IFS_SEARCH_INT8_X736_SCALE;
        default:
            return 0;
    }
}

float int8_score_divisor(ifs_search_profile_t profile) {
    const float scale = static_cast<float>(int8_scale_for(profile));
    return scale * scale;
}

ifs_cpu_storage_t storage_for(uint32_t profile) {
    switch (profile) {
        case IFS_SEARCH_PROFILE_FP32_V1: return IFS_CPU_STORAGE_FP32;
        case IFS_SEARCH_PROFILE_BF16_V1: return IFS_CPU_STORAGE_BF16;
        case IFS_SEARCH_PROFILE_INT8_X1000_V1: return IFS_CPU_STORAGE_INT8;
        case IFS_SEARCH_PROFILE_INT8_X736_V1: return IFS_CPU_STORAGE_INT8;
        default: throw std::invalid_argument("profile is not supported by the CPU backend");
    }
}

uint64_t bytes_per_vector(ifs_search_profile_t profile) {
    switch (profile) {
        case IFS_SEARCH_PROFILE_FP32_V1:
            return IFS_SEARCH_DIMENSION * sizeof(float);
        case IFS_SEARCH_PROFILE_BF16_V1:
            return IFS_SEARCH_DIMENSION * sizeof(uint16_t);
        case IFS_SEARCH_PROFILE_INT8_X1000_V1:
        case IFS_SEARCH_PROFILE_INT8_X736_V1:
            return IFS_SEARCH_DIMENSION * sizeof(int8_t);
        default:
            return 0;
    }
}

bool validate_normalized_rows(const float *vectors, uint64_t count,
                              std::string *message) {
    if (count && vectors == nullptr) {
        *message = "vectors must not be null for a non-empty batch";
        return false;
    }
    if (count > std::numeric_limits<size_t>::max() / IFS_SEARCH_DIMENSION) {
        *message = "vector batch is too large for this process";
        return false;
    }
    for (uint64_t row = 0; row < count; ++row) {
        double squared_norm = 0.0;
        const float *vector = vectors + static_cast<size_t>(row) * IFS_SEARCH_DIMENSION;
        for (size_t column = 0; column < IFS_SEARCH_DIMENSION; ++column) {
            const float value = vector[column];
            if (!std::isfinite(value)) {
                *message = "vector contains a non-finite value";
                return false;
            }
            squared_norm += static_cast<double>(value) * value;
        }
        if (std::abs(squared_norm - 1.0) > kNormalizedSquaredTolerance) {
            *message = "vector must be L2-normalized before indexing";
            return false;
        }
    }
    return true;
}

ifs_search_status_t map_legacy_status(ifs_cpu_status_t status) {
    switch (status) {
        case IFS_CPU_OK: return IFS_SEARCH_OK;
        case IFS_CPU_INVALID_ARGUMENT: return IFS_SEARCH_INVALID_ARGUMENT;
        case IFS_CPU_OUT_OF_MEMORY: return IFS_SEARCH_OUT_OF_MEMORY;
        case IFS_CPU_DUPLICATE_ID: return IFS_SEARCH_DUPLICATE_ID;
        case IFS_CPU_ID_NOT_FOUND: return IFS_SEARCH_ID_NOT_FOUND;
        case IFS_CPU_INTERNAL_ERROR: return IFS_SEARCH_INTERNAL_ERROR;
    }
    return IFS_SEARCH_INTERNAL_ERROR;
}

ifs_search_status_t legacy_failure(ifs_cpu_status_t status,
                                   const char *fallback) {
    const char *legacy = ifs_cpu_last_error();
    return fail(map_legacy_status(status),
                legacy && legacy[0] ? legacy : fallback);
}

ifs_search_status_t reserve_locked(CpuIndex *index, uint64_t rows,
                                   bool count_reallocation) {
    if (index->max_rows && rows > index->max_rows) {
        return fail(IFS_SEARCH_CAPACITY_EXCEEDED,
                    "requested capacity exceeds the configured max_rows");
    }
    if (rows > std::numeric_limits<size_t>::max()) {
        return fail(IFS_SEARCH_CAPACITY_EXCEEDED,
                    "requested capacity exceeds the process address space");
    }
    const size_t before = ifs_cpu_index_capacity(index->core);
    if (rows <= before) return IFS_SEARCH_OK;
    const ifs_cpu_status_t status = ifs_cpu_index_reserve(
        index->core, static_cast<size_t>(rows));
    if (status != IFS_CPU_OK) return legacy_failure(status, "CPU reserve failed");
    if (count_reallocation) ++index->reallocations;
    return IFS_SEARCH_OK;
}

ifs_search_status_t ensure_add_capacity(CpuIndex *index, uint64_t count) {
    const uint64_t reusable = std::min(count, index->tombstone_rows);
    const uint64_t append_rows = count - reusable;
    if (append_rows > std::numeric_limits<uint64_t>::max() - index->physical_rows) {
        return fail(IFS_SEARCH_CAPACITY_EXCEEDED, "index row count overflow");
    }
    const uint64_t required = index->physical_rows + append_rows;
    if (index->max_rows && required > index->max_rows) {
        return fail(IFS_SEARCH_CAPACITY_EXCEEDED,
                    "add would exceed the configured max_rows");
    }
    const uint64_t current = static_cast<uint64_t>(
        ifs_cpu_index_capacity(index->core));
    if (required <= current) return IFS_SEARCH_OK;
    uint64_t grown = current ? current : 1;
    while (grown < required) {
        const long double candidate = std::ceil(
            static_cast<long double>(grown) * index->growth_factor);
        if (candidate > std::numeric_limits<uint64_t>::max()) {
            grown = required;
            break;
        }
        const uint64_t next = static_cast<uint64_t>(candidate);
        if (next <= grown) {
            grown = required;
            break;
        }
        grown = next;
    }
    if (index->max_rows) grown = std::min(grown, index->max_rows);
    return reserve_locked(index, std::max(grown, required), true);
}

void zero_timings(ifs_search_timings_t *timings) {
    if (!timings) return;
    timings->kernel_ms = 0.0;
    timings->topk_ms = 0.0;
    timings->total_ms = 0.0;
}

}  // namespace

extern "C" {

uint32_t ifs_search_abi_version(void) { return IFS_SEARCH_ABI_VERSION; }

uint32_t ifs_search_dimension(void) { return IFS_SEARCH_DIMENSION; }

const char *ifs_search_build_info(void) {
    static thread_local std::string info;
    const char *features = ifs_cpu_runtime_features();
    info = "ifs-search-cpu ABI=2 d512 exact-flat raw-cosine "
           "profiles=fp32,bf16,int8_x1000,int8_x736 "
           "grouped_topk=host-reference; ";
    info += features && features[0] ? features : "runtime-features-unavailable";
    return info.c_str();
}

const char *ifs_search_last_error(void) { return g_last_error.c_str(); }

const char *ifs_search_status_string(ifs_search_status_t status) {
    switch (status) {
        case IFS_SEARCH_OK: return "ok";
        case IFS_SEARCH_INVALID_ARGUMENT: return "invalid_argument";
        case IFS_SEARCH_OUT_OF_MEMORY: return "out_of_memory";
        case IFS_SEARCH_UNSUPPORTED: return "unsupported";
        case IFS_SEARCH_DUPLICATE_ID: return "duplicate_id";
        case IFS_SEARCH_ID_NOT_FOUND: return "id_not_found";
        case IFS_SEARCH_CAPACITY_EXCEEDED: return "capacity_exceeded";
        case IFS_SEARCH_BACKEND_ERROR: return "backend_error";
        case IFS_SEARCH_INTERNAL_ERROR: return "internal_error";
    }
    return "unknown_status";
}

ifs_search_status_t ifs_search_get_capabilities(
    int32_t device, ifs_search_capabilities_t *out) {
    clear_error();
    if (!out || out->struct_size < sizeof(*out)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "capabilities struct_size is too small");
    }
    if (device != -1 && device != 0) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "the CPU backend accepts device -1 or 0 only");
    }
    const uint32_t requested_size = out->struct_size;
    *out = {};
    out->struct_size = requested_size;
    out->abi_version = IFS_SEARCH_ABI_VERSION;
    out->dimension = IFS_SEARCH_DIMENSION;
    out->backend = IFS_SEARCH_BACKEND_CPU;
    out->profile_mask =
        (UINT64_C(1) << IFS_SEARCH_PROFILE_FP32_V1) |
        (UINT64_C(1) << IFS_SEARCH_PROFILE_BF16_V1) |
        (UINT64_C(1) << IFS_SEARCH_PROFILE_INT8_X1000_V1) |
        (UINT64_C(1) << IFS_SEARCH_PROFILE_INT8_X736_V1);
    out->flags = IFS_SEARCH_CAP_EXACT_FLAT_SCAN |
                 IFS_SEARCH_CAP_BATCH_ADD |
                 IFS_SEARCH_CAP_BATCH_DELETE |
                 IFS_SEARCH_CAP_RESERVE |
                 IFS_SEARCH_CAP_GROUPED_PERSON_TOPK |
                 IFS_SEARCH_CAP_DELETED_SLOT_REUSE |
                 IFS_SEARCH_CAP_GROUPED_HOST_REFERENCE;
    out->device = -1;
    out->compute_capability_major = -1;
    out->compute_capability_minor = -1;
    return IFS_SEARCH_OK;
}

ifs_search_status_t ifs_search_create(
    const ifs_search_create_options_t *options,
    ifs_search_index_t *out_index) {
    clear_error();
    if (!out_index) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT, "out_index must not be null");
    }
    *out_index = nullptr;
    if (!options || options->struct_size < sizeof(*options)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "create options struct_size is too small");
    }
    if (options->profile == IFS_SEARCH_PROFILE_FP16_V1) {
        return fail(IFS_SEARCH_UNSUPPORTED,
                    "fp16_v1 is GPU-only and is not supported by the CPU backend");
    }
    if (!valid_profile(options->profile)) {
        return fail(IFS_SEARCH_UNSUPPORTED, "unsupported CPU search profile");
    }
    if (options->device != -1 && options->device != 0) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "the CPU backend accepts device -1 or 0 only");
    }
    if (options->topk_mode == IFS_SEARCH_TOPK_DEVICE) {
        return fail(IFS_SEARCH_UNSUPPORTED,
                    "device Top-K is not supported by the CPU backend");
    }
    if (options->topk_mode > IFS_SEARCH_TOPK_DEVICE) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT, "invalid Top-K mode");
    }
    if (options->max_rows && options->reserve_rows > options->max_rows) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "reserve_rows must not exceed max_rows");
    }
    const double growth = options->growth_factor == 0.0
        ? kDefaultGrowthFactor : options->growth_factor;
    if (!(growth >= 1.1 && growth <= 4.0)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "growth_factor must be between 1.1 and 4.0");
    }
    if (options->reserve_rows > std::numeric_limits<size_t>::max()) {
        return fail(IFS_SEARCH_CAPACITY_EXCEEDED,
                    "reserve_rows exceeds the process address space");
    }

    try {
        std::unique_ptr<CpuIndex> index(new CpuIndex());
        index->profile = static_cast<ifs_search_profile_t>(options->profile);
        index->max_rows = options->max_rows;
        index->growth_factor = growth;
        const ifs_cpu_status_t status = ifs_cpu_index_create(
            storage_for(options->profile),
            static_cast<size_t>(options->reserve_rows),
            int8_scale_for(options->profile), &index->core);
        if (status != IFS_CPU_OK) {
            return legacy_failure(status, "CPU index creation failed");
        }
        *out_index = index.release();
        return IFS_SEARCH_OK;
    } catch (const std::bad_alloc &) {
        return fail(IFS_SEARCH_OUT_OF_MEMORY, "CPU index allocation failed");
    } catch (const std::exception &error) {
        return fail(IFS_SEARCH_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(IFS_SEARCH_INTERNAL_ERROR, "unknown CPU creation failure");
    }
}

void ifs_search_destroy(ifs_search_index_t index) {
    clear_error();
    delete as_index(index);
}

ifs_search_status_t ifs_search_reserve(ifs_search_index_t opaque,
                                       uint64_t rows) {
    clear_error();
    if (!opaque) return fail(IFS_SEARCH_INVALID_ARGUMENT, "index is null");
    CpuIndex *index = as_index(opaque);
    std::unique_lock<std::shared_mutex> guard(index->state_mutex);
    return reserve_locked(index, rows, false);
}

ifs_search_status_t ifs_search_add_batch(
    ifs_search_index_t opaque, const uint64_t *ids,
    const uint64_t *group_ids, const float *vectors, uint64_t count) {
    clear_error();
    if (!opaque || (count && (!ids || !group_ids || !vectors))) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "index, vector IDs, group IDs, and vectors are required for a non-empty add");
    }
    if (count > std::numeric_limits<size_t>::max()) {
        return fail(IFS_SEARCH_CAPACITY_EXCEEDED,
                    "batch count exceeds the process address space");
    }
    std::string validation_error;
    if (!validate_normalized_rows(vectors, count, &validation_error)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT, validation_error);
    }
    CpuIndex *index = as_index(opaque);
    std::unique_lock<std::shared_mutex> guard(index->state_mutex);
    try {
        index->vector_to_group.reserve(
            index->vector_to_group.size() + static_cast<size_t>(count));
        std::unordered_set<uint64_t> batch_ids;
        batch_ids.reserve(static_cast<size_t>(count));
        for (uint64_t offset = 0; offset < count; ++offset) {
            if (index->vector_to_group.find(ids[offset]) !=
                index->vector_to_group.end()) {
                return fail(IFS_SEARCH_DUPLICATE_ID,
                            "duplicate active vector ID in batch add");
            }
            if (!batch_ids.insert(ids[offset]).second) {
                return fail(IFS_SEARCH_DUPLICATE_ID,
                            "duplicate vector ID within batch add");
            }
        }
    } catch (const std::bad_alloc &) {
        return fail(IFS_SEARCH_OUT_OF_MEMORY,
                    "unable to reserve vector-to-group mapping");
    }
    ifs_search_status_t capacity_status = ensure_add_capacity(index, count);
    if (capacity_status != IFS_SEARCH_OK) return capacity_status;
    try {
        for (uint64_t offset = 0; offset < count; ++offset) {
            index->vector_to_group.emplace(ids[offset], group_ids[offset]);
        }
    } catch (const std::bad_alloc &) {
        for (uint64_t offset = 0; offset < count; ++offset) {
            index->vector_to_group.erase(ids[offset]);
        }
        return fail(IFS_SEARCH_OUT_OF_MEMORY,
                    "unable to populate vector-to-group mapping");
    } catch (...) {
        for (uint64_t offset = 0; offset < count; ++offset) {
            index->vector_to_group.erase(ids[offset]);
        }
        return fail(IFS_SEARCH_INTERNAL_ERROR,
                    "unable to populate vector-to-group mapping");
    }
    const ifs_cpu_status_t status = ifs_cpu_index_add_batch(
        index->core, ids, vectors, static_cast<size_t>(count));
    if (status != IFS_CPU_OK) {
        for (uint64_t offset = 0; offset < count; ++offset) {
            index->vector_to_group.erase(ids[offset]);
        }
        return legacy_failure(status, "CPU batch add failed");
    }
    const uint64_t reused = std::min(count, index->tombstone_rows);
    index->tombstone_rows -= reused;
    index->physical_rows += count - reused;
    index->live_rows += count;
    return IFS_SEARCH_OK;
}

ifs_search_status_t ifs_search_delete_batch(
    ifs_search_index_t opaque, const uint64_t *ids, uint64_t count,
    uint64_t *out_removed) {
    clear_error();
    if (!out_removed) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT, "out_removed must not be null");
    }
    *out_removed = 0;
    if (!opaque || (count && !ids)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "index and IDs are required for a non-empty delete");
    }
    CpuIndex *index = as_index(opaque);
    std::unique_lock<std::shared_mutex> guard(index->state_mutex);
    uint64_t removed = 0;
    for (uint64_t offset = 0; offset < count; ++offset) {
        const ifs_cpu_status_t status = ifs_cpu_index_delete(index->core, ids[offset]);
        if (status == IFS_CPU_OK) {
            index->vector_to_group.erase(ids[offset]);
            ++removed;
        } else if (status != IFS_CPU_ID_NOT_FOUND) {
            return legacy_failure(status, "CPU batch delete failed");
        }
    }
    index->live_rows -= removed;
    index->tombstone_rows += removed;
    *out_removed = removed;
    return IFS_SEARCH_OK;
}

ifs_search_status_t ifs_search_topk(
    ifs_search_index_t opaque, const float *query, uint64_t top_k,
    uint64_t *out_ids, float *out_scores, uint64_t *out_count,
    ifs_search_timings_t *timings) {
    clear_error();
    if (!out_count) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT, "out_count must not be null");
    }
    *out_count = 0;
    if (timings && timings->struct_size < sizeof(*timings)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "timings struct_size is too small");
    }
    zero_timings(timings);
    if (!opaque || !query || (top_k && (!out_ids || !out_scores))) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "invalid index, query, or Top-K output buffers");
    }
    if (top_k > std::numeric_limits<size_t>::max()) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "top_k exceeds the process address space");
    }
    std::string validation_error;
    if (!validate_normalized_rows(query, 1, &validation_error)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT, validation_error);
    }
    CpuIndex *index = as_index(opaque);
    size_t count = 0;
    const auto begin = std::chrono::steady_clock::now();
    const ifs_cpu_status_t status = ifs_cpu_index_search(
        index->core, query, static_cast<size_t>(top_k), out_ids, out_scores, &count);
    const auto end = std::chrono::steady_clock::now();
    if (status != IFS_CPU_OK) return legacy_failure(status, "CPU search failed");
    if (int8_scale_for(index->profile) != 0) {
        const float divisor = int8_score_divisor(index->profile);
        for (size_t offset = 0; offset < count; ++offset) {
            out_scores[offset] /= divisor;
        }
    }
    for (size_t offset = 0; offset < count; ++offset) {
        out_scores[offset] = public_cosine(out_scores[offset]);
    }
    *out_count = static_cast<uint64_t>(count);
    if (timings) {
        timings->total_ms = std::chrono::duration<double, std::milli>(
            end - begin).count();
        timings->kernel_ms = timings->total_ms;
    }
    return IFS_SEARCH_OK;
}

ifs_search_status_t ifs_search_grouped_topk(
    ifs_search_index_t opaque, const float *query, uint64_t top_k,
    uint64_t *out_group_ids, uint64_t *out_vector_ids, float *out_scores,
    uint64_t *out_count, ifs_search_timings_t *timings) {
    clear_error();
    if (!out_count) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT, "out_count must not be null");
    }
    *out_count = 0;
    if (timings && timings->struct_size < sizeof(*timings)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "timings struct_size is too small");
    }
    zero_timings(timings);
    if (!opaque || !query ||
        (top_k && (!out_group_ids || !out_vector_ids || !out_scores))) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "invalid grouped Top-K input or output buffers");
    }
    std::string validation_error;
    if (!validate_normalized_rows(query, 1, &validation_error)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT, validation_error);
    }

    struct GroupCandidate {
        uint64_t group_id;
        uint64_t vector_id;
        float score;
    };
    const auto better = [](const GroupCandidate &left,
                           const GroupCandidate &right) {
        if (left.score != right.score) return left.score > right.score;
        return left.group_id < right.group_id;
    };

    CpuIndex *index = as_index(opaque);
    std::shared_lock<std::shared_mutex> guard(index->state_mutex);
    const auto begin = std::chrono::steady_clock::now();
    try {
        const size_t live = static_cast<size_t>(index->live_rows);
        std::vector<uint64_t> face_ids(live);
        std::vector<float> face_scores(live);
        size_t face_count = 0;
        const ifs_cpu_status_t status = ifs_cpu_index_search(
            index->core, query, live, face_ids.data(), face_scores.data(),
            &face_count);
        if (status != IFS_CPU_OK) {
            return legacy_failure(status, "CPU grouped reference search failed");
        }
        if (int8_scale_for(index->profile) != 0) {
            const float divisor = int8_score_divisor(index->profile);
            for (size_t offset = 0; offset < face_count; ++offset) {
                face_scores[offset] /= divisor;
            }
        }

        std::unordered_map<uint64_t, GroupCandidate> best_by_group;
        best_by_group.reserve(index->vector_to_group.size());
        for (size_t offset = 0; offset < face_count; ++offset) {
            const auto group = index->vector_to_group.find(face_ids[offset]);
            if (group == index->vector_to_group.end()) {
                return fail(IFS_SEARCH_INTERNAL_ERROR,
                            "active vector is missing its group mapping");
            }
            /* face results are score descending, vector ID ascending, so the
             * first face observed for a group is its deterministic maximum. */
            best_by_group.emplace(
                group->second,
                GroupCandidate{group->second, face_ids[offset],
                               face_scores[offset]});
        }
        std::vector<GroupCandidate> groups;
        groups.reserve(best_by_group.size());
        for (const auto &entry : best_by_group) groups.push_back(entry.second);
        std::sort(groups.begin(), groups.end(), better);
        const size_t result_count = std::min<size_t>(
            static_cast<size_t>(std::min<uint64_t>(
                top_k, std::numeric_limits<size_t>::max())), groups.size());
        for (size_t offset = 0; offset < result_count; ++offset) {
            out_group_ids[offset] = groups[offset].group_id;
            out_vector_ids[offset] = groups[offset].vector_id;
            out_scores[offset] = public_cosine(groups[offset].score);
        }
        *out_count = static_cast<uint64_t>(result_count);
    } catch (const std::bad_alloc &) {
        return fail(IFS_SEARCH_OUT_OF_MEMORY,
                    "CPU grouped reference allocation failed");
    } catch (const std::exception &error) {
        return fail(IFS_SEARCH_INTERNAL_ERROR, error.what());
    }
    const auto end = std::chrono::steady_clock::now();
    if (timings) {
        timings->total_ms = std::chrono::duration<double, std::milli>(
            end - begin).count();
        timings->kernel_ms = timings->total_ms;
    }
    return IFS_SEARCH_OK;
}

ifs_search_status_t ifs_search_get_stats(
    ifs_search_index_t opaque, ifs_search_stats_t *out) {
    clear_error();
    if (!opaque || !out || out->struct_size < sizeof(*out)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "index and a complete stats structure are required");
    }
    CpuIndex *index = as_index(opaque);
    std::shared_lock<std::shared_mutex> guard(index->state_mutex);
    const uint32_t requested_size = out->struct_size;
    *out = {};
    out->struct_size = requested_size;
    out->backend = IFS_SEARCH_BACKEND_CPU;
    out->profile = index->profile;
    out->device = -1;
    out->physical_rows = index->physical_rows;
    out->live_rows = index->live_rows;
    out->capacity_rows = static_cast<uint64_t>(
        ifs_cpu_index_capacity(index->core));
    out->max_rows = index->max_rows;
    out->tombstone_rows = index->tombstone_rows;
    out->reallocations = index->reallocations;
    out->bytes_per_vector = bytes_per_vector(index->profile);
    return IFS_SEARCH_OK;
}

}  // extern "C"
