#include "ifs_search.h"

#include "ifs_cuda_legacy.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>

namespace {

constexpr double kDefaultGrowthFactor = 1.5;
constexpr double kNormalizedSquaredTolerance = 1.0e-3;
constexpr uint64_t kDeviceTopKLimit = 100;

thread_local std::string g_last_error;

struct CudaIndex {
    void *core = nullptr;
    ifs_search_profile_t profile = IFS_SEARCH_PROFILE_FP32_V1;
    int32_t device = 0;
    ifs_search_topk_mode_t topk_mode = IFS_SEARCH_TOPK_AUTO;
    uint64_t max_rows = 0;
    mutable std::mutex operation_mutex;

    ~CudaIndex() { ifs_cuda_destroy(core); }
};

ifs_search_status_t fail(ifs_search_status_t status, const std::string &message) {
    g_last_error = message;
    return status;
}

void clear_error() { g_last_error.clear(); }

float public_cosine(float score) {
    return std::clamp(score, -1.0f, 1.0f);
}

CudaIndex *as_index(ifs_search_index_t index) {
    return static_cast<CudaIndex *>(index);
}

bool valid_profile(uint32_t profile) {
    return profile == IFS_SEARCH_PROFILE_FP32_V1 ||
           profile == IFS_SEARCH_PROFILE_FP16_V1 ||
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

uint64_t profile_mask_for_compute_capability(int major, int minor) {
    const int capability = major * 10 + minor;
    if (capability < 75) return 0;
    uint64_t mask =
        (UINT64_C(1) << IFS_SEARCH_PROFILE_FP32_V1) |
        (UINT64_C(1) << IFS_SEARCH_PROFILE_FP16_V1) |
        (UINT64_C(1) << IFS_SEARCH_PROFILE_INT8_X1000_V1) |
        (UINT64_C(1) << IFS_SEARCH_PROFILE_INT8_X736_V1);
    /* Native BF16 matrix operations require Ampere (SM80) or newer. */
    if (capability >= 80) {
        mask |= UINT64_C(1) << IFS_SEARCH_PROFILE_BF16_V1;
    }
    return mask;
}

int legacy_dtype(uint32_t profile) {
    switch (profile) {
        case IFS_SEARCH_PROFILE_FP32_V1: return IFS_CUDA_FP32;
        case IFS_SEARCH_PROFILE_FP16_V1: return IFS_CUDA_FP16;
        case IFS_SEARCH_PROFILE_BF16_V1: return IFS_CUDA_BF16;
        case IFS_SEARCH_PROFILE_INT8_X1000_V1: return IFS_CUDA_INT8;
        case IFS_SEARCH_PROFILE_INT8_X736_V1: return IFS_CUDA_INT8;
        default: return -1;
    }
}

ifs_search_status_t classify_legacy_error(const char *error) {
    if (!error) return IFS_SEARCH_BACKEND_ERROR;
    if (std::strstr(error, "duplicate")) return IFS_SEARCH_DUPLICATE_ID;
    if (std::strstr(error, "out of memory") ||
        std::strstr(error, "allocation") || std::strstr(error, "allocate")) {
        return IFS_SEARCH_OUT_OF_MEMORY;
    }
    if (std::strstr(error, "unsupported")) return IFS_SEARCH_UNSUPPORTED;
    if (std::strstr(error, "null") || std::strstr(error, "must be") ||
        std::strstr(error, "invalid")) {
        return IFS_SEARCH_INVALID_ARGUMENT;
    }
    return IFS_SEARCH_BACKEND_ERROR;
}

ifs_search_status_t legacy_failure(const char *fallback) {
    const char *legacy = ifs_cuda_last_error();
    const std::string message = legacy && legacy[0] ? legacy : fallback;
    return fail(classify_legacy_error(message.c_str()), message);
}

bool validate_normalized_rows(const float *vectors, uint64_t count,
                              std::string *message) {
    if (count && !vectors) {
        *message = "vectors must not be null for a non-empty batch";
        return false;
    }
    if (count > std::numeric_limits<size_t>::max() / IFS_SEARCH_DIMENSION) {
        *message = "vector batch is too large for this process";
        return false;
    }
    for (uint64_t row = 0; row < count; ++row) {
        const float *vector = vectors + static_cast<size_t>(row) * IFS_SEARCH_DIMENSION;
        double squared_norm = 0.0;
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

int resolve_topk_mode(ifs_search_topk_mode_t configured, uint64_t top_k) {
    if (configured == IFS_SEARCH_TOPK_HOST) return IFS_CUDA_TOPK_HOST;
    if (configured == IFS_SEARCH_TOPK_DEVICE) return IFS_CUDA_TOPK_DEVICE;
    return top_k <= kDeviceTopKLimit
        ? IFS_CUDA_TOPK_DEVICE : IFS_CUDA_TOPK_HOST;
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
    info = "ifs-search-cuda ABI=2 d512 exact-flat raw-cosine "
           "profiles=fp32,fp16,bf16,int8_x1000,int8_x736 "
           "grouped_topk=device-resident-exact; ";
    const char *legacy = ifs_cuda_build_info();
    info += legacy && legacy[0] ? legacy : "legacy-build-info-unavailable";
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
    if (device < 0) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "CUDA device ordinal must be non-negative");
    }
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return fail(IFS_SEARCH_BACKEND_ERROR,
                    std::string("cudaGetDeviceProperties: ") +
                    cudaGetErrorString(status));
    }
    int runtime_version = 0;
    status = cudaRuntimeGetVersion(&runtime_version);
    if (status != cudaSuccess) {
        return fail(IFS_SEARCH_BACKEND_ERROR,
                    std::string("cudaRuntimeGetVersion: ") +
                    cudaGetErrorString(status));
    }
    int driver_version = 0;
    status = cudaDriverGetVersion(&driver_version);
    if (status != cudaSuccess) {
        return fail(IFS_SEARCH_BACKEND_ERROR,
                    std::string("cudaDriverGetVersion: ") +
                    cudaGetErrorString(status));
    }
    const uint32_t requested_size = out->struct_size;
    *out = {};
    out->struct_size = requested_size;
    out->abi_version = IFS_SEARCH_ABI_VERSION;
    out->dimension = IFS_SEARCH_DIMENSION;
    out->backend = IFS_SEARCH_BACKEND_CUDA;
    out->profile_mask = profile_mask_for_compute_capability(
        properties.major, properties.minor);
    out->flags = IFS_SEARCH_CAP_EXACT_FLAT_SCAN |
                 IFS_SEARCH_CAP_BATCH_ADD |
                 IFS_SEARCH_CAP_BATCH_DELETE |
                 IFS_SEARCH_CAP_RESERVE |
                 IFS_SEARCH_CAP_DEVICE_TOPK |
                 IFS_SEARCH_CAP_GROUPED_PERSON_TOPK |
                 IFS_SEARCH_CAP_TOMBSTONE_DELETE |
                 IFS_SEARCH_CAP_GROUPED_DEVICE_RESIDENT;
    out->device_topk_limit = kDeviceTopKLimit;
    out->device = device;
    out->compute_capability_major = properties.major;
    out->compute_capability_minor = properties.minor;
    out->cuda_runtime_version = runtime_version;
    out->cuda_driver_version = driver_version;
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
    if (!valid_profile(options->profile)) {
        return fail(IFS_SEARCH_UNSUPPORTED, "unsupported CUDA search profile");
    }
    if (options->device < 0) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "CUDA device ordinal must be non-negative");
    }
    cudaDeviceProp properties{};
    const cudaError_t property_status = cudaGetDeviceProperties(
        &properties, options->device);
    if (property_status != cudaSuccess) {
        return fail(IFS_SEARCH_BACKEND_ERROR,
                    std::string("cudaGetDeviceProperties: ") +
                    cudaGetErrorString(property_status));
    }
    const uint64_t supported_profiles = profile_mask_for_compute_capability(
        properties.major, properties.minor);
    if ((supported_profiles & (UINT64_C(1) << options->profile)) == 0) {
        return fail(IFS_SEARCH_UNSUPPORTED,
                    "requested profile is unsupported by this CUDA compute capability");
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
    try {
        std::unique_ptr<CudaIndex> index(new CudaIndex());
        index->profile = static_cast<ifs_search_profile_t>(options->profile);
        index->device = options->device;
        index->topk_mode = static_cast<ifs_search_topk_mode_t>(
            options->topk_mode);
        index->max_rows = options->max_rows;
        index->core = ifs_cuda_create(legacy_dtype(options->profile),
                                      options->reserve_rows, options->device,
                                      growth, int8_scale_for(options->profile));
        if (!index->core) return legacy_failure("CUDA index creation failed");
        *out_index = index.release();
        return IFS_SEARCH_OK;
    } catch (const std::bad_alloc &) {
        return fail(IFS_SEARCH_OUT_OF_MEMORY, "CUDA wrapper allocation failed");
    } catch (const std::exception &error) {
        return fail(IFS_SEARCH_INTERNAL_ERROR, error.what());
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
    CudaIndex *index = as_index(opaque);
    std::lock_guard<std::mutex> guard(index->operation_mutex);
    if (index->max_rows && rows > index->max_rows) {
        return fail(IFS_SEARCH_CAPACITY_EXCEEDED,
                    "requested capacity exceeds the configured max_rows");
    }
    if (ifs_cuda_reserve(index->core, rows) != 0) {
        return legacy_failure("CUDA reserve failed");
    }
    return IFS_SEARCH_OK;
}

ifs_search_status_t ifs_search_add_batch(
    ifs_search_index_t opaque, const uint64_t *ids,
    const uint64_t *group_ids, const float *vectors, uint64_t count) {
    clear_error();
    if (!opaque || (count && (!ids || !group_ids || !vectors))) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT,
                    "index, vector IDs, group IDs, and vectors are required for a non-empty add");
    }
    std::string validation_error;
    if (!validate_normalized_rows(vectors, count, &validation_error)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT, validation_error);
    }
    CudaIndex *index = as_index(opaque);
    std::lock_guard<std::mutex> guard(index->operation_mutex);
    ifs_cuda_stats stats{};
    if (ifs_cuda_get_stats(index->core, &stats) != 0) {
        return legacy_failure("CUDA stats failed before add");
    }
    if (count > std::numeric_limits<uint64_t>::max() - stats.size ||
        (index->max_rows && stats.size + count > index->max_rows)) {
        return fail(IFS_SEARCH_CAPACITY_EXCEEDED,
                    "add would exceed CUDA max_rows; tombstones require a rebuild");
    }
    if (ifs_cuda_add(index->core, vectors, ids, group_ids, count) != 0) {
        return legacy_failure("CUDA batch add failed");
    }
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
    CudaIndex *index = as_index(opaque);
    std::lock_guard<std::mutex> guard(index->operation_mutex);
    if (ifs_cuda_remove(index->core, ids, count, out_removed) != 0) {
        return legacy_failure("CUDA batch delete failed");
    }
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
    std::string validation_error;
    if (!validate_normalized_rows(query, 1, &validation_error)) {
        return fail(IFS_SEARCH_INVALID_ARGUMENT, validation_error);
    }
    CudaIndex *index = as_index(opaque);
    std::lock_guard<std::mutex> guard(index->operation_mutex);
    const int mode = resolve_topk_mode(index->topk_mode, top_k);
    if (mode == IFS_CUDA_TOPK_DEVICE && top_k > kDeviceTopKLimit) {
        return fail(IFS_SEARCH_UNSUPPORTED,
                    "explicit CUDA device Top-K supports at most 100 results");
    }
    double kernel_ms = 0.0;
    double topk_ms = 0.0;
    double total_ms = 0.0;
    if (ifs_cuda_search_ex(index->core, query, top_k, mode, out_ids,
                           out_scores, out_count, &kernel_ms, &topk_ms,
                           &total_ms) != 0) {
        return legacy_failure("CUDA search failed");
    }
    for (uint64_t offset = 0; offset < *out_count; ++offset) {
        out_scores[offset] = public_cosine(out_scores[offset]);
    }
    if (timings) {
        timings->kernel_ms = kernel_ms;
        timings->topk_ms = topk_ms;
        timings->total_ms = total_ms;
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

    CudaIndex *index = as_index(opaque);
    std::lock_guard<std::mutex> guard(index->operation_mutex);
    if (top_k > kDeviceTopKLimit) {
        return fail(IFS_SEARCH_UNSUPPORTED,
                    "CUDA grouped device Top-K supports at most 100 results");
    }
    double kernel_ms = 0.0;
    double topk_ms = 0.0;
    double total_ms = 0.0;
    if (ifs_cuda_grouped_search(
            index->core, query, top_k, out_group_ids, out_vector_ids,
            out_scores, out_count, &kernel_ms, &topk_ms, &total_ms) != 0) {
        return legacy_failure("CUDA GPU-resident grouped search failed");
    }
    for (uint64_t offset = 0; offset < *out_count; ++offset) {
        out_scores[offset] = public_cosine(out_scores[offset]);
    }
    if (timings) {
        timings->kernel_ms = kernel_ms;
        timings->topk_ms = topk_ms;
        timings->total_ms = total_ms;
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
    CudaIndex *index = as_index(opaque);
    std::lock_guard<std::mutex> guard(index->operation_mutex);
    ifs_cuda_stats legacy{};
    if (ifs_cuda_get_stats(index->core, &legacy) != 0) {
        return legacy_failure("CUDA stats failed");
    }
    const uint32_t requested_size = out->struct_size;
    *out = {};
    out->struct_size = requested_size;
    out->backend = IFS_SEARCH_BACKEND_CUDA;
    out->profile = index->profile;
    out->device = index->device;
    out->physical_rows = legacy.size;
    out->live_rows = legacy.live_size;
    out->capacity_rows = legacy.capacity;
    out->max_rows = index->max_rows;
    out->tombstone_rows = legacy.size - legacy.live_size;
    out->reallocations = legacy.reallocations;
    out->bytes_per_vector = legacy.bytes_per_vector;
    return IFS_SEARCH_OK;
}

}  // extern "C"
