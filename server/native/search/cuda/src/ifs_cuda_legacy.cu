#include "ifs_cuda_legacy.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr int kDimension = 512;
constexpr int kTopKBlockSize = 256;
constexpr uint64_t kTopKChunkRows = 8192;
constexpr uint64_t kDeviceTopKLimit = 100;
thread_local std::string g_last_error;

void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << what << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

void check_cublas(cublasStatus_t status, const char *what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream message;
        message << what << ": cuBLAS status " << static_cast<int>(status);
        throw std::runtime_error(message.str());
    }
}

size_t element_bytes(int dtype) {
    switch (dtype) {
        case IFS_CUDA_FP32: return sizeof(float);
        case IFS_CUDA_FP16: return sizeof(__half);
        case IFS_CUDA_BF16: return sizeof(__nv_bfloat16);
        case IFS_CUDA_INT8: return sizeof(int8_t);
        default: throw std::invalid_argument("unsupported CUDA index dtype");
    }
}

__global__ void float_to_half(const float *input, __half *output, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2half_rn(input[i]);
}

__global__ void float_to_bfloat16(const float *input, __nv_bfloat16 *output,
                                  size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2bfloat16_rn(input[i]);
}

__global__ void float_to_int8(const float *input, int8_t *output,
                              size_t count, float scale) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float scaled = input[i] * scale;
        int value = 0;
        if (scaled >= 127.0f) value = 127;
        else if (scaled <= -128.0f) value = -128;
        else value = static_cast<int>(roundf(scaled));
        output[i] = static_cast<int8_t>(value);
    }
}

__global__ void mark_dead(uint8_t *alive, const uint64_t *slots,
                          uint64_t count) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) alive[slots[i]] = 0;
}

template <typename Score>
struct alignas(16) DeviceCandidate {
    Score score;
    uint64_t id;
};

template <typename Score>
__device__ __forceinline__ Score worst_score();

template <>
__device__ __forceinline__ float worst_score<float>() {
    return -INFINITY;
}

template <>
__device__ __forceinline__ int32_t worst_score<int32_t>() {
    return INT32_MIN;
}

template <typename Score>
__device__ __forceinline__ DeviceCandidate<Score> worst_candidate() {
    return DeviceCandidate<Score>{worst_score<Score>(), UINT64_MAX};
}

template <typename Score>
__device__ __forceinline__ bool device_better(
    const DeviceCandidate<Score> &left,
    const DeviceCandidate<Score> &right) {
    if (left.score != right.score) return left.score > right.score;
    return left.id < right.id;
}

template <typename Score>
__device__ __forceinline__ void bitonic_sort_best_first(
    DeviceCandidate<Score> *values) {
    const unsigned int tid = threadIdx.x;
    #pragma unroll
    for (unsigned int sequence = 2; sequence <= kTopKBlockSize;
         sequence <<= 1) {
        #pragma unroll
        for (unsigned int stride = sequence >> 1; stride > 0; stride >>= 1) {
            const unsigned int peer = tid ^ stride;
            if (peer > tid) {
                const DeviceCandidate<Score> mine = values[tid];
                const DeviceCandidate<Score> theirs = values[peer];
                const bool best_first_segment = (tid & sequence) == 0;
                const bool swap = best_first_segment
                    ? device_better(theirs, mine)
                    : device_better(mine, theirs);
                if (swap) {
                    values[tid] = theirs;
                    values[peer] = mine;
                }
            }
            __syncthreads();
        }
    }
}

template <typename Score>
__global__ void topk_from_scores(
    const Score *scores, const uint64_t *ids, const uint8_t *alive,
    uint64_t count, uint64_t k, DeviceCandidate<Score> *output) {
    __shared__ DeviceCandidate<Score> candidates[kTopKBlockSize];
    const uint64_t range_begin = static_cast<uint64_t>(blockIdx.x) * kTopKChunkRows;
    const uint64_t range_end = min(count, range_begin + kTopKChunkRows);
    const uint64_t first = range_begin + threadIdx.x;
    candidates[threadIdx.x] = first < range_end && alive[first]
        ? DeviceCandidate<Score>{scores[first], ids[first]}
        : worst_candidate<Score>();
    __syncthreads();
    bitonic_sort_best_first(candidates);

    uint64_t cursor = range_begin + kTopKBlockSize;
    const uint64_t fresh_per_round = kTopKBlockSize - k;
    while (cursor < range_end) {
        if (threadIdx.x >= k) {
            const uint64_t row = cursor + threadIdx.x - k;
            candidates[threadIdx.x] = row < range_end && alive[row]
                ? DeviceCandidate<Score>{scores[row], ids[row]}
                : worst_candidate<Score>();
        }
        __syncthreads();
        bitonic_sort_best_first(candidates);
        cursor += fresh_per_round;
    }
    if (threadIdx.x < k) {
        output[static_cast<uint64_t>(blockIdx.x) * k + threadIdx.x] =
            candidates[threadIdx.x];
    }
}

template <typename Score>
__global__ void topk_from_candidates(
    const DeviceCandidate<Score> *input, uint64_t count, uint64_t k,
    DeviceCandidate<Score> *output) {
    __shared__ DeviceCandidate<Score> candidates[kTopKBlockSize];
    const uint64_t range_begin = static_cast<uint64_t>(blockIdx.x) * kTopKChunkRows;
    const uint64_t range_end = min(count, range_begin + kTopKChunkRows);
    const uint64_t first = range_begin + threadIdx.x;
    candidates[threadIdx.x] = first < range_end
        ? input[first] : worst_candidate<Score>();
    __syncthreads();
    bitonic_sort_best_first(candidates);

    uint64_t cursor = range_begin + kTopKBlockSize;
    const uint64_t fresh_per_round = kTopKBlockSize - k;
    while (cursor < range_end) {
        if (threadIdx.x >= k) {
            const uint64_t row = cursor + threadIdx.x - k;
            candidates[threadIdx.x] = row < range_end
                ? input[row] : worst_candidate<Score>();
        }
        __syncthreads();
        bitonic_sort_best_first(candidates);
        cursor += fresh_per_round;
    }
    if (threadIdx.x < k) {
        output[static_cast<uint64_t>(blockIdx.x) * k + threadIdx.x] =
            candidates[threadIdx.x];
    }
}

template <typename Score>
__device__ __forceinline__ uint32_t ordered_score_key(Score score);

template <>
__device__ __forceinline__ uint32_t ordered_score_key<float>(float score) {
    /* Canonicalize signed zero so tie-breaking agrees with C++ float equality
     * and the host reference path. */
    if (score == 0.0f) return UINT32_C(0x80000000);
    const uint32_t bits = __float_as_uint(score);
    return (bits & UINT32_C(0x80000000)) ? ~bits
                                         : bits ^ UINT32_C(0x80000000);
}

template <>
__device__ __forceinline__ uint32_t ordered_score_key<int32_t>(int32_t score) {
    return static_cast<uint32_t>(score) ^ UINT32_C(0x80000000);
}

template <typename Score>
__device__ __forceinline__ Score score_from_ordered_key(uint32_t key);

template <>
__device__ __forceinline__ float score_from_ordered_key<float>(uint32_t key) {
    const uint32_t bits = (key & UINT32_C(0x80000000))
        ? key ^ UINT32_C(0x80000000) : ~key;
    return __uint_as_float(bits);
}

template <>
__device__ __forceinline__ int32_t score_from_ordered_key<int32_t>(
    uint32_t key) {
    return static_cast<int32_t>(key ^ UINT32_C(0x80000000));
}

template <typename Score>
__global__ void reduce_group_best_scores(
    const Score *scores, const uint32_t *row_group_slots,
    const uint8_t *alive, uint64_t count, uint32_t *group_best_keys) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (row < count && alive[row]) {
        atomicMax(group_best_keys + row_group_slots[row],
                  ordered_score_key<Score>(scores[row]));
    }
}

template <typename Score>
__global__ void reduce_group_best_vector_ids(
    const Score *scores, const uint64_t *vector_ids,
    const uint32_t *row_group_slots, const uint8_t *alive, uint64_t count,
    const uint32_t *group_best_keys, uint64_t *group_best_vector_ids) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (row < count && alive[row]) {
        const uint32_t group_slot = row_group_slots[row];
        if (ordered_score_key<Score>(scores[row]) ==
            group_best_keys[group_slot]) {
            atomicMin(reinterpret_cast<unsigned long long *>(
                          group_best_vector_ids + group_slot),
                      static_cast<unsigned long long>(vector_ids[row]));
        }
    }
}

template <typename Score>
struct alignas(16) DeviceGroupCandidate {
    Score score;
    uint64_t group_id;
    uint64_t vector_id;
};

template <typename Score>
__device__ __forceinline__ DeviceGroupCandidate<Score>
worst_group_candidate() {
    return DeviceGroupCandidate<Score>{worst_score<Score>(), UINT64_MAX,
                                       UINT64_MAX};
}

template <typename Score>
__device__ __forceinline__ bool device_group_better(
    const DeviceGroupCandidate<Score> &left,
    const DeviceGroupCandidate<Score> &right) {
    if (left.score != right.score) return left.score > right.score;
    if (left.group_id != right.group_id) {
        return left.group_id < right.group_id;
    }
    return left.vector_id < right.vector_id;
}

template <typename Score>
__device__ __forceinline__ void bitonic_sort_groups_best_first(
    DeviceGroupCandidate<Score> *values) {
    const unsigned int tid = threadIdx.x;
    #pragma unroll
    for (unsigned int sequence = 2; sequence <= kTopKBlockSize;
         sequence <<= 1) {
        #pragma unroll
        for (unsigned int stride = sequence >> 1; stride > 0; stride >>= 1) {
            const unsigned int peer = tid ^ stride;
            if (peer > tid) {
                const DeviceGroupCandidate<Score> mine = values[tid];
                const DeviceGroupCandidate<Score> theirs = values[peer];
                const bool best_first_segment = (tid & sequence) == 0;
                const bool swap = best_first_segment
                    ? device_group_better(theirs, mine)
                    : device_group_better(mine, theirs);
                if (swap) {
                    values[tid] = theirs;
                    values[peer] = mine;
                }
            }
            __syncthreads();
        }
    }
}

template <typename Score>
__global__ void grouped_topk_from_best(
    const uint32_t *best_keys, const uint64_t *best_vector_ids,
    const uint64_t *group_ids, uint64_t group_count, uint64_t k,
    DeviceGroupCandidate<Score> *output) {
    __shared__ DeviceGroupCandidate<Score> candidates[kTopKBlockSize];
    const uint64_t range_begin =
        static_cast<uint64_t>(blockIdx.x) * kTopKChunkRows;
    const uint64_t range_end = min(group_count, range_begin + kTopKChunkRows);
    const uint64_t first = range_begin + threadIdx.x;
    candidates[threadIdx.x] =
        first < range_end && best_vector_ids[first] != UINT64_MAX
        ? DeviceGroupCandidate<Score>{score_from_ordered_key<Score>(
                                          best_keys[first]),
                                      group_ids[first],
                                      best_vector_ids[first]}
        : worst_group_candidate<Score>();
    __syncthreads();
    bitonic_sort_groups_best_first(candidates);

    uint64_t cursor = range_begin + kTopKBlockSize;
    const uint64_t fresh_per_round = kTopKBlockSize - k;
    while (cursor < range_end) {
        if (threadIdx.x >= k) {
            const uint64_t group_slot = cursor + threadIdx.x - k;
            candidates[threadIdx.x] =
                group_slot < range_end &&
                    best_vector_ids[group_slot] != UINT64_MAX
                ? DeviceGroupCandidate<Score>{
                      score_from_ordered_key<Score>(best_keys[group_slot]),
                      group_ids[group_slot], best_vector_ids[group_slot]}
                : worst_group_candidate<Score>();
        }
        __syncthreads();
        bitonic_sort_groups_best_first(candidates);
        cursor += fresh_per_round;
    }
    if (threadIdx.x < k) {
        output[static_cast<uint64_t>(blockIdx.x) * k + threadIdx.x] =
            candidates[threadIdx.x];
    }
}

template <typename Score>
__global__ void grouped_topk_from_candidates(
    const DeviceGroupCandidate<Score> *input, uint64_t count, uint64_t k,
    DeviceGroupCandidate<Score> *output) {
    __shared__ DeviceGroupCandidate<Score> candidates[kTopKBlockSize];
    const uint64_t range_begin =
        static_cast<uint64_t>(blockIdx.x) * kTopKChunkRows;
    const uint64_t range_end = min(count, range_begin + kTopKChunkRows);
    const uint64_t first = range_begin + threadIdx.x;
    candidates[threadIdx.x] = first < range_end
        ? input[first] : worst_group_candidate<Score>();
    __syncthreads();
    bitonic_sort_groups_best_first(candidates);

    uint64_t cursor = range_begin + kTopKBlockSize;
    const uint64_t fresh_per_round = kTopKBlockSize - k;
    while (cursor < range_end) {
        if (threadIdx.x >= k) {
            const uint64_t row = cursor + threadIdx.x - k;
            candidates[threadIdx.x] = row < range_end
                ? input[row] : worst_group_candidate<Score>();
        }
        __syncthreads();
        bitonic_sort_groups_best_first(candidates);
        cursor += fresh_per_round;
    }
    if (threadIdx.x < k) {
        output[static_cast<uint64_t>(blockIdx.x) * k + threadIdx.x] =
            candidates[threadIdx.x];
    }
}

struct Candidate {
    float score;
    uint64_t id;
};

bool better(const Candidate &left, const Candidate &right) {
    if (left.score != right.score) return left.score > right.score;
    return left.id < right.id;
}

/* priority_queue top is the worst retained candidate. */
struct BetterComparator {
    bool operator()(const Candidate &left, const Candidate &right) const {
        return better(left, right);
    }
};

class CudaFlatIndex {
public:
    CudaFlatIndex(int dtype, uint64_t reserve_rows, int device,
                  double growth_factor, uint32_t int8_scale)
        : dtype_(dtype), device_(device), growth_factor_(growth_factor),
          int8_scale_(int8_scale),
          int8_score_divisor_(static_cast<float>(int8_scale) *
                              static_cast<float>(int8_scale)),
          bytes_per_vector_(static_cast<uint64_t>(kDimension) * element_bytes(dtype)) {
        if (!(growth_factor_ >= 1.1 && growth_factor_ <= 4.0)) {
            throw std::invalid_argument("growth_factor must be between 1.1 and 4.0");
        }
        if ((dtype_ == IFS_CUDA_INT8) != (int8_scale_ > 0)) {
            throw std::invalid_argument(
                "INT8 dtype requires a positive scale and other dtypes require zero");
        }
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        check_cublas(cublasCreate(&cublas_), "cublasCreate");
        check_cublas(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH),
                     "cublasSetMathMode");
        try {
            allocate_query();
            if (reserve_rows > 0) reserve_impl(reserve_rows, false);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~CudaFlatIndex() { cleanup(); }

    void reserve(uint64_t rows) {
        std::lock_guard<std::mutex> guard(mutex_);
        set_device();
        reserve_impl(rows, false);
    }

    void add(const float *vectors, const uint64_t *ids,
             const uint64_t *group_ids, uint64_t count) {
        if (count == 0) return;
        if (!vectors || !ids || !group_ids) {
            throw std::invalid_argument("add received a null pointer");
        }
        std::lock_guard<std::mutex> guard(mutex_);
        set_device();

        std::unordered_set<uint64_t> batch_ids;
        batch_ids.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            if (ids[i] == UINT64_MAX) {
                throw std::invalid_argument(
                    "invalid vector ID: UINT64_MAX is reserved");
            }
            if (id_to_slot_.count(ids[i]) || !batch_ids.insert(ids[i]).second) {
                throw std::invalid_argument("duplicate external ID in add");
            }
        }
        if (count > std::numeric_limits<uint64_t>::max() - size_) {
            throw std::overflow_error("index row count overflow");
        }
        const uint64_t required = size_ + count;
        if (required > capacity_) {
            uint64_t grown = capacity_ ? capacity_ : 1;
            while (grown < required) {
                const uint64_t next = static_cast<uint64_t>(
                    std::ceil(static_cast<double>(grown) * growth_factor_));
                if (next <= grown) { grown = required; break; }
                grown = next;
            }
            reserve_impl(std::max(grown, required), true);
        }

        const size_t old_group_count = group_ids_.size();
        std::vector<uint64_t> new_group_ids;
        std::unordered_map<uint64_t, uint32_t> pending_group_slots;
        std::vector<uint32_t> row_group_slots(static_cast<size_t>(count));
        new_group_ids.reserve(static_cast<size_t>(count));
        pending_group_slots.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            const auto existing = group_to_slot_.find(group_ids[i]);
            if (existing != group_to_slot_.end()) {
                row_group_slots[static_cast<size_t>(i)] = existing->second;
                continue;
            }
            const auto pending = pending_group_slots.find(group_ids[i]);
            if (pending != pending_group_slots.end()) {
                row_group_slots[static_cast<size_t>(i)] = pending->second;
                continue;
            }
            const uint64_t next_slot = old_group_count + new_group_ids.size();
            if (next_slot >= static_cast<uint64_t>(UINT32_MAX)) {
                throw std::overflow_error("CUDA group slot count exceeds UINT32_MAX");
            }
            const uint32_t slot = static_cast<uint32_t>(next_slot);
            pending_group_slots.emplace(group_ids[i], slot);
            new_group_ids.push_back(group_ids[i]);
            row_group_slots[static_cast<size_t>(i)] = slot;
        }

        ids_.reserve(static_cast<size_t>(required));
        alive_.reserve(static_cast<size_t>(required));
        row_group_slots_.reserve(static_cast<size_t>(required));
        id_to_slot_.reserve(id_to_slot_.size() + static_cast<size_t>(count));
        group_ids_.reserve(old_group_count + new_group_ids.size());
        group_live_counts_.reserve(old_group_count + new_group_ids.size());
        group_to_slot_.reserve(group_to_slot_.size() + new_group_ids.size());

        try {
            for (const auto &entry : pending_group_slots) {
                group_to_slot_.emplace(entry.first, entry.second);
            }
            group_ids_.insert(group_ids_.end(), new_group_ids.begin(),
                              new_group_ids.end());
            group_live_counts_.insert(group_live_counts_.end(),
                                      new_group_ids.size(), UINT64_C(0));
            for (uint64_t i = 0; i < count; ++i) {
                id_to_slot_.emplace(ids[i], size_ + i);
            }
        } catch (...) {
            for (uint64_t i = 0; i < count; ++i) id_to_slot_.erase(ids[i]);
            for (const auto &entry : pending_group_slots) {
                group_to_slot_.erase(entry.first);
            }
            group_ids_.resize(old_group_count);
            group_live_counts_.resize(old_group_count);
            throw;
        }

        const auto rollback_metadata = [&] {
            for (uint64_t i = 0; i < count; ++i) id_to_slot_.erase(ids[i]);
            for (const auto &entry : pending_group_slots) {
                group_to_slot_.erase(entry.first);
            }
            group_ids_.resize(old_group_count);
            group_live_counts_.resize(old_group_count);
        };

        try {
            const size_t elements = checked_elements(count);
            char *destination = static_cast<char *>(data_) +
                                static_cast<size_t>(size_) * bytes_per_vector_;
            if (dtype_ == IFS_CUDA_FP32) {
                check_cuda(cudaMemcpy(destination, vectors,
                                      elements * sizeof(float),
                                      cudaMemcpyHostToDevice),
                           "copy FP32 database rows to GPU");
            } else {
                ensure_staging(elements);
                check_cuda(cudaMemcpy(staging_, vectors,
                                      elements * sizeof(float),
                                      cudaMemcpyHostToDevice),
                           "copy conversion staging rows to GPU");
                convert(staging_, destination, elements);
            }
            check_cuda(cudaMemcpy(device_ids_ + size_, ids,
                                  static_cast<size_t>(count) * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice),
                       "copy external IDs to GPU");
            check_cuda(cudaMemcpy(device_row_group_slots_ + size_,
                                  row_group_slots.data(),
                                  static_cast<size_t>(count) * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "copy row group slots to GPU");
            if (!new_group_ids.empty()) {
                check_cuda(cudaMemcpy(device_group_ids_ + old_group_count,
                                      new_group_ids.data(),
                                      new_group_ids.size() * sizeof(uint64_t),
                                      cudaMemcpyHostToDevice),
                           "copy stable group IDs to GPU");
            }
            check_cuda(cudaMemset(device_alive_ + size_, 1,
                                  static_cast<size_t>(count)),
                       "initialize live-row flags on GPU");
            check_cuda(cudaDeviceSynchronize(), "synchronize database add");
        } catch (...) {
            rollback_metadata();
            throw;
        }

        for (uint64_t i = 0; i < count; ++i) {
            ids_.push_back(ids[i]);
            alive_.push_back(1);
            const uint32_t group_slot = row_group_slots[static_cast<size_t>(i)];
            row_group_slots_.push_back(group_slot);
            if (group_live_counts_[group_slot]++ == 0) ++active_group_count_;
        }
        size_ = required;
        live_size_ += count;
    }

    uint64_t remove(const uint64_t *ids, uint64_t count) {
        if (count && !ids) throw std::invalid_argument("remove received a null pointer");
        std::lock_guard<std::mutex> guard(mutex_);
        set_device();
        std::vector<uint64_t> removed_slots;
        removed_slots.reserve(static_cast<size_t>(count));
        std::unordered_set<uint64_t> scheduled_slots;
        scheduled_slots.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            const auto found = id_to_slot_.find(ids[i]);
            if (found == id_to_slot_.end()) continue;
            const uint64_t slot = found->second;
            if (alive_[static_cast<size_t>(slot)] &&
                scheduled_slots.insert(slot).second) {
                const uint32_t group_slot =
                    row_group_slots_[static_cast<size_t>(slot)];
                if (group_live_counts_[group_slot] == 0) {
                    throw std::logic_error("CUDA group live count underflow");
                }
                removed_slots.push_back(slot);
            }
        }
        if (!removed_slots.empty()) {
            check_cuda(cudaMemcpy(device_delete_slots_, removed_slots.data(),
                                  removed_slots.size() * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice),
                       "copy tombstone update slots");
            constexpr int threads = 256;
            const int blocks = static_cast<int>(
                (removed_slots.size() + threads - 1) / threads);
            mark_dead<<<blocks, threads>>>(device_alive_, device_delete_slots_,
                                           removed_slots.size());
            check_cuda(cudaGetLastError(), "launch tombstone update kernel");
            check_cuda(cudaDeviceSynchronize(),
                       "synchronize tombstone update kernel");

            for (const uint64_t slot : removed_slots) {
                alive_[static_cast<size_t>(slot)] = 0;
                id_to_slot_.erase(ids_[static_cast<size_t>(slot)]);
                const uint32_t group_slot =
                    row_group_slots_[static_cast<size_t>(slot)];
                if (--group_live_counts_[group_slot] == 0) {
                    --active_group_count_;
                }
            }
            live_size_ -= static_cast<uint64_t>(removed_slots.size());
        }
        return static_cast<uint64_t>(removed_slots.size());
    }

    void search(const float *query, uint64_t k, int topk_mode,
                uint64_t *out_ids,
                float *out_scores, uint64_t *out_count, double *kernel_ms,
                double *topk_ms, double *total_ms) {
        if (!query || !out_count) throw std::invalid_argument("search received a null pointer");
        if (k && (!out_ids || !out_scores)) {
            throw std::invalid_argument("search output buffers are null");
        }
        if (topk_mode != IFS_CUDA_TOPK_HOST &&
            topk_mode != IFS_CUDA_TOPK_DEVICE) {
            throw std::invalid_argument("unsupported CUDA Top-K mode");
        }
        if (topk_mode == IFS_CUDA_TOPK_DEVICE && k > kDeviceTopKLimit) {
            throw std::invalid_argument("device Top-K supports k <= 100");
        }
        std::lock_guard<std::mutex> guard(mutex_);
        set_device();
        const auto total_start = std::chrono::steady_clock::now();
        if (k == 0 || live_size_ == 0) {
            *out_count = 0;
            if (kernel_ms) *kernel_ms = 0.0;
            if (topk_ms) *topk_ms = 0.0;
            if (total_ms) *total_ms = 0.0;
            return;
        }
        compute_scores(query, kernel_ms);

        if (topk_mode == IFS_CUDA_TOPK_DEVICE) {
            if (dtype_ == IFS_CUDA_INT8) {
                select_device_topk<int32_t>(k, out_ids, out_scores, out_count,
                                            topk_ms);
            } else {
                select_device_topk<float>(k, out_ids, out_scores, out_count,
                                          topk_ms);
            }
        } else {
            const auto selection_start = std::chrono::steady_clock::now();
            select_host_topk(k, out_ids, out_scores, out_count);
            if (topk_ms) {
                const auto selection_end = std::chrono::steady_clock::now();
                *topk_ms = std::chrono::duration<double, std::milli>(
                    selection_end - selection_start).count();
            }
        }
        if (total_ms) {
            const auto total_end = std::chrono::steady_clock::now();
            *total_ms = std::chrono::duration<double, std::milli>(
                            total_end - total_start).count();
        }
    }

    void grouped_search(const float *query, uint64_t k,
                        uint64_t *out_group_ids, uint64_t *out_vector_ids,
                        float *out_scores, uint64_t *out_count,
                        double *kernel_ms, double *topk_ms,
                        double *total_ms) {
        if (!query || !out_count) {
            throw std::invalid_argument("grouped search received a null pointer");
        }
        if (k && (!out_group_ids || !out_vector_ids || !out_scores)) {
            throw std::invalid_argument("grouped search output buffers are null");
        }
        if (k > kDeviceTopKLimit) {
            throw std::invalid_argument("grouped device Top-K supports k <= 100");
        }
        std::lock_guard<std::mutex> guard(mutex_);
        set_device();
        const auto total_start = std::chrono::steady_clock::now();
        if (k == 0 || live_size_ == 0 || active_group_count_ == 0) {
            *out_count = 0;
            if (kernel_ms) *kernel_ms = 0.0;
            if (topk_ms) *topk_ms = 0.0;
            if (total_ms) *total_ms = 0.0;
            return;
        }
        compute_scores(query, kernel_ms);
        if (dtype_ == IFS_CUDA_INT8) {
            select_device_grouped_topk<int32_t>(
                k, out_group_ids, out_vector_ids, out_scores, out_count,
                topk_ms);
        } else {
            select_device_grouped_topk<float>(
                k, out_group_ids, out_vector_ids, out_scores, out_count,
                topk_ms);
        }
        if (total_ms) {
            const auto total_end = std::chrono::steady_clock::now();
            *total_ms = std::chrono::duration<double, std::milli>(
                            total_end - total_start).count();
        }
    }

    ifs_cuda_stats stats() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return ifs_cuda_stats{size_, live_size_, capacity_, reallocations_,
                              bytes_per_vector_, device_, dtype_};
    }

private:
    void set_device() const { check_cuda(cudaSetDevice(device_), "cudaSetDevice"); }

    size_t checked_elements(uint64_t rows) const {
        if (rows > std::numeric_limits<size_t>::max() / kDimension) {
            throw std::overflow_error("vector element count overflow");
        }
        return static_cast<size_t>(rows) * kDimension;
    }

    void compute_scores(const float *query, double *kernel_ms) {
        if (size_ > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "cuBLAS flat search supports at most INT_MAX rows");
        }
        upload_query(query);
        ensure_scores(size_);

        cudaEvent_t begin = nullptr;
        cudaEvent_t end = nullptr;
        check_cuda(cudaEventCreate(&begin), "create CUDA timing event");
        try {
            check_cuda(cudaEventCreate(&end), "create CUDA timing event");
            check_cuda(cudaEventRecord(begin), "record CUDA timing start");
            run_gemm();
            check_cuda(cudaEventRecord(end), "record CUDA timing end");
            check_cuda(cudaEventSynchronize(end),
                       "synchronize CUDA dot product");
            float elapsed = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed, begin, end),
                       "measure CUDA dot product");
            if (kernel_ms) *kernel_ms = static_cast<double>(elapsed);
        } catch (...) {
            if (begin) cudaEventDestroy(begin);
            if (end) cudaEventDestroy(end);
            throw;
        }
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }

    void reserve_impl(uint64_t rows, bool automatic) {
        if (rows <= capacity_) return;
        if (rows > std::numeric_limits<size_t>::max() / bytes_per_vector_) {
            throw std::overflow_error("GPU index allocation size overflow");
        }
        if (rows > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
            throw std::overflow_error("GPU index metadata allocation overflow");
        }
        void *next = nullptr;
        uint64_t *next_ids = nullptr;
        uint8_t *next_alive = nullptr;
        uint32_t *next_row_group_slots = nullptr;
        uint64_t *next_group_ids = nullptr;
        uint32_t *next_group_best_keys = nullptr;
        uint64_t *next_group_best_vector_ids = nullptr;
        uint64_t *next_delete_slots = nullptr;
        void *next_scores = nullptr;
        void *next_topk_first = nullptr;
        void *next_topk_second = nullptr;
        void *next_group_topk_first = nullptr;
        void *next_group_topk_second = nullptr;
        const uint64_t topk_blocks = rows / kTopKChunkRows +
            (rows % kTopKChunkRows != 0 ? 1 : 0);
        if (topk_blocks > std::numeric_limits<uint64_t>::max() /
                              kDeviceTopKLimit) {
            throw std::overflow_error("GPU Top-K reserve size overflow");
        }
        const uint64_t topk_entries = topk_blocks * kDeviceTopKLimit;
        if (topk_entries > std::numeric_limits<size_t>::max() /
                               sizeof(DeviceGroupCandidate<float>)) {
            throw std::overflow_error("GPU Top-K reserve byte size overflow");
        }
        check_cuda(cudaMalloc(&next, static_cast<size_t>(rows) * bytes_per_vector_),
                   "allocate GPU index storage");
        try {
            check_cuda(cudaMalloc(&next_ids,
                                  static_cast<size_t>(rows) * sizeof(uint64_t)),
                       "allocate GPU external IDs");
            check_cuda(cudaMalloc(&next_alive, static_cast<size_t>(rows)),
                       "allocate GPU live-row flags");
            check_cuda(cudaMalloc(&next_row_group_slots,
                                  static_cast<size_t>(rows) * sizeof(uint32_t)),
                       "allocate GPU row group slots");
            check_cuda(cudaMalloc(&next_group_ids,
                                  static_cast<size_t>(rows) * sizeof(uint64_t)),
                       "allocate GPU stable group IDs");
            check_cuda(cudaMalloc(&next_group_best_keys,
                                  static_cast<size_t>(rows) * sizeof(uint32_t)),
                       "allocate GPU group score reduction storage");
            check_cuda(cudaMalloc(&next_group_best_vector_ids,
                                  static_cast<size_t>(rows) * sizeof(uint64_t)),
                       "allocate GPU group vector reduction storage");
            check_cuda(cudaMalloc(&next_delete_slots,
                                  static_cast<size_t>(rows) * sizeof(uint64_t)),
                       "allocate GPU tombstone update slots");
            const size_t score_bytes = dtype_ == IFS_CUDA_INT8
                ? sizeof(int32_t) : sizeof(float);
            check_cuda(cudaMalloc(&next_scores,
                                  static_cast<size_t>(rows) * score_bytes),
                       "allocate GPU score workspace");
            const size_t face_topk_bytes = static_cast<size_t>(topk_entries) *
                                           sizeof(DeviceCandidate<float>);
            const size_t group_topk_bytes = static_cast<size_t>(topk_entries) *
                                            sizeof(DeviceGroupCandidate<float>);
            check_cuda(cudaMalloc(&next_topk_first, face_topk_bytes),
                       "allocate first reserved GPU Top-K workspace");
            check_cuda(cudaMalloc(&next_topk_second, face_topk_bytes),
                       "allocate second reserved GPU Top-K workspace");
            check_cuda(cudaMalloc(&next_group_topk_first, group_topk_bytes),
                       "allocate first reserved GPU grouped Top-K workspace");
            check_cuda(cudaMalloc(&next_group_topk_second, group_topk_bytes),
                       "allocate second reserved GPU grouped Top-K workspace");
            if (data_ && size_ > 0) {
                check_cuda(cudaMemcpy(next, data_,
                                      static_cast<size_t>(size_) * bytes_per_vector_,
                                      cudaMemcpyDeviceToDevice),
                           "copy GPU index during growth");
                check_cuda(cudaMemcpy(next_ids, device_ids_,
                                      static_cast<size_t>(size_) * sizeof(uint64_t),
                                      cudaMemcpyDeviceToDevice),
                           "copy GPU external IDs during growth");
                check_cuda(cudaMemcpy(next_alive, device_alive_,
                                      static_cast<size_t>(size_),
                                      cudaMemcpyDeviceToDevice),
                           "copy GPU live-row flags during growth");
                check_cuda(cudaMemcpy(
                               next_row_group_slots, device_row_group_slots_,
                               static_cast<size_t>(size_) * sizeof(uint32_t),
                               cudaMemcpyDeviceToDevice),
                           "copy GPU row group slots during growth");
            }
            if (device_group_ids_ && !group_ids_.empty()) {
                check_cuda(cudaMemcpy(
                               next_group_ids, device_group_ids_,
                               group_ids_.size() * sizeof(uint64_t),
                               cudaMemcpyDeviceToDevice),
                           "copy GPU stable group IDs during growth");
            }
        } catch (...) {
            cudaFree(next);
            if (next_ids) cudaFree(next_ids);
            if (next_alive) cudaFree(next_alive);
            if (next_row_group_slots) cudaFree(next_row_group_slots);
            if (next_group_ids) cudaFree(next_group_ids);
            if (next_group_best_keys) cudaFree(next_group_best_keys);
            if (next_group_best_vector_ids) {
                cudaFree(next_group_best_vector_ids);
            }
            if (next_delete_slots) cudaFree(next_delete_slots);
            if (next_scores) cudaFree(next_scores);
            if (next_topk_first) cudaFree(next_topk_first);
            if (next_topk_second) cudaFree(next_topk_second);
            if (next_group_topk_first) cudaFree(next_group_topk_first);
            if (next_group_topk_second) cudaFree(next_group_topk_second);
            throw;
        }
        if (data_) check_cuda(cudaFree(data_), "free previous GPU index storage");
        if (device_ids_) {
            check_cuda(cudaFree(device_ids_), "free previous GPU external IDs");
        }
        if (device_alive_) {
            check_cuda(cudaFree(device_alive_), "free previous GPU live-row flags");
        }
        if (device_row_group_slots_) {
            check_cuda(cudaFree(device_row_group_slots_),
                       "free previous GPU row group slots");
        }
        if (device_group_ids_) {
            check_cuda(cudaFree(device_group_ids_),
                       "free previous GPU stable group IDs");
        }
        if (group_best_keys_) {
            check_cuda(cudaFree(group_best_keys_),
                       "free previous GPU group score reduction storage");
        }
        if (group_best_vector_ids_) {
            check_cuda(cudaFree(group_best_vector_ids_),
                       "free previous GPU group vector reduction storage");
        }
        if (device_delete_slots_) {
            check_cuda(cudaFree(device_delete_slots_),
                       "free previous GPU tombstone update slots");
        }
        if (scores_) {
            check_cuda(cudaFree(scores_), "free previous GPU score workspace");
        }
        if (topk_first_) {
            check_cuda(cudaFree(topk_first_),
                       "free first previous GPU Top-K workspace");
        }
        if (topk_second_) {
            check_cuda(cudaFree(topk_second_),
                       "free second previous GPU Top-K workspace");
        }
        if (group_topk_first_) {
            check_cuda(cudaFree(group_topk_first_),
                       "free first previous GPU grouped Top-K workspace");
        }
        if (group_topk_second_) {
            check_cuda(cudaFree(group_topk_second_),
                       "free second previous GPU grouped Top-K workspace");
        }
        data_ = next;
        device_ids_ = next_ids;
        device_alive_ = next_alive;
        device_row_group_slots_ = next_row_group_slots;
        device_group_ids_ = next_group_ids;
        group_best_keys_ = next_group_best_keys;
        group_best_vector_ids_ = next_group_best_vector_ids;
        device_delete_slots_ = next_delete_slots;
        scores_ = next_scores;
        scores_capacity_ = rows;
        topk_first_ = next_topk_first;
        topk_second_ = next_topk_second;
        topk_capacity_entries_ = topk_entries;
        group_topk_first_ = next_group_topk_first;
        group_topk_second_ = next_group_topk_second;
        group_topk_capacity_entries_ = topk_entries;
        capacity_ = rows;
        if (automatic) ++reallocations_;
    }

    void allocate_query() {
        check_cuda(cudaMalloc(&query_, static_cast<size_t>(kDimension) * element_bytes(dtype_)),
                   "allocate GPU query");
    }

    void ensure_staging(size_t elements) {
        if (elements <= staging_elements_) return;
        if (staging_) check_cuda(cudaFree(staging_), "free old conversion staging");
        check_cuda(cudaMalloc(&staging_, elements * sizeof(float)),
                   "allocate conversion staging");
        staging_elements_ = elements;
    }

    void ensure_scores(uint64_t rows) {
        if (rows <= scores_capacity_) return;
        if (scores_) check_cuda(cudaFree(scores_), "free old score buffer");
        const size_t score_bytes = dtype_ == IFS_CUDA_INT8 ? sizeof(int32_t) : sizeof(float);
        check_cuda(cudaMalloc(&scores_, static_cast<size_t>(rows) * score_bytes),
                   "allocate GPU score buffer");
        scores_capacity_ = rows;
    }

    void convert(const float *input, void *output, size_t elements) {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((elements + threads - 1) / threads);
        switch (dtype_) {
            case IFS_CUDA_FP16:
                float_to_half<<<blocks, threads>>>(input, static_cast<__half *>(output), elements);
                break;
            case IFS_CUDA_BF16:
                float_to_bfloat16<<<blocks, threads>>>(
                    input, static_cast<__nv_bfloat16 *>(output), elements);
                break;
            case IFS_CUDA_INT8:
                float_to_int8<<<blocks, threads>>>(
                    input, static_cast<int8_t *>(output), elements,
                    static_cast<float>(int8_scale_));
                break;
            default:
                throw std::logic_error("conversion requested for FP32 index");
        }
        check_cuda(cudaGetLastError(), "launch storage conversion kernel");
    }

    void upload_query(const float *query) {
        if (dtype_ == IFS_CUDA_FP32) {
            check_cuda(cudaMemcpy(query_, query, kDimension * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       "copy FP32 query to GPU");
            return;
        }
        ensure_staging(kDimension);
        check_cuda(cudaMemcpy(staging_, query, kDimension * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "copy query conversion staging to GPU");
        convert(staging_, query_, kDimension);
    }

    void run_gemm() {
        const int rows = static_cast<int>(size_);
        const int dimension = kDimension;
        if (dtype_ == IFS_CUDA_INT8) {
            const int32_t alpha = 1;
            const int32_t beta = 0;
            check_cublas(cublasGemmEx(
                cublas_, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, dimension,
                &alpha, data_, CUDA_R_8I, dimension, query_, CUDA_R_8I, dimension,
                &beta, scores_, CUDA_R_32I, rows, CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "INT8 exact flat inner-product GEMM");
            return;
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudaDataType_t storage_type = CUDA_R_32F;
        if (dtype_ == IFS_CUDA_FP16) storage_type = CUDA_R_16F;
        if (dtype_ == IFS_CUDA_BF16) storage_type = CUDA_R_16BF;
        check_cublas(cublasGemmEx(
            cublas_, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, dimension,
            &alpha, data_, storage_type, dimension, query_, storage_type, dimension,
            &beta, scores_, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "floating-point exact flat inner-product GEMM");
    }

    void select_host_topk(uint64_t k, uint64_t *out_ids,
                          float *out_scores, uint64_t *out_count) {
        const size_t row_count = static_cast<size_t>(size_);
        std::priority_queue<Candidate, std::vector<Candidate>, BetterComparator> heap;
        if (dtype_ == IFS_CUDA_INT8) {
            host_int_scores_.resize(row_count);
            check_cuda(cudaMemcpy(host_int_scores_.data(), scores_,
                                  row_count * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost),
                       "copy INT8 dot scores to host");
            for (size_t slot = 0; slot < row_count; ++slot) {
                if (!alive_[slot]) continue;
                consider(heap, k, Candidate{
                    static_cast<float>(host_int_scores_[slot]) /
                        int8_score_divisor_,
                    ids_[slot]});
            }
        } else {
            host_float_scores_.resize(row_count);
            check_cuda(cudaMemcpy(host_float_scores_.data(), scores_,
                                  row_count * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "copy floating-point dot scores to host");
            for (size_t slot = 0; slot < row_count; ++slot) {
                if (!alive_[slot]) continue;
                consider(heap, k,
                         Candidate{host_float_scores_[slot], ids_[slot]});
            }
        }
        std::vector<Candidate> ordered;
        ordered.reserve(heap.size());
        while (!heap.empty()) {
            ordered.push_back(heap.top());
            heap.pop();
        }
        std::sort(ordered.begin(), ordered.end(), better);
        for (size_t i = 0; i < ordered.size(); ++i) {
            out_ids[i] = ordered[i].id;
            out_scores[i] = ordered[i].score;
        }
        *out_count = static_cast<uint64_t>(ordered.size());
    }

    void ensure_topk_buffers(uint64_t entries) {
        if (entries <= topk_capacity_entries_) return;
        if (entries > std::numeric_limits<size_t>::max() /
                          sizeof(DeviceCandidate<float>)) {
            throw std::overflow_error("GPU Top-K candidate allocation overflow");
        }
        void *first = nullptr;
        void *second = nullptr;
        const size_t bytes = static_cast<size_t>(entries) *
                             sizeof(DeviceCandidate<float>);
        check_cuda(cudaMalloc(&first, bytes),
                   "allocate first GPU Top-K candidate buffer");
        try {
            check_cuda(cudaMalloc(&second, bytes),
                       "allocate second GPU Top-K candidate buffer");
        } catch (...) {
            cudaFree(first);
            throw;
        }
        if (topk_first_) check_cuda(cudaFree(topk_first_),
                                    "free first GPU Top-K candidate buffer");
        if (topk_second_) check_cuda(cudaFree(topk_second_),
                                     "free second GPU Top-K candidate buffer");
        topk_first_ = first;
        topk_second_ = second;
        topk_capacity_entries_ = entries;
    }

    template <typename Score>
    void select_device_topk(uint64_t k, uint64_t *out_ids,
                            float *out_scores, uint64_t *out_count,
                            double *topk_ms) {
        static_assert(sizeof(DeviceCandidate<Score>) ==
                      sizeof(DeviceCandidate<float>),
                      "all device candidate layouts must have equal size");
        const uint64_t first_blocks =
            (size_ + kTopKChunkRows - 1) / kTopKChunkRows;
        if (first_blocks > std::numeric_limits<unsigned int>::max()) {
            throw std::overflow_error("GPU Top-K grid dimension overflow");
        }
        if (first_blocks > std::numeric_limits<uint64_t>::max() / k) {
            throw std::overflow_error("GPU Top-K candidate count overflow");
        }
        ensure_topk_buffers(first_blocks * k);
        auto *first = static_cast<DeviceCandidate<Score> *>(topk_first_);
        auto *second = static_cast<DeviceCandidate<Score> *>(topk_second_);

        cudaEvent_t begin = nullptr;
        cudaEvent_t end = nullptr;
        check_cuda(cudaEventCreate(&begin), "create GPU Top-K timing event");
        try {
            check_cuda(cudaEventCreate(&end), "create GPU Top-K timing event");
            check_cuda(cudaEventRecord(begin), "record GPU Top-K start");
            topk_from_scores<Score><<<static_cast<unsigned int>(first_blocks),
                                      kTopKBlockSize>>>(
                static_cast<const Score *>(scores_), device_ids_, device_alive_,
                size_, k, first);
            check_cuda(cudaGetLastError(), "launch first GPU Top-K stage");

            uint64_t candidate_count = first_blocks * k;
            DeviceCandidate<Score> *current = first;
            DeviceCandidate<Score> *next = second;
            while (candidate_count > k) {
                const uint64_t blocks =
                    (candidate_count + kTopKChunkRows - 1) /
                    kTopKChunkRows;
                if (blocks > std::numeric_limits<unsigned int>::max()) {
                    throw std::overflow_error("GPU Top-K reduction grid overflow");
                }
                topk_from_candidates<Score><<<static_cast<unsigned int>(blocks),
                                              kTopKBlockSize>>>(
                    current, candidate_count, k, next);
                check_cuda(cudaGetLastError(),
                           "launch GPU Top-K reduction stage");
                candidate_count = blocks * k;
                std::swap(current, next);
            }
            check_cuda(cudaEventRecord(end), "record GPU Top-K end");
            check_cuda(cudaEventSynchronize(end),
                       "synchronize GPU Top-K selection");
            float elapsed = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed, begin, end),
                       "measure GPU Top-K selection");
            if (topk_ms) *topk_ms = static_cast<double>(elapsed);

            const uint64_t result_count = std::min(k, live_size_);
            std::vector<DeviceCandidate<Score>> selected(
                static_cast<size_t>(result_count));
            check_cuda(cudaMemcpy(selected.data(), current,
                                  static_cast<size_t>(result_count) *
                                      sizeof(DeviceCandidate<Score>),
                                  cudaMemcpyDeviceToHost),
                       "copy final GPU Top-K candidates to host");
            for (size_t i = 0; i < selected.size(); ++i) {
                out_ids[i] = selected[i].id;
                if constexpr (std::is_same<Score, int32_t>::value) {
                    out_scores[i] =
                        static_cast<float>(selected[i].score) /
                        int8_score_divisor_;
                } else {
                    out_scores[i] = selected[i].score;
                }
            }
            *out_count = result_count;
        } catch (...) {
            if (begin) cudaEventDestroy(begin);
            if (end) cudaEventDestroy(end);
            throw;
        }
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }

    void ensure_group_topk_buffers(uint64_t entries) {
        if (entries <= group_topk_capacity_entries_) return;
        if (entries > std::numeric_limits<size_t>::max() /
                          sizeof(DeviceGroupCandidate<float>)) {
            throw std::overflow_error(
                "GPU grouped Top-K candidate allocation overflow");
        }
        void *first = nullptr;
        void *second = nullptr;
        const size_t bytes = static_cast<size_t>(entries) *
                             sizeof(DeviceGroupCandidate<float>);
        check_cuda(cudaMalloc(&first, bytes),
                   "allocate first GPU grouped Top-K candidate buffer");
        try {
            check_cuda(cudaMalloc(&second, bytes),
                       "allocate second GPU grouped Top-K candidate buffer");
        } catch (...) {
            cudaFree(first);
            throw;
        }
        if (group_topk_first_) {
            check_cuda(cudaFree(group_topk_first_),
                       "free first GPU grouped Top-K candidate buffer");
        }
        if (group_topk_second_) {
            check_cuda(cudaFree(group_topk_second_),
                       "free second GPU grouped Top-K candidate buffer");
        }
        group_topk_first_ = first;
        group_topk_second_ = second;
        group_topk_capacity_entries_ = entries;
    }

    template <typename Score>
    void select_device_grouped_topk(
        uint64_t k, uint64_t *out_group_ids, uint64_t *out_vector_ids,
        float *out_scores, uint64_t *out_count, double *topk_ms) {
        static_assert(sizeof(DeviceGroupCandidate<Score>) ==
                          sizeof(DeviceGroupCandidate<float>),
                      "all grouped candidate layouts must have equal size");
        const uint64_t group_count = static_cast<uint64_t>(group_ids_.size());
        const uint64_t first_blocks =
            (group_count + kTopKChunkRows - 1) / kTopKChunkRows;
        if (first_blocks > std::numeric_limits<unsigned int>::max()) {
            throw std::overflow_error(
                "GPU grouped Top-K grid dimension overflow");
        }
        if (first_blocks > std::numeric_limits<uint64_t>::max() / k) {
            throw std::overflow_error(
                "GPU grouped Top-K candidate count overflow");
        }
        ensure_group_topk_buffers(first_blocks * k);
        auto *first = static_cast<DeviceGroupCandidate<Score> *>(
            group_topk_first_);
        auto *second = static_cast<DeviceGroupCandidate<Score> *>(
            group_topk_second_);

        cudaEvent_t begin = nullptr;
        cudaEvent_t end = nullptr;
        check_cuda(cudaEventCreate(&begin),
                   "create GPU grouped Top-K timing event");
        try {
            check_cuda(cudaEventCreate(&end),
                       "create GPU grouped Top-K timing event");
            check_cuda(cudaEventRecord(begin),
                       "record GPU grouped Top-K start");
            check_cuda(cudaMemset(group_best_keys_, 0,
                                  static_cast<size_t>(group_count) *
                                      sizeof(uint32_t)),
                       "reset GPU group best scores");
            check_cuda(cudaMemset(group_best_vector_ids_, 0xff,
                                  static_cast<size_t>(group_count) *
                                      sizeof(uint64_t)),
                       "reset GPU group best vector IDs");

            constexpr int reduction_threads = 256;
            const uint64_t reduction_blocks =
                (size_ + reduction_threads - 1) / reduction_threads;
            if (reduction_blocks > std::numeric_limits<unsigned int>::max()) {
                throw std::overflow_error(
                    "GPU group reduction grid dimension overflow");
            }
            reduce_group_best_scores<Score>
                <<<static_cast<unsigned int>(reduction_blocks),
                   reduction_threads>>>(
                    static_cast<const Score *>(scores_),
                    device_row_group_slots_, device_alive_, size_,
                    group_best_keys_);
            check_cuda(cudaGetLastError(),
                       "launch GPU group score reduction");
            reduce_group_best_vector_ids<Score>
                <<<static_cast<unsigned int>(reduction_blocks),
                   reduction_threads>>>(
                    static_cast<const Score *>(scores_), device_ids_,
                    device_row_group_slots_, device_alive_, size_,
                    group_best_keys_, group_best_vector_ids_);
            check_cuda(cudaGetLastError(),
                       "launch GPU group vector reduction");

            grouped_topk_from_best<Score>
                <<<static_cast<unsigned int>(first_blocks),
                   kTopKBlockSize>>>(
                    group_best_keys_, group_best_vector_ids_,
                    device_group_ids_, group_count, k, first);
            check_cuda(cudaGetLastError(),
                       "launch first GPU grouped Top-K stage");

            uint64_t candidate_count = first_blocks * k;
            DeviceGroupCandidate<Score> *current = first;
            DeviceGroupCandidate<Score> *next = second;
            while (candidate_count > k) {
                const uint64_t blocks =
                    (candidate_count + kTopKChunkRows - 1) /
                    kTopKChunkRows;
                if (blocks > std::numeric_limits<unsigned int>::max()) {
                    throw std::overflow_error(
                        "GPU grouped Top-K reduction grid overflow");
                }
                grouped_topk_from_candidates<Score>
                    <<<static_cast<unsigned int>(blocks),
                       kTopKBlockSize>>>(current, candidate_count, k, next);
                check_cuda(cudaGetLastError(),
                           "launch GPU grouped Top-K reduction stage");
                candidate_count = blocks * k;
                std::swap(current, next);
            }
            check_cuda(cudaEventRecord(end),
                       "record GPU grouped Top-K end");
            check_cuda(cudaEventSynchronize(end),
                       "synchronize GPU grouped Top-K selection");
            float elapsed = 0.0f;
            check_cuda(cudaEventElapsedTime(&elapsed, begin, end),
                       "measure GPU grouped Top-K selection");
            if (topk_ms) *topk_ms = static_cast<double>(elapsed);

            const uint64_t result_count = std::min(k, active_group_count_);
            std::vector<DeviceGroupCandidate<Score>> selected(
                static_cast<size_t>(result_count));
            check_cuda(cudaMemcpy(
                           selected.data(), current,
                           static_cast<size_t>(result_count) *
                               sizeof(DeviceGroupCandidate<Score>),
                           cudaMemcpyDeviceToHost),
                       "copy final GPU grouped Top-K candidates to host");
            for (size_t i = 0; i < selected.size(); ++i) {
                out_group_ids[i] = selected[i].group_id;
                out_vector_ids[i] = selected[i].vector_id;
                if constexpr (std::is_same<Score, int32_t>::value) {
                    out_scores[i] =
                        static_cast<float>(selected[i].score) /
                        int8_score_divisor_;
                } else {
                    out_scores[i] = selected[i].score;
                }
            }
            *out_count = result_count;
        } catch (...) {
            if (begin) cudaEventDestroy(begin);
            if (end) cudaEventDestroy(end);
            throw;
        }
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }

    static void consider(
        std::priority_queue<Candidate, std::vector<Candidate>, BetterComparator> &heap,
        uint64_t k, const Candidate &candidate) {
        if (heap.size() < static_cast<size_t>(k)) {
            heap.push(candidate);
        } else if (better(candidate, heap.top())) {
            heap.pop();
            heap.push(candidate);
        }
    }

    void cleanup() noexcept {
        if (device_ >= 0) cudaSetDevice(device_);
        if (group_topk_second_) cudaFree(group_topk_second_);
        if (group_topk_first_) cudaFree(group_topk_first_);
        if (topk_second_) cudaFree(topk_second_);
        if (topk_first_) cudaFree(topk_first_);
        if (group_best_vector_ids_) cudaFree(group_best_vector_ids_);
        if (group_best_keys_) cudaFree(group_best_keys_);
        if (device_delete_slots_) cudaFree(device_delete_slots_);
        if (scores_) cudaFree(scores_);
        if (staging_) cudaFree(staging_);
        if (query_) cudaFree(query_);
        if (device_group_ids_) cudaFree(device_group_ids_);
        if (device_row_group_slots_) cudaFree(device_row_group_slots_);
        if (device_alive_) cudaFree(device_alive_);
        if (device_ids_) cudaFree(device_ids_);
        if (data_) cudaFree(data_);
        if (cublas_) cublasDestroy(cublas_);
        scores_ = nullptr;
        staging_ = nullptr;
        query_ = nullptr;
        data_ = nullptr;
        device_ids_ = nullptr;
        device_alive_ = nullptr;
        device_row_group_slots_ = nullptr;
        device_group_ids_ = nullptr;
        group_best_keys_ = nullptr;
        group_best_vector_ids_ = nullptr;
        device_delete_slots_ = nullptr;
        topk_first_ = nullptr;
        topk_second_ = nullptr;
        group_topk_first_ = nullptr;
        group_topk_second_ = nullptr;
        cublas_ = nullptr;
    }

    int dtype_;
    int device_;
    double growth_factor_;
    uint32_t int8_scale_;
    float int8_score_divisor_;
    uint64_t bytes_per_vector_;
    uint64_t size_ = 0;
    uint64_t live_size_ = 0;
    uint64_t capacity_ = 0;
    uint64_t reallocations_ = 0;
    void *data_ = nullptr;
    uint64_t *device_ids_ = nullptr;
    uint8_t *device_alive_ = nullptr;
    uint32_t *device_row_group_slots_ = nullptr;
    uint64_t *device_group_ids_ = nullptr;
    uint32_t *group_best_keys_ = nullptr;
    uint64_t *group_best_vector_ids_ = nullptr;
    uint64_t *device_delete_slots_ = nullptr;
    void *query_ = nullptr;
    float *staging_ = nullptr;
    size_t staging_elements_ = 0;
    void *scores_ = nullptr;
    uint64_t scores_capacity_ = 0;
    void *topk_first_ = nullptr;
    void *topk_second_ = nullptr;
    uint64_t topk_capacity_entries_ = 0;
    void *group_topk_first_ = nullptr;
    void *group_topk_second_ = nullptr;
    uint64_t group_topk_capacity_entries_ = 0;
    cublasHandle_t cublas_ = nullptr;
    std::vector<uint64_t> ids_;
    std::vector<uint8_t> alive_;
    std::unordered_map<uint64_t, uint64_t> id_to_slot_;
    std::vector<uint32_t> row_group_slots_;
    std::unordered_map<uint64_t, uint32_t> group_to_slot_;
    std::vector<uint64_t> group_ids_;
    std::vector<uint64_t> group_live_counts_;
    uint64_t active_group_count_ = 0;
    std::vector<float> host_float_scores_;
    std::vector<int32_t> host_int_scores_;
    mutable std::mutex mutex_;
};

template <typename Function>
int protect(Function &&function) {
    try {
        function();
        g_last_error.clear();
        return 0;
    } catch (const std::exception &error) {
        g_last_error = error.what();
        return -1;
    } catch (...) {
        g_last_error = "unknown C++ exception";
        return -1;
    }
}

CudaFlatIndex &index_from(void *handle) {
    if (!handle) throw std::invalid_argument("index handle is null");
    return *static_cast<CudaFlatIndex *>(handle);
}

}  // namespace

extern "C" {

void *ifs_cuda_create(int dtype, uint64_t reserve_rows, int device,
                      double growth_factor, uint32_t int8_scale) {
    try {
        std::unique_ptr<CudaFlatIndex> index(
            new CudaFlatIndex(
                dtype, reserve_rows, device, growth_factor, int8_scale));
        g_last_error.clear();
        return index.release();
    } catch (const std::exception &error) {
        g_last_error = error.what();
        return nullptr;
    } catch (...) {
        g_last_error = "unknown C++ exception";
        return nullptr;
    }
}

void ifs_cuda_destroy(void *handle) {
    delete static_cast<CudaFlatIndex *>(handle);
}

int ifs_cuda_reserve(void *handle, uint64_t rows) {
    return protect([&] { index_from(handle).reserve(rows); });
}

int ifs_cuda_add(void *handle, const float *vectors, const uint64_t *ids,
                 const uint64_t *group_ids, uint64_t count) {
    return protect(
        [&] { index_from(handle).add(vectors, ids, group_ids, count); });
}

int ifs_cuda_remove(void *handle, const uint64_t *ids, uint64_t count,
                    uint64_t *removed) {
    return protect([&] {
        if (!removed) throw std::invalid_argument("removed output pointer is null");
        *removed = index_from(handle).remove(ids, count);
    });
}

int ifs_cuda_search(void *handle, const float *query, uint64_t k,
                    uint64_t *out_ids, float *out_scores,
                    uint64_t *out_count, double *kernel_ms,
                    double *total_ms) {
    return protect([&] {
        double ignored_topk_ms = 0.0;
        index_from(handle).search(query, k, IFS_CUDA_TOPK_HOST, out_ids,
                                  out_scores, out_count, kernel_ms,
                                  &ignored_topk_ms, total_ms);
    });
}

int ifs_cuda_search_ex(void *handle, const float *query, uint64_t k,
                       int topk_mode, uint64_t *out_ids,
                       float *out_scores, uint64_t *out_count,
                       double *kernel_ms, double *topk_ms,
                       double *total_ms) {
    return protect([&] {
        index_from(handle).search(query, k, topk_mode, out_ids, out_scores,
                                  out_count, kernel_ms, topk_ms, total_ms);
    });
}

int ifs_cuda_grouped_search(void *handle, const float *query, uint64_t k,
                            uint64_t *out_group_ids,
                            uint64_t *out_vector_ids, float *out_scores,
                            uint64_t *out_count, double *kernel_ms,
                            double *topk_ms, double *total_ms) {
    return protect([&] {
        index_from(handle).grouped_search(
            query, k, out_group_ids, out_vector_ids, out_scores, out_count,
            kernel_ms, topk_ms, total_ms);
    });
}

int ifs_cuda_get_stats(void *handle, ifs_cuda_stats *out) {
    return protect([&] {
        if (!out) throw std::invalid_argument("stats output pointer is null");
        *out = index_from(handle).stats();
    });
}

const char *ifs_cuda_last_error(void) { return g_last_error.c_str(); }

const char *ifs_cuda_build_info(void) {
    return "ifs_cuda flat-ip d512; cuBLAS; FP32/FP16/BF16/INT8(per-index-scale); "
           "float-accumulation except INT8 int32; exact scan; exact CUDA "
           "face/grouped Top-K<=100; grouped device-resident";
}

}  // extern "C"
