#include "ifs_cpu_legacy.h"

#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ifs_cpu {
namespace {

class Index {
public:
    explicit Index(ifs_cpu_storage_t storage, std::size_t initial_capacity,
                   std::uint32_t int8_scale)
        : storage_(storage), int8_scale_(int8_scale) {
        reserve_capacity(initial_capacity);
    }

    bool add(std::uint64_t id, const float *input) {
        std::array<float, kDimension> fp32{};
        std::array<std::uint16_t, kDimension> bf16{};
        std::array<std::int8_t, kDimension> int8{};
        for (std::size_t offset = 0; offset < kDimension; ++offset) {
            if (!std::isfinite(input[offset])) {
                throw std::invalid_argument("vector contains a non-finite value");
            }
            switch (storage_) {
                case IFS_CPU_STORAGE_FP32:
                    fp32[offset] = input[offset];
                    break;
                case IFS_CPU_STORAGE_BF16:
                    bf16[offset] = float_to_bf16(input[offset]);
                    break;
                case IFS_CPU_STORAGE_INT8:
                    int8[offset] = quantize_int8(input[offset], int8_scale_);
                    break;
            }
        }

        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (id_to_slot_.find(id) != id_to_slot_.end()) {
            return false;
        }

        std::size_t slot = 0;
        if (!free_slots_.empty()) {
            slot = free_slots_.back();
            free_slots_.pop_back();
            ids_[slot] = id;
            active_[slot] = 1;
            write_slot(slot, fp32, bf16, int8);
        } else {
            ensure_capacity(ids_.size() + 1);
            slot = ids_.size();
            ids_.push_back(id);
            active_.push_back(1);
            append_slot(fp32, bf16, int8);
        }

        try {
            id_to_slot_.emplace(id, slot);
        } catch (...) {
            active_[slot] = 0;
            free_slots_.push_back(slot);
            throw;
        }
        ++active_count_;
        return true;
    }

    bool add_batch(const std::uint64_t *ids, const float *inputs,
                   std::size_t count) {
        if (count == 0) {
            return true;
        }
        std::unordered_map<std::uint64_t, std::uint8_t> batch_ids;
        batch_ids.reserve(count);
        for (std::size_t row = 0; row < count; ++row) {
            if (!batch_ids.emplace(ids[row], 1).second) {
                return false;
            }
            const float *input = inputs + row * kDimension;
            for (std::size_t offset = 0; offset < kDimension; ++offset) {
                if (!std::isfinite(input[offset])) {
                    throw std::invalid_argument("vector batch contains a non-finite value");
                }
            }
        }

        std::unique_lock<std::shared_mutex> lock(mutex_);
        for (std::size_t row = 0; row < count; ++row) {
            if (id_to_slot_.find(ids[row]) != id_to_slot_.end()) {
                return false;
            }
        }
        const std::size_t reusable = std::min(count, free_slots_.size());
        if (count - reusable > std::numeric_limits<std::size_t>::max() - ids_.size()) {
            throw std::length_error("batch row count overflows index size");
        }
        ensure_capacity(ids_.size() + count - reusable);
        id_to_slot_.reserve(id_to_slot_.size() + count);

        for (std::size_t row = 0; row < count; ++row) {
            std::array<float, kDimension> fp32{};
            std::array<std::uint16_t, kDimension> bf16{};
            std::array<std::int8_t, kDimension> int8{};
            const float *input = inputs + row * kDimension;
            for (std::size_t offset = 0; offset < kDimension; ++offset) {
                switch (storage_) {
                    case IFS_CPU_STORAGE_FP32:
                        fp32[offset] = input[offset];
                        break;
                    case IFS_CPU_STORAGE_BF16:
                        bf16[offset] = float_to_bf16(input[offset]);
                        break;
                    case IFS_CPU_STORAGE_INT8:
                        int8[offset] = quantize_int8(input[offset], int8_scale_);
                        break;
                }
            }
            std::size_t slot = 0;
            if (!free_slots_.empty()) {
                slot = free_slots_.back();
                free_slots_.pop_back();
                ids_[slot] = ids[row];
                active_[slot] = 1;
                write_slot(slot, fp32, bf16, int8);
            } else {
                slot = ids_.size();
                ids_.push_back(ids[row]);
                active_.push_back(1);
                append_slot(fp32, bf16, int8);
            }
            id_to_slot_.emplace(ids[row], slot);
            ++active_count_;
        }
        return true;
    }

    bool erase(std::uint64_t id) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        const auto found = id_to_slot_.find(id);
        if (found == id_to_slot_.end()) {
            return false;
        }
        const std::size_t slot = found->second;
        id_to_slot_.erase(found);
        active_[slot] = 0;
        free_slots_.push_back(slot);
        --active_count_;
        return true;
    }

    std::vector<std::pair<std::uint64_t, float>> search(
        const float *query,
        std::size_t top_k) const {
        std::array<float, kDimension> fp32{};
        std::array<std::uint16_t, kDimension> bf16{};
        std::array<std::int8_t, kDimension> int8{};
        std::int32_t query_sum = 0;
        for (std::size_t offset = 0; offset < kDimension; ++offset) {
            if (!std::isfinite(query[offset])) {
                throw std::invalid_argument("query contains a non-finite value");
            }
            switch (storage_) {
                case IFS_CPU_STORAGE_FP32:
                    fp32[offset] = query[offset];
                    break;
                case IFS_CPU_STORAGE_BF16:
                    bf16[offset] = float_to_bf16(query[offset]);
                    break;
                case IFS_CPU_STORAGE_INT8:
                    int8[offset] = quantize_int8(query[offset], int8_scale_);
                    query_sum += static_cast<std::int32_t>(int8[offset]);
                    break;
            }
        }

        if (top_k == 0) {
            return {};
        }

        struct Candidate {
            std::uint64_t id;
            float score;
        };
        const auto better = [](const Candidate &left, const Candidate &right) {
            return left.score > right.score ||
                   (left.score == right.score && left.id < right.id);
        };
        struct BetterComparator {
            bool operator()(const Candidate &left, const Candidate &right) const {
                return left.score > right.score ||
                       (left.score == right.score && left.id < right.id);
            }
        };

        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto score_slot = [&](std::size_t slot) {
            if (active_[slot] == 0) {
                return Candidate{0, -std::numeric_limits<float>::infinity()};
            }
            const std::size_t base = slot * kDimension;
            float score = 0.0f;
            switch (storage_) {
                case IFS_CPU_STORAGE_FP32:
                    score = dispatch().fp32(fp32_.data() + base, fp32.data());
                    break;
                case IFS_CPU_STORAGE_BF16:
                    score = dispatch().bf16(bf16_.data() + base, bf16.data());
                    break;
                case IFS_CPU_STORAGE_INT8:
                    score = static_cast<float>(dispatch().int8(
                        int8_.data() + base, int8.data(), query_sum));
                    break;
            }
            return Candidate{ids_[slot], score};
        };
        auto consider = [&](auto &heap, const Candidate &candidate) {
            if (!std::isfinite(candidate.score)) return;
            if (heap.size() < top_k) heap.push(candidate);
            else if (better(candidate, heap.top())) {
                heap.pop();
                heap.push(candidate);
            }
        };

        std::priority_queue<Candidate, std::vector<Candidate>, BetterComparator> heap;
#ifdef _OPENMP
        const int thread_count = std::max(1, omp_get_max_threads());
        std::vector<std::vector<Candidate>> partials(
            static_cast<std::size_t>(thread_count));
#pragma omp parallel
        {
            std::priority_queue<Candidate, std::vector<Candidate>, BetterComparator>
                local_heap;
#pragma omp for schedule(static)
            for (std::int64_t signed_slot = 0;
                 signed_slot < static_cast<std::int64_t>(ids_.size()); ++signed_slot) {
                consider(local_heap, score_slot(static_cast<std::size_t>(signed_slot)));
            }
            auto &partial = partials[static_cast<std::size_t>(omp_get_thread_num())];
            partial.reserve(local_heap.size());
            while (!local_heap.empty()) {
                partial.push_back(local_heap.top());
                local_heap.pop();
            }
        }
        for (const auto &partial : partials) {
            for (const Candidate &candidate : partial) consider(heap, candidate);
        }
#else
        for (std::size_t slot = 0; slot < ids_.size(); ++slot) {
            consider(heap, score_slot(slot));
        }
#endif

        std::vector<Candidate> candidates;
        candidates.reserve(heap.size());
        while (!heap.empty()) {
            candidates.push_back(heap.top());
            heap.pop();
        }
        std::sort(candidates.begin(), candidates.end(), better);

        std::vector<std::pair<std::uint64_t, float>> results;
        results.reserve(candidates.size());
        for (const Candidate &candidate : candidates) {
            results.emplace_back(candidate.id, candidate.score);
        }
        return results;
    }

    std::size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return active_count_;
    }

    std::size_t capacity() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return capacity_;
    }

    ifs_cpu_storage_t storage() const noexcept {
        return storage_;
    }

    void reserve(std::size_t rows) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        reserve_capacity(rows);
    }

private:
    void ensure_capacity(std::size_t required) {
        if (required <= capacity_) {
            return;
        }
        std::size_t grown = capacity_ == 0 ? 1 : capacity_;
        while (grown < required) {
            if (grown > std::numeric_limits<std::size_t>::max() / 2) {
                grown = required;
                break;
            }
            grown *= 2;
        }
        reserve_capacity(grown);
    }

    void reserve_capacity(std::size_t requested) {
        if (requested < capacity_) {
            return;
        }
        if (requested > std::numeric_limits<std::size_t>::max() / kDimension) {
            throw std::length_error("capacity exceeds addressable storage");
        }
        const std::size_t elements = requested * kDimension;
        ids_.reserve(requested);
        active_.reserve(requested);
        free_slots_.reserve(requested);
        id_to_slot_.reserve(requested);
        switch (storage_) {
            case IFS_CPU_STORAGE_FP32:
                fp32_.reserve(elements);
                break;
            case IFS_CPU_STORAGE_BF16:
                bf16_.reserve(elements);
                break;
            case IFS_CPU_STORAGE_INT8:
                int8_.reserve(elements);
                break;
        }
        capacity_ = requested;
    }

    void write_slot(
        std::size_t slot,
        const std::array<float, kDimension> &fp32,
        const std::array<std::uint16_t, kDimension> &bf16,
        const std::array<std::int8_t, kDimension> &int8) {
        const std::size_t base = slot * kDimension;
        switch (storage_) {
            case IFS_CPU_STORAGE_FP32:
                std::copy(fp32.begin(), fp32.end(), fp32_.begin() + base);
                break;
            case IFS_CPU_STORAGE_BF16:
                std::copy(bf16.begin(), bf16.end(), bf16_.begin() + base);
                break;
            case IFS_CPU_STORAGE_INT8:
                std::copy(int8.begin(), int8.end(), int8_.begin() + base);
                break;
        }
    }

    void append_slot(
        const std::array<float, kDimension> &fp32,
        const std::array<std::uint16_t, kDimension> &bf16,
        const std::array<std::int8_t, kDimension> &int8) {
        switch (storage_) {
            case IFS_CPU_STORAGE_FP32:
                fp32_.insert(fp32_.end(), fp32.begin(), fp32.end());
                break;
            case IFS_CPU_STORAGE_BF16:
                bf16_.insert(bf16_.end(), bf16.begin(), bf16.end());
                break;
            case IFS_CPU_STORAGE_INT8:
                int8_.insert(int8_.end(), int8.begin(), int8.end());
                break;
        }
    }

    ifs_cpu_storage_t storage_;
    std::uint32_t int8_scale_;
    mutable std::shared_mutex mutex_;
    std::size_t capacity_ = 0;
    std::size_t active_count_ = 0;
    std::vector<std::uint64_t> ids_;
    std::vector<std::uint8_t> active_;
    std::vector<std::size_t> free_slots_;
    std::unordered_map<std::uint64_t, std::size_t> id_to_slot_;
    std::vector<float> fp32_;
    std::vector<std::uint16_t> bf16_;
    std::vector<std::int8_t> int8_;
};

thread_local std::string last_error;

void clear_error() {
    last_error.clear();
}

ifs_cpu_status_t fail(ifs_cpu_status_t status, const std::string &message) {
    last_error = message;
    return status;
}

Index *as_index(ifs_cpu_index_t index) {
    return static_cast<Index *>(index);
}

bool valid_storage(ifs_cpu_storage_t storage) {
    return storage == IFS_CPU_STORAGE_FP32 || storage == IFS_CPU_STORAGE_BF16 ||
           storage == IFS_CPU_STORAGE_INT8;
}

}  // namespace
}  // namespace ifs_cpu

extern "C" {

const char *ifs_cpu_last_error(void) {
    return ifs_cpu::last_error.c_str();
}

const char *ifs_cpu_version(void) {
    ifs_cpu::clear_error();
    return "0.1.0";
}

std::uint32_t ifs_cpu_dimension(void) {
    ifs_cpu::clear_error();
    return static_cast<std::uint32_t>(ifs_cpu::kDimension);
}

const char *ifs_cpu_runtime_features(void) {
    ifs_cpu::clear_error();
    try {
        return ifs_cpu::dispatch().features.c_str();
    } catch (const std::exception &error) {
        ifs_cpu::last_error = error.what();
        return "";
    } catch (...) {
        ifs_cpu::last_error = "unknown runtime dispatch failure";
        return "";
    }
}

const char *ifs_cpu_kernel_name(ifs_cpu_storage_t storage) {
    ifs_cpu::clear_error();
    try {
        const ifs_cpu::Dispatch &selected = ifs_cpu::dispatch();
        switch (storage) {
            case IFS_CPU_STORAGE_FP32:
                return selected.fp32_name;
            case IFS_CPU_STORAGE_BF16:
                return selected.bf16_name;
            case IFS_CPU_STORAGE_INT8:
                return selected.int8_name;
        }
        ifs_cpu::last_error = "invalid storage mode";
        return "invalid-storage";
    } catch (const std::exception &error) {
        ifs_cpu::last_error = error.what();
        return "";
    } catch (...) {
        ifs_cpu::last_error = "unknown runtime dispatch failure";
        return "";
    }
}

ifs_cpu_status_t ifs_cpu_index_create(
    ifs_cpu_storage_t storage,
    std::size_t initial_capacity,
    std::uint32_t int8_scale,
    ifs_cpu_index_t *out_index) {
    ifs_cpu::clear_error();
    if (out_index == nullptr) {
        return ifs_cpu::fail(IFS_CPU_INVALID_ARGUMENT, "out_index must not be null");
    }
    *out_index = nullptr;
    if (!ifs_cpu::valid_storage(storage)) {
        return ifs_cpu::fail(IFS_CPU_INVALID_ARGUMENT, "invalid storage mode");
    }
    if ((storage == IFS_CPU_STORAGE_INT8) != (int8_scale > 0)) {
        return ifs_cpu::fail(
            IFS_CPU_INVALID_ARGUMENT,
            "INT8 storage requires a positive scale and other storage requires zero");
    }
    try {
        *out_index = new ifs_cpu::Index(storage, initial_capacity, int8_scale);
        return IFS_CPU_OK;
    } catch (const std::bad_alloc &) {
        return ifs_cpu::fail(IFS_CPU_OUT_OF_MEMORY, "unable to reserve index capacity");
    } catch (const std::exception &error) {
        return ifs_cpu::fail(IFS_CPU_INVALID_ARGUMENT, error.what());
    } catch (...) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, "unknown index creation failure");
    }
}

void ifs_cpu_index_destroy(ifs_cpu_index_t index) {
    ifs_cpu::clear_error();
    delete ifs_cpu::as_index(index);
}

std::size_t ifs_cpu_index_size(ifs_cpu_index_t index) {
    ifs_cpu::clear_error();
    if (index == nullptr) {
        ifs_cpu::last_error = "index must not be null";
        return 0;
    }
    try {
        return ifs_cpu::as_index(index)->size();
    } catch (const std::exception &error) {
        ifs_cpu::last_error = error.what();
    } catch (...) {
        ifs_cpu::last_error = "unknown size failure";
    }
    return 0;
}

std::size_t ifs_cpu_index_capacity(ifs_cpu_index_t index) {
    ifs_cpu::clear_error();
    if (index == nullptr) {
        ifs_cpu::last_error = "index must not be null";
        return 0;
    }
    try {
        return ifs_cpu::as_index(index)->capacity();
    } catch (const std::exception &error) {
        ifs_cpu::last_error = error.what();
    } catch (...) {
        ifs_cpu::last_error = "unknown capacity failure";
    }
    return 0;
}

ifs_cpu_storage_t ifs_cpu_index_storage(ifs_cpu_index_t index) {
    ifs_cpu::clear_error();
    if (index == nullptr) {
        ifs_cpu::last_error = "index must not be null";
        return IFS_CPU_STORAGE_FP32;
    }
    try {
        return ifs_cpu::as_index(index)->storage();
    } catch (const std::exception &error) {
        ifs_cpu::last_error = error.what();
    } catch (...) {
        ifs_cpu::last_error = "unknown storage failure";
    }
    return IFS_CPU_STORAGE_FP32;
}

ifs_cpu_status_t ifs_cpu_index_reserve(
    ifs_cpu_index_t index,
    std::size_t rows) {
    ifs_cpu::clear_error();
    if (index == nullptr) {
        return ifs_cpu::fail(IFS_CPU_INVALID_ARGUMENT, "index must not be null");
    }
    try {
        ifs_cpu::as_index(index)->reserve(rows);
        return IFS_CPU_OK;
    } catch (const std::bad_alloc &) {
        return ifs_cpu::fail(IFS_CPU_OUT_OF_MEMORY, "unable to reserve index capacity");
    } catch (const std::invalid_argument &error) {
        return ifs_cpu::fail(IFS_CPU_INVALID_ARGUMENT, error.what());
    } catch (const std::exception &error) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, error.what());
    } catch (...) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, "unknown reserve failure");
    }
}

ifs_cpu_status_t ifs_cpu_index_add(
    ifs_cpu_index_t index,
    std::uint64_t id,
    const float *vector512) {
    ifs_cpu::clear_error();
    if (index == nullptr || vector512 == nullptr) {
        return ifs_cpu::fail(
            IFS_CPU_INVALID_ARGUMENT, "index and vector512 must not be null");
    }
    try {
        if (!ifs_cpu::as_index(index)->add(id, vector512)) {
            return ifs_cpu::fail(IFS_CPU_DUPLICATE_ID, "ID is already active");
        }
        return IFS_CPU_OK;
    } catch (const std::invalid_argument &error) {
        return ifs_cpu::fail(IFS_CPU_INVALID_ARGUMENT, error.what());
    } catch (const std::bad_alloc &) {
        return ifs_cpu::fail(IFS_CPU_OUT_OF_MEMORY, "index growth failed");
    } catch (const std::exception &error) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, error.what());
    } catch (...) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, "unknown add failure");
    }
}

ifs_cpu_status_t ifs_cpu_index_add_batch(
    ifs_cpu_index_t index,
    const std::uint64_t *ids,
    const float *vectors,
    std::size_t count) {
    ifs_cpu::clear_error();
    if (index == nullptr || (count > 0 && (ids == nullptr || vectors == nullptr))) {
        return ifs_cpu::fail(
            IFS_CPU_INVALID_ARGUMENT,
            "index, ids, and vectors must not be null for a non-empty batch");
    }
    try {
        if (!ifs_cpu::as_index(index)->add_batch(ids, vectors, count)) {
            return ifs_cpu::fail(IFS_CPU_DUPLICATE_ID, "duplicate active ID in batch add");
        }
        return IFS_CPU_OK;
    } catch (const std::bad_alloc &) {
        return ifs_cpu::fail(IFS_CPU_OUT_OF_MEMORY, "unable to grow index for batch");
    } catch (const std::invalid_argument &error) {
        return ifs_cpu::fail(IFS_CPU_INVALID_ARGUMENT, error.what());
    } catch (const std::exception &error) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, error.what());
    } catch (...) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, "unknown batch add failure");
    }
}

ifs_cpu_status_t ifs_cpu_index_delete(ifs_cpu_index_t index, std::uint64_t id) {
    ifs_cpu::clear_error();
    if (index == nullptr) {
        return ifs_cpu::fail(IFS_CPU_INVALID_ARGUMENT, "index must not be null");
    }
    try {
        if (!ifs_cpu::as_index(index)->erase(id)) {
            return ifs_cpu::fail(IFS_CPU_ID_NOT_FOUND, "ID is not active");
        }
        return IFS_CPU_OK;
    } catch (const std::exception &error) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, error.what());
    } catch (...) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, "unknown delete failure");
    }
}

ifs_cpu_status_t ifs_cpu_index_search(
    ifs_cpu_index_t index,
    const float *query512,
    std::size_t top_k,
    std::uint64_t *out_ids,
    float *out_scores,
    std::size_t *out_count) {
    ifs_cpu::clear_error();
    if (out_count == nullptr) {
        return ifs_cpu::fail(IFS_CPU_INVALID_ARGUMENT, "out_count must not be null");
    }
    *out_count = 0;
    if (index == nullptr || query512 == nullptr) {
        return ifs_cpu::fail(
            IFS_CPU_INVALID_ARGUMENT, "index and query512 must not be null");
    }
    if (top_k > 0 && (out_ids == nullptr || out_scores == nullptr)) {
        return ifs_cpu::fail(
            IFS_CPU_INVALID_ARGUMENT,
            "out_ids and out_scores must not be null when top_k is positive");
    }
    try {
        const auto results = ifs_cpu::as_index(index)->search(query512, top_k);
        for (std::size_t offset = 0; offset < results.size(); ++offset) {
            out_ids[offset] = results[offset].first;
            out_scores[offset] = results[offset].second;
        }
        *out_count = results.size();
        return IFS_CPU_OK;
    } catch (const std::invalid_argument &error) {
        return ifs_cpu::fail(IFS_CPU_INVALID_ARGUMENT, error.what());
    } catch (const std::bad_alloc &) {
        return ifs_cpu::fail(IFS_CPU_OUT_OF_MEMORY, "search allocation failed");
    } catch (const std::exception &error) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, error.what());
    } catch (...) {
        return ifs_cpu::fail(IFS_CPU_INTERNAL_ERROR, "unknown search failure");
    }
}

}  // extern "C"
