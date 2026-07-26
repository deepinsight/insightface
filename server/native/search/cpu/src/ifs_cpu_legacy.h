#ifndef IFS_CPU_H
#define IFS_CPU_H

#include <stddef.h>
#include <stdint.h>

/* Internal compatibility API imported from the standalone benchmark. The
 * product library exports only ifs_search.h ABI v2. */
#define IFS_CPU_API

#ifdef __cplusplus
extern "C" {
#endif

#define IFS_CPU_DIMENSION 512u

typedef void *ifs_cpu_index_t;

typedef enum ifs_cpu_storage {
    IFS_CPU_STORAGE_FP32 = 0,
    IFS_CPU_STORAGE_BF16 = 1,
    IFS_CPU_STORAGE_INT8 = 2
} ifs_cpu_storage_t;

typedef enum ifs_cpu_status {
    IFS_CPU_OK = 0,
    IFS_CPU_INVALID_ARGUMENT = 1,
    IFS_CPU_OUT_OF_MEMORY = 2,
    IFS_CPU_DUPLICATE_ID = 3,
    IFS_CPU_ID_NOT_FOUND = 4,
    IFS_CPU_INTERNAL_ERROR = 5
} ifs_cpu_status_t;

/* Error text is thread-local and remains valid until the next API call on the
 * same thread. Successful calls clear it. */
IFS_CPU_API const char *ifs_cpu_last_error(void);
IFS_CPU_API const char *ifs_cpu_version(void);
IFS_CPU_API uint32_t ifs_cpu_dimension(void);

/* Human-readable process-wide dispatch diagnostics. */
IFS_CPU_API const char *ifs_cpu_runtime_features(void);
IFS_CPU_API const char *ifs_cpu_kernel_name(ifs_cpu_storage_t storage);

IFS_CPU_API ifs_cpu_status_t ifs_cpu_index_create(
    ifs_cpu_storage_t storage,
    size_t initial_capacity,
    uint32_t int8_scale,
    ifs_cpu_index_t *out_index);

IFS_CPU_API void ifs_cpu_index_destroy(ifs_cpu_index_t index);

IFS_CPU_API size_t ifs_cpu_index_size(ifs_cpu_index_t index);
IFS_CPU_API size_t ifs_cpu_index_capacity(ifs_cpu_index_t index);
IFS_CPU_API ifs_cpu_storage_t ifs_cpu_index_storage(ifs_cpu_index_t index);
IFS_CPU_API ifs_cpu_status_t ifs_cpu_index_reserve(
    ifs_cpu_index_t index,
    size_t rows);

/* Vectors must contain exactly IFS_CPU_DIMENSION finite FP32 values. IDs are
 * unique among active entries; a deleted ID may be added again. */
IFS_CPU_API ifs_cpu_status_t ifs_cpu_index_add(
    ifs_cpu_index_t index,
    uint64_t id,
    const float *vector512);

/* Row-major vectors[count][512]. This performs one validation and write lock
 * for the complete batch and is the intended ingestion path for large bases. */
IFS_CPU_API ifs_cpu_status_t ifs_cpu_index_add_batch(
    ifs_cpu_index_t index,
    const uint64_t *ids,
    const float *vectors,
    size_t count);

IFS_CPU_API ifs_cpu_status_t ifs_cpu_index_delete(
    ifs_cpu_index_t index,
    uint64_t id);

/* Results are ordered by descending score, then ascending ID. For INT8,
 * out_scores contains the exact INT32 accumulator converted to float. */
IFS_CPU_API ifs_cpu_status_t ifs_cpu_index_search(
    ifs_cpu_index_t index,
    const float *query512,
    size_t top_k,
    uint64_t *out_ids,
    float *out_scores,
    size_t *out_count);

#ifdef __cplusplus
}
#endif

#endif
