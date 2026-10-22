#ifndef IFS_SEARCH_H
#define IFS_SEARCH_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(IFS_SEARCH_BUILDING)
#    define IFS_SEARCH_API __declspec(dllexport)
#  else
#    define IFS_SEARCH_API __declspec(dllimport)
#  endif
#else
#  define IFS_SEARCH_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define IFS_SEARCH_ABI_VERSION 2u
#define IFS_SEARCH_DIMENSION 512u
#define IFS_SEARCH_INT8_X1000_SCALE 1000u
#define IFS_SEARCH_INT8_X736_SCALE 736u
/* Source-compatibility alias for ABI-v2 clients built before x736 existed. */
#define IFS_SEARCH_INT8_SCALE IFS_SEARCH_INT8_X1000_SCALE

typedef void *ifs_search_index_t;

typedef enum ifs_search_status {
    IFS_SEARCH_OK = 0,
    IFS_SEARCH_INVALID_ARGUMENT = 1,
    IFS_SEARCH_OUT_OF_MEMORY = 2,
    IFS_SEARCH_UNSUPPORTED = 3,
    IFS_SEARCH_DUPLICATE_ID = 4,
    IFS_SEARCH_ID_NOT_FOUND = 5,
    IFS_SEARCH_CAPACITY_EXCEEDED = 6,
    IFS_SEARCH_BACKEND_ERROR = 7,
    IFS_SEARCH_INTERNAL_ERROR = 8
} ifs_search_status_t;

typedef enum ifs_search_backend {
    IFS_SEARCH_BACKEND_CPU = 1,
    IFS_SEARCH_BACKEND_CUDA = 2
} ifs_search_backend_t;

/* These values are persisted by the Server. Do not renumber them. */
typedef enum ifs_search_profile {
    IFS_SEARCH_PROFILE_FP32_V1 = 0,
    IFS_SEARCH_PROFILE_FP16_V1 = 1,
    IFS_SEARCH_PROFILE_BF16_V1 = 2,
    IFS_SEARCH_PROFILE_INT8_X1000_V1 = 3,
    IFS_SEARCH_PROFILE_INT8_X736_V1 = 4
} ifs_search_profile_t;

typedef enum ifs_search_topk_mode {
    IFS_SEARCH_TOPK_AUTO = 0,
    IFS_SEARCH_TOPK_HOST = 1,
    IFS_SEARCH_TOPK_DEVICE = 2
} ifs_search_topk_mode_t;

enum ifs_search_capability_flag {
    IFS_SEARCH_CAP_EXACT_FLAT_SCAN = UINT64_C(1) << 0,
    IFS_SEARCH_CAP_BATCH_ADD = UINT64_C(1) << 1,
    IFS_SEARCH_CAP_BATCH_DELETE = UINT64_C(1) << 2,
    IFS_SEARCH_CAP_RESERVE = UINT64_C(1) << 3,
    IFS_SEARCH_CAP_DEVICE_TOPK = UINT64_C(1) << 4,
    IFS_SEARCH_CAP_GROUPED_PERSON_TOPK = UINT64_C(1) << 5,
    IFS_SEARCH_CAP_DELETED_SLOT_REUSE = UINT64_C(1) << 6,
    IFS_SEARCH_CAP_TOMBSTONE_DELETE = UINT64_C(1) << 7,
    /* Grouped Top-K is exact but aggregates after transferring/selecting all
     * face scores on the host. It is a correctness reference path. */
    IFS_SEARCH_CAP_GROUPED_HOST_REFERENCE = UINT64_C(1) << 8,
    /* Group reduction and exact Top-K selection remain on the accelerator;
     * only the final K group candidates cross the device boundary. */
    IFS_SEARCH_CAP_GROUPED_DEVICE_RESIDENT = UINT64_C(1) << 9
};

typedef struct ifs_search_capabilities {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t dimension;
    uint32_t backend;
    uint64_t profile_mask;
    uint64_t flags;
    uint64_t device_topk_limit;
    int32_t device;
    int32_t compute_capability_major;
    int32_t compute_capability_minor;
    int32_t cuda_runtime_version;
    int32_t cuda_driver_version;
} ifs_search_capabilities_t;

typedef struct ifs_search_create_options {
    uint32_t struct_size;
    uint32_t profile;
    uint64_t reserve_rows;
    /* Zero means no logical hard limit. A non-zero limit is checked before
     * every add. Set reserve_rows == max_rows to guarantee no data reallocation
     * while the collection remains within its configured capacity. */
    uint64_t max_rows;
    int32_t device;
    uint32_t topk_mode;
    double growth_factor;
    uint64_t reserved[4];
} ifs_search_create_options_t;

typedef struct ifs_search_stats {
    uint32_t struct_size;
    uint32_t backend;
    uint32_t profile;
    int32_t device;
    uint64_t physical_rows;
    uint64_t live_rows;
    uint64_t capacity_rows;
    uint64_t max_rows;
    uint64_t tombstone_rows;
    /* Number of automatic data-storage growth events after creation. Explicit
     * reserve calls are not counted. */
    uint64_t reallocations;
    uint64_t bytes_per_vector;
} ifs_search_stats_t;

typedef struct ifs_search_timings {
    uint32_t struct_size;
    uint32_t reserved;
    double kernel_ms;
    double topk_ms;
    double total_ms;
} ifs_search_timings_t;

IFS_SEARCH_API uint32_t ifs_search_abi_version(void);
IFS_SEARCH_API uint32_t ifs_search_dimension(void);
IFS_SEARCH_API const char *ifs_search_build_info(void);
IFS_SEARCH_API const char *ifs_search_last_error(void);
IFS_SEARCH_API const char *ifs_search_status_string(ifs_search_status_t status);

/* The CPU library accepts device=-1. The CUDA library requires a valid CUDA
 * device ordinal. profile_mask is authoritative: unsupported profiles fail
 * creation instead of falling back to another representation or backend. */
IFS_SEARCH_API ifs_search_status_t ifs_search_get_capabilities(
    int32_t device,
    ifs_search_capabilities_t *out_capabilities);

IFS_SEARCH_API ifs_search_status_t ifs_search_create(
    const ifs_search_create_options_t *options,
    ifs_search_index_t *out_index);
IFS_SEARCH_API void ifs_search_destroy(ifs_search_index_t index);

IFS_SEARCH_API ifs_search_status_t ifs_search_reserve(
    ifs_search_index_t index,
    uint64_t rows);

/* All vectors and queries are row-major finite FP32 input with exactly 512
 * values per row and must already be L2-normalized. Low-precision profiles
 * quantize that normalized FP32 representation internally. UINT64_MAX is
 * reserved as an internal sentinel and is not a valid vector ID. */
IFS_SEARCH_API ifs_search_status_t ifs_search_add_batch(
    ifs_search_index_t index,
    const uint64_t *vector_ids,
    const uint64_t *group_ids,
    const float *vectors,
    uint64_t count);

/* Missing IDs are ignored. out_removed receives the number actually removed.
 * CPU indexes reuse deleted slots. CUDA indexes retain tombstones until the
 * Server rebuilds the collection generation. */
IFS_SEARCH_API ifs_search_status_t ifs_search_delete_batch(
    ifs_search_index_t index,
    const uint64_t *vector_ids,
    uint64_t count,
    uint64_t *out_removed);

/* Exact face-vector Top-K. Results are score descending, vector ID ascending.
 * Scores always use raw cosine/inner-product semantics. INT8 scores are the
 * int32 accumulator divided by the selected profile's scale squared, never
 * the unscaled accumulator. */
IFS_SEARCH_API ifs_search_status_t ifs_search_topk(
    ifs_search_index_t index,
    const float *query512,
    uint64_t top_k,
    uint64_t *out_vector_ids,
    float *out_cosine_scores,
    uint64_t *out_count,
    ifs_search_timings_t *out_timings);

/* Exact grouped Top-K. Each vector's group ID is supplied at add time. Every
 * live face score participates, each group retains its maximum-scoring face,
 * then groups are sorted by score descending, group ID ascending, and vector
 * ID ascending. This is not a fixed face-candidate oversample and therefore
 * cannot omit a Person. Capability flags distinguish host-reference and
 * accelerator-resident implementations. */
IFS_SEARCH_API ifs_search_status_t ifs_search_grouped_topk(
    ifs_search_index_t index,
    const float *query512,
    uint64_t top_k,
    uint64_t *out_group_ids,
    uint64_t *out_vector_ids,
    float *out_cosine_scores,
    uint64_t *out_count,
    ifs_search_timings_t *out_timings);

IFS_SEARCH_API ifs_search_status_t ifs_search_get_stats(
    ifs_search_index_t index,
    ifs_search_stats_t *out_stats);

#ifdef __cplusplus
}
#endif

#endif
