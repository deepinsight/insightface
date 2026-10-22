#ifndef IFS_CUDA_H
#define IFS_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ifs_cuda_dtype {
    IFS_CUDA_FP32 = 0,
    IFS_CUDA_FP16 = 1,
    IFS_CUDA_BF16 = 2,
    IFS_CUDA_INT8 = 3,
};

enum ifs_cuda_topk_mode {
    /* Compatibility path: copy every score to the host, then select Top-K. */
    IFS_CUDA_TOPK_HOST = 0,
    /* Exact multi-stage CUDA Top-K; only the final K candidates cross PCIe. */
    IFS_CUDA_TOPK_DEVICE = 1,
};

typedef struct ifs_cuda_stats {
    uint64_t size;
    uint64_t live_size;
    uint64_t capacity;
    uint64_t reallocations;
    uint64_t bytes_per_vector;
    int device;
    int dtype;
} ifs_cuda_stats;

/*
 * Exact, append-friendly, flat inner-product index for fixed 512-D vectors.
 * Input vectors and queries are always normalized float32. Storage is selected
 * by dtype. INT8 conversion rounds half away from zero after multiplying by
 * the positive per-index scale, then clamps to [-128, 127]. Non-INT8 indexes
 * require int8_scale=0.
 */
void *ifs_cuda_create(int dtype, uint64_t reserve_rows, int device,
                      double growth_factor, uint32_t int8_scale);
void ifs_cuda_destroy(void *handle);

int ifs_cuda_reserve(void *handle, uint64_t rows);
int ifs_cuda_add(void *handle, const float *vectors, const uint64_t *ids,
                 const uint64_t *group_ids, uint64_t count);
int ifs_cuda_remove(void *handle, const uint64_t *ids, uint64_t count,
                    uint64_t *removed);

/*
 * Returns up to k live results, sorted by descending dot product and then
 * ascending stable external ID. out_count may be less than k.
 * kernel_ms measures cuBLAS dot-product work only; total_ms includes device to
 * host score transfer and exact host Top-K selection.
 */
int ifs_cuda_search(void *handle, const float *query, uint64_t k,
                    uint64_t *out_ids, float *out_scores,
                    uint64_t *out_count, double *kernel_ms,
                    double *total_ms);

/*
 * Extended search with a selectable Top-K implementation. topk_ms reports
 * host selection time in HOST mode and CUDA selection time in DEVICE mode.
 * DEVICE mode supports k in [1, 100], performs exact selection on the GPU,
 * skips deleted rows, and preserves the same score-descending/ID-ascending
 * ordering as the compatibility path.
 */
int ifs_cuda_search_ex(void *handle, const float *query, uint64_t k,
                       int topk_mode, uint64_t *out_ids,
                       float *out_scores, uint64_t *out_count,
                       double *kernel_ms, double *topk_ms,
                       double *total_ms);

/*
 * Exact GPU-resident Person Top-K. Every live row contributes to its group;
 * each group keeps the highest score (then the smallest vector ID on a tie).
 * Groups are returned score-descending/group-ID-ascending. k must be <= 100,
 * and only the final k candidates are copied to host memory.
 */
int ifs_cuda_grouped_search(void *handle, const float *query, uint64_t k,
                            uint64_t *out_group_ids,
                            uint64_t *out_vector_ids, float *out_scores,
                            uint64_t *out_count, double *kernel_ms,
                            double *topk_ms, double *total_ms);

int ifs_cuda_get_stats(void *handle, ifs_cuda_stats *out);
const char *ifs_cuda_last_error(void);
const char *ifs_cuda_build_info(void);

#ifdef __cplusplus
}
#endif

#endif
