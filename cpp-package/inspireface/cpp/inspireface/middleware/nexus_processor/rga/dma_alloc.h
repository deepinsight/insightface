#ifndef RGA_ALLOCATOR_DMA_ALLOC_H__
#define RGA_ALLOCATOR_DMA_ALLOC_H__

#if defined(ISF_ENABLE_RGA)

#define DMA_HEAP_UNCACHE_PATH "/dev/dma_heap/system-uncached"
#define DMA_HEAP_PATH "/dev/dma_heap/system"
#define DMA_HEAP_DMA32_UNCACHE_PATCH "/dev/dma_heap/system-uncached-dma32"
#define DMA_HEAP_DMA32_PATCH "/dev/dma_heap/system-dma32"
#define CMA_HEAP_UNCACHE_PATH "/dev/dma_heap/cma-uncached"
#define RV1106_CMA_HEAP_PATH "/dev/rk_dma_heap/rk-dma-heap-cma"

int dma_sync_device_to_cpu(int fd);
int dma_sync_cpu_to_device(int fd);

int dma_buf_alloc(const char *path, size_t size, int *fd, void **va);
void dma_buf_free(size_t size, int *fd, void *va);

#endif /* #ifndef RGA_ALLOCATOR_DMA_ALLOC_H__ */

#endif  // ISF_ENABLE_RGA