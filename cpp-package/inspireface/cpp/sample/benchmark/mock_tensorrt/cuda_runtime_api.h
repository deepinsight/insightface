#ifndef INSPIREFACE_SAMPLE_FAKE_CUDA_RUNTIME_API_H
#define INSPIREFACE_SAMPLE_FAKE_CUDA_RUNTIME_API_H

#include <cstddef>

struct InspireFaceFakeCudaStream {
    int id;
};

typedef InspireFaceFakeCudaStream *cudaStream_t;

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorUnknown = 999,
};

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
};

cudaError_t cudaSetDevice(int device);
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaMalloc(void **pointer, size_t size);
cudaError_t cudaFree(void *pointer);
cudaError_t cudaMemcpyAsync(void *destination, const void *source, size_t size, cudaMemcpyKind kind, cudaStream_t stream);
const char *cudaGetErrorString(cudaError_t error);

#endif
