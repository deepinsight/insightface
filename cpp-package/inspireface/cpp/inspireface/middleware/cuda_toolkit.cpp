#ifdef ISF_ENABLE_TENSORRT
#include <cuda_toolkit.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#endif  // ISF_ENABLE_TENSORRT
#include <log.h>
#include <herror.h>

namespace inspire {

int32_t INSPIRE_API_EXPORT GetCudaDeviceCount(int32_t *device_count) {
#ifdef ISF_ENABLE_TENSORRT
    cudaError_t error = cudaGetDeviceCount(device_count);
    if (error != cudaSuccess) {
        INSPIRE_LOGE("CUDA error: %s", cudaGetErrorString(error));
        return HERR_DEVICE_CUDA_UNKNOWN_ERROR;
    }
    return HSUCCEED;
#else
    *device_count = 0;
    return HERR_DEVICE_CUDA_NOT_SUPPORT;
#endif
}

int32_t INSPIRE_API_EXPORT CheckCudaUsability(int32_t *is_support) {
#ifdef ISF_ENABLE_TENSORRT
    int device_count;
    auto ret = GetCudaDeviceCount(&device_count);
    if (ret != HSUCCEED) {
        return ret;
    }
    if (device_count == 0) {
        *is_support = 0;
        INSPIRE_LOGE("No CUDA devices found");
        return HERR_DEVICE_CUDA_NOT_SUPPORT;
    }
    *is_support = device_count > 0;
    return HSUCCEED;
#else
    *is_support = 0;
    return HERR_DEVICE_CUDA_NOT_SUPPORT;
#endif
}

int32_t INSPIRE_API_EXPORT _PrintCudaDeviceInfo() {
#ifdef ISF_ENABLE_TENSORRT
    try {
        INSPIRE_LOGW("TensorRT version: %d.%d.%d", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

        // check if CUDA is available
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess) {
            INSPIRE_LOGE("CUDA error: %s", cudaGetErrorString(error));
            return HERR_DEVICE_CUDA_UNKNOWN_ERROR;
        }
        INSPIRE_LOGW("available CUDA devices: %d", device_count);

        // get current CUDA device
        int currentDevice;
        error = cudaGetDevice(&currentDevice);
        if (error != cudaSuccess) {
            INSPIRE_LOGE("[CUDA error] failed to get current CUDA device: %s", cudaGetErrorString(error));
            return HERR_DEVICE_CUDA_UNKNOWN_ERROR;
        }
        INSPIRE_LOGW("current CUDA device ID: %d", currentDevice);

        // get GPU device properties
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, currentDevice);
        if (error != cudaSuccess) {
            INSPIRE_LOGE("[CUDA error] failed to get CUDA device properties: %s", cudaGetErrorString(error));
            return HERR_DEVICE_CUDA_UNKNOWN_ERROR;
        }

        // print device detailed information
        INSPIRE_LOGW("\nCUDA device details:");
        INSPIRE_LOGW("device name: %s", prop.name);
        INSPIRE_LOGW("compute capability: %d.%d", prop.major, prop.minor);
        INSPIRE_LOGW("global memory: %d MB", prop.totalGlobalMem / (1024 * 1024));
        INSPIRE_LOGW("max shared memory/block: %d KB", prop.sharedMemPerBlock / 1024);
        INSPIRE_LOGW("max threads/block: %d", prop.maxThreadsPerBlock);
        INSPIRE_LOGW("max block dimensions: (%d, %d, %d)", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        INSPIRE_LOGW("max grid size: (%d, %d, %d)", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        INSPIRE_LOGW("total constant memory: %d KB", prop.totalConstMem / 1024);
        INSPIRE_LOGW("multi-processor count: %d", prop.multiProcessorCount);
        INSPIRE_LOGW("max blocks per multi-processor: %d", prop.maxBlocksPerMultiProcessor);
        INSPIRE_LOGW("clock frequency: %d MHz", prop.clockRate / 1000);
        INSPIRE_LOGW("memory frequency: %d MHz", prop.memoryClockRate / 1000);
        INSPIRE_LOGW("memory bus width: %d bits", prop.memoryBusWidth);
        INSPIRE_LOGW("L2 cache size: %d KB", prop.l2CacheSize / 1024);
        INSPIRE_LOGW("theoretical memory bandwidth: %f GB/s", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

        // check if FP16 is supported
        bool supportsFP16 = prop.major >= 6 || (prop.major == 5 && prop.minor >= 3);
        INSPIRE_LOGW("FP16 support: %s", supportsFP16 ? "yes" : "no");

        // check if unified memory is supported
        INSPIRE_LOGW("unified memory support: %s", prop.unifiedAddressing ? "yes" : "no");

        // check if concurrent kernel execution is supported
        INSPIRE_LOGW("concurrent kernel execution: %s", prop.concurrentKernels ? "yes" : "no");

        // check if asynchronous engine is supported
        INSPIRE_LOGW("asynchronous engine count: %d", prop.asyncEngineCount);

        return HSUCCEED;
    } catch (const std::exception &e) {
        INSPIRE_LOGE("error when printing CUDA device info: %s", e.what());
        return HERR_DEVICE_CUDA_UNKNOWN_ERROR;
    }
#else
    INSPIRE_LOGE("CUDA/TensorRT support is not enabled");
    return HERR_DEVICE_CUDA_NOT_SUPPORT;
#endif
}

int32_t INSPIRE_API_EXPORT PrintCudaDeviceInfo() {
#ifdef ISF_ENABLE_TENSORRT
    INSPIRE_LOGW("================================================");
    auto ret = _PrintCudaDeviceInfo();
    INSPIRE_LOGW("================================================");
    return ret;
#else
    INSPIRE_LOGE("CUDA/TensorRT support is not enabled");
    return HERR_DEVICE_CUDA_NOT_SUPPORT;
#endif
}

}  // namespace inspire