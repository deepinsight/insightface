/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIREFACE_LAUNCH_H
#define INSPIREFACE_LAUNCH_H
#include "middleware/model_archive/inspire_archive.h"
#if defined(ISF_ENABLE_RGA)
#include "middleware/nexus_processor/rga/dma_alloc.h"
#endif
#include <mutex>
#include "middleware/inference_wrapper/inference_wrapper.h"
#include "middleware/system.h"

#ifndef INSPIRE_API
#define INSPIRE_API
#endif

#define INSPIRE_LAUNCH inspire::Launch::GetInstance()

namespace inspire {

// The Launch class acts as the main entry point for the InspireFace system.
// It is responsible for loading static resources such as models, configurations, and parameters.
class INSPIRE_API Launch {
public:
    Launch(const Launch&) = delete;             // Delete the copy constructor to prevent copying.
    Launch& operator=(const Launch&) = delete;  // Delete the assignment operator to prevent assignment.

    // Retrieves the singleton instance of Launch, ensuring that only one instance exists.
    static std::shared_ptr<Launch> GetInstance();

    // Loads the necessary resources from a specified path.
    // Returns an integer status code: 0 on success, non-zero on failure.
    int32_t Load(const std::string& path);

    // Reloads the resources from a specified path.
    // Returns an integer status code: 0 on success, non-zero on failure.
    int32_t Reload(const std::string& path);

    // Provides access to the loaded InspireArchive instance.
    InspireArchive& getMArchive();

    // Checks if the resources have been successfully loaded.
    bool isMLoad() const;

    // Unloads the resources and resets the system to its initial state.
    void Unload();

    // Set the rockchip dma heap path
    void SetRockchipDmaHeapPath(const std::string& path);

    // Get the rockchip dma heap path
    std::string GetRockchipDmaHeapPath() const;

    // Set the extension path
    void ConfigurationExtensionPath(const std::string& path);

    // Get the extension path
    std::string GetExtensionPath() const;

    // Set the global coreml inference mode
    void SetGlobalCoreMLInferenceMode(InferenceWrapper::SpecialBackend mode);

    // Get the global coreml inference mode
    InferenceWrapper::SpecialBackend GetGlobalCoreMLInferenceMode() const;

    // Build the extension path
    void BuildAppleExtensionPath(const std::string& resource_path);

    // Set the cuda device id
    void SetCudaDeviceId(int32_t device_id);

    // Get the cuda device id
    int32_t GetCudaDeviceId() const;

private:
    // Parameters
    std::string m_rockchip_dma_heap_path_;

    // Constructor
    Launch() : m_load_(false), m_archive_(nullptr) {
#if defined(ISF_ENABLE_RGA)
#if defined(ISF_RKNPU_RV1106)
        m_rockchip_dma_heap_path_ = RV1106_CMA_HEAP_PATH;
#else
        m_rockchip_dma_heap_path_ = DMA_HEAP_DMA32_UNCACHE_PATCH;
#endif
        INSPIRE_LOGW("Rockchip dma heap configured path: %s", m_rockchip_dma_heap_path_.c_str());
#endif
    }  ///< Private constructor for the singleton pattern.

    static std::mutex mutex_;                  ///< Mutex for synchronizing access to the singleton instance.
    static std::shared_ptr<Launch> instance_;  ///< The singleton instance of Launch.

    std::string m_extension_path_;

    std::unique_ptr<InspireArchive> m_archive_;  ///< The archive containing all necessary resources.
    bool m_load_;                                ///< Flag indicating whether the resources have been successfully loaded.

    int32_t m_cuda_device_id_{0};

    InferenceWrapper::SpecialBackend m_global_coreml_inference_mode_{InferenceWrapper::COREML_ANE};  ///< The global coreml inference mode
};

}  // namespace inspire

#endif  // INSPIREFACE_LAUNCH_H
