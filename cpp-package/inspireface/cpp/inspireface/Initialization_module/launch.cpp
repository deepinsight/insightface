/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "launch.h"
#include "log.h"
#include "herror.h"
#include "isf_check.h"
#include "middleware/cuda_toolkit.h"
#if defined(ISF_ENABLE_TENSORRT)
#include "middleware/cuda_toolkit.h"
#endif

#define APPLE_EXTENSION_SUFFIX ".bundle"

namespace inspire {

std::mutex Launch::mutex_;
std::shared_ptr<Launch> Launch::instance_ = nullptr;

InspireArchive& Launch::getMArchive() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!m_archive_) {
        throw std::runtime_error("Archive not initialized");
    }
    return *m_archive_;
}

std::shared_ptr<Launch> Launch::GetInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!instance_) {
        instance_ = std::shared_ptr<Launch>(new Launch());
    }
    return instance_;
}

int32_t Launch::Load(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(ISF_ENABLE_TENSORRT)
    int32_t support_cuda;
    auto ret = CheckCudaUsability(&support_cuda);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("An error occurred while checking CUDA device support. Please ensure that your environment supports CUDA!");
        return ret;
    }
    if (!support_cuda) {
        INSPIRE_LOGE("Your environment does not support CUDA! Please ensure that your environment supports CUDA!");
        return HERR_DEVICE_CUDA_NOT_SUPPORT;
    }
#endif
    INSPIREFACE_CHECK_MSG(os::IsExists(path), "The package path does not exist because the launch failed.");
#if defined(ISF_ENABLE_APPLE_EXTENSION)
    BuildAppleExtensionPath(path);
#endif
    if (!m_load_) {
        try {
            m_archive_ = std::make_unique<InspireArchive>();
            m_archive_->ReLoad(path);

            if (m_archive_->QueryStatus() == SARC_SUCCESS) {
                m_load_ = true;
                INSPIRE_LOGI("Successfully loaded resources");
                return HSUCCEED;
            } else {
                m_archive_.reset();
                INSPIRE_LOGE("Failed to load resources");
                return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
            }
        } catch (const std::exception& e) {
            m_archive_.reset();
            INSPIRE_LOGE("Exception during resource loading: %s", e.what());
            return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
        }
    } else {
        INSPIRE_LOGW("There is no need to call launch more than once, as subsequent calls will not affect the initialization.");
        return HSUCCEED;
    }
}

int32_t Launch::Reload(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    INSPIREFACE_CHECK_MSG(os::IsExists(path), "The package path does not exist because the launch failed.");
#if defined(ISF_ENABLE_APPLE_EXTENSION)
    BuildAppleExtensionPath(path);
#endif
    try {
        // Clean up existing archive if it exists
        if (m_archive_) {
            m_archive_.reset();
            m_load_ = false;
        }

        // Create and load new archive
        m_archive_ = std::make_unique<InspireArchive>();
        m_archive_->ReLoad(path);

        if (m_archive_->QueryStatus() == SARC_SUCCESS) {
            m_load_ = true;
            INSPIRE_LOGI("Successfully reloaded resources");
            return HSUCCEED;
        } else {
            m_archive_.reset();
            INSPIRE_LOGE("Failed to reload resources");
            return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
        }
    } catch (const std::exception& e) {
        m_archive_.reset();
        INSPIRE_LOGE("Exception during resource reloading: %s", e.what());
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }
}

bool Launch::isMLoad() const {
    return m_load_;
}

void Launch::Unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (m_load_) {
        m_archive_.reset();
        m_load_ = false;
        INSPIRE_LOGI("All resources have been successfully unloaded and system is reset.");
    } else {
        INSPIRE_LOGW("Unload called but system was not loaded.");
    }
}

void Launch::SetRockchipDmaHeapPath(const std::string& path) {
    m_rockchip_dma_heap_path_ = path;
}

std::string Launch::GetRockchipDmaHeapPath() const {
    return m_rockchip_dma_heap_path_;
}

void Launch::ConfigurationExtensionPath(const std::string& path) {
#if defined(ISF_ENABLE_APPLE_EXTENSION)
    INSPIREFACE_CHECK_MSG(os::IsDir(path), "The apple extension path is not a directory, please check.");
#endif
    INSPIREFACE_CHECK_MSG(os::IsExists(path), "The extension path is not exists, please check.");
    m_extension_path_ = path;
}

std::string Launch::GetExtensionPath() const {
    return m_extension_path_;
}

void Launch::SetGlobalCoreMLInferenceMode(InferenceWrapper::SpecialBackend mode) {
    m_global_coreml_inference_mode_ = mode;
    if (m_global_coreml_inference_mode_ == InferenceWrapper::COREML_CPU) {
        INSPIRE_LOGW("Global CoreML Compute Units set to CPU Only.");
    } else if (m_global_coreml_inference_mode_ == InferenceWrapper::COREML_GPU) {
        INSPIRE_LOGW("Global CoreML Compute Units set to CPU and GPU.");
    } else if (m_global_coreml_inference_mode_ == InferenceWrapper::COREML_ANE) {
        INSPIRE_LOGW("Global CoreML Compute Units set to Auto Switch (ANE, GPU, CPU).");
    }
}

InferenceWrapper::SpecialBackend Launch::GetGlobalCoreMLInferenceMode() const {
    return m_global_coreml_inference_mode_;
}

void Launch::BuildAppleExtensionPath(const std::string& resource_path) {
    std::string basename = os::Basename(resource_path);
    m_extension_path_ = os::PathJoin(os::Dirname(resource_path), basename + APPLE_EXTENSION_SUFFIX);
    INSPIREFACE_CHECK_MSG(os::IsExists(m_extension_path_), "The apple extension path is not exists, please check.");
    INSPIREFACE_CHECK_MSG(os::IsDir(m_extension_path_), "The apple extension path is not a directory, please check.");
}

void Launch::SetCudaDeviceId(int32_t device_id) {
    m_cuda_device_id_ = device_id;
}

int32_t Launch::GetCudaDeviceId() const {
    return m_cuda_device_id_;
}

}  // namespace inspire