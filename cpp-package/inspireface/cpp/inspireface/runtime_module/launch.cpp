/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "launch.h"
#include "log.h"
#include "herror.h"
#include "isf_check.h"

// Include the implementation details here, hidden from public header
#include "middleware/model_archive/inspire_archive.h"
#if defined(ISF_ENABLE_RGA)
#include "image_process/nexus_processor/rga/dma_alloc.h"
#endif
#include <mutex>
#include "middleware/inference_wrapper/inference_wrapper.h"
#include "middleware/system.h"
#if defined(ISF_ENABLE_TENSORRT)
#include "cuda_toolkit.h"
#endif
#include "meta.h"

#define APPLE_EXTENSION_SUFFIX ".bundle"

namespace inspire {

// Implementation class definition
class Launch::Impl {
public:
    Impl()
    : m_load_(false),
      m_archive_(nullptr),
      m_cuda_device_id_(0),
      m_global_coreml_inference_mode_(InferenceWrapper::COREML_ANE),
      m_image_processing_backend_(IMAGE_PROCESSING_CPU) {
#if defined(ISF_ENABLE_RGA)
        m_image_processing_backend_ = IMAGE_PROCESSING_RGA;
        INSPIRE_LOGW("Default image processing backend is RGA.");
#if defined(ISF_RKNPU_RV1106)
        m_rockchip_dma_heap_path_ = RV1106_CMA_HEAP_PATH;
#else
        m_rockchip_dma_heap_path_ = DMA_HEAP_DMA32_UNCACHE_PATCH;
#endif
        INSPIRE_LOGW("Rockchip dma heap configured path: %s", m_rockchip_dma_heap_path_.c_str());
#endif
        m_face_detect_pixel_list_ = {160, 320, 640};
        m_face_detect_model_list_ = {"face_detect_160", "face_detect_320", "face_detect_640"};
        INSPIRE_LOGI("%s", GetSDKInfo().GetFullVersionInfo().c_str());
    }
    // Face Detection pixel size
    std::vector<int32_t> m_face_detect_pixel_list_;

    // Face Detection model list
    std::vector<std::string> m_face_detect_model_list_;

    // Static members
    static std::mutex mutex_;
    static std::shared_ptr<Launch> instance_;

    // Data members
    std::string m_rockchip_dma_heap_path_;
    std::string m_extension_path_;
    std::unique_ptr<InspireArchive> m_archive_;
    bool m_load_;
    int32_t m_cuda_device_id_;
    InferenceWrapper::SpecialBackend m_global_coreml_inference_mode_;
    Launch::ImageProcessingBackend m_image_processing_backend_;
    int32_t m_image_process_aligned_width_{4};
};

// Initialize static members
std::mutex Launch::Impl::mutex_;
std::shared_ptr<Launch> Launch::Impl::instance_ = nullptr;

// Constructor implementation
Launch::Launch() : pImpl(std::make_unique<Impl>()) {}

// Destructor implementation
Launch::~Launch() = default;

std::shared_ptr<Launch> Launch::GetInstance() {
    std::lock_guard<std::mutex> lock(Impl::mutex_);
    if (!Impl::instance_) {
        Impl::instance_ = std::shared_ptr<Launch>(new Launch());
    }
    return Impl::instance_;
}

InspireArchive& Launch::getMArchive() {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    if (!pImpl->m_archive_) {
        throw std::runtime_error("Archive not initialized");
    }
    return *(pImpl->m_archive_);
}

int32_t Launch::Load(const std::string& path) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
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
    if (!pImpl->m_load_) {
        try {
            pImpl->m_archive_ = std::make_unique<InspireArchive>();
            pImpl->m_archive_->ReLoad(path);

            // Update face detect pixel list and model list
            pImpl->m_face_detect_pixel_list_ = pImpl->m_archive_->GetFaceDetectPixelList();
            pImpl->m_face_detect_model_list_ = pImpl->m_archive_->GetFaceDetectModelList();

            if (pImpl->m_archive_->QueryStatus() == SARC_SUCCESS) {
                pImpl->m_load_ = true;
                INSPIRE_LOGI("Successfully loaded resources");
                return HSUCCEED;
            } else {
                pImpl->m_archive_.reset();
                INSPIRE_LOGE("Failed to load resources");
                return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
            }
        } catch (const std::exception& e) {
            pImpl->m_archive_.reset();
            INSPIRE_LOGE("Exception during resource loading: %s", e.what());
            return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
        }
    } else {
        INSPIRE_LOGW("There is no need to call launch more than once, as subsequent calls will not affect the initialization.");
        return HSUCCEED;
    }
}

int32_t Launch::Reload(const std::string& path) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    INSPIREFACE_CHECK_MSG(os::IsExists(path), "The package path does not exist because the launch failed.");
#if defined(ISF_ENABLE_APPLE_EXTENSION)
    BuildAppleExtensionPath(path);
#endif
    try {
        // Clean up existing archive if it exists
        if (pImpl->m_archive_) {
            pImpl->m_archive_.reset();
            pImpl->m_load_ = false;
        }

        // Create and load new archive
        pImpl->m_archive_ = std::make_unique<InspireArchive>();
        pImpl->m_archive_->ReLoad(path);

        if (pImpl->m_archive_->QueryStatus() == SARC_SUCCESS) {
            pImpl->m_load_ = true;
            INSPIRE_LOGI("Successfully reloaded resources");
            return HSUCCEED;
        } else {
            pImpl->m_archive_.reset();
            INSPIRE_LOGE("Failed to reload resources");
            return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
        }
    } catch (const std::exception& e) {
        pImpl->m_archive_.reset();
        INSPIRE_LOGE("Exception during resource reloading: %s", e.what());
        return HERR_ARCHIVE_LOAD_MODEL_FAILURE;
    }
}

bool Launch::isMLoad() const {
    return pImpl->m_load_;
}

void Launch::Unload() {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    if (pImpl->m_load_) {
        pImpl->m_archive_.reset();
        pImpl->m_load_ = false;
        INSPIRE_LOGI("All resources have been successfully unloaded and system is reset.");
    } else {
        INSPIRE_LOGW("Unload called but system was not loaded.");
    }
}

void Launch::SetRockchipDmaHeapPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    pImpl->m_rockchip_dma_heap_path_ = path;
}

std::string Launch::GetRockchipDmaHeapPath() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    return pImpl->m_rockchip_dma_heap_path_;
}

void Launch::ConfigurationExtensionPath(const std::string& path) {
#if defined(ISF_ENABLE_APPLE_EXTENSION)
    INSPIREFACE_CHECK_MSG(os::IsDir(path), "The apple extension path is not a directory, please check.");
#endif
    INSPIREFACE_CHECK_MSG(os::IsExists(path), "The extension path is not exists, please check.");
    pImpl->m_extension_path_ = path;
}

std::string Launch::GetExtensionPath() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    return pImpl->m_extension_path_;
}

void Launch::SetGlobalCoreMLInferenceMode(NNInferenceBackend mode) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    if (mode == NN_INFERENCE_CPU) {
        pImpl->m_global_coreml_inference_mode_ = InferenceWrapper::COREML_CPU;
    } else if (mode == NN_INFERENCE_COREML_GPU) {
        pImpl->m_global_coreml_inference_mode_ = InferenceWrapper::COREML_GPU;
    } else if (mode == NN_INFERENCE_COREML_ANE) {
        pImpl->m_global_coreml_inference_mode_ = InferenceWrapper::COREML_ANE;
    } else {
        INSPIRE_LOGE("Invalid CoreML inference mode");
    }
    if (pImpl->m_global_coreml_inference_mode_ == InferenceWrapper::COREML_CPU) {
        INSPIRE_LOGW("Global CoreML Compute Units set to CPU Only.");
    } else if (pImpl->m_global_coreml_inference_mode_ == InferenceWrapper::COREML_GPU) {
        INSPIRE_LOGW("Global CoreML Compute Units set to CPU and GPU.");
    } else if (pImpl->m_global_coreml_inference_mode_ == InferenceWrapper::COREML_ANE) {
        INSPIRE_LOGW("Global CoreML Compute Units set to Auto Switch (ANE, GPU, CPU).");
    }
}

Launch::NNInferenceBackend Launch::GetGlobalCoreMLInferenceMode() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    if (pImpl->m_global_coreml_inference_mode_ == InferenceWrapper::COREML_CPU) {
        return NN_INFERENCE_CPU;
    } else if (pImpl->m_global_coreml_inference_mode_ == InferenceWrapper::COREML_GPU) {
        return NN_INFERENCE_COREML_GPU;
    } else if (pImpl->m_global_coreml_inference_mode_ == InferenceWrapper::COREML_ANE) {
        return NN_INFERENCE_COREML_ANE;
    } else {
        INSPIRE_LOGE("Invalid CoreML inference mode");
        return NN_INFERENCE_CPU;
    }
}

void Launch::BuildAppleExtensionPath(const std::string& resource_path) {
    std::string basename = os::Basename(resource_path);
    pImpl->m_extension_path_ = os::PathJoin(os::Dirname(resource_path), basename + APPLE_EXTENSION_SUFFIX);
    INSPIREFACE_CHECK_MSG(os::IsExists(pImpl->m_extension_path_), "The apple extension path is not exists, please check.");
    INSPIREFACE_CHECK_MSG(os::IsDir(pImpl->m_extension_path_), "The apple extension path is not a directory, please check.");
}

void Launch::SetCudaDeviceId(int32_t device_id) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    pImpl->m_cuda_device_id_ = device_id;
}

int32_t Launch::GetCudaDeviceId() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    return pImpl->m_cuda_device_id_;
}

void Launch::SetFaceDetectPixelList(const std::vector<int32_t>& pixel_list) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    pImpl->m_face_detect_pixel_list_ = pixel_list;
}

std::vector<int32_t> Launch::GetFaceDetectPixelList() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    return pImpl->m_face_detect_pixel_list_;
}

void Launch::SetFaceDetectModelList(const std::vector<std::string>& model_list) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    pImpl->m_face_detect_model_list_ = model_list;
}

std::vector<std::string> Launch::GetFaceDetectModelList() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    return pImpl->m_face_detect_model_list_;
}

void Launch::SwitchLandmarkEngine(LandmarkEngine engine) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    if (pImpl->m_archive_->QueryStatus() != SARC_SUCCESS) {
        INSPIRE_LOGE("The InspireFace is not initialized, please call launch first.");
        return;
    }
    auto landmark_param = pImpl->m_archive_->GetLandmarkParam();
    bool ret = false;
    if (engine == LANDMARK_HYPLMV2_0_25) {
        ret = landmark_param->ReLoad("landmark");
    } else if (engine == LANDMARK_HYPLMV2_0_50) {
        ret = landmark_param->ReLoad("landmark_0_50");
    } else if (engine == LANDMARK_INSIGHTFACE_2D106_TRACK) {
        ret = landmark_param->ReLoad("landmark_insightface_2d106");
    }
    INSPIREFACE_CHECK_MSG(ret, "Failed to switch landmark engine");
}

void Launch::SwitchImageProcessingBackend(ImageProcessingBackend backend) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    if (backend == IMAGE_PROCESSING_RGA) {
#if defined(ISF_ENABLE_RGA)
        pImpl->m_image_processing_backend_ = backend;
#else
        INSPIRE_LOGE("RKRGA is not enabled, please check the build configuration.");
#endif  // ISF_ENABLE_RGA
        return;
    } else {
        pImpl->m_image_processing_backend_ = backend;
    }
}

Launch::ImageProcessingBackend Launch::GetImageProcessingBackend() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    return pImpl->m_image_processing_backend_;
}

void Launch::SetImageProcessAlignedWidth(int32_t width) {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    pImpl->m_image_process_aligned_width_ = width;
}

int32_t Launch::GetImageProcessAlignedWidth() const {
    std::lock_guard<std::mutex> lock(pImpl->mutex_);
    return pImpl->m_image_process_aligned_width_;
}

}  // namespace inspire