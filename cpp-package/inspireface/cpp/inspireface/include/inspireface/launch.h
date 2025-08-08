/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIREFACE_LAUNCH_H
#define INSPIREFACE_LAUNCH_H

#include <memory>
#include <string>
#include <cstdint>
#include "data_type.h"

#define INSPIREFACE_CONTEXT inspire::Launch::GetInstance()

namespace inspire {

// Forward declarations
class InspireArchive;

// The Launch class acts as the main entry point for the InspireFace system.
// It is responsible for loading static resources such as models, configurations, and parameters.
class INSPIRE_API_EXPORT Launch {
public:
    // Special Backend enum for CoreML
    enum NNInferenceBackend {
        NN_INFERENCE_CPU = 0,
        NN_INFERENCE_MMM_CUDA,
        NN_INFERENCE_COREML_CPU,
        NN_INFERENCE_COREML_GPU,
        NN_INFERENCE_COREML_ANE,
        NN_INFERENCE_TENSORRT_CUDA,
    };

    // Landmark engine enum
    enum LandmarkEngine {
        LANDMARK_HYPLMV2_0_25 = 0,
        LANDMARK_HYPLMV2_0_50,
        LANDMARK_INSIGHTFACE_2D106_TRACK,
    };

    // Image processing backend engine
    enum ImageProcessingBackend {
        IMAGE_PROCESSING_CPU = 0,  // CPU backend(Default)
        IMAGE_PROCESSING_RGA,      // Rockchip RGA backend(Hardware support is mandatory)
    };

    Launch(const Launch&) = delete;             // Delete the copy constructor to prevent copying.
    Launch& operator=(const Launch&) = delete;  // Delete the assignment operator to prevent assignment.
    ~Launch();                                  // Destructor needs to be defined where the implementation is complete

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
    void SetGlobalCoreMLInferenceMode(NNInferenceBackend mode);

    // Get the global coreml inference mode
    NNInferenceBackend GetGlobalCoreMLInferenceMode() const;

    // Build the extension path
    void BuildAppleExtensionPath(const std::string& resource_path);

    // Set the cuda device id
    void SetCudaDeviceId(int32_t device_id);

    // Get the cuda device id
    int32_t GetCudaDeviceId() const;

    // Set the face detect pixel list
    void SetFaceDetectPixelList(const std::vector<int32_t>& pixel_list);

    // Get the face detect pixel list
    std::vector<int32_t> GetFaceDetectPixelList() const;

    // Set the face detect model list
    void SetFaceDetectModelList(const std::vector<std::string>& model_list);

    // Get the face detect model list
    std::vector<std::string> GetFaceDetectModelList() const;

    // Switch the landmark engine(It must be used before creating a session)
    void SwitchLandmarkEngine(LandmarkEngine engine);

    // Switch the image processing backend(It must be used before creating a session)
    void SwitchImageProcessingBackend(ImageProcessingBackend backend);

    // Get the image processing backend
    ImageProcessingBackend GetImageProcessingBackend() const;

    // Set the ImageProcess Aligned Width(It must be used before creating a session)
    void SetImageProcessAlignedWidth(int32_t width);

    // Get the ImageProcess Aligned Width
    int32_t GetImageProcessAlignedWidth() const;

private:
    // Private constructor for the singleton pattern
    Launch();

    // Private implementation class
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

}  // namespace inspire

#endif  // INSPIREFACE_LAUNCH_H