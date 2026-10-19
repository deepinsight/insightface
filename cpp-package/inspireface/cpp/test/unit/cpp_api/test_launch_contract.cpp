#include <string>
#include <vector>

#include <inspireface/include/inspireface/inspireface.hpp>

#include "settings/test_settings.h"

namespace {

class LaunchConfigurationReset {
public:
    explicit LaunchConfigurationReset(const std::shared_ptr<inspire::Launch>& launch)
    : launch_(launch),
      rockchip_path_(launch->GetRockchipDmaHeapPath()),
      coreml_mode_(launch->GetGlobalCoreMLInferenceMode()),
      cpu_engine_mode_(launch->GetGlobalCPUEnginePowerMode()),
      cuda_device_id_(launch->GetCudaDeviceId()),
      detect_pixels_(launch->GetFaceDetectPixelList()),
      detect_models_(launch->GetFaceDetectModelList()),
      image_backend_(launch->GetImageProcessingBackend()),
      aligned_width_(launch->GetImageProcessAlignedWidth()) {}

    ~LaunchConfigurationReset() {
        launch_->SetRockchipDmaHeapPath(rockchip_path_);
        launch_->SetGlobalCoreMLInferenceMode(coreml_mode_);
        launch_->SetGlobalCPUEnginePowerMode(cpu_engine_mode_);
        launch_->SetCudaDeviceId(cuda_device_id_);
        launch_->SetFaceDetectPixelList(detect_pixels_);
        launch_->SetFaceDetectModelList(detect_models_);
        launch_->SwitchImageProcessingBackend(image_backend_);
        launch_->SetImageProcessAlignedWidth(aligned_width_);
    }

private:
    std::shared_ptr<inspire::Launch> launch_;
    std::string rockchip_path_;
    inspire::Launch::NNInferenceBackend coreml_mode_;
    inspire::Launch::CPUEnginePowerMode cpu_engine_mode_;
    int32_t cuda_device_id_;
    std::vector<int32_t> detect_pixels_;
    std::vector<std::string> detect_models_;
    inspire::Launch::ImageProcessingBackend image_backend_;
    int32_t aligned_width_;
};

}  // namespace

TEST_CASE("C++ Launch singleton exposes a stable loaded resource", "[cpp_api][contract][launch]") {
    const auto first = inspire::Launch::GetInstance();
    const auto second = inspire::Launch::GetInstance();
    REQUIRE(first);
    CHECK(first == second);
    REQUIRE(first->isMLoad());
    CHECK_NOTHROW(first->getMArchive());

    CHECK(first->Load(GET_RUNTIME_FULLPATH_NAME) == HSUCCEED);
    CHECK(first->isMLoad());
}

TEST_CASE("C++ Launch failed reload preserves the active resource", "[cpp_api][contract][launch][boundary]") {
    const auto launch = inspire::Launch::GetInstance();
    REQUIRE(launch->isMLoad());
    const auto pixels = launch->GetFaceDetectPixelList();
    const auto models = launch->GetFaceDetectModelList();

    CHECK(launch->Reload(GET_DATA("missing/does-not-exist.pack")) == HERR_ARCHIVE_LOAD_FAILURE);
    CHECK(launch->isMLoad());
    CHECK(launch->GetFaceDetectPixelList() == pixels);
    CHECK(launch->GetFaceDetectModelList() == models);
    CHECK_NOTHROW(launch->getMArchive());
}

TEST_CASE("C++ Launch configuration getters round-trip and restore global state", "[cpp_api][contract][launch][configuration]") {
    const auto launch = inspire::Launch::GetInstance();
    LaunchConfigurationReset reset(launch);

    launch->SetRockchipDmaHeapPath("/contract/dma_heap");
    CHECK(launch->GetRockchipDmaHeapPath() == "/contract/dma_heap");

    launch->SetGlobalCoreMLInferenceMode(inspire::Launch::NN_INFERENCE_CPU);
    CHECK(launch->GetGlobalCoreMLInferenceMode() == inspire::Launch::NN_INFERENCE_CPU);
    launch->SetGlobalCoreMLInferenceMode(inspire::Launch::NN_INFERENCE_COREML_GPU);
    CHECK(launch->GetGlobalCoreMLInferenceMode() == inspire::Launch::NN_INFERENCE_COREML_GPU);
    launch->SetGlobalCoreMLInferenceMode(inspire::Launch::NN_INFERENCE_COREML_ANE);
    CHECK(launch->GetGlobalCoreMLInferenceMode() == inspire::Launch::NN_INFERENCE_COREML_ANE);

    launch->SetCudaDeviceId(7);
    CHECK(launch->GetCudaDeviceId() == 7);

    const std::vector<int32_t> pixels = {128, 256, 512};
    const std::vector<std::string> models = {"detect_128", "detect_256", "detect_512"};
    launch->SetFaceDetectPixelList(pixels);
    launch->SetFaceDetectModelList(models);
    CHECK(launch->GetFaceDetectPixelList() == pixels);
    CHECK(launch->GetFaceDetectModelList() == models);

    launch->SwitchImageProcessingBackend(inspire::Launch::IMAGE_PROCESSING_CPU);
    CHECK(launch->GetImageProcessingBackend() == inspire::Launch::IMAGE_PROCESSING_CPU);
    launch->SetImageProcessAlignedWidth(16);
    CHECK(launch->GetImageProcessAlignedWidth() == 16);
}

TEST_CASE("C++ Launch CPU engine power modes default to Normal and support inference", "[cpp_api][contract][launch][configuration][cpu_engine]") {
    using Launch = inspire::Launch;
    const auto launch = Launch::GetInstance();
    LaunchConfigurationReset reset(launch);
    REQUIRE(launch->GetGlobalCPUEnginePowerMode() == Launch::CPU_ENGINE_POWER_NORMAL);

    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    inspire::CustomPipelineParameter parameter;
    parameter.enable_recognition = true;
    std::vector<float> reference;

    for (const auto mode : {Launch::CPU_ENGINE_POWER_NORMAL, Launch::CPU_ENGINE_POWER_HIGH, Launch::CPU_ENGINE_POWER_LOW}) {
        CAPTURE(mode);
        REQUIRE(launch->SetGlobalCPUEnginePowerMode(mode) == HSUCCEED);
        CHECK(launch->GetGlobalCPUEnginePowerMode() == mode);
        CHECK(launch->SetGlobalCPUEnginePowerMode(static_cast<Launch::CPUEnginePowerMode>(-1)) == HERR_INVALID_PARAM);
        CHECK(launch->SetGlobalCPUEnginePowerMode(static_cast<Launch::CPUEnginePowerMode>(99)) == HERR_INVALID_PARAM);
        CHECK(launch->GetGlobalCPUEnginePowerMode() == mode);

        auto session = inspire::Session::Create(inspire::DETECT_MODE_ALWAYS_DETECT, 1, parameter);
        std::vector<inspire::FaceTrackWrap> faces;
        REQUIRE(session.FaceDetectAndTrack(process, faces) == HSUCCEED);
        REQUIRE(faces.size() == 1);
        inspire::FaceEmbedding feature{};
        REQUIRE(session.FaceFeatureExtract(process, faces[0], feature, true) == HSUCCEED);
        REQUIRE(!feature.embedding.empty());
        if (reference.empty()) {
            reference = feature.embedding;
        } else {
            float similarity = 0.0f;
            REQUIRE(inspire::FeatureHubDB::CosineSimilarity(reference, feature.embedding, similarity, true) == HSUCCEED);
            CHECK(similarity == Approx(1.0f).margin(1e-5f));
        }
    }
}
