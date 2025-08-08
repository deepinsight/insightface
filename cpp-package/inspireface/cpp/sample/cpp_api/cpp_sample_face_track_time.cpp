#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <inspirecv/inspirecv.h>
#include <inspireface/inspireface.hpp>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    // Global init(only once)
    INSPIREFACE_CONTEXT->Reload(model_path);

    // Create image and frame process
    inspirecv::Image image = inspirecv::Image::Create(image_path);
    inspirecv::FrameProcess process =
      inspirecv::FrameProcess::Create(image.Data(), image.Height(), image.Width(), inspirecv::BGR, inspirecv::ROTATION_0);

    // Create session
    inspire::CustomPipelineParameter param;
    param.enable_recognition = true;
    param.enable_liveness = true;
    param.enable_mask_detect = true;
    param.enable_face_attribute = true;
    param.enable_face_quality = true;
    std::shared_ptr<inspire::Session> session(inspire::Session::CreatePtr(inspire::DETECT_MODE_LIGHT_TRACK, 100, param, 640));
    session->SetTrackPreviewSize(640);
    session->SetTrackModeDetectInterval(10);

    INSPIREFACE_CHECK_MSG(session != nullptr, "Session is not valid");


    for (int i = 0; i < 100; i++) {
        
        // Detect and track
        std::vector<inspire::FaceTrackWrap> results;
        int32_t ret;
        // Start time
        auto start = std::chrono::high_resolution_clock::now();
        ret = session->FaceDetectAndTrack(process, results);
        INSPIREFACE_CHECK_MSG(ret == 0, "FaceDetectAndTrack failed");
        auto end = std::chrono::high_resolution_clock::now();
        // End time
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << i << " MultipleFacePipelineProcess: " << duration.count() / 1000.0 << " ms" << std::endl;

    }
    

    return 0;
}
