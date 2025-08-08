#include <iostream>
#include <vector>
#include <string>
#include <memory>
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
      inspirecv::FrameProcess::Create(image.Data(), image.Height(), image.Width(), inspirecv::BGR, inspirecv::ROTATION_90);

    // Create session
    inspire::CustomPipelineParameter param;
    param.enable_recognition = true;
    param.enable_liveness = true;
    param.enable_mask_detect = true;
    param.enable_face_attribute = true;
    param.enable_face_quality = true;
    std::shared_ptr<inspire::Session> session(inspire::Session::CreatePtr(inspire::DETECT_MODE_ALWAYS_DETECT, 100, param, 640));
    session->SetTrackPreviewSize(640);

    INSPIREFACE_CHECK_MSG(session != nullptr, "Session is not valid");

    // Detect and track
    std::vector<inspire::FaceTrackWrap> results;
    int32_t ret;
    ret = session->FaceDetectAndTrack(process, results);
    INSPIREFACE_CHECK_MSG(ret == 0, "FaceDetectAndTrack failed");

    // Run pipeline for each face
    ret = session->MultipleFacePipelineProcess(process, param, results);
    INSPIREFACE_CHECK_MSG(ret == 0, "MultipleFacePipelineProcess failed");

    for (auto& result : results) {
        std::cout << "result: " << result.trackId << std::endl;
        std::cout << "quality: " << result.quality[0] << ", " << result.quality[1] << ", " << result.quality[2] << ", " << result.quality[3] << ", "
                  << result.quality[4] << std::endl;
        inspirecv::Rect2i rect = inspirecv::Rect2i::Create(result.rect.x, result.rect.y, result.rect.width, result.rect.height);
        std::cout << rect << std::endl;
        image.DrawRect(rect, inspirecv::Color::Red);
        inspirecv::TransformMatrix trans = inspirecv::TransformMatrix::Create(result.trans.m00, result.trans.m01, result.trans.tx, result.trans.m10, result.trans.m11, result.trans.ty);
        std::cout << "trans: " << trans.GetInverse() << std::endl;

        std::vector<inspirecv::Point2f> landmark = session->GetFaceDenseLandmark(result);
        for (auto& point : landmark) {
            image.DrawCircle(point.As<int>(), 2, inspirecv::Color::Green);
        }
    }
    image.Write("result.jpg");

    return 0;
}
