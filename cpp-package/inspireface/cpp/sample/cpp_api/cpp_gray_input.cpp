#include <iostream>
#include <inspirecv/inspirecv.h>
#include <inspireface/inspireface.hpp>

int main() {
    INSPIREFACE_CONTEXT->Reload("test_res/pack/Pikachu");
    // Create session
    inspire::CustomPipelineParameter param;
    param.enable_recognition = true;
    param.enable_liveness = true;
    param.enable_mask_detect = true;
    param.enable_face_attribute = true;
    param.enable_face_quality = true;
    std::shared_ptr<inspire::Session> session(inspire::Session::CreatePtr(inspire::DETECT_MODE_ALWAYS_DETECT, 1, param, 320));
    // Prepare image
    inspirecv::Image img = inspirecv::Image::Create("data.jpg", 3);
    inspirecv::Image gray_img = img.ToGray();

    // Create image and frame process
    inspirecv::FrameProcess process =
      inspirecv::FrameProcess::Create(gray_img.Data(), gray_img.Height(), gray_img.Width(), inspirecv::GRAY, inspirecv::ROTATION_0);
    auto decode = process.ExecutePreviewImageProcessing(true);
    decode.Write("decode.jpg");

    // Detect
    std::vector<inspire::FaceTrackWrap> results;
    int32_t ret;
    ret = session->FaceDetectAndTrack(process, results);
    std::cout << "Face size: " << results.size() << std::endl;
    for (const auto& result : results) {
        // Draw face
        inspirecv::Rect2i rect = inspirecv::Rect2i::Create(result.rect.x, result.rect.y, result.rect.width, result.rect.height);
        img.DrawRect(rect, inspirecv::Color::Red);
    }
    img.Write("result.jpg");

    return 0;
}