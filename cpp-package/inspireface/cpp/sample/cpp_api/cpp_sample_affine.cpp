#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <inspirecv/inspirecv.h>
#include <inspireface/inspireface.hpp>
#include "inspireface/track_module/landmark/order_of_hyper_landmark.h"

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
    param.enable_interaction_liveness = true;
    std::shared_ptr<inspire::Session> session(inspire::Session::CreatePtr(inspire::DETECT_MODE_ALWAYS_DETECT, 1, param, 320));

    INSPIREFACE_CHECK_MSG(session != nullptr, "Session is not valid");

    // Detect and track
    std::vector<inspire::FaceTrackWrap> results;
    int32_t ret;
    ret = session->FaceDetectAndTrack(process, results);
    INSPIREFACE_CHECK_MSG(ret == 0, "FaceDetectAndTrack failed");

    auto first = results[0];
    auto lmk = session->GetFaceDenseLandmark(first);
    std::cout << "lmk: " << lmk.size() << std::endl;
    for (size_t i = 0; i < lmk.size(); i++) {
        image.DrawCircle(lmk[i].As<int>(), 5, inspirecv::Color::Red);
    }

    inspirecv::TransformMatrix rotation_mode_affine = process.GetAffineMatrix();

    std::vector<inspirecv::Point2f> stand_lmk = ApplyTransformToPoints(lmk, rotation_mode_affine.GetInverse());

    // Use total lmk
    auto rect = inspirecv::MinBoundingRect(stand_lmk);
    auto rect_pts = rect.As<float>().ToFourVertices();
    std::vector<inspirecv::Point2f> dst_pts = {{0, 0}, {112, 0}, {112, 112}, {0, 112}};
    std::vector<inspirecv::Point2f> camera_pts = ApplyTransformToPoints(rect_pts, rotation_mode_affine);

    auto affine = inspirecv::SimilarityTransformEstimate(camera_pts, dst_pts);
    auto image_affine = process.ExecuteImageAffineProcessing(affine, 112, 112);
    image_affine.Write("affine.jpg");

    // image.DrawRect(rect.As<int>(), inspirecv::Color::Red);
    // image.Write("lmk.jpg");

    std::vector<inspirecv::Point2i> points;
    for (const auto& idx : inspire::HLMK_LEFT_EYE_POINTS_INDEX) {
        points.emplace_back(stand_lmk[idx].GetX(), stand_lmk[idx].GetY());
    }
    std::cout << "points: " << points.size() << std::endl;
    auto rect_eye = inspirecv::MinBoundingRect(points).Square(1.4f);
    // draw debug
    image.DrawRect(rect_eye.As<int>(), inspirecv::Color::Red);
    auto rect_pts_eye = rect_eye.As<float>().ToFourVertices();
    std::vector<inspirecv::Point2f> dst_pts_eye = {{0, 0}, {64, 0}, {64, 64}, {0, 64}};
    std::vector<inspirecv::Point2f> camera_pts_eye = ApplyTransformToPoints(rect_pts_eye, rotation_mode_affine);

    auto affine_eye = inspirecv::SimilarityTransformEstimate(camera_pts_eye, dst_pts_eye);
    auto eye_affine = process.ExecuteImageAffineProcessing(affine_eye, 64, 64);
    eye_affine.Write("eye.jpg");
    return 0;
}
