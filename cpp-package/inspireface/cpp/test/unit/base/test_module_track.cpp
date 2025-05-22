
#include <iostream>
#include "settings/test_settings.h"
#include "unit/test_helper/help.h"
#include <inspireface/include/inspireface/feature_hub_db.h>
#include "middleware/costman.h"
#include "track_module/face_detect/all.h"
#include "track_module/face_track_module.h"
#include <inspireface/include/inspireface/frame_process.h>

using namespace inspire;

TEST_CASE("test_FaceDetect", "[track_module") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);
    auto archive = INSPIREFACE_CONTEXT->getMArchive();
    const std::vector<int32_t> supported_sizes = {160, 320, 640};
    const std::vector<std::string> scheme_names = {"face_detect_160", "face_detect_320", "face_detect_640"};
    for (size_t i = 0; i < scheme_names.size(); i++) {
        InspireModel model;
        auto ret = archive.LoadModel(scheme_names[i], model);
        REQUIRE(ret == 0);
        FaceDetectAdapt face_detector(supported_sizes[i]);
        face_detector.LoadData(model, model.modelType, false);

        inspirecv::Image img = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
        auto result = face_detector(img);
        REQUIRE(result.size() == 1);
    }
}

TEST_CASE("test_RefineNet", "[track_module") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);
    auto archive = INSPIREFACE_CONTEXT->getMArchive();
    InspireModel model;
    auto ret = archive.LoadModel("refine_net", model);
    REQUIRE(ret == 0);
    RNetAdapt rnet;
    rnet.LoadData(model, model.modelType, false);

    inspirecv::Image face = inspirecv::Image::Create(GET_DATA("data/crop/crop.png"));
    auto result1 = rnet(face);
    REQUIRE(result1 > 0.5f);

    inspirecv::Image no_face = inspirecv::Image::Create(GET_DATA("data/crop/no_face.png"));
    auto result2 = rnet(no_face);
    REQUIRE(result2 < 0.5f);
}

TEST_CASE("test_Landmark", "[track_module") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);
    auto archive = INSPIREFACE_CONTEXT->getMArchive();
    InspireModel model;
    auto ret = archive.LoadModel("landmark", model);
    REQUIRE(ret == 0);

    FaceLandmarkAdapt face_landmark(112);
    face_landmark.LoadData(model, model.modelType);

    inspirecv::Image img = inspirecv::Image::Create(GET_DATA("data/crop/crop.png"));
    auto result = face_landmark(img);
    REQUIRE(result.size() == 106 * 2);
}

TEST_CASE("test_Quality", "[track_module") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);
    auto archive = INSPIREFACE_CONTEXT->getMArchive();
    InspireModel model;
    auto ret = archive.LoadModel("pose_quality", model);
    REQUIRE(ret == 0);
    FacePoseQualityAdapt quality;
    ret = quality.LoadData(model, model.modelType);
    REQUIRE(ret == 0);

    inspirecv::Image img = inspirecv::Image::Create(GET_DATA("data/crop/crop.png"));
    auto result = quality(img);
    REQUIRE(result.lmk.size() == 5);
    REQUIRE(result.lmk_quality.size() == 5);
}

TEST_CASE("test_FaceTrackModule", "[track_module") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);
    auto archive = INSPIREFACE_CONTEXT->getMArchive();

    SECTION("Test face detect rotate 0") {
        auto mode = DetectModuleMode::DETECT_MODE_ALWAYS_DETECT;
        int max_detected_faces = 10;
        FaceTrackModule face_track(mode, max_detected_faces);
        face_track.Configuration(archive);
        inspirecv::Image img = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
        inspirecv::FrameProcess image = inspirecv::FrameProcess::Create(img.Data(), img.Height(), img.Width(), inspirecv::BGR);
        face_track.UpdateStream(image);
        REQUIRE(face_track.trackingFace.size() == 1);
    }

    SECTION("Test face detect rotate 90") {
        auto mode = DetectModuleMode::DETECT_MODE_ALWAYS_DETECT;
        int max_detected_faces = 10;
        FaceTrackModule face_track(mode, max_detected_faces);
        face_track.Configuration(archive);
        inspirecv::Image img = inspirecv::Image::Create(GET_DATA("data/bulk/r90.jpg"));
        inspirecv::FrameProcess image =
          inspirecv::FrameProcess::Create(img.Data(), img.Height(), img.Width(), inspirecv::BGR, inspirecv::ROTATION_90);
        face_track.UpdateStream(image);
        REQUIRE(face_track.trackingFace.size() == 1);
    }
}