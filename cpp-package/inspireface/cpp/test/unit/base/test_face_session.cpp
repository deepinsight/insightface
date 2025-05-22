
#include <iostream>
#include "settings/test_settings.h"
#include "unit/test_helper/help.h"
#include "middleware/costman.h"
#include <inspireface/include/inspireface/launch.h>
#include "frame_process.h"
#include "inspireface/engine/face_session.h"
#include <inspireface/include/inspireface/feature_hub_db.h>

using namespace inspire;

TEST_CASE("test_FaceSession", "[face_session") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    int32_t ret;
    CustomPipelineParameter param;
    param.enable_recognition = true;
    param.enable_liveness = true;
    param.enable_mask_detect = true;
    param.enable_face_attribute = true;
    param.enable_face_quality = true;

    FaceSession session;
    ret = session.Configuration(DetectModuleMode::DETECT_MODE_ALWAYS_DETECT, 1, param);
    REQUIRE(ret == HSUCCEED);

    inspirecv::Image kun1 = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    inspirecv::Image kun2 = inspirecv::Image::Create(GET_DATA("data/bulk/jntm.jpg"));
    inspirecv::FrameProcess proc1 = inspirecv::FrameProcess::Create(kun1.Data(), kun1.Height(), kun1.Width(), inspirecv::BGR, inspirecv::ROTATION_0);
    inspirecv::FrameProcess proc2 = inspirecv::FrameProcess::Create(kun2.Data(), kun2.Height(), kun2.Width(), inspirecv::BGR, inspirecv::ROTATION_0);
    std::vector<std::vector<float>> features;
    std::vector<inspirecv::FrameProcess> processes = {proc1, proc2};
    for (auto &process : processes) {
        ret = session.FaceDetectAndTrack(process);
        REQUIRE(ret == HSUCCEED);
        if (session.GetDetectCache().size() > 0) {
            FaceBasicData data = session.GetFaceBasicDataCache()[0];
            ret = session.FaceFeatureExtract(process, data);
            REQUIRE(ret == HSUCCEED);
            const auto &faces = session.GetTrackingFaceList();
            REQUIRE(faces.size() > 0);
            Embedded feature;
            FaceTrackWrap hyper_face_data = FaceObjectInternalToHyperFaceData(faces[0]);
            float norm;
            ret = session.FaceRecognitionModule()->FaceExtract(process, hyper_face_data, feature, norm);
            REQUIRE(ret == HSUCCEED);
            features.push_back(feature);
        }
    }
    REQUIRE(features.size() == 2);
    float res;
    ret = FeatureHubDB::CosineSimilarity(features[0].data(), features[1].data(), features[0].size(), res);
    REQUIRE(ret == HSUCCEED);
    REQUIRE(res > 0.5f);

    inspirecv::Image other = inspirecv::Image::Create(GET_DATA("data/bulk/woman.png"));
    inspirecv::FrameProcess proc3 =
      inspirecv::FrameProcess::Create(other.Data(), other.Height(), other.Width(), inspirecv::BGR, inspirecv::ROTATION_0);
    ret = session.FaceDetectAndTrack(proc3);
    REQUIRE(ret == HSUCCEED);
    if (session.GetDetectCache().size() > 0) {
        FaceBasicData data = session.GetFaceBasicDataCache()[0];
        ret = session.FaceFeatureExtract(proc3, data);
        auto faces = session.GetTrackingFaceList();
        REQUIRE(ret == HSUCCEED);
        Embedded feature;
        FaceTrackWrap hyper_face_data = FaceObjectInternalToHyperFaceData(faces[0]);
        float norm;
        ret = session.FaceRecognitionModule()->FaceExtract(proc3, hyper_face_data, feature, norm);
        REQUIRE(ret == HSUCCEED);
        features.push_back(feature);
    }
    REQUIRE(features.size() == 3);
    float other_v_kun1, other_v_kun2;
    ret = FeatureHubDB::CosineSimilarity(features[0].data(), features[2].data(), features[0].size(), other_v_kun1);
    REQUIRE(ret == HSUCCEED);
    REQUIRE(other_v_kun1 < 0.5f);
    ret = FeatureHubDB::CosineSimilarity(features[1].data(), features[2].data(), features[0].size(), other_v_kun2);
    REQUIRE(ret == HSUCCEED);
    REQUIRE(other_v_kun2 < 0.5f);
}