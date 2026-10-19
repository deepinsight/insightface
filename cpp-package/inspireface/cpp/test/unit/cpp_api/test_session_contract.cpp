#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include <inspireface/include/inspireface/inspireface.hpp>

#include "settings/test_settings.h"

using inspire::CustomPipelineParameter;
using inspire::FaceEmbedding;
using inspire::FaceTrackWrap;
using inspire::Session;

TEST_CASE("C++ Session reports an unconfigured default instance without leaking stale results", "[cpp_api][contract][session][boundary]") {
    static_assert(!std::is_copy_constructible<Session>::value, "Session must remain non-copyable");
    static_assert(std::is_move_constructible<Session>::value, "Session must remain movable");

    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);

    Session session;
    session.ClearTrackingFace();
    session.SetTrackLostRecoveryMode(true);
    session.SetLightTrackConfidenceThreshold(std::numeric_limits<float>::quiet_NaN());
    session.SetTrackPreviewSize(0);
    session.SetFilterMinimumFacePixelSize(-1);
    session.SetFaceDetectThreshold(std::numeric_limits<float>::infinity());
    session.SetTrackModeSmoothRatio(0.25f);
    session.SetTrackModeNumSmoothCacheFrame(0);
    session.SetTrackModeDetectInterval(0);
    std::vector<FaceTrackWrap> faces(1);
    CHECK(session.FaceDetectAndTrack(process, faces) == HERR_SESS_INVALID_RESOURCE);
    CHECK(faces.empty());

    FaceTrackWrap face = {};
    FaceEmbedding embedding = {1, 7.0f, std::vector<float>(512, 1.0f)};
    CHECK(session.FaceFeatureExtract(process, face, embedding) == HERR_SESS_INVALID_RESOURCE);
    CHECK(embedding.isNormal == 0);
    CHECK(embedding.norm == 0.0f);
    CHECK(embedding.embedding.empty());
    CHECK(session.MultipleFacePipelineProcess(process, CustomPipelineParameter(), faces) == HERR_SESS_INVALID_RESOURCE);
    CHECK(session.GetRGBLivenessConfidence().empty());
    CHECK(session.GetFaceMaskConfidence().empty());
    CHECK(session.GetFaceQualityConfidence().empty());
    CHECK(session.GetFaceInteractionState().empty());
    CHECK(session.GetFaceInteractionAction().empty());
    CHECK(session.GetFaceAttributeResult().empty());
    CHECK(session.GetFaceEmotionResult().empty());
}

TEST_CASE("C++ Session creation rejects invalid configuration transactionally", "[cpp_api][contract][session][boundary]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    const CustomPipelineParameter parameter;

    std::array<Session, 3> invalid_sessions = {
      Session::Create(static_cast<inspire::DetectModuleMode>(99), 1, parameter),
      Session::Create(inspire::DETECT_MODE_ALWAYS_DETECT, 0, parameter),
      Session::Create(inspire::DETECT_MODE_TRACK_BY_DETECT, 1, parameter, -1, 0)};
    for (auto& session : invalid_sessions) {
        std::vector<FaceTrackWrap> faces(1);
        CHECK(session.FaceDetectAndTrack(process, faces) == HERR_INVALID_PARAM);
        CHECK(faces.empty());
    }
}

TEST_CASE("C++ Session reports unsupported IR requests explicitly", "[cpp_api][contract][session][unsupported][ir]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);

    CustomPipelineParameter ir_parameter;
    ir_parameter.enable_ir_liveness = true;
    Session ir_session = Session::Create(inspire::DETECT_MODE_ALWAYS_DETECT, 1, ir_parameter);
    std::vector<FaceTrackWrap> faces;
    CHECK(ir_session.FaceDetectAndTrack(process, faces) == HERR_UNSUPPORTED);
    CHECK(faces.empty());

    Session session = Session::Create(inspire::DETECT_MODE_ALWAYS_DETECT, 1, CustomPipelineParameter());
    REQUIRE(session.FaceDetectAndTrack(process, faces) == HSUCCEED);
    REQUIRE(!faces.empty());
    CHECK(session.MultipleFacePipelineProcess(process, ir_parameter, faces) == HERR_UNSUPPORTED);
}

TEST_CASE("C++ Session move ownership retains detection and landmark behavior", "[cpp_api][contract][session][face_track]") {
    CustomPipelineParameter parameter;
    Session original = Session::Create(inspire::DETECT_MODE_ALWAYS_DETECT, 3, parameter);
    Session session(std::move(original));
    session.SetTrackPreviewSize(320);
    session.SetTrackLostRecoveryMode(true);
    session.SetLightTrackConfidenceThreshold(0.1f);
    session.SetFilterMinimumFacePixelSize(0);
    session.SetFaceDetectThreshold(0.5f);
    session.SetTrackModeSmoothRatio(1);
    session.SetTrackModeNumSmoothCacheFrame(5);
    session.SetTrackModeDetectInterval(20);
    session.ClearTrackingFace();

    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    std::vector<FaceTrackWrap> faces;
    REQUIRE(session.FaceDetectAndTrack(process, faces) == HSUCCEED);
    REQUIRE(faces.size() == 1);

    const auto box = session.GetFaceBoundingBox(faces[0]);
    CHECK(box.GetWidth() > 0);
    CHECK(box.GetHeight() > 0);
    const auto dense = session.GetFaceDenseLandmark(faces[0]);
    const auto five = session.GetFaceFiveKeyPoints(faces[0]);
    REQUIRE(dense.size() == 106);
    REQUIRE(five.size() == 5);
    for (const auto& point : five) {
        CHECK(std::isfinite(point.GetX()));
        CHECK(std::isfinite(point.GetY()));
    }

    const auto no_face = inspirecv::Image::Create(GET_DATA("data/crop/no_face.png"));
    REQUIRE(!no_face.Empty());
    auto no_face_process = inspirecv::FrameProcess::Create(no_face, inspirecv::BGR, inspirecv::ROTATION_0);
    REQUIRE(session.FaceDetectAndTrack(no_face_process, faces) == HSUCCEED);
    CHECK(faces.empty());
}

TEST_CASE("C++ Session CreatePtr transfers explicit ownership to the caller", "[cpp_api][contract][session][lifetime]") {
    CustomPipelineParameter parameter;
    std::unique_ptr<Session> session(Session::CreatePtr(inspire::DETECT_MODE_ALWAYS_DETECT, 1, parameter));
    REQUIRE(session);

    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    std::vector<FaceTrackWrap> faces;
    REQUIRE(session->FaceDetectAndTrack(process, faces) == HSUCCEED);
    CHECK(faces.size() == 1);
}

TEST_CASE("C++ Session feature and pipeline outputs have one result per processed face", "[cpp_api][contract][session][pipeline][face_feature]") {
    CustomPipelineParameter parameter;
    parameter.enable_recognition = true;
    parameter.enable_liveness = true;
    parameter.enable_mask_detect = true;
    parameter.enable_face_quality = true;
    parameter.enable_face_attribute = true;
    parameter.enable_interaction_liveness = true;
    parameter.enable_face_emotion = true;

    Session session = Session::Create(inspire::DETECT_MODE_ALWAYS_DETECT, 3, parameter);
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    std::vector<FaceTrackWrap> faces;
    REQUIRE(session.FaceDetectAndTrack(process, faces) == HSUCCEED);
    REQUIRE(faces.size() == 1);

    FaceEmbedding direct = {};
    REQUIRE(session.FaceFeatureExtract(process, faces[0], direct, true) == HSUCCEED);
    REQUIRE(direct.embedding.size() == 512);
    CHECK(direct.isNormal == 1);
    CHECK(std::isfinite(direct.norm));

    inspirecv::Image alignment;
    session.GetFaceAlignmentImage(process, faces[0], alignment);
    REQUIRE(!alignment.Empty());
    CHECK(alignment.Width() == alignment.Height());

    FaceEmbedding from_image = {};
    REQUIRE(session.FaceFeatureExtractWithAlignmentImage(alignment, from_image, true) == HSUCCEED);
    REQUIRE(from_image.embedding.size() == direct.embedding.size());
    float similarity = 0.0f;
    REQUIRE(inspire::FeatureHubDB::CosineSimilarity(direct.embedding, from_image.embedding, similarity, true) == HSUCCEED);
    CHECK(similarity == Approx(1.0f).margin(1e-4f));

    auto alignment_process = inspirecv::FrameProcess::Create(alignment, inspirecv::BGR, inspirecv::ROTATION_0);
    FaceEmbedding from_process = {};
    REQUIRE(session.FaceFeatureExtractWithAlignmentImage(alignment_process, from_process, true) == HSUCCEED);
    REQUIRE(inspire::FeatureHubDB::CosineSimilarity(direct.embedding, from_process.embedding, similarity, true) == HSUCCEED);
    CHECK(similarity == Approx(1.0f).margin(1e-4f));

    REQUIRE(session.MultipleFacePipelineProcess(process, parameter, faces) == HSUCCEED);
    CHECK(session.GetRGBLivenessConfidence().size() == faces.size());
    CHECK(session.GetFaceMaskConfidence().size() == faces.size());
    CHECK(session.GetFaceQualityConfidence().size() == faces.size());
    CHECK(session.GetFaceInteractionState().size() == faces.size());
    CHECK(session.GetFaceInteractionAction().size() == faces.size());
    CHECK(session.GetFaceAttributeResult().size() == faces.size());
    CHECK(session.GetFaceEmotionResult().size() == faces.size());
}
