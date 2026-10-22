#include <cmath>
#include <vector>

#include <inspireface/include/inspireface/inspireface.hpp>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/test_tools.h"

using inspireface_test::UniqueImageStream;
using inspireface_test::UniqueSession;

TEST_CASE("C and C++ APIs agree on detection landmarks and embeddings", "[cpp_api][contract][parity][consistency]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());

    inspire::CustomPipelineParameter cpp_parameter;
    cpp_parameter.enable_recognition = true;
    inspire::Session cpp_session = inspire::Session::Create(inspire::DETECT_MODE_ALWAYS_DETECT, 1, cpp_parameter);
    auto cpp_process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    std::vector<inspire::FaceTrackWrap> cpp_faces;
    REQUIRE(cpp_session.FaceDetectAndTrack(cpp_process, cpp_faces) == HSUCCEED);
    REQUIRE(cpp_faces.size() == 1);

    HFSession c_session_handle = nullptr;
    REQUIRE(HFCreateInspireFaceSessionOptional(HF_ENABLE_FACE_RECOGNITION, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &c_session_handle) == HSUCCEED);
    UniqueSession c_session(c_session_handle);
    HFImageStream c_stream_handle = nullptr;
    REQUIRE(CVImageToImageStream(image, c_stream_handle) == HSUCCEED);
    UniqueImageStream c_stream(c_stream_handle);
    HFMultipleFaceData c_faces = {};
    REQUIRE(HFExecuteFaceTrack(c_session.Get(), c_stream.Get(), &c_faces) == HSUCCEED);
    REQUIRE(c_faces.detectedNum == 1);

    const auto cpp_box = cpp_session.GetFaceBoundingBox(cpp_faces[0]);
    CHECK(std::abs(cpp_box.GetX() - c_faces.rects[0].x) <= 1);
    CHECK(std::abs(cpp_box.GetY() - c_faces.rects[0].y) <= 1);
    CHECK(std::abs(cpp_box.GetWidth() - c_faces.rects[0].width) <= 1);
    CHECK(std::abs(cpp_box.GetHeight() - c_faces.rects[0].height) <= 1);

    const auto cpp_five = cpp_session.GetFaceFiveKeyPoints(cpp_faces[0]);
    std::vector<HPoint2f> c_five(5);
    REQUIRE(HFGetFaceFiveKeyPointsFromFaceToken(c_faces.tokens[0], c_five.data(), 5) == HSUCCEED);
    for (size_t i = 0; i < cpp_five.size(); ++i) {
        CHECK(cpp_five[i].GetX() == Approx(c_five[i].x).margin(1e-4f));
        CHECK(cpp_five[i].GetY() == Approx(c_five[i].y).margin(1e-4f));
    }

    inspire::FaceEmbedding cpp_embedding = {};
    REQUIRE(cpp_session.FaceFeatureExtract(cpp_process, cpp_faces[0], cpp_embedding, true) == HSUCCEED);
    HFFaceFeature c_embedding = {};
    REQUIRE(HFFaceFeatureExtract(c_session.Get(), c_stream.Get(), c_faces.tokens[0], &c_embedding) == HSUCCEED);
    REQUIRE(cpp_embedding.embedding.size() == static_cast<size_t>(c_embedding.size));
    const std::vector<float> c_embedding_copy(c_embedding.data, c_embedding.data + c_embedding.size);

    float similarity = 0.0f;
    REQUIRE(inspire::FeatureHubDB::CosineSimilarity(cpp_embedding.embedding, c_embedding_copy, similarity, true) == HSUCCEED);
    CHECK(similarity == Approx(1.0f).margin(1e-5f));
}
