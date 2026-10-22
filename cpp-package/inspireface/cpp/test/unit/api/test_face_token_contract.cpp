#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/test_tools.h"

using inspireface_test::UniqueImageStream;
using inspireface_test::UniqueSession;

TEST_CASE("C API face token can be copied and used independently of session cache", "[api][contract][face_token][landmark]") {
    HFSession session_handle = nullptr;
    REQUIRE(HFCreateInspireFaceSessionOptional(HF_ENABLE_QUALITY, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &session_handle) == HSUCCEED);
    UniqueSession session(session_handle);

    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    HFImageStream stream_handle = nullptr;
    REQUIRE(CVImageToImageStream(image, stream_handle) == HSUCCEED);
    UniqueImageStream stream(stream_handle);

    HFMultipleFaceData faces = {};
    REQUIRE(HFExecuteFaceTrack(session.Get(), stream.Get(), &faces) == HSUCCEED);
    REQUIRE(faces.detectedNum == 1);

    HInt32 token_size = 0;
    REQUIRE(HFGetFaceBasicTokenSize(&token_size) == HSUCCEED);
    REQUIRE(token_size > 0);
    CHECK(HFGetFaceBasicTokenSize(nullptr) == HERR_INVALID_PARAM);
    REQUIRE(faces.tokens[0].size == token_size);

    std::vector<char> copied(static_cast<size_t>(token_size));
    CHECK(HFCopyFaceBasicToken(faces.tokens[0], copied.data(), token_size - 1) == HERR_INVALID_BUFFER_SIZE);
    CHECK(HFCopyFaceBasicToken(faces.tokens[0], nullptr, token_size) == HERR_INVALID_PARAM);
    REQUIRE(HFCopyFaceBasicToken(faces.tokens[0], copied.data(), token_size) == HSUCCEED);

    const HFFaceBasicToken persistent = {token_size, copied.data()};
    HInt32 dense_count = 0;
    REQUIRE(HFGetNumOfFaceDenseLandmark(&dense_count) == HSUCCEED);
    REQUIRE(dense_count == 106);
    CHECK(HFGetNumOfFaceDenseLandmark(nullptr) == HERR_INVALID_PARAM);

    std::vector<HPoint2f> dense(static_cast<size_t>(dense_count));
    std::vector<HPoint2f> five(5);
    REQUIRE(HFGetFaceDenseLandmarkFromFaceToken(persistent, dense.data(), dense_count) == HSUCCEED);
    REQUIRE(HFGetFaceFiveKeyPointsFromFaceToken(persistent, five.data(), 5) == HSUCCEED);
    for (const auto& point : dense) {
        CHECK(std::isfinite(point.x));
        CHECK(std::isfinite(point.y));
    }
    for (const auto& point : five) {
        CHECK(std::isfinite(point.x));
        CHECK(std::isfinite(point.y));
    }

    REQUIRE(session.Reset() == HSUCCEED);
    std::vector<HPoint2f> after_release(5);
    CHECK(HFGetFaceFiveKeyPointsFromFaceToken(persistent, after_release.data(), 5) == HSUCCEED);
    for (size_t i = 0; i < five.size(); ++i) {
        CHECK(after_release[i].x == Approx(five[i].x));
        CHECK(after_release[i].y == Approx(five[i].y));
    }
}

TEST_CASE("C API face token and landmark functions reject malformed inputs", "[api][contract][face_token][landmark][boundary]") {
    HFFaceBasicToken invalid = {0, nullptr};
    std::vector<char> buffer(1024);
    std::vector<HPoint2f> dense(106);
    std::vector<HPoint2f> five(5);

    CHECK(HFCopyFaceBasicToken(invalid, buffer.data(), static_cast<int>(buffer.size())) == HERR_INVALID_FACE_TOKEN);
    CHECK(HFGetFaceDenseLandmarkFromFaceToken(invalid, dense.data(), 106) == HERR_INVALID_FACE_TOKEN);
    CHECK(HFGetFaceFiveKeyPointsFromFaceToken(invalid, five.data(), 5) == HERR_INVALID_FACE_TOKEN);

    invalid.data = buffer.data();
    invalid.size = 1;
    CHECK(HFGetFaceDenseLandmarkFromFaceToken(invalid, dense.data(), 106) == HERR_INVALID_FACE_TOKEN);
    CHECK(HFGetFaceFiveKeyPointsFromFaceToken(invalid, five.data(), 5) == HERR_INVALID_FACE_TOKEN);
    CHECK(HFGetFaceDenseLandmarkFromFaceToken(invalid, dense.data(), 105) == HERR_SESS_LANDMARK_NUM_NOT_MATCH);
    CHECK(HFGetFaceFiveKeyPointsFromFaceToken(invalid, five.data(), 4) == HERR_SESS_KEY_POINT_NUM_NOT_MATCH);

    HInt32 token_size = 0;
    REQUIRE(HFGetFaceBasicTokenSize(&token_size) == HSUCCEED);
    invalid.size = token_size;
    CHECK(HFGetFaceDenseLandmarkFromFaceToken(invalid, nullptr, 106) == HERR_INVALID_PARAM);
    CHECK(HFGetFaceFiveKeyPointsFromFaceToken(invalid, nullptr, 5) == HERR_INVALID_PARAM);
}
