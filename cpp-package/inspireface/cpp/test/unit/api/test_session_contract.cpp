#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/test_tools.h"

using inspireface_test::UniqueImageBitmap;
using inspireface_test::UniqueImageStream;
using inspireface_test::UniqueSession;

namespace {

struct StreamInput {
    inspirecv::Image image;
    UniqueImageStream stream;
};

StreamInput StreamFromTestImage(const std::string& relative_path) {
    StreamInput input;
    input.image = inspirecv::Image::Create(GET_DATA(relative_path));
    REQUIRE(!input.image.Empty());
    HFImageStream handle = nullptr;
    REQUIRE(CVImageToImageStream(input.image, handle) == HSUCCEED);
    input.stream = UniqueImageStream(handle);
    return input;
}

UniqueSession CreateSession(HOption option = HF_ENABLE_NONE, HFDetectMode mode = HF_DETECT_MODE_ALWAYS_DETECT, int max_faces = 3) {
    HFSession handle = nullptr;
    REQUIRE(HFCreateInspireFaceSessionOptional(option, mode, max_faces, -1, -1, &handle) == HSUCCEED);
    REQUIRE(handle != nullptr);
    return UniqueSession(handle);
}

}  // namespace

TEST_CASE("C API session creation validates modes, limits, options, and outputs", "[api][contract][session][boundary]") {
    HFFaceDetectPixelList levels = {};
    REQUIRE(HFQuerySupportedPixelLevelsForFaceDetection(&levels) == HSUCCEED);
    REQUIRE(levels.size > 0);
    REQUIRE(levels.size <= 20);
    for (int i = 0; i < levels.size; ++i) {
        CHECK(levels.pixel_level[i] > 0);
    }
    CHECK(HFQuerySupportedPixelLevelsForFaceDetection(nullptr) == HERR_INVALID_PARAM);

    HFSessionCustomParameter parameter = {0};
    HFSession output = reinterpret_cast<HFSession>(static_cast<uintptr_t>(1));
    CHECK(HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, nullptr) == HERR_INVALID_PARAM);
    CHECK(HFCreateInspireFaceSession(parameter, static_cast<HFDetectMode>(99), 1, -1, -1, &output) == HERR_INVALID_PARAM);
    CHECK(output == nullptr);
    CHECK(HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 0, -1, -1, &output) == HERR_INVALID_PARAM);
    CHECK(HFCreateInspireFaceSessionOptional(0x40000000, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &output) == HERR_INVALID_PARAM);
    CHECK(HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_TRACK_BY_DETECTION, 1, -1, 0, &output) == HERR_INVALID_PARAM);
    CHECK(output == nullptr);

    REQUIRE(HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, 1,
                                                std::numeric_limits<HInt32>::min(), -1, &output) == HSUCCEED);
    REQUIRE(output != nullptr);
    REQUIRE(HFReleaseInspireFaceSession(output) == HSUCCEED);
    output = nullptr;

    const std::array<HFDetectMode, 3> modes = {
      {HF_DETECT_MODE_ALWAYS_DETECT, HF_DETECT_MODE_LIGHT_TRACK, HF_DETECT_MODE_TRACK_BY_DETECTION}};
    for (const auto mode : modes) {
        auto session = CreateSession(HF_ENABLE_NONE, mode, 1);
        CHECK(session.Get() != nullptr);
    }
}

TEST_CASE("C API session setters enforce documented domains", "[api][contract][session][boundary]") {
    auto session = CreateSession(HF_ENABLE_NONE, HF_DETECT_MODE_LIGHT_TRACK);

    CHECK(HFSessionSetTrackLostRecoveryMode(session.Get(), 0) == HSUCCEED);
    CHECK(HFSessionSetTrackLostRecoveryMode(session.Get(), 1) == HSUCCEED);
    CHECK(HFSessionSetTrackLostRecoveryMode(session.Get(), 2) == HERR_INVALID_PARAM);

    CHECK(HFSessionSetLightTrackConfidenceThreshold(session.Get(), 0.0f) == HSUCCEED);
    CHECK(HFSessionSetLightTrackConfidenceThreshold(session.Get(), 1.0f) == HSUCCEED);
    CHECK(HFSessionSetLightTrackConfidenceThreshold(session.Get(), -0.01f) == HERR_INVALID_PARAM);
    CHECK(HFSessionSetLightTrackConfidenceThreshold(session.Get(), std::numeric_limits<float>::quiet_NaN()) == HERR_INVALID_PARAM);

    CHECK(HFSessionSetTrackPreviewSize(session.Get(), -1) == HSUCCEED);
    CHECK(HFSessionSetTrackPreviewSize(session.Get(), 50) == HSUCCEED);
    HInt32 preview_size = 0;
    REQUIRE(HFSessionGetTrackPreviewSize(session.Get(), &preview_size) == HSUCCEED);
    CHECK(preview_size == 160);
    CHECK(HFSessionGetTrackPreviewSize(session.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFSessionSetTrackPreviewSize(session.Get(), 0) == HERR_INVALID_PARAM);

    CHECK(HFSessionSetFilterMinimumFacePixelSize(session.Get(), 0) == HSUCCEED);
    CHECK(HFSessionSetFilterMinimumFacePixelSize(session.Get(), -1) == HERR_INVALID_PARAM);
    CHECK(HFSessionSetFaceDetectThreshold(session.Get(), 0.0f) == HSUCCEED);
    CHECK(HFSessionSetFaceDetectThreshold(session.Get(), 1.0f) == HSUCCEED);
    CHECK(HFSessionSetFaceDetectThreshold(session.Get(), 1.01f) == HERR_INVALID_PARAM);
    CHECK(HFSessionSetFaceDetectThreshold(session.Get(), std::numeric_limits<float>::infinity()) == HERR_INVALID_PARAM);

    CHECK(HFSessionSetTrackModeSmoothRatio(session.Get(), 0.0f) == HSUCCEED);
    CHECK(HFSessionSetTrackModeSmoothRatio(session.Get(), 1.0f) == HSUCCEED);
    CHECK(HFSessionSetTrackModeSmoothRatio(session.Get(), -0.1f) == HERR_INVALID_PARAM);
    CHECK(HFSessionSetTrackModeNumSmoothCacheFrame(session.Get(), 1) == HSUCCEED);
    CHECK(HFSessionSetTrackModeNumSmoothCacheFrame(session.Get(), 0) == HERR_INVALID_PARAM);
    CHECK(HFSessionSetTrackModeDetectInterval(session.Get(), 1) == HSUCCEED);
    CHECK(HFSessionSetTrackModeDetectInterval(session.Get(), 0) == HERR_INVALID_PARAM);
    CHECK(HFSessionSetLandmarkAugmentationNum(session.Get(), 1) == HSUCCEED);
    CHECK(HFSessionSetLandmarkAugmentationNum(session.Get(), 3) == HSUCCEED);
    CHECK(HFSessionSetLandmarkAugmentationNum(session.Get(), 0) == HERR_INVALID_PARAM);

    CHECK(HFSessionSetEnableTrackCostSpend(session.Get(), 0) == HSUCCEED);
    CHECK(HFSessionSetEnableTrackCostSpend(session.Get(), 1) == HSUCCEED);
    CHECK(HFSessionPrintTrackCostSpend(session.Get()) == HSUCCEED);
    CHECK(HFSessionSetEnableTrackCostSpend(session.Get(), -1) == HERR_INVALID_PARAM);
    CHECK(HFSessionClearTrackingFace(session.Get()) == HSUCCEED);
}

TEST_CASE("C API face tracking resets no-face outputs and rejects invalid handles", "[api][contract][session][face_track]") {
    auto session = CreateSession();
    auto face_stream = StreamFromTestImage("data/bulk/kun.jpg");
    auto empty_stream = StreamFromTestImage("data/crop/no_face.png");

    HFMultipleFaceData faces = {};
    REQUIRE(HFExecuteFaceTrack(session.Get(), face_stream.stream.Get(), &faces) == HSUCCEED);
    REQUIRE(faces.detectedNum >= 1);
    REQUIRE(faces.rects != nullptr);
    REQUIRE(faces.trackIds != nullptr);
    REQUIRE(faces.detConfidence != nullptr);
    REQUIRE(faces.tokens != nullptr);
    for (int i = 0; i < faces.detectedNum; ++i) {
        CHECK(faces.rects[i].width > 0);
        CHECK(faces.rects[i].height > 0);
        CHECK(std::isfinite(faces.detConfidence[i]));
        CHECK(faces.tokens[i].data != nullptr);
        CHECK(faces.tokens[i].size > 0);
    }

    REQUIRE(HFExecuteFaceTrack(session.Get(), empty_stream.stream.Get(), &faces) == HSUCCEED);
    CHECK(faces.detectedNum == 0);
    CHECK(faces.rects == nullptr);
    CHECK(faces.trackIds == nullptr);
    CHECK(faces.trackCounts == nullptr);
    CHECK(faces.detConfidence == nullptr);
    CHECK(faces.tokens == nullptr);

    CHECK(HFExecuteFaceTrack(session.Get(), face_stream.stream.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFExecuteFaceTrack(nullptr, face_stream.stream.Get(), &faces) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFExecuteFaceTrack(session.Get(), nullptr, &faces) == HERR_INVALID_IMAGE_STREAM_HANDLE);

    HInt32 debug_size = 0;
    CHECK(HFSessionLastFaceDetectionGetDebugPreviewImageSize(session.Get(), &debug_size) == HSUCCEED);
    CHECK(debug_size > 0);
    CHECK(HFSessionLastFaceDetectionGetDebugPreviewImageSize(session.Get(), nullptr) == HERR_INVALID_PARAM);

    HFSession stale = session.ReleaseOwnership();
    REQUIRE(HFReleaseInspireFaceSession(stale) == HSUCCEED);
    CHECK(HFReleaseInspireFaceSession(stale) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFSessionClearTrackingFace(stale) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFSessionSetFaceDetectThreshold(stale, 0.5f) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFSessionSetLandmarkAugmentationNum(stale, 1) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFExecuteFaceTrack(stale, face_stream.stream.Get(), &faces) == HERR_INVALID_CONTEXT_HANDLE);
}

TEST_CASE("C API session rejects a handle belonging to another resource type", "[api][contract][session][boundary]") {
    std::vector<uint8_t> pixels(4 * 4 * 3, 1);
    HFImageBitmapData data = {pixels.data(), 4, 4, 3};
    UniqueImageBitmap bitmap;
    REQUIRE(HFCreateImageBitmap(&data, bitmap.Put()) == HSUCCEED);
    const auto wrong_session = reinterpret_cast<HFSession>(bitmap.Get());

    CHECK(HFSessionClearTrackingFace(wrong_session) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFSessionSetTrackPreviewSize(wrong_session, 192) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFReleaseInspireFaceSession(wrong_session) == HERR_INVALID_CONTEXT_HANDLE);
}
