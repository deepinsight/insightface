#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/test_tools.h"

namespace {

using Clock = std::chrono::steady_clock;
using LegacyCreateFunction = HResult (*)(HFSessionCustomParameter, HFDetectMode, HInt32, HInt32, HInt32, PHFSession);
using LegacyOptionalCreateFunction = HResult (*)(HOption, HFDetectMode, HInt32, HInt32, HInt32, PHFSession);

static_assert(sizeof(HInt32) == 4, "The legacy C ABI requires a 32-bit HInt32");
static_assert(sizeof(HFStatus) == 4, "C API level-2 status values must be 32-bit");
static_assert(sizeof(HFUInt32) == 4, "C API level-2 uint32 values must be 32-bit");
static_assert(sizeof(HFUInt64) == 8, "C API level-2 uint64 values must be 64-bit");
static_assert(sizeof(HFSessionCustomParameter) == 40, "The legacy session configuration ABI changed");
static_assert(offsetof(HFSessionCustomParameter, enable_recognition) == 0, "Legacy field offset changed");
static_assert(offsetof(HFSessionCustomParameter, enable_ir_liveness) == 8, "Legacy field offset changed");
static_assert(offsetof(HFSessionCustomParameter, enable_face_emotion) == 36, "Legacy field offset changed");
static_assert(std::is_same<decltype(&HFCreateInspireFaceSession), LegacyCreateFunction>::value,
              "The legacy session creation signature changed");
static_assert(std::is_same<decltype(&HFCreateInspireFaceSessionOptional), LegacyOptionalCreateFunction>::value,
              "The legacy optional session creation signature changed");

static_assert(sizeof(HFSessionConfigV2) == 64, "HFSessionConfigV2 must have a stable 64-byte layout");
static_assert(offsetof(HFSessionConfigV2, structSize) == 0, "HFSessionConfigV2 layout changed");
static_assert(offsetof(HFSessionConfigV2, structVersion) == 4, "HFSessionConfigV2 layout changed");
static_assert(offsetof(HFSessionConfigV2, featureMask) == 8, "HFSessionConfigV2 layout changed");
static_assert(offsetof(HFSessionConfigV2, detectMode) == 16, "HFSessionConfigV2 layout changed");
static_assert(offsetof(HFSessionConfigV2, reserved) == 32, "HFSessionConfigV2 layout changed");

HFSessionConfigV2 DefaultConfig(HFUInt64 feature_mask = HF_ENABLE_NONE, HInt32 max_faces = 3) {
    HFSessionConfigV2 config = {};
    config.structSize = sizeof(config);
    config.structVersion = HF_SESSION_CONFIG_V2_VERSION;
    config.featureMask = feature_mask;
    config.detectMode = HF_DETECT_MODE_ALWAYS_DETECT;
    config.maxDetectFaceNum = max_faces;
    config.detectPixelLevel = -1;
    config.trackByDetectModeFPS = -1;
    return config;
}

inspireface_test::UniqueSession CreateV1(HOption options = HF_ENABLE_NONE, HInt32 max_faces = 3) {
    HFSession handle = nullptr;
    REQUIRE(HFCreateInspireFaceSessionOptional(options, HF_DETECT_MODE_ALWAYS_DETECT, max_faces, -1, -1, &handle) == HSUCCEED);
    REQUIRE(handle != nullptr);
    return inspireface_test::UniqueSession(handle);
}

inspireface_test::UniqueSession CreateV2(HFUInt64 options = HF_ENABLE_NONE, HInt32 max_faces = 3) {
    auto config = DefaultConfig(options, max_faces);
    HFSession handle = nullptr;
    REQUIRE(HFCreateInspireFaceSessionV2(&config, &handle) == HSUCCEED);
    REQUIRE(handle != nullptr);
    return inspireface_test::UniqueSession(handle);
}

inspireface_test::UniqueImageStream StreamFromImage(const inspirecv::Image& image) {
    HFImageStream handle = nullptr;
    REQUIRE(CVImageToImageStream(image, handle) == HSUCCEED);
    REQUIRE(handle != nullptr);
    return inspireface_test::UniqueImageStream(handle);
}

int64_t Median(std::vector<int64_t> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <typename Callable>
int64_t MeasureMicroseconds(Callable&& callable, HResult& status) {
    const auto begin = Clock::now();
    status = callable();
    const auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

void CheckFaceDataEqual(const HFMultipleFaceData& legacy, const HFMultipleFaceData& v2) {
    REQUIRE(legacy.detectedNum == v2.detectedNum);
    for (HInt32 index = 0; index < legacy.detectedNum; ++index) {
        CHECK(legacy.rects[index].x == v2.rects[index].x);
        CHECK(legacy.rects[index].y == v2.rects[index].y);
        CHECK(legacy.rects[index].width == v2.rects[index].width);
        CHECK(legacy.rects[index].height == v2.rects[index].height);
        CHECK(legacy.detConfidence[index] == Approx(v2.detConfidence[index]).margin(1e-6f));
        CHECK(legacy.angles.roll[index] == Approx(v2.angles.roll[index]).margin(1e-6f));
        CHECK(legacy.angles.yaw[index] == Approx(v2.angles.yaw[index]).margin(1e-6f));
        CHECK(legacy.angles.pitch[index] == Approx(v2.angles.pitch[index]).margin(1e-6f));
    }
}

}  // namespace

TEST_CASE("C API level and V2 configuration have fixed boundary behavior", "[api][contract][v2][boundary]") {
    HFUInt32 api_level = 0;
    CHECK(HFQueryCAPILevel(nullptr) == HERR_INVALID_PARAM);
    REQUIRE(HFQueryCAPILevel(&api_level) == HSUCCEED);
    CHECK(api_level == HF_C_API_LEVEL);
    CHECK(api_level == 2);

    HFSession output = reinterpret_cast<HFSession>(static_cast<uintptr_t>(1));
    CHECK(HFCreateInspireFaceSessionV2(nullptr, &output) == HERR_INVALID_PARAM);
    CHECK(output == nullptr);

    auto config = DefaultConfig();
    CHECK(HFCreateInspireFaceSessionV2(&config, nullptr) == HERR_INVALID_PARAM);

    config.structSize = sizeof(config) - 1;
    output = reinterpret_cast<HFSession>(static_cast<uintptr_t>(1));
    CHECK(HFCreateInspireFaceSessionV2(&config, &output) == HERR_INVALID_PARAM);
    CHECK(output == nullptr);

    config = DefaultConfig();
    config.structSize = sizeof(config) + 32;
    REQUIRE(HFCreateInspireFaceSessionV2(&config, &output) == HSUCCEED);
    REQUIRE(output != nullptr);
    REQUIRE(HFReleaseInspireFaceSession(output) == HSUCCEED);

    config = DefaultConfig();
    config.structVersion = 0;
    CHECK(HFCreateInspireFaceSessionV2(&config, &output) == HERR_INVALID_PARAM);
    config = DefaultConfig();
    config.reserved[4] = 1;
    CHECK(HFCreateInspireFaceSessionV2(&config, &output) == HERR_INVALID_PARAM);
    config = DefaultConfig();
    config.detectMode = 99;
    CHECK(HFCreateInspireFaceSessionV2(&config, &output) == HERR_INVALID_PARAM);
    config = DefaultConfig();
    config.maxDetectFaceNum = 0;
    CHECK(HFCreateInspireFaceSessionV2(&config, &output) == HERR_INVALID_PARAM);

    config = DefaultConfig(HF_ENABLE_IR_LIVENESS);
    CHECK(HFCreateInspireFaceSessionV2(&config, &output) == HERR_UNSUPPORTED);
    CHECK(output == nullptr);
    config = DefaultConfig(HF_ENABLE_PLACEHOLDER_);
    CHECK(HFCreateInspireFaceSessionV2(&config, &output) == HERR_UNSUPPORTED);
    config = DefaultConfig(HFUInt64{1} << 63);
    CHECK(HFCreateInspireFaceSessionV2(&config, &output) == HERR_UNSUPPORTED);
}

TEST_CASE("IR requests fail explicitly through legacy creation and pipeline entry points", "[api][contract][unsupported][ir]") {
    HFSession output = reinterpret_cast<HFSession>(static_cast<uintptr_t>(1));
    HFSessionCustomParameter parameter = {};
    parameter.enable_ir_liveness = 1;
    CHECK(HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &output) == HERR_UNSUPPORTED);
    CHECK(output == nullptr);
    CHECK(HFCreateInspireFaceSessionOptional(HF_ENABLE_IR_LIVENESS, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &output) ==
          HERR_UNSUPPORTED);
    CHECK(output == nullptr);

    auto session = CreateV1();
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto stream = StreamFromImage(image);
    HFMultipleFaceData faces = {};
    REQUIRE(HFExecuteFaceTrack(session.Get(), stream.Get(), &faces) == HSUCCEED);
    REQUIRE(faces.detectedNum > 0);

    CHECK(HFMultipleFacePipelineProcess(session.Get(), stream.Get(), &faces, parameter) == HERR_UNSUPPORTED);
    CHECK(HFMultipleFacePipelineProcessOptional(session.Get(), stream.Get(), &faces, HF_ENABLE_IR_LIVENESS) == HERR_UNSUPPORTED);
}

TEST_CASE("V2 handles work with legacy detection result and release APIs", "[api][contract][v2][compatibility]") {
    auto session = CreateV2();
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto stream = StreamFromImage(image);

    HFMultipleFaceData faces = {};
    REQUIRE(HFExecuteFaceTrack(session.Get(), stream.Get(), &faces) == HSUCCEED);
    REQUIRE(faces.detectedNum > 0);
    std::array<HPoint2f, 5> points = {};
    REQUIRE(HFGetFaceFiveKeyPointsFromFaceToken(faces.tokens[0], points.data(), points.size()) == HSUCCEED);
    CHECK(std::isfinite(points[0].x));
    CHECK(std::isfinite(points[0].y));

    HFSession raw_handle = session.ReleaseOwnership();
    REQUIRE(HFReleaseInspireFaceSession(raw_handle) == HSUCCEED);
}

TEST_CASE("Legacy and V2 sessions return identical detections across representative images", "[api][v2][accuracy][consistency]") {
    auto legacy_session = CreateV1(HF_ENABLE_NONE, 5);
    auto v2_session = CreateV2(HF_ENABLE_NONE, 5);
    const std::vector<std::string> image_paths = {
      "data/bulk/kun.jpg", "data/bulk/r0.jpg", "data/bulk/woman.png", "data/bulk/pedestrian.png", "data/crop/no_face.png"};

    for (const auto& relative_path : image_paths) {
        DYNAMIC_SECTION(relative_path) {
            const auto image = inspirecv::Image::Create(GET_DATA(relative_path));
            REQUIRE(!image.Empty());
            auto legacy_stream = StreamFromImage(image);
            auto v2_stream = StreamFromImage(image);
            HFMultipleFaceData legacy_faces = {};
            HFMultipleFaceData v2_faces = {};
            REQUIRE(HFExecuteFaceTrack(legacy_session.Get(), legacy_stream.Get(), &legacy_faces) == HSUCCEED);
            REQUIRE(HFExecuteFaceTrack(v2_session.Get(), v2_stream.Get(), &v2_faces) == HSUCCEED);
            CheckFaceDataEqual(legacy_faces, v2_faces);
        }
    }
}

TEST_CASE("V2 session creation and detection latency stay within the legacy envelope", "[api][v2][performance][latency]") {
    std::vector<int64_t> legacy_creation_samples;
    std::vector<int64_t> v2_creation_samples;
    for (int iteration = 0; iteration < 5; ++iteration) {
        const bool v2_first = iteration % 2 != 0;
        for (int order = 0; order < 2; ++order) {
            const bool use_v2 = (order == 0) == v2_first;
            HFSession handle = nullptr;
            HResult status = HSUCCEED;
            int64_t elapsed = 0;
            if (use_v2) {
                auto config = DefaultConfig(HF_ENABLE_NONE, 1);
                elapsed = MeasureMicroseconds([&] { return static_cast<HResult>(HFCreateInspireFaceSessionV2(&config, &handle)); }, status);
                v2_creation_samples.push_back(elapsed);
            } else {
                elapsed = MeasureMicroseconds(
                  [&] { return HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &handle); }, status);
                legacy_creation_samples.push_back(elapsed);
            }
            REQUIRE(status == HSUCCEED);
            REQUIRE(handle != nullptr);
            REQUIRE(HFReleaseInspireFaceSession(handle) == HSUCCEED);
        }
    }

    const int64_t legacy_creation_median = Median(legacy_creation_samples);
    const int64_t v2_creation_median = Median(v2_creation_samples);
    TEST_PRINT("Legacy median session creation: {} us; V2: {} us", legacy_creation_median, v2_creation_median);
    CHECK(v2_creation_median <= static_cast<int64_t>(legacy_creation_median * 1.35) + 3000);

    auto legacy_session = CreateV1(HF_ENABLE_NONE, 1);
    auto v2_session = CreateV2(HF_ENABLE_NONE, 1);
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto legacy_stream = StreamFromImage(image);
    auto v2_stream = StreamFromImage(image);
    HFMultipleFaceData legacy_faces = {};
    HFMultipleFaceData v2_faces = {};

    for (int warmup = 0; warmup < 2; ++warmup) {
        REQUIRE(HFExecuteFaceTrack(legacy_session.Get(), legacy_stream.Get(), &legacy_faces) == HSUCCEED);
        REQUIRE(HFExecuteFaceTrack(v2_session.Get(), v2_stream.Get(), &v2_faces) == HSUCCEED);
        CheckFaceDataEqual(legacy_faces, v2_faces);
    }

    std::vector<int64_t> legacy_detection_samples;
    std::vector<int64_t> v2_detection_samples;
    for (int iteration = 0; iteration < 9; ++iteration) {
        HResult legacy_status = HSUCCEED;
        HResult v2_status = HSUCCEED;
        if (iteration % 2 == 0) {
            legacy_detection_samples.push_back(
              MeasureMicroseconds([&] { return HFExecuteFaceTrack(legacy_session.Get(), legacy_stream.Get(), &legacy_faces); }, legacy_status));
            v2_detection_samples.push_back(
              MeasureMicroseconds([&] { return HFExecuteFaceTrack(v2_session.Get(), v2_stream.Get(), &v2_faces); }, v2_status));
        } else {
            v2_detection_samples.push_back(
              MeasureMicroseconds([&] { return HFExecuteFaceTrack(v2_session.Get(), v2_stream.Get(), &v2_faces); }, v2_status));
            legacy_detection_samples.push_back(
              MeasureMicroseconds([&] { return HFExecuteFaceTrack(legacy_session.Get(), legacy_stream.Get(), &legacy_faces); }, legacy_status));
        }
        REQUIRE(legacy_status == HSUCCEED);
        REQUIRE(v2_status == HSUCCEED);
        CheckFaceDataEqual(legacy_faces, v2_faces);
    }

    const int64_t legacy_detection_median = Median(legacy_detection_samples);
    const int64_t v2_detection_median = Median(v2_detection_samples);
    TEST_PRINT("Legacy median detection: {} us; V2: {} us", legacy_detection_median, v2_detection_median);
    CHECK(v2_detection_median <= static_cast<int64_t>(legacy_detection_median * 1.35) + 2000);
}
