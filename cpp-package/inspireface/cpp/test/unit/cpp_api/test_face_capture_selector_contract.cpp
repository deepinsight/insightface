#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

#include <inspireface/include/inspireface/inspireface.hpp>

#include "settings/test_settings.h"

namespace {

inspire::FaceTrackWrap MakeFace(int trackId = 7, int x = 30, int y = 20,
                                int width = 40, int height = 50) {
    inspire::FaceTrackWrap face{};
    face.trackId = trackId;
    face.trackCount = 1;
    face.rect = {x, y, width, height};
    for (float& value : face.quality) value = 0.1f;
    return face;
}

inspire::FaceCaptureConfig GeometryConfig(uint64_t filterMask) {
    inspire::FaceCaptureConfig config;
    config.filterMask = filterMask;
    config.stableDurationMs = 0;
    config.collectDurationMs = 0;
    config.maxCollectDurationMs = 1000;
    config.minCandidateIntervalMs = 0;
    return config;
}

inspire::FaceCaptureUpdate Feed(inspire::FaceCaptureSelector& selector,
                                const std::vector<inspire::FaceTrackWrap>& faces,
                                uint64_t frameId = 1, uint64_t timestampMs = 1) {
    inspire::FaceCaptureUpdate update;
    REQUIRE(selector.Update(100, 100, faces, frameId, timestampMs, update) == HSUCCEED);
    return update;
}

}  // namespace

TEST_CASE("C++ face capture selector configuration is explicit and transactional",
          "[cpp_api][capture][contract][boundary]") {
    using inspire::FaceCaptureSelector;
    static_assert(!std::is_copy_constructible<FaceCaptureSelector>::value,
                  "A stateful selector must not be copied");
    static_assert(std::is_move_constructible<FaceCaptureSelector>::value,
                  "A selector must support explicit ownership transfer");

    FaceCaptureSelector selector;
    CHECK(selector.GetStatus() == HERR_CAPTURE_INVALID_CONFIG);
    inspire::FaceCaptureUpdate update;
    CHECK(selector.Update(100, 100, {}, 1, 1, update) == HERR_CAPTURE_INVALID_CONFIG);

    const auto valid = GeometryConfig(inspire::CAPTURE_FILTER_FACE_COUNT);
    REQUIRE(selector.Configure(valid) == HSUCCEED);
    CHECK(selector.GetStatus() == HSUCCEED);

    const std::vector<float> invalidFloats = {
      -0.01f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    for (float value : invalidFloats) {
        auto invalid = valid;
        invalid.minFaceWidthRatio = value;
        CHECK(selector.Configure(invalid) == HERR_CAPTURE_INVALID_CONFIG);
        CHECK(selector.GetResults().empty());
        REQUIRE(selector.Configure(valid) == HSUCCEED);
    }

    auto invalid = valid;
    invalid.filterMask = UINT64_C(1) << 63;
    CHECK(selector.Configure(invalid) == HERR_CAPTURE_INVALID_CONFIG);
    invalid = valid;
    invalid.outputCount = 0;
    CHECK(selector.Configure(invalid) == HERR_CAPTURE_INVALID_CONFIG);
    invalid.outputCount = 9;
    CHECK(selector.Configure(invalid) == HERR_CAPTURE_INVALID_CONFIG);
    invalid = valid;
    invalid.minFaceWidthRatio = 0.8f;
    invalid.maxFaceWidthRatio = 0.2f;
    CHECK(selector.Configure(invalid) == HERR_CAPTURE_INVALID_CONFIG);
    invalid = valid;
    invalid.collectDurationMs = 1001;
    invalid.maxCollectDurationMs = 1000;
    CHECK(selector.Configure(invalid) == HERR_CAPTURE_INVALID_CONFIG);

    inspire::CustomPipelineParameter features;
    auto pose = valid;
    pose.filterMask = inspire::CAPTURE_FILTER_POSE;
    CHECK(selector.Configure(pose, features) == HERR_CAPTURE_REQUIRED_FEATURE_OFF);
    features.enable_face_pose = true;
    CHECK(selector.Configure(pose, features) == HSUCCEED);
    auto quality = valid;
    quality.filterMask = inspire::CAPTURE_FILTER_QUALITY;
    CHECK(selector.Configure(quality, features) == HERR_CAPTURE_REQUIRED_FEATURE_OFF);
    features.enable_face_quality = true;
    CHECK(selector.Configure(quality, features) == HSUCCEED);
}

TEST_CASE("C++ face capture selector reports every geometry and model-data rejection",
          "[cpp_api][capture][accuracy][boundary]") {
    inspire::FaceCaptureSelector selector;

    auto config = GeometryConfig(inspire::CAPTURE_FILTER_FACE_COUNT);
    REQUIRE(selector.Configure(config) == HSUCCEED);
    const auto noFace = Feed(selector, {});
    CHECK((noFace.rejectReasons & inspire::CAPTURE_REJECT_NO_FACE) != 0);
    CHECK(noFace.progress == Approx(0.0f));
    selector.Reset();
    CHECK((Feed(selector, {MakeFace(1), MakeFace(2, 10, 10, 20, 20)}).rejectReasons &
           inspire::CAPTURE_REJECT_MULTIPLE_FACES) != 0);

    config = GeometryConfig(inspire::CAPTURE_FILTER_FACE_SIZE);
    REQUIRE(selector.Configure(config) == HSUCCEED);
    CHECK((Feed(selector, {MakeFace(1, 40, 40, 5, 5)}).rejectReasons &
           inspire::CAPTURE_REJECT_FACE_TOO_SMALL) != 0);
    selector.Reset();
    CHECK((Feed(selector, {MakeFace(1, 5, 5, 90, 90)}).rejectReasons &
           inspire::CAPTURE_REJECT_FACE_TOO_LARGE) != 0);

    config = GeometryConfig(inspire::CAPTURE_FILTER_FACE_POSITION);
    config.maxCenterOffsetX = 0.05f;
    config.maxCenterOffsetY = 0.05f;
    REQUIRE(selector.Configure(config) == HSUCCEED);
    CHECK((Feed(selector, {MakeFace(1, 0, 0, 20, 20)}).rejectReasons &
           inspire::CAPTURE_REJECT_FACE_OFF_CENTER) != 0);

    config = GeometryConfig(inspire::CAPTURE_FILTER_FACE_BOUNDARY);
    config.boundaryMarginRatio = 0.1f;
    REQUIRE(selector.Configure(config) == HSUCCEED);
    CHECK((Feed(selector, {MakeFace(1, 5, 20, 40, 40)}).rejectReasons &
           inspire::CAPTURE_REJECT_FACE_OUT_OF_BOUNDS) != 0);

    inspire::CustomPipelineParameter features;
    features.enable_face_pose = true;
    config = GeometryConfig(inspire::CAPTURE_FILTER_POSE);
    config.maxAbsYaw = 10.0f;
    REQUIRE(selector.Configure(config, features) == HSUCCEED);
    auto turned = MakeFace();
    turned.face3DAngle.yaw = 10.01f;
    CHECK((Feed(selector, {turned}).rejectReasons & inspire::CAPTURE_REJECT_POSE) != 0);
    turned.face3DAngle.yaw = 10.0f;
    selector.Reset();
    CHECK(Feed(selector, {turned}).rejectReasons == inspire::CAPTURE_REJECT_NONE);

    features.enable_face_quality = true;
    config = GeometryConfig(inspire::CAPTURE_FILTER_QUALITY);
    config.minQualityScore = 0.7f;
    REQUIRE(selector.Configure(config, features) == HSUCCEED);
    auto poor = MakeFace();
    for (float& value : poor.quality) value = 0.5f;
    CHECK((Feed(selector, {poor}).rejectReasons & inspire::CAPTURE_REJECT_QUALITY) != 0);
    for (float& value : poor.quality) value = 0.3f;
    selector.Reset();
    CHECK(Feed(selector, {poor}).rejectReasons == inspire::CAPTURE_REJECT_NONE);
}

TEST_CASE("C++ face capture track-count gate is exact, optional, and resets on a new track",
          "[cpp_api][capture][track_count][accuracy][boundary]") {
    auto config = GeometryConfig(inspire::CAPTURE_FILTER_TRACK_COUNT);
    config.minTrackCount = 5;
    inspire::FaceCaptureSelector selector;
    REQUIRE(selector.Configure(config) == HSUCCEED);

    auto face = MakeFace(7);
    face.trackCount = 1;
    auto update = Feed(selector, {face}, 1, 1);
    CHECK(update.trackCount == 1);
    CHECK((update.rejectReasons & inspire::CAPTURE_REJECT_TRACK_COUNT_TOO_LOW) != 0);
    CHECK(update.candidateCount == 0);
    CHECK(update.progress == Approx(0.1f));

    face.trackCount = 4;
    update = Feed(selector, {face}, 2, 2);
    CHECK(update.trackCount == 4);
    CHECK((update.rejectReasons & inspire::CAPTURE_REJECT_TRACK_COUNT_TOO_LOW) != 0);
    CHECK(update.candidateCount == 0);

    face.trackCount = 5;
    update = Feed(selector, {face}, 3, 3);
    CHECK(update.trackCount == 5);
    CHECK(update.rejectReasons == inspire::CAPTURE_REJECT_NONE);
    CHECK(update.state == inspire::CAPTURE_STATE_READY);
    CHECK(update.candidateCount == 1);

    selector.Reset();
    auto newTrack = MakeFace(8);
    newTrack.trackCount = 1;
    update = Feed(selector, {newTrack}, 1, 1);
    CHECK(update.trackId == 8);
    CHECK(update.trackCount == 1);
    CHECK((update.rejectReasons & inspire::CAPTURE_REJECT_TRACK_COUNT_TOO_LOW) != 0);

    config.filterMask = inspire::CAPTURE_FILTER_FACE_COUNT;
    config.minTrackCount = std::numeric_limits<uint32_t>::max();
    REQUIRE(selector.Configure(config) == HSUCCEED);
    update = Feed(selector, {newTrack}, 1, 1);
    CHECK(update.rejectReasons == inspire::CAPTURE_REJECT_NONE);
    CHECK(update.state == inspire::CAPTURE_STATE_READY);

    config.filterMask = inspire::CAPTURE_FILTER_TRACK_COUNT;
    config.minTrackCount = 0;
    CHECK(selector.Configure(config) == HERR_CAPTURE_INVALID_CONFIG);
    config.minTrackCount = static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1U;
    CHECK(selector.Configure(config) == HERR_CAPTURE_INVALID_CONFIG);
}

TEST_CASE("C++ face capture temporal state uses timestamps, track identity, grace, and deterministic Top-N",
          "[cpp_api][capture][stability][accuracy]") {
    inspire::FaceCaptureConfig config = GeometryConfig(
      inspire::CAPTURE_FILTER_FACE_COUNT | inspire::CAPTURE_FILTER_STABILITY |
      inspire::CAPTURE_FILTER_FACE_POSITION);
    config.outputCount = 3;
    config.stableDurationMs = 100;
    config.collectDurationMs = 200;
    config.maxCollectDurationMs = 1000;
    config.trackLostGraceMs = 60;
    config.minCandidateIntervalMs = 50;
    inspire::FaceCaptureSelector selector;
    REQUIRE(selector.Configure(config) == HSUCCEED);

    auto face = MakeFace();
    auto first = Feed(selector, {face}, 1, 1000);
    CHECK(first.state == inspire::CAPTURE_STATE_STABILIZING);
    CHECK((first.rejectReasons & inspire::CAPTURE_REJECT_UNSTABLE) != 0);
    auto stable = Feed(selector, {face}, 2, 1100);
    CHECK(stable.state == inspire::CAPTURE_STATE_COLLECTING);
    CHECK(stable.candidateCount == 1);
    auto duplicate = Feed(selector, {face}, 3, 1120);
    CHECK(duplicate.candidateCount == 1);
    face.rect.x += 1;
    REQUIRE(Feed(selector, {face}, 4, 1160).candidateCount == 2);
    face.rect.x -= 2;
    REQUIRE(Feed(selector, {face}, 5, 1220).candidateCount == 3);
    auto ready = Feed(selector, {face}, 6, 1300);
    CHECK(ready.state == inspire::CAPTURE_STATE_READY);
    CHECK(ready.progress == Approx(1.0f));

    const auto results = selector.GetResults();
    REQUIRE(results.size() == 3);
    CHECK(results[0].score >= results[1].score);
    CHECK(results[1].score >= results[2].score);
    CHECK(results[0].face.trackId == 7);
    CHECK(selector.Update(100, 100, {face}, 6, 1301, ready) == HERR_CAPTURE_FRAME_OUT_OF_ORDER);
    CHECK(selector.Update(100, 100, {face}, 7, 1300, ready) == HERR_CAPTURE_FRAME_OUT_OF_ORDER);

    selector.Reset();
    REQUIRE(Feed(selector, {face}, 1, 1).state == inspire::CAPTURE_STATE_STABILIZING);
    auto lost = Feed(selector, {}, 2, 50);
    CHECK(lost.state == inspire::CAPTURE_STATE_TRACK_LOST);
    CHECK(lost.trackId == 7);
    auto expired = Feed(selector, {}, 3, 70);
    CHECK(expired.state == inspire::CAPTURE_STATE_IDLE);
    auto switched = Feed(selector, {MakeFace(9)}, 4, 100);
    CHECK(switched.state == inspire::CAPTURE_STATE_STABILIZING);
    CHECK(switched.trackId == 9);

    inspire::FaceCaptureUpdate finished;
    REQUIRE(selector.Finish(finished) == HSUCCEED);
    CHECK(finished.state == inspire::CAPTURE_STATE_FINISHED);
    CHECK(selector.Finish(finished) == HSUCCEED);
    CHECK(Feed(selector, {MakeFace(9)}, 5, 200).state == inspire::CAPTURE_STATE_FINISHED);
}

TEST_CASE("C++ face capture pixel filters distinguish texture and brightness without changing input faces",
          "[cpp_api][capture][pixel][accuracy]") {
    std::vector<uint8_t> checker(100 * 100 * 3);
    for (int y = 0; y < 100; ++y) {
        for (int x = 0; x < 100; ++x) {
            const uint8_t value = ((x / 4 + y / 4) & 1) != 0 ? 230 : 30;
            const size_t offset = static_cast<size_t>(y * 100 + x) * 3;
            checker[offset] = checker[offset + 1] = checker[offset + 2] = value;
        }
    }
    auto process = inspirecv::FrameProcess::Create(checker.data(), 100, 100, inspirecv::BGR);
    auto config = GeometryConfig(inspire::CAPTURE_FILTER_SHARPNESS |
                                 inspire::CAPTURE_FILTER_BRIGHTNESS);
    config.minSharpnessScore = 0.10f;
    config.minBrightnessScore = 0.2f;
    config.maxBrightnessScore = 0.8f;
    inspire::FaceCaptureSelector selector;
    REQUIRE(selector.Configure(config) == HSUCCEED);
    const std::vector<inspire::FaceTrackWrap> input{MakeFace(7, 20, 20, 60, 60)};
    const auto original = input;
    inspire::FaceCaptureUpdate update;
    REQUIRE(selector.Update(process, input, 1, 1, update) == HSUCCEED);
    CHECK(update.rejectReasons == inspire::CAPTURE_REJECT_NONE);
    CHECK(update.metrics.sharpnessScore >= config.minSharpnessScore);
    CHECK(update.metrics.brightnessScore > 0.2f);
    CHECK(std::memcmp(input.data(), original.data(), sizeof(inspire::FaceTrackWrap)) == 0);

    std::vector<uint8_t> flat(100 * 100 * 3, 128);
    process.SetDataBuffer(flat.data(), 100, 100);
    selector.Reset();
    REQUIRE(selector.Update(process, input, 1, 1, update) == HSUCCEED);
    CHECK((update.rejectReasons & inspire::CAPTURE_REJECT_SHARPNESS) != 0);
    CHECK((update.rejectReasons & inspire::CAPTURE_REJECT_BRIGHTNESS) == 0);

    std::fill(flat.begin(), flat.end(), 255);
    process.SetDataBuffer(flat.data(), 100, 100);
    selector.Reset();
    REQUIRE(selector.Update(process, input, 1, 1, update) == HSUCCEED);
    CHECK((update.rejectReasons & inspire::CAPTURE_REJECT_BRIGHTNESS) != 0);

    // A caller-selected RGB output must remain untouched, while the selector
    // independently converts BGR input to a known layout for luminance math.
    std::vector<uint8_t> blue(100 * 100 * 3, 0);
    for (size_t offset = 0; offset < blue.size(); offset += 3) blue[offset] = 255;
    process = inspirecv::FrameProcess::Create(blue.data(), 100, 100, inspirecv::BGR);
    process.SetDestFormat(inspirecv::RGB);
    config = GeometryConfig(inspire::CAPTURE_FILTER_BRIGHTNESS);
    config.minBrightnessScore = 0.0f;
    config.maxBrightnessScore = 0.2f;
    REQUIRE(selector.Configure(config) == HSUCCEED);
    REQUIRE(selector.Update(process, input, 1, 1, update) == HSUCCEED);
    CHECK(update.rejectReasons == inspire::CAPTURE_REJECT_NONE);
    CHECK(update.metrics.availableMetrics == inspire::CAPTURE_FILTER_BRIGHTNESS);
    CHECK(update.metrics.brightnessScore < config.maxBrightnessScore);
    CHECK(update.metrics.sharpnessScore == Approx(-1.0f));
    const auto callerView = process.ExecuteImageScaleProcessing(1.0f, true);
    REQUIRE(!callerView.Empty());
    REQUIRE(callerView.Channels() == 3);
    CHECK(callerView.Data()[0] == 0);
    CHECK(callerView.Data()[1] == 0);
    CHECK(callerView.Data()[2] == 255);
}

TEST_CASE("C++ face capture selector is deterministic, instance-safe, and bounded in latency",
          "[cpp_api][capture][concurrency][performance]") {
    auto config = GeometryConfig(inspire::CAPTURE_FILTER_FACE_COUNT |
                                 inspire::CAPTURE_FILTER_FACE_SIZE |
                                 inspire::CAPTURE_FILTER_FACE_POSITION |
                                 inspire::CAPTURE_FILTER_TRACK_COUNT);
    config.minTrackCount = 5;
    config.outputCount = 8;
    config.maxCollectDurationMs = 1000000;
    inspire::FaceCaptureSelector first;
    inspire::FaceCaptureSelector second;
    REQUIRE(first.Configure(config) == HSUCCEED);
    REQUIRE(second.Configure(config) == HSUCCEED);
    auto trackedFace = MakeFace();
    trackedFace.trackCount = 5;
    const std::vector<inspire::FaceTrackWrap> faces{trackedFace};
    std::atomic<int> failures(0);
    auto run = [&](inspire::FaceCaptureSelector& selector) {
        for (uint64_t index = 1; index <= 2000; ++index) {
            inspire::FaceCaptureUpdate update;
            if (selector.Update(100, 100, faces, index, index, update) != HSUCCEED) ++failures;
        }
    };
    std::thread a(run, std::ref(first));
    std::thread b(run, std::ref(second));
    a.join();
    b.join();
    CHECK(failures.load() == 0);
    const auto firstResults = first.GetResults();
    const auto secondResults = second.GetResults();
    REQUIRE(firstResults.size() == secondResults.size());
    REQUIRE(!firstResults.empty());
    for (size_t index = 0; index < firstResults.size(); ++index) {
        CHECK(firstResults[index].frameId == secondResults[index].frameId);
        CHECK(firstResults[index].score == Approx(secondResults[index].score));
    }

    inspire::FaceCaptureSelector benchmark;
    REQUIRE(benchmark.Configure(config) == HSUCCEED);
    constexpr uint64_t kIterations = 50000;
    inspire::FaceCaptureUpdate update;
    const auto started = std::chrono::steady_clock::now();
    int32_t status = HSUCCEED;
    for (uint64_t index = 1; index <= kIterations; ++index) {
        status = benchmark.Update(100, 100, faces, index, index, update);
        if (status != HSUCCEED) break;
    }
    const double averageUs = std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - started).count() /
                             static_cast<double>(kIterations);
    TEST_PRINT("Face capture selector average: {:.3f} us", averageUs);
    CHECK(status == HSUCCEED);
    CHECK(averageUs < 100.0);
    CHECK(benchmark.GetResults().size() <= 8);
}
