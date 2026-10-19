#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/test_tools.h"

using inspireface_test::UniqueFaceCaptureSession;
using inspireface_test::UniqueFaceResultSnapshot;
using inspireface_test::UniqueImageStream;
using inspireface_test::UniqueSession;

namespace {

UniqueSession CaptureSession(HOption features = HF_ENABLE_NONE, int maxFaces = 4) {
    UniqueSession session;
    REQUIRE(HFCreateInspireFaceSessionOptional(features, HF_DETECT_MODE_ALWAYS_DETECT,
                                                maxFaces, -1, -1, session.Put()) == HSUCCEED);
    return session;
}

struct ImageStreamInput {
    inspirecv::Image image;
    UniqueImageStream stream;
};

ImageStreamInput CaptureStream(const std::string& path) {
    ImageStreamInput input;
    input.image = inspirecv::Image::Create(GET_DATA(path));
    REQUIRE_FALSE(input.image.Empty());
    HFImageStream stream = nullptr;
    REQUIRE(CVImageToImageStream(input.image, stream) == HSUCCEED);
    input.stream = UniqueImageStream(stream);
    return input;
}

HFFaceCaptureConfig DefaultConfig() {
    HFFaceCaptureConfig config{};
    REQUIRE(HFGetDefaultFaceCaptureConfig(&config) == HSUCCEED);
    return config;
}

UniqueFaceCaptureSession CreateCapture(HFSession session, HFFaceCaptureConfig config) {
    UniqueFaceCaptureSession capture;
    REQUIRE(HFCreateFaceCaptureSession(session, &config, capture.Put()) == HSUCCEED);
    return capture;
}

UniqueFaceResultSnapshot DetectSnapshot(HFSession session, HFImageStream stream) {
    UniqueFaceResultSnapshot snapshot;
    REQUIRE(HFExecuteFaceTrackSnapshot(session, stream, snapshot.Put()) == HSUCCEED);
    return snapshot;
}

}  // namespace

TEST_CASE("C API face capture configuration has fixed defaults and rejects malformed values",
          "[api][capture][contract][boundary]") {
    CHECK(HFGetDefaultFaceCaptureConfig(nullptr) == HERR_INVALID_PARAM);
    const HFFaceCaptureConfig defaults = DefaultConfig();
    CHECK(defaults.structSize == sizeof(HFFaceCaptureConfig));
    CHECK(defaults.structVersion == HF_FACE_CAPTURE_CONFIG_VERSION);
    CHECK(defaults.outputCount == 1);
    CHECK(defaults.minTrackCount == 5);
    CHECK(defaults.filterMask == (HF_CAPTURE_FILTER_FACE_COUNT | HF_CAPTURE_FILTER_FACE_SIZE |
                                  HF_CAPTURE_FILTER_FACE_POSITION | HF_CAPTURE_FILTER_FACE_BOUNDARY |
                                  HF_CAPTURE_FILTER_STABILITY | HF_CAPTURE_FILTER_TRACK_COUNT));
    CHECK(sizeof(HFFaceCaptureConfig) == 152);
    CHECK(offsetof(HFFaceCaptureConfig, minTrackCount) == 20);
    CHECK(sizeof(HFFaceCaptureProgress) == 96);
    CHECK(offsetof(HFFaceCaptureProgress, trackCount) == 28);
    for (HFUInt32 value : defaults.reserved) CHECK(value == 0);

    auto session = CaptureSession();
    HFFaceCaptureSession output = reinterpret_cast<HFFaceCaptureSession>(static_cast<uintptr_t>(1));
    CHECK(HFCreateFaceCaptureSession(session.Get(), nullptr, &output) == HERR_CAPTURE_INVALID_CONFIG);
    CHECK(output == nullptr);
    CHECK(HFCreateFaceCaptureSession(session.Get(), &defaults, nullptr) == HERR_INVALID_PARAM);

    std::vector<HFFaceCaptureConfig> invalid;
    auto value = defaults;
    value.structSize = sizeof(value) - 1;
    invalid.push_back(value);
    value = defaults;
    value.structVersion += 1;
    invalid.push_back(value);
    value = defaults;
    value.minTrackCount = 0;
    invalid.push_back(value);
    value = defaults;
    value.minTrackCount = static_cast<HFUInt32>(std::numeric_limits<HInt32>::max()) + 1U;
    invalid.push_back(value);
    value = defaults;
    value.reserved[3] = 1;
    invalid.push_back(value);
    value = defaults;
    value.filterMask = UINT64_C(1) << 63;
    invalid.push_back(value);
    value = defaults;
    value.outputCount = 0;
    invalid.push_back(value);
    value = defaults;
    value.outputCount = HF_FACE_CAPTURE_MAX_RESULTS + 1;
    invalid.push_back(value);
    value = defaults;
    value.minFaceWidthRatio = std::numeric_limits<float>::quiet_NaN();
    invalid.push_back(value);
    value = defaults;
    value.minBrightnessScore = 0.9f;
    value.maxBrightnessScore = 0.1f;
    invalid.push_back(value);
    for (const auto& config : invalid) {
        output = reinterpret_cast<HFFaceCaptureSession>(static_cast<uintptr_t>(1));
        CHECK(HFCreateFaceCaptureSession(session.Get(), &config, &output) == HERR_CAPTURE_INVALID_CONFIG);
        CHECK(output == nullptr);
    }

    auto pose = defaults;
    pose.filterMask = HF_CAPTURE_FILTER_POSE;
    CHECK(HFCreateFaceCaptureSession(session.Get(), &pose, &output) == HERR_CAPTURE_REQUIRED_FEATURE_OFF);
    auto poseSession = CaptureSession(HF_ENABLE_FACE_POSE);
    REQUIRE(HFCreateFaceCaptureSession(poseSession.Get(), &pose, &output) == HSUCCEED);
    REQUIRE(HFReleaseFaceCaptureSession(output) == HSUCCEED);

    auto quality = defaults;
    quality.filterMask = HF_CAPTURE_FILTER_QUALITY;
    CHECK(HFCreateFaceCaptureSession(session.Get(), &quality, &output) == HERR_CAPTURE_REQUIRED_FEATURE_OFF);
    auto qualitySession = CaptureSession(HF_ENABLE_QUALITY);
    REQUIRE(HFCreateFaceCaptureSession(qualitySession.Get(), &quality, &output) == HSUCCEED);
    REQUIRE(HFReleaseFaceCaptureSession(output) == HSUCCEED);
}

TEST_CASE("C API face capture track-count gate reports the exact real tracker age and is optional",
          "[api][capture][track_count][accuracy][boundary]") {
    auto session = CaptureSession();
    auto input = CaptureStream("data/bulk/kun.jpg");
    auto snapshot = DetectSnapshot(session.Get(), input.stream.Get());
    HFMultipleFaceData detected{};
    REQUIRE(HFGetFaceResultSnapshotData(snapshot.Get(), &detected) == HSUCCEED);
    REQUIRE(detected.detectedNum == 1);
    REQUIRE(detected.trackCounts[0] > 0);

    auto config = DefaultConfig();
    config.filterMask = HF_CAPTURE_FILTER_TRACK_COUNT;
    config.minTrackCount = static_cast<HFUInt32>(detected.trackCounts[0] + 1);
    config.collectDurationMs = 0;
    config.maxCollectDurationMs = 1000;
    config.minCandidateIntervalMs = 0;
    auto capture = CreateCapture(session.Get(), config);

    HFFaceCaptureProgress progress{};
    REQUIRE(HFUpdateFaceCaptureSessionWithSnapshot(capture.Get(), input.stream.Get(), snapshot.Get(),
                                                    1, 1, &progress) == HSUCCEED);
    CHECK(progress.trackCount == detected.trackCounts[0]);
    CHECK((progress.rejectReasons & HF_CAPTURE_REJECT_TRACK_COUNT_TOO_LOW) != 0);
    CHECK(progress.candidateCount == 0);

    config.minTrackCount = static_cast<HFUInt32>(detected.trackCounts[0]);
    auto exact = CreateCapture(session.Get(), config);
    REQUIRE(HFUpdateFaceCaptureSessionWithSnapshot(exact.Get(), input.stream.Get(), snapshot.Get(),
                                                    1, 1, &progress) == HSUCCEED);
    CHECK(progress.trackCount == detected.trackCounts[0]);
    CHECK(progress.rejectReasons == HF_CAPTURE_REJECT_NONE);
    CHECK(progress.candidateCount == 1);

    config.filterMask = HF_CAPTURE_FILTER_FACE_COUNT;
    config.minTrackCount = std::numeric_limits<HFUInt32>::max();
    auto disabled = CreateCapture(session.Get(), config);
    REQUIRE(HFUpdateFaceCaptureSessionWithSnapshot(disabled.Get(), input.stream.Get(), snapshot.Get(),
                                                    1, 1, &progress) == HSUCCEED);
    CHECK(progress.rejectReasons == HF_CAPTURE_REJECT_NONE);
    CHECK(progress.candidateCount == 1);
}

TEST_CASE("C API face capture reuses detection snapshots and preserves exact face data",
          "[api][capture][accuracy][snapshot]") {
    auto session = CaptureSession();
    auto input = CaptureStream("data/bulk/kun.jpg");
    auto snapshot = DetectSnapshot(session.Get(), input.stream.Get());
    HFMultipleFaceData detected{};
    REQUIRE(HFGetFaceResultSnapshotData(snapshot.Get(), &detected) == HSUCCEED);
    REQUIRE(detected.detectedNum == 1);

    auto config = DefaultConfig();
    config.filterMask = HF_CAPTURE_FILTER_FACE_COUNT;
    config.outputCount = 2;
    config.stableDurationMs = 0;
    config.collectDurationMs = 0;
    config.maxCollectDurationMs = 1000;
    config.minCandidateIntervalMs = 0;
    auto capture = CreateCapture(session.Get(), config);

    HFFaceCaptureProgress progress{};
    REQUIRE(HFUpdateFaceCaptureSessionWithSnapshot(capture.Get(), input.stream.Get(), snapshot.Get(),
                                                    1, 100, &progress) == HSUCCEED);
    CHECK(progress.state == HF_CAPTURE_STATE_COLLECTING);
    CHECK(progress.rejectReasons == HF_CAPTURE_REJECT_NONE);
    CHECK(progress.candidateCount == 1);
    REQUIRE(HFUpdateFaceCaptureSessionWithSnapshot(capture.Get(), input.stream.Get(), snapshot.Get(),
                                                    2, 200, &progress) == HSUCCEED);
    CHECK(progress.state == HF_CAPTURE_STATE_READY);
    CHECK(progress.candidateCount == 2);

    HFUInt32 count = 99;
    REQUIRE(HFGetFaceCaptureResults(capture.Get(), nullptr, 0, &count) == HSUCCEED);
    REQUIRE(count == 2);
    HFFaceCaptureResult tooSmall[1]{};
    CHECK(HFGetFaceCaptureResults(capture.Get(), tooSmall, 1, &count) == HERR_INVALID_BUFFER_SIZE);
    REQUIRE(count == 2);
    HFFaceCaptureResult results[2]{};
    REQUIRE(HFGetFaceCaptureResults(capture.Get(), results, 2, &count) == HSUCCEED);
    REQUIRE(count == 2);
    for (const auto& result : results) {
        CHECK(result.trackId == detected.trackIds[0]);
        CHECK(result.rect.x == detected.rects[0].x);
        CHECK(result.rect.y == detected.rects[0].y);
        CHECK(result.rect.width == detected.rects[0].width);
        CHECK(result.rect.height == detected.rects[0].height);
        CHECK(result.token.data != nullptr);
        CHECK(result.token.size == detected.tokens[0].size);
        CHECK(std::isfinite(result.score));
    }
    CHECK(results[0].score >= results[1].score);
    std::vector<char> selectedToken(static_cast<size_t>(results[0].token.size));
    std::vector<char> detectedToken(static_cast<size_t>(detected.tokens[0].size));
    REQUIRE(HFCopyFaceBasicToken(results[0].token, selectedToken.data(), results[0].token.size) == HSUCCEED);
    REQUIRE(HFCopyFaceBasicToken(detected.tokens[0], detectedToken.data(), detected.tokens[0].size) == HSUCCEED);
    CHECK(selectedToken == detectedToken);
    void* firstTokenAddress = results[0].token.data;
    HFFaceCaptureResult repeated[2]{};
    REQUIRE(HFGetFaceCaptureResults(capture.Get(), repeated, 2, &count) == HSUCCEED);
    CHECK(repeated[0].token.data == firstTokenAddress);

    HFMultipleFaceData after{};
    REQUIRE(HFGetFaceResultSnapshotData(snapshot.Get(), &after) == HSUCCEED);
    CHECK(after.detectedNum == detected.detectedNum);
    CHECK(after.rects[0].x == detected.rects[0].x);
    CHECK(HFGetFaceCaptureResults(capture.Get(), nullptr, 1, &count) == HERR_INVALID_PARAM);
    CHECK(HFGetFaceCaptureResults(capture.Get(), nullptr, 0, nullptr) == HERR_INVALID_PARAM);
}

TEST_CASE("C API face capture direct frame path, ordering, reset, finish, and pinned lifetime are safe",
          "[api][capture][lifecycle][stability]") {
    auto sessionOwner = CaptureSession();
    auto input = CaptureStream("data/bulk/kun.jpg");
    auto config = DefaultConfig();
    config.filterMask = HF_CAPTURE_FILTER_FACE_COUNT;
    config.stableDurationMs = 0;
    config.collectDurationMs = 0;
    config.maxCollectDurationMs = 1000;
    auto capture = CreateCapture(sessionOwner.Get(), config);

    const HFSession pinnedSession = sessionOwner.ReleaseOwnership();
    REQUIRE(HFReleaseInspireFaceSession(pinnedSession) == HSUCCEED);
    HFFaceCaptureProgress progress{};
    REQUIRE(HFUpdateFaceCaptureSession(capture.Get(), input.stream.Get(), 1, 10, &progress) == HSUCCEED);
    CHECK(progress.candidateCount == 1);
    CHECK(HFUpdateFaceCaptureSession(capture.Get(), input.stream.Get(), 1, 11, &progress) ==
          HERR_CAPTURE_FRAME_OUT_OF_ORDER);
    CHECK(HFUpdateFaceCaptureSession(capture.Get(), input.stream.Get(), 2, 10, &progress) ==
          HERR_CAPTURE_FRAME_OUT_OF_ORDER);

    REQUIRE(HFFinishFaceCaptureSession(capture.Get(), &progress) == HSUCCEED);
    CHECK(progress.state == HF_CAPTURE_STATE_FINISHED);
    REQUIRE(HFFinishFaceCaptureSession(capture.Get(), &progress) == HSUCCEED);
    CHECK(progress.state == HF_CAPTURE_STATE_FINISHED);
    REQUIRE(HFResetFaceCaptureSession(capture.Get()) == HSUCCEED);
    REQUIRE(HFUpdateFaceCaptureSession(capture.Get(), input.stream.Get(), 1, 1, &progress) == HSUCCEED);

    CHECK(HFUpdateFaceCaptureSession(nullptr, input.stream.Get(), 1, 1, &progress) == HERR_CAPTURE_INVALID_HANDLE);
    CHECK(HFUpdateFaceCaptureSession(capture.Get(), nullptr, 2, 2, &progress) == HERR_INVALID_IMAGE_STREAM_HANDLE);
    CHECK(HFUpdateFaceCaptureSession(capture.Get(), input.stream.Get(), 2, 2, nullptr) == HERR_INVALID_PARAM);
    CHECK(HFFinishFaceCaptureSession(capture.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFResetFaceCaptureSession(nullptr) == HERR_CAPTURE_INVALID_HANDLE);

    const HFFaceCaptureSession stale = capture.ReleaseOwnership();
    REQUIRE(HFReleaseFaceCaptureSession(stale) == HSUCCEED);
    CHECK(HFReleaseFaceCaptureSession(stale) == HERR_CAPTURE_INVALID_HANDLE);
    CHECK(HFResetFaceCaptureSession(stale) == HERR_CAPTURE_INVALID_HANDLE);
    HFUInt32 count = 0;
    CHECK(HFGetFaceCaptureResults(stale, nullptr, 0, &count) == HERR_CAPTURE_INVALID_HANDLE);
}

TEST_CASE("C API face capture reports no-face and multiple-face outcomes across real images",
          "[api][capture][accuracy][multi_image]") {
    auto session = CaptureSession(HF_ENABLE_NONE, 8);
    auto config = DefaultConfig();
    config.filterMask = HF_CAPTURE_FILTER_FACE_COUNT;
    config.stableDurationMs = 0;
    config.collectDurationMs = 0;
    config.maxCollectDurationMs = 1000;
    auto capture = CreateCapture(session.Get(), config);

    auto noFace = CaptureStream("data/crop/no_face.png");
    auto noFaceSnapshot = DetectSnapshot(session.Get(), noFace.stream.Get());
    HFFaceCaptureProgress progress{};
    REQUIRE(HFUpdateFaceCaptureSessionWithSnapshot(capture.Get(), noFace.stream.Get(), noFaceSnapshot.Get(),
                                                    1, 1, &progress) == HSUCCEED);
    CHECK((progress.rejectReasons & HF_CAPTURE_REJECT_NO_FACE) != 0);
    CHECK(progress.state == HF_CAPTURE_STATE_IDLE);
    CHECK(progress.progress == Approx(0.0f));

    auto multiple = CaptureStream("data/bulk/pedestrian.png");
    auto multipleSnapshot = DetectSnapshot(session.Get(), multiple.stream.Get());
    HFMultipleFaceData faces{};
    REQUIRE(HFGetFaceResultSnapshotData(multipleSnapshot.Get(), &faces) == HSUCCEED);
    REQUIRE(faces.detectedNum > 1);
    REQUIRE(HFUpdateFaceCaptureSessionWithSnapshot(capture.Get(), multiple.stream.Get(), multipleSnapshot.Get(),
                                                    2, 2, &progress) == HSUCCEED);
    CHECK((progress.rejectReasons & HF_CAPTURE_REJECT_MULTIPLE_FACES) != 0);
    CHECK(progress.candidateCount == 0);
}

TEST_CASE("C API snapshot capture is deterministic across instances and adds bounded overhead",
          "[api][capture][concurrency][performance]") {
    auto session = CaptureSession();
    auto input = CaptureStream("data/bulk/kun.jpg");
    auto snapshot = DetectSnapshot(session.Get(), input.stream.Get());
    auto config = DefaultConfig();
    config.filterMask = HF_CAPTURE_FILTER_FACE_COUNT | HF_CAPTURE_FILTER_FACE_SIZE |
                        HF_CAPTURE_FILTER_FACE_POSITION;
    config.outputCount = HF_FACE_CAPTURE_MAX_RESULTS;
    config.stableDurationMs = 0;
    config.collectDurationMs = 0;
    config.maxCollectDurationMs = 100000;
    config.minCandidateIntervalMs = 0;
    auto first = CreateCapture(session.Get(), config);
    auto second = CreateCapture(session.Get(), config);
    std::atomic<int> failures(0);
    auto run = [&](HFFaceCaptureSession capture) {
        for (HFUInt64 index = 1; index <= 500; ++index) {
            HFFaceCaptureProgress progress{};
            if (HFUpdateFaceCaptureSessionWithSnapshot(capture, input.stream.Get(), snapshot.Get(),
                                                       index, index, &progress) != HSUCCEED) {
                ++failures;
            }
        }
    };
    std::thread a(run, first.Get());
    std::thread b(run, second.Get());
    a.join();
    b.join();
    CHECK(failures.load() == 0);

    HFUInt32 firstCount = 0;
    HFUInt32 secondCount = 0;
    REQUIRE(HFGetFaceCaptureResults(first.Get(), nullptr, 0, &firstCount) == HSUCCEED);
    REQUIRE(HFGetFaceCaptureResults(second.Get(), nullptr, 0, &secondCount) == HSUCCEED);
    CHECK(firstCount == secondCount);
    CHECK(firstCount <= HF_FACE_CAPTURE_MAX_RESULTS);

    auto benchmark = CreateCapture(session.Get(), config);
    constexpr HFUInt64 kIterations = 2000;
    HFFaceCaptureProgress progress{};
    HResult status = HSUCCEED;
    const auto started = std::chrono::steady_clock::now();
    for (HFUInt64 index = 1; index <= kIterations; ++index) {
        status = HFUpdateFaceCaptureSessionWithSnapshot(benchmark.Get(), input.stream.Get(), snapshot.Get(),
                                                        index, index, &progress);
        if (status != HSUCCEED) break;
    }
    const double averageUs = std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - started).count() /
                             static_cast<double>(kIterations);
    TEST_PRINT("Face capture snapshot update average: {:.3f} us", averageUs);
    CHECK(status == HSUCCEED);
    CHECK(averageUs < 500.0);
}
