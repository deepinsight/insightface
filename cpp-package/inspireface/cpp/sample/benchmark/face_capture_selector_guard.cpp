#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <inspireface.h>

namespace {

using Clock = std::chrono::steady_clock;

struct SessionOwner {
    HFSession value = nullptr;
    ~SessionOwner() { if (value != nullptr) HFReleaseInspireFaceSession(value); }
};
struct BitmapOwner {
    HFImageBitmap value = nullptr;
    ~BitmapOwner() { if (value != nullptr) HFReleaseImageBitmap(value); }
};
struct StreamOwner {
    HFImageStream value = nullptr;
    ~StreamOwner() { if (value != nullptr) HFReleaseImageStream(value); }
};
struct SnapshotOwner {
    HFFaceResultSnapshot value = nullptr;
    ~SnapshotOwner() { if (value != nullptr) HFReleaseFaceResultSnapshot(value); }
};
struct CaptureOwner {
    HFFaceCaptureSession value = nullptr;
    ~CaptureOwner() { if (value != nullptr) HFReleaseFaceCaptureSession(value); }
};

std::string Join(const std::string& root, const std::string& relative) {
    return root.empty() || root.back() == '/' ? root + relative : root + "/" + relative;
}

bool Load(const std::string& path, BitmapOwner& bitmap, StreamOwner& stream) {
    return HFCreateImageBitmapFromFilePath(path.c_str(), 3, &bitmap.value) == HSUCCEED &&
           HFCreateImageStreamFromImageBitmap(bitmap.value, HF_CAMERA_ROTATION_0, &stream.value) == HSUCCEED;
}

bool CreateSession(SessionOwner& session) {
    return HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT,
                                               16, -1, -1, &session.value) == HSUCCEED;
}

HFFaceCaptureConfig CaptureConfig(uint32_t outputCount = 1) {
    HFFaceCaptureConfig config{};
    if (HFGetDefaultFaceCaptureConfig(&config) != HSUCCEED) return config;
    config.filterMask = HF_CAPTURE_FILTER_FACE_COUNT | HF_CAPTURE_FILTER_FACE_SIZE |
                        HF_CAPTURE_FILTER_FACE_POSITION | HF_CAPTURE_FILTER_FACE_BOUNDARY;
    config.outputCount = outputCount;
    config.stableDurationMs = 0;
    config.collectDurationMs = 0;
    config.maxCollectDurationMs = 100000000;
    config.minCandidateIntervalMs = 0;
    // Keep the guard focused on selector consistency, not policy tuning.
    config.minFaceWidthRatio = 0.0f;
    config.maxFaceWidthRatio = 1.0f;
    config.maxCenterOffsetX = 1.0f;
    config.maxCenterOffsetY = 1.0f;
    config.boundaryMarginRatio = 0.0f;
    return config;
}

double Percentile(std::vector<double> samples, double fraction) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const size_t index = static_cast<size_t>(fraction * (samples.size() - 1));
    return samples[index];
}

bool SameFace(const HFFaceCaptureResult& selected, const HFMultipleFaceData& detected) {
    return detected.detectedNum == 1 && selected.trackId == detected.trackIds[0] &&
           selected.rect.x == detected.rects[0].x && selected.rect.y == detected.rects[0].y &&
           selected.rect.width == detected.rects[0].width && selected.rect.height == detected.rects[0].height &&
           selected.token.size == detected.tokens[0].size && selected.token.data != nullptr;
}

bool RunImageCase(SessionOwner& session, const std::string& root, const char* relative,
                  uint64_t frameId) {
    BitmapOwner bitmap;
    StreamOwner stream;
    SnapshotOwner snapshot;
    CaptureOwner capture;
    const std::string path = Join(root, relative);
    if (!Load(path, bitmap, stream) ||
        HFExecuteFaceTrackSnapshot(session.value, stream.value, &snapshot.value) != HSUCCEED) {
        std::cerr << "ERROR,case=" << relative << ",reason=input_or_track_failed,path=" << path << '\n';
        return false;
    }
    HFMultipleFaceData detected{};
    if (HFGetFaceResultSnapshotData(snapshot.value, &detected) != HSUCCEED) return false;
    const auto config = CaptureConfig();
    if (HFCreateFaceCaptureSession(session.value, &config, &capture.value) != HSUCCEED) return false;
    HFFaceCaptureProgress progress{};
    if (HFUpdateFaceCaptureSessionWithSnapshot(capture.value, stream.value, snapshot.value,
                                               frameId, frameId, &progress) != HSUCCEED) return false;
    HFUInt32 count = 0;
    if (HFGetFaceCaptureResults(capture.value, nullptr, 0, &count) != HSUCCEED) return false;

    bool passed = false;
    if (detected.detectedNum == 0) {
        passed = count == 0 && (progress.rejectReasons & HF_CAPTURE_REJECT_NO_FACE) != 0;
    } else if (detected.detectedNum == 1) {
        HFFaceCaptureResult result{};
        passed = count == 1 && progress.rejectReasons == HF_CAPTURE_REJECT_NONE &&
                 HFGetFaceCaptureResults(capture.value, &result, 1, &count) == HSUCCEED &&
                 SameFace(result, detected);
    } else {
        passed = count == 0 && (progress.rejectReasons & HF_CAPTURE_REJECT_MULTIPLE_FACES) != 0;
    }

    HFMultipleFaceData after{};
    passed = passed && HFGetFaceResultSnapshotData(snapshot.value, &after) == HSUCCEED &&
             after.detectedNum == detected.detectedNum;
    if (after.detectedNum > 0) {
        passed = passed && after.rects[0].x == detected.rects[0].x &&
                 after.rects[0].y == detected.rects[0].y &&
                 after.rects[0].width == detected.rects[0].width &&
                 after.rects[0].height == detected.rects[0].height;
    }
    std::cout << "CASE,image=" << relative << ",faces=" << detected.detectedNum
              << ",selected=" << count << ",reject=0x" << std::hex << progress.rejectReasons
              << std::dec << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool RunTiming(SessionOwner& session, const std::string& root, int iterations) {
    BitmapOwner bitmap;
    StreamOwner stream;
    SnapshotOwner snapshot;
    CaptureOwner capture;
    if (!Load(Join(root, "data/bulk/kun.jpg"), bitmap, stream) ||
        HFExecuteFaceTrackSnapshot(session.value, stream.value, &snapshot.value) != HSUCCEED) return false;
    const auto config = CaptureConfig(HF_FACE_CAPTURE_MAX_RESULTS);
    if (HFCreateFaceCaptureSession(session.value, &config, &capture.value) != HSUCCEED) return false;

    std::vector<double> baseline;
    std::vector<double> captureSamples;
    baseline.reserve(iterations);
    captureSamples.reserve(iterations);
    HFFaceCaptureProgress progress{};
    for (int index = 0; index < iterations; ++index) {
        HFMultipleFaceData view{};
        auto started = Clock::now();
        if (HFGetFaceResultSnapshotData(snapshot.value, &view) != HSUCCEED) return false;
        baseline.push_back(std::chrono::duration<double, std::micro>(Clock::now() - started).count());

        started = Clock::now();
        if (HFUpdateFaceCaptureSessionWithSnapshot(capture.value, stream.value, snapshot.value,
                                                   static_cast<HFUInt64>(index + 1),
                                                   static_cast<HFUInt64>(index + 1), &progress) != HSUCCEED) return false;
        captureSamples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - started).count());
    }
    const double baselineP50 = Percentile(baseline, 0.50);
    const double captureP50 = Percentile(captureSamples, 0.50);
    const double captureP95 = Percentile(captureSamples, 0.95);
    const double addedP50 = std::max(0.0, captureP50 - baselineP50);
    const bool passed = captureP95 < 1000.0;
    std::cout << "TIMING,iterations=" << iterations << ",baseline_p50_us=" << baselineP50
              << ",capture_p50_us=" << captureP50 << ",capture_p95_us=" << captureP95
              << ",added_p50_us=" << addedP50 << ",limit_p95_us=1000,status="
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <pack_path> [test_res_root] [iterations]\n";
        return 2;
    }
    const std::string pack = argv[1];
    const std::string root = argc >= 3 ? argv[2] : "test_res";
    const int iterations = argc >= 4 ? std::max(10, std::stoi(argv[3])) : 2000;
    if (HFLaunchInspireFace(pack.c_str()) != HSUCCEED) return 3;
    SessionOwner session;
    bool passed = CreateSession(session);
    const std::vector<const char*> images = {
      "data/bulk/kun.jpg", "data/bulk/r0.jpg", "data/bulk/woman.png",
      "data/bulk/pedestrian.png", "data/crop/no_face.png"};
    uint64_t frameId = 1;
    for (const char* image : images) {
        passed = RunImageCase(session, root, image, frameId++) && passed;
    }
    passed = RunTiming(session, root, iterations) && passed;
    if (session.value != nullptr) {
        passed = HFReleaseInspireFaceSession(session.value) == HSUCCEED && passed;
        session.value = nullptr;
    }
    if (HFTerminateInspireFace() != HSUCCEED) passed = false;
    std::cout << "SUMMARY,status=" << (passed ? "PASS" : "FAIL")
              << ",images=" << images.size() << ",iterations=" << iterations << '\n';
    return passed ? 0 : 1;
}
