#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <inspireface.h>
#include <launch.h>
#include "middleware/model_archive/inspire_archive.h"

namespace {

using Clock = std::chrono::steady_clock;

struct ImageCase {
    const char* name;
    const char* relative_path;
    int minimum_faces;
    int maximum_faces;
};

const ImageCase kImageCases[] = {
  {"no_face", "data/bulk/view.jpg", 0, 0},
  {"single_frontal", "data/bulk/kun.jpg", 1, 1},
  {"single_profile", "data/pose/right_face.png", 1, 1},
  {"multi_face", "data/bulk/pedestrian.png", 10, 11},
};

struct ImageHandles {
    HFImageBitmap bitmap = nullptr;
    HFImageStream stream = nullptr;

    ~ImageHandles() {
        if (stream != nullptr) {
            HFReleaseImageStream(stream);
        }
        if (bitmap != nullptr) {
            HFReleaseImageBitmap(bitmap);
        }
    }
};

struct SessionHandle {
    HFSession value = nullptr;

    ~SessionHandle() {
        if (value != nullptr) {
            HFReleaseInspireFaceSession(value);
        }
    }
};

struct CaseResult {
    std::string name;
    int faces = -1;
    uint64_t digest = 0;
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    bool valid = false;
};

std::string JoinPath(const std::string& root, const std::string& relative) {
    if (root.empty()) {
        return relative;
    }
    if (root.back() == '/' || root.back() == '\\') {
        return root + relative;
    }
    return root + "/" + relative;
}

void HashBytes(uint64_t& hash, const void* data, size_t size) {
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
}

template <typename T>
void HashValue(uint64_t& hash, const T& value) {
    HashBytes(hash, &value, sizeof(value));
}

double Percentile(const std::vector<double>& sorted_samples, double percentile) {
    if (sorted_samples.empty()) {
        return 0.0;
    }
    const double position = percentile * static_cast<double>(sorted_samples.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sorted_samples.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return sorted_samples[lower] * (1.0 - fraction) + sorted_samples[upper] * fraction;
}

bool LoadImage(const std::string& path, ImageHandles& image) {
    if (HFCreateImageBitmapFromFilePath(path.c_str(), 3, &image.bitmap) != HSUCCEED) {
        return false;
    }
    return HFCreateImageStreamFromImageBitmap(image.bitmap, HF_CAMERA_ROTATION_0, &image.stream) == HSUCCEED;
}

bool CreateSession(SessionHandle& session) {
    HFSessionCustomParameter parameter = {0};
    parameter.enable_detect_mode_landmark = 1;
    return HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 25, 320, -1, &session.value) == HSUCCEED &&
           HFSessionSetTrackPreviewSize(session.value, 320) == HSUCCEED &&
           HFSessionSetFilterMinimumFacePixelSize(session.value, 0) == HSUCCEED;
}

bool CreateConcurrentSession(SessionHandle& session) {
    return HFCreateInspireFaceSessionOptional(HF_ENABLE_INTERACTION, HF_DETECT_MODE_TRACK_BY_DETECTION, 1, 320, 30, &session.value) ==
             HSUCCEED &&
           HFSessionSetTrackPreviewSize(session.value, 320) == HSUCCEED &&
           HFSessionSetFilterMinimumFacePixelSize(session.value, 0) == HSUCCEED;
}

bool HashFaces(const HFMultipleFaceData& faces, uint64_t& digest) {
    digest = 1469598103934665603ULL;
    HashValue(digest, faces.detectedNum);
    for (int i = 0; i < faces.detectedNum; ++i) {
        HashValue(digest, faces.rects[i].x);
        HashValue(digest, faces.rects[i].y);
        HashValue(digest, faces.rects[i].width);
        HashValue(digest, faces.rects[i].height);
        HashValue(digest, faces.detConfidence[i]);
        HashValue(digest, faces.angles.roll[i]);
        HashValue(digest, faces.angles.yaw[i]);
        HashValue(digest, faces.angles.pitch[i]);
        HPoint2f points[5] = {};
        if (HFGetFaceFiveKeyPointsFromFaceToken(faces.tokens[i], points, 5) != HSUCCEED) {
            return false;
        }
        for (const auto& point : points) {
            HashValue(digest, point.x);
            HashValue(digest, point.y);
        }
    }
    return true;
}

bool RunCase(HFSession session, const ImageCase& image_case, const std::string& test_root, int iterations, CaseResult& result) {
    ImageHandles image;
    result.name = image_case.name;
    if (!LoadImage(JoinPath(test_root, image_case.relative_path), image)) {
        return false;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iterations));
    uint64_t reference_digest = 0;
    bool repeatable = true;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        HFMultipleFaceData faces = {0};
        const auto begin = Clock::now();
        const HResult status = HFExecuteFaceTrack(session, image.stream, &faces);
        const auto end = Clock::now();
        uint64_t digest = 0;
        if (status != HSUCCEED || !HashFaces(faces, digest)) {
            return false;
        }
        if (iteration == 0) {
            reference_digest = digest;
            result.faces = faces.detectedNum;
        } else {
            repeatable = repeatable && digest == reference_digest;
        }
        samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }

    result.digest = reference_digest;
    result.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    std::sort(samples.begin(), samples.end());
    result.p50_us = Percentile(samples, 0.50);
    result.p95_us = Percentile(samples, 0.95);
    result.valid = repeatable && result.faces >= image_case.minimum_faces && result.faces <= image_case.maximum_faces;
    return result.valid;
}

bool RunSuite(const char* phase, const std::string& test_root, int iterations, std::vector<CaseResult>& results) {
    SessionHandle session;
    if (!CreateSession(session)) {
        std::cout << "LIFECYCLE_SUITE,phase=" << phase << ",session=FAIL,status=FAIL\n";
        return false;
    }
    bool passed = true;
    for (const auto& image_case : kImageCases) {
        CaseResult result;
        const bool case_passed = RunCase(session.value, image_case, test_root, iterations, result);
        std::cout << std::fixed << std::setprecision(3) << "LIFECYCLE_CASE,phase=" << phase << ",image=" << image_case.name
                  << ",faces=" << result.faces << ",digest=0x" << std::hex << result.digest << std::dec
                  << ",mean_us=" << result.mean_us << ",p50_us=" << result.p50_us << ",p95_us=" << result.p95_us
                  << ",status=" << (case_passed ? "PASS" : "FAIL") << '\n';
        results.push_back(result);
        passed = case_passed && passed;
    }
    return passed;
}

bool TestLandmarkSwitchIsolation(const std::string& pack_path, const std::string& test_root) {
    inspire::InspireArchive original(pack_path);
    if (original.QueryStatus() != inspire::SARC_SUCCESS || !original.GetLandmarkParam()) {
        return false;
    }
    const auto original_parameter = original.GetLandmarkParam();
    inspire::InspireArchive replacement(original);
    const bool archive_isolated = replacement.SwitchLandmarkEngine("landmark") && replacement.GetLandmarkParam() &&
                                  replacement.GetLandmarkParam() != original_parameter && original.GetLandmarkParam() == original_parameter &&
                                  replacement.GetLandmarkParam()->num_of_landmark == original_parameter->num_of_landmark &&
                                  replacement.GetLandmarkParam()->semantic_index.left_eye_center ==
                                    original_parameter->semantic_index.left_eye_center;

    SessionHandle old_session;
    CaseResult before;
    CaseResult after;
    CaseResult fresh;
    const ImageCase& image_case = kImageCases[1];
    bool inference_isolated = CreateSession(old_session) && RunCase(old_session.value, image_case, test_root, 3, before);
    const auto switch_begin = Clock::now();
    const HResult switch_status = HFSwitchLandmarkEngine(HF_LANDMARK_HYPLMV2_0_25);
    const auto switch_end = Clock::now();
    inference_isolated = inference_isolated && switch_status == HSUCCEED && RunCase(old_session.value, image_case, test_root, 3, after);
    SessionHandle fresh_session;
    inference_isolated = inference_isolated && CreateSession(fresh_session) && RunCase(fresh_session.value, image_case, test_root, 3, fresh) &&
                         before.faces == after.faces && before.faces == fresh.faces && before.digest == after.digest && before.digest == fresh.digest;
    const double switch_us = std::chrono::duration<double, std::micro>(switch_end - switch_begin).count();
    const bool passed = archive_isolated && inference_isolated;
    std::cout << std::fixed << std::setprecision(3) << "LANDMARK_SWITCH_ISOLATION,archive=" << (archive_isolated ? "PASS" : "FAIL")
              << ",old_session_exact=" << (before.digest == after.digest ? "PASS" : "FAIL")
              << ",new_session_exact=" << (before.digest == fresh.digest ? "PASS" : "FAIL") << ",switch_us=" << switch_us
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool ResultsMatch(const std::vector<CaseResult>& baseline, const std::vector<CaseResult>& candidate, const char* phase) {
    bool exact = baseline.size() == candidate.size();
    bool latency = exact;
    double baseline_total = 0.0;
    double candidate_total = 0.0;
    for (size_t i = 0; exact && i < baseline.size(); ++i) {
        exact = baseline[i].valid && candidate[i].valid && baseline[i].name == candidate[i].name &&
                baseline[i].faces == candidate[i].faces && baseline[i].digest == candidate[i].digest;
        baseline_total += baseline[i].mean_us;
        candidate_total += candidate[i].mean_us;
        const double limit = std::max(baseline[i].mean_us * 1.75, baseline[i].mean_us + 5000.0);
        latency = latency && candidate[i].mean_us <= limit;
    }
    latency = latency && candidate_total <= baseline_total * 1.30;
    std::cout << std::fixed << std::setprecision(3) << "LIFECYCLE_COMPARE,phase=" << phase
              << ",baseline_total_mean_us=" << baseline_total << ",candidate_total_mean_us=" << candidate_total
              << ",exact=" << (exact ? "PASS" : "FAIL") << ",latency=" << (latency ? "PASS" : "FAIL")
              << ",status=" << (exact && latency ? "PASS" : "FAIL") << '\n';
    return exact && latency;
}

bool TestFailedReloadRollback(const std::string& invalid_pack, const std::vector<int32_t>& expected_pixels,
                              const std::vector<std::string>& expected_models, double& reload_us) {
    const auto begin = Clock::now();
    const HResult reload_status = HFReloadInspireFace(invalid_pack.c_str());
    const auto end = Clock::now();
    reload_us = std::chrono::duration<double, std::micro>(end - begin).count();
    HInt32 loaded = 0;
    const HResult query_status = HFQueryInspireFaceLaunchStatus(&loaded);
    const auto pixels = INSPIREFACE_CONTEXT->GetFaceDetectPixelList();
    const auto models = INSPIREFACE_CONTEXT->GetFaceDetectModelList();
    const bool passed = reload_status != HSUCCEED && query_status == HSUCCEED && loaded == HF_STATUS_ENABLE &&
                        pixels == expected_pixels && models == expected_models;
    std::cout << std::fixed << std::setprecision(3) << "FAILED_RELOAD,return_rejected="
              << (reload_status != HSUCCEED ? "PASS" : "FAIL") << ",old_state_loaded="
              << (loaded == HF_STATUS_ENABLE ? "PASS" : "FAIL") << ",metadata="
              << (pixels == expected_pixels && models == expected_models ? "PASS" : "FAIL") << ",reload_us=" << reload_us
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestConcurrentLifecycle(const std::string& pack_path, const std::string& test_root, int worker_iterations,
                             int reload_iterations) {
    if (worker_iterations <= 0 || reload_iterations <= 0) {
        std::cout << "LIFECYCLE_CONCURRENCY,status=SKIP\n";
        return true;
    }

    constexpr int kWorkers = 2;
    std::vector<SessionHandle> sessions(kWorkers);
    std::vector<ImageHandles> images(kWorkers);
    for (int i = 0; i < kWorkers; ++i) {
        if (!CreateConcurrentSession(sessions[i]) || !LoadImage(JoinPath(test_root, "data/bulk/kun.jpg"), images[i])) {
            return false;
        }
    }

    std::atomic<bool> start(false);
    std::atomic<bool> valid(true);
    std::vector<uint64_t> worker_digests(kWorkers, 1469598103934665603ULL);
    std::vector<std::thread> workers;
    for (int worker = 0; worker < kWorkers; ++worker) {
        workers.emplace_back([&, worker]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            try {
                for (int iteration = 0; iteration < worker_iterations; ++iteration) {
                    auto& archive = INSPIREFACE_CONTEXT->getMArchive();
                    std::this_thread::yield();
                    if (archive.QueryStatus() != inspire::SARC_SUCCESS || archive.GetFaceDetectPixelList().empty() ||
                        archive.GetFaceDetectPixelList().size() != archive.GetFaceDetectModelList().size()) {
                        valid.store(false, std::memory_order_relaxed);
                        return;
                    }
                    HFMultipleFaceData faces = {0};
                    uint64_t digest = 0;
                    if (HFExecuteFaceTrack(sessions[worker].value, images[worker].stream, &faces) != HSUCCEED ||
                        !HashFaces(faces, digest) || faces.detectedNum != 1) {
                        valid.store(false, std::memory_order_relaxed);
                        return;
                    }
                    if (HFMultipleFacePipelineProcessOptional(sessions[worker].value, images[worker].stream, &faces, HF_ENABLE_INTERACTION) !=
                        HSUCCEED) {
                        valid.store(false, std::memory_order_relaxed);
                        return;
                    }
                    HFFaceInteractionState state = {0};
                    HFFaceInteractionsActions actions = {0};
                    if (HFGetFaceInteractionStateResult(sessions[worker].value, &state) != HSUCCEED ||
                        HFGetFaceInteractionActionsResult(sessions[worker].value, &actions) != HSUCCEED || state.num != 1 || actions.num != 1 ||
                        state.leftEyeStatusConfidence == nullptr || state.rightEyeStatusConfidence == nullptr || actions.normal == nullptr ||
                        actions.shake == nullptr || actions.jawOpen == nullptr || actions.headRaise == nullptr || actions.blink == nullptr ||
                        !std::isfinite(state.leftEyeStatusConfidence[0]) || !std::isfinite(state.rightEyeStatusConfidence[0])) {
                        valid.store(false, std::memory_order_relaxed);
                        return;
                    }
                    HashValue(worker_digests[worker], digest);
                    HashValue(worker_digests[worker], state.leftEyeStatusConfidence[0]);
                    HashValue(worker_digests[worker], state.rightEyeStatusConfidence[0]);
                    HashValue(worker_digests[worker], actions.normal[0]);
                    HashValue(worker_digests[worker], actions.shake[0]);
                    HashValue(worker_digests[worker], actions.jawOpen[0]);
                    HashValue(worker_digests[worker], actions.headRaise[0]);
                    HashValue(worker_digests[worker], actions.blink[0]);
                }
            } catch (const std::exception&) {
                valid.store(false, std::memory_order_relaxed);
            }
        });
    }

    std::vector<double> reload_samples;
    reload_samples.reserve(static_cast<size_t>(reload_iterations));
    start.store(true, std::memory_order_release);
    for (int iteration = 0; iteration < reload_iterations; ++iteration) {
        const auto begin = Clock::now();
        const HResult status = HFReloadInspireFace(pack_path.c_str());
        const auto end = Clock::now();
        reload_samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        if (status != HSUCCEED) {
            valid.store(false, std::memory_order_relaxed);
        }
    }
    for (auto& worker : workers) {
        worker.join();
    }
    std::sort(reload_samples.begin(), reload_samples.end());
    const double mean = std::accumulate(reload_samples.begin(), reload_samples.end(), 0.0) /
                        static_cast<double>(reload_samples.size());
    const bool passed = valid.load(std::memory_order_relaxed) && worker_digests[0] == worker_digests[1];
    std::cout << std::fixed << std::setprecision(3) << "LIFECYCLE_CONCURRENCY,workers=" << kWorkers
              << ",worker_iterations=" << worker_iterations << ",reload_iterations=" << reload_iterations
              << ",reload_mean_us=" << mean << ",reload_p50_us=" << Percentile(reload_samples, 0.50)
              << ",reload_p95_us=" << Percentile(reload_samples, 0.95) << ",digest=0x" << std::hex
              << worker_digests[0] << std::dec << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <pack_path> [test_res_root] [image_iterations] [worker_iterations] [reload_iterations]\n";
        return 2;
    }
    const std::string pack_path = argv[1];
    const std::string test_root = argc >= 3 ? argv[2] : "test_res";
    const int image_iterations = argc >= 4 ? std::max(5, std::stoi(argv[3])) : 10;
    const int worker_iterations = argc >= 5 ? std::max(0, std::stoi(argv[4])) : 20;
    const int reload_iterations = argc >= 6 ? std::max(0, std::stoi(argv[5])) : 3;
    const std::string invalid_pack = JoinPath(test_root, "data/bulk/face_sample.png");

    HFTerminateInspireFace();
    INSPIREFACE_CONTEXT->SwitchLandmarkEngine(inspire::Launch::LANDMARK_HYPLMV2_0_25);
    HInt32 unloaded_status = HF_STATUS_ENABLE;
    const bool unloaded_guard = HFQueryInspireFaceLaunchStatus(&unloaded_status) == HSUCCEED && unloaded_status == HF_STATUS_DISABLE;
    std::cout << "UNLOADED_ACCESS,landmark_switch=no_crash,still_unloaded=" << (unloaded_guard ? "PASS" : "FAIL")
              << ",status=" << (unloaded_guard ? "PASS" : "FAIL") << '\n';
    if (HFLaunchInspireFace(pack_path.c_str()) != HSUCCEED) {
        std::cerr << "ERROR,reason=launch_failed,pack=" << pack_path << '\n';
        return 3;
    }

    bool passed = unloaded_guard;
    passed = TestLandmarkSwitchIsolation(pack_path, test_root) && passed;
    std::vector<CaseResult> baseline;
    passed = RunSuite("baseline", test_root, image_iterations, baseline) && passed;
    const auto expected_pixels = INSPIREFACE_CONTEXT->GetFaceDetectPixelList();
    const auto expected_models = INSPIREFACE_CONTEXT->GetFaceDetectModelList();

    double failed_reload_us = 0.0;
    const bool rollback = TestFailedReloadRollback(invalid_pack, expected_pixels, expected_models, failed_reload_us);
    passed = rollback && passed;
    if (!rollback && HFReloadInspireFace(pack_path.c_str()) != HSUCCEED) {
        std::cerr << "ERROR,reason=baseline_recovery_failed\n";
        return 4;
    }

    std::vector<CaseResult> after_failed_reload;
    const bool failed_suite = RunSuite("after_failed_reload", test_root, image_iterations, after_failed_reload);
    passed = failed_suite && ResultsMatch(baseline, after_failed_reload, "after_failed_reload") && passed;

    const auto reload_begin = Clock::now();
    const HResult same_pack_status = HFReloadInspireFace(pack_path.c_str());
    const auto reload_end = Clock::now();
    const double same_pack_reload_us = std::chrono::duration<double, std::micro>(reload_end - reload_begin).count();
    std::vector<CaseResult> after_successful_reload;
    const bool successful_suite = same_pack_status == HSUCCEED &&
                                  RunSuite("after_successful_reload", test_root, image_iterations, after_successful_reload);
    passed = successful_suite && ResultsMatch(baseline, after_successful_reload, "after_successful_reload") && passed;
    std::cout << std::fixed << std::setprecision(3) << "SUCCESSFUL_RELOAD,reload_us=" << same_pack_reload_us
              << ",status=" << (same_pack_status == HSUCCEED ? "PASS" : "FAIL") << '\n';

    passed = TestConcurrentLifecycle(pack_path, test_root, worker_iterations, reload_iterations) && passed;
    if (HFTerminateInspireFace() != HSUCCEED) {
        passed = false;
    }
    std::cout << "SUMMARY,status=" << (passed ? "PASS" : "FAIL") << ",images=" << (sizeof(kImageCases) / sizeof(kImageCases[0]))
              << ",image_iterations=" << image_iterations << ",worker_iterations=" << worker_iterations
              << ",reload_iterations=" << reload_iterations << '\n';
    return passed ? 0 : 1;
}
