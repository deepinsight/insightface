#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <inspireface.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Scenario {
    const char* name;
    const char* relative_path;
    HFDetectMode mode;
    int input_size;
    int detect_interval;
    int minimum_faces;
    int maximum_faces;
    bool require_unique_ids;
    bool verify_dependent_apis;
};

const std::vector<Scenario> kScenarios = {
  {"always_multi", "data/bulk/pedestrian.png", HF_DETECT_MODE_ALWAYS_DETECT, 640, 1, 16, 20, false, false},
  {"light_single_cache", "data/bulk/kun.jpg", HF_DETECT_MODE_LIGHT_TRACK, 320, 5, 1, 1, true, true},
  {"light_multi_redetect", "data/bulk/pedestrian.png", HF_DETECT_MODE_LIGHT_TRACK, 640, 1, 16, 20, true, false},
  {"track_by_detect_single", "data/bulk/kun.jpg", HF_DETECT_MODE_TRACK_BY_DETECTION, 320, 1, 0, 1, true, false},
  {"always_no_face", "data/bulk/view.jpg", HF_DETECT_MODE_ALWAYS_DETECT, 320, 1, 0, 0, false, false},
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

struct OwnedToken {
    std::vector<char> bytes;

    HFFaceBasicToken View() {
        HFFaceBasicToken token = {};
        token.size = static_cast<HInt32>(bytes.size());
        token.data = bytes.data();
        return token;
    }
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
    HResult status = HFCreateImageBitmapFromFilePath(path.c_str(), 3, &image.bitmap);
    if (status != HSUCCEED) {
        std::cerr << "ERROR,reason=image_load_failed,status=" << status << ",path=" << path << '\n';
        return false;
    }
    status = HFCreateImageStreamFromImageBitmap(image.bitmap, HF_CAMERA_ROTATION_0, &image.stream);
    if (status != HSUCCEED) {
        std::cerr << "ERROR,reason=stream_create_failed,status=" << status << ",path=" << path << '\n';
        return false;
    }
    return true;
}

bool CreateSession(const Scenario& scenario, SessionHandle& session) {
    HFSessionCustomParameter parameter = {0};
    parameter.enable_recognition = scenario.verify_dependent_apis ? 1 : 0;
    parameter.enable_interaction_liveness = scenario.verify_dependent_apis ? 1 : 0;
    parameter.enable_detect_mode_landmark = 1;
    parameter.enable_face_emotion = scenario.verify_dependent_apis ? 1 : 0;
    HResult status = HFCreateInspireFaceSession(parameter, scenario.mode, 25, scenario.input_size, 30, &session.value);
    if (status != HSUCCEED) {
        std::cerr << "ERROR,scenario=" << scenario.name << ",reason=session_create_failed,status=" << status << '\n';
        return false;
    }
    const HResult preview_status = HFSessionSetTrackPreviewSize(session.value, scenario.input_size);
    const HResult filter_status = HFSessionSetFilterMinimumFacePixelSize(session.value, 0);
    const HResult interval_status = HFSessionSetTrackModeDetectInterval(session.value, scenario.detect_interval);
    if (preview_status != HSUCCEED || filter_status != HSUCCEED || interval_status != HSUCCEED) {
        std::cerr << "ERROR,scenario=" << scenario.name << ",reason=session_config_failed,preview_status=" << preview_status
                  << ",filter_status=" << filter_status << ",interval_status=" << interval_status << '\n';
        return false;
    }
    return true;
}

bool CopyToken(const HFFaceBasicToken& token, OwnedToken& copy) {
    HInt32 token_size = 0;
    if (HFGetFaceBasicTokenSize(&token_size) != HSUCCEED || token_size <= 0 || token.size != token_size || token.data == nullptr) {
        return false;
    }
    copy.bytes.resize(static_cast<size_t>(token_size));
    return HFCopyFaceBasicToken(token, copy.bytes.data(), token_size) == HSUCCEED;
}

bool ExtractLandmarks(HFFaceBasicToken token, HPoint2f (&five_points)[5], std::vector<HPoint2f>& dense_points) {
    HInt32 dense_count = 0;
    if (HFGetNumOfFaceDenseLandmark(&dense_count) != HSUCCEED || dense_count != 106) {
        return false;
    }
    dense_points.resize(static_cast<size_t>(dense_count));
    if (HFGetFaceFiveKeyPointsFromFaceToken(token, five_points, 5) != HSUCCEED ||
        HFGetFaceDenseLandmarkFromFaceToken(token, dense_points.data(), dense_count) != HSUCCEED) {
        return false;
    }
    for (const auto& point : five_points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return false;
        }
    }
    for (const auto& point : dense_points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return false;
        }
    }
    return true;
}

bool ValidateAndHashFrame(const Scenario& scenario, const HFMultipleFaceData& faces, uint64_t& frame_hash,
                          OwnedToken* first_token = nullptr, HPoint2f (*first_five_points)[5] = nullptr,
                          std::vector<HPoint2f>* first_dense_points = nullptr) {
    if (faces.detectedNum < scenario.minimum_faces || faces.detectedNum > scenario.maximum_faces) {
        return false;
    }
    if (faces.detectedNum > 0 &&
        (faces.rects == nullptr || faces.trackIds == nullptr || faces.trackCounts == nullptr || faces.detConfidence == nullptr ||
         faces.angles.roll == nullptr || faces.angles.yaw == nullptr || faces.angles.pitch == nullptr || faces.tokens == nullptr)) {
        return false;
    }

    frame_hash = 1469598103934665603ULL;
    HashValue(frame_hash, faces.detectedNum);
    std::set<int> track_ids;
    for (int i = 0; i < faces.detectedNum; ++i) {
        const HFaceRect& rect = faces.rects[i];
        if (rect.width <= 0 || rect.height <= 0 || faces.trackCounts[i] < 0 || !std::isfinite(faces.detConfidence[i]) ||
            !std::isfinite(faces.angles.roll[i]) || !std::isfinite(faces.angles.yaw[i]) || !std::isfinite(faces.angles.pitch[i])) {
            return false;
        }
        if (scenario.require_unique_ids && !track_ids.insert(faces.trackIds[i]).second) {
            return false;
        }

        HashValue(frame_hash, rect.x);
        HashValue(frame_hash, rect.y);
        HashValue(frame_hash, rect.width);
        HashValue(frame_hash, rect.height);
        HashValue(frame_hash, faces.trackIds[i]);
        HashValue(frame_hash, faces.trackCounts[i]);
        HashValue(frame_hash, faces.detConfidence[i]);
        HashValue(frame_hash, faces.angles.roll[i]);
        HashValue(frame_hash, faces.angles.yaw[i]);
        HashValue(frame_hash, faces.angles.pitch[i]);

        OwnedToken token_copy;
        if (!CopyToken(faces.tokens[i], token_copy)) {
            return false;
        }
        HashValue(frame_hash, faces.tokens[i].size);
        HashBytes(frame_hash, token_copy.bytes.data(), token_copy.bytes.size());

        HPoint2f five_points[5] = {};
        std::vector<HPoint2f> dense_points;
        if (!ExtractLandmarks(token_copy.View(), five_points, dense_points)) {
            return false;
        }
        for (const auto& point : five_points) {
            HashValue(frame_hash, point.x);
            HashValue(frame_hash, point.y);
        }
        for (const auto& point : dense_points) {
            HashValue(frame_hash, point.x);
            HashValue(frame_hash, point.y);
        }

        if (i == 0 && first_token != nullptr && first_five_points != nullptr && first_dense_points != nullptr) {
            *first_token = std::move(token_copy);
            std::memcpy(*first_five_points, five_points, sizeof(five_points));
            *first_dense_points = dense_points;
        }
    }
    return true;
}

bool ValidateCopiedTokenAfterNextTrack(HFSession session, HFImageStream stream, OwnedToken& token_copy,
                                       const HPoint2f (&expected_five_points)[5], const std::vector<HPoint2f>& expected_dense_points,
                                       uint64_t& dependent_hash) {
    HPoint2f five_points[5] = {};
    std::vector<HPoint2f> dense_points;
    HFFaceBasicToken token = token_copy.View();
    if (!ExtractLandmarks(token, five_points, dense_points) || dense_points.size() != expected_dense_points.size() ||
        std::memcmp(five_points, expected_five_points, sizeof(five_points)) != 0 ||
        std::memcmp(dense_points.data(), expected_dense_points.data(), dense_points.size() * sizeof(HPoint2f)) != 0) {
        return false;
    }

    HFloat quality = 0.0f;
    if (HFFaceQualityDetect(session, token, &quality) != HSUCCEED || !std::isfinite(quality)) {
        return false;
    }

    HFFaceFeature feature = {0};
    if (HFCreateFaceFeature(&feature) != HSUCCEED) {
        return false;
    }
    const HResult feature_status = HFFaceFeatureExtractTo(session, stream, token, feature);
    if (feature_status != HSUCCEED || feature.size <= 0 || feature.data == nullptr) {
        HFReleaseFaceFeature(&feature);
        return false;
    }

    dependent_hash = 1469598103934665603ULL;
    HashBytes(dependent_hash, token_copy.bytes.data(), token_copy.bytes.size());
    HashValue(dependent_hash, quality);
    HashValue(dependent_hash, feature.size);
    for (int i = 0; i < feature.size; ++i) {
        if (!std::isfinite(feature.data[i])) {
            HFReleaseFaceFeature(&feature);
            return false;
        }
        HashValue(dependent_hash, feature.data[i]);
    }
    return HFReleaseFaceFeature(&feature) == HSUCCEED;
}

bool ValidatePipelineLink(HFSession session, HFImageStream stream, HFMultipleFaceData& faces, uint64_t& pipeline_hash) {
    const HOption option = HF_ENABLE_INTERACTION | HF_ENABLE_FACE_EMOTION;
    if (HFMultipleFacePipelineProcessOptional(session, stream, &faces, option) != HSUCCEED) {
        return false;
    }

    HFFaceInteractionState states = {0};
    HFFaceInteractionsActions actions = {0};
    HFFaceEmotionResult emotions = {0};
    if (HFGetFaceInteractionStateResult(session, &states) != HSUCCEED ||
        HFGetFaceInteractionActionsResult(session, &actions) != HSUCCEED || HFGetFaceEmotionResult(session, &emotions) != HSUCCEED ||
        states.num != faces.detectedNum || actions.num != faces.detectedNum || emotions.num != faces.detectedNum) {
        return false;
    }
    if (faces.detectedNum > 0 &&
        (states.leftEyeStatusConfidence == nullptr || states.rightEyeStatusConfidence == nullptr || actions.normal == nullptr ||
         actions.shake == nullptr || actions.jawOpen == nullptr || actions.headRaise == nullptr || actions.blink == nullptr ||
         emotions.emotion == nullptr)) {
        return false;
    }

    pipeline_hash = 1469598103934665603ULL;
    HashValue(pipeline_hash, states.num);
    for (int i = 0; i < faces.detectedNum; ++i) {
        if (!std::isfinite(states.leftEyeStatusConfidence[i]) || !std::isfinite(states.rightEyeStatusConfidence[i]) ||
            emotions.emotion[i] < -1 || emotions.emotion[i] > 6) {
            return false;
        }
        HashValue(pipeline_hash, states.leftEyeStatusConfidence[i]);
        HashValue(pipeline_hash, states.rightEyeStatusConfidence[i]);
        HashValue(pipeline_hash, actions.normal[i]);
        HashValue(pipeline_hash, actions.shake[i]);
        HashValue(pipeline_hash, actions.jawOpen[i]);
        HashValue(pipeline_hash, actions.headRaise[i]);
        HashValue(pipeline_hash, actions.blink[i]);
        HashValue(pipeline_hash, emotions.emotion[i]);
    }
    return true;
}

bool RunScenario(const Scenario& scenario, const std::string& test_root, int warmup_iterations, int benchmark_iterations) {
    ImageHandles image;
    if (!LoadImage(JoinPath(test_root, scenario.relative_path), image)) {
        return false;
    }
    SessionHandle session;
    if (!CreateSession(scenario, session)) {
        return false;
    }

    bool scenario_ok = true;
    uint64_t sequence_hash = 1469598103934665603ULL;
    OwnedToken copied_token;
    HPoint2f copied_five_points[5] = {};
    std::vector<HPoint2f> copied_dense_points;

    HFMultipleFaceData first_faces = {0};
    const auto first_begin = Clock::now();
    HResult status = HFExecuteFaceTrack(session.value, image.stream, &first_faces);
    const auto first_end = Clock::now();
    const double first_us = std::chrono::duration<double, std::micro>(first_end - first_begin).count();
    uint64_t first_hash = 0;
    if (status != HSUCCEED || !ValidateAndHashFrame(scenario, first_faces, first_hash,
                                                    scenario.verify_dependent_apis ? &copied_token : nullptr,
                                                    scenario.verify_dependent_apis ? &copied_five_points : nullptr,
                                                    scenario.verify_dependent_apis ? &copied_dense_points : nullptr)) {
        scenario_ok = false;
    } else {
        HashValue(sequence_hash, first_hash);
    }

    HFMultipleFaceData second_faces = {0};
    if (scenario_ok) {
        status = HFExecuteFaceTrack(session.value, image.stream, &second_faces);
        uint64_t second_hash = 0;
        if (status != HSUCCEED || !ValidateAndHashFrame(scenario, second_faces, second_hash)) {
            scenario_ok = false;
        } else {
            HashValue(sequence_hash, second_hash);
        }
    }

    uint64_t dependent_hash = 0;
    uint64_t pipeline_hash = 0;
    if (scenario_ok && scenario.verify_dependent_apis) {
        if (copied_token.bytes.empty() ||
            !ValidateCopiedTokenAfterNextTrack(session.value, image.stream, copied_token, copied_five_points, copied_dense_points,
                                               dependent_hash) ||
            !ValidatePipelineLink(session.value, image.stream, second_faces, pipeline_hash)) {
            scenario_ok = false;
        }
    }

    for (int i = 0; i < warmup_iterations && scenario_ok; ++i) {
        HFMultipleFaceData faces = {0};
        status = HFExecuteFaceTrack(session.value, image.stream, &faces);
        uint64_t frame_hash = 0;
        if (status != HSUCCEED || !ValidateAndHashFrame(scenario, faces, frame_hash)) {
            scenario_ok = false;
            break;
        }
        HashValue(sequence_hash, frame_hash);
    }

    std::vector<double> samples_us;
    samples_us.reserve(static_cast<size_t>(benchmark_iterations));
    for (int i = 0; i < benchmark_iterations && scenario_ok; ++i) {
        HFMultipleFaceData faces = {0};
        const auto begin = Clock::now();
        status = HFExecuteFaceTrack(session.value, image.stream, &faces);
        const auto end = Clock::now();
        uint64_t frame_hash = 0;
        if (status != HSUCCEED || !ValidateAndHashFrame(scenario, faces, frame_hash)) {
            scenario_ok = false;
            break;
        }
        samples_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        HashValue(sequence_hash, frame_hash);
    }
    if (samples_us.size() != static_cast<size_t>(benchmark_iterations)) {
        scenario_ok = false;
    }

    std::cout << "CORRECTNESS"
              << ",scenario=" << scenario.name
              << ",mode=" << static_cast<int>(scenario.mode)
              << ",frames=" << 2 + warmup_iterations + static_cast<int>(samples_us.size())
              << ",first_faces=" << first_faces.detectedNum
              << ",second_faces=" << second_faces.detectedNum
              << ",sequence_digest=0x" << std::hex << sequence_hash << std::dec
              << ",status=" << (scenario_ok ? "PASS" : "FAIL") << '\n';
    if (scenario.verify_dependent_apis) {
        std::cout << "CAPI_CACHE"
                  << ",scenario=" << scenario.name
                  << ",copied_token_after_next_track=" << (scenario_ok ? "PASS" : "FAIL")
                  << ",dependent_digest=0x" << std::hex << dependent_hash
                  << ",pipeline_digest=0x" << pipeline_hash << std::dec
                  << ",status=" << (scenario_ok ? "PASS" : "FAIL") << '\n';
    }

    if (!samples_us.empty()) {
        const double total_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
        const double mean_us = total_us / static_cast<double>(samples_us.size());
        std::sort(samples_us.begin(), samples_us.end());
        std::cout << std::fixed << std::setprecision(3)
                  << "TIMING"
                  << ",scenario=" << scenario.name
                  << ",first_us=" << first_us
                  << ",iterations=" << benchmark_iterations
                  << ",mean_us=" << mean_us
                  << ",p50_us=" << Percentile(samples_us, 0.50)
                  << ",p95_us=" << Percentile(samples_us, 0.95)
                  << ",min_us=" << samples_us.front()
                  << ",max_us=" << samples_us.back() << '\n';
    }
    return scenario_ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <pack_path> [test_res_root] [iterations] [warmup_iterations]\n";
        return 2;
    }

    const std::string pack_path = argv[1];
    const std::string test_root = argc >= 3 ? argv[2] : "test_res";
    const int benchmark_iterations = argc >= 4 ? std::max(1, std::stoi(argv[3])) : 50;
    const int warmup_iterations = argc >= 5 ? std::max(0, std::stoi(argv[4])) : 5;

    const HResult launch_status = HFLaunchInspireFace(pack_path.c_str());
    if (launch_status != HSUCCEED) {
        std::cerr << "ERROR,reason=launch_failed,status=" << launch_status << ",pack=" << pack_path << '\n';
        return 3;
    }

    bool all_passed = true;
    for (const auto& scenario : kScenarios) {
        all_passed = RunScenario(scenario, test_root, warmup_iterations, benchmark_iterations) && all_passed;
    }

    const HResult terminate_status = HFTerminateInspireFace();
    if (terminate_status != HSUCCEED) {
        all_passed = false;
    }
    std::cout << "SUMMARY,status=" << (all_passed ? "PASS" : "FAIL")
              << ",scenarios=" << kScenarios.size()
              << ",iterations=" << benchmark_iterations
              << ",warmup=" << warmup_iterations << '\n';
    return all_passed ? 0 : 1;
}
