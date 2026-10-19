#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include <inspireface.h>

#include "inspireface/common/face_info/face_object_internal.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kLandmarkCount = 106;
constexpr int kInternalLandmarkCount = 116;
constexpr int kGeneratedFrameCount = 257;

struct SmoothScenario {
    const char* name;
    int cache_frames;
    float smooth_ratio;
};

struct TrackingScenario {
    const char* name;
    const char* relative_path;
    int input_size;
    int detect_interval;
    int cache_frames;
    float smooth_ratio;
    int minimum_faces;
    int maximum_faces;
};

const std::vector<SmoothScenario> kSmoothScenarios = {
  {"smooth_n1_h0", 1, 0.0f},
  {"smooth_n5_default", 5, 0.05f},
  {"smooth_n8_h020", 8, 0.20f},
};

const std::vector<TrackingScenario> kTrackingScenarios = {
  {"frontal_n1", "data/bulk/kun.jpg", 320, 5, 1, 0.0f, 1, 1},
  {"profile_n5", "data/bulk/jntm.jpg", 320, 5, 5, 0.05f, 1, 1},
  {"woman_n8", "data/bulk/woman.png", 320, 5, 8, 0.20f, 1, 1},
  {"multi_n5", "data/bulk/pedestrian.png", 640, 1, 5, 0.05f, 16, 20},
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

std::vector<std::vector<inspirecv::Point2f>> GenerateLandmarkFrames() {
    std::vector<std::vector<inspirecv::Point2f>> frames(
      kGeneratedFrameCount, std::vector<inspirecv::Point2f>(kInternalLandmarkCount));
    for (int frame = 0; frame < kGeneratedFrameCount; ++frame) {
        for (int point = 0; point < kInternalLandmarkCount; ++point) {
            const float phase = static_cast<float>(frame * 13 + point * 7);
            const float x = static_cast<float>(point) * 0.73f + static_cast<float>((frame + point * 3) % 17) * 0.11f +
                            std::sin(phase * 0.017f) * 2.25f;
            const float y = static_cast<float>(point) * 0.41f + static_cast<float>((frame * 5 + point) % 23) * 0.09f +
                            std::cos(phase * 0.013f) * 1.75f;
            frames[frame][point] = inspirecv::Point2f(x, y);
        }
    }
    return frames;
}

void HashSmoothingState(uint64_t& hash, const std::vector<inspirecv::Point2f>& landmarks,
                        const std::vector<std::vector<inspirecv::Point2f>>& history) {
    HashValue(hash, history.size());
    for (int i = 0; i < kLandmarkCount; ++i) {
        const float x = landmarks[i].GetX();
        const float y = landmarks[i].GetY();
        HashValue(hash, x);
        HashValue(hash, y);
    }
    for (const auto& frame : history) {
        HashValue(hash, frame.size());
        for (const auto& point : frame) {
            const float x = point.GetX();
            const float y = point.GetY();
            HashValue(hash, x);
            HashValue(hash, y);
        }
    }
}

bool RunSmoothingScenario(const SmoothScenario& scenario, const std::vector<std::vector<inspirecv::Point2f>>& generated_frames,
                          int benchmark_iterations) {
    inspire::FaceObjectInternal smoother(1, inspirecv::Rect2i(0, 0, 100, 100), kInternalLandmarkCount);
    std::vector<std::vector<inspirecv::Point2f>> history;
    std::vector<inspirecv::Point2f> landmarks(kInternalLandmarkCount);
    uint64_t sequence_hash = 1469598103934665603ULL;

    const int warmup_iterations = scenario.cache_frames + 4;
    for (int i = 0; i < warmup_iterations; ++i) {
        landmarks = generated_frames[static_cast<size_t>(i) % generated_frames.size()];
        smoother.DynamicSmoothParamUpdate(landmarks, history, kLandmarkCount * 2, scenario.smooth_ratio, scenario.cache_frames);
        const size_t expected_history_size = static_cast<size_t>(std::min(i + 1, scenario.cache_frames));
        if (history.size() != expected_history_size || history.back().size() != kLandmarkCount) {
            return false;
        }
        HashSmoothingState(sequence_hash, landmarks, history);
    }

    std::vector<double> samples_ns;
    samples_ns.reserve(static_cast<size_t>(benchmark_iterations));
    for (int i = 0; i < benchmark_iterations; ++i) {
        landmarks = generated_frames[static_cast<size_t>(i + warmup_iterations) % generated_frames.size()];
        const auto begin = Clock::now();
        smoother.DynamicSmoothParamUpdate(landmarks, history, kLandmarkCount * 2, scenario.smooth_ratio, scenario.cache_frames);
        const auto end = Clock::now();
        if (history.size() != static_cast<size_t>(scenario.cache_frames) || history.back().size() != kLandmarkCount) {
            return false;
        }
        samples_ns.push_back(std::chrono::duration<double, std::nano>(end - begin).count());
        HashSmoothingState(sequence_hash, landmarks, history);
    }

    const double total_ns = std::accumulate(samples_ns.begin(), samples_ns.end(), 0.0);
    const double mean_ns = total_ns / static_cast<double>(samples_ns.size());
    std::sort(samples_ns.begin(), samples_ns.end());
    std::cout << "SMOOTH_CORRECTNESS"
              << ",scenario=" << scenario.name
              << ",cache_frames=" << scenario.cache_frames
              << ",ratio=" << scenario.smooth_ratio
              << ",sequence_digest=0x" << std::hex << sequence_hash << std::dec
              << ",status=PASS\n";
    std::cout << std::fixed << std::setprecision(3)
              << "SMOOTH_TIMING"
              << ",scenario=" << scenario.name
              << ",iterations=" << benchmark_iterations
              << ",mean_ns=" << mean_ns
              << ",p50_ns=" << Percentile(samples_ns, 0.50)
              << ",p95_ns=" << Percentile(samples_ns, 0.95)
              << ",min_ns=" << samples_ns.front()
              << ",max_ns=" << samples_ns.back() << '\n';
    return true;
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

bool CreateSession(const TrackingScenario& scenario, SessionHandle& session) {
    HFSessionCustomParameter parameter = {0};
    parameter.enable_detect_mode_landmark = 1;
    HResult status = HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_LIGHT_TRACK, 25, scenario.input_size, -1, &session.value);
    if (status != HSUCCEED) {
        std::cerr << "ERROR,scenario=" << scenario.name << ",reason=session_create_failed,status=" << status << '\n';
        return false;
    }
    const HResult preview_status = HFSessionSetTrackPreviewSize(session.value, scenario.input_size);
    const HResult filter_status = HFSessionSetFilterMinimumFacePixelSize(session.value, 0);
    const HResult interval_status = HFSessionSetTrackModeDetectInterval(session.value, scenario.detect_interval);
    const HResult ratio_status = HFSessionSetTrackModeSmoothRatio(session.value, scenario.smooth_ratio);
    const HResult cache_status = HFSessionSetTrackModeNumSmoothCacheFrame(session.value, scenario.cache_frames);
    if (preview_status != HSUCCEED || filter_status != HSUCCEED || interval_status != HSUCCEED || ratio_status != HSUCCEED ||
        cache_status != HSUCCEED) {
        std::cerr << "ERROR,scenario=" << scenario.name << ",reason=session_config_failed,preview_status=" << preview_status
                  << ",filter_status=" << filter_status << ",interval_status=" << interval_status << ",ratio_status=" << ratio_status
                  << ",cache_status=" << cache_status << '\n';
        return false;
    }
    return true;
}

bool HashAndValidateFrame(const TrackingScenario& scenario, const HFMultipleFaceData& faces, const std::set<int>& initial_track_ids,
                          std::map<int, int>& last_track_counts, uint64_t& frame_hash) {
    if (faces.detectedNum < scenario.minimum_faces || faces.detectedNum > scenario.maximum_faces) {
        return false;
    }
    if (faces.detectedNum > 0 &&
        (faces.rects == nullptr || faces.trackIds == nullptr || faces.trackCounts == nullptr || faces.detConfidence == nullptr ||
         faces.angles.roll == nullptr || faces.angles.yaw == nullptr || faces.angles.pitch == nullptr || faces.tokens == nullptr)) {
        return false;
    }

    HInt32 token_size = 0;
    HInt32 dense_count = 0;
    if (HFGetFaceBasicTokenSize(&token_size) != HSUCCEED || token_size <= 0 || HFGetNumOfFaceDenseLandmark(&dense_count) != HSUCCEED ||
        dense_count != kLandmarkCount) {
        return false;
    }

    frame_hash = 1469598103934665603ULL;
    HashValue(frame_hash, faces.detectedNum);
    std::set<int> frame_track_ids;
    for (int i = 0; i < faces.detectedNum; ++i) {
        const HFaceRect& rect = faces.rects[i];
        if (rect.width <= 0 || rect.height <= 0 || faces.trackCounts[i] < 0 || !std::isfinite(faces.detConfidence[i]) ||
            !std::isfinite(faces.angles.roll[i]) || !std::isfinite(faces.angles.yaw[i]) || !std::isfinite(faces.angles.pitch[i]) ||
            faces.tokens[i].size != token_size || faces.tokens[i].data == nullptr || !frame_track_ids.insert(faces.trackIds[i]).second) {
            return false;
        }
        if (!initial_track_ids.empty() && initial_track_ids.count(faces.trackIds[i]) == 0) {
            return false;
        }
        const auto previous = last_track_counts.find(faces.trackIds[i]);
        if (previous != last_track_counts.end() && faces.trackCounts[i] < previous->second) {
            return false;
        }
        last_track_counts[faces.trackIds[i]] = faces.trackCounts[i];

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

        std::vector<char> token_bytes(static_cast<size_t>(token_size));
        if (HFCopyFaceBasicToken(faces.tokens[i], token_bytes.data(), token_size) != HSUCCEED) {
            return false;
        }
        HashBytes(frame_hash, token_bytes.data(), token_bytes.size());

        HPoint2f five_points[5] = {};
        std::vector<HPoint2f> dense_points(static_cast<size_t>(dense_count));
        if (HFGetFaceFiveKeyPointsFromFaceToken(faces.tokens[i], five_points, 5) != HSUCCEED ||
            HFGetFaceDenseLandmarkFromFaceToken(faces.tokens[i], dense_points.data(), dense_count) != HSUCCEED) {
            return false;
        }
        for (const auto& point : five_points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                return false;
            }
            HashValue(frame_hash, point.x);
            HashValue(frame_hash, point.y);
        }
        for (const auto& point : dense_points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                return false;
            }
            HashValue(frame_hash, point.x);
            HashValue(frame_hash, point.y);
        }
    }
    return true;
}

bool RunTrackingScenario(const TrackingScenario& scenario, const std::string& test_root, int warmup_iterations, int benchmark_iterations) {
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
    std::set<int> initial_track_ids;
    std::map<int, int> last_track_counts;
    int observed_min_faces = scenario.maximum_faces;
    int observed_max_faces = 0;

    HFMultipleFaceData first_faces = {0};
    const auto first_begin = Clock::now();
    HResult status = HFExecuteFaceTrack(session.value, image.stream, &first_faces);
    const auto first_end = Clock::now();
    const double first_us = std::chrono::duration<double, std::micro>(first_end - first_begin).count();
    uint64_t first_hash = 0;
    if (status != HSUCCEED || !HashAndValidateFrame(scenario, first_faces, {}, last_track_counts, first_hash)) {
        scenario_ok = false;
    } else {
        HashValue(sequence_hash, first_hash);
        observed_min_faces = first_faces.detectedNum;
        observed_max_faces = first_faces.detectedNum;
        for (int i = 0; i < first_faces.detectedNum; ++i) {
            initial_track_ids.insert(first_faces.trackIds[i]);
        }
    }

    for (int i = 0; i < warmup_iterations && scenario_ok; ++i) {
        HFMultipleFaceData faces = {0};
        status = HFExecuteFaceTrack(session.value, image.stream, &faces);
        uint64_t frame_hash = 0;
        if (status != HSUCCEED || !HashAndValidateFrame(scenario, faces, initial_track_ids, last_track_counts, frame_hash)) {
            scenario_ok = false;
            break;
        }
        HashValue(sequence_hash, frame_hash);
        observed_min_faces = std::min(observed_min_faces, faces.detectedNum);
        observed_max_faces = std::max(observed_max_faces, faces.detectedNum);
    }

    std::vector<double> samples_us;
    samples_us.reserve(static_cast<size_t>(benchmark_iterations));
    for (int i = 0; i < benchmark_iterations && scenario_ok; ++i) {
        HFMultipleFaceData faces = {0};
        const auto begin = Clock::now();
        status = HFExecuteFaceTrack(session.value, image.stream, &faces);
        const auto end = Clock::now();
        uint64_t frame_hash = 0;
        if (status != HSUCCEED || !HashAndValidateFrame(scenario, faces, initial_track_ids, last_track_counts, frame_hash)) {
            scenario_ok = false;
            break;
        }
        samples_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        HashValue(sequence_hash, frame_hash);
        observed_min_faces = std::min(observed_min_faces, faces.detectedNum);
        observed_max_faces = std::max(observed_max_faces, faces.detectedNum);
    }
    if (samples_us.size() != static_cast<size_t>(benchmark_iterations)) {
        scenario_ok = false;
    }

    std::cout << "TRACK_CORRECTNESS"
              << ",scenario=" << scenario.name
              << ",frames=" << 1 + warmup_iterations + static_cast<int>(samples_us.size())
              << ",first_faces=" << first_faces.detectedNum
              << ",observed_faces=" << observed_min_faces << '-' << observed_max_faces
              << ",sequence_digest=0x" << std::hex << sequence_hash << std::dec
              << ",status=" << (scenario_ok ? "PASS" : "FAIL") << '\n';
    if (!samples_us.empty()) {
        const double total_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
        const double mean_us = total_us / static_cast<double>(samples_us.size());
        std::sort(samples_us.begin(), samples_us.end());
        std::cout << std::fixed << std::setprecision(3)
                  << "TRACK_TIMING"
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
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <pack_path> [test_res_root] [tracking_iterations] [tracking_warmups] [smoothing_iterations]\n";
        return 2;
    }

    const std::string pack_path = argv[1];
    const std::string test_root = argc >= 3 ? argv[2] : "test_res";
    const int tracking_iterations = argc >= 4 ? std::max(1, std::stoi(argv[3])) : 50;
    const int tracking_warmups = argc >= 5 ? std::max(0, std::stoi(argv[4])) : 8;
    const int smoothing_iterations = argc >= 6 ? std::max(1, std::stoi(argv[5])) : 10000;

    const HResult launch_status = HFLaunchInspireFace(pack_path.c_str());
    if (launch_status != HSUCCEED) {
        std::cerr << "ERROR,reason=launch_failed,status=" << launch_status << ",pack=" << pack_path << '\n';
        return 3;
    }

    const auto generated_frames = GenerateLandmarkFrames();
    bool all_passed = true;
    for (const auto& scenario : kSmoothScenarios) {
        all_passed = RunSmoothingScenario(scenario, generated_frames, smoothing_iterations) && all_passed;
    }
    for (const auto& scenario : kTrackingScenarios) {
        all_passed = RunTrackingScenario(scenario, test_root, tracking_warmups, tracking_iterations) && all_passed;
    }

    const HResult terminate_status = HFTerminateInspireFace();
    if (terminate_status != HSUCCEED) {
        all_passed = false;
    }
    std::cout << "SUMMARY,status=" << (all_passed ? "PASS" : "FAIL")
              << ",smooth_scenarios=" << kSmoothScenarios.size()
              << ",track_scenarios=" << kTrackingScenarios.size()
              << ",tracking_iterations=" << tracking_iterations
              << ",tracking_warmups=" << tracking_warmups
              << ",smoothing_iterations=" << smoothing_iterations << '\n';
    return all_passed ? 0 : 1;
}
