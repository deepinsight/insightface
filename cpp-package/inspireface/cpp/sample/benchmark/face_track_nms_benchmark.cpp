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

namespace {

using Clock = std::chrono::steady_clock;

struct Scenario {
    const char* name;
    const char* relative_path;
    int input_size;
    int detect_interval;
    int initial_min_faces;
    int initial_max_faces;
    int steady_min_faces;
    int steady_max_faces;
    bool require_stable_ids;
};

const std::vector<Scenario> kScenarios = {
  {"single_static", "data/bulk/kun.jpg", 320, 5, 1, 1, 1, 1, true},
  {"multi_static", "data/bulk/pedestrian.png", 640, 1, 16, 20, 16, 20, true},
  {"no_face_static", "data/bulk/view.jpg", 320, 1, 0, 0, 0, 0, true},
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

bool HashAndValidateFrame(const HFMultipleFaceData& faces, int minimum_faces, int maximum_faces, uint64_t& frame_hash) {
    if (faces.detectedNum < minimum_faces || faces.detectedNum > maximum_faces) {
        return false;
    }

    frame_hash = 1469598103934665603ULL;
    const uint64_t count = static_cast<uint64_t>(faces.detectedNum);
    HashBytes(frame_hash, &count, sizeof(count));
    std::set<int> track_ids;
    for (int i = 0; i < faces.detectedNum; ++i) {
        const HFaceRect& rect = faces.rects[i];
        if (rect.width <= 0 || rect.height <= 0 || faces.trackCounts[i] < 0 || !std::isfinite(faces.detConfidence[i]) ||
            !std::isfinite(faces.angles.roll[i]) ||
            !std::isfinite(faces.angles.yaw[i]) || !std::isfinite(faces.angles.pitch[i])) {
            return false;
        }
        if (!track_ids.insert(faces.trackIds[i]).second) {
            return false;
        }

        HashBytes(frame_hash, &rect, sizeof(rect));
        HashBytes(frame_hash, &faces.trackIds[i], sizeof(faces.trackIds[i]));
        HashBytes(frame_hash, &faces.trackCounts[i], sizeof(faces.trackCounts[i]));
        HashBytes(frame_hash, &faces.detConfidence[i], sizeof(faces.detConfidence[i]));
        HashBytes(frame_hash, &faces.angles.roll[i], sizeof(faces.angles.roll[i]));
        HashBytes(frame_hash, &faces.angles.yaw[i], sizeof(faces.angles.yaw[i]));
        HashBytes(frame_hash, &faces.angles.pitch[i], sizeof(faces.angles.pitch[i]));

        HPoint2f key_points[5] = {};
        if (HFGetFaceFiveKeyPointsFromFaceToken(faces.tokens[i], key_points, 5) != HSUCCEED) {
            return false;
        }
        for (const auto& point : key_points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                return false;
            }
        }
        HashBytes(frame_hash, key_points, sizeof(key_points));
    }
    return true;
}

bool ValidateStableTracking(const HFMultipleFaceData& faces, const std::set<int>& initial_track_ids, std::map<int, int>& last_track_counts) {
    for (int i = 0; i < faces.detectedNum; ++i) {
        const int track_id = faces.trackIds[i];
        const int track_count = faces.trackCounts[i];
        if (initial_track_ids.count(track_id) == 0) {
            return false;
        }
        const auto previous = last_track_counts.find(track_id);
        if (previous != last_track_counts.end() && track_count < previous->second) {
            return false;
        }
        last_track_counts[track_id] = track_count;
    }
    return true;
}

bool ConfigureSession(const Scenario& scenario, HFSession& session) {
    HFSessionCustomParameter parameter = {0};
    HResult status = HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_LIGHT_TRACK, 25, scenario.input_size, -1, &session);
    if (status != HSUCCEED) {
        std::cerr << "ERROR,scenario=" << scenario.name << ",reason=session_create_failed,status=" << status << '\n';
        return false;
    }

    const HResult preview_status = HFSessionSetTrackPreviewSize(session, scenario.input_size);
    const HResult filter_status = HFSessionSetFilterMinimumFacePixelSize(session, 0);
    const HResult interval_status = HFSessionSetTrackModeDetectInterval(session, scenario.detect_interval);
    if (preview_status != HSUCCEED || filter_status != HSUCCEED || interval_status != HSUCCEED) {
        std::cerr << "ERROR,scenario=" << scenario.name << ",reason=session_config_failed,preview_status=" << preview_status
                  << ",filter_status=" << filter_status << ",interval_status=" << interval_status << '\n';
        HFReleaseInspireFaceSession(session);
        session = nullptr;
        return false;
    }
    return true;
}

bool RunScenario(const Scenario& scenario, const std::string& test_root, int warmup_iterations, int benchmark_iterations) {
    const std::string image_path = JoinPath(test_root, scenario.relative_path);
    HFImageBitmap bitmap = nullptr;
    HResult status = HFCreateImageBitmapFromFilePath(image_path.c_str(), 3, &bitmap);
    if (status != HSUCCEED) {
        std::cerr << "ERROR,scenario=" << scenario.name << ",reason=load_failed,status=" << status << ",path=" << image_path << '\n';
        return false;
    }

    HFImageStream stream = nullptr;
    status = HFCreateImageStreamFromImageBitmap(bitmap, HF_CAMERA_ROTATION_0, &stream);
    if (status != HSUCCEED) {
        std::cerr << "ERROR,scenario=" << scenario.name << ",reason=stream_create_failed,status=" << status << '\n';
        HFReleaseImageBitmap(bitmap);
        return false;
    }

    HFSession session = nullptr;
    if (!ConfigureSession(scenario, session)) {
        HFReleaseImageStream(stream);
        HFReleaseImageBitmap(bitmap);
        return false;
    }

    bool scenario_ok = true;
    uint64_t sequence_hash = 1469598103934665603ULL;
    int observed_min_faces = 25;
    int observed_max_faces = 0;
    std::set<int> initial_track_ids;
    std::map<int, int> last_track_counts;

    HFMultipleFaceData first_faces = {0};
    const auto first_begin = Clock::now();
    status = HFExecuteFaceTrack(session, stream, &first_faces);
    const auto first_end = Clock::now();
    const double first_us = std::chrono::duration<double, std::micro>(first_end - first_begin).count();
    uint64_t first_hash = 0;
    if (status != HSUCCEED ||
        !HashAndValidateFrame(first_faces, scenario.initial_min_faces, scenario.initial_max_faces, first_hash)) {
        scenario_ok = false;
    } else {
        HashBytes(sequence_hash, &first_hash, sizeof(first_hash));
        observed_min_faces = std::min(observed_min_faces, first_faces.detectedNum);
        observed_max_faces = std::max(observed_max_faces, first_faces.detectedNum);
        for (int i = 0; i < first_faces.detectedNum; ++i) {
            initial_track_ids.insert(first_faces.trackIds[i]);
            last_track_counts[first_faces.trackIds[i]] = first_faces.trackCounts[i];
        }
    }

    for (int i = 0; i < warmup_iterations && scenario_ok; ++i) {
        HFMultipleFaceData faces = {0};
        status = HFExecuteFaceTrack(session, stream, &faces);
        uint64_t frame_hash = 0;
        if (status != HSUCCEED || !HashAndValidateFrame(faces, scenario.steady_min_faces, scenario.steady_max_faces, frame_hash)) {
            scenario_ok = false;
            break;
        }
        if (scenario.require_stable_ids && !ValidateStableTracking(faces, initial_track_ids, last_track_counts)) {
            scenario_ok = false;
        }
        HashBytes(sequence_hash, &frame_hash, sizeof(frame_hash));
        observed_min_faces = std::min(observed_min_faces, faces.detectedNum);
        observed_max_faces = std::max(observed_max_faces, faces.detectedNum);
    }

    std::vector<double> samples_us;
    samples_us.reserve(benchmark_iterations);
    for (int i = 0; i < benchmark_iterations && scenario_ok; ++i) {
        HFMultipleFaceData faces = {0};
        const auto begin = Clock::now();
        status = HFExecuteFaceTrack(session, stream, &faces);
        const auto end = Clock::now();
        uint64_t frame_hash = 0;
        if (status != HSUCCEED || !HashAndValidateFrame(faces, scenario.steady_min_faces, scenario.steady_max_faces, frame_hash)) {
            scenario_ok = false;
            break;
        }
        if (scenario.require_stable_ids && !ValidateStableTracking(faces, initial_track_ids, last_track_counts)) {
            scenario_ok = false;
        }
        samples_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        HashBytes(sequence_hash, &frame_hash, sizeof(frame_hash));
        observed_min_faces = std::min(observed_min_faces, faces.detectedNum);
        observed_max_faces = std::max(observed_max_faces, faces.detectedNum);
    }

    if (samples_us.size() != static_cast<size_t>(benchmark_iterations)) {
        scenario_ok = false;
    }

    std::cout << "CORRECTNESS"
              << ",scenario=" << scenario.name
              << ",frames=" << 1 + warmup_iterations + static_cast<int>(samples_us.size())
              << ",first_faces=" << first_faces.detectedNum
              << ",observed_faces=" << observed_min_faces << '-' << observed_max_faces
              << ",stable_ids=" << (scenario.require_stable_ids ? (scenario_ok ? "PASS" : "FAIL") : "CHECKED_UNIQUE")
              << ",sequence_digest=0x" << std::hex << sequence_hash << std::dec
              << ",status=" << (scenario_ok ? "PASS" : "FAIL") << '\n';

    if (!samples_us.empty()) {
        const double total_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
        const double mean_us = total_us / static_cast<double>(samples_us.size());
        std::sort(samples_us.begin(), samples_us.end());
        std::cout << std::fixed << std::setprecision(3)
                  << "TIMING"
                  << ",scenario=" << scenario.name
                  << ",detect_interval=" << scenario.detect_interval
                  << ",first_us=" << first_us
                  << ",iterations=" << benchmark_iterations
                  << ",mean_us=" << mean_us
                  << ",p50_us=" << Percentile(samples_us, 0.50)
                  << ",p95_us=" << Percentile(samples_us, 0.95)
                  << ",min_us=" << samples_us.front()
                  << ",max_us=" << samples_us.back() << '\n';
    }

    const HResult session_release_status = HFReleaseInspireFaceSession(session);
    const HResult stream_release_status = HFReleaseImageStream(stream);
    const HResult bitmap_release_status = HFReleaseImageBitmap(bitmap);
    if (session_release_status != HSUCCEED || stream_release_status != HSUCCEED || bitmap_release_status != HSUCCEED) {
        scenario_ok = false;
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
