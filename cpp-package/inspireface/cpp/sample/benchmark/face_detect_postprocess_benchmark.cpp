#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <inspireface.h>

namespace {

using Clock = std::chrono::steady_clock;

struct ExpectedCount {
    int minimum;
    int maximum;
};

struct ImageCase {
    const char* name;
    const char* relative_path;
    ExpectedCount expected_160;
    ExpectedCount expected_320;
    ExpectedCount expected_640;
};

struct DetectorConfig {
    int input_size;
};

const std::vector<ImageCase> kImageCases = {
  {"no_face", "data/bulk/view.jpg", {0, 0}, {0, 0}, {0, 0}},
  {"single_frontal", "data/bulk/kun.jpg", {1, 1}, {1, 1}, {1, 1}},
  {"single_profile", "data/pose/right_face.png", {1, 1}, {1, 1}, {1, 1}},
  {"single_raise_head", "data/pose/rise_face.jpeg", {1, 1}, {1, 1}, {1, 1}},
  {"multi_pedestrian", "data/bulk/pedestrian.png", {1, 6}, {10, 11}, {16, 20}},
};

const std::vector<DetectorConfig> kDetectorConfigs = {
  {160},
  {320},
  {640},
};

ExpectedCount GetExpectedCount(const ImageCase& image_case, int input_size) {
    if (input_size == 160) {
        return image_case.expected_160;
    }
    if (input_size == 320) {
        return image_case.expected_320;
    }
    return image_case.expected_640;
}

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

bool HashFaces(const HFMultipleFaceData& faces, uint64_t& hash) {
    hash = 1469598103934665603ULL;
    const uint64_t count = static_cast<uint64_t>(faces.detectedNum);
    HashBytes(hash, &count, sizeof(count));
    for (int i = 0; i < faces.detectedNum; ++i) {
        HashBytes(hash, &faces.rects[i], sizeof(faces.rects[i]));
        HashBytes(hash, &faces.detConfidence[i], sizeof(faces.detConfidence[i]));
        HashBytes(hash, &faces.angles.roll[i], sizeof(faces.angles.roll[i]));
        HashBytes(hash, &faces.angles.yaw[i], sizeof(faces.angles.yaw[i]));
        HashBytes(hash, &faces.angles.pitch[i], sizeof(faces.angles.pitch[i]));

        HPoint2f key_points[5] = {};
        if (HFGetFaceFiveKeyPointsFromFaceToken(faces.tokens[i], key_points, 5) != HSUCCEED) {
            return false;
        }
        HashBytes(hash, key_points, sizeof(key_points));
    }
    return true;
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

bool RunCase(HFSession session, int input_size, const ImageCase& image_case, const std::string& test_root, int warmup_iterations,
             int benchmark_iterations) {
    const std::string image_path = JoinPath(test_root, image_case.relative_path);
    HFImageBitmap bitmap = nullptr;
    HResult status = HFCreateImageBitmapFromFilePath(image_path.c_str(), 3, &bitmap);
    if (status != HSUCCEED) {
        std::cerr << "ERROR,image=" << image_case.name << ",reason=load_failed,status=" << status << ",path=" << image_path << '\n';
        return false;
    }

    HFImageStream stream = nullptr;
    status = HFCreateImageStreamFromImageBitmap(bitmap, HF_CAMERA_ROTATION_0, &stream);
    if (status != HSUCCEED) {
        std::cerr << "ERROR,image=" << image_case.name << ",reason=stream_create_failed,status=" << status << '\n';
        HFReleaseImageBitmap(bitmap);
        return false;
    }

    HFMultipleFaceData reference = {0};
    status = HFExecuteFaceTrack(session, stream, &reference);
    if (status != HSUCCEED) {
        std::cerr << "ERROR,image=" << image_case.name << ",reason=detect_failed,status=" << status << '\n';
        HFReleaseImageStream(stream);
        HFReleaseImageBitmap(bitmap);
        return false;
    }
    uint64_t reference_hash = 0;
    if (!HashFaces(reference, reference_hash)) {
        std::cerr << "ERROR,image=" << image_case.name << ",reason=result_hash_failed\n";
        HFReleaseImageStream(stream);
        HFReleaseImageBitmap(bitmap);
        return false;
    }
    const ExpectedCount expected = GetExpectedCount(image_case, input_size);
    const bool count_ok = reference.detectedNum >= expected.minimum && reference.detectedNum <= expected.maximum;

    bool repeat_ok = true;
    for (int i = 0; i < 3; ++i) {
        HFMultipleFaceData repeated = {0};
        status = HFExecuteFaceTrack(session, stream, &repeated);
        uint64_t repeated_hash = 0;
        if (status != HSUCCEED || !HashFaces(repeated, repeated_hash) || repeated_hash != reference_hash) {
            repeat_ok = false;
            break;
        }
    }

    std::cout << "CORRECTNESS"
              << ",size=" << input_size
              << ",image=" << image_case.name
              << ",faces=" << reference.detectedNum
              << ",expected=" << expected.minimum << '-' << expected.maximum
              << ",digest=0x" << std::hex << reference_hash << std::dec
              << ",repeat=" << (repeat_ok ? "PASS" : "FAIL")
              << ",status=" << (count_ok && repeat_ok ? "PASS" : "FAIL") << '\n';

    for (int i = 0; i < warmup_iterations; ++i) {
        HFMultipleFaceData ignored = {0};
        status = HFExecuteFaceTrack(session, stream, &ignored);
        if (status != HSUCCEED) {
            std::cerr << "ERROR,image=" << image_case.name << ",reason=warmup_detect_failed,status=" << status << '\n';
            HFReleaseImageStream(stream);
            HFReleaseImageBitmap(bitmap);
            return false;
        }
    }

    std::vector<double> samples_us;
    samples_us.reserve(benchmark_iterations);
    uint64_t benchmark_hash = 1469598103934665603ULL;
    for (int i = 0; i < benchmark_iterations; ++i) {
        HFMultipleFaceData faces = {0};
        const auto begin = Clock::now();
        status = HFExecuteFaceTrack(session, stream, &faces);
        const auto end = Clock::now();
        if (status != HSUCCEED) {
            std::cerr << "ERROR,image=" << image_case.name << ",reason=benchmark_detect_failed,status=" << status << '\n';
            HFReleaseImageStream(stream);
            HFReleaseImageBitmap(bitmap);
            return false;
        }
        samples_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        uint64_t iteration_hash = 0;
        if (!HashFaces(faces, iteration_hash)) {
            std::cerr << "ERROR,image=" << image_case.name << ",reason=benchmark_hash_failed\n";
            HFReleaseImageStream(stream);
            HFReleaseImageBitmap(bitmap);
            return false;
        }
        HashBytes(benchmark_hash, &iteration_hash, sizeof(iteration_hash));
    }

    const double total_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
    const double mean_us = total_us / static_cast<double>(samples_us.size());
    std::sort(samples_us.begin(), samples_us.end());

    std::cout << std::fixed << std::setprecision(3)
              << "TIMING"
              << ",size=" << input_size
              << ",image=" << image_case.name
              << ",iterations=" << benchmark_iterations
              << ",total_us=" << total_us
              << ",mean_us=" << mean_us
              << ",p50_us=" << Percentile(samples_us, 0.50)
              << ",p95_us=" << Percentile(samples_us, 0.95)
              << ",min_us=" << samples_us.front()
              << ",max_us=" << samples_us.back()
              << ",run_digest=0x" << std::hex << benchmark_hash << std::dec << '\n';

    const HResult stream_release_status = HFReleaseImageStream(stream);
    const HResult bitmap_release_status = HFReleaseImageBitmap(bitmap);
    if (stream_release_status != HSUCCEED || bitmap_release_status != HSUCCEED) {
        std::cerr << "ERROR,image=" << image_case.name << ",reason=image_release_failed,stream_status=" << stream_release_status
                  << ",bitmap_status=" << bitmap_release_status << '\n';
        return false;
    }
    return count_ok && repeat_ok;
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
    for (const auto& config : kDetectorConfigs) {
        HFSessionCustomParameter parameter = {0};
        HFSession session = nullptr;
        const HResult create_status =
          HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 25, config.input_size, -1, &session);
        if (create_status != HSUCCEED) {
            std::cerr << "ERROR,reason=session_create_failed,size=" << config.input_size << ",status=" << create_status << '\n';
            all_passed = false;
            continue;
        }
        const HResult preview_status = HFSessionSetTrackPreviewSize(session, config.input_size);
        const HResult filter_status = HFSessionSetFilterMinimumFacePixelSize(session, 0);
        if (preview_status != HSUCCEED || filter_status != HSUCCEED) {
            std::cerr << "ERROR,reason=session_config_failed,size=" << config.input_size << ",preview_status=" << preview_status
                      << ",filter_status=" << filter_status << '\n';
            HFReleaseInspireFaceSession(session);
            all_passed = false;
            continue;
        }

        for (const auto& image_case : kImageCases) {
            all_passed = RunCase(session, config.input_size, image_case, test_root, warmup_iterations, benchmark_iterations) && all_passed;
        }

        const HResult release_status = HFReleaseInspireFaceSession(session);
        if (release_status != HSUCCEED) {
            std::cerr << "ERROR,reason=session_release_failed,size=" << config.input_size << ",status=" << release_status << '\n';
            all_passed = false;
        }
    }

    const HResult terminate_status = HFTerminateInspireFace();
    if (terminate_status != HSUCCEED) {
        std::cerr << "ERROR,reason=terminate_failed,status=" << terminate_status << '\n';
        all_passed = false;
    }
    std::cout << "SUMMARY,status=" << (all_passed ? "PASS" : "FAIL")
              << ",detector_sizes=" << kDetectorConfigs.size()
              << ",image_cases=" << kImageCases.size()
              << ",iterations=" << benchmark_iterations
              << ",warmup=" << warmup_iterations << '\n';
    return all_passed ? 0 : 1;
}
