#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <inspireface.h>

#include "middleware/model_archive/core_archive/microtar/microtar.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

enum class FixtureKind {
    kMissingTracker,
    kMissingLandmark,
    kMissingRefine,
    kMissingPose,
    kMissingPipeline,
    kMissingLiveness,
    kMissingAttribute,
    kMissingBlink,
    kMissingEmotion,
    kMissingRecognition,
    kCorruptTracker,
    kMalformedTracker,
    kUnsupportedBackend,
};

struct SessionHandle {
    HFSession value = nullptr;

    ~SessionHandle() {
        if (value != nullptr) {
            HFReleaseInspireFaceSession(value);
        }
    }

    void Reset() {
        if (value != nullptr) {
            HFReleaseInspireFaceSession(value);
            value = nullptr;
        }
    }
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

struct TemporaryFile {
    std::string path;

    ~TemporaryFile() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
};

void HashBytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

template <typename T>
void HashValue(uint64_t& hash, const T& value) {
    HashBytes(hash, &value, sizeof(value));
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

void PrintTiming(const char* label, std::vector<double> samples_ms) {
    std::sort(samples_ms.begin(), samples_ms.end());
    const double mean = samples_ms.empty() ? 0.0 :
                                                   std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) /
                                                     static_cast<double>(samples_ms.size());
    std::cout << std::fixed << std::setprecision(3) << label << ",samples=" << samples_ms.size() << ",mean_ms=" << mean
              << ",p50_ms=" << Percentile(samples_ms, 0.50) << ",p95_ms=" << Percentile(samples_ms, 0.95) << '\n';
}

bool ReplaceInSection(std::string& manifest, const std::string& section, const std::string& next_section,
                      const std::string& from, const std::string& to) {
    const size_t begin = manifest.find(section);
    if (begin == std::string::npos) {
        return false;
    }
    const size_t end = next_section.empty() ? manifest.size() : manifest.find(next_section, begin + section.size());
    if (end == std::string::npos) {
        return false;
    }
    const size_t position = manifest.find(from, begin + section.size());
    if (position == std::string::npos || position >= end) {
        return false;
    }
    manifest.replace(position, from.size(), to);
    return true;
}

bool MutateManifest(std::string& manifest, FixtureKind kind, std::string& extra_entry_name) {
    switch (kind) {
        case FixtureKind::kMissingTracker:
            return ReplaceInSection(manifest, "face_detect_320:", "face_detect_640:",
                                    "name: _00_scrfd_500m_bnkps_shape320x320_fp16", "name: __guard_missing_detector");
        case FixtureKind::kMissingLandmark:
            return ReplaceInSection(manifest, "landmark:", "refine_net:", "name: _01_hyplmkv2_0.25_112x_fp16",
                                    "name: __guard_missing_landmark");
        case FixtureKind::kMissingRefine:
            return ReplaceInSection(manifest, "refine_net:", "pose_quality:", "name: _04_refine_net",
                                    "name: __guard_missing_refine");
        case FixtureKind::kMissingPose:
            return ReplaceInSection(manifest, "pose_quality:", "feature:", "name: _07_pose_q_fp16", "name: __guard_missing_pose");
        case FixtureKind::kMissingPipeline:
            return ReplaceInSection(manifest, "mask_detect:", "rgb_anti_spoofing:", "name: _05_mask",
                                    "name: __guard_missing_mask");
        case FixtureKind::kMissingLiveness:
            return ReplaceInSection(manifest, "rgb_anti_spoofing:", "face_attribute:", "name: _06_msafa27",
                                    "name: __guard_missing_liveness");
        case FixtureKind::kMissingAttribute:
            return ReplaceInSection(manifest, "face_attribute:", "blink_predict:", "name: _08_fairface_fp16",
                                    "name: __guard_missing_attribute");
        case FixtureKind::kMissingBlink:
            return ReplaceInSection(manifest, "blink_predict:", "face_emotion:", "name: _09_blink_crop",
                                    "name: __guard_missing_blink");
        case FixtureKind::kMissingEmotion:
            return ReplaceInSection(manifest, "face_emotion:", "", "name: _10_emotion_fp16", "name: __guard_missing_emotion");
        case FixtureKind::kMissingRecognition:
            return ReplaceInSection(manifest, "feature:", "mask_detect:", "name: _03_extract",
                                    "name: __guard_missing_feature");
        case FixtureKind::kCorruptTracker:
            extra_entry_name = "__guard_corrupt_detector";
            return ReplaceInSection(manifest, "face_detect_320:", "face_detect_640:",
                                    "name: _00_scrfd_500m_bnkps_shape320x320_fp16", "name: " + extra_entry_name);
        case FixtureKind::kMalformedTracker:
            return ReplaceInSection(manifest, "face_detect_320:", "face_detect_640:", "input_size:\n    - 320\n    - 320", "input_size: []");
        case FixtureKind::kUnsupportedBackend:
            return ReplaceInSection(manifest, "face_detect_320:", "face_detect_640:", "model_type: MNN", "model_type: RKNN");
    }
    return false;
}

bool WriteArchiveEntry(mtar_t& archive, const std::string& name, const std::vector<char>& content) {
    if (mtar_write_file_header(&archive, name.c_str(), static_cast<unsigned>(content.size())) != MTAR_ESUCCESS) {
        return false;
    }
    return content.empty() || mtar_write_data(&archive, content.data(), static_cast<unsigned>(content.size())) == MTAR_ESUCCESS;
}

std::string MakeTemporaryPath() {
    const char* temporary_root = std::getenv("TMPDIR");
    if (temporary_root == nullptr || temporary_root[0] == '\0') {
#if defined(_WIN32)
        temporary_root = ".";
#else
        temporary_root = "/tmp";
#endif
    }
    const auto timestamp = Clock::now().time_since_epoch().count();
    return JoinPath(temporary_root, "inspireface_session_initialization_guard_" + std::to_string(timestamp) + ".tar");
}

bool CreateFixtureArchive(const std::string& source_path, FixtureKind kind, TemporaryFile& fixture) {
    mtar_t source = {};
    mtar_t destination = {};
    fixture.path = MakeTemporaryPath();
    if (mtar_open(&source, source_path.c_str(), "r") != MTAR_ESUCCESS) {
        return false;
    }
    if (mtar_open(&destination, fixture.path.c_str(), "w") != MTAR_ESUCCESS) {
        mtar_close(&source);
        return false;
    }

    bool passed = mtar_rewind(&source) == MTAR_ESUCCESS;
    std::string extra_entry_name;
    mtar_header_t header = {};
    while (passed && mtar_read_header(&source, &header) == MTAR_ESUCCESS) {
        std::vector<char> content(header.size);
        if (header.size != 0 && mtar_read_data(&source, content.data(), header.size) != MTAR_ESUCCESS) {
            passed = false;
            break;
        }
        if (std::strcmp(header.name, "__inspire__") == 0) {
            std::string manifest(content.begin(), content.end());
            if (!MutateManifest(manifest, kind, extra_entry_name)) {
                passed = false;
                break;
            }
            content.assign(manifest.begin(), manifest.end());
        }
        passed = WriteArchiveEntry(destination, header.name, content);
        if (passed && mtar_next(&source) != MTAR_ESUCCESS) {
            passed = false;
        }
    }

    if (passed && !extra_entry_name.empty()) {
        const std::vector<char> corrupt_model = {'N', 'O', 'T', '_', 'A', '_', 'M', 'O', 'D', 'E', 'L'};
        passed = WriteArchiveEntry(destination, extra_entry_name, corrupt_model);
    }
    if (passed) {
        passed = mtar_finalize(&destination) == MTAR_ESUCCESS;
    }
    mtar_close(&destination);
    mtar_close(&source);
    return passed;
}

bool NoUnreleasedSessions() {
    HInt32 count = -1;
    return HFDeBugGetUnreleasedSessionsCount(&count) == HSUCCEED && count == 0;
}

HResult CreateSession(HOption option, int detect_pixel_level, HFSession* session) {
    return HFCreateInspireFaceSessionOptional(option, HF_DETECT_MODE_ALWAYS_DETECT, 8, detect_pixel_level, -1, session);
}

bool LoadImage(const std::string& path, ImageHandles& image) {
    return HFCreateImageBitmapFromFilePath(path.c_str(), 3, &image.bitmap) == HSUCCEED &&
           HFCreateImageStreamFromImageBitmap(image.bitmap, HF_CAMERA_ROTATION_0, &image.stream) == HSUCCEED;
}

bool TestDetectionConsistency(const std::string& test_root, uint64_t& digest) {
    const char* image_paths[] = {"data/bulk/kun.jpg", "data/bulk/jntm.jpg", "data/bulk/woman.png"};
    const int expected_faces[] = {1, 1, 1};
    SessionHandle session;
    if (CreateSession(HF_ENABLE_NONE, 320, &session.value) != HSUCCEED ||
        HFSessionSetFilterMinimumFacePixelSize(session.value, 0) != HSUCCEED) {
        return false;
    }

    digest = kFnvOffset;
    for (size_t image_index = 0; image_index < sizeof(image_paths) / sizeof(image_paths[0]); ++image_index) {
        ImageHandles image;
        if (!LoadImage(JoinPath(test_root, image_paths[image_index]), image)) {
            return false;
        }
        HFMultipleFaceData faces = {};
        if (HFExecuteFaceTrack(session.value, image.stream, &faces) != HSUCCEED || faces.detectedNum != expected_faces[image_index] ||
            faces.rects == nullptr || faces.detConfidence == nullptr || faces.tokens == nullptr) {
            return false;
        }
        HashValue(digest, faces.detectedNum);
        for (int face_index = 0; face_index < faces.detectedNum; ++face_index) {
            HashValue(digest, faces.rects[face_index].x);
            HashValue(digest, faces.rects[face_index].y);
            HashValue(digest, faces.rects[face_index].width);
            HashValue(digest, faces.rects[face_index].height);
            HashValue(digest, faces.detConfidence[face_index]);
            if (faces.tokens[face_index].data == nullptr || faces.tokens[face_index].size <= 0) {
                return false;
            }
            HashBytes(digest, faces.tokens[face_index].data, static_cast<size_t>(faces.tokens[face_index].size));
        }
    }
    session.Reset();
    return NoUnreleasedSessions();
}

bool RunValidCase(const std::string& pack_path, const std::string& test_root, int iterations) {
    if (HFLaunchInspireFace(pack_path.c_str()) != HSUCCEED) {
        return false;
    }
    struct OptionCase {
        const char* name;
        HOption option;
    };
    const HOption all_options = HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_LIVENESS | HF_ENABLE_MASK_DETECT |
                                HF_ENABLE_FACE_ATTRIBUTE | HF_ENABLE_QUALITY | HF_ENABLE_INTERACTION | HF_ENABLE_FACE_POSE |
                                HF_ENABLE_FACE_EMOTION;
    const OptionCase option_cases[] = {{"none", HF_ENABLE_NONE},
                                       {"recognition", HF_ENABLE_FACE_RECOGNITION},
                                       {"liveness", HF_ENABLE_LIVENESS},
                                       {"mask", HF_ENABLE_MASK_DETECT},
                                       {"attribute", HF_ENABLE_FACE_ATTRIBUTE},
                                       {"quality", HF_ENABLE_QUALITY},
                                       {"interaction", HF_ENABLE_INTERACTION},
                                       {"pose", HF_ENABLE_FACE_POSE},
                                       {"emotion", HF_ENABLE_FACE_EMOTION},
                                       {"all", all_options}};
    bool passed = true;
    std::vector<double> samples_ms;
    for (const auto& option_case : option_cases) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
            SessionHandle session;
            const auto begin = Clock::now();
            const HResult status = CreateSession(option_case.option, 320, &session.value);
            const auto end = Clock::now();
            samples_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
            if (status != HSUCCEED || session.value == nullptr) {
                std::cerr << "SESSION_MATRIX,name=" << option_case.name << ",status=" << status << ",result=FAIL\n";
                passed = false;
                break;
            }
        }
    }
    uint64_t digest = 0;
    passed = TestDetectionConsistency(test_root, digest) && passed;
    passed = NoUnreleasedSessions() && passed;
    PrintTiming("SESSION_CREATE_TIMING", samples_ms);
    std::cout << "SESSION_MATRIX,cases=" << (sizeof(option_cases) / sizeof(option_cases[0])) << ",iterations=" << iterations
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    std::cout << "SESSION_DETECTION_CONSISTENCY,images=3,digest=0x" << std::hex << digest << std::dec
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    if (HFTerminateInspireFace() != HSUCCEED) {
        passed = false;
    }
    return passed;
}

bool RunFailureCase(const std::string& pack_path, FixtureKind kind, HOption failing_option, HResult expected_status, int iterations,
                    const char* case_name) {
    TemporaryFile fixture;
    HFSetLogLevel(HF_LOG_NONE);
    if (!CreateFixtureArchive(pack_path, kind, fixture) || HFLaunchInspireFace(fixture.path.c_str()) != HSUCCEED) {
        return false;
    }

    bool passed = true;
    std::vector<double> failure_samples_ms;
    std::vector<double> recovery_samples_ms;
    const int failing_pixel_level = 320;
    const bool recovery_supported = kind != FixtureKind::kMissingLandmark && kind != FixtureKind::kMissingRefine;
    const int recovery_pixel_level = kind == FixtureKind::kMissingTracker || kind == FixtureKind::kCorruptTracker ||
                                               kind == FixtureKind::kMalformedTracker || kind == FixtureKind::kUnsupportedBackend
                                       ? 160
                                       : 320;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        HFSession failed_handle = reinterpret_cast<HFSession>(static_cast<uintptr_t>(1));
        const auto failure_begin = Clock::now();
        const HResult failure_status = CreateSession(failing_option, failing_pixel_level, &failed_handle);
        const auto failure_end = Clock::now();
        failure_samples_ms.push_back(std::chrono::duration<double, std::milli>(failure_end - failure_begin).count());
        if (failure_status != expected_status || failed_handle != nullptr) {
            std::cerr << "SESSION_FAILURE_DETAIL,case=" << case_name << ",iteration=" << iteration << ",expected=" << expected_status
                      << ",actual=" << failure_status << ",handle_null=" << (failed_handle == nullptr ? "YES" : "NO") << '\n';
            passed = false;
            if (failure_status == HSUCCEED && failed_handle != nullptr) {
                HFReleaseInspireFaceSession(failed_handle);
            }
        }
        if (!NoUnreleasedSessions()) {
            passed = false;
        }

        if (recovery_supported) {
            SessionHandle recovery;
            const auto recovery_begin = Clock::now();
            const HResult recovery_status = CreateSession(HF_ENABLE_NONE, recovery_pixel_level, &recovery.value);
            const auto recovery_end = Clock::now();
            recovery_samples_ms.push_back(std::chrono::duration<double, std::milli>(recovery_end - recovery_begin).count());
            if (recovery_status != HSUCCEED || recovery.value == nullptr) {
                passed = false;
            }
            recovery.Reset();
            if (!NoUnreleasedSessions()) {
                passed = false;
            }
        }
    }

    PrintTiming("SESSION_FAILURE_TIMING", failure_samples_ms);
    if (recovery_supported) {
        PrintTiming("SESSION_RECOVERY_TIMING", recovery_samples_ms);
    } else {
        std::cout << "SESSION_RECOVERY_TIMING,samples=0,status=NOT_APPLICABLE\n";
    }
    std::cout << "SESSION_FAILURE_GUARD,case=" << case_name << ",iterations=" << iterations << ",expected=" << expected_status
              << ",resource_leak=" << (NoUnreleasedSessions() ? "NO" : "YES") << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    if (HFTerminateInspireFace() != HSUCCEED) {
        passed = false;
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <pack_path> [test_res_root] [iterations] [mode]\n"
                  << "Modes: valid, missing-tracker, missing-landmark, missing-refine, missing-pose, missing-pipeline, missing-liveness, "
                     "missing-attribute, missing-blink, missing-emotion, missing-recognition, corrupt-tracker, malformed-tracker, "
                     "unsupported-backend\n";
        return 2;
    }
    const std::string pack_path = argv[1];
    const std::string test_root = argc >= 3 ? argv[2] : "test_res";
    const int iterations = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 20;
    const std::string mode = argc >= 5 ? argv[4] : "valid";

    bool passed = false;
    if (mode == "valid") {
        passed = RunValidCase(pack_path, test_root, iterations);
    } else if (mode == "missing-tracker") {
        passed = RunFailureCase(pack_path, FixtureKind::kMissingTracker, HF_ENABLE_NONE, HERR_ARCHIVE_LOAD_MODEL_FAILURE, iterations,
                                "missing-tracker");
    } else if (mode == "missing-landmark") {
        passed = RunFailureCase(pack_path, FixtureKind::kMissingLandmark, HF_ENABLE_NONE, HERR_ARCHIVE_LOAD_MODEL_FAILURE, iterations,
                                "missing-landmark");
    } else if (mode == "missing-refine") {
        passed = RunFailureCase(pack_path, FixtureKind::kMissingRefine, HF_ENABLE_NONE, HERR_ARCHIVE_LOAD_MODEL_FAILURE, iterations,
                                "missing-refine");
    } else if (mode == "missing-pose") {
        passed = RunFailureCase(pack_path, FixtureKind::kMissingPose, HF_ENABLE_QUALITY, HERR_ARCHIVE_LOAD_MODEL_FAILURE, iterations,
                                "missing-pose");
    } else if (mode == "missing-pipeline") {
        passed = RunFailureCase(pack_path, FixtureKind::kMissingPipeline, HF_ENABLE_MASK_DETECT, HERR_ARCHIVE_LOAD_MODEL_FAILURE, iterations,
                                "missing-pipeline");
    } else if (mode == "missing-liveness") {
        passed = RunFailureCase(pack_path, FixtureKind::kMissingLiveness, HF_ENABLE_LIVENESS, HERR_ARCHIVE_LOAD_MODEL_FAILURE, iterations,
                                "missing-liveness");
    } else if (mode == "missing-attribute") {
        passed = RunFailureCase(pack_path, FixtureKind::kMissingAttribute, HF_ENABLE_FACE_ATTRIBUTE, HERR_ARCHIVE_LOAD_MODEL_FAILURE,
                                iterations, "missing-attribute");
    } else if (mode == "missing-blink") {
        passed = RunFailureCase(pack_path, FixtureKind::kMissingBlink, HF_ENABLE_INTERACTION, HERR_ARCHIVE_LOAD_MODEL_FAILURE, iterations,
                                "missing-blink");
    } else if (mode == "missing-emotion") {
        passed = RunFailureCase(pack_path, FixtureKind::kMissingEmotion, HF_ENABLE_FACE_EMOTION, HERR_ARCHIVE_LOAD_MODEL_FAILURE, iterations,
                                "missing-emotion");
    } else if (mode == "missing-recognition") {
        passed = RunFailureCase(pack_path, FixtureKind::kMissingRecognition, HF_ENABLE_FACE_RECOGNITION, HERR_ARCHIVE_LOAD_MODEL_FAILURE,
                                iterations, "missing-recognition");
    } else if (mode == "corrupt-tracker") {
        passed = RunFailureCase(pack_path, FixtureKind::kCorruptTracker, HF_ENABLE_NONE, HERR_ARCHIVE_LOAD_FAILURE, iterations,
                                "corrupt-tracker");
    } else if (mode == "malformed-tracker") {
        passed = RunFailureCase(pack_path, FixtureKind::kMalformedTracker, HF_ENABLE_NONE, HERR_ARCHIVE_LOAD_FAILURE, iterations,
                                "malformed-tracker");
    } else if (mode == "unsupported-backend") {
        passed = RunFailureCase(pack_path, FixtureKind::kUnsupportedBackend, HF_ENABLE_NONE, HERR_ARCHIVE_LOAD_FAILURE, iterations,
                                "unsupported-backend");
    } else {
        std::cerr << "Unknown mode: " << mode << '\n';
        return 2;
    }
    std::cout << "SUMMARY,mode=" << mode << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
