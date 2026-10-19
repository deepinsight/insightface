#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <inspireface.h>
#include <inspireface/frame_process.h>

#include "common/face_data/face_serialize_tools.h"
#include "track_module/tracker_optional/bytetrack/BYTETracker.h"

namespace {

using Clock = std::chrono::steady_clock;

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

bool EqualMatrix(const inspirecv::TransformMatrix& lhs, const inspirecv::TransformMatrix& rhs) {
    for (int i = 0; i < 6; ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

bool TestTransformState() {
    constexpr int kWidth = 37;
    constexpr int kHeight = 23;
    std::vector<uint8_t> pixels(static_cast<size_t>(kWidth * kHeight * 3));
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<uint8_t>((i * 37 + 19) % 251);
    }

    bool passed = true;
    uint64_t digest = 1469598103934665603ULL;
    const inspirecv::ROTATION_MODE rotations[] = {inspirecv::ROTATION_0, inspirecv::ROTATION_90, inspirecv::ROTATION_180,
                                                  inspirecv::ROTATION_270};
    for (const auto rotation : rotations) {
        auto process = inspirecv::FrameProcess::Create(pixels.data(), kHeight, kWidth, inspirecv::BGR, rotation);
        process.SetPreviewSize(320);
        const auto before = process.GetAffineMatrix();
        const auto scaled = process.ExecuteImageScaleProcessing(1.0f, true);
        const auto after = process.GetAffineMatrix();
        passed = passed && !scaled.Empty() && EqualMatrix(before, after);
        for (int i = 0; i < 6; ++i) {
            HashValue(digest, before[i]);
            HashValue(digest, after[i]);
        }
        HashBytes(digest, scaled.Data(), static_cast<size_t>(scaled.Width() * scaled.Height() * scaled.Channels()));
    }

    std::cout << "TRANSFORM_STATE,rotations=4,digest=0x" << std::hex << digest << std::dec
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

inspire::FaceObjectInternal MakeFaceObject() {
    inspire::FaceObjectInternal face(7, inspirecv::Rect2i(10, 20, 80, 80), 116);
    std::vector<inspirecv::Point2f> landmarks(116);
    for (size_t i = 0; i < landmarks.size(); ++i) {
        landmarks[i] = inspirecv::Point2f(static_cast<float>(i) * 0.75f + 10.0f, static_cast<float>(i) * 0.5f + 20.0f);
    }
    face.SetLandmark(landmarks, true, false, 0.05f, 5, 212);
    face.high_result.lmk = {landmarks[55], landmarks[105], landmarks[69], landmarks[45], landmarks[50]};
    face.high_result.lmk_quality = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    face.high_result.pitch = 1.25f;
    face.high_result.yaw = -2.5f;
    face.high_result.roll = 3.75f;
    face.setTransMatrix(inspirecv::TransformMatrix::Create(1.0f, 0.0f, 4.0f, 0.0f, 1.0f, 8.0f));
    return face;
}

bool TestTokenDeterminism() {
    const auto face = MakeFaceObject();
    std::vector<char> reference;
    bool passed = true;
    bool padding_zero = true;
    uint64_t digest = 1469598103934665603ULL;
    const size_t represented_end = offsetof(inspire::FaceTrackWrap, densityLandmarkEnable) + sizeof(int);

    for (int iteration = 0; iteration < 512; ++iteration) {
        volatile uint8_t stack_noise[257];
        for (size_t i = 0; i < sizeof(stack_noise); ++i) {
            stack_noise[i] = static_cast<uint8_t>((iteration * 31 + static_cast<int>(i)) & 0xff);
        }
        const auto wrapped = inspire::FaceObjectInternalToHyperFaceData(face, 0);
        inspire::ByteArray bytes;
        if (inspire::RunSerializeHyperFaceData(wrapped, bytes) != HSUCCEED) {
            return false;
        }
        if (iteration == 0) {
            reference.assign(bytes.begin(), bytes.end());
        } else if (bytes.size() != reference.size() || std::memcmp(bytes.data(), reference.data(), bytes.size()) != 0) {
            passed = false;
        }
        for (size_t i = represented_end; i < bytes.size(); ++i) {
            padding_zero = padding_zero && bytes[i] == 0;
        }
        HashBytes(digest, bytes.data(), bytes.size());
    }

    passed = passed && padding_zero;

    inspire::ByteArray overwritten(17, 'x');
    passed = passed && inspire::RunSerializeHyperFaceData(
                         inspire::FaceObjectInternalToHyperFaceData(face, 0), overwritten) == HSUCCEED &&
             overwritten.size() == sizeof(inspire::FaceTrackWrap) &&
             std::memcmp(overwritten.data(), reference.data(), reference.size()) == 0;
    inspire::FaceTrackWrap unchanged = {};
    unchanged.trackId = 91;
    passed = passed && inspire::RunDeserializeHyperFaceData(nullptr, sizeof(unchanged), unchanged) == HERR_SESS_FACE_DATA_ERROR &&
             unchanged.trackId == 91;

    auto oversized = MakeFaceObject();
    oversized.high_result.lmk.assign(9, inspirecv::Point2f(777.0f, 888.0f));
    oversized.high_result.lmk_quality = {0.25f};
    oversized.landmark_smooth_aux_.assign(1, std::vector<inspirecv::Point2f>(3));
    const auto bounded = inspire::FaceObjectInternalToHyperFaceData(oversized, 0);
    passed = passed && bounded.keyPoints[4].x == 777.0f && bounded.quality[0] == 0.25f && bounded.quality[1] == -1.0f &&
             bounded.densityLandmarkEnable == 0;

    std::cout << "TOKEN_DETERMINISM,size=" << reference.size() << ",tail_padding=" << (reference.size() - represented_end)
              << ",padding_zero=" << (padding_zero ? "PASS" : "FAIL") << ",digest=0x" << std::hex << digest << std::dec
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestTensorMaterializationTiming(int benchmark_iterations) {
    const int strides[] = {8, 16, 32};
    std::vector<std::string> names;
    std::vector<std::vector<float>> tensors;
    names.reserve(9);
    tensors.reserve(9);
    for (int group = 0; group < 3; ++group) {
        const int values_per_anchor = group == 0 ? 1 : (group == 1 ? 4 : 10);
        for (const int stride : strides) {
            const size_t anchors = static_cast<size_t>(640 / stride) * static_cast<size_t>(640 / stride) * 2;
            names.push_back("output_" + std::to_string(names.size()));
            tensors.emplace_back(anchors * static_cast<size_t>(values_per_anchor));
            auto& tensor = tensors.back();
            for (size_t i = 0; i < tensor.size(); ++i) {
                tensor[i] = static_cast<float>((i * 17 + tensor.size()) % 1009) / 1009.0f;
            }
        }
    }

    inspire::AnyTensorOutputs copied;
    inspire::AnyTensorViews views;
    copied.reserve(tensors.size());
    views.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        copied.emplace_back(names[i], tensors[i]);
        views.emplace_back(&names[i], tensors[i].data(), tensors[i].size());
    }
    bool exact = copied.size() == views.size();
    size_t total_floats = 0;
    for (size_t i = 0; exact && i < copied.size(); ++i) {
        total_floats += views[i].size;
        exact = copied[i].first == *views[i].name && copied[i].second.size() == views[i].size &&
                std::memcmp(copied[i].second.data(), views[i].data, views[i].size * sizeof(float)) == 0;
    }

    const int iterations = std::max(100, benchmark_iterations);
    volatile float sink = 0.0f;
    const auto copy_begin = Clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        inspire::AnyTensorOutputs outputs;
        outputs.reserve(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            outputs.emplace_back(names[i], tensors[i]);
            sink += outputs.back().second[static_cast<size_t>(iteration) % outputs.back().second.size()];
        }
    }
    const auto copy_end = Clock::now();
    const auto view_begin = Clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        inspire::AnyTensorViews outputs;
        outputs.reserve(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            outputs.emplace_back(&names[i], tensors[i].data(), tensors[i].size());
            sink += outputs.back().data[static_cast<size_t>(iteration) % outputs.back().size];
        }
    }
    const auto view_end = Clock::now();
    const double copy_us = std::chrono::duration<double, std::micro>(copy_end - copy_begin).count() / iterations;
    const double view_us = std::chrono::duration<double, std::micro>(view_end - view_begin).count() / iterations;
    const bool passed = exact && std::isfinite(sink) && view_us < copy_us;
    std::cout << std::fixed << std::setprecision(3) << "TENSOR_MATERIALIZATION,tensors=9,floats=" << total_floats
              << ",copy_mean_us=" << copy_us << ",view_mean_us=" << view_us << ",speedup=" << (copy_us / view_us)
              << ",exact=" << (exact ? "PASS" : "FAIL") << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

std::vector<Object> MakeTrackObjects(int frame, int count) {
    std::vector<Object> objects;
    objects.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        Object object;
        object.rect = inspirecv::Rect<int>(20 + i * 31 + frame % 3, 15 + i * 17 + (frame + i) % 2, 24 + i % 4, 28 + i % 5);
        object.label = 0;
        object.prob = 0.91f - static_cast<float>(i % 5) * 0.025f;
        objects.push_back(object);
    }
    return objects;
}

std::vector<Object> MakeLifecycleTrackObjects(int frame) {
    if (frame >= 14 && frame <= 22) {
        return {};
    }
    const int count = frame < 11 ? 8 : (frame < 14 ? 3 : 6);
    std::vector<Object> objects;
    objects.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        Object object;
        const int phase = frame > 22 ? frame - 22 : frame;
        object.rect = inspirecv::Rect<int>(15 + i * 27 + phase * (i % 2 == 0 ? 2 : -1), 20 + i * 13 + phase % 4, 22 + i % 3,
                                           26 + i % 4);
        object.label = 0;
        object.prob = (frame >= 6 && frame <= 10 && i % 3 == 0) ? 0.35f : 0.93f - static_cast<float>(i) * 0.02f;
        objects.push_back(object);
    }
    return objects;
}

uint64_t HashTracks(const std::vector<STrack>& tracks) {
    uint64_t digest = 1469598103934665603ULL;
    HashValue(digest, tracks.size());
    for (const auto& track : tracks) {
        HashValue(digest, track.track_id);
        HashValue(digest, track.state);
        HashValue(digest, track.score);
        for (const float value : track.tlwh) {
            HashValue(digest, value);
        }
    }
    return digest;
}

bool TestTrackerIsolationAndTiming(int benchmark_iterations) {
    BYTETracker first(30, 30);
    BYTETracker second(30, 30);
    const auto initial_objects = MakeTrackObjects(0, 1);
    const auto first_tracks = first.update(initial_objects);
    const auto second_tracks = second.update(initial_objects);
    bool isolated = first_tracks.size() == 1 && second_tracks.size() == 1 && first_tracks[0].track_id == second_tracks[0].track_id;

    BYTETracker reference_tracker(30, 30);
    BYTETracker repeated_tracker(30, 30);
    uint64_t reference_digest = 1469598103934665603ULL;
    uint64_t repeated_digest = 1469598103934665603ULL;
    std::vector<double> samples_us;
    samples_us.reserve(static_cast<size_t>(benchmark_iterations));
    for (int frame = 0; frame < benchmark_iterations; ++frame) {
        const auto objects = MakeTrackObjects(frame, 12);
        const auto begin = Clock::now();
        const auto reference_output = reference_tracker.update(objects);
        const auto end = Clock::now();
        const auto repeated_output = repeated_tracker.update(objects);
        const uint64_t frame_reference_digest = HashTracks(reference_output);
        const uint64_t frame_repeated_digest = HashTracks(repeated_output);
        HashValue(reference_digest, frame_reference_digest);
        HashValue(repeated_digest, frame_repeated_digest);
        samples_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }

    BYTETracker lifecycle_reference(30, 5);
    BYTETracker lifecycle_repeated(30, 5);
    uint64_t lifecycle_reference_digest = 1469598103934665603ULL;
    uint64_t lifecycle_repeated_digest = 1469598103934665603ULL;
    bool lifecycle_valid = true;
    for (int frame = 0; frame < 40; ++frame) {
        const auto objects = MakeLifecycleTrackObjects(frame);
        const auto reference_output = lifecycle_reference.update(objects);
        const auto repeated_output = lifecycle_repeated.update(objects);
        const uint64_t reference_frame_digest = HashTracks(reference_output);
        const uint64_t repeated_frame_digest = HashTracks(repeated_output);
        HashValue(lifecycle_reference_digest, reference_frame_digest);
        HashValue(lifecycle_repeated_digest, repeated_frame_digest);
        std::vector<int> ids;
        for (const auto &track : reference_output) {
            lifecycle_valid = lifecycle_valid && track.track_id > 0 && std::isfinite(track.score) && track.tlwh[2] > 0.0f && track.tlwh[3] > 0.0f;
            ids.push_back(track.track_id);
        }
        std::sort(ids.begin(), ids.end());
        lifecycle_valid = lifecycle_valid && std::adjacent_find(ids.begin(), ids.end()) == ids.end();
    }

    const bool repeatable = reference_digest == repeated_digest;
    const bool lifecycle_repeatable = lifecycle_valid && lifecycle_reference_digest == lifecycle_repeated_digest;
    std::sort(samples_us.begin(), samples_us.end());
    const double mean = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) / static_cast<double>(samples_us.size());
    std::cout << "BYTETRACK_CORRECTNESS,isolation=" << (isolated ? "PASS" : "FAIL")
              << ",repeatability=" << (repeatable ? "PASS" : "FAIL") << ",lifecycle=" << (lifecycle_repeatable ? "PASS" : "FAIL")
              << ",digest=0x" << std::hex << reference_digest << ",lifecycle_digest=0x" << lifecycle_reference_digest << std::dec
              << ",status=" << (isolated && repeatable && lifecycle_repeatable ? "PASS" : "FAIL") << '\n';
    std::cout << std::fixed << std::setprecision(3) << "BYTETRACK_TIMING,iterations=" << benchmark_iterations << ",objects=12,mean_us=" << mean
              << ",p50_us=" << Percentile(samples_us, 0.50) << ",p95_us=" << Percentile(samples_us, 0.95) << '\n';
    return isolated && repeatable && lifecycle_repeatable;
}

bool LoadImage(const std::string& path, ImageHandles& image) {
    if (HFCreateImageBitmapFromFilePath(path.c_str(), 3, &image.bitmap) != HSUCCEED) {
        return false;
    }
    return HFCreateImageStreamFromImageBitmap(image.bitmap, HF_CAMERA_ROTATION_0, &image.stream) == HSUCCEED;
}

bool CreatePipelineSession(SessionHandle& session, int maximum_faces = 5, int preview_size = 320) {
    HFSessionCustomParameter parameter = {0};
    parameter.enable_liveness = 1;
    parameter.enable_mask_detect = 1;
    parameter.enable_face_attribute = 1;
    parameter.enable_interaction_liveness = 1;
    parameter.enable_face_emotion = 1;
    parameter.enable_detect_mode_landmark = 1;
    if (HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_LIGHT_TRACK, maximum_faces, preview_size, -1, &session.value) != HSUCCEED) {
        return false;
    }
    return HFSessionSetTrackPreviewSize(session.value, preview_size) == HSUCCEED &&
           HFSessionSetFilterMinimumFacePixelSize(session.value, 0) == HSUCCEED &&
           HFSessionSetTrackModeDetectInterval(session.value, 5) == HSUCCEED;
}

bool ValidatePipelineResults(HFSession session, int expected_faces, bool expect_sentinel, uint64_t& digest) {
    HFRGBLivenessConfidence liveness = {0};
    HFFaceMaskConfidence mask = {0};
    HFFaceAttributeResult attribute = {0};
    HFFaceInteractionState interaction = {0};
    HFFaceInteractionsActions actions = {0};
    HFFaceEmotionResult emotion = {0};
    if (HFGetRGBLivenessConfidence(session, &liveness) != HSUCCEED || HFGetFaceMaskConfidence(session, &mask) != HSUCCEED ||
        HFGetFaceAttributeResult(session, &attribute) != HSUCCEED || HFGetFaceInteractionStateResult(session, &interaction) != HSUCCEED ||
        HFGetFaceInteractionActionsResult(session, &actions) != HSUCCEED || HFGetFaceEmotionResult(session, &emotion) != HSUCCEED) {
        return false;
    }
    if (liveness.num != expected_faces || mask.num != expected_faces || attribute.num != expected_faces || interaction.num != expected_faces ||
        actions.num != expected_faces || emotion.num != expected_faces) {
        return false;
    }
    for (int i = 0; i < expected_faces; ++i) {
        if (liveness.confidence == nullptr || mask.confidence == nullptr || attribute.race == nullptr || attribute.gender == nullptr ||
            attribute.ageBracket == nullptr || interaction.leftEyeStatusConfidence == nullptr || interaction.rightEyeStatusConfidence == nullptr ||
            actions.normal == nullptr || actions.shake == nullptr || actions.jawOpen == nullptr || actions.headRaise == nullptr || actions.blink == nullptr ||
            emotion.emotion == nullptr) {
            return false;
        }
        if (expect_sentinel) {
            if (liveness.confidence[i] != -1.0f || mask.confidence[i] != -1.0f || attribute.race[i] != -1 || attribute.gender[i] != -1 ||
                attribute.ageBracket[i] != -1 || interaction.leftEyeStatusConfidence[i] != -1.0f ||
                interaction.rightEyeStatusConfidence[i] != -1.0f || actions.normal[i] != -1 || actions.shake[i] != -1 ||
                actions.jawOpen[i] != -1 || actions.headRaise[i] != -1 || actions.blink[i] != -1 || emotion.emotion[i] != -1) {
                return false;
            }
        } else {
            if (!std::isfinite(liveness.confidence[i]) || !std::isfinite(mask.confidence[i]) ||
                !std::isfinite(interaction.leftEyeStatusConfidence[i]) || !std::isfinite(interaction.rightEyeStatusConfidence[i]) ||
                attribute.race[i] < 0 || attribute.race[i] > 4 || attribute.gender[i] < 0 || attribute.gender[i] > 1 ||
                attribute.ageBracket[i] < 0 || emotion.emotion[i] < 0 || emotion.emotion[i] > 6) {
                return false;
            }
        }
        HashValue(digest, liveness.confidence[i]);
        HashValue(digest, mask.confidence[i]);
        HashValue(digest, attribute.race[i]);
        HashValue(digest, attribute.gender[i]);
        HashValue(digest, attribute.ageBracket[i]);
        HashValue(digest, interaction.leftEyeStatusConfidence[i]);
        HashValue(digest, interaction.rightEyeStatusConfidence[i]);
        HashValue(digest, actions.normal[i]);
        HashValue(digest, actions.shake[i]);
        HashValue(digest, actions.jawOpen[i]);
        HashValue(digest, actions.headRaise[i]);
        HashValue(digest, actions.blink[i]);
        HashValue(digest, emotion.emotion[i]);
    }
    return true;
}

bool TestPipelineCachesAndTiming(const std::string& test_root, int benchmark_iterations, bool include_multi_face) {
    struct PipelineCase {
        const char* relative_path;
        int expected_faces;
        int preview_size;
    };
    const PipelineCase image_cases[] = {{"data/bulk/kun.jpg", 1, 320},
                                        {"data/bulk/jntm.jpg", 1, 320},
                                        {"data/bulk/woman.png", 1, 320},
                                        {"data/bulk/pedestrian.png", 5, 640}};
    constexpr HOption kAllPipelineOptions = HF_ENABLE_LIVENESS | HF_ENABLE_MASK_DETECT | HF_ENABLE_FACE_ATTRIBUTE | HF_ENABLE_INTERACTION |
                                             HF_ENABLE_FACE_EMOTION;
    bool passed = true;
    uint64_t digest = 1469598103934665603ULL;
    std::vector<double> samples_ms;

    constexpr size_t kPipelineCaseCount = sizeof(image_cases) / sizeof(image_cases[0]);
    const size_t case_count = include_multi_face ? kPipelineCaseCount : kPipelineCaseCount - 1;
    for (size_t case_index = 0; case_index < case_count; ++case_index) {
        const auto& image_case = image_cases[case_index];
        ImageHandles image;
        SessionHandle session;
        if (!LoadImage(JoinPath(test_root, image_case.relative_path), image) ||
            !CreatePipelineSession(session, image_case.expected_faces, image_case.preview_size)) {
            passed = false;
            continue;
        }
        HFMultipleFaceData faces = {0};
        if (HFExecuteFaceTrack(session.value, image.stream, &faces) != HSUCCEED || faces.detectedNum != image_case.expected_faces) {
            passed = false;
            continue;
        }

        for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
            const auto begin = Clock::now();
            const HResult status = HFMultipleFacePipelineProcessOptional(session.value, image.stream, &faces, kAllPipelineOptions);
            const auto end = Clock::now();
            if (status != HSUCCEED || !ValidatePipelineResults(session.value, faces.detectedNum, false, digest)) {
                passed = false;
                break;
            }
            samples_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        }
        if (HFMultipleFacePipelineProcessOptional(session.value, image.stream, &faces, HF_ENABLE_NONE) != HSUCCEED ||
            !ValidatePipelineResults(session.value, faces.detectedNum, true, digest)) {
            passed = false;
        }
        HFMultipleFaceData next_faces = {0};
        if (HFExecuteFaceTrack(session.value, image.stream, &next_faces) != HSUCCEED ||
            !ValidatePipelineResults(session.value, 0, true, digest)) {
            passed = false;
        }
    }

    std::sort(samples_ms.begin(), samples_ms.end());
    const double mean = samples_ms.empty() ? 0.0 : std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / samples_ms.size();
    std::cout << "PIPELINE_CACHE,images=" << case_count << ",multi_face=" << (include_multi_face ? "ON" : "OFF") << ",digest=0x" << std::hex
              << digest << std::dec
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    if (!samples_ms.empty()) {
        std::cout << std::fixed << std::setprecision(3) << "PIPELINE_TIMING,iterations=" << samples_ms.size() << ",mean_ms=" << mean
                  << ",p50_ms=" << Percentile(samples_ms, 0.50) << ",p95_ms=" << Percentile(samples_ms, 0.95) << '\n';
    }
    return passed;
}

uint64_t HashConcurrentFaces(const HFMultipleFaceData& faces, bool& valid) {
    uint64_t digest = 1469598103934665603ULL;
    valid = faces.detectedNum == 1 && faces.rects != nullptr && faces.trackIds != nullptr && faces.trackCounts != nullptr &&
            faces.detConfidence != nullptr && faces.tokens != nullptr;
    HashValue(digest, faces.detectedNum);
    for (int i = 0; valid && i < faces.detectedNum; ++i) {
        valid = faces.rects[i].width > 0 && faces.rects[i].height > 0 && faces.trackIds[i] > 0 && std::isfinite(faces.detConfidence[i]) &&
                faces.tokens[i].data != nullptr && faces.tokens[i].size == static_cast<int>(sizeof(inspire::FaceTrackWrap));
        HashValue(digest, faces.rects[i].x);
        HashValue(digest, faces.rects[i].y);
        HashValue(digest, faces.rects[i].width);
        HashValue(digest, faces.rects[i].height);
        HashValue(digest, faces.trackIds[i]);
        HashValue(digest, faces.trackCounts[i]);
        HashValue(digest, faces.detConfidence[i]);
        if (valid) {
            HashBytes(digest, faces.tokens[i].data, static_cast<size_t>(faces.tokens[i].size));
        }
    }
    return digest;
}

bool CreateConcurrentSession(SessionHandle& session) {
    HFSessionCustomParameter parameter = {0};
    parameter.enable_detect_mode_landmark = 1;
    return HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_TRACK_BY_DETECTION, 1, 320, 30, &session.value) == HSUCCEED &&
           HFSessionSetTrackPreviewSize(session.value, 320) == HSUCCEED &&
           HFSessionSetFilterMinimumFacePixelSize(session.value, 0) == HSUCCEED;
}

bool TestConcurrentSessions(const std::string& test_root) {
    std::array<ImageHandles, 2> images;
    std::array<SessionHandle, 2> sessions;
    for (size_t i = 0; i < sessions.size(); ++i) {
        if (!LoadImage(JoinPath(test_root, "data/bulk/kun.jpg"), images[i]) || !CreateConcurrentSession(sessions[i])) {
            return false;
        }
    }

    struct Result {
        bool valid = true;
        uint64_t digest = 1469598103934665603ULL;
    };
    std::array<Result, 2> results;
    std::array<std::thread, 2> workers;
    for (size_t index = 0; index < workers.size(); ++index) {
        workers[index] = std::thread([&, index]() {
            for (int frame = 0; frame < 8; ++frame) {
                HFMultipleFaceData faces = {0};
                if (HFExecuteFaceTrack(sessions[index].value, images[index].stream, &faces) != HSUCCEED) {
                    results[index].valid = false;
                    return;
                }
                bool frame_valid = false;
                const uint64_t frame_digest = HashConcurrentFaces(faces, frame_valid);
                results[index].valid = results[index].valid && frame_valid;
                HashValue(results[index].digest, frame_digest);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    const bool passed = results[0].valid && results[1].valid && results[0].digest == results[1].digest;
    std::cout << "CONCURRENT_SESSIONS,sessions=2,frames=8,digest=0x" << std::hex << results[0].digest << std::dec
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <pack_path> [test_res_root] [pipeline_iterations] [tracker_iterations] [include_multi_face]\n";
        return 2;
    }
    const std::string pack_path = argv[1];
    const std::string test_root = argc >= 3 ? argv[2] : "test_res";
    const int pipeline_iterations = argc >= 4 ? std::max(1, std::stoi(argv[3])) : 10;
    const int tracker_iterations = argc >= 5 ? std::max(1, std::stoi(argv[4])) : 200;
    const bool include_multi_face = argc < 6 || std::stoi(argv[5]) != 0;

    if (HFLaunchInspireFace(pack_path.c_str()) != HSUCCEED) {
        return 3;
    }
    bool passed = true;
    passed = TestTransformState() && passed;
    passed = TestTokenDeterminism() && passed;
    passed = TestTensorMaterializationTiming(tracker_iterations) && passed;
    passed = TestTrackerIsolationAndTiming(tracker_iterations) && passed;
    passed = TestPipelineCachesAndTiming(test_root, pipeline_iterations, include_multi_face) && passed;
    passed = TestConcurrentSessions(test_root) && passed;
    if (HFTerminateInspireFace() != HSUCCEED) {
        passed = false;
    }
    std::cout << "SUMMARY,status=" << (passed ? "PASS" : "FAIL") << ",pipeline_iterations=" << pipeline_iterations
              << ",tracker_iterations=" << tracker_iterations << '\n';
    return passed ? 0 : 1;
}
