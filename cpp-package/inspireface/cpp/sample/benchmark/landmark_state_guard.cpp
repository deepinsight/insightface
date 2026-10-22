#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "../../inspireface/common/face_info/face_object_internal.h"

namespace {

void Require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

YAML::Node MakeLandmarkTable() {
    return YAML::Load(R"yaml(
valid_a:
  num_of_landmark: 5
  expansion_scale: 1.2
  input_size: 112
  normalization_mode: MinMax
  semantic_index:
    left_eye_center: 0
    right_eye_center: 1
    nose_corner: 2
    mouth_left_corner: 3
    mouth_right_corner: 4
    mouth_lower: 3
    mouth_upper: 4
    left_eye_region: [0, 1]
    right_eye_region: [1, 2]
  mesh_shape: [0, 0, 2, 0, 3, 1, 2, 3, 0, 2]
valid_b:
  num_of_landmark: 5
  expansion_scale: 1.4
  input_size: 192
  normalization_mode: CenterScaling
  semantic_index:
    left_eye_center: 1
    right_eye_center: 2
    nose_corner: 3
    mouth_left_corner: 0
    mouth_right_corner: 4
    mouth_lower: 0
    mouth_upper: 4
    left_eye_region: [0, 1, 2]
    right_eye_region: [2, 3, 4]
  mesh_shape: [0, 0, 4, 0, 5, 2, 3, 5, 0, 3]
bad_missing_semantic:
  num_of_landmark: 5
  expansion_scale: 1.3
  input_size: 112
  normalization_mode: MinMax
  mesh_shape: [0, 0, 2, 0, 3, 1, 2, 3, 0, 2]
bad_mesh:
  num_of_landmark: 5
  expansion_scale: 1.3
  input_size: 112
  normalization_mode: MinMax
  semantic_index:
    left_eye_center: 0
    right_eye_center: 1
    nose_corner: 2
    mouth_left_corner: 3
    mouth_right_corner: 4
    mouth_lower: 3
    mouth_upper: 4
    left_eye_region: [0]
    right_eye_region: [1]
  mesh_shape: [0, 0]
bad_index:
  num_of_landmark: 5
  expansion_scale: 1.3
  input_size: 112
  normalization_mode: MinMax
  semantic_index:
    left_eye_center: 99
    right_eye_center: 1
    nose_corner: 2
    mouth_left_corner: 3
    mouth_right_corner: 4
    mouth_lower: 3
    mouth_upper: 4
    left_eye_region: [0]
    right_eye_region: [1]
  mesh_shape: [0, 0, 2, 0, 3, 1, 2, 3, 0, 2]
bad_type:
  num_of_landmark: invalid
  expansion_scale: 1.3
  input_size: 112
  normalization_mode: MinMax
)yaml");
}

struct ParamSnapshot {
    int count;
    float scale;
    int input;
    int leftEye;
    std::string engine;
    std::string normalization;
    std::vector<inspirecv::Point2f> shape;
};

ParamSnapshot Snapshot(const inspire::LandmarkParam &parameter) {
    return {parameter.num_of_landmark, parameter.expansion_scale, parameter.input_size, parameter.semantic_index.left_eye_center,
            parameter.landmark_engine_name, parameter.normalization_mode, parameter.mean_shape_points};
}

bool Equal(const ParamSnapshot &left, const ParamSnapshot &right) {
    if (left.count != right.count || left.scale != right.scale || left.input != right.input || left.leftEye != right.leftEye ||
        left.engine != right.engine || left.normalization != right.normalization || left.shape.size() != right.shape.size()) {
        return false;
    }
    for (size_t index = 0; index < left.shape.size(); ++index) {
        if (left.shape[index].GetX() != right.shape[index].GetX() || left.shape[index].GetY() != right.shape[index].GetY()) {
            return false;
        }
    }
    return true;
}

void TestTransactionalLandmarkReload() {
    inspire::LandmarkParam legacy_default{YAML::Node()};
    Require(legacy_default.ReLoad("landmark"), "legacy pack without landmark table must use defaults");
    Require(legacy_default.landmark_engine_name == "landmark" && legacy_default.num_of_landmark == 106,
            "legacy default landmark state must remain coherent");
    inspire::LandmarkParam explicit_null(YAML::Load("null"));
    Require(explicit_null.ReLoad("landmark_0_50"), "explicit null landmark table must use defaults");

    inspire::LandmarkParam parameter(MakeLandmarkTable());
    Require(parameter.ReLoad("valid_a"), "valid landmark config A must load");
    const ParamSnapshot stable = Snapshot(parameter);
    Require(!parameter.ReLoad("missing"), "missing config must fail");
    Require(Equal(Snapshot(parameter), stable), "missing config must not partially mutate state");
    Require(!parameter.ReLoad("bad_missing_semantic"), "missing semantic config must fail");
    Require(Equal(Snapshot(parameter), stable), "missing semantic config must roll back");
    Require(!parameter.ReLoad("bad_mesh"), "bad mesh length must fail");
    Require(Equal(Snapshot(parameter), stable), "bad mesh length must roll back");
    Require(!parameter.ReLoad("bad_index"), "out-of-range semantic index must fail");
    Require(Equal(Snapshot(parameter), stable), "bad semantic index must roll back");
    Require(!parameter.ReLoad("bad_type"), "YAML conversion failure must be contained");
    Require(Equal(Snapshot(parameter), stable), "YAML conversion failure must roll back");
    Require(parameter.ReLoad("valid_b"), "valid landmark config B must load");
    Require(parameter.landmark_engine_name == "valid_b" && parameter.input_size == 192 && parameter.semantic_index.left_eye_center == 1,
            "valid config B must commit as one state");
}

std::vector<inspirecv::Point2f> MakeLandmarks(float offset) {
    std::vector<inspirecv::Point2f> landmarks(106);
    for (size_t index = 0; index < landmarks.size(); ++index) {
        landmarks[index] = inspirecv::Point2f(offset + static_cast<float>(index) * 0.75f,
                                              offset * 0.5f + static_cast<float>((index * 7) % 29));
    }
    return landmarks;
}

void TestFaceObjectBoundaries() {
    inspire::FaceObjectInternal face(7, inspirecv::Rect2i(10, 20, 80, 90), 116);
    Require(!face.isStandard(), "standard-pose state must initialize false");
    auto landmarks = MakeLandmarks(3.0f);
    Require(face.SetLandmark(landmarks, true, false, 0.06f, 5, 212), "valid landmarks must commit");
    const auto stableLandmarks = face.landmark_;
    const auto stableHistory = face.landmark_smooth_aux_;
    const auto stableBbox = face.bbox_;

    std::vector<inspirecv::Point2f> shortLandmarks(12);
    Require(!face.SetLandmark(shortLandmarks, true, false, 0.06f, 5, 212), "short landmarks must fail");
    Require(face.landmark_.size() == stableLandmarks.size() && face.landmark_smooth_aux_.size() == stableHistory.size() &&
              face.bbox_.GetX() == stableBbox.GetX() && face.bbox_.GetY() == stableBbox.GetY(),
            "failed landmark update must leave all state unchanged");
    Require(!face.SetLandmark(landmarks, true, false, 0.06f, 0, 212), "zero smoothing cache must fail safely");
    Require(!face.DynamicSmoothParamUpdate(landmarks, face.landmark_smooth_aux_, 211, 0.06f, 5), "odd coordinate count must fail safely");

    const auto initialPose = face.getPoseEulerAngle();
    Require(!face.setPoseEulerAngle({0.1f, 0.2f}), "short pose vector must fail");
    Require(face.getPoseEulerAngle() == initialPose && !face.isStandard(), "failed pose update must not mutate state");
    Require(face.setPoseEulerAngle({0.1f, 0.2f, 0.3f}) && face.isStandard(), "frontal pose must be standard");
    Require(face.setPoseEulerAngle({0.8f, 0.7f, 0.0f}) && !face.isStandard(), "non-frontal pose must reset standard state");
    Require(!face.setAlignMeanSquareError(std::vector<inspirecv::Point2f>(4)), "short alignment landmarks must fail safely");
    Require(face.setAlignMeanSquareError(std::vector<inspirecv::Point2f>(5)), "five alignment landmarks must succeed");

    inspire::FaceObjectInternal rectangular(8, inspirecv::Rect2i(10, 20, 80, 90));
    const auto square = rectangular.GetRectSquare();
    Require(square.GetX() == 5 && square.GetY() == 20 && square.GetWidth() == 90 && square.GetHeight() == 90,
            "valid square conversion must preserve legacy coordinates");
    const auto padded = rectangular.GetRectSquare(0.5f);
    Require(padded.GetX() == -17 && padded.GetY() == -2 && padded.GetWidth() == 134 && padded.GetHeight() == 134,
            "valid padding must preserve legacy integer rounding");

    inspire::FaceObjectInternal empty(9, inspirecv::Rect2i());
    Require(empty.GetRectSquare().Area() == 0, "empty rectangle must not assert");
    inspire::FaceObjectInternal single_pixel(10, inspirecv::Rect2i(4, 5, 1, 1));
    Require(single_pixel.GetRectSquare().Area() == 0, "single-pixel rectangle must fail safely");
    Require(rectangular.GetRectSquare(-1.0f).Area() == 0, "collapsing padding must fail safely");
    Require(rectangular.GetRectSquare(std::numeric_limits<float>::infinity()).Area() == 0,
            "non-finite padding must fail safely");
}

void RunSmoothingAccuracyAndPerformanceGate() {
    inspire::FaceObjectInternal face(9, inspirecv::Rect2i(0, 0, 100, 100), 116);
    constexpr int iterations = 20000;
    std::vector<double> samples;
    samples.reserve(iterations);
    uint64_t digest = 1469598103934665603ULL;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto landmarks = MakeLandmarks(static_cast<float>(iteration % 41) * 0.125f);
        const auto begin = std::chrono::steady_clock::now();
        Require(face.SetLandmark(landmarks, false, false, 0.06f, 5, 212), "benchmark landmark update must succeed");
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        const auto &latest = face.landmark_smooth_aux_.back();
        Require(latest.size() == 106 && face.landmark_smooth_aux_.size() <= 5, "smoothing history shape must remain exact");
        const uint32_t bits = static_cast<uint32_t>(std::llround(latest[iteration % 106].GetX() * 1000.0f));
        digest = (digest ^ bits) * 1099511628211ULL;
    }
    std::sort(samples.begin(), samples.end());
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double p50 = samples[samples.size() / 2];
    const double p95 = samples[samples.size() * 95 / 100];
    Require(p95 < 1000.0, "landmark state update p95 must stay below 1 ms");
    std::cout << "Landmark state guard: iterations=" << iterations << " digest=0x" << std::hex << digest << std::dec << " mean_us=" << mean
              << " p50_us=" << p50 << " p95_us=" << p95 << std::endl;
}

}  // namespace

int main() {
    TestTransactionalLandmarkReload();
    TestFaceObjectBoundaries();
    RunSmoothingAccuracyAndPerformanceGate();
    std::cout << "Landmark transactional reload, FaceObject boundary, accuracy, and performance gates: PASS" << std::endl;
    return 0;
}
