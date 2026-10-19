/**
 * Created by Jingyu Yan
 * @date 2025-03-29
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/simple_csv_writer.h"
#include "unit/test_helper/test_help.h"
#include "unit/test_helper/test_tools.h"
#include <inspireface/include/inspireface/inspireface.hpp>

using namespace inspire;

namespace {

using Clock = std::chrono::steady_clock;

enum class PoseAxis {
    Yaw,
    Pitch,
    Roll,
};

struct PoseCase {
    const char* name;
    const char* image_path;
    PoseAxis axis;
    float minimum;
    float maximum;
};

float SelectAngle(const Face3DAngle& angle, PoseAxis axis) {
    if (axis == PoseAxis::Yaw) {
        return angle.yaw;
    }
    if (axis == PoseAxis::Pitch) {
        return angle.pitch;
    }
    return angle.roll;
}

int64_t Median(std::vector<int64_t> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

template <typename Callable>
int64_t MeasureMicroseconds(Callable&& callable, HResult& status) {
    const auto begin = Clock::now();
    status = callable();
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin).count();
}

}  // namespace

TEST_CASE("test_SessionFaceTrack", "[session_face_track][model_accuracy]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Face detection from image") {
        CustomPipelineParameter param;
        int32_t ret;

        std::unique_ptr<Session> session(Session::CreatePtr(DetectModuleMode::DETECT_MODE_ALWAYS_DETECT, 3, param));
        REQUIRE(session != nullptr);

        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
        auto process = inspirecv::FrameProcess::Create(image, inspirecv::DATA_FORMAT::BGR, inspirecv::ROTATION_MODE::ROTATION_0);

        std::vector<FaceTrackWrap> results;
        ret = session->FaceDetectAndTrack(process, results);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(results.size() == 1);
    }

    SECTION("Head pose disabled returns neutral angles without affecting detection") {
        CustomPipelineParameter param;
        REQUIRE_FALSE(param.enable_face_pose);

        std::unique_ptr<Session> session(Session::CreatePtr(DetectModuleMode::DETECT_MODE_ALWAYS_DETECT, 3, param));
        REQUIRE(session != nullptr);

        const auto image = inspirecv::Image::Create(GET_DATA("data/pose/left_face.jpeg"));
        REQUIRE(!image.Empty());
        auto process = inspirecv::FrameProcess::Create(image, inspirecv::DATA_FORMAT::BGR, inspirecv::ROTATION_MODE::ROTATION_0);
        std::vector<FaceTrackWrap> results;
        REQUIRE(session->FaceDetectAndTrack(process, results) == HSUCCEED);
        REQUIRE(results.size() == 1);
        CHECK(results[0].face3DAngle.yaw == Approx(0.0f).margin(1e-6f));
        CHECK(results[0].face3DAngle.pitch == Approx(0.0f).margin(1e-6f));
        CHECK(results[0].face3DAngle.roll == Approx(0.0f).margin(1e-6f));
    }

    SECTION("Head pose estimation matches the C API") {
        CustomPipelineParameter cppParameter;
        cppParameter.enable_face_pose = true;
        std::unique_ptr<Session> cppSession(Session::CreatePtr(DetectModuleMode::DETECT_MODE_ALWAYS_DETECT, 3, cppParameter));
        REQUIRE(cppSession != nullptr);

        HFSessionCustomParameter cParameter = {};
        cParameter.enable_face_pose = 1;
        HFSession cHandle = nullptr;
        REQUIRE(HFCreateInspireFaceSession(cParameter, HF_DETECT_MODE_ALWAYS_DETECT, 3, -1, -1, &cHandle) == HSUCCEED);
        inspireface_test::UniqueSession cSession(cHandle);

        const std::array<PoseCase, 6> poseCases = {{{"left", "data/pose/left_face.jpeg", PoseAxis::Yaw, -90.0f, -10.0f},
                                                    {"right", "data/pose/right_face.png", PoseAxis::Yaw, 10.0f, 90.0f},
                                                    {"rise", "data/pose/rise_face.jpeg", PoseAxis::Pitch, 3.0f, 90.0f},
                                                    {"lower", "data/pose/lower_face.jpeg", PoseAxis::Pitch, -90.0f, -10.0f},
                                                    {"left roll", "data/pose/left_wryneck.png", PoseAxis::Roll, -90.0f, -30.0f},
                                                    {"right roll", "data/pose/right_wryneck.png", PoseAxis::Roll, 25.0f, 90.0f}}};
        std::vector<int64_t> cppLatencies;
        std::vector<int64_t> cLatencies;
        cppLatencies.reserve(poseCases.size());
        cLatencies.reserve(poseCases.size());

        for (size_t caseIndex = 0; caseIndex < poseCases.size(); ++caseIndex) {
            const auto& poseCase = poseCases[caseIndex];
            INFO("Pose case: " << poseCase.name << ", image: " << poseCase.image_path);

            const auto image = inspirecv::Image::Create(GET_DATA(poseCase.image_path));
            REQUIRE(!image.Empty());
            auto cppProcess = inspirecv::FrameProcess::Create(image, inspirecv::DATA_FORMAT::BGR, inspirecv::ROTATION_MODE::ROTATION_0);
            std::vector<FaceTrackWrap> cppResults;

            HFImageStream streamHandle = nullptr;
            REQUIRE(CVImageToImageStream(image, streamHandle) == HSUCCEED);
            inspireface_test::UniqueImageStream cStream(streamHandle);
            HFMultipleFaceData cResults = {};

            HResult cppStatus = HERR_UNKNOWN;
            HResult cStatus = HERR_UNKNOWN;
            int64_t cppMicroseconds = 0;
            int64_t cMicroseconds = 0;
            const auto runCpp = [&]() { return cppSession->FaceDetectAndTrack(cppProcess, cppResults); };
            const auto runC = [&]() { return HFExecuteFaceTrack(cSession.Get(), cStream.Get(), &cResults); };
            if (caseIndex % 2 == 0) {
                cppMicroseconds = MeasureMicroseconds(runCpp, cppStatus);
                cMicroseconds = MeasureMicroseconds(runC, cStatus);
            } else {
                cMicroseconds = MeasureMicroseconds(runC, cStatus);
                cppMicroseconds = MeasureMicroseconds(runCpp, cppStatus);
            }
            REQUIRE(cppStatus == HSUCCEED);
            REQUIRE(cStatus == HSUCCEED);
            REQUIRE(cppMicroseconds > 0);
            REQUIRE(cMicroseconds > 0);
            cppLatencies.push_back(cppMicroseconds);
            cLatencies.push_back(cMicroseconds);

            REQUIRE(cppResults.size() == 1);
            REQUIRE(cResults.detectedNum == 1);
            REQUIRE(cResults.rects != nullptr);
            REQUIRE(cResults.angles.yaw != nullptr);
            REQUIRE(cResults.angles.pitch != nullptr);
            REQUIRE(cResults.angles.roll != nullptr);

            const auto& cppFace = cppResults[0];
            CHECK(std::abs(cppFace.rect.x - cResults.rects[0].x) <= 1);
            CHECK(std::abs(cppFace.rect.y - cResults.rects[0].y) <= 1);
            CHECK(std::abs(cppFace.rect.width - cResults.rects[0].width) <= 1);
            CHECK(std::abs(cppFace.rect.height - cResults.rects[0].height) <= 1);

            const Face3DAngle cAngle = {cResults.angles.roll[0], cResults.angles.yaw[0], cResults.angles.pitch[0]};
            const std::array<float, 3> cppAngles =
              {{cppFace.face3DAngle.yaw, cppFace.face3DAngle.pitch, cppFace.face3DAngle.roll}};
            const std::array<float, 3> cAngles = {{cAngle.yaw, cAngle.pitch, cAngle.roll}};
            for (size_t angleIndex = 0; angleIndex < cppAngles.size(); ++angleIndex) {
                CHECK(std::isfinite(cppAngles[angleIndex]));
                CHECK(std::isfinite(cAngles[angleIndex]));
                CHECK(cppAngles[angleIndex] >= -90.0f);
                CHECK(cppAngles[angleIndex] <= 90.0f);
                CHECK(cppAngles[angleIndex] == Approx(cAngles[angleIndex]).margin(1e-4f));
            }

            const float cppExpectedAngle = SelectAngle(cppFace.face3DAngle, poseCase.axis);
            const float cExpectedAngle = SelectAngle(cAngle, poseCase.axis);
            CHECK(cppExpectedAngle > poseCase.minimum);
            CHECK(cppExpectedAngle < poseCase.maximum);
            CHECK(cExpectedAngle > poseCase.minimum);
            CHECK(cExpectedAngle < poseCase.maximum);

            TEST_PRINT("Head pose {}: C++ {} us, C API {} us, yaw {:.3f}, pitch {:.3f}, roll {:.3f}", poseCase.name,
                       cppMicroseconds, cMicroseconds, cppFace.face3DAngle.yaw, cppFace.face3DAngle.pitch, cppFace.face3DAngle.roll);
        }

        const int64_t cppMedianMicroseconds = Median(cppLatencies);
        const int64_t cMedianMicroseconds = Median(cLatencies);
        TEST_PRINT("Head pose median latency: C++ {} us, C API {} us", cppMedianMicroseconds, cMedianMicroseconds);
        // Keep this hardware-independent: the C++ wrapper should remain within
        // the C API latency envelope because both call the same pose pipeline.
        CHECK(cppMedianMicroseconds <= static_cast<int64_t>(cMedianMicroseconds * 1.35) + 2000);
    }
}
