#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <inspireface/include/inspireface/inspireface.hpp>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/test_tools.h"

namespace {

using Clock = std::chrono::steady_clock;

int64_t Median(std::vector<int64_t> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

template <typename Callable>
int64_t MeasureMicroseconds(Callable&& callable, int32_t& status) {
    const auto begin = Clock::now();
    status = callable();
    const auto end = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

}  // namespace

TEST_CASE("C and C++ detection stay consistent across representative images", "[cpp_api][performance][consistency][accuracy]") {
    inspire::CustomPipelineParameter cpp_parameter;
    auto cpp_session = inspire::Session::Create(inspire::DETECT_MODE_ALWAYS_DETECT, 5, cpp_parameter);

    HFSession c_handle = nullptr;
    REQUIRE(HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, 5, -1, -1, &c_handle) == HSUCCEED);
    inspireface_test::UniqueSession c_session(c_handle);

    const std::vector<std::string> image_paths = {
      "data/bulk/kun.jpg", "data/bulk/r0.jpg", "data/bulk/woman.png", "data/crop/no_face.png"};
    for (const auto& relative_path : image_paths) {
        DYNAMIC_SECTION(relative_path) {
            const auto image = inspirecv::Image::Create(GET_DATA(relative_path));
            REQUIRE(!image.Empty());
            auto cpp_process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
            std::vector<inspire::FaceTrackWrap> cpp_faces;
            REQUIRE(cpp_session.FaceDetectAndTrack(cpp_process, cpp_faces) == HSUCCEED);

            HFImageStream stream_handle = nullptr;
            REQUIRE(CVImageToImageStream(image, stream_handle) == HSUCCEED);
            inspireface_test::UniqueImageStream stream(stream_handle);
            HFMultipleFaceData c_faces = {};
            REQUIRE(HFExecuteFaceTrack(c_session.Get(), stream.Get(), &c_faces) == HSUCCEED);
            REQUIRE(cpp_faces.size() == static_cast<size_t>(c_faces.detectedNum));

            for (size_t index = 0; index < cpp_faces.size(); ++index) {
                const auto cpp_box = cpp_session.GetFaceBoundingBox(cpp_faces[index]);
                CHECK(std::abs(cpp_box.GetX() - c_faces.rects[index].x) <= 1);
                CHECK(std::abs(cpp_box.GetY() - c_faces.rects[index].y) <= 1);
                CHECK(std::abs(cpp_box.GetWidth() - c_faces.rects[index].width) <= 1);
                CHECK(std::abs(cpp_box.GetHeight() - c_faces.rects[index].height) <= 1);

                const auto cpp_five = cpp_session.GetFaceFiveKeyPoints(cpp_faces[index]);
                std::vector<HPoint2f> c_five(5);
                REQUIRE(HFGetFaceFiveKeyPointsFromFaceToken(c_faces.tokens[index], c_five.data(), 5) == HSUCCEED);
                for (size_t point = 0; point < cpp_five.size(); ++point) {
                    CHECK(cpp_five[point].GetX() == Approx(c_five[point].x).margin(1e-4f));
                    CHECK(cpp_five[point].GetY() == Approx(c_five[point].y).margin(1e-4f));
                }
            }
        }
    }
}

TEST_CASE("C++ detection wrapper latency stays within the C API envelope", "[cpp_api][performance][latency]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto cpp_process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    inspire::CustomPipelineParameter cpp_parameter;
    auto cpp_session = inspire::Session::Create(inspire::DETECT_MODE_ALWAYS_DETECT, 1, cpp_parameter);
    std::vector<inspire::FaceTrackWrap> cpp_faces;

    HFSession c_handle = nullptr;
    REQUIRE(HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &c_handle) == HSUCCEED);
    inspireface_test::UniqueSession c_session(c_handle);
    HFImageStream stream_handle = nullptr;
    REQUIRE(CVImageToImageStream(image, stream_handle) == HSUCCEED);
    inspireface_test::UniqueImageStream stream(stream_handle);
    HFMultipleFaceData c_faces = {};

    for (int warmup = 0; warmup < 2; ++warmup) {
        REQUIRE(cpp_session.FaceDetectAndTrack(cpp_process, cpp_faces) == HSUCCEED);
        REQUIRE(HFExecuteFaceTrack(c_session.Get(), stream.Get(), &c_faces) == HSUCCEED);
        REQUIRE(cpp_faces.size() == static_cast<size_t>(c_faces.detectedNum));
    }

    std::vector<int64_t> cpp_samples;
    std::vector<int64_t> c_samples;
    for (int iteration = 0; iteration < 9; ++iteration) {
        int32_t cpp_status = HSUCCEED;
        int32_t c_status = HSUCCEED;
        if (iteration % 2 == 0) {
            cpp_samples.push_back(MeasureMicroseconds(
              [&] { return cpp_session.FaceDetectAndTrack(cpp_process, cpp_faces); }, cpp_status));
            c_samples.push_back(MeasureMicroseconds(
              [&] { return HFExecuteFaceTrack(c_session.Get(), stream.Get(), &c_faces); }, c_status));
        } else {
            c_samples.push_back(MeasureMicroseconds(
              [&] { return HFExecuteFaceTrack(c_session.Get(), stream.Get(), &c_faces); }, c_status));
            cpp_samples.push_back(MeasureMicroseconds(
              [&] { return cpp_session.FaceDetectAndTrack(cpp_process, cpp_faces); }, cpp_status));
        }
        REQUIRE(cpp_status == HSUCCEED);
        REQUIRE(c_status == HSUCCEED);
        REQUIRE(cpp_faces.size() == static_cast<size_t>(c_faces.detectedNum));
    }

    const int64_t cpp_median_us = Median(cpp_samples);
    const int64_t c_median_us = Median(c_samples);
    TEST_PRINT("C++ API median detection latency: {} us; C API median: {} us", cpp_median_us, c_median_us);
    REQUIRE(cpp_median_us > 0);
    REQUIRE(c_median_us > 0);
    // The fixed allowance absorbs timer/scheduler noise; the ratio catches a
    // meaningful wrapper-side regression without imposing a hardware budget.
    CHECK(cpp_median_us <= static_cast<int64_t>(c_median_us * 1.35) + 2000);
}
