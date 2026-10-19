#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"

namespace {

struct DetectionDigest {
    HInt32 detected_num{0};
    std::vector<HFaceRect> rects;
    std::vector<HInt32> track_ids;
    std::vector<HInt32> track_counts;
    std::vector<HFloat> confidence;
    std::vector<HFloat> roll;
    std::vector<HFloat> yaw;
    std::vector<HFloat> pitch;
    std::vector<std::vector<HBuffer>> tokens;
};

DetectionDigest Detect(HFSession session, const inspirecv::Image& image) {
    HFImageData image_data = {};
    image_data.data = const_cast<HUInt8*>(image.Data());
    image_data.width = image.Width();
    image_data.height = image.Height();
    image_data.format = HF_STREAM_BGR;
    image_data.rotation = HF_CAMERA_ROTATION_0;
    inspireface_test::UniqueImageStream stream;
    REQUIRE(HFCreateImageStream(&image_data, stream.Put()) == HSUCCEED);

    HFMultipleFaceData data = {};
    REQUIRE(HFExecuteFaceTrack(session, stream.Get(), &data) == HSUCCEED);
    DetectionDigest result;
    result.detected_num = data.detectedNum;
    for (HInt32 index = 0; index < data.detectedNum; ++index) {
        result.rects.push_back(data.rects[index]);
        result.track_ids.push_back(data.trackIds[index]);
        result.track_counts.push_back(data.trackCounts[index]);
        result.confidence.push_back(data.detConfidence[index]);
        result.roll.push_back(data.angles.roll[index]);
        result.yaw.push_back(data.angles.yaw[index]);
        result.pitch.push_back(data.angles.pitch[index]);
        const HFFaceBasicToken& token = data.tokens[index];
        REQUIRE(token.size >= 0);
        if (token.size == 0) {
            result.tokens.emplace_back();
        } else {
            REQUIRE(token.data != nullptr);
            const auto* token_begin = static_cast<const HBuffer*>(token.data);
            result.tokens.emplace_back(token_begin, token_begin + token.size);
        }
    }
    return result;
}

void CheckEqual(const DetectionDigest& expected, const DetectionDigest& actual) {
    REQUIRE(actual.detected_num == expected.detected_num);
    REQUIRE(actual.rects.size() == expected.rects.size());
    for (size_t index = 0; index < expected.rects.size(); ++index) {
        CHECK(std::memcmp(&actual.rects[index], &expected.rects[index], sizeof(HFaceRect)) == 0);
    }
    CHECK(actual.track_ids == expected.track_ids);
    CHECK(actual.track_counts == expected.track_counts);
    CHECK(actual.confidence == expected.confidence);
    CHECK(actual.roll == expected.roll);
    CHECK(actual.yaw == expected.yaw);
    CHECK(actual.pitch == expected.pitch);
    CHECK(actual.tokens == expected.tokens);
}

}  // namespace

TEST_CASE("Resource-pack validation preserves multi-image detection results and latency",
          "[api][resource_pack][accuracy][consistency][performance]") {
    HFSessionCustomParameter parameter = {};
    inspireface_test::UniqueSession session;
    REQUIRE(HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 8, -1, -1, session.Put()) == HSUCCEED);

    const std::vector<std::string> image_paths = {
      "data/bulk/kun.jpg",
      "data/bulk/r0.jpg",
      "data/bulk/woman.png",
      "data/bulk/pedestrian.png",
    };
    std::vector<inspirecv::Image> images;
    std::vector<DetectionDigest> expected;
    for (const auto& relative_path : image_paths) {
        images.emplace_back(inspirecv::Image::Create(GET_DATA(relative_path)));
        REQUIRE_FALSE(images.back().Empty());
        expected.emplace_back(Detect(session.Get(), images.back()));
    }

    HFResourcePackInfo info = {};
    info.structSize = sizeof(info);
    info.structVersion = HF_RESOURCE_PACK_INFO_VERSION;
    const auto started = std::chrono::steady_clock::now();
    REQUIRE(HFValidateResourcePack(GET_RUNTIME_FULLPATH_NAME.c_str(), &info) == HSUCCEED);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    TEST_PRINT("Resource-pack validation consistency gate: {:.3f} ms",
               std::chrono::duration<double, std::milli>(elapsed).count());
    CHECK(elapsed < std::chrono::seconds(2));

    for (size_t index = 0; index < images.size(); ++index) {
        CheckEqual(expected[index], Detect(session.Get(), images[index]));
    }
}
