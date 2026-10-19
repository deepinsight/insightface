#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"

using inspireface_test::UniqueImageBitmap;
using inspireface_test::UniqueImageStream;

namespace {

HInt32 LiveStreamCount() {
    HInt32 count = -1;
    REQUIRE(HFDeBugGetUnreleasedStreamsCount(&count) == HSUCCEED);
    return count;
}

}  // namespace

TEST_CASE("C API image stream validates creation parameters", "[api][contract][image_stream][boundary]") {
    std::vector<uint8_t> pixels(8 * 8 * 4, 127);
    HFImageData data = {pixels.data(), 8, 8, HF_STREAM_BGR, HF_CAMERA_ROTATION_0};
    HFImageStream output = reinterpret_cast<HFImageStream>(static_cast<uintptr_t>(1));

    CHECK(HFCreateImageStream(nullptr, &output) == HERR_INVALID_IMAGE_STREAM_HANDLE);
    CHECK(HFCreateImageStream(&data, nullptr) == HERR_INVALID_IMAGE_STREAM_HANDLE);

    data.data = nullptr;
    CHECK(HFCreateImageStream(&data, &output) == HERR_INVALID_IMAGE_STREAM_PARAM);
    CHECK(output == nullptr);
    data.data = pixels.data();

    for (const auto dimension : std::array<int, 2>{{-1, 0}}) {
        data.width = dimension;
        output = reinterpret_cast<HFImageStream>(static_cast<uintptr_t>(1));
        CHECK(HFCreateImageStream(&data, &output) == HERR_INVALID_IMAGE_STREAM_PARAM);
        CHECK(output == nullptr);
    }
    data.width = 8;

    data.height = 0;
    CHECK(HFCreateImageStream(&data, &output) == HERR_INVALID_IMAGE_STREAM_PARAM);
    data.height = 8;

    data.format = static_cast<HFImageFormat>(-1);
    CHECK(HFCreateImageStream(&data, &output) == HERR_INVALID_IMAGE_STREAM_PARAM);
    data.format = HF_STREAM_BGR;

    data.rotation = static_cast<HFRotation>(99);
    CHECK(HFCreateImageStream(&data, &output) == HERR_INVALID_IMAGE_STREAM_PARAM);
    CHECK(output == nullptr);

    data.rotation = HF_CAMERA_ROTATION_0;
    for (const auto format : std::array<HFImageFormat, 3>{{HF_STREAM_YUV_NV12, HF_STREAM_YUV_NV21, HF_STREAM_I420}}) {
        data.format = format;
        data.width = 7;
        data.height = 8;
        CHECK(HFCreateImageStream(&data, &output) == HERR_INVALID_IMAGE_STREAM_PARAM);
        data.width = 8;
        data.height = 7;
        CHECK(HFCreateImageStream(&data, &output) == HERR_INVALID_IMAGE_STREAM_PARAM);
    }
}

TEST_CASE("C API image stream accepts every declared format and rotation", "[api][contract][image_stream]") {
    const HInt32 before = LiveStreamCount();
    std::vector<uint8_t> pixels(8 * 8 * 4, 127);

    const std::array<HFImageFormat, 8> formats = {{HF_STREAM_RGB, HF_STREAM_BGR, HF_STREAM_RGBA, HF_STREAM_BGRA,
                                                   HF_STREAM_YUV_NV12, HF_STREAM_YUV_NV21, HF_STREAM_I420, HF_STREAM_GRAY}};
    const std::array<HFRotation, 4> rotations = {
      {HF_CAMERA_ROTATION_0, HF_CAMERA_ROTATION_90, HF_CAMERA_ROTATION_180, HF_CAMERA_ROTATION_270}};

    for (const auto format : formats) {
        for (const auto rotation : rotations) {
            HFImageData data = {pixels.data(), 8, 8, format, rotation};
            UniqueImageStream stream;
            REQUIRE(HFCreateImageStream(&data, stream.Put()) == HSUCCEED);
            REQUIRE(stream.Get() != nullptr);
        }
    }
    CHECK(LiveStreamCount() == before);
}

TEST_CASE("C API valid NV21 buffers can be processed without crossing their bounds", "[api][contract][image_stream][image_process]") {
    constexpr HInt32 width = 8;
    constexpr HInt32 height = 8;
    std::vector<uint8_t> pixels(width * height * 3 / 2, 128);
    HFImageData data = {pixels.data(), width, height, HF_STREAM_YUV_NV21, HF_CAMERA_ROTATION_90};
    UniqueImageStream stream;
    REQUIRE(HFCreateImageStream(&data, stream.Put()) == HSUCCEED);

    UniqueImageBitmap bitmap;
    REQUIRE(HFCreateImageBitmapFromImageStreamProcess(stream.Get(), bitmap.Put(), 1, 1.0f) == HSUCCEED);
    HFImageBitmapData output = {};
    REQUIRE(HFImageBitmapGetData(bitmap.Get(), &output) == HSUCCEED);
    CHECK(output.width == height);
    CHECK(output.height == width);
    CHECK(output.channels == 3);
}

TEST_CASE("C API bitmap-derived streams retain an immutable pixel snapshot", "[api][contract][image_stream][lifetime]") {
    constexpr HInt32 width = 8;
    constexpr HInt32 height = 6;
    std::vector<uint8_t> pixels(width * height * 3);
    for (size_t index = 0; index < pixels.size(); ++index) {
        pixels[index] = static_cast<uint8_t>((index * 31 + 7) & 0xff);
    }
    HFImageBitmapData input = {pixels.data(), width, height, 3};
    UniqueImageBitmap source;
    UniqueImageStream stream;
    REQUIRE(HFCreateImageBitmap(&input, source.Put()) == HSUCCEED);
    REQUIRE(HFCreateImageStreamFromImageBitmap(source.Get(), HF_CAMERA_ROTATION_0, stream.Put()) == HSUCCEED);

    auto decode_pixels = [&]() {
        UniqueImageBitmap decoded;
        REQUIRE(HFCreateImageBitmapFromImageStreamProcess(stream.Get(), decoded.Put(), 0, 1.0f) == HSUCCEED);
        HFImageBitmapData actual = {};
        REQUIRE(HFImageBitmapGetData(decoded.Get(), &actual) == HSUCCEED);
        REQUIRE(actual.width == width);
        REQUIRE(actual.height == height);
        REQUIRE(actual.channels == 3);
        return std::vector<uint8_t>(actual.data, actual.data + pixels.size());
    };

    // Compare decoded snapshots before/after mutation and release. Even at unit
    // scale, the image-processing backend can round pixels during resampling;
    // that must not be confused with a change in the stream's owned snapshot.
    const auto expected = decode_pixels();
    REQUIRE(std::any_of(expected.begin(), expected.end(), [](uint8_t value) { return value != 0; }));

    HFImageBitmapData source_data = {};
    REQUIRE(HFImageBitmapGetData(source.Get(), &source_data) == HSUCCEED);
    REQUIRE(std::memcmp(source_data.data, pixels.data(), pixels.size()) == 0);
    std::fill(source_data.data, source_data.data + expected.size(), 0);
    CHECK(decode_pixels() == expected);
    REQUIRE(source.Reset() == HSUCCEED);
    CHECK(decode_pixels() == expected);
}

TEST_CASE("C API image stream rejects arithmetic overflow and unconfigured processing", "[api][contract][image_stream][boundary]") {
    uint8_t pixel = 0;
    HFImageData oversized = {&pixel, std::numeric_limits<HInt32>::max(), 2, HF_STREAM_BGRA, HF_CAMERA_ROTATION_0};
    HFImageStream output = reinterpret_cast<HFImageStream>(static_cast<uintptr_t>(1));
    CHECK(HFCreateImageStream(&oversized, &output) == HERR_INVALID_IMAGE_STREAM_PARAM);
    CHECK(output == nullptr);

    UniqueImageStream empty;
    REQUIRE(HFCreateImageStreamEmpty(empty.Put()) == HSUCCEED);
    HFImageBitmap bitmap = reinterpret_cast<HFImageBitmap>(static_cast<uintptr_t>(1));
    CHECK(HFCreateImageBitmapFromImageStreamProcess(empty.Get(), &bitmap, 0, 1.0f) == HERR_INVALID_PARAM);
    CHECK(bitmap == nullptr);

    std::vector<uint8_t> pixels(8 * 8 * 3, 1);
    CHECK(HFImageStreamSetFormat(empty.Get(), HF_STREAM_BGR) == HSUCCEED);
    CHECK(HFImageStreamSetBuffer(empty.Get(), pixels.data(), 8, 8) == HSUCCEED);
    CHECK(HFCreateImageBitmapFromImageStreamProcess(empty.Get(), &bitmap, 2, 1.0f) == HERR_INVALID_PARAM);
    CHECK(HFCreateImageBitmapFromImageStreamProcess(empty.Get(), &bitmap, 0, 0.01f) == HERR_INVALID_PARAM);
    CHECK(HFCreateImageBitmapFromImageStreamProcess(empty.Get(), &bitmap, 0, std::numeric_limits<float>::max()) == HERR_INVALID_PARAM);
}

TEST_CASE("C API empty image stream supports ordered configuration and rejects stale handles",
          "[api][contract][image_stream][boundary]") {
    std::vector<uint8_t> pixels(8 * 8 * 3, 64);
    UniqueImageStream stream;
    REQUIRE(HFCreateImageStreamEmpty(stream.Put()) == HSUCCEED);

    CHECK(HFImageStreamSetFormat(stream.Get(), HF_STREAM_BGR) == HSUCCEED);
    CHECK(HFImageStreamSetRotation(stream.Get(), HF_CAMERA_ROTATION_270) == HSUCCEED);
    CHECK(HFImageStreamSetBuffer(stream.Get(), pixels.data(), 8, 8) == HSUCCEED);

    CHECK(HFImageStreamSetBuffer(stream.Get(), nullptr, 8, 8) == HERR_INVALID_IMAGE_STREAM_PARAM);
    CHECK(HFImageStreamSetBuffer(stream.Get(), pixels.data(), 0, 8) == HERR_INVALID_IMAGE_STREAM_PARAM);
    CHECK(HFImageStreamSetBuffer(stream.Get(), pixels.data(), 8, -1) == HERR_INVALID_IMAGE_STREAM_PARAM);
    CHECK(HFImageStreamSetFormat(stream.Get(), HF_STREAM_YUV_NV21) == HSUCCEED);
    CHECK(HFImageStreamSetBuffer(stream.Get(), pixels.data(), 7, 8) == HERR_INVALID_IMAGE_STREAM_PARAM);
    CHECK(HFImageStreamSetBuffer(stream.Get(), pixels.data(), 8, 7) == HERR_INVALID_IMAGE_STREAM_PARAM);
    CHECK(HFImageStreamSetFormat(stream.Get(), HF_STREAM_BGR) == HSUCCEED);
    CHECK(HFImageStreamSetBuffer(stream.Get(), pixels.data(), 7, 7) == HSUCCEED);
    CHECK(HFImageStreamSetFormat(stream.Get(), static_cast<HFImageFormat>(99)) == HERR_INVALID_IMAGE_STREAM_PARAM);
    CHECK(HFImageStreamSetRotation(stream.Get(), static_cast<HFRotation>(99)) == HERR_INVALID_IMAGE_STREAM_PARAM);

    HFImageStream stale = stream.ReleaseOwnership();
    REQUIRE(HFReleaseImageStream(stale) == HSUCCEED);
    CHECK(HFReleaseImageStream(stale) == HERR_INVALID_IMAGE_STREAM_HANDLE);
    CHECK(HFImageStreamSetBuffer(stale, pixels.data(), 8, 8) == HERR_INVALID_IMAGE_STREAM_HANDLE);
    CHECK(HFImageStreamSetFormat(stale, HF_STREAM_BGR) == HERR_INVALID_IMAGE_STREAM_HANDLE);
    CHECK(HFImageStreamSetRotation(stale, HF_CAMERA_ROTATION_0) == HERR_INVALID_IMAGE_STREAM_HANDLE);
}

TEST_CASE("C API image stream rejects handles belonging to another resource type", "[api][contract][image_stream][boundary]") {
    std::vector<uint8_t> pixels(4 * 4 * 3, 32);
    HFImageBitmapData bitmap_data = {pixels.data(), 4, 4, 3};
    UniqueImageBitmap bitmap;
    REQUIRE(HFCreateImageBitmap(&bitmap_data, bitmap.Put()) == HSUCCEED);

    const auto wrong_handle = reinterpret_cast<HFImageStream>(bitmap.Get());
    CHECK(HFImageStreamSetFormat(wrong_handle, HF_STREAM_BGR) == HERR_INVALID_IMAGE_STREAM_HANDLE);
    CHECK(HFImageStreamSetRotation(wrong_handle, HF_CAMERA_ROTATION_0) == HERR_INVALID_IMAGE_STREAM_HANDLE);
    CHECK(HFReleaseImageStream(wrong_handle) == HERR_INVALID_IMAGE_STREAM_HANDLE);
}

TEST_CASE("C API image stream debug decoder validates handles and writes into the ignored output directory",
          "[api][contract][image_stream][debug]") {
    std::vector<uint8_t> pixels(8 * 8 * 3, 64);
    HFImageData data = {pixels.data(), 8, 8, HF_STREAM_BGR, HF_CAMERA_ROTATION_0};
    UniqueImageStream stream;
    REQUIRE(HFCreateImageStream(&data, stream.Put()) == HSUCCEED);

    const std::string output_path = GET_SAVE_DATA("contract_stream_decode.bmp");
    REQUIRE(HFDeBugImageStreamDecodeSave(stream.Get(), output_path.c_str()) == HSUCCEED);
    std::ifstream output(output_path, std::ios::binary);
    CHECK(output.good());
    CHECK(HFDeBugImageStreamDecodeSave(stream.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFDeBugImageStreamDecodeSave(nullptr, output_path.c_str()) == HERR_INVALID_IMAGE_STREAM_HANDLE);

    // The display helper is intentionally exercised only on an invalid handle;
    // a unit test must not open a GUI window or create an implicit tmp.jpg.
    HFDeBugImageStreamImShow(nullptr);
}
