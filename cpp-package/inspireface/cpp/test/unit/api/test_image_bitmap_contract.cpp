#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"

using inspireface_test::UniqueImageBitmap;
using inspireface_test::UniqueImageStream;

TEST_CASE("C API image bitmap validates input and copies caller-owned pixels", "[api][contract][image_bitmap][boundary]") {
    std::vector<uint8_t> pixels(4 * 3 * 3);
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<uint8_t>(i + 1);
    }
    const std::vector<uint8_t> original = pixels;
    HFImageBitmapData data = {pixels.data(), 4, 3, 3};
    HFImageBitmap output = reinterpret_cast<HFImageBitmap>(static_cast<uintptr_t>(1));

    CHECK(HFCreateImageBitmap(nullptr, &output) == HERR_INVALID_IMAGE_BITMAP_HANDLE);
    CHECK(HFCreateImageBitmap(&data, nullptr) == HERR_INVALID_IMAGE_BITMAP_HANDLE);

    data.data = nullptr;
    CHECK(HFCreateImageBitmap(&data, &output) == HERR_INVALID_PARAM);
    CHECK(output == nullptr);
    data.data = pixels.data();

    data.width = 0;
    CHECK(HFCreateImageBitmap(&data, &output) == HERR_INVALID_PARAM);
    data.width = 4;
    data.height = -1;
    CHECK(HFCreateImageBitmap(&data, &output) == HERR_INVALID_PARAM);
    data.height = 3;
    data.channels = 2;
    CHECK(HFCreateImageBitmap(&data, &output) == HERR_INVALID_PARAM);
    data.channels = 3;

    data.width = std::numeric_limits<HInt32>::max();
    data.height = 2;
    CHECK(HFCreateImageBitmap(&data, &output) == HERR_INVALID_PARAM);
    CHECK(output == nullptr);
    data.width = 4;
    data.height = 3;

    UniqueImageBitmap bitmap;
    REQUIRE(HFCreateImageBitmap(&data, bitmap.Put()) == HSUCCEED);
    std::fill(pixels.begin(), pixels.end(), 0);

    HFImageBitmapData actual = {nullptr, 0, 0, 0};
    REQUIRE(HFImageBitmapGetData(bitmap.Get(), &actual) == HSUCCEED);
    REQUIRE(actual.data != nullptr);
    CHECK(actual.width == 4);
    CHECK(actual.height == 3);
    CHECK(actual.channels == 3);
    CHECK(std::equal(original.begin(), original.end(), actual.data));
}

TEST_CASE("C API image bitmap copy owns independent pixel storage", "[api][contract][image_bitmap]") {
    std::vector<uint8_t> pixels(3 * 3 * 3, 17);
    HFImageBitmapData data = {pixels.data(), 3, 3, 3};
    UniqueImageBitmap source;
    UniqueImageBitmap copy;
    REQUIRE(HFCreateImageBitmap(&data, source.Put()) == HSUCCEED);
    REQUIRE(HFImageBitmapCopy(source.Get(), copy.Put()) == HSUCCEED);

    HFImageBitmapData source_data = {nullptr, 0, 0, 0};
    HFImageBitmapData copy_data = {nullptr, 0, 0, 0};
    REQUIRE(HFImageBitmapGetData(source.Get(), &source_data) == HSUCCEED);
    REQUIRE(HFImageBitmapGetData(copy.Get(), &copy_data) == HSUCCEED);
    REQUIRE(source_data.data != copy_data.data);

    source_data.data[0] = 99;
    CHECK(copy_data.data[0] == 17);
}

TEST_CASE("C API image bitmap file and stream conversions preserve valid dimensions", "[api][contract][image_bitmap]") {
    UniqueImageBitmap bitmap;
    REQUIRE(HFCreateImageBitmapFromFilePath(GET_DATA("data/bulk/pedestrian.png").c_str(), 3, bitmap.Put()) == HSUCCEED);

    HFImageBitmapData source = {nullptr, 0, 0, 0};
    REQUIRE(HFImageBitmapGetData(bitmap.Get(), &source) == HSUCCEED);
    REQUIRE(source.width > 0);
    REQUIRE(source.height > 0);
    REQUIRE(source.channels == 3);

    UniqueImageStream stream;
    REQUIRE(HFCreateImageStreamFromImageBitmap(bitmap.Get(), HF_CAMERA_ROTATION_90, stream.Put()) == HSUCCEED);

    UniqueImageBitmap processed;
    REQUIRE(HFCreateImageBitmapFromImageStreamProcess(stream.Get(), processed.Put(), 1, 0.5f) == HSUCCEED);
    HFImageBitmapData transformed = {nullptr, 0, 0, 0};
    REQUIRE(HFImageBitmapGetData(processed.Get(), &transformed) == HSUCCEED);
    CHECK(transformed.width > 0);
    CHECK(transformed.height > 0);
    CHECK(transformed.channels == 3);

    const std::string output_path = GET_SAVE_DATA("contract_bitmap_output.bmp");
    REQUIRE(HFImageBitmapWriteToFile(processed.Get(), output_path.c_str()) == HSUCCEED);
    std::ifstream output(output_path, std::ios::binary);
    CHECK(output.good());
}

TEST_CASE("C API image bitmap rejects invalid files, transforms, drawing parameters, and stale handles",
          "[api][contract][image_bitmap][boundary]") {
    HFImageBitmap output = reinterpret_cast<HFImageBitmap>(static_cast<uintptr_t>(1));
    CHECK(HFCreateImageBitmapFromFilePath(nullptr, 3, &output) == HERR_INVALID_PARAM);
    CHECK(output == nullptr);
    CHECK(HFCreateImageBitmapFromFilePath("", 3, &output) == HERR_INVALID_PARAM);
    CHECK(HFCreateImageBitmapFromFilePath(GET_DATA("data/does-not-exist.png").c_str(), 3, &output) == HERR_IMAGE_STREAM_DECODE_FAILED);
    CHECK(HFCreateImageBitmapFromFilePath(GET_DATA("data/bulk/pedestrian.png").c_str(), 2, &output) == HERR_INVALID_PARAM);

    std::vector<uint8_t> pixels(8 * 8 * 3, 127);
    HFImageBitmapData data = {pixels.data(), 8, 8, 3};
    UniqueImageBitmap bitmap;
    REQUIRE(HFCreateImageBitmap(&data, bitmap.Put()) == HSUCCEED);

    const HColor color = {255.0f, 0.0f, 0.0f};
    CHECK(HFImageBitmapDrawRect(bitmap.Get(), {-2, -2, 5, 5}, color, 1) == HSUCCEED);
    CHECK(HFImageBitmapDrawCircle(bitmap.Get(), {0, 0}, 3, color, 1) == HSUCCEED);
    CHECK(HFImageBitmapDrawCircleF(bitmap.Get(), {7.5f, 7.5f}, 3, color, 1) == HSUCCEED);
    CHECK(HFImageBitmapDrawRect(bitmap.Get(), {0, 0, 0, 2}, color, 1) == HERR_INVALID_PARAM);
    CHECK(HFImageBitmapDrawCircle(bitmap.Get(), {0, 0}, -1, color, 1) == HERR_INVALID_PARAM);
    CHECK(HFImageBitmapDrawCircleF(bitmap.Get(), {std::numeric_limits<float>::quiet_NaN(), 0.0f}, 1, color, 1) == HERR_INVALID_PARAM);
    CHECK(HFImageBitmapShow(bitmap.Get(), nullptr, 0) == HERR_INVALID_PARAM);
    char unused_title[] = "unused";
    CHECK(HFImageBitmapShow(nullptr, unused_title, 0) == HERR_INVALID_IMAGE_BITMAP_HANDLE);

    UniqueImageStream stream;
    REQUIRE(HFCreateImageStreamFromImageBitmap(bitmap.Get(), HF_CAMERA_ROTATION_0, stream.Put()) == HSUCCEED);
    CHECK(HFCreateImageBitmapFromImageStreamProcess(stream.Get(), &output, 0, 0.0f) == HERR_INVALID_PARAM);
    CHECK(HFCreateImageBitmapFromImageStreamProcess(stream.Get(), &output, 0, std::numeric_limits<float>::infinity()) == HERR_INVALID_PARAM);
    CHECK(HFCreateImageStreamFromImageBitmap(bitmap.Get(), static_cast<HFRotation>(99), stream.Put()) == HERR_INVALID_IMAGE_STREAM_PARAM);

    HFImageBitmap stale = bitmap.ReleaseOwnership();
    REQUIRE(HFReleaseImageBitmap(stale) == HSUCCEED);
    CHECK(HFReleaseImageBitmap(stale) == HERR_INVALID_IMAGE_BITMAP_HANDLE);
    data = {pixels.data(), 8, 8, 3};
    CHECK(HFImageBitmapGetData(stale, &data) == HERR_INVALID_IMAGE_BITMAP_HANDLE);
    CHECK(data.data == nullptr);
    CHECK(data.width == 0);
    CHECK(data.height == 0);
    CHECK(data.channels == 0);
    CHECK(HFImageBitmapCopy(stale, &output) == HERR_INVALID_IMAGE_BITMAP_HANDLE);
    CHECK(HFImageBitmapWriteToFile(stale, GET_SAVE_DATA("stale.bmp").c_str()) == HERR_INVALID_IMAGE_BITMAP_HANDLE);
}
