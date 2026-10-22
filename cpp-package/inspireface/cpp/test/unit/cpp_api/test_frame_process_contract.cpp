#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <inspireface/include/inspireface/inspireface.hpp>

#include "settings/test_settings.h"

TEST_CASE("C++ FrameProcess preserves configuration across copy and move", "[cpp_api][contract][frame_process]") {
    static_assert(std::is_copy_constructible<inspirecv::FrameProcess>::value, "FrameProcess must remain copy constructible");
    static_assert(std::is_move_constructible<inspirecv::FrameProcess>::value, "FrameProcess must remain move constructible");

    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto original = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    original.SetPreviewScale(0.5f);

    inspirecv::FrameProcess copied(original);
    CHECK(copied.GetWidth() == image.Width());
    CHECK(copied.GetHeight() == image.Height());
    CHECK(copied.GetPreviewScale() == Approx(0.5f));
    CHECK(copied.getRotationMode() == inspirecv::ROTATION_0);

    original.SetRotationMode(inspirecv::ROTATION_90);
    original.SetPreviewScale(0.25f);
    CHECK(copied.getRotationMode() == inspirecv::ROTATION_0);
    CHECK(copied.GetPreviewScale() == Approx(0.5f));

    inspirecv::FrameProcess moved(std::move(original));
    CHECK(moved.GetWidth() == image.Width());
    CHECK(moved.getRotationMode() == inspirecv::ROTATION_90);

    // A moved-from object must at least remain assignable.
    original = copied;
    CHECK(original.GetWidth() == image.Width());
    CHECK(original.getRotationMode() == inspirecv::ROTATION_0);

    inspirecv::FrameProcess move_assigned;
    move_assigned = std::move(moved);
    CHECK(move_assigned.GetHeight() == image.Height());
    CHECK(move_assigned.getRotationMode() == inspirecv::ROTATION_90);
}

TEST_CASE("C++ FrameProcess rotations and scales produce coherent dimensions", "[cpp_api][contract][frame_process][consistency]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/r0.jpg"));
    REQUIRE(!image.Empty());

    const std::array<inspirecv::ROTATION_MODE, 4> rotations = {
      {inspirecv::ROTATION_0, inspirecv::ROTATION_90, inspirecv::ROTATION_180, inspirecv::ROTATION_270}};
    for (const auto rotation : rotations) {
        auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, rotation);
        const auto full = process.ExecuteImageScaleProcessing(1.0f, true);
        REQUIRE(!full.Empty());
        if (rotation == inspirecv::ROTATION_90 || rotation == inspirecv::ROTATION_270) {
            CHECK(full.Width() == image.Height());
            CHECK(full.Height() == image.Width());
        } else {
            CHECK(full.Width() == image.Width());
            CHECK(full.Height() == image.Height());
        }

        const auto half = process.ExecuteImageScaleProcessing(0.5f, true);
        REQUIRE(!half.Empty());
        CHECK(std::abs(half.Width() * 2 - full.Width()) <= 1);
        CHECK(std::abs(half.Height() * 2 - full.Height()) <= 1);
    }
}

TEST_CASE("C++ FrameProcess affine output honors the requested size", "[cpp_api][contract][frame_process]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    auto matrix = inspirecv::TransformMatrix::Create();
    const auto output = process.ExecuteImageAffineProcessing(matrix, 112, 96);
    REQUIRE(!output.Empty());
    CHECK(output.Width() == 112);
    CHECK(output.Height() == 96);
    CHECK(output.Channels() == 3);
}

TEST_CASE("C++ FrameProcess raw buffers and mutable preprocessing settings remain usable", "[cpp_api][contract][frame_process]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/r0.jpg"));
    REQUIRE(!image.Empty());

    auto process = inspirecv::FrameProcess::Create(
      image.Data(), image.Height(), image.Width(), inspirecv::BGR, inspirecv::ROTATION_0);
    CHECK(process.GetWidth() == image.Width());
    CHECK(process.GetHeight() == image.Height());

    process.SetPreviewSize(160);
    CHECK(process.GetPreviewScale() == Approx(160.0f / std::max(image.Width(), image.Height())));
    auto preview = process.ExecutePreviewImageProcessing(false);
    REQUIRE(!preview.Empty());
    CHECK(std::max(preview.Width(), preview.Height()) <= 160);

    process.SetDataBuffer(image.Data(), image.Height(), image.Width());
    process.SetDataFormat(inspirecv::BGR);
    process.SetDestFormat(inspirecv::RGB);
    process.SetRotationMode(inspirecv::ROTATION_90);
    const auto rotated = process.ExecutePreviewImageProcessing(true);
    REQUIRE(!rotated.Empty());
    CHECK(rotated.Width() == static_cast<int>(image.Height() * process.GetPreviewScale()));
    CHECK(rotated.Height() == static_cast<int>(image.Width() * process.GetPreviewScale()));

    const auto affine = process.GetAffineMatrix();
    const auto rotation_affine = process.GetRotationModeAffineMatrix();
    for (size_t index = 0; index < 6; ++index) {
        CHECK(std::isfinite(affine[index]));
        CHECK(std::isfinite(rotation_affine[index]));
    }
}

TEST_CASE("C++ FrameProcess rejects incomplete and arithmetically unsafe inputs", "[cpp_api][contract][frame_process][boundary]") {
    inspirecv::FrameProcess empty;
    CHECK(empty.GetWidth() == 0);
    CHECK(empty.GetHeight() == 0);
    CHECK(empty.ExecutePreviewImageProcessing(false).Empty());
    CHECK(empty.ExecuteImageScaleProcessing(1.0f, true).Empty());

    auto identity = inspirecv::TransformMatrix::Create();
    CHECK(empty.ExecuteImageAffineProcessing(identity, 8, 8).Empty());
    const auto empty_affine = empty.GetAffineMatrix();
    const auto empty_rotation = empty.GetRotationModeAffineMatrix();
    for (size_t index = 0; index < 6; ++index) {
        CHECK(std::isfinite(empty_affine[index]));
        CHECK(std::isfinite(empty_rotation[index]));
    }

    uint8_t pixel = 0;
    auto oversized = inspirecv::FrameProcess::Create(
      &pixel, std::numeric_limits<int>::max(), 2, inspirecv::GRAY, inspirecv::ROTATION_0);
    CHECK(oversized.ExecuteImageScaleProcessing(1.0f, false).Empty());

    std::vector<uint8_t> odd_yuv(7 * 8 * 3 / 2, 128);
    auto odd_planar = inspirecv::FrameProcess::Create(
      odd_yuv.data(), 8, 7, inspirecv::NV21, inspirecv::ROTATION_0);
    CHECK(odd_planar.ExecuteImageScaleProcessing(1.0f, false).Empty());
}

TEST_CASE("C++ FrameProcess invalid settings preserve the last valid configuration", "[cpp_api][contract][frame_process][boundary]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/r0.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_90);
    process.SetPreviewScale(0.5f);

    const float valid_scale = process.GetPreviewScale();
    const auto valid_rotation = process.getRotationMode();
    process.SetPreviewScale(0.0f);
    process.SetPreviewScale(-1.0f);
    process.SetPreviewScale(std::numeric_limits<float>::quiet_NaN());
    process.SetPreviewScale(std::numeric_limits<float>::infinity());
    process.SetPreviewSize(0);
    process.SetPreviewSize(-1);
    process.SetRotationMode(static_cast<inspirecv::ROTATION_MODE>(99));
    process.SetDataFormat(static_cast<inspirecv::DATA_FORMAT>(99));
    process.SetDestFormat(static_cast<inspirecv::DATA_FORMAT>(99));
    CHECK(process.GetPreviewScale() == Approx(valid_scale));
    CHECK(process.getRotationMode() == valid_rotation);

    const auto after_invalid_settings = process.ExecutePreviewImageProcessing(true);
    REQUIRE(!after_invalid_settings.Empty());
    CHECK(after_invalid_settings.Channels() == 3);

    CHECK(process.ExecuteImageScaleProcessing(std::numeric_limits<float>::denorm_min(), false).Empty());
    CHECK(process.ExecuteImageScaleProcessing(std::numeric_limits<float>::max(), false).Empty());
    CHECK(process.ExecuteImageScaleProcessing(std::numeric_limits<float>::quiet_NaN(), false).Empty());
    CHECK(process.ExecuteImageScaleProcessing(-1.0f, false).Empty());

    process.SetDataBuffer(nullptr, image.Height(), image.Width());
    CHECK(process.GetWidth() == 0);
    CHECK(process.GetHeight() == 0);
    CHECK(process.ExecutePreviewImageProcessing(true).Empty());
}

TEST_CASE("C++ FrameProcess rejects singular transforms and honors packed destination formats",
          "[cpp_api][contract][frame_process][boundary]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);

    auto singular = inspirecv::TransformMatrix::Create();
    for (size_t index = 0; index < 6; ++index) {
        singular[index] = 0.0f;
    }
    CHECK(process.ExecuteImageAffineProcessing(singular, 32, 32).Empty());

    auto non_finite = inspirecv::TransformMatrix::Create();
    non_finite[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(process.ExecuteImageAffineProcessing(non_finite, 32, 32).Empty());
    auto identity = inspirecv::TransformMatrix::Create();
    CHECK(process.ExecuteImageAffineProcessing(identity, 0, 32).Empty());
    CHECK(process.ExecuteImageAffineProcessing(identity, std::numeric_limits<int>::max(), 2).Empty());

    process.SetDestFormat(inspirecv::GRAY);
    const auto gray = process.ExecuteImageScaleProcessing(0.25f, false);
    REQUIRE(!gray.Empty());
    CHECK(gray.Channels() == 1);

    process.SetDestFormat(inspirecv::RGBA);
    const auto rgba = process.ExecuteImageScaleProcessing(0.25f, false);
    REQUIRE(!rgba.Empty());
    CHECK(rgba.Channels() == 4);

    process.SetDestFormat(inspirecv::NV21);
    CHECK(process.ExecuteImageScaleProcessing(0.25f, false).Empty());
}

TEST_CASE("C++ FrameProcess expands RGB and BGR with correct color order and opaque alpha",
          "[cpp_api][contract][frame_process][consistency]") {
    const int width = 4;
    const int height = 2;
    const std::array<uint8_t, width * height * 3> pixels = {{
      7, 31, 211, 43, 101, 239, 67, 127, 197, 83, 151, 181,
      11, 37, 223, 47, 103, 241, 71, 131, 199, 89, 157, 191}};
    for (const auto source : {inspirecv::RGB, inspirecv::BGR}) {
        auto process = inspirecv::FrameProcess::Create(pixels.data(), height, width, source);
        // Switch formats on the same owner and reuse each converter repeatedly.
        for (const auto destination : {inspirecv::RGBA, inspirecv::BGRA, inspirecv::RGBA}) {
            INFO("source=" << source << ", destination=" << destination);
            process.SetDestFormat(destination);
            const bool reverse = (source == inspirecv::RGB && destination == inspirecv::BGRA) ||
                                 (source == inspirecv::BGR && destination == inspirecv::RGBA);
            auto identity = inspirecv::TransformMatrix::Create();
            const std::array<inspirecv::Image, 3> outputs = {{
              process.ExecuteImageScaleProcessing(1.0f, false),
              process.ExecuteImageScaleProcessing(1.0f, false),
              process.ExecuteImageAffineProcessing(identity, width, height)}};
            for (const auto& output : outputs) {
                REQUIRE(!output.Empty());
                REQUIRE(output.Width() == width);
                REQUIRE(output.Height() == height);
                REQUIRE(output.Channels() == 4);
                for (size_t index = 0; index < width * height; ++index) {
                    CHECK(output.Data()[index * 4] == pixels[index * 3 + (reverse ? 2 : 0)]);
                    CHECK(output.Data()[index * 4 + 1] == pixels[index * 3 + 1]);
                    CHECK(output.Data()[index * 4 + 2] == pixels[index * 3 + (reverse ? 0 : 2)]);
                    CHECK(output.Data()[index * 4 + 3] == 255);
                }
            }
        }
    }
}
