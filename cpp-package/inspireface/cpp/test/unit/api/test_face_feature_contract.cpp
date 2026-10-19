#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/test_tools.h"

using inspireface_test::UniqueFaceFeature;
using inspireface_test::UniqueImageBitmap;
using inspireface_test::UniqueImageStream;
using inspireface_test::UniqueSession;

namespace {

struct FaceInput {
    // HFImageStream is a non-owning view of the caller's pixels. Keep the
    // image first so it is destroyed after the stream and session.
    inspirecv::Image image;
    UniqueSession session;
    UniqueImageStream stream;
    HFMultipleFaceData faces = {};
};

FaceInput CreateFaceInput(HOption option) {
    FaceInput input;
    HFSession session = nullptr;
    REQUIRE(HFCreateInspireFaceSessionOptional(option, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &session) == HSUCCEED);
    input.session = UniqueSession(session);

    input.image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!input.image.Empty());
    HFImageStream stream = nullptr;
    REQUIRE(CVImageToImageStream(input.image, stream) == HSUCCEED);
    input.stream = UniqueImageStream(stream);

    REQUIRE(HFExecuteFaceTrack(input.session.Get(), input.stream.Get(), &input.faces) == HSUCCEED);
    REQUIRE(input.faces.detectedNum == 1);
    return input;
}

}  // namespace

TEST_CASE("C API face feature extraction variants agree", "[api][contract][face_feature][consistency]") {
    auto input = CreateFaceInput(HF_ENABLE_FACE_RECOGNITION);

    HInt32 feature_length = 0;
    REQUIRE(HFGetFeatureLength(&feature_length) == HSUCCEED);
    REQUIRE(feature_length > 0);
    CHECK(HFGetFeatureLength(nullptr) == HERR_INVALID_PARAM);

    HFFaceFeature cached = {0, nullptr};
    REQUIRE(HFFaceFeatureExtract(input.session.Get(), input.stream.Get(), input.faces.tokens[0], &cached) == HSUCCEED);
    REQUIRE(cached.size == feature_length);
    REQUIRE(cached.data != nullptr);
    const std::vector<float> cached_copy(cached.data, cached.data + cached.size);

    UniqueFaceFeature owned;
    REQUIRE(HFCreateFaceFeature(owned.Put()) == HSUCCEED);
    REQUIRE(owned.Get().size == feature_length);
    REQUIRE(owned.Get().data != nullptr);
    REQUIRE(HFFaceFeatureExtractTo(input.session.Get(), input.stream.Get(), input.faces.tokens[0], owned.Get()) == HSUCCEED);

    std::vector<float> copied(static_cast<size_t>(feature_length));
    REQUIRE(HFFaceFeatureExtractCpy(input.session.Get(), input.stream.Get(), input.faces.tokens[0], copied.data()) == HSUCCEED);

    HFFaceFeature cached_snapshot = {feature_length, const_cast<float*>(cached_copy.data())};
    HFFaceFeature copied_view = {feature_length, copied.data()};
    float similarity = 0.0f;
    REQUIRE(HFFaceComparison(cached_snapshot, owned.Get(), &similarity) == HSUCCEED);
    CHECK(similarity == Approx(1.0f).margin(1e-5f));
    REQUIRE(HFFaceComparison(cached_snapshot, copied_view, &similarity) == HSUCCEED);
    CHECK(similarity == Approx(1.0f).margin(1e-5f));

    UniqueImageBitmap alignment;
    REQUIRE(HFFaceGetFaceAlignmentImage(input.session.Get(), input.stream.Get(), input.faces.tokens[0], alignment.Put()) == HSUCCEED);
    HFImageBitmapData alignment_data = {nullptr, 0, 0, 0};
    REQUIRE(HFImageBitmapGetData(alignment.Get(), &alignment_data) == HSUCCEED);
    CHECK(alignment_data.width > 0);
    CHECK(alignment_data.height > 0);
    CHECK(alignment_data.width == alignment_data.height);
    CHECK(alignment_data.channels == 3);

    UniqueImageStream alignment_stream;
    REQUIRE(HFCreateImageStreamFromImageBitmap(alignment.Get(), HF_CAMERA_ROTATION_0, alignment_stream.Put()) == HSUCCEED);
    REQUIRE(HFFaceFeatureExtractWithAlignmentImage(input.session.Get(), alignment_stream.Get(), owned.Get()) == HSUCCEED);
    REQUIRE(HFFaceComparison(cached_snapshot, owned.Get(), &similarity) == HSUCCEED);
    CHECK(similarity == Approx(1.0f).margin(1e-4f));
}

TEST_CASE("C API face feature functions reject disabled modules and malformed buffers", "[api][contract][face_feature][boundary]") {
    auto input = CreateFaceInput(HF_ENABLE_NONE);
    HFFaceFeature output = {123, reinterpret_cast<float*>(static_cast<uintptr_t>(1))};

    CHECK(HFFaceFeatureExtract(input.session.Get(), input.stream.Get(), input.faces.tokens[0], &output) == HERR_SESS_REC_EXTRACT_FAILURE);
    CHECK(output.size == 0);
    CHECK(output.data == nullptr);
    CHECK(HFFaceFeatureExtract(input.session.Get(), input.stream.Get(), input.faces.tokens[0], nullptr) == HERR_INVALID_FACE_FEATURE);

    HFFaceBasicToken invalid_token = {0, nullptr};
    CHECK(HFFaceFeatureExtract(input.session.Get(), input.stream.Get(), invalid_token, &output) == HERR_INVALID_FACE_TOKEN);

    std::vector<float> too_small(4);
    HFFaceFeature small = {static_cast<int>(too_small.size()), too_small.data()};
    CHECK(HFFaceFeatureExtractTo(input.session.Get(), input.stream.Get(), input.faces.tokens[0], small) == HERR_INVALID_FACE_FEATURE);
    CHECK(HFFaceFeatureExtractCpy(input.session.Get(), input.stream.Get(), input.faces.tokens[0], nullptr) == HERR_INVALID_FACE_FEATURE);
    CHECK(HFFaceFeatureExtractWithAlignmentImage(input.session.Get(), input.stream.Get(), small) == HERR_INVALID_FACE_FEATURE);

    HFImageBitmap alignment = reinterpret_cast<HFImageBitmap>(static_cast<uintptr_t>(1));
    CHECK(HFFaceGetFaceAlignmentImage(input.session.Get(), input.stream.Get(), invalid_token, &alignment) == HERR_INVALID_FACE_TOKEN);
    CHECK(alignment == nullptr);
    CHECK(HFFaceGetFaceAlignmentImage(input.session.Get(), input.stream.Get(), input.faces.tokens[0], nullptr) == HERR_INVALID_IMAGE_BITMAP_HANDLE);

    HFFaceFeature empty = {0, nullptr};
    float similarity = 7.0f;
    CHECK(HFFaceComparison(empty, empty, &similarity) == HERR_INVALID_FACE_FEATURE);
    CHECK(HFFaceComparison(small, small, &similarity) == HERR_INVALID_FACE_FEATURE);
    CHECK(HFFaceComparison(empty, empty, nullptr) == HERR_INVALID_PARAM);
}

TEST_CASE("C API owned face feature has deterministic release semantics", "[api][contract][face_feature][resource]") {
    CHECK(HFCreateFaceFeature(nullptr) == HERR_INVALID_FACE_FEATURE);
    CHECK(HFReleaseFaceFeature(nullptr) == HERR_INVALID_FACE_FEATURE);

    HFFaceFeature feature = {0, nullptr};
    REQUIRE(HFCreateFaceFeature(&feature) == HSUCCEED);
    REQUIRE(feature.size > 0);
    REQUIRE(feature.data != nullptr);
    CHECK(HFCreateFaceFeature(&feature) == HERR_INVALID_FACE_FEATURE);
    REQUIRE(HFReleaseFaceFeature(&feature) == HSUCCEED);
    CHECK(feature.size == 0);
    CHECK(feature.data == nullptr);
    CHECK(HFReleaseFaceFeature(&feature) == HERR_INVALID_FACE_FEATURE);
}
