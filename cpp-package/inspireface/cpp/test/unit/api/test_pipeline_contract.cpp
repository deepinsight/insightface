#include <cmath>
#include <cstdint>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"
#include "unit/test_helper/test_tools.h"

using inspireface_test::UniqueImageStream;
using inspireface_test::UniqueSession;

namespace {

constexpr HOption kPipelineOptions = HF_ENABLE_LIVENESS | HF_ENABLE_MASK_DETECT | HF_ENABLE_QUALITY |
                                     HF_ENABLE_FACE_ATTRIBUTE | HF_ENABLE_INTERACTION | HF_ENABLE_FACE_EMOTION;

struct PipelineInput {
    inspirecv::Image image;
    UniqueSession session;
    UniqueImageStream stream;
    HFMultipleFaceData faces = {};
};

PipelineInput CreatePipelineInput() {
    PipelineInput input;
    HFSession session = nullptr;
    REQUIRE(HFCreateInspireFaceSessionOptional(kPipelineOptions, HF_DETECT_MODE_ALWAYS_DETECT, 3, -1, -1, &session) == HSUCCEED);
    input.session = UniqueSession(session);

    input.image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!input.image.Empty());
    HFImageStream stream = nullptr;
    REQUIRE(CVImageToImageStream(input.image, stream) == HSUCCEED);
    input.stream = UniqueImageStream(stream);

    REQUIRE(HFExecuteFaceTrack(input.session.Get(), input.stream.Get(), &input.faces) == HSUCCEED);
    REQUIRE(input.faces.detectedNum > 0);
    return input;
}

void CheckEmptyPipelineOutputs(HFSession session, HInt32 expected_quality_count) {
    HFRGBLivenessConfidence liveness = {};
    HFFaceMaskConfidence mask = {};
    HFFaceQualityConfidence quality = {};
    HFFaceInteractionState state = {};
    HFFaceInteractionsActions actions = {};
    HFFaceAttributeResult attributes = {};
    HFFaceEmotionResult emotion = {};
    REQUIRE(HFGetRGBLivenessConfidence(session, &liveness) == HSUCCEED);
    REQUIRE(HFGetFaceMaskConfidence(session, &mask) == HSUCCEED);
    REQUIRE(HFGetFaceQualityConfidence(session, &quality) == HSUCCEED);
    REQUIRE(HFGetFaceInteractionStateResult(session, &state) == HSUCCEED);
    REQUIRE(HFGetFaceInteractionActionsResult(session, &actions) == HSUCCEED);
    REQUIRE(HFGetFaceAttributeResult(session, &attributes) == HSUCCEED);
    REQUIRE(HFGetFaceEmotionResult(session, &emotion) == HSUCCEED);
    CHECK(liveness.num == 0);
    CHECK(mask.num == 0);
    // Face quality is produced by tracking, not by the optional pipeline, so an
    // empty pipeline request must not erase the most recent detection quality.
    CHECK(quality.num == expected_quality_count);
    CHECK(state.num == 0);
    CHECK(actions.num == 0);
    CHECK(attributes.num == 0);
    CHECK(emotion.num == 0);
}

}  // namespace

TEST_CASE("C API pipeline output counts follow the processed face list", "[api][contract][pipeline][consistency]") {
    auto input = CreatePipelineInput();
    CheckEmptyPipelineOutputs(input.session.Get(), input.faces.detectedNum);

    REQUIRE(HFMultipleFacePipelineProcessOptional(input.session.Get(), input.stream.Get(), &input.faces, kPipelineOptions) == HSUCCEED);

    HFRGBLivenessConfidence liveness = {};
    HFFaceMaskConfidence mask = {};
    HFFaceQualityConfidence quality = {};
    HFFaceInteractionState state = {};
    HFFaceInteractionsActions actions = {};
    HFFaceAttributeResult attributes = {};
    HFFaceEmotionResult emotion = {};
    REQUIRE(HFGetRGBLivenessConfidence(input.session.Get(), &liveness) == HSUCCEED);
    REQUIRE(HFGetFaceMaskConfidence(input.session.Get(), &mask) == HSUCCEED);
    REQUIRE(HFGetFaceQualityConfidence(input.session.Get(), &quality) == HSUCCEED);
    REQUIRE(HFGetFaceInteractionStateResult(input.session.Get(), &state) == HSUCCEED);
    REQUIRE(HFGetFaceInteractionActionsResult(input.session.Get(), &actions) == HSUCCEED);
    REQUIRE(HFGetFaceAttributeResult(input.session.Get(), &attributes) == HSUCCEED);
    REQUIRE(HFGetFaceEmotionResult(input.session.Get(), &emotion) == HSUCCEED);

    CHECK(liveness.num == input.faces.detectedNum);
    CHECK(mask.num == input.faces.detectedNum);
    CHECK(quality.num == input.faces.detectedNum);
    CHECK(state.num == input.faces.detectedNum);
    CHECK(actions.num == input.faces.detectedNum);
    CHECK(attributes.num == input.faces.detectedNum);
    CHECK(emotion.num == input.faces.detectedNum);
    REQUIRE(liveness.confidence != nullptr);
    REQUIRE(mask.confidence != nullptr);
    REQUIRE(quality.confidence != nullptr);
    for (int i = 0; i < input.faces.detectedNum; ++i) {
        CHECK(std::isfinite(liveness.confidence[i]));
        CHECK(std::isfinite(mask.confidence[i]));
        CHECK(std::isfinite(quality.confidence[i]));
    }

    float direct_quality = 0.0f;
    REQUIRE(HFFaceQualityDetect(input.session.Get(), input.faces.tokens[0], &direct_quality) == HSUCCEED);
    CHECK(std::isfinite(direct_quality));

    HFMultipleFaceData empty_faces = {};
    REQUIRE(HFMultipleFacePipelineProcessOptional(input.session.Get(), input.stream.Get(), &empty_faces, kPipelineOptions) == HSUCCEED);
    CheckEmptyPipelineOutputs(input.session.Get(), input.faces.detectedNum);
}

TEST_CASE("C API pipeline rejects null outputs, invalid lists, and malformed tokens", "[api][contract][pipeline][boundary]") {
    auto input = CreatePipelineInput();
    HFSessionCustomParameter parameter = {0};

    CHECK(HFMultipleFacePipelineProcess(input.session.Get(), input.stream.Get(), nullptr, parameter) == HERR_INVALID_FACE_LIST);
    CHECK(HFMultipleFacePipelineProcessOptional(input.session.Get(), input.stream.Get(), nullptr, kPipelineOptions) == HERR_INVALID_FACE_LIST);
    CHECK(HFMultipleFacePipelineProcess(nullptr, input.stream.Get(), &input.faces, parameter) == HERR_INVALID_CONTEXT_HANDLE);
    CHECK(HFMultipleFacePipelineProcess(input.session.Get(), nullptr, &input.faces, parameter) == HERR_INVALID_IMAGE_STREAM_HANDLE);

    HFMultipleFaceData invalid_list = {};
    invalid_list.detectedNum = -1;
    CHECK(HFMultipleFacePipelineProcess(input.session.Get(), input.stream.Get(), &invalid_list, parameter) == HERR_INVALID_FACE_LIST);
    invalid_list.detectedNum = 1;
    CHECK(HFMultipleFacePipelineProcess(input.session.Get(), input.stream.Get(), &invalid_list, parameter) == HERR_INVALID_FACE_LIST);

    HFFaceBasicToken invalid_token = {0, nullptr};
    invalid_list.tokens = &invalid_token;
    CHECK(HFMultipleFacePipelineProcess(input.session.Get(), input.stream.Get(), &invalid_list, parameter) == HERR_INVALID_FACE_TOKEN);

    CHECK(HFGetRGBLivenessConfidence(input.session.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFGetFaceMaskConfidence(input.session.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFGetFaceQualityConfidence(input.session.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFGetFaceInteractionStateResult(input.session.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFGetFaceInteractionActionsResult(input.session.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFGetFaceAttributeResult(input.session.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFGetFaceEmotionResult(input.session.Get(), nullptr) == HERR_INVALID_PARAM);
    CHECK(HFFaceQualityDetect(input.session.Get(), invalid_token, nullptr) == HERR_INVALID_FACE_TOKEN);
}
