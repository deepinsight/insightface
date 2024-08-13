//
// Created by tunm on 2023/10/12.
//

#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "../test_helper/test_tools.h"

TEST_CASE("test_FacePipelineAttribute", "[face_pipeline_attribute]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    enum AGE_BRACKED {
        AGE_0_2 = 0,   ///< Age 0-2 years old
        AGE_3_9,       ///< Age 3-9 years old
        AGE_10_19,     ///< Age 10-19 years old
        AGE_20_29,     ///< Age 20-29 years old
        AGE_30_39,     ///< Age 30-39 years old
        AGE_40_49,     ///< Age 40-49 years old
        AGE_50_59,     ///< Age 50-59 years old
        AGE_60_69,     ///< Age 60-69 years old
        MORE_THAN_70,  ///< Age more than 70 years old
    };
    enum GENDER {
        FEMALE = 0,  ///< Female
        MALE,        ///< Male
    };
    enum RACE {
        BLACK = 0,        ///< Black
        ASIAN,            ///< Asian
        LATINO_HISPANIC,  ///< Latino/Hispanic
        MIDDLE_EASTERN,   ///< Middle Eastern
        WHITE,            ///< White
    };

    HResult ret;
    HFSessionCustomParameter parameter = {0};
    parameter.enable_face_attribute = 1;
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    HFSession session;
    HInt32 faceDetectPixelLevel = 160;
    ret = HFCreateInspireFaceSession(parameter, detMode, 5, faceDetectPixelLevel, -1, &session);
    REQUIRE(ret == HSUCCEED);

    SECTION("a black girl") {
        HFImageStream imgHandle;
        auto img = cv::imread(GET_DATA("data/attribute/1423.jpg"));
        REQUIRE(!img.empty());
        ret = CVImageToImageStream(img, imgHandle);
        REQUIRE(ret == HSUCCEED);

        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);

        // Run pipeline
        ret = HFMultipleFacePipelineProcessOptional(session, imgHandle, &multipleFaceData,
                                                    HF_ENABLE_FACE_ATTRIBUTE);
        REQUIRE(ret == HSUCCEED);

        HFFaceAttributeResult result = {0};
        ret = HFGetFaceAttributeResult(session, &result);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(result.num == 1);

        // Check attribute
        CHECK(result.race[0] == BLACK);
        CHECK(result.ageBracket[0] == AGE_10_19);
        CHECK(result.gender[0] == FEMALE);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);
        imgHandle = nullptr;
    }

    SECTION("two young white women") {
        HFImageStream imgHandle;
        auto img = cv::imread(GET_DATA("data/attribute/7242.jpg"));
        REQUIRE(!img.empty());
        ret = CVImageToImageStream(img, imgHandle);
        REQUIRE(ret == HSUCCEED);

        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 2);

        // Run pipeline
        ret = HFMultipleFacePipelineProcessOptional(session, imgHandle, &multipleFaceData,
                                                    HF_ENABLE_FACE_ATTRIBUTE);
        REQUIRE(ret == HSUCCEED);

        HFFaceAttributeResult result = {0};
        ret = HFGetFaceAttributeResult(session, &result);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(result.num == 2);

        // Check attribute
        for (size_t i = 0; i < result.num; i++) {
            CHECK(result.race[i] == WHITE);
            CHECK(result.ageBracket[i] == AGE_20_29);
            CHECK(result.gender[i] == FEMALE);
        }

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);
        imgHandle = nullptr;
    }

    ret = HFReleaseInspireFaceSession(session);
    session = nullptr;
    REQUIRE(ret == HSUCCEED);
}

TEST_CASE("test_FacePipelineRobustness", "[robustness]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Exception") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Input exception data
        HFImageStream nullHandle = {0};
        HFMultipleFaceData nullfaces = {0};
        ret =
          HFMultipleFacePipelineProcessOptional(session, nullHandle, &nullfaces, HF_ENABLE_NONE);
        REQUIRE(ret == HERR_INVALID_IMAGE_STREAM_HANDLE);

        // Get a face picture
        HFImageStream img1Handle;
        auto img1 = cv::imread(GET_DATA("data/bulk/image_T1.jpeg"));
        ret = CVImageToImageStream(img1, img1Handle);
        REQUIRE(ret == HSUCCEED);

        // Input correct Image and exception faces struct
        ret =
          HFMultipleFacePipelineProcessOptional(session, img1Handle, &nullfaces, HF_ENABLE_NONE);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageStream(img1Handle);
        REQUIRE(ret == HSUCCEED);
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);

        // Multiple release
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HERR_INVALID_CONTEXT_HANDLE);

        HFDeBugShowResourceStatistics();
    }
}

TEST_CASE("test_FacePipeline", "[face_pipeline]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("rgb liveness detect") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_liveness = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream img1Handle;
        auto img1 = cv::imread(GET_DATA("data/bulk/image_T1.jpeg"));
        ret = CVImageToImageStream(img1, img1Handle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, img1Handle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        ret = HFMultipleFacePipelineProcess(session, img1Handle, &multipleFaceData, parameter);
        REQUIRE(ret == HSUCCEED);
        HFRGBLivenessConfidence confidence;
        ret = HFGetRGBLivenessConfidence(session, &confidence);
        TEST_PRINT("{}", confidence.confidence[0]);
        REQUIRE(ret == HSUCCEED);
        CHECK(confidence.num > 0);
        CHECK(confidence.confidence[0] > 0.9);

        ret = HFReleaseImageStream(img1Handle);
        REQUIRE(ret == HSUCCEED);
        img1Handle = nullptr;

        // fake face
        HFImageStream img2Handle;
        auto img2 = cv::imread(GET_DATA("data/bulk/rgb_fake.jpg"));
        ret = CVImageToImageStream(img2, img2Handle);
        REQUIRE(ret == HSUCCEED);
        ret = HFExecuteFaceTrack(session, img2Handle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        ret = HFMultipleFacePipelineProcess(session, img2Handle, &multipleFaceData, parameter);
        REQUIRE(ret == HSUCCEED);
        ret = HFGetRGBLivenessConfidence(session, &confidence);
        REQUIRE(ret == HSUCCEED);
        CHECK(confidence.num > 0);
        CHECK(confidence.confidence[0] < 0.9);

        ret = HFReleaseImageStream(img2Handle);
        REQUIRE(ret == HSUCCEED);
        img2Handle = nullptr;

        ret = HFReleaseInspireFaceSession(session);
        session = nullptr;
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("face mask detect") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_mask_detect = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream img1Handle;
        auto img1 = cv::imread(GET_DATA("data/bulk/mask2.jpg"));
        ret = CVImageToImageStream(img1, img1Handle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, img1Handle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        ret = HFMultipleFacePipelineProcess(session, img1Handle, &multipleFaceData, parameter);
        REQUIRE(ret == HSUCCEED);
        HFFaceMaskConfidence confidence;
        ret = HFGetFaceMaskConfidence(session, &confidence);
        REQUIRE(ret == HSUCCEED);
        CHECK(confidence.num > 0);
        CHECK(confidence.confidence[0] > 0.9);

        ret = HFReleaseImageStream(img1Handle);
        REQUIRE(ret == HSUCCEED);
        img1Handle = nullptr;

        // no mask face
        HFImageStream img2Handle;
        auto img2 = cv::imread(GET_DATA("data/bulk/face_sample.png"));
        ret = CVImageToImageStream(img2, img2Handle);
        REQUIRE(ret == HSUCCEED);
        ret = HFExecuteFaceTrack(session, img2Handle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        ret = HFMultipleFacePipelineProcess(session, img2Handle, &multipleFaceData, parameter);
        REQUIRE(ret == HSUCCEED);
        ret = HFGetFaceMaskConfidence(session, &confidence);
        REQUIRE(ret == HSUCCEED);
        //        spdlog::info("mask {}", confidence.confidence[0]);
        CHECK(confidence.num > 0);
        CHECK(confidence.confidence[0] < 0.1);

        ret = HFReleaseImageStream(img2Handle);
        REQUIRE(ret == HSUCCEED);
        img2Handle = nullptr;

        ret = HFReleaseInspireFaceSession(session);
        session = nullptr;
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("face quality") {
        HResult ret;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HInt32 option = HF_ENABLE_QUALITY;
        HFSession session;
        ret = HFCreateInspireFaceSessionOptional(option, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream superiorHandle;
        auto superior = cv::imread(GET_DATA("data/bulk/yifei.jpg"));
        ret = CVImageToImageStream(superior, superiorHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, superiorHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        ret =
          HFMultipleFacePipelineProcessOptional(session, superiorHandle, &multipleFaceData, option);
        REQUIRE(ret == HSUCCEED);

        HFloat quality;
        ret = HFFaceQualityDetect(session, multipleFaceData.tokens[0], &quality);
        REQUIRE(ret == HSUCCEED);
        CHECK(quality > 0.85);

        // blur image
        HFImageStream blurHandle;
        auto blur = cv::imread(GET_DATA("data/bulk/blur.jpg"));
        ret = CVImageToImageStream(blur, blurHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        ret = HFExecuteFaceTrack(session, blurHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        ret = HFMultipleFacePipelineProcessOptional(session, blurHandle, &multipleFaceData, option);
        REQUIRE(ret == HSUCCEED);

        ret = HFFaceQualityDetect(session, multipleFaceData.tokens[0], &quality);
        REQUIRE(ret == HSUCCEED);
        CHECK(quality < 0.85);

        ret = HFReleaseImageStream(superiorHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageStream(blurHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }
}

TEST_CASE("test_FaceReaction", "[face_reaction]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    HResult ret;
    HFSessionCustomParameter parameter = {0};
    parameter.enable_interaction_liveness = 1;
    parameter.enable_liveness = 1;
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    HFSession session;
    ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
    REQUIRE(ret == HSUCCEED);

    SECTION("open eyes") {
        // Get a face picture
        HFImageStream imgHandle;
        auto img = cv::imread(GET_DATA("data/reaction/open_eyes.png"));
        ret = CVImageToImageStream(img, imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        // Predict eyes status
        ret = HFMultipleFacePipelineProcess(session, imgHandle, &multipleFaceData, parameter);
        REQUIRE(ret == HSUCCEED);

        // Get results
        HFFaceIntereactionState result;
        ret = HFGetFaceIntereactionStateResult(session, &result);
        REQUIRE(multipleFaceData.detectedNum == result.num);
        REQUIRE(ret == HSUCCEED);

        // Check
        CHECK(result.leftEyeStatusConfidence[0] > 0.5f);
        CHECK(result.rightEyeStatusConfidence[0] > 0.5f);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("close eyes") {
        // Get a face picture
        HFImageStream imgHandle;
        auto img = cv::imread(GET_DATA("data/reaction/close_eyes.jpeg"));
        ret = CVImageToImageStream(img, imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        // Predict eyes status
        ret = HFMultipleFacePipelineProcess(session, imgHandle, &multipleFaceData, parameter);
        REQUIRE(ret == HSUCCEED);

        // Get results
        HFFaceIntereactionState result;
        ret = HFGetFaceIntereactionStateResult(session, &result);
        REQUIRE(multipleFaceData.detectedNum == result.num);
        REQUIRE(ret == HSUCCEED);

        // Check
        CHECK(result.leftEyeStatusConfidence[0] < 0.5f);
        CHECK(result.rightEyeStatusConfidence[0] < 0.5f);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Close one eye and open the other") {
        // Get a face picture
        HFImageStream imgHandle;
        auto img = cv::imread(GET_DATA("data/reaction/close_open_eyes.jpeg"));
        ret = CVImageToImageStream(img, imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        // Predict eyes status
        ret = HFMultipleFacePipelineProcess(session, imgHandle, &multipleFaceData, parameter);
        REQUIRE(ret == HSUCCEED);

        // Get results
        HFFaceIntereactionState result;
        ret = HFGetFaceIntereactionStateResult(session, &result);
        REQUIRE(multipleFaceData.detectedNum == result.num);
        REQUIRE(ret == HSUCCEED);

        // Check
        CHECK(result.leftEyeStatusConfidence[0] < 0.5f);
        CHECK(result.rightEyeStatusConfidence[0] > 0.5f);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);
    }

    ret = HFReleaseInspireFaceSession(session);
    REQUIRE(ret == HSUCCEED);
}