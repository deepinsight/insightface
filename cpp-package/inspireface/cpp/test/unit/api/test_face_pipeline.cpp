//
// Created by tunm on 2023/10/12.
//

#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "../test_helper/test_tools.h"

TEST_CASE("test_FacePipeline", "[face_pipeline]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("rgb liveness detect") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_liveness = 1;
        HFDetectMode detMode = HF_DETECT_MODE_IMAGE;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream img1Handle;
        auto img1 = cv::imread(GET_DATA("images/image_T1.jpeg"));
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
        auto img2 = cv::imread(GET_DATA("images/rgb_fake.jpg"));
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
        HFDetectMode detMode = HF_DETECT_MODE_IMAGE;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream img1Handle;
        auto img1 = cv::imread(GET_DATA("images/mask2.jpg"));
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
        auto img2 = cv::imread(GET_DATA("images/face_sample.png"));
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
        HFDetectMode detMode = HF_DETECT_MODE_IMAGE;
        HInt32 option = HF_ENABLE_QUALITY;
        HFSession session;
        ret = HFCreateInspireFaceSessionOptional(option, detMode, 3, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream superiorHandle;
        auto superior = cv::imread(GET_DATA("images/yifei.jpg"));
        ret = CVImageToImageStream(superior, superiorHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, superiorHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum > 0);

        ret = HFMultipleFacePipelineProcessOptional(session, superiorHandle, &multipleFaceData, option);
        REQUIRE(ret == HSUCCEED);

        HFloat quality;
        ret = HFFaceQualityDetect(session, multipleFaceData.tokens[0], &quality);
        REQUIRE(ret == HSUCCEED);
        CHECK(quality > 0.85);

        // blur image
        HFImageStream blurHandle;
        auto blur = cv::imread(GET_DATA("images/blur.jpg"));
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