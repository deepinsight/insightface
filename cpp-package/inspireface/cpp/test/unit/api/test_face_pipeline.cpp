/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "../test_helper/test_tools.h"
#include "../test_helper/test_help.h"

TEST_CASE("test_FaceEmotion", "[face_emotion]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    enum EMOTION {
        NEUTRAL = 0,   ///< Emotion: neutral
        HAPPY = 1,     ///< Emotion: happy
        SAD = 2,       ///< Emotion: sad
        SURPRISE = 3,  ///< Emotion: surprise
        FEAR = 4,      ///< Emotion: fear
        DISGUST = 5,   ///< Emotion: disgust
        ANGER = 6,     ///< Emotion: anger
    };

    HResult ret;
    HFSessionCustomParameter parameter = {0};
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    HFSession session;
    HInt32 faceDetectPixelLevel = 320;
    HInt32 option = HF_ENABLE_FACE_EMOTION;
    ret = HFCreateInspireFaceSessionOptional(option, detMode, 5, faceDetectPixelLevel, -1, &session);
    REQUIRE(ret == HSUCCEED);

    std::vector<std::string> test_images = {
      "data/emotion/anger.png",
      "data/emotion/sad.png",
      "data/emotion/happy.png",
    };

    std::vector<EMOTION> expected_emotions = {
      ANGER,
      SAD,
      HAPPY,
    };
    REQUIRE(test_images.size() == expected_emotions.size());

    for (size_t i = 0; i < test_images.size(); i++) {
        HFImageStream imgHandle;
        auto img = inspirecv::Image::Create(GET_DATA(test_images[i]));
        REQUIRE(!img.Empty());
        ret = CVImageToImageStream(img, imgHandle);
        REQUIRE(ret == HSUCCEED);

        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);

        ret = HFMultipleFacePipelineProcessOptional(session, imgHandle, &multipleFaceData, option);
        REQUIRE(ret == HSUCCEED);

        HFFaceEmotionResult result = {0};
        ret = HFGetFaceEmotionResult(session, &result);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(result.num == 1);
        CHECK(result.emotion[0] == (HInt32)expected_emotions[i]);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);
        imgHandle = nullptr;
    }

    ret = HFReleaseInspireFaceSession(session);
    REQUIRE(ret == HSUCCEED);
    session = nullptr;
}

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
    HInt32 faceDetectPixelLevel = 320;
    ret = HFCreateInspireFaceSession(parameter, detMode, 5, faceDetectPixelLevel, -1, &session);
    REQUIRE(ret == HSUCCEED);

    SECTION("a black girl") {
        HFImageStream imgHandle;
        auto img = inspirecv::Image::Create(GET_DATA("data/attribute/1423.jpg"));
        REQUIRE(!img.Empty());
        ret = CVImageToImageStream(img, imgHandle);
        REQUIRE(ret == HSUCCEED);

        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);

        // Run pipeline
        ret = HFMultipleFacePipelineProcessOptional(session, imgHandle, &multipleFaceData, HF_ENABLE_FACE_ATTRIBUTE);
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
        auto img = inspirecv::Image::Create(GET_DATA("data/attribute/7242.jpg"));
        REQUIRE(!img.Empty());
        ret = CVImageToImageStream(img, imgHandle);
        REQUIRE(ret == HSUCCEED);

        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 2);

        // Run pipeline
        ret = HFMultipleFacePipelineProcessOptional(session, imgHandle, &multipleFaceData, HF_ENABLE_FACE_ATTRIBUTE);
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

TEST_CASE("test_FacePipeline", "[face_pipeline]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("rgb liveness detect") {
#ifndef INFERENCE_WRAPPER_ENABLE_RKNN2
        /** The anti spoofing model based on RGB faces seems to have some problems with quantization under RKNPU2, so it is not started yet */
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_liveness = 1;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, 320, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream img1Handle;
        auto img1 = inspirecv::Image::Create(GET_DATA("data/bulk/image_T1.jpeg"));
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
        // CHECK(confidence.confidence[0] > 0.70f);

        ret = HFReleaseImageStream(img1Handle);
        REQUIRE(ret == HSUCCEED);
        img1Handle = nullptr;

        // fake face
        HFImageStream img2Handle;
        auto img2 = inspirecv::Image::Create(GET_DATA("data/bulk/rgb_fake.jpg"));
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
#else
        TEST_PRINT("The anti spoofing model based on RGB faces seems to have some problems with quantization under RKNPU2, so we skip this test.");
#endif
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
        auto img1 = inspirecv::Image::Create(GET_DATA("data/bulk/mask2.jpg"));
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
        auto img2 = inspirecv::Image::Create(GET_DATA("data/bulk/face_sample.png"));
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
        ret = HFCreateInspireFaceSessionOptional(option, detMode, 3, 320, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream superiorHandle;
        auto superior = inspirecv::Image::Create(GET_DATA("data/bulk/yifei.jpg"));
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
        CHECK(quality > 0.8);

        // blur image
        HFImageStream blurHandle;
        auto blur = inspirecv::Image::Create(GET_DATA("data/bulk/blur.jpg"));
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
        auto img = inspirecv::Image::Create(GET_DATA("data/reaction/open_eyes.png"));
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
        HFFaceInteractionState result;
        ret = HFGetFaceInteractionStateResult(session, &result);
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
        auto img = inspirecv::Image::Create(GET_DATA("data/reaction/close_eyes.jpeg"));
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
        HFFaceInteractionState result;
        ret = HFGetFaceInteractionStateResult(session, &result);
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
        auto img = inspirecv::Image::Create(GET_DATA("data/reaction/close_open_eyes.jpeg"));
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
        HFFaceInteractionState result;
        ret = HFGetFaceInteractionStateResult(session, &result);
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

TEST_CASE("test_TrackModeFaceAction", "[face_action]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    HResult ret;
    HFSessionCustomParameter parameter = {0};
    parameter.enable_interaction_liveness = 1;
    HFDetectMode detMode = HF_DETECT_MODE_LIGHT_TRACK;
    HFSession session;
    ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
    REQUIRE(ret == HSUCCEED);

#if 0  // Temporarily shut down the processing of frame related cases to prevent excessive data from occupying the github memory
    SECTION("Action Blink") {
        auto start = 130, end = 150;
        std::vector<std::string> filenames = generateFilenames("frame-%04d.jpg", start, end);
        int count = 0;
        for (size_t i = 0; i < filenames.size(); i++) {
            auto filename = filenames[i];
            HFImageStream imgHandle;
            auto image = inspirecv::Image::Create(GET_DATA("data/video_frames/" + filename));
            ret = CVImageToImageStream(image, imgHandle);
            REQUIRE(ret == HSUCCEED);

            HFMultipleFaceData multipleFaceData = {0};
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(multipleFaceData.detectedNum > 0);

            ret = HFMultipleFacePipelineProcessOptional(session, imgHandle, &multipleFaceData, HF_ENABLE_INTERACTION);
            REQUIRE(ret == HSUCCEED);

            HFFaceInteractionsActions result;
            ret = HFGetFaceInteractionActionsResult(session, &result);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(multipleFaceData.detectedNum == result.num);

            count += result.blink[0];
            ret = HFReleaseImageStream(imgHandle);
            REQUIRE(ret == HSUCCEED);
        }
        // Blink at least once
        REQUIRE(count > 0);
    }
#endif

#if 0  // Temporarily shut down the processing of frame related cases to prevent excessive data from occupying the github memory
    SECTION("Action Jaw Open") {
        auto start = 110, end = 150;
        std::vector<std::string> filenames = generateFilenames("frame-%04d.jpg", start, end);
        int count = 0;
        for (size_t i = 0; i < filenames.size(); i++) {
            auto filename = filenames[i];
            HFImageStream imgHandle;
            auto image = inspirecv::Image::Create(GET_DATA("data/video_frames/" + filename));
            ret = CVImageToImageStream(image, imgHandle);
            REQUIRE(ret == HSUCCEED);

            HFMultipleFaceData multipleFaceData = {0};
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(multipleFaceData.detectedNum > 0);

            ret = HFMultipleFacePipelineProcessOptional(session, imgHandle, &multipleFaceData, HF_ENABLE_INTERACTION);
            REQUIRE(ret == HSUCCEED);

            HFFaceInteractionsActions result;
            ret = HFGetFaceInteractionActionsResult(session, &result);
            REQUIRE(ret == HSUCCEED);
            REQUIRE(multipleFaceData.detectedNum == result.num);

            count += result.jawOpen[0];
            ret = HFReleaseImageStream(imgHandle);
            REQUIRE(ret == HSUCCEED);
        }
        // Jaw open at least once
        REQUIRE(count > 0);
    }
#endif
    ret = HFReleaseInspireFaceSession(session);
    REQUIRE(ret == HSUCCEED);
}
