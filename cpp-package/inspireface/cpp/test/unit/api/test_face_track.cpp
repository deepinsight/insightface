//
// Created by tunm on 2023/10/11.
//

#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "opencv2/opencv.hpp"
#include "unit/test_helper/simple_csv_writer.h"
#include "unit/test_helper/test_help.h"
#include "unit/test_helper/test_tools.h"


TEST_CASE("test_FaceTrack", "[face_track]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Face detection from image") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_IMAGE;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, &session);
        spdlog::error("error ret :{}", ret);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream imgHandle;
        auto image = cv::imread(GET_DATA("data/bulk/kun.jpg"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);

        // Detect face position
        auto rect = multipleFaceData.rects[0];
        HFaceRect expect = {0};
        expect.x = 98;
        expect.y = 146;
        expect.width = 233 - expect.x;
        expect.height = 272 - expect.y;

        auto iou = CalculateOverlap(rect, expect);
        cv::Rect cvRect(rect.x, rect.y, rect.width, rect.height);
        cv::rectangle(image, cvRect, cv::Scalar(255, 0, 124), 2);
        cv::imwrite("ww.jpg", image);
        // The iou is allowed to have an error of 10%
        CHECK(iou == Approx(1.0f).epsilon(0.1));

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Prepare non-face images
        HFImageStream viewHandle;
        auto view = cv::imread(GET_DATA("data/bulk/view.jpg"));
        ret = CVImageToImageStream(view, viewHandle);
        REQUIRE(ret == HSUCCEED);
        ret = HFExecuteFaceTrack(session, viewHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 0);

        ret = HFReleaseImageStream(viewHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);

    }

    SECTION("Face tracking stability from frames") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_VIDEO;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, &session);
        REQUIRE(ret == HSUCCEED);

        auto expectedId = 1;
        int start = 1, end = 288;
        std::vector<std::string> filenames = generateFilenames("frame-%04d.jpg", start, end);
        auto count_loss = 0;
        for (int i = 0; i < filenames.size(); ++i) {
            auto filename = filenames[i];
            HFImageStream imgHandle;
            auto image = cv::imread(GET_DATA("video_frames/" + filename));
            ret = CVImageToImageStream(image, imgHandle);
            REQUIRE(ret == HSUCCEED);

            HFMultipleFaceData multipleFaceData = {0};
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
            REQUIRE(ret == HSUCCEED);
//            CHECK(multipleFaceData.detectedNum == 1);
            if (multipleFaceData.detectedNum != 1) {
                count_loss++;
                continue;
            }
            auto rect = multipleFaceData.rects[0];
            cv::Rect cvRect(rect.x, rect.y, rect.width, rect.height);
            cv::rectangle(image, cvRect, cv::Scalar(255, 0, 124), 2);
            std::string save = GET_SAVE_DATA("video_frames") + "/" + std::to_string(i) + ".jpg";
            cv::imwrite(save, image);
            auto id = multipleFaceData.trackIds[0];
//            TEST_PRINT("{}", id);
            if (id != expectedId) {
                count_loss++;
            }

            ret = HFReleaseImageStream(imgHandle);
            REQUIRE(ret == HSUCCEED);
        }
        float loss = (float )count_loss / filenames.size();
        // The face track loss is allowed to have an error of 5%
//        CHECK(loss == Approx(0.0f).epsilon(0.05));

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Head pose estimation") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_IMAGE;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, &session);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};

        // Left side face
        HFImageStream leftHandle;
        auto left = cv::imread(GET_DATA("data/pose/left_face.jpeg"));
        ret = CVImageToImageStream(left, leftHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFExecuteFaceTrack(session, leftHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);

        HFloat yaw, pitch, roll;
        bool checked;

        // Left-handed rotation
        yaw = multipleFaceData.angles.yaw[0];
        checked = (yaw > -90 && yaw < -10);
        CHECK(checked);

        HFReleaseImageStream(leftHandle);

        // Right-handed rotation
        HFImageStream rightHandle;
        auto right = cv::imread(GET_DATA("data/pose/right_face.png"));
        ret = CVImageToImageStream(right, rightHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFExecuteFaceTrack(session, rightHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        yaw = multipleFaceData.angles.yaw[0];
        checked = (yaw > 10 && yaw < 90);
        CHECK(checked);

        HFReleaseImageStream(rightHandle);

        // Rise head
        HFImageStream riseHandle;
        auto rise = cv::imread(GET_DATA("data/pose/rise_face.jpeg"));
        ret = CVImageToImageStream(rise, riseHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFExecuteFaceTrack(session, riseHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        pitch = multipleFaceData.angles.pitch[0];
        CHECK(pitch > 5);
        HFReleaseImageStream(riseHandle);

        // Lower head
        HFImageStream lowerHandle;
        auto lower = cv::imread(GET_DATA("data/pose/lower_face.jpeg"));
        ret = CVImageToImageStream(lower, lowerHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFExecuteFaceTrack(session, lowerHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        pitch = multipleFaceData.angles.pitch[0];
        CHECK(pitch < -10);
        HFReleaseImageStream(lowerHandle);

        // Roll head
        HFImageStream leftWryneckHandle;
        auto leftWryneck = cv::imread(GET_DATA("data/pose/left_wryneck.png"));
        ret = CVImageToImageStream(leftWryneck, leftWryneckHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFExecuteFaceTrack(session, leftWryneckHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        roll = multipleFaceData.angles.roll[0];
        CHECK(roll < -30);
        HFReleaseImageStream(leftWryneckHandle);

        // Roll head
        HFImageStream rightWryneckHandle;
        auto rightWryneck = cv::imread(GET_DATA("data/pose/right_wryneck.png"));
        ret = CVImageToImageStream(rightWryneck, rightWryneckHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFExecuteFaceTrack(session, rightWryneckHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        roll = multipleFaceData.angles.roll[0];
        CHECK(roll > 30);
        HFReleaseImageStream(rightWryneckHandle);

        //  finish
        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);

    }

    SECTION("Face detection benchmark") {
#ifdef ISF_ENABLE_BENCHMARK
        int loop = 1000;
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_IMAGE;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, &session);
        REQUIRE(ret == HSUCCEED);

        // Prepare an image
        HFImageStream imgHandle;
        auto image = cv::imread(GET_DATA("data/bulk/kun.jpg"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);
        BenchmarkRecord record(getBenchmarkRecordFile());

        // Case: Execute the benchmark using the IMAGE mode
        ret = HFSessionSetFaceTrackMode(session, HF_DETECT_MODE_IMAGE);
        REQUIRE(ret == HSUCCEED);
        HFMultipleFaceData multipleFaceData = {0};
        auto start = (double) cv::getTickCount();
        for (int i = 0; i < loop; ++i) {
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        }
        auto cost = ((double) cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        TEST_PRINT("<Benchmark> Face Detect -> Loop: {}, Total Time: {:.5f}ms, Average Time: {:.5f}ms", loop, cost, cost / loop);
        record.insertBenchmarkData("Face Detect", loop, cost, cost / loop);

        // Case: Execute the benchmark using the VIDEO mode(Track)
        ret = HFSessionSetFaceTrackMode(session, HF_DETECT_MODE_VIDEO);
        REQUIRE(ret == HSUCCEED);
        multipleFaceData = {0};
        start = (double) cv::getTickCount();
        for (int i = 0; i < loop; ++i) {
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        }
        cost = ((double) cv::getTickCount() - start) / cv::getTickFrequency() * 1000;
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        TEST_PRINT("<Benchmark> Face Track -> Loop: {}, Total Time: {:.5f}ms, Average Time: {:.5f}ms", loop, cost, cost / loop);
        record.insertBenchmarkData("Face Track", loop, cost, cost / loop);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
#else
        TEST_PRINT("Skip the face detection benchmark test. To run it, you need to turn on the benchmark test.");
#endif
    }

}