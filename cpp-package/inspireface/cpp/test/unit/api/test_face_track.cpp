/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include "unit/test_helper/simple_csv_writer.h"
#include "unit/test_helper/test_help.h"
#include "unit/test_helper/test_tools.h"
#include "middleware/costman.h"

TEST_CASE("test_FaceTrack", "[face_track]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Face detection from image") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        TEST_ERROR_PRINT("error ret :{}", ret);
        REQUIRE(ret == HSUCCEED);

        // Get a face picture
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
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
        expect.x = 79;
        expect.y = 104;
        expect.width = 168;
        expect.height = 167;

        auto iou = CalculateOverlap(rect, expect);
        auto cvRect = inspirecv::Rect<int>::Create(rect.x, rect.y, rect.width, rect.height);
        image.DrawRect(cvRect, {0, 0, 255}, 2);
        image.Write("ww.jpg");
        // The iou is allowed to have an error of 50%
        CHECK(iou == Approx(1.0f).epsilon(0.5));

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Prepare non-face images
        HFImageStream viewHandle;
        auto view = inspirecv::Image::Create(GET_DATA("data/bulk/view.jpg"));
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

// Temporarily shut down the processing of frame related cases to prevent excessive data from occupying the github memory
#if 0
    SECTION("Face tracking stability from frames") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_LIGHT_TRACK;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        auto expectedId = 1;
        int start = 1, end = 288;
        std::vector<std::string> filenames = generateFilenames("frame-%04d.jpg", start, end);
        auto count_loss = 0;
        for (int i = 0; i < filenames.size(); ++i) {
            auto filename = filenames[i];
            HFImageStream imgHandle;
            auto image = inspirecv::Image::Create(GET_DATA("data/video_frames/" + filename));
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
            auto cvRect = inspirecv::Rect<int>::Create(rect.x, rect.y, rect.width, rect.height);
            image.DrawRect(cvRect, {0, 0, 255}, 2);
            std::string save = GET_SAVE_DATA("video_frames") + "/" + std::to_string(i) + ".jpg";
            image.Write(save);
            auto id = multipleFaceData.trackIds[0];
            //            TEST_PRINT("{}", id);
            if (id != expectedId) {
                count_loss++;
            }

            ret = HFReleaseImageStream(imgHandle);
            REQUIRE(ret == HSUCCEED);
        }
        float loss = (float)count_loss / filenames.size();
        // The face track loss is allowed to have an error of 5%
        //        CHECK(loss == Approx(0.0f).epsilon(0.05));

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }
#endif

    SECTION("Head pose estimation") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        parameter.enable_face_pose = true;
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};

        // Left side face
        HFImageStream leftHandle;
        auto left = inspirecv::Image::Create(GET_DATA("data/pose/left_face.jpeg"));
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
        auto right = inspirecv::Image::Create(GET_DATA("data/pose/right_face.png"));
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
        auto rise = inspirecv::Image::Create(GET_DATA("data/pose/rise_face.jpeg"));
        ret = CVImageToImageStream(rise, riseHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFExecuteFaceTrack(session, riseHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        pitch = multipleFaceData.angles.pitch[0];
        CHECK(pitch > 3);
        HFReleaseImageStream(riseHandle);

        // Lower head
        HFImageStream lowerHandle;
        auto lower = inspirecv::Image::Create(GET_DATA("data/pose/lower_face.jpeg"));
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
        auto leftWryneck = inspirecv::Image::Create(GET_DATA("data/pose/left_wryneck.png"));
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
        auto rightWryneck = inspirecv::Image::Create(GET_DATA("data/pose/right_wryneck.png"));
        ret = CVImageToImageStream(rightWryneck, rightWryneckHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFExecuteFaceTrack(session, rightWryneckHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        roll = multipleFaceData.angles.roll[0];
        CHECK(roll > 25);
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
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Prepare an image
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);
        BenchmarkRecord record(getBenchmarkRecordFile());

        REQUIRE(ret == HSUCCEED);
        HFMultipleFaceData multipleFaceData = {0};
        auto timer = inspire::Timer();
        for (int i = 0; i < loop; ++i) {
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        }
        auto cost = timer.GetCostTime();
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum == 1);
        TEST_PRINT("<Benchmark> Face Detect -> Loop: {}, Total Time: {:.5f}ms, Average Time: {:.5f}ms", loop, cost, cost / loop);
        record.insertBenchmarkData("Face Detect", loop, cost, cost / loop);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
#else
        TEST_PRINT("Skip the face detection benchmark test. To run it, you need to turn on the benchmark test.");
#endif
    }

    SECTION("Face light track benchmark") {
#ifdef ISF_ENABLE_BENCHMARK
        int loop = 1000;
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_LIGHT_TRACK;
        HFSession session;
        ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Prepare an image
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);
        BenchmarkRecord record(getBenchmarkRecordFile());

        // Case: Execute the benchmark using the VIDEO mode(Track)
        REQUIRE(ret == HSUCCEED);
        HFMultipleFaceData multipleFaceData = {0};
        auto timer = inspire::Timer();
        for (int i = 0; i < loop; ++i) {
            ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        }
        auto cost = timer.GetCostTime();
        REQUIRE(ret == HSUCCEED);
        REQUIRE(multipleFaceData.detectedNum >= 1);
        TEST_PRINT("<Benchmark> Face Track -> Loop: {}, Total Time: {:.5f}ms, Average Time: {:.5f}ms", loop, cost, cost / loop);
        record.insertBenchmarkData("Face Track", loop, cost, cost / loop);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
#else
        TEST_PRINT("Skip the face light track benchmark test. To run it, you need to turn on the benchmark test.");
#endif
    }
}

TEST_CASE("test_MultipleLevelFaceDetect", "[face_detect]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Detect input 192px") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        HInt32 detectPixelLevel = 192;
        ret = HFCreateInspireFaceSession(parameter, detMode, 20, detectPixelLevel, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFSessionSetTrackPreviewSize(session, detectPixelLevel);
        HFSessionSetFilterMinimumFacePixelSize(session, 0);
        
        // Check the preview size
        HInt32 previewSize;
        ret = HFSessionGetTrackPreviewSize(session, &previewSize);
        REQUIRE(ret == HSUCCEED);
        CHECK(previewSize == detectPixelLevel);

        // Get a face picture
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/pedestrian.png"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);

        CHECK(multipleFaceData.detectedNum > 0);
        CHECK(multipleFaceData.detectedNum < 7);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Detect input 320px") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        HInt32 detectPixelLevel = 320;
        ret = HFCreateInspireFaceSession(parameter, detMode, 20, detectPixelLevel, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFSessionSetTrackPreviewSize(session, detectPixelLevel);
        HFSessionSetFilterMinimumFacePixelSize(session, 0);

        // Check the preview size
        HInt32 previewSize;
        ret = HFSessionGetTrackPreviewSize(session, &previewSize);
        REQUIRE(ret == HSUCCEED);
        CHECK(previewSize == detectPixelLevel);

        // Get a face picture
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/pedestrian.png"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);

        CHECK(multipleFaceData.detectedNum > 9);
        CHECK(multipleFaceData.detectedNum < 12);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Detect input 640px") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        HInt32 detectPixelLevel = 640;
        ret = HFCreateInspireFaceSession(parameter, detMode, 25, detectPixelLevel, -1, &session);
        REQUIRE(ret == HSUCCEED);
        HFSessionSetTrackPreviewSize(session, detectPixelLevel);
        HFSessionSetFilterMinimumFacePixelSize(session, 0);

        // Check the preview size
        HInt32 previewSize;
        ret = HFSessionGetTrackPreviewSize(session, &previewSize);
        REQUIRE(ret == HSUCCEED);
        CHECK(previewSize == detectPixelLevel);

        // Get a face picture
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/pedestrian.png"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);

        CHECK(multipleFaceData.detectedNum > 15);
        CHECK(multipleFaceData.detectedNum < 21);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }
}

TEST_CASE("test_FaceTrackPreviewSizeSetting", "[face_track]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Default preview size and detection level size") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        HInt32 levelSize = -1;
        ret = HFCreateInspireFaceSession(parameter, detMode, 20, levelSize, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Check the preview size
        HInt32 previewSize;
        ret = HFSessionGetTrackPreviewSize(session, &previewSize);
        REQUIRE(ret == HSUCCEED);
        
        CHECK(previewSize == 320);

        // Get a face picture
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/pedestrian.png"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);

        CHECK(multipleFaceData.detectedNum > 0);
        // Check the preview size
        HInt32 debugPreviewSize;
        ret = HFSessionLastFaceDetectionGetDebugPreviewImageSize(session, &debugPreviewSize);
        REQUIRE(ret == HSUCCEED);
        CHECK(debugPreviewSize == 320);

        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Set preview size to 320px") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        HInt32 levelSize = 320;
        ret = HFCreateInspireFaceSession(parameter, detMode, 20, levelSize, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Check the preview size
        HInt32 previewSize;
        ret = HFSessionGetTrackPreviewSize(session, &previewSize);
        REQUIRE(ret == HSUCCEED);
        CHECK(previewSize == levelSize);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("Set the detect level to an invalid value") {
        HResult ret;
        HFSessionCustomParameter parameter = {0};
        HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
        HFSession session;
        HInt32 levelSize = 1000;
        ret = HFCreateInspireFaceSession(parameter, detMode, 20, levelSize, -1, &session);
        REQUIRE(ret == HSUCCEED);

        // Check the preview size
        HInt32 previewSize;
        ret = HFSessionGetTrackPreviewSize(session, &previewSize);
        REQUIRE(ret == HSUCCEED);
        // If the detect level value is invalid, the value will be automatically adjusted to the nearest legal size. 
        // If the default value of preview_size is -1, the value will also be adjusted
        CHECK(previewSize == 640);

        // Get a face picture
        HFImageStream imgHandle;
        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/pedestrian.png"));
        ret = CVImageToImageStream(image, imgHandle);
        REQUIRE(ret == HSUCCEED);

        // Extract basic face information from photos
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);

        CHECK(multipleFaceData.detectedNum > 0);
        
        // Check the preview size
        HInt32 debugPreviewSize;
        ret = HFSessionLastFaceDetectionGetDebugPreviewImageSize(session, &debugPreviewSize);
        REQUIRE(ret == HSUCCEED);
        CHECK(debugPreviewSize == 640);

        // Set a value manually
        ret = HFSessionSetTrackPreviewSize(session, 192);
        REQUIRE(ret == HSUCCEED);

        // Check the preview size
        ret = HFSessionGetTrackPreviewSize(session, &previewSize);
        REQUIRE(ret == HSUCCEED);
        CHECK(previewSize == 192);
        
        ret = HFExecuteFaceTrack(session, imgHandle, &multipleFaceData);
        REQUIRE(ret == HSUCCEED);

        CHECK(multipleFaceData.detectedNum > 0);

        // Check the preview size
        ret = HFSessionLastFaceDetectionGetDebugPreviewImageSize(session, &debugPreviewSize);
        REQUIRE(ret == HSUCCEED);
        CHECK(debugPreviewSize == 192);


        ret = HFReleaseImageStream(imgHandle);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseInspireFaceSession(session);
        REQUIRE(ret == HSUCCEED);
    }
    
}