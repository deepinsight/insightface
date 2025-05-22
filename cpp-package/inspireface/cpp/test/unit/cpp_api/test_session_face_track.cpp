/**
 * Created by Jingyu Yan
 * @date 2025-03-29
 */

#include <iostream>
#include "settings/test_settings.h"
#include "unit/test_helper/simple_csv_writer.h"
#include "unit/test_helper/test_help.h"
#include "unit/test_helper/test_tools.h"
#include <inspireface/include/inspireface/inspireface.hpp>

using namespace inspire;

TEST_CASE("test_SessionFaceTrack", "[session_face_track]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("Face detection from image") {
        CustomPipelineParameter param;
        int32_t ret;

        std::shared_ptr<Session> session = std::shared_ptr<Session>(Session::CreatePtr(DetectModuleMode::DETECT_MODE_ALWAYS_DETECT, 3, param));
        REQUIRE(session != nullptr);

        auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
        auto process = inspirecv::FrameProcess::Create(image, inspirecv::DATA_FORMAT::BGR, inspirecv::ROTATION_MODE::ROTATION_0);

        std::vector<FaceTrackWrap> results;
        ret = session->FaceDetectAndTrack(process, results);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(results.size() == 1);
    }

    SECTION("Head pose estimation") {
        CustomPipelineParameter param;
        int32_t ret;

        std::shared_ptr<Session> session = std::shared_ptr<Session>(Session::CreatePtr(DetectModuleMode::DETECT_MODE_ALWAYS_DETECT, 3, param));
        REQUIRE(session != nullptr);

        HFMultipleFaceData multipleFaceData = {0};
        std::vector<FaceTrackWrap> results;

        // Left side face
        HFImageStream leftHandle;
        auto left = inspirecv::Image::Create(GET_DATA("data/pose/left_face.jpeg"));
        auto leftProcess = inspirecv::FrameProcess::Create(left, inspirecv::DATA_FORMAT::BGR, inspirecv::ROTATION_MODE::ROTATION_0);

        ret = session->FaceDetectAndTrack(leftProcess, results);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(results.size() == 1);

        HFloat yaw, pitch, roll;
        bool checked;

        // Left-handed rotation
        yaw = results[0].face3DAngle.yaw;
        checked = (yaw > -90 && yaw < -10);
        CHECK(checked);
        
        // Right-handed rotation
        auto right = inspirecv::Image::Create(GET_DATA("data/pose/right_face.png"));
        auto rightProcess = inspirecv::FrameProcess::Create(right, inspirecv::DATA_FORMAT::BGR, inspirecv::ROTATION_MODE::ROTATION_0);

        ret = session->FaceDetectAndTrack(rightProcess, results);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(results.size() == 1);

        yaw = results[0].face3DAngle.yaw;
        checked = (yaw > 10 && yaw < 90);
        CHECK(checked);

        // Rise head
        auto rise = inspirecv::Image::Create(GET_DATA("data/pose/rise_face.jpeg"));
        auto riseProcess = inspirecv::FrameProcess::Create(rise, inspirecv::DATA_FORMAT::BGR, inspirecv::ROTATION_MODE::ROTATION_0);

        ret = session->FaceDetectAndTrack(riseProcess, results);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(results.size() == 1);

        pitch = results[0].face3DAngle.pitch;
        CHECK(pitch > 3);

        // Lower head
        auto lower = inspirecv::Image::Create(GET_DATA("data/pose/lower_face.jpeg"));
        auto lowerProcess = inspirecv::FrameProcess::Create(lower, inspirecv::DATA_FORMAT::BGR, inspirecv::ROTATION_MODE::ROTATION_0);

        ret = session->FaceDetectAndTrack(lowerProcess, results);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(results.size() == 1);

        pitch = results[0].face3DAngle.pitch;
        CHECK(pitch < -10);

        // Roll head
        auto leftWryneck = inspirecv::Image::Create(GET_DATA("data/pose/left_wryneck.png"));
        auto leftWryneckProcess = inspirecv::FrameProcess::Create(leftWryneck, inspirecv::DATA_FORMAT::BGR, inspirecv::ROTATION_MODE::ROTATION_0);

        ret = session->FaceDetectAndTrack(leftWryneckProcess, results);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(results.size() == 1);

        roll = results[0].face3DAngle.roll;
        CHECK(roll < -30);

        // Roll head
        auto rightWryneck = inspirecv::Image::Create(GET_DATA("data/pose/right_wryneck.png"));
        auto rightWryneckProcess = inspirecv::FrameProcess::Create(rightWryneck, inspirecv::DATA_FORMAT::BGR, inspirecv::ROTATION_MODE::ROTATION_0);

        ret = session->FaceDetectAndTrack(rightWryneckProcess, results);
        REQUIRE(ret == HSUCCEED);
        REQUIRE(results.size() == 1);

        roll = results[0].face3DAngle.roll;
        CHECK(roll > 25);
    }
}