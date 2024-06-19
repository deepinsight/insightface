//
// Created by tunm on 2023/9/13.
//

#include "settings/test_settings.h"
#include "inspireface/face_context.h"
#include "herror.h"

using namespace inspire;

TEST_CASE("test_FacePipeline", "[face_pipe") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);


    SECTION("FaceContextInit") {
        FaceContext ctx;
        CustomPipelineParameter param;
        auto ret = ctx.Configuration(DetectMode::DETECT_MODE_ALWAYS_DETECT, 1, param);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("FaceContextMaskPredict") {
        FaceContext ctx;
        CustomPipelineParameter param;
        param.enable_mask_detect = true;
        auto ret = ctx.Configuration(DetectMode::DETECT_MODE_ALWAYS_DETECT, 1, param);
        REQUIRE(ret == HSUCCEED);

        {
            // Prepare a photo of your face without a mask
            auto image = cv::imread(GET_DATA("images/kun.jpg"));
            CameraStream stream;
            stream.SetDataFormat(BGR);
            stream.SetRotationMode(ROTATION_0);
            stream.SetDataBuffer(image.data, image.rows, image.cols);
            ret = ctx.FaceDetectAndTrack(stream);
            REQUIRE(ret == HSUCCEED);
            // Face detection
            ctx.FaceDetectAndTrack(stream);
            auto &faces = ctx.GetTrackingFaceList();
            REQUIRE(faces.size() > 0);
            auto &face = faces[0];
            ctx.FacePipelineModule()->Process(stream, face);
            CHECK(face.faceProcess.maskInfo == MaskInfo::UNMASKED);
        }
        {
            // Prepare a face picture with a mask in advance
            auto image = cv::imread(GET_DATA("images/mask.png"));
            CameraStream stream;
            stream.SetDataFormat(BGR);
            stream.SetRotationMode(ROTATION_0);
            stream.SetDataBuffer(image.data, image.rows, image.cols);
            ret = ctx.FaceDetectAndTrack(stream);
            REQUIRE(ret == HSUCCEED);
            // Face detection
            ctx.FaceDetectAndTrack(stream);
            auto &faces = ctx.GetTrackingFaceList();
            REQUIRE(faces.size() > 0);
            auto &face = faces[0];
            ctx.FacePipelineModule()->Process(stream, face);
            CHECK(face.faceProcess.maskInfo == MaskInfo::MASKED);
        }


        SECTION("FaceContextLiveness") {
            FaceContext ctx;
            CustomPipelineParameter param;
            param.enable_liveness = true;
            auto ret = ctx.Configuration(DetectMode::DETECT_MODE_ALWAYS_DETECT, 1, param);
            REQUIRE(ret == HSUCCEED);

            {
                // Prepare realistic face images
                auto image = cv::imread(GET_DATA("images/face_sample.png"));
                CameraStream stream;
                stream.SetDataFormat(BGR);
                stream.SetRotationMode(ROTATION_0);
                stream.SetDataBuffer(image.data, image.rows, image.cols);
                ret = ctx.FaceDetectAndTrack(stream);
                REQUIRE(ret == HSUCCEED);
                // Face detection
                ctx.FaceDetectAndTrack(stream);
                auto &faces = ctx.GetTrackingFaceList();
                REQUIRE(faces.size() > 0);
                auto &face = faces[0];
                ctx.FacePipelineModule()->Process(stream, face);
                CHECK(face.faceProcess.rgbLivenessInfo == RGBLivenessInfo::LIVENESS_REAL);
            }

            {
                // Prepare a fake photo that wasn't actually taken
                auto image = cv::imread(GET_DATA("images/rgb_fake.jpg"));
                CameraStream stream;
                stream.SetDataFormat(BGR);
                stream.SetRotationMode(ROTATION_0);
                stream.SetDataBuffer(image.data, image.rows, image.cols);
                ret = ctx.FaceDetectAndTrack(stream);
                REQUIRE(ret == HSUCCEED);
                // Face detection
                ctx.FaceDetectAndTrack(stream);
                auto &faces = ctx.GetTrackingFaceList();
                REQUIRE(faces.size() > 0);
                auto &face = faces[0];
                ctx.FacePipelineModule()->Process(stream, face);
                CHECK(face.faceProcess.rgbLivenessInfo == RGBLivenessInfo::LIVENESS_FAKE);
            }

        }
    }

}