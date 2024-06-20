//
// Created by tunm on 2023/9/16.
//

#include "settings/test_settings.h"
#include "inspireface/face_context.h"
#include "herror.h"

using namespace inspire;

TEST_CASE("test_FaceDetectTrack", "[face_track]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("TrackBenchmark") {
        // Initialize
        FaceContext ctx;
        CustomPipelineParameter param;
        param.enable_face_quality = true;
        auto ret = ctx.Configuration(DetectMode::DETECT_MODE_LIGHT_TRACK, 1, param);
        REQUIRE(ret == HSUCCEED);

        // Prepare a picture of a face
        auto image = cv::imread(GET_DATA("images/face_sample.png"));
        CameraStream stream;
        stream.SetDataFormat(BGR);
        stream.SetRotationMode(ROTATION_0);
        stream.SetDataBuffer(image.data, image.rows, image.cols);

        const auto loop = 1000;
        double total = 0.0f;
        spdlog::info("begin {} times tracking: ", loop);

        auto out = (double) cv::getTickCount();
        for (int i = 0; i < loop; ++i) {
            auto timeStart = (double) cv::getTickCount();
            // Face detection
            ctx.FaceDetectAndTrack(stream);
            auto &faces = ctx.GetTrackingFaceList();
            double cost = ((double) cv::getTickCount() - timeStart) / cv::getTickFrequency() * 1000;
            REQUIRE(ret == HSUCCEED);
            REQUIRE(faces.size() > 0);
            total += cost;
        }
        auto end = ((double) cv::getTickCount() - out) / cv::getTickFrequency() * 1000;

        spdlog::info("[Face Tracking]{} times, Total cost: {}ms, Average cost: {}ms", loop, end, total / loop);
    }

}