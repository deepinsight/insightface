//
// Created by Tunm-Air13 on 2024/2/2.
//

#include "settings/test_settings.h"
#include "inspireface/face_context.h"
#include "common/face_data/data_tools.h"
#include "../test_helper/test_tools.h"
#include "herror.h"

using namespace inspire;

TEST_CASE("test_CameraStream", "[camera_stream") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    SECTION("DecodingRotatedImages") {
        FaceContext ctx;
        CustomPipelineParameter param;
        auto ret = ctx.Configuration(DetectMode::DETECT_MODE_ALWAYS_DETECT, 1, param);
        REQUIRE(ret == HSUCCEED);

        std::vector<std::string> rotated_filename_list = {
                getTestData("images/rotate/rot_0.jpg"),
                getTestData("images/rotate/rot_90.jpg"),
                getTestData("images/rotate/rot_180.jpg"),
                getTestData("images/rotate/rot_270.jpg"),
        };
        std::vector<ROTATION_MODE> rotate_list = {ROTATION_0, ROTATION_90, ROTATION_180, ROTATION_270};

        CHECK(rotate_list.size() == rotated_filename_list.size());

        for (int i = 0; i < rotate_list.size(); ++i) {
            cv::Mat image = cv::imread(rotated_filename_list[i]);
            REQUIRE(!image.empty());
            auto rotated = rotate_list[i];

            CameraStream stream;
            stream.SetDataBuffer(image.data, image.rows, image.cols);
            stream.SetDataFormat(BGR);
            stream.SetRotationMode(rotated);

            ret = ctx.FaceDetectAndTrack(stream);
            REQUIRE(ret == HSUCCEED);
            const auto &faces = ctx.GetTrackingFaceList();
            CHECK(faces.size() == 1);
        }

    }

    SECTION("DecodingNV21Image") {
        FaceContext ctx;
        CustomPipelineParameter param;
        auto ret = ctx.Configuration(DetectMode::DETECT_MODE_ALWAYS_DETECT, 1, param);
        REQUIRE(ret == HSUCCEED);

        int32_t width = 402;
        int32_t height = 324;
        auto rotated = ROTATION_90;
        auto format = NV21;
        auto nv21 = ReadNV21Data(getTestData("images/rotate/rot_90_324x402.nv21").c_str(), width, height);
        REQUIRE(nv21 != nullptr);

        CameraStream stream;
        stream.SetDataBuffer(nv21, height, width);
        stream.SetDataFormat(format);
        stream.SetRotationMode(rotated);

        ret = ctx.FaceDetectAndTrack(stream);
        REQUIRE(ret == HSUCCEED);
        const auto &faces = ctx.GetTrackingFaceList();
        CHECK(faces.size() == 1);
    }

}