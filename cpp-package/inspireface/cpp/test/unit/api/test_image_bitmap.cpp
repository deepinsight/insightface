
#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include <cstdio>

TEST_CASE("test_ImageBitmap", "[image_bitmap]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    HFImageBitmap handle;
    HResult ret = HFCreateImageBitmapFromFilePath(GET_DATA("data/bulk/r90.jpg").c_str(), 3, &handle);
    REQUIRE(ret == HSUCCEED);

    HFImageStream stream;
    ret = HFCreateImageStreamFromImageBitmap(handle, HF_CAMERA_ROTATION_90, &stream);
    REQUIRE(ret == HSUCCEED);

    HFSessionCustomParameter parameter = {0};
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    HFSession session;
    ret = HFCreateInspireFaceSession(parameter, detMode, 3, -1, -1, &session);
    TEST_ERROR_PRINT("error ret :{}", ret);
    REQUIRE(ret == HSUCCEED);

    // Extract basic face information from photos
    HFMultipleFaceData multipleFaceData = {0};
    ret = HFExecuteFaceTrack(session, stream, &multipleFaceData);
    REQUIRE(ret == HSUCCEED);
    REQUIRE(multipleFaceData.detectedNum == 1);

    auto rect = multipleFaceData.rects[0];
    HColor color = {0, 0, 255};
    HFImageBitmapDrawRect(handle, rect, color, 2);
    HFImageBitmapWriteToFile(handle, "bitmap_draw_test.jpg");

    ret = HFReleaseInspireFaceSession(session);
    REQUIRE(ret == HSUCCEED);

    ret = HFReleaseImageStream(stream);
    REQUIRE(ret == HSUCCEED);

    ret = HFReleaseImageBitmap(handle);
    REQUIRE(ret == HSUCCEED);
}