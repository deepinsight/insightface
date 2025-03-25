#include <iostream>
#include "settings/test_settings.h"
#include "inspireface/c_api/inspireface.h"
#include <cstdio>

uint8_t* ReadNV21File(const char* filepath, size_t* fileSize) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        if (fileSize)
            *fileSize = 0;
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t* data = new uint8_t[size];

    size_t read_size = fread(data, 1, size, fp);
    fclose(fp);

    if (read_size != size) {
        delete[] data;
        if (fileSize)
            *fileSize = 0;
        return nullptr;
    }

    if (fileSize)
        *fileSize = size;
    return data;
}

TEST_CASE("test_ImageProcessRotateNV21", "[image_process]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    HFImageBitmap originBmp;
    size_t fileSize;
    uint8_t* data = ReadNV21File(GET_DATA("data/bulk/r0_w330_h409_c3.nv21").c_str(), &fileSize);
    REQUIRE(data != nullptr);

    HFImageData imageData;
    imageData.data = data;
    imageData.width = 330;
    imageData.height = 409;
    imageData.rotation = HF_CAMERA_ROTATION_0;
    imageData.format = HF_STREAM_YUV_NV21;

    HFImageStream stream;
    HResult ret = HFCreateImageStream(&imageData, &stream);
    REQUIRE(ret == HSUCCEED);

    ret = HFCreateImageBitmapFromImageStreamProcess(stream, &originBmp, 1, 1.0f);
    REQUIRE(ret == HSUCCEED);

    HFImageBitmapData originData;
    ret = HFImageBitmapGetData(originBmp, &originData);
    REQUIRE(ret == HSUCCEED);

    // compare with eps(0~1)
    float eps = 0.01;

    SECTION("rotate 90") {
        size_t fileSize;
        uint8_t* r90nv21 = ReadNV21File(GET_DATA("data/bulk/r90_w409_h330_c3.nv21").c_str(), &fileSize);
        REQUIRE(r90nv21 != nullptr);

        HFImageData imageData;
        imageData.data = r90nv21;
        imageData.width = 409;
        imageData.height = 330;
        imageData.rotation = HF_CAMERA_ROTATION_90;
        imageData.format = HF_STREAM_YUV_NV21;

        HFImageStream stream;
        ret = HFCreateImageStream(&imageData, &stream);
        REQUIRE(ret == HSUCCEED);

        HFImageBitmap rot90;
        ret = HFCreateImageBitmapFromImageStreamProcess(stream, &rot90, 1, 1.0f);
        REQUIRE(ret == HSUCCEED);

        // HFImageBitmapShow(rot90, "w", 0);

        HFImageBitmapData rot90Data;
        ret = HFImageBitmapGetData(rot90, &rot90Data);
        REQUIRE(ret == HSUCCEED);

        REQUIRE_EQ_IMAGE_WITH_EPS(originData.data, rot90Data.data, originData.height, originData.width, originData.channels, eps);

        ret = HFReleaseImageBitmap(rot90);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageStream(stream);
        REQUIRE(ret == HSUCCEED);

        delete[] r90nv21;
    }

    ret = HFReleaseImageStream(stream);
    REQUIRE(ret == HSUCCEED);

    ret = HFReleaseImageBitmap(originBmp);
    REQUIRE(ret == HSUCCEED);

    delete[] data;
}

TEST_CASE("test_ImageProcessRotate", "[image_process]") {
    DRAW_SPLIT_LINE
    TEST_PRINT_OUTPUT(true);

    HFImageBitmap originBmp;
    HResult ret = HFCreateImageBitmapFromFilePath(GET_DATA("data/bulk/r0.jpg").c_str(), 3, &originBmp);
    REQUIRE(ret == HSUCCEED);

    HFImageBitmapData originData;
    ret = HFImageBitmapGetData(originBmp, &originData);
    REQUIRE(ret == HSUCCEED);

    // compare with eps(0~1)
    float eps = 0.001;

    SECTION("rotate 90") {
        HFImageBitmap bitmap;
        HResult ret = HFCreateImageBitmapFromFilePath(GET_DATA("data/bulk/r90.jpg").c_str(), 3, &bitmap);
        REQUIRE(ret == HSUCCEED);

        HFImageStream stream;
        ret = HFCreateImageStreamFromImageBitmap(bitmap, HF_CAMERA_ROTATION_90, &stream);
        REQUIRE(ret == HSUCCEED);

        HFImageBitmap rot90;
        ret = HFCreateImageBitmapFromImageStreamProcess(stream, &rot90, 1, 1.0f);
        REQUIRE(ret == HSUCCEED);

        HFImageBitmapData rot90Data;
        ret = HFImageBitmapGetData(rot90, &rot90Data);
        REQUIRE(ret == HSUCCEED);

        REQUIRE_EQ_IMAGE_WITH_EPS(originData.data, rot90Data.data, originData.height, originData.width, originData.channels, eps);

        ret = HFReleaseImageBitmap(rot90);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageStream(stream);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageBitmap(bitmap);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("rotate 180") {
        HFImageBitmap bitmap;
        HResult ret = HFCreateImageBitmapFromFilePath(GET_DATA("data/bulk/r180.jpg").c_str(), 3, &bitmap);
        REQUIRE(ret == HSUCCEED);

        HFImageStream stream;
        ret = HFCreateImageStreamFromImageBitmap(bitmap, HF_CAMERA_ROTATION_180, &stream);
        REQUIRE(ret == HSUCCEED);

        HFImageBitmap rot180;
        ret = HFCreateImageBitmapFromImageStreamProcess(stream, &rot180, 1, 1.0f);
        REQUIRE(ret == HSUCCEED);

        HFImageBitmapData rot180Data;
        ret = HFImageBitmapGetData(rot180, &rot180Data);
        REQUIRE(ret == HSUCCEED);

        REQUIRE_EQ_IMAGE_WITH_EPS(originData.data, rot180Data.data, originData.height, originData.width, originData.channels, eps);

        ret = HFReleaseImageBitmap(rot180);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageStream(stream);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageBitmap(bitmap);
        REQUIRE(ret == HSUCCEED);
    }

    SECTION("rotate 270") {
        HFImageBitmap bitmap;
        HResult ret = HFCreateImageBitmapFromFilePath(GET_DATA("data/bulk/r270.jpg").c_str(), 3, &bitmap);
        REQUIRE(ret == HSUCCEED);

        HFImageStream stream;
        ret = HFCreateImageStreamFromImageBitmap(bitmap, HF_CAMERA_ROTATION_270, &stream);
        REQUIRE(ret == HSUCCEED);

        HFImageBitmap rot270;
        ret = HFCreateImageBitmapFromImageStreamProcess(stream, &rot270, 1, 1.0f);
        REQUIRE(ret == HSUCCEED);

        HFImageBitmapData rot270Data;
        ret = HFImageBitmapGetData(rot270, &rot270Data);
        REQUIRE(ret == HSUCCEED);

        REQUIRE_EQ_IMAGE_WITH_EPS(originData.data, rot270Data.data, originData.height, originData.width, originData.channels, eps);

        ret = HFReleaseImageBitmap(rot270);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageStream(stream);
        REQUIRE(ret == HSUCCEED);

        ret = HFReleaseImageBitmap(bitmap);
        REQUIRE(ret == HSUCCEED);
    }

    ret = HFReleaseImageBitmap(originBmp);
    REQUIRE(ret == HSUCCEED);
}