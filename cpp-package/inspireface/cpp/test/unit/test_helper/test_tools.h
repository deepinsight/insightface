//
// Created by tunm on 2023/10/12.
//
#pragma
#ifndef HYPERFACEREPO_TEST_TOOLS_H
#define HYPERFACEREPO_TEST_TOOLS_H

#include "opencv2/opencv.hpp"
#include "inspireface/c_api/inspireface.h"
#include <fstream>
#include <cstdint> // For uint8_t

// Bad function
inline HResult ReadImageToImageStream(const char *path, HFImageStream &handle, HFImageFormat format = HF_STREAM_BGR,
                                      HFRotation rot = HF_CAMERA_ROTATION_0) {
    cv::Mat image = cv::imread(path);
    if (image.empty()) {
        return -1;
    }
    HFImageData imageData = {0};
    imageData.data = image.data;
    imageData.height = image.rows;
    imageData.width = image.cols;
    imageData.format = format;
    imageData.rotation = rot;

    auto ret = HFCreateImageStream(&imageData, &handle);

    return ret;
}

inline HResult CVImageToImageStream(const cv::Mat& image, HFImageStream &handle, HFImageFormat format = HF_STREAM_BGR,
                                    HFRotation rot = HF_CAMERA_ROTATION_0) {
    if (image.empty()) {
        return -1;
    }
    HFImageData imageData = {0};
    imageData.data = image.data;
    imageData.height = image.rows;
    imageData.width = image.cols;
    imageData.format = format;
    imageData.rotation = rot;

    auto ret = HFCreateImageStream(&imageData, &handle);

    return ret;
}


inline uint8_t *ReadNV21Data(const char *filePath, int width, int height) {
    const int nv21Size = width * height * 3 / 2; // Calculate the NV21 data size

    // Memory is allocated dynamically to store NV21 data
    uint8_t *nv21Data = new uint8_t[nv21Size];

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Unable to open the file " << filePath << std::endl;
        delete[] nv21Data;
        return nullptr;
    }

    // Read data
    file.read(reinterpret_cast<char *>(nv21Data), nv21Size);
    if (!file) {
        std::cerr << "Read error or incomplete file" << std::endl;
        file.close();
        delete[] nv21Data;
        return nullptr;
    }

    // Open file
    file.close();

    // Returns a pointer to NV21 data
    return nv21Data;
}

#endif //HYPERFACEREPO_TEST_TOOLS_H
