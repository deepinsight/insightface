//
// Created by Tunm-Air13 on 2024/4/17.
//
#include <iostream>
#include "opencv2/opencv.hpp"
#include "inspireface/c_api/inspireface.h"
#include <thread>

void runFaceTrack(HFSession session, HFImageStream imageHandle) {
    HFMultipleFaceData multipleFaceData = {0};
    auto ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
    if (ret != HSUCCEED) {
        std::cout << "Thread " << std::this_thread::get_id() << " Execute HFExecuteFaceTrack error: " << ret << std::endl;
    } else {
        std::cout << "Thread " << std::this_thread::get_id() << " successfully executed HFExecuteFaceTrack.\n";
    }
}

int main(int argc, char* argv[]) {
    // Check whether the number of parameters is correct
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <pack_path> <source_path>\n";
        return 1;
    }

    auto packPath = argv[1];
    auto sourcePath = argv[2];

    std::cout << "Pack file Path: " << packPath << std::endl;
    std::cout << "Source file Path: " << sourcePath << std::endl;

    HResult ret;
    // The resource file must be loaded before it can be used
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        std::cout << "Load Resource error: " << ret << std::endl;
        return ret;
    }

    // Enable the functions in the pipeline: mask detection, live detection, and face quality detection
    HOption option = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS;
    // Non-video or frame sequence mode uses IMAGE-MODE, which is always face detection without tracking
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    // Maximum number of faces detected
    HInt32 maxDetectNum = 5;
    // Handle of the current face SDK algorithm session
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, -1, -1, &session);
    if (ret != HSUCCEED) {
        std::cout << "Create FaceContext error: " << ret << std::endl;
        return ret;
    }

    // Load a image
    cv::Mat image = cv::imread(sourcePath);
    if (image.empty()) {
        std::cout << "The source entered is not a picture or read error." << std::endl;
        return 1;
    }
    // Prepare an image parameter structure for configuration
    HFImageData imageParam = {0};
    imageParam.data = image.data;       // Data buffer
    imageParam.width = image.cols;      // Target view width
    imageParam.height = image.rows;      // Target view width
    imageParam.rotation = HF_CAMERA_ROTATION_0;      // Data source rotate
    imageParam.format = HF_STREAM_BGR;      // Data source format

    // Create an image data stream
    HFImageStream imageHandle = {0};
    ret = HFCreateImageStream(&imageParam, &imageHandle);
    if (ret != HSUCCEED) {
        std::cout << "Create ImageStream error: " << ret << std::endl;
        return ret;
    }

    // Create and start multiple threads
    const size_t numThreads = 10;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(runFaceTrack, session, imageHandle);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // The memory must be freed at the end of the program
    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        printf("Release FaceContext error: %lu\n", ret);
        return ret;
    }

    return 0;
}