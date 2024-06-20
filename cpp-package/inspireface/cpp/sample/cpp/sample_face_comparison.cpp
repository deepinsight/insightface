//
// Created by tunm on 2024/4/20.
//
#include <iostream>
#include "opencv2/opencv.hpp"
#include "inspireface/c_api/inspireface.h"

int main(int argc, char* argv[]) {
    // Check whether the number of parameters is correct
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <pack_path> <img1_path> <img2_path>\n";
        return 1;
    }

    auto packPath = argv[1];
    auto imgPath1 = argv[2];
    auto imgPath2 = argv[3];

    std::cout << "Pack file Path: " << packPath << std::endl;
    std::cout << "Source file Path 1: " << imgPath1 << std::endl;
    std::cout << "Source file Path 2: " << imgPath2 << std::endl;

    HResult ret;
    // The resource file must be loaded before it can be used
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        std::cout << "Load Resource error: " << ret << std::endl;
        return ret;
    }

    // Create a session for face recognition
    HOption option = HF_ENABLE_FACE_RECOGNITION;
    HFSession session;
    ret = HFCreateInspireFaceSessionOptional(option, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &session);
    if (ret != HSUCCEED) {
        std::cout << "Create session error: " << ret << std::endl;
        return ret;
    }


    std::vector<char* > twoImg = {imgPath1, imgPath2};
    std::vector<std::vector<float>> vec(2, std::vector<float>(512));
    for (int i = 0; i < twoImg.size(); ++i) {
        auto image = cv::imread(twoImg[i]);
        if (image.empty()) {
            std::cout << "Image is empty: " << twoImg[i] << std::endl;
            return 0;
        }
        // Prepare image data for processing
        HFImageData imageData = {0};
        imageData.data = image.data; // Pointer to the image data
        imageData.format = HF_STREAM_BGR; // Image format (BGR in this case)
        imageData.height = image.rows; // Image height
        imageData.width = image.cols; // Image width
        imageData.rotation = HF_CAMERA_ROTATION_0; // Image rotation
        HFImageStream stream;
        ret = HFCreateImageStream(&imageData, &stream); // Create an image stream for processing
        if (ret != HSUCCEED) {
            std::cout << "Create stream error: " << ret << std::endl;
            return ret;
        }

        // Execute face tracking on the image
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, stream, &multipleFaceData); // Track faces in the image
        if (ret != HSUCCEED) {
            std::cout << "Run face track error: " << ret << std::endl;
            return ret;
        }
        if (multipleFaceData.detectedNum == 0) { // Check if any faces were detected
            std::cout << "No face was detected: " << twoImg[i] << ret << std::endl;
            return ret;
        }

        // Extract facial features from the first detected face, an interface that uses copy features in a comparison scenario
        ret = HFFaceFeatureExtractCpy(session, stream, multipleFaceData.tokens[0], vec[i].data()); // Extract features
        if (ret != HSUCCEED) {
            std::cout << "Extract feature error: " << ret << std::endl;
            return ret;
        }

        ret = HFReleaseImageStream(stream);
        if (ret != HSUCCEED) {
            printf("Release image stream error: %lu\n", ret);
        }

    }

    // Make feature1
    HFFaceFeature feature1 = {0};
    feature1.data = vec[0].data();
    feature1.size = vec[0].size();

    // Make feature2
    HFFaceFeature feature2 = {0};
    feature2.data = vec[1].data();
    feature2.size = vec[1].size();

    // Run comparison
    HFloat similarity;
    ret = HFFaceComparison(feature1, feature2, &similarity);
    if (ret != HSUCCEED) {
        std::cout << "Feature comparison error: " << ret << std::endl;
        return ret;
    }

    std::cout << "Similarity: " << similarity << std::endl;

    // The memory must be freed at the end of the program
    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        printf("Release session error: %lu\n", ret);
        return ret;
    }

}