/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#include <iostream>
#include <inspireface.h>

int main(int argc, char* argv[]) {
    // Check whether the number of parameters is correct
    if (argc < 3 || argc > 4) {
        HFLogPrint(HF_LOG_ERROR, "Usage: %s <pack_path> <source_path> [rotation]", argv[0]);
        return 1;
    }

    auto packPath = argv[1];
    auto sourcePath = argv[2];
    int rotation = 0;

    // If rotation is provided, check and set the value
    if (argc == 4) {
        rotation = std::atoi(argv[3]);
        if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) {
            HFLogPrint(HF_LOG_ERROR, "Invalid rotation value. Allowed values are 0, 90, 180, 270.");
            return 1;
        }
    }
    HFRotation rotation_enum;
    // Set rotation based on input parameter
    switch (rotation) {
        case 90:
            rotation_enum = HF_CAMERA_ROTATION_90;
            break;
        case 180:
            rotation_enum = HF_CAMERA_ROTATION_180;
            break;
        case 270:
            rotation_enum = HF_CAMERA_ROTATION_270;
            break;
        case 0:
        default:
            rotation_enum = HF_CAMERA_ROTATION_0;
            break;
    }

    HFLogPrint(HF_LOG_INFO, "Pack file Path: %s", packPath);
    HFLogPrint(HF_LOG_INFO, "Source file Path: %s", sourcePath);
    HFLogPrint(HF_LOG_INFO, "Rotation: %d", rotation);

    HFSetLogLevel(HF_LOG_INFO);

    HResult ret;
    // The resource file must be loaded before it can be used
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
        return ret;
    }

    // Enable the functions in the pipeline: mask detection, live detection, and face quality
    // detection
    HOption option = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS | HF_ENABLE_DETECT_MODE_LANDMARK;
    // Non-video or frame sequence mode uses IMAGE-MODE, which is always face detection without
    // tracking
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    // Maximum number of faces detected
    HInt32 maxDetectNum = 20;
    // Face detection image input level
    HInt32 detectPixelLevel = 160;
    // Handle of the current face SDK algorithm context
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create FaceContext error: %d", ret);
        return ret;
    }

    HFSessionSetTrackPreviewSize(session, detectPixelLevel);
    HFSessionSetFilterMinimumFacePixelSize(session, 4);

    // Load a image
    HFImageBitmap image;
    ret = HFCreateImageBitmapFromFilePath(sourcePath, 3, &image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
        return ret;
    }
    // Prepare an image parameter structure for configuration
    HFImageStream imageHandle = {0};
    ret = HFCreateImageStreamFromImageBitmap(image, rotation_enum, &imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
        return ret;
    }

    // Execute HF_FaceContextRunFaceTrack captures face information in an image
    HFMultipleFaceData multipleFaceData = {0};
    ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Execute HFExecuteFaceTrack error: %d", ret);
        return ret;
    }

    // Print the number of faces detected
    auto faceNum = multipleFaceData.detectedNum;
    HFLogPrint(HF_LOG_INFO, "Num of face: %d", faceNum);

    // Copy a new image to draw
    HFImageBitmap drawImage = {0};
    ret = HFImageBitmapCopy(image, &drawImage);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Copy ImageBitmap error: %d", ret);
        return ret;
    }
    HFImageBitmapData data;
    ret = HFImageBitmapGetData(drawImage, &data);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Get ImageBitmap data error: %d", ret);
        return ret;
    }
    for (int index = 0; index < faceNum; ++index) {
        HFLogPrint(HF_LOG_INFO, "========================================");
        HFLogPrint(HF_LOG_INFO, "Token size: %d", multipleFaceData.tokens[index].size);
        HFLogPrint(HF_LOG_INFO, "Process face index: %d", index);
        HFLogPrint(HF_LOG_INFO, "DetConfidence: %f", multipleFaceData.detConfidence[index]);
        HFImageBitmapDrawRect(drawImage, multipleFaceData.rects[index], {0, 100, 255}, 4);

        // Print FaceID, In IMAGE-MODE it is changing, in VIDEO-MODE it is fixed, but it may be lost
        HFLogPrint(HF_LOG_INFO, "FaceID: %d", multipleFaceData.trackIds[index]);

        // Print Head euler angle, It can often be used to judge the quality of a face by the Angle
        // of the head
        HFLogPrint(HF_LOG_INFO, "Roll: %f, Yaw: %f, Pitch: %f", multipleFaceData.angles.roll[index], multipleFaceData.angles.yaw[index],
                   multipleFaceData.angles.pitch[index]);

        HInt32 numOfLmk;
        HFGetNumOfFaceDenseLandmark(&numOfLmk);
        HPoint2f denseLandmarkPoints[numOfLmk];
        ret = HFGetFaceDenseLandmarkFromFaceToken(multipleFaceData.tokens[index], denseLandmarkPoints, numOfLmk);
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "HFGetFaceDenseLandmarkFromFaceToken error!!");
            return -1;
        }
        for (size_t i = 0; i < numOfLmk; i++) {
            HFImageBitmapDrawCircleF(drawImage, {denseLandmarkPoints[i].x, denseLandmarkPoints[i].y}, 0, {100, 100, 0}, 2);
        }
        auto& rt = multipleFaceData.rects[index];
        float area = ((float)(rt.height * rt.width)) / (data.width * data.height);
        HFLogPrint(HF_LOG_INFO, "area: %f", area);

        HPoint2f fiveKeyPoints[5];
        ret = HFGetFaceFiveKeyPointsFromFaceToken(multipleFaceData.tokens[index], fiveKeyPoints, 5);
        if (ret != HSUCCEED) {
            HFLogPrint(HF_LOG_ERROR, "HFGetFaceFiveKeyPointsFromFaceToken error!!");
            return -1;
        }
        for (size_t i = 0; i < 5; i++) {
            HFImageBitmapDrawCircleF(drawImage, {fiveKeyPoints[i].x, fiveKeyPoints[i].y}, 0, {0, 0, 232}, 2);
        }
    }
    HFImageBitmapWriteToFile(drawImage, "draw_detected.jpg");
    HFLogPrint(HF_LOG_WARN, "Write to file success: %s", "draw_detected.jpg");

    // Run pipeline function
    // Select the pipeline function that you want to execute, provided that it is already enabled
    // when FaceContext is created!
    auto pipelineOption = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_LIVENESS;
    // In this loop, all faces are processed
    ret = HFMultipleFacePipelineProcessOptional(session, imageHandle, &multipleFaceData, pipelineOption);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Execute Pipeline error: %d", ret);
        return ret;
    }

    // Get mask detection results from the pipeline cache
    HFFaceMaskConfidence maskConfidence = {0};
    ret = HFGetFaceMaskConfidence(session, &maskConfidence);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Get mask detect result error: %d", ret);
        return -1;
    }

    // Get face quality results from the pipeline cache
    HFFaceQualityConfidence qualityConfidence = {0};
    ret = HFGetFaceQualityConfidence(session, &qualityConfidence);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Get face quality result error: %d", ret);
        return -1;
    }

    for (int index = 0; index < faceNum; ++index) {
        HFLogPrint(HF_LOG_INFO, "========================================");
        HFLogPrint(HF_LOG_INFO, "Process face index from pipeline: %d", index);
        HFLogPrint(HF_LOG_INFO, "Mask detect result: %f", maskConfidence.confidence[index]);
        HFLogPrint(HF_LOG_INFO, "Quality predict result: %f", qualityConfidence.confidence[index]);
        // We set the threshold of wearing a mask as 0.85. If it exceeds the threshold, it will be
        // judged as wearing a mask. The threshold can be adjusted according to the scene
        if (maskConfidence.confidence[index] > 0.85) {
            HFLogPrint(HF_LOG_INFO, "Mask");
        } else {
            HFLogPrint(HF_LOG_INFO, "Non Mask");
        }
    }

    ret = HFReleaseImageStream(imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release image stream error: %d", ret);
    }
    // The memory must be freed at the end of the program
    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release session error: %d", ret);
        return ret;
    }

    ret = HFReleaseImageBitmap(image);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release image bitmap error: %d", ret);
        return ret;
    }

    ret = HFReleaseImageBitmap(drawImage);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Release draw image bitmap error: %d", ret);
        return ret;
    }

    return 0;
}
