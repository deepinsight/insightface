//
// Created by tunm on 2023/10/3.
//
#include <iostream>
#include "inspireface/c_api/inspireface.h"
#include "opencv2/opencv.hpp"
#include "inspireface/log.h"

using namespace inspire;

std::string basename(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");  // Take into account the cross-platform separator
    if (lastSlash == std::string::npos) {
        return path;  // Without the slash, the entire path is the base name
    } else {
        return path.substr(lastSlash + 1);  // Returns the part after the last slash
    }
}

int compare() {
    HResult ret;
    // Initialize context
#ifdef ISF_ENABLE_RKNN
    HPath path = "test_res/pack/Gundam_RV1109";
#else
    HPath path = "test_res/pack/Pikachu";
#endif
    HF_ContextCustomParameter parameter = {0};
    parameter.enable_liveness = 1;
    parameter.enable_mask_detect = 1;
    parameter.enable_recognition = 1;
    parameter.enable_face_quality = 1;
    HF_DetectMode detMode = HF_DETECT_MODE_IMAGE;   // Selecting the image mode is always detection
    HContextHandle session;
    ret = HF_CreateFaceContextFromResourceFile(path, parameter, detMode, 3, &session);
    if (ret != HSUCCEED) {
        INSPIRE_LOGD("An error occurred while creating ctx: %ld", ret);
    }

    std::vector<std::string> names = {
            "/Users/tunm/datasets/lfw_funneled/Abel_Pacheco/Abel_Pacheco_0001.jpg",
            "/Users/tunm/datasets/lfw_funneled/Abel_Pacheco/Abel_Pacheco_0004.jpg",
    };
    HInt32 featureNum;
    HF_GetFeatureLength(&featureNum);
    INSPIRE_LOGD("Feature length: %d", featureNum);
    HFloat featuresCache[names.size()][featureNum];     // Store the cached vector

    for (int i = 0; i < names.size(); ++i) {
        auto &name = names[i];
        cv::Mat image = cv::imread(name);
        if (image.empty()) {
            INSPIRE_LOGD("%s is empty!", name.c_str());
            return -1;
        }
        HF_ImageData imageData = {0};
        imageData.data = image.data;
        imageData.height = image.rows;
        imageData.width = image.cols;
        imageData.rotation = CAMERA_ROTATION_0;
        imageData.format = STREAM_BGR;

        HImageHandle imageSteamHandle;
        ret = HF_CreateImageStream(&imageData, &imageSteamHandle);
        if (ret == HSUCCEED) {
            INSPIRE_LOGD("image handle: %ld", (long )imageSteamHandle);
        }

        HF_MultipleFaceData multipleFaceData = {0};
        HF_FaceContextRunFaceTrack(session, imageSteamHandle, &multipleFaceData);
        INSPIRE_LOGD("Number of faces detected: %d", multipleFaceData.detectedNum);

        for (int i = 0; i < multipleFaceData.detectedNum; ++i) {
            cv::Rect rect = cv::Rect(multipleFaceData.rects[i].x, multipleFaceData.rects[i].y, multipleFaceData.rects[i].width, multipleFaceData.rects[i].height);
            cv::rectangle(image, rect, cv::Scalar(0, 255, 200), 2);
            INSPIRE_LOGD("%d, track_id: %d, pitch: %f, yaw: %f, roll: %f", i, multipleFaceData.trackIds[i], multipleFaceData.angles.pitch[i], multipleFaceData.angles.yaw[i], multipleFaceData.angles.roll[i]);
            INSPIRE_LOGD("token size: %d", multipleFaceData.tokens->size);
        }
#ifndef DISABLE_GUI
//        cv::imshow("wq", image);
//        cv::waitKey(0);
#endif

        ret = HF_FaceFeatureExtractCpy(session, imageSteamHandle, multipleFaceData.tokens[0], featuresCache[i]);

        std::cout << "wtg" << std::endl;
        if (ret != HSUCCEED) {
            INSPIRE_LOGE("Abnormal feature extraction: %d", ret);
            return -1;
        }

//        for (int j = 0; j < 512; ++j) {
//            std::cout << featuresCache[0][j] << ", ";
//        }
//        std::cout << std::endl;

//        HSize size;
//        HF_GetFaceBasicTokenSize(&size);
//        LOGD("in size: %ld", size);
//
//        LOGD("o size %d", multipleFaceData.tokens[0].size);

        HBuffer buffer[multipleFaceData.tokens[0].size];
        HF_CopyFaceBasicToken(multipleFaceData.tokens[0], buffer, multipleFaceData.tokens[0].size);

        HF_FaceBasicToken token = {0};
        token.size = multipleFaceData.tokens[0].size;
        token.data = buffer;

        HFloat quality;
//        ret = HF_FaceQualityDetect(session, multipleFaceData.tokens[0], &quality);
        ret = HF_FaceQualityDetect(session, token, &quality);
        INSPIRE_LOGD("RET : %d", ret);
        INSPIRE_LOGD("Q: %f", quality);

        ret = HF_ReleaseImageStream(imageSteamHandle);
        if (ret == HSUCCEED) {
            imageSteamHandle = nullptr;
            INSPIRE_LOGD("image released");
        } else {
            INSPIRE_LOGE("image release error: %ld", ret);
        }

    }

    HFloat compResult;
    HF_FaceFeature compFeature1 = {0};
    HF_FaceFeature compFeature2 = {0};
    compFeature1.size = featureNum;
    compFeature1.data = featuresCache[0];
    compFeature2.size = featureNum;
    compFeature2.data = featuresCache[1];
    ret = HF_FaceComparison1v1(compFeature1, compFeature2, &compResult);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Contrast failure: %d", ret);
        return -1;
    }
    INSPIRE_LOGD("similarity: %f", compResult);

    ret = HF_ReleaseFaceContext(session);
    if (ret != HSUCCEED) {
        INSPIRE_LOGD("Release error");
    }

    return 0;
}

int search() {
    HResult ret;
    // 初始化context
    HString path = "test_res/pack/Pikachu";
    HF_ContextCustomParameter parameter = {0};
    parameter.enable_liveness = 1;
    parameter.enable_mask_detect = 1;
    parameter.enable_recognition = 1;
    HF_DetectMode detMode = HF_DETECT_MODE_IMAGE;
    HContextHandle session;
    ret = HF_CreateFaceContextFromResourceFile(path, parameter, detMode, 3, &session);
    if (ret != HSUCCEED) {
        INSPIRE_LOGD("An error occurred while creating ctx: %ld", ret);
    }
    HF_FeatureHubConfiguration databaseConfiguration = {0};
    databaseConfiguration.enablePersistence = 1;
    databaseConfiguration.dbPath = "./";
    ret = HF_FeatureHubDataEnable(databaseConfiguration);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Database configuration failure: %ld", ret);
        return -1;
    }

    std::vector<std::string> files_list = {

    };

    for (int i = 0; i < files_list.size(); ++i) {
        auto &name = files_list[i];
        cv::Mat image = cv::imread(name);
        HF_ImageData imageData = {0};
        imageData.data = image.data;
        imageData.height = image.rows;
        imageData.width = image.cols;
        imageData.rotation = CAMERA_ROTATION_0;
        imageData.format = STREAM_BGR;

        HImageHandle imageSteamHandle;
        ret = HF_CreateImageStream(&imageData, &imageSteamHandle);
        if (ret != HSUCCEED) {
            INSPIRE_LOGE("image handle error: %ld", (long )imageSteamHandle);
            return -1;
        }

        HF_MultipleFaceData multipleFaceData = {0};
        HF_FaceContextRunFaceTrack(session, imageSteamHandle, &multipleFaceData);

        if (multipleFaceData.detectedNum <= 0) {
            INSPIRE_LOGE("%s No face detected", name.c_str());
            return -1;
        }

        HF_FaceFeature feature = {0};
        ret = HF_FaceFeatureExtract(session, imageSteamHandle, multipleFaceData.tokens[0], &feature);
        if (ret != HSUCCEED) {
            INSPIRE_LOGE("Feature extraction error: %ld", ret);
            return -1;
        }

        auto tag = basename(name);
        char *tagName = new char[tag.size() + 1];
        std::strcpy(tagName, tag.c_str());
        HF_FaceFeatureIdentity identity = {0};
        identity.feature = &feature;
        identity.customId = i;
        identity.tag = tagName;

        ret = HF_FeatureHubInsertFeature(identity);
        if (ret != HSUCCEED) {
            INSPIRE_LOGE("插入失败: %ld", ret);
            return -1;
        }


//        // 在插入一次测试一下重复操作问题
//        ret = HF_FeaturesGroupInsertFeature(session, identity);
//        if (ret != HSUCCEED) {
//            LOGE("不能重复id插入: %ld", ret);
//        }

        delete[] tagName;

        ret = HF_ReleaseImageStream(imageSteamHandle);
        if (ret == HSUCCEED) {
            imageSteamHandle = nullptr;
            INSPIRE_LOGD("image released");
        } else {
            INSPIRE_LOGE("image release error: %ld", ret);
        }
    }

    cv::Mat image = cv::imread("test_res/images/kun.jpg");
    HF_ImageData imageData = {0};
    imageData.data = image.data;
    imageData.height = image.rows;
    imageData.width = image.cols;
    imageData.rotation = CAMERA_ROTATION_0;
    imageData.format = STREAM_BGR;

    HImageHandle imageSteamHandle;
    ret = HF_CreateImageStream(&imageData, &imageSteamHandle);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("image handle error: %ld", (long )imageSteamHandle);
        return -1;
    }
    HF_MultipleFaceData multipleFaceData = {0};
    HF_FaceContextRunFaceTrack(session, imageSteamHandle, &multipleFaceData);

    if (multipleFaceData.detectedNum <= 0) {
        INSPIRE_LOGE("No face detected");
        return -1;
    }

    HF_FaceFeature feature = {0};
    ret = HF_FaceFeatureExtract(session, imageSteamHandle, multipleFaceData.tokens[0], &feature);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Feature extraction error: %ld", ret);
        return -1;
    }

//    ret = HF_FaceContextFeatureRemove(session, 3);
//    if (ret != HSUCCEED) {
//        LOGE("delete failed: %ld", ret);
//    }

    std::string newName = "Six";
    char *newTagName = new char[newName.size() + 1];
    std::strcpy(newTagName, newName.c_str());
    HF_FaceFeatureIdentity updateIdentity = {0};
    updateIdentity.customId = 1;
    updateIdentity.tag = newTagName;
    updateIdentity.feature = &feature;
    ret = HF_FeatureHubFaceUpdate(updateIdentity);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Update failure: %ld", ret);
    }
    delete[] newTagName;


    HF_FaceFeatureIdentity searchIdentity = {0};
//    HF_FaceFeature featureSearched = {0};
//    searchIdentity.feature = &featureSearched;
    HFloat confidence;
    ret = HF_FeatureHubFaceSearch(feature, &confidence, &searchIdentity);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Search failure: %ld", ret);
        return -1;
    }

    INSPIRE_LOGD("Search for confidence: %f", confidence);
    INSPIRE_LOGD("The matched tag: %s", searchIdentity.tag);
    INSPIRE_LOGD("The matched customId: %d", searchIdentity.customId);


    // Face Pipeline
    ret = HF_MultipleFacePipelineProcess(session, imageSteamHandle, &multipleFaceData, parameter);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("pipeline execution failed: %ld", ret);
        return -1;
    }

    HF_RGBLivenessConfidence livenessConfidence = {0};
    ret = HF_GetRGBLivenessConfidence(session, &livenessConfidence);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Failed to obtain live data");
        return -1;
    }
    INSPIRE_LOGD("Failed to obtain live data: %f", livenessConfidence.confidence[0]);

    HF_FaceMaskConfidence maskConfidence = {0};
    ret = HF_GetFaceMaskConfidence(session, &maskConfidence);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Failed to obtain live data");
        return -1;
    }
    INSPIRE_LOGD("Mask wearing confidence: %f", maskConfidence.confidence[0]);

    HInt32 faceNum;
    ret = HF_FeatureHubGetFaceCount(&faceNum);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("fail to get");
    }
    INSPIRE_LOGD("Number of facial features: %d", faceNum);

    HF_FeatureHubViewDBTable();


    HF_FaceFeatureIdentity identity;
    ret = HF_FeatureHubGetFaceIdentity(100, &identity);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Feature acquisition failure");
    }

    ret = HF_ReleaseImageStream(imageSteamHandle);
    if (ret == HSUCCEED) {
        imageSteamHandle = nullptr;
        INSPIRE_LOGD("image released");
    } else {
        INSPIRE_LOGE("image release error: %ld", ret);
    }

    return 0;
}

int opiton() {
//    HInt32 mask = HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_LIVENESS;

    return 0;
}

int main() {

    HResult ret;

//    {
//        // 测试ImageStream
//        cv::Mat image = cv::imread("test_res/images/kun.jpg");
//        HF_ImageData imageData = {0};
//        imageData.data = image.data;
//        imageData.height = image.rows;
//        imageData.width = image.cols;
//        imageData.rotation = CAMERA_ROTATION_0;
//        imageData.format = STREAM_BGR;
//
//        HImageHandle imageSteamHandle;
//        ret = HF_CreateImageStream(&imageData, &imageSteamHandle);
//        if (ret == HSUCCEED) {
//            LOGD("image handle: %ld", (long )imageSteamHandle);
//        }
//        HF_DeBugImageStreamImShow(imageSteamHandle);
//
//        ret = HF_ReleaseImageStream(imageSteamHandle);
//        if (ret == HSUCCEED) {
//            imageSteamHandle = nullptr;
//            LOGD("image released");
//        } else {
//            LOGE("image release error: %ld", ret);
//        }
//
//    }


//    compare();

    search();


    opiton();
}