/**
 * Created by Jingyu Yan
 * @date 2024-11-26
 */

#ifdef ANDROID

#include <jni.h>
#include <string>
#include <android/log.h>
#include <stdlib.h>
#include <android/bitmap.h>
#include "../common/common.h"
#include "c_api/inspireface.h"
#include "log.h"
#include "herror.h"

/**
 * Java Native Interface (JNI) macro for generating JNI function names.
 */
#define INSPIRE_FACE_JNI(sig) Java_com_insightface_sdk_inspireface_##sig

extern "C" {

/**
 * @brief Launch InspireFace.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param resourcePath The resource path.
 * @return True if the launch is successful, false otherwise.
 */
JNIEXPORT jboolean INSPIRE_FACE_JNI(InspireFace_GlobalLaunch)(JNIEnv *env, jobject thiz, jstring resourcePath) {
    std::string path = jstring2str(env, resourcePath);
    auto result = HFLaunchInspireFace(path.c_str());
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to launch InspireFace, error code: %d", result);
        return false;
    }
    return true;
}

/**
 * @brief Terminate InspireFace.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @return True if the termination is successful, false otherwise.
 */
JNIEXPORT jboolean INSPIRE_FACE_JNI(InspireFace_GlobalTerminate)(JNIEnv *env, jobject thiz) {
    auto result = HFTerminateInspireFace();
    if (result != 0) {
        INSPIRE_LOGE("Failed to terminate InspireFace, error code: %d", result);
        return false;
    }
    return true;
}

/**
 * @brief Query InspireFace launch status.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @return True if the launch status is successful, false otherwise.
 */
JNIEXPORT jboolean INSPIRE_FACE_JNI(InspireFace_QueryLaunchStatus)(JNIEnv *env, jobject thiz) {
    HInt32 status;
    auto result = HFQueryInspireFaceLaunchStatus(&status);
    if (result != 0) {
        INSPIRE_LOGE("Failed to query InspireFace launch status, error code: %d", result);
        return false;
    }
    return status;
}

/**
 * @brief Create a new InspireFace session.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param customParameter The custom parameter object.
 * @param detectMode The detection mode.
 * @param maxDetectFaceNum The maximum number of detected faces.
 * @param detectPixelLevel The detection pixel level.
 * @param trackByDetectModeFPS The tracking frame rate.
 * @return The new InspireFace session object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_CreateSession)(JNIEnv *env, jobject thiz, jobject customParameter, jint detectMode,
                                                              jint maxDetectFaceNum, jint detectPixelLevel, jint trackByDetectModeFPS) {
    // Get CustomParameter class and fields
    jclass customParamClass = env->GetObjectClass(customParameter);
    jfieldID enableRecognitionField = env->GetFieldID(customParamClass, "enableRecognition", "I");
    jfieldID enableLivenessField = env->GetFieldID(customParamClass, "enableLiveness", "I");
    jfieldID enableIrLivenessField = env->GetFieldID(customParamClass, "enableIrLiveness", "I");
    jfieldID enableMaskDetectField = env->GetFieldID(customParamClass, "enableMaskDetect", "I");
    jfieldID enableFaceQualityField = env->GetFieldID(customParamClass, "enableFaceQuality", "I");
    jfieldID enableFaceAttributeField = env->GetFieldID(customParamClass, "enableFaceAttribute", "I");
    jfieldID enableInteractionLivenessField = env->GetFieldID(customParamClass, "enableInteractionLiveness", "I");
    jfieldID enableFacePoseField = env->GetFieldID(customParamClass, "enableFacePose", "I");
    jfieldID enableFaceEmotionField = env->GetFieldID(customParamClass, "enableFaceEmotion", "I");

    // Create HFSessionCustomParameter struct
    HFSessionCustomParameter parameter;
    parameter.enable_recognition = env->GetIntField(customParameter, enableRecognitionField);
    parameter.enable_liveness = env->GetIntField(customParameter, enableLivenessField);
    parameter.enable_ir_liveness = env->GetIntField(customParameter, enableIrLivenessField);
    parameter.enable_mask_detect = env->GetIntField(customParameter, enableMaskDetectField);
    parameter.enable_face_quality = env->GetIntField(customParameter, enableFaceQualityField);
    parameter.enable_face_attribute = env->GetIntField(customParameter, enableFaceAttributeField);
    parameter.enable_interaction_liveness = env->GetIntField(customParameter, enableInteractionLivenessField);
    parameter.enable_face_pose = env->GetIntField(customParameter, enableFacePoseField);
    parameter.enable_face_emotion = env->GetIntField(customParameter, enableFaceEmotionField);

    // Create session
    HFSession handle;
    HResult result =
      HFCreateInspireFaceSession(parameter, (HFDetectMode)detectMode, maxDetectFaceNum, detectPixelLevel, trackByDetectModeFPS, &handle);

    if (result != 0) {
        INSPIRE_LOGE("Failed to create session, error code: %d", result);
        return nullptr;
    }

    // Create Session object
    jclass sessionClass = env->FindClass("com/insightface/sdk/inspireface/base/Session");
    jmethodID constructor = env->GetMethodID(sessionClass, "<init>", "()V");
    jobject session = env->NewObject(sessionClass, constructor);

    // Set handle
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    env->SetLongField(session, handleField, (jlong)handle);

    return session;
}

/**
 * @brief Release an InspireFace session.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The InspireFace session object.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_ReleaseSession)(JNIEnv *env, jobject thiz, jobject session) {
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    HFSession handle = (HFSession)env->GetLongField(session, handleField);
    auto result = HFReleaseInspireFaceSession(handle);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to release session, error code: %d", result);
    }
}

/**
 * @brief Create an InspireFace image stream from a bitmap.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param bitmap The bitmap object.
 * @param rotation The rotation.
 * @return The InspireFace image stream object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_CreateImageStreamFromBitmap)(JNIEnv *env, jobject thiz, jobject bitmap, jint rotation) {
    AndroidBitmapInfo info;
    void *pixels = nullptr;
    HFImageFormat format = HF_STREAM_RGB;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
        INSPIRE_LOGE("Failed to get bitmap info");
        return nullptr;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        AndroidBitmap_unlockPixels(env, bitmap);
        INSPIRE_LOGE("Failed to lock pixels");
        return nullptr;
    }
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        format = HF_STREAM_RGBA;
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        format = HF_STREAM_RGB;
    } else {
        AndroidBitmap_unlockPixels(env, bitmap);
        INSPIRE_LOGE("Unsupported bitmap format: %d", info.format);
        return nullptr;
    }

    HFImageData imageData;
    imageData.data = (uint8_t *)pixels;
    imageData.width = info.width;
    imageData.height = info.height;
    imageData.format = (HFImageFormat)format;
    imageData.rotation = (HFRotation)rotation;

    HFImageStream streamHandle;
    auto result = HFCreateImageStream(&imageData, &streamHandle);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to create image stream, error code: %d", result);
        return nullptr;
    }
    AndroidBitmap_unlockPixels(env, bitmap);

    jclass streamClass = env->FindClass("com/insightface/sdk/inspireface/base/ImageStream");
    jmethodID constructor = env->GetMethodID(streamClass, "<init>", "()V");
    jfieldID streamHandleField = env->GetFieldID(streamClass, "handle", "J");
    jobject imageStreamObj = env->NewObject(streamClass, constructor);
    env->SetLongField(imageStreamObj, streamHandleField, (jlong)streamHandle);

    return imageStreamObj;
}

/**
 * @brief Create an InspireFace image stream from a byte buffer.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param data The byte buffer.
 * @param width The width.
 * @param height The height.
 * @param format The format.
 * @param rotation The rotation.
 * @return The InspireFace image stream object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_CreateImageStreamFromByteBuffer)(JNIEnv *env, jobject thiz, jbyteArray data, jint width, jint height,
                                                                                jint format, jint rotation) {
    // Convert jbyteArray to byte*
    uint8_t *buf = (uint8_t *)env->GetByteArrayElements(data, 0);
    HFImageData imageData;
    imageData.data = buf;
    imageData.width = width;
    imageData.height = height;
    imageData.format = (HFImageFormat)format;
    imageData.rotation = (HFRotation)rotation;

    HFImageStream streamHandle;
    auto result = HFCreateImageStream(&imageData, &streamHandle);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to create image stream, error code: %d", result);
        return nullptr;
    }

    jclass streamClass = env->FindClass("com/insightface/sdk/inspireface/base/ImageStream");
    jmethodID constructor = env->GetMethodID(streamClass, "<init>", "()V");
    jfieldID streamHandleField = env->GetFieldID(streamClass, "handle", "J");
    jobject imageStreamObj = env->NewObject(streamClass, constructor);
    env->SetLongField(imageStreamObj, streamHandleField, (jlong)streamHandle);
    env->ReleaseByteArrayElements(data, (jbyte *)buf, JNI_ABORT);

    return imageStreamObj;
}

/**
 * @brief Write an InspireFace image stream to a file.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param imageStream The InspireFace image stream object.
 * @param filePath The file path.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_WriteImageStreamToFile)(JNIEnv *env, jobject thiz, jobject imageStream, jstring filePath) {
    jclass streamClass = env->GetObjectClass(imageStream);
    jfieldID streamHandleField = env->GetFieldID(streamClass, "handle", "J");
    HFImageStream streamHandle = (HFImageStream)env->GetLongField(imageStream, streamHandleField);

    std::string path = jstring2str(env, filePath);
    HFDeBugImageStreamDecodeSave(streamHandle, path.c_str());
}

/**
 * @brief Release an InspireFace image stream.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param imageStream The InspireFace image stream object.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_ReleaseImageStream)(JNIEnv *env, jobject thiz, jobject imageStream) {
    jclass streamClass = env->GetObjectClass(imageStream);
    jfieldID streamHandleField = env->GetFieldID(streamClass, "handle", "J");
    HFImageStream streamHandle = (HFImageStream)env->GetLongField(imageStream, streamHandleField);
    auto result = HFReleaseImageStream(streamHandle);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to release image stream, error code: %d", result);
    }
}

/**
 * @brief Execute face track.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The InspireFace session object.
 * @param streamHandle The InspireFace image stream object.
 * @return The face track results.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_ExecuteFaceTrack)(JNIEnv *env, jobject thiz, jobject session, jobject streamHandle) {
    // Get session handle
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    HFSession sessionHandle = (HFSession)env->GetLongField(session, handleField);

    // Get stream handle
    jclass streamClass = env->GetObjectClass(streamHandle);
    jfieldID streamHandleField = env->GetFieldID(streamClass, "handle", "J");
    HFImageStream imageStreamHandle = (HFImageStream)env->GetLongField(streamHandle, streamHandleField);

    // Execute face track
    HFMultipleFaceData results;
    HResult result = HFExecuteFaceTrack(sessionHandle, imageStreamHandle, &results);
    if (result != 0) {
        INSPIRE_LOGE("Failed to execute face track, error code: %d", result);
        return nullptr;
    }

    // Create MultipleFaceData object
    jclass multipleFaceDataClass = env->FindClass("com/insightface/sdk/inspireface/base/MultipleFaceData");
    jmethodID constructor = env->GetMethodID(multipleFaceDataClass, "<init>", "()V");
    jobject multipleFaceData = env->NewObject(multipleFaceDataClass, constructor);

    // Set detected number
    jfieldID detectedNumField = env->GetFieldID(multipleFaceDataClass, "detectedNum", "I");
    env->SetIntField(multipleFaceData, detectedNumField, results.detectedNum);

    if (results.detectedNum > 0) {
        // Create and set face rects array
        jclass faceRectClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceRect");
        jobjectArray rectArray = env->NewObjectArray(results.detectedNum, faceRectClass, nullptr);
        jmethodID rectConstructor = env->GetMethodID(faceRectClass, "<init>", "()V");
        jfieldID xField = env->GetFieldID(faceRectClass, "x", "I");
        jfieldID yField = env->GetFieldID(faceRectClass, "y", "I");
        jfieldID widthField = env->GetFieldID(faceRectClass, "width", "I");
        jfieldID heightField = env->GetFieldID(faceRectClass, "height", "I");

        // Create and set track IDs array
        jintArray trackIdsArray = env->NewIntArray(results.detectedNum);
        env->SetIntArrayRegion(trackIdsArray, 0, results.detectedNum, results.trackIds);

        // Create and set detection confidence array
        jfloatArray detConfArray = env->NewFloatArray(results.detectedNum);
        env->SetFloatArrayRegion(detConfArray, 0, results.detectedNum, results.detConfidence);

        // Create and set angles array
        jclass angleClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceEulerAngle");
        jobjectArray angleArray = env->NewObjectArray(results.detectedNum, angleClass, nullptr);
        jmethodID angleConstructor = env->GetMethodID(angleClass, "<init>", "()V");
        jfieldID rollField = env->GetFieldID(angleClass, "roll", "F");
        jfieldID yawField = env->GetFieldID(angleClass, "yaw", "F");
        jfieldID pitchField = env->GetFieldID(angleClass, "pitch", "F");

        // Create and set tokens array
        jclass tokenClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceBasicToken");
        jobjectArray tokenArray = env->NewObjectArray(results.detectedNum, tokenClass, nullptr);
        jmethodID tokenConstructor = env->GetMethodID(tokenClass, "<init>", "()V");
        jfieldID tokenDataField = env->GetFieldID(tokenClass, "data", "[B");
        jfieldID sizeField = env->GetFieldID(tokenClass, "size", "I");

        // Get token size first
        HInt32 tokenSize = 0;
        HFGetFaceBasicTokenSize(&tokenSize);

        for (int i = 0; i < results.detectedNum; i++) {
            // Set face rect
            jobject rect = env->NewObject(faceRectClass, rectConstructor);
            env->SetIntField(rect, xField, results.rects[i].x);
            env->SetIntField(rect, yField, results.rects[i].y);
            env->SetIntField(rect, widthField, results.rects[i].width);
            env->SetIntField(rect, heightField, results.rects[i].height);
            env->SetObjectArrayElement(rectArray, i, rect);

            // Set angle
            jobject angle = env->NewObject(angleClass, angleConstructor);
            env->SetFloatField(angle, rollField, *results.angles.roll);
            env->SetFloatField(angle, yawField, *results.angles.yaw);
            env->SetFloatField(angle, pitchField, *results.angles.pitch);
            env->SetObjectArrayElement(angleArray, i, angle);

            // Create token object
            jobject token = env->NewObject(tokenClass, tokenConstructor);
            
            // Create byte array to hold token data
            jbyteArray dataArray = env->NewByteArray(tokenSize);
            jbyte* buffer = env->GetByteArrayElements(dataArray, nullptr);
            
            // Copy token data using HFCopyFaceBasicToken
            HResult copyResult = HFCopyFaceBasicToken(results.tokens[i], reinterpret_cast<char*>(buffer), tokenSize);
            if (copyResult == HSUCCEED) {
                // Set data and size fields
                env->SetObjectField(token, tokenDataField, dataArray);
                env->SetIntField(token, sizeField, tokenSize);
                env->ReleaseByteArrayElements(dataArray, buffer, 0);
            } else {
                INSPIRE_LOGE("Failed to copy token data for face %d, error code: %d", i, copyResult);
                env->ReleaseByteArrayElements(dataArray, buffer, JNI_ABORT);
                env->DeleteLocalRef(dataArray);
            }
            env->SetObjectArrayElement(tokenArray, i, token);

            // Release local references
            env->DeleteLocalRef(rect);
            env->DeleteLocalRef(angle);
            env->DeleteLocalRef(token);
        }

        // Set arrays to MultipleFaceData
        jfieldID rectsField = env->GetFieldID(multipleFaceDataClass, "rects", "[Lcom/insightface/sdk/inspireface/base/FaceRect;");
        jfieldID trackIdsField = env->GetFieldID(multipleFaceDataClass, "trackIds", "[I");
        jfieldID detConfidenceField = env->GetFieldID(multipleFaceDataClass, "detConfidence", "[F");
        jfieldID anglesField = env->GetFieldID(multipleFaceDataClass, "angles", "[Lcom/insightface/sdk/inspireface/base/FaceEulerAngle;");
        jfieldID tokensField = env->GetFieldID(multipleFaceDataClass, "tokens", "[Lcom/insightface/sdk/inspireface/base/FaceBasicToken;");

        env->SetObjectField(multipleFaceData, rectsField, rectArray);
        env->SetObjectField(multipleFaceData, trackIdsField, trackIdsArray);
        env->SetObjectField(multipleFaceData, detConfidenceField, detConfArray);
        env->SetObjectField(multipleFaceData, anglesField, angleArray);
        env->SetObjectField(multipleFaceData, tokensField, tokenArray);
    }

    return multipleFaceData;
}

/**
 * @brief Get face dense landmark from face token.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param token The face token object.
 * @return The face dense landmark array.
 */
JNIEXPORT jobjectArray INSPIRE_FACE_JNI(InspireFace_GetFaceDenseLandmarkFromFaceToken)(JNIEnv *env, jobject thiz, jobject token) {
    // Get token data and size from FaceBasicToken object
    jclass tokenClass = env->GetObjectClass(token);
    jfieldID tokenDataField = env->GetFieldID(tokenClass, "data", "[B");
    jfieldID tokenSizeField = env->GetFieldID(tokenClass, "size", "I");
    jbyteArray tokenDataArray = (jbyteArray)env->GetObjectField(token, tokenDataField);
    jint tokenSize = env->GetIntField(token, tokenSizeField);
    
    if (tokenDataArray == nullptr) {
        INSPIRE_LOGE("Token data array is null");
        return nullptr;
    }

    // Get number of landmarks
    int32_t numLandmarks = 0;
    HFGetNumOfFaceDenseLandmark(&numLandmarks);

    // Allocate memory for landmarks
    HPoint2f *landmarks = new HPoint2f[numLandmarks];

    // Create face token struct from byte array data
    HFFaceBasicToken faceToken;
    faceToken.size = tokenSize;
    faceToken.data = env->GetByteArrayElements(tokenDataArray, nullptr);

    // Get landmarks from token
    HResult result = HFGetFaceDenseLandmarkFromFaceToken(faceToken, landmarks, numLandmarks);

    // Release byte array elements
    env->ReleaseByteArrayElements(tokenDataArray, (jbyte*)faceToken.data, JNI_ABORT);

    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to get face dense landmark from face token, error code: %d", result);
        delete[] landmarks;
        return nullptr;
    }

    // Create Point2f array to return
    jclass point2fClass = env->FindClass("com/insightface/sdk/inspireface/base/Point2f");
    jobjectArray landmarkArray = env->NewObjectArray(numLandmarks, point2fClass, nullptr);
    jmethodID constructor = env->GetMethodID(point2fClass, "<init>", "()V");
    jfieldID xField = env->GetFieldID(point2fClass, "x", "F");
    jfieldID yField = env->GetFieldID(point2fClass, "y", "F");

    // Fill array with landmark points
    for (int i = 0; i < numLandmarks; i++) {
        jobject point = env->NewObject(point2fClass, constructor);
        env->SetFloatField(point, xField, landmarks[i].x);
        env->SetFloatField(point, yField, landmarks[i].y);
        env->SetObjectArrayElement(landmarkArray, i, point);
    }

    delete[] landmarks;
    return landmarkArray;
}

/**
 * @brief Extract face feature from a face token.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The InspireFace session object.
 * @param streamHandle The InspireFace image stream object.
 * @param token The face token object.
 * @return The face feature object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_ExtractFaceFeature)(JNIEnv *env, jobject thiz, jobject session, jobject streamHandle, jobject token) {
    // Get session handle
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID sessionHandleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, sessionHandleField);

    // Get stream handle
    jclass streamClass = env->GetObjectClass(streamHandle);
    jfieldID streamHandleField = env->GetFieldID(streamClass, "handle", "J");
    jlong streamHandleValue = env->GetLongField(streamHandle, streamHandleField);

    // Get token data and size from FaceBasicToken object
    jclass tokenClass = env->GetObjectClass(token);
    jfieldID tokenDataField = env->GetFieldID(tokenClass, "data", "[B");
    jfieldID tokenSizeField = env->GetFieldID(tokenClass, "size", "I");
    jbyteArray tokenDataArray = (jbyteArray)env->GetObjectField(token, tokenDataField);
    jint tokenSize = env->GetIntField(token, tokenSizeField);
    
    if (tokenDataArray == nullptr) {
        INSPIRE_LOGE("Token data array is null");
        return nullptr;
    }

    // Create face token struct from byte array data
    HFFaceBasicToken faceToken;
    faceToken.size = tokenSize;
    faceToken.data = env->GetByteArrayElements(tokenDataArray, nullptr);

    // Extract face feature
    HFFaceFeature feature;
    HResult result =
      HFFaceFeatureExtract(reinterpret_cast<HFSession>(sessionHandle), reinterpret_cast<HFImageStream>(streamHandleValue), faceToken, &feature);

    // Release byte array elements
    env->ReleaseByteArrayElements(tokenDataArray, (jbyte*)faceToken.data, JNI_ABORT);

    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to extract face feature, error code: %d", result);
        return nullptr;
    }

    // Create FaceFeature object to return
    jclass faceFeatureClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceFeature");
    jobject faceFeature = env->NewObject(faceFeatureClass, env->GetMethodID(faceFeatureClass, "<init>", "()V"));

    // Create float array and copy feature data
    jfloatArray dataArray = env->NewFloatArray(feature.size);
    env->SetFloatArrayRegion(dataArray, 0, feature.size, reinterpret_cast<jfloat *>(feature.data));

    // Set data field in FaceFeature object
    jfieldID dataField = env->GetFieldID(faceFeatureClass, "data", "[F");
    env->SetObjectField(faceFeature, dataField, dataArray);

    return faceFeature;
}

/**
 * @brief Get face alignment image.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The InspireFace session object.
 * @param streamHandle The InspireFace image stream object.
 * @param token The face token object.
 * @return The face alignment image object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_GetFaceAlignmentImage)(JNIEnv *env, jobject thiz, jobject session, jobject streamHandle,
                                                                      jobject token) {
    // Get session handle
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID sessionHandleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, sessionHandleField);

    // Get stream handle
    jclass streamClass = env->GetObjectClass(streamHandle);
    jfieldID streamHandleField = env->GetFieldID(streamClass, "handle", "J");
    jlong streamHandleValue = env->GetLongField(streamHandle, streamHandleField);

    // Get token data and size from FaceBasicToken object
    jclass tokenClass = env->GetObjectClass(token);
    jfieldID tokenDataField = env->GetFieldID(tokenClass, "data", "[B");
    jfieldID tokenSizeField = env->GetFieldID(tokenClass, "size", "I");
    jbyteArray tokenDataArray = (jbyteArray)env->GetObjectField(token, tokenDataField);
    jint tokenSize = env->GetIntField(token, tokenSizeField);
    
    if (tokenDataArray == nullptr) {
        INSPIRE_LOGE("Token data array is null");
        return nullptr;
    }

    // Create face token struct from byte array data
    HFFaceBasicToken faceToken;
    faceToken.size = tokenSize;
    faceToken.data = env->GetByteArrayElements(tokenDataArray, nullptr);

    // Get face alignment image
    HFImageBitmap imageBitmap;
    HResult result = HFFaceGetFaceAlignmentImage(reinterpret_cast<HFSession>(sessionHandle), reinterpret_cast<HFImageStream>(streamHandleValue),
                                                 faceToken, &imageBitmap);

    // Release byte array elements
    env->ReleaseByteArrayElements(tokenDataArray, (jbyte*)faceToken.data, JNI_ABORT);

    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to get face alignment image, error code: %d", result);
        return nullptr;
    }

    // Get image bitmap data
    HFImageBitmapData bitmapData;
    result = HFImageBitmapGetData(imageBitmap, &bitmapData);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to get image bitmap data, error code: %d", result);
        HFReleaseImageBitmap(imageBitmap);
        return nullptr;
    }

    // Create Android Bitmap
    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapMethod =
      env->GetStaticMethodID(bitmapClass, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");

    // Get Bitmap.Config.ARGB_8888
    jclass bitmapConfigClass = env->FindClass("android/graphics/Bitmap$Config");
    jfieldID argb8888FieldID = env->GetStaticFieldID(bitmapConfigClass, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    jobject argb8888Obj = env->GetStaticObjectField(bitmapConfigClass, argb8888FieldID);

    // Create bitmap with alignment image dimensions
    jobject bitmap = env->CallStaticObjectMethod(bitmapClass, createBitmapMethod, bitmapData.width, bitmapData.height, argb8888Obj);

    // Copy pixels to bitmap
    AndroidBitmapInfo bitmapInfo;
    void *pixels;
    if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) < 0 || AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        INSPIRE_LOGE("Failed to get bitmap info or lock pixels, error code: %d", result);
        HFReleaseImageBitmap(imageBitmap);
        return nullptr;
    }

    // Convert BGR to ARGB
    uint8_t *src = (uint8_t *)bitmapData.data;
    uint32_t *dst = (uint32_t *)pixels;
    for (int i = 0; i < bitmapData.width * bitmapData.height; i++) {
        // BGR to ARGB conversion
        uint8_t r = src[i * 3 + 2];
        uint8_t g = src[i * 3 + 1];
        uint8_t b = src[i * 3];
        dst[i] = (0xFF << 24) | (b << 16) | (g << 8) | r;
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    HFReleaseImageBitmap(imageBitmap);

    return bitmap;
}

/**
 * @brief Set track preview size.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The InspireFace session object.
 * @param previewSize The preview size.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_SetTrackPreviewSize)(JNIEnv *env, jobject thiz, jobject session, jint previewSize) {
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID sessionHandleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, sessionHandleField);
    auto result = HFSessionSetTrackPreviewSize((HFSession)sessionHandle, previewSize);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to set track preview size, error code: %d", result);
    }
}

/**
 * @brief Set filter minimum face pixel size.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The InspireFace session object.
 * @param minSize The minimum face pixel size.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_SetFilterMinimumFacePixelSize)(JNIEnv *env, jobject thiz, jobject session, jint minSize) {
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID sessionHandleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, sessionHandleField);
    auto result = HFSessionSetFilterMinimumFacePixelSize((HFSession)sessionHandle, minSize);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to set filter minimum face pixel size, error code: %d", result);
    }
}

/**
 * @brief Set face detect threshold.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The InspireFace session object.
 * @param threshold The face detect threshold.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_SetFaceDetectThreshold)(JNIEnv *env, jobject thiz, jobject session, jfloat threshold) {
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID sessionHandleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, sessionHandleField);
    auto result = HFSessionSetFaceDetectThreshold((HFSession)sessionHandle, threshold);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to set face detect threshold, error code: %d", result);
    }
}

/**
 * @brief Set track mode smooth ratio.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The InspireFace session object.
 * @param ratio The track mode smooth ratio.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_SetTrackModeSmoothRatio)(JNIEnv *env, jobject thiz, jobject session, jfloat ratio) {
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID sessionHandleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, sessionHandleField);
    auto result = HFSessionSetTrackModeSmoothRatio((HFSession)sessionHandle, ratio);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to set track mode smooth ratio, error code: %d", result);
    }
}

/**
 * @brief Set track mode num smooth cache frame.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The InspireFace session object.
 * @param num The track mode num smooth cache frame.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_SetTrackModeNumSmoothCacheFrame)(JNIEnv *env, jobject thiz, jobject session, jint num) {
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID sessionHandleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, sessionHandleField);
    auto result = HFSessionSetTrackModeNumSmoothCacheFrame((HFSession)sessionHandle, num);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to set track mode num smooth cache frame, error code: %d", result);
    }
}

/**
 * @brief Set track mode detect interval.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The InspireFace session object.
 * @param interval The track mode detect interval.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_SetTrackModeDetectInterval)(JNIEnv *env, jobject thiz, jobject session, jint interval) {
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID sessionHandleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, sessionHandleField);
    auto result = HFSessionSetTrackModeDetectInterval((HFSession)sessionHandle, interval);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to set track mode detect interval, error code: %d", result);
    }
}

/**
 * @brief Enable feature hub data.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param configuration The configuration object.
 */
JNIEXPORT jboolean INSPIRE_FACE_JNI(InspireFace_FeatureHubDataEnable)(JNIEnv *env, jobject thiz, jobject configuration) {
    jclass configClass = env->GetObjectClass(configuration);

    jfieldID primaryKeyModeField = env->GetFieldID(configClass, "primaryKeyMode", "I");
    jfieldID enablePersistenceField = env->GetFieldID(configClass, "enablePersistence", "I");
    jfieldID persistenceDbPathField = env->GetFieldID(configClass, "persistenceDbPath", "Ljava/lang/String;");
    jfieldID searchThresholdField = env->GetFieldID(configClass, "searchThreshold", "F");
    jfieldID searchModeField = env->GetFieldID(configClass, "searchMode", "I");
    HFFeatureHubConfiguration config;
    config.primaryKeyMode = (HFPKMode)env->GetIntField(configuration, primaryKeyModeField);
    config.enablePersistence = env->GetIntField(configuration, enablePersistenceField);

    // Add null check for dbPath
    jstring dbPath = (jstring)env->GetObjectField(configuration, persistenceDbPathField);
    if (dbPath != nullptr) {
        const char *nativeDbPath = env->GetStringUTFChars(dbPath, nullptr);
        if (nativeDbPath != nullptr) {
            config.persistenceDbPath = const_cast<char *>(nativeDbPath);
        } else {
            config.persistenceDbPath[0] = '\0';
        }
    } else {
        config.persistenceDbPath[0] = '\0';
    }

    config.searchThreshold = env->GetFloatField(configuration, searchThresholdField);
    config.searchMode = (HFSearchMode)env->GetIntField(configuration, searchModeField);

    // Remove debug logs that might interfere with error handling
    auto result = HFFeatureHubDataEnable(config);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to enable feature hub data, error code: %d", result);
        return false;
    }

    return true;
}

/**
 * @brief Disable feature hub data.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @return True if the feature hub data is disabled successfully, false otherwise.
 */
JNIEXPORT jboolean INSPIRE_FACE_JNI(InspireFace_FeatureHubDataDisable)(JNIEnv *env, jobject thiz) {
    auto result = HFFeatureHubDataDisable();
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to disable feature hub data, error code: %d", result);
        return false;
    }

    return true;
}

/**
 * @brief Insert a feature into the feature hub.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param newIdentity The new identity object.
 * @return True if the feature is inserted successfully, false otherwise.
 */
JNIEXPORT jboolean INSPIRE_FACE_JNI(InspireFace_FeatureHubInsertFeature)(JNIEnv *env, jobject thiz, jobject newIdentity) {
    // Get FaceFeatureIdentity class and fields
    jclass identityClass = env->GetObjectClass(newIdentity);
    jfieldID idField = env->GetFieldID(identityClass, "id", "J");
    jfieldID featureField = env->GetFieldID(identityClass, "feature", "Lcom/insightface/sdk/inspireface/base/FaceFeature;");

    // Get feature object
    jobject featureObj = env->GetObjectField(newIdentity, featureField);
    if (featureObj == nullptr) {
        INSPIRE_LOGE("Feature object is null");
        return false;
    }

    // Get FaceFeature class and fields
    jclass featureClass = env->GetObjectClass(featureObj);
    jfieldID dataField = env->GetFieldID(featureClass, "data", "[F");

    // Get feature data array
    jfloatArray dataArray = (jfloatArray)env->GetObjectField(featureObj, dataField);
    if (dataArray == nullptr) {
        INSPIRE_LOGE("Feature data array is null");
        return false;
    }

    // Get feature data
    jsize length = env->GetArrayLength(dataArray);
    jfloat *data = env->GetFloatArrayElements(dataArray, nullptr);

    // Create HFFaceFeature
    HFFaceFeature feature;
    feature.data = data;
    feature.size = length;

    // Create HFFaceFeatureIdentity
    HFFaceFeatureIdentity identity;
    identity.id = env->GetLongField(newIdentity, idField);
    identity.feature = &feature;

    // Insert feature and get allocated ID
    HFaceId allocId = -1;
    auto result = HFFeatureHubInsertFeature(identity, &allocId);

    // Release array elements
    env->ReleaseFloatArrayElements(dataArray, data, 0);

    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to insert feature, error code: %d", result);
        return false;
    }

    // Update ID field with allocated ID
    env->SetLongField(newIdentity, idField, allocId);
    return true;
}

/**
 * @brief Search for a feature in the feature hub.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param searchFeature The search feature object.
 * @param confidence The confidence.
 * @return The search result object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_FeatureHubFaceSearch)(JNIEnv *env, jobject thiz, jobject searchFeature, jfloat confidence) {
    // Get FaceFeature class and fields
    jclass featureClass = env->GetObjectClass(searchFeature);
    jfieldID dataField = env->GetFieldID(featureClass, "data", "[F");

    // Get feature data array
    jfloatArray dataArray = (jfloatArray)env->GetObjectField(searchFeature, dataField);
    if (dataArray == nullptr) {
        INSPIRE_LOGE("Feature data array is null");
        return nullptr;
    }

    // Get feature data
    jsize length = env->GetArrayLength(dataArray);
    jfloat *data = env->GetFloatArrayElements(dataArray, nullptr);

    // Create HFFaceFeature
    HFFaceFeature feature;
    feature.data = data;
    feature.size = length;

    // Create HFFaceFeatureIdentity to store search result
    float searchConfidence = 0.0f;
    HFFaceFeatureIdentity mostSimilar;
    auto result = HFFeatureHubFaceSearch(feature, &searchConfidence, &mostSimilar);

    // Release array elements
    env->ReleaseFloatArrayElements(dataArray, data, 0);

    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to search feature, error code: %d", result);
        return nullptr;
    }

    // Create FaceFeatureIdentity object to return
    jclass identityClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceFeatureIdentity");
    jmethodID constructor = env->GetMethodID(identityClass, "<init>", "()V");
    jobject identityObj = env->NewObject(identityClass, constructor);

    // Set id field
    jfieldID idField = env->GetFieldID(identityClass, "id", "J");
    env->SetLongField(identityObj, idField, mostSimilar.id);

    // Create FaceFeature object and set data
    jclass faceFeatureClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceFeature");
    jmethodID featureConstructor = env->GetMethodID(faceFeatureClass, "<init>", "()V");
    jobject featureObj = env->NewObject(faceFeatureClass, featureConstructor);

    // Create and set float array for feature data
    jfloatArray featureArray = env->NewFloatArray(mostSimilar.feature->size);
    env->SetFloatArrayRegion(featureArray, 0, mostSimilar.feature->size, mostSimilar.feature->data);
    env->SetObjectField(featureObj, dataField, featureArray);

    // Set feature field in identity object
    jfieldID featureField = env->GetFieldID(identityClass, "feature", "Lcom/insightface/sdk/inspireface/base/FaceFeature;");
    env->SetObjectField(identityObj, featureField, featureObj);

    // Set searchConfidence field
    jfieldID searchConfidenceField = env->GetFieldID(identityClass, "searchConfidence", "F");
    env->SetFloatField(identityObj, searchConfidenceField, searchConfidence);

    return identityObj;
}

/**
 * @brief Search for top-k features in the feature hub.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param searchFeature The search feature object.
 * @param topK The top-k.
 * @return The search results object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_FeatureHubFaceSearchTopK)(JNIEnv *env, jobject thiz, jobject searchFeature, jint topK) {
    // Get feature data array
    jclass faceFeatureClass = env->GetObjectClass(searchFeature);
    jfieldID dataField = env->GetFieldID(faceFeatureClass, "data", "[F");
    jfloatArray dataArray = (jfloatArray)env->GetObjectField(searchFeature, dataField);
    jfloat *data = env->GetFloatArrayElements(dataArray, 0);
    jsize length = env->GetArrayLength(dataArray);

    // Create HFFaceFeature
    HFFaceFeature feature;
    feature.data = data;
    feature.size = length;

    // Create HFSearchTopKResults to store search results
    HFSearchTopKResults results;
    auto result = HFFeatureHubFaceSearchTopK(feature, topK, &results);

    // Release array elements
    env->ReleaseFloatArrayElements(dataArray, data, 0);

    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to search top-k features, error code: %d", result);
        return nullptr;
    }

    // Create SearchTopKResults object to return
    jclass searchResultsClass = env->FindClass("com/insightface/sdk/inspireface/base/SearchTopKResults");
    jmethodID constructor = env->GetMethodID(searchResultsClass, "<init>", "()V");
    jobject searchResultsObj = env->NewObject(searchResultsClass, constructor);

    // Set num field
    jfieldID numField = env->GetFieldID(searchResultsClass, "num", "I");
    env->SetIntField(searchResultsObj, numField, results.size);

    // Create and set confidence array
    jfloatArray confidenceArray = env->NewFloatArray(results.size);
    env->SetFloatArrayRegion(confidenceArray, 0, results.size, results.confidence);
    jfieldID confidenceField = env->GetFieldID(searchResultsClass, "confidence", "[F");
    env->SetObjectField(searchResultsObj, confidenceField, confidenceArray);

    // Create and set ids array
    jlongArray idsArray = env->NewLongArray(results.size);
    env->SetLongArrayRegion(idsArray, 0, results.size, results.ids);
    jfieldID idsField = env->GetFieldID(searchResultsClass, "ids", "[J");
    env->SetObjectField(searchResultsObj, idsField, idsArray);

    return searchResultsObj;
}

/**
 * @brief Remove a feature from the feature hub.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param id The id of the feature to remove.
 * @return True if the feature is removed successfully, false otherwise.
 */
JNIEXPORT jboolean INSPIRE_FACE_JNI(InspireFace_FeatureHubFaceRemove)(JNIEnv *env, jobject thiz, jlong id) {
    auto result = HFFeatureHubFaceRemove(id);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to remove feature, error code: %d", result);
        return false;
    }
    return true;
}

/**
 * @brief Update a feature in the feature hub.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param featureIdentity The feature identity object.
 * @return True if the feature is updated successfully, false otherwise.
 */
JNIEXPORT jboolean INSPIRE_FACE_JNI(InspireFace_FeatureHubFaceUpdate)(JNIEnv *env, jobject thiz, jobject featureIdentity) {
    // Get FaceFeatureIdentity class and fields
    jclass identityClass = env->GetObjectClass(featureIdentity);
    jfieldID idField = env->GetFieldID(identityClass, "id", "J");
    jfieldID featureField = env->GetFieldID(identityClass, "feature", "Lcom/insightface/sdk/inspireface/base/FaceFeature;");

    // Get id value
    jlong id = env->GetLongField(featureIdentity, idField);

    // Get feature object
    jobject featureObj = env->GetObjectField(featureIdentity, featureField);
    jclass featureClass = env->GetObjectClass(featureObj);
    jfieldID dataField = env->GetFieldID(featureClass, "data", "[F");

    // Get feature data array
    jfloatArray dataArray = (jfloatArray)env->GetObjectField(featureObj, dataField);
    jsize length = env->GetArrayLength(dataArray);
    jfloat *data = env->GetFloatArrayElements(dataArray, nullptr);

    // Create HFFaceFeature
    HFFaceFeature feature;
    feature.data = data;
    feature.size = length;

    // Create HFFaceFeatureIdentity
    HFFaceFeatureIdentity identity;
    identity.id = id;
    identity.feature = &feature;

    // Update feature
    auto result = HFFeatureHubFaceUpdate(identity);

    // Release array elements
    env->ReleaseFloatArrayElements(dataArray, data, 0);

    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to update feature, error code: %d", result);
        return false;
    }
    return true;
}

/**
 * @brief Get a face identity from the feature hub.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param id The id of the face identity to get.
 * @return The face identity object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_FeatureHubGetFaceIdentity)(JNIEnv *env, jobject thiz, jlong id) {
    // Create HFFaceFeatureIdentity
    HFFaceFeatureIdentity identity;
    auto result = HFFeatureHubGetFaceIdentity(id, &identity);
    if (result != HSUCCEED) {
        return nullptr;
    }

    // Create FaceFeature object
    jclass featureClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceFeature");
    jmethodID featureConstructor = env->GetMethodID(featureClass, "<init>", "()V");
    jobject featureObj = env->NewObject(featureClass, featureConstructor);

    // Set feature data array
    jfieldID dataField = env->GetFieldID(featureClass, "data", "[F");
    jfloatArray dataArray = env->NewFloatArray(identity.feature->size);
    env->SetFloatArrayRegion(dataArray, 0, identity.feature->size, identity.feature->data);
    env->SetObjectField(featureObj, dataField, dataArray);

    // Create FaceFeatureIdentity object
    jclass identityClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceFeatureIdentity");
    jmethodID identityConstructor = env->GetMethodID(identityClass, "<init>", "()V");
    jobject identityObj = env->NewObject(identityClass, identityConstructor);

    // Set id and feature fields
    jfieldID idField = env->GetFieldID(identityClass, "id", "J");
    jfieldID featureField = env->GetFieldID(identityClass, "feature", "Lcom/insightface/sdk/inspireface/base/FaceFeature;");
    env->SetLongField(identityObj, idField, identity.id);
    env->SetObjectField(identityObj, featureField, featureObj);

    return identityObj;
}

/**
 * @brief Get the number of faces in the feature hub.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @return The number of faces in the feature hub.
 */
JNIEXPORT jint INSPIRE_FACE_JNI(InspireFace_FeatureHubGetFaceCount)(JNIEnv *env, jobject thiz) {
    HInt32 count;
    auto result = HFFeatureHubGetFaceCount(&count);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to get face count, error code: %d", result);
        return -1;
    }
    return count;
}

/**
 * @brief Set the face search threshold.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param threshold The face search threshold.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_FeatureHubFaceSearchThresholdSetting)(JNIEnv *env, jobject thiz, jfloat threshold) {
    auto result = HFFeatureHubFaceSearchThresholdSetting(threshold);
    if (result != HSUCCEED) {
        INSPIRE_LOGE("Failed to set face search threshold, error code: %d", result);
    }
}

/**
 * @brief Get the length of the feature.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @return The length of the feature.
 */
JNIEXPORT jint INSPIRE_FACE_JNI(InspireFace_GetFeatureLength)(JNIEnv *env, jobject thiz) {
    HInt32 length;
    auto result = HFGetFeatureLength(&length);
    return length;
}

/**
 * @brief Update the cosine similarity converter.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param config The cosine similarity converter configuration object.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_UpdateCosineSimilarityConverter)(JNIEnv *env, jobject thiz, jobject config) {
    jclass configClass = env->GetObjectClass(config);
    jfieldID thresholdField = env->GetFieldID(configClass, "threshold", "F");
    jfieldID middleScoreField = env->GetFieldID(configClass, "middleScore", "F");
    jfieldID steepnessField = env->GetFieldID(configClass, "steepness", "F");
    jfieldID outputMinField = env->GetFieldID(configClass, "outputMin", "F");
    jfieldID outputMaxField = env->GetFieldID(configClass, "outputMax", "F");

    HFSimilarityConverterConfig converterConfig;
    converterConfig.threshold = env->GetFloatField(config, thresholdField);
    converterConfig.middleScore = env->GetFloatField(config, middleScoreField);
    converterConfig.steepness = env->GetFloatField(config, steepnessField);
    converterConfig.outputMin = env->GetFloatField(config, outputMinField);
    converterConfig.outputMax = env->GetFloatField(config, outputMaxField);
    HFUpdateCosineSimilarityConverter(converterConfig);
}

/**
 * @brief Get the cosine similarity converter.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @return The cosine similarity converter object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_GetCosineSimilarityConverter)(JNIEnv *env, jobject thiz) {
    HFSimilarityConverterConfig converterConfig;
    HFGetCosineSimilarityConverter(&converterConfig);
    jclass configClass = env->FindClass("com/insightface/sdk/inspireface/base/SimilarityConverterConfig");
    jmethodID constructor = env->GetMethodID(configClass, "<init>", "()V");
    jobject configObj = env->NewObject(configClass, constructor);

    // Get field IDs
    jfieldID thresholdField = env->GetFieldID(configClass, "threshold", "F");
    jfieldID middleScoreField = env->GetFieldID(configClass, "middleScore", "F");
    jfieldID steepnessField = env->GetFieldID(configClass, "steepness", "F");
    jfieldID outputMinField = env->GetFieldID(configClass, "outputMin", "F");
    jfieldID outputMaxField = env->GetFieldID(configClass, "outputMax", "F");

    // Set fields
    env->SetFloatField(configObj, thresholdField, converterConfig.threshold);
    env->SetFloatField(configObj, middleScoreField, converterConfig.middleScore);
    env->SetFloatField(configObj, steepnessField, converterConfig.steepness);
    env->SetFloatField(configObj, outputMinField, converterConfig.outputMin);
    env->SetFloatField(configObj, outputMaxField, converterConfig.outputMax);

    return configObj;
}

/**
 * @brief Convert cosine similarity to percentage.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param similarity The cosine similarity.
 * @return The percentage.
 */
JNIEXPORT jfloat INSPIRE_FACE_JNI(InspireFace_CosineSimilarityConvertToPercentage)(JNIEnv *env, jobject thiz, jfloat similarity) {
    HFloat result;
    HFCosineSimilarityConvertToPercentage(similarity, &result);
    return result;
}

/**
 * @brief Compare two face features.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param feature1 The first face feature.
 * @param feature2 The second face feature.
 * @return The similarity between the two face features.
 */
JNIEXPORT jfloat INSPIRE_FACE_JNI(InspireFace_FaceComparison)(JNIEnv *env, jobject thiz, jobject feature1, jobject feature2) {
    if (feature1 == nullptr || feature2 == nullptr) {
        return -1.0f;
    }

    // Get FaceFeature class and data field
    jclass featureClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceFeature");
    jfieldID dataField = env->GetFieldID(featureClass, "data", "[F");

    // Get float arrays from FaceFeature objects
    jfloatArray data1Array = (jfloatArray)env->GetObjectField(feature1, dataField);
    jfloatArray data2Array = (jfloatArray)env->GetObjectField(feature2, dataField);

    if (data1Array == nullptr || data2Array == nullptr) {
        return -1.0f;
    }

    // Get array lengths
    jsize len1 = env->GetArrayLength(data1Array);
    jsize len2 = env->GetArrayLength(data2Array);

    if (len1 != len2) {
        return -1.0f;
    }

    // Get float data
    jfloat *data1 = env->GetFloatArrayElements(data1Array, nullptr);
    jfloat *data2 = env->GetFloatArrayElements(data2Array, nullptr);

    // Create HFFaceFeature structs
    HFFaceFeature ft1, ft2;
    ft1.data = data1;
    ft1.size = len1;
    ft2.data = data2;
    ft2.size = len2;

    // Compare features
    float compareResult;
    HResult ret = HFFaceComparison(ft1, ft2, &compareResult);

    // Release arrays
    env->ReleaseFloatArrayElements(data1Array, data1, 0);
    env->ReleaseFloatArrayElements(data2Array, data2, 0);

    if (ret != HSUCCEED) {
        return -1.0f;
    }

    return compareResult;
}

/**
 * @brief Get the recommended cosine threshold.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @return The recommended cosine threshold.
 */
JNIEXPORT jfloat INSPIRE_FACE_JNI(InspireFace_GetRecommendedCosineThreshold)(JNIEnv *env, jobject thiz) {
    HFloat threshold;
    HFGetRecommendedCosineThreshold(&threshold);
    return threshold;
}

/**
 * @brief Process multiple faces in the pipeline.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The session object.
 * @param streamHandle The stream handle object.
 * @param faces The faces object.
 * @param parameter The parameter object.
 * @return True if the multiple faces are processed successfully, false otherwise.
 */
JNIEXPORT jboolean INSPIRE_FACE_JNI(InspireFace_MultipleFacePipelineProcess)(JNIEnv *env, jobject thiz, jobject session, jobject streamHandle,
                                                                             jobject faces, jobject parameter) {
    // Get session handle
    jclass sessionClass = env->FindClass("com/insightface/sdk/inspireface/base/Session");
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, handleField);

    // Get stream handle
    jclass streamClass = env->FindClass("com/insightface/sdk/inspireface/base/ImageStream");
    handleField = env->GetFieldID(streamClass, "handle", "J");
    jlong streamHandleValue = env->GetLongField(streamHandle, handleField);

    // Get faces data
    jclass facesClass = env->FindClass("com/insightface/sdk/inspireface/base/MultipleFaceData");
    jfieldID numField = env->GetFieldID(facesClass, "detectedNum", "I");
    jfieldID tokensField = env->GetFieldID(facesClass, "tokens", "[Lcom/insightface/sdk/inspireface/base/FaceBasicToken;");

    jint detectedNum = env->GetIntField(faces, numField);
    jobjectArray tokenArray = (jobjectArray)env->GetObjectField(faces, tokensField);

    // Create HFMultipleFaceData struct
    HFMultipleFaceData faceData;
    faceData.detectedNum = detectedNum;

    // Get token data
    HFFaceBasicToken *tokens = nullptr;
    std::vector<jbyteArray> tokenDataArrays;
    std::vector<jbyte*> tokenBuffers;
    
    if (detectedNum > 0) {
        jclass tokenClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceBasicToken");
        jfieldID tokenDataField = env->GetFieldID(tokenClass, "data", "[B");
        jfieldID sizeField = env->GetFieldID(tokenClass, "size", "I");

        tokens = new HFFaceBasicToken[detectedNum];
        tokenDataArrays.resize(detectedNum);
        tokenBuffers.resize(detectedNum);
        
        for (int i = 0; i < detectedNum; i++) {
            jobject token = env->GetObjectArrayElement(tokenArray, i);
            jbyteArray dataArray = (jbyteArray)env->GetObjectField(token, tokenDataField);
            jint size = env->GetIntField(token, sizeField);
            
            if (dataArray != nullptr) {
                tokens[i].size = size;
                tokens[i].data = env->GetByteArrayElements(dataArray, nullptr);
                tokenDataArrays[i] = dataArray;
                tokenBuffers[i] = (jbyte*)tokens[i].data;
            } else {
                tokens[i].size = 0;
                tokens[i].data = nullptr;
                tokenDataArrays[i] = nullptr;
                tokenBuffers[i] = nullptr;
            }
            env->DeleteLocalRef(token);
        }
        faceData.tokens = tokens;
    } else {
        faceData.tokens = nullptr;
    }

    // Get custom parameter fields
    jclass paramClass = env->FindClass("com/insightface/sdk/inspireface/base/CustomParameter");
    jfieldID enableRecognitionField = env->GetFieldID(paramClass, "enableRecognition", "I");
    jfieldID enableLivenessField = env->GetFieldID(paramClass, "enableLiveness", "I");
    jfieldID enableIrLivenessField = env->GetFieldID(paramClass, "enableIrLiveness", "I");
    jfieldID enableMaskDetectField = env->GetFieldID(paramClass, "enableMaskDetect", "I");
    jfieldID enableFaceQualityField = env->GetFieldID(paramClass, "enableFaceQuality", "I");
    jfieldID enableFaceAttributeField = env->GetFieldID(paramClass, "enableFaceAttribute", "I");
    jfieldID enableInteractionLivenessField = env->GetFieldID(paramClass, "enableInteractionLiveness", "I");
    jfieldID enableFacePoseField = env->GetFieldID(paramClass, "enableFacePose", "I");
    jfieldID enableFaceEmotionField = env->GetFieldID(paramClass, "enableFaceEmotion", "I");
    // Get parameter values
    HFSessionCustomParameter customParam;
    customParam.enable_recognition = env->GetIntField(parameter, enableRecognitionField);
    customParam.enable_liveness = env->GetIntField(parameter, enableLivenessField);
    customParam.enable_ir_liveness = env->GetIntField(parameter, enableIrLivenessField);
    customParam.enable_mask_detect = env->GetIntField(parameter, enableMaskDetectField);
    customParam.enable_face_quality = env->GetIntField(parameter, enableFaceQualityField);
    customParam.enable_face_attribute = env->GetIntField(parameter, enableFaceAttributeField);
    customParam.enable_interaction_liveness = env->GetIntField(parameter, enableInteractionLivenessField);
    customParam.enable_face_pose = env->GetIntField(parameter, enableFacePoseField);
    customParam.enable_face_emotion = env->GetIntField(parameter, enableFaceEmotionField);
    // Call native function
    HResult ret = HFMultipleFacePipelineProcess((HFSession)sessionHandle, (HFImageStream)streamHandleValue, &faceData, customParam);

    // Clean up allocated memory
    if (tokens != nullptr) {
        for (int i = 0; i < detectedNum; i++) {
            if (tokenDataArrays[i] != nullptr && tokenBuffers[i] != nullptr) {
                env->ReleaseByteArrayElements(tokenDataArrays[i], tokenBuffers[i], JNI_ABORT);
            }
        }
        delete[] tokens;
    }

    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Failed to process multiple faces, error code: %d", ret);
        return false;
    }

    return true;
}

/**
 * @brief Get the RGB liveness confidence.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The session object.
 * @return The RGB liveness confidence object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_GetRGBLivenessConfidence)(JNIEnv *env, jobject thiz, jobject session) {
    // Get session handle
    jclass sessionClass = env->FindClass("com/insightface/sdk/inspireface/base/Session");
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, handleField);

    // Create native confidence struct
    HFRGBLivenessConfidence confidence;
    HResult ret = HFGetRGBLivenessConfidence((HFSession)sessionHandle, &confidence);

    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Failed to get RGB liveness confidence, error code: %d", ret);
        return nullptr;
    }

    // Create Java RGBLivenessConfidence object
    jclass confidenceClass = env->FindClass("com/insightface/sdk/inspireface/base/RGBLivenessConfidence");
    jobject confidenceObj = env->AllocObject(confidenceClass);

    // Set num field
    jfieldID numField = env->GetFieldID(confidenceClass, "num", "I");
    env->SetIntField(confidenceObj, numField, confidence.num);

    // Set confidence array field
    jfieldID confidenceField = env->GetFieldID(confidenceClass, "confidence", "[F");
    jfloatArray confidenceArray = env->NewFloatArray(confidence.num);
    float *confidencePtr = confidence.confidence;
    env->SetFloatArrayRegion(confidenceArray, 0, confidence.num, confidencePtr);
    env->SetObjectField(confidenceObj, confidenceField, confidenceArray);

    return confidenceObj;
}

/**
 * @brief Get the face quality confidence.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The session object.
 * @return The face quality confidence object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_GetFaceQualityConfidence)(JNIEnv *env, jobject thiz, jobject session) {
    // Get session handle
    jclass sessionClass = env->FindClass("com/insightface/sdk/inspireface/base/Session");
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, handleField);

    // Create native confidence struct
    HFFaceQualityConfidence confidence;
    HResult ret = HFGetFaceQualityConfidence((HFSession)sessionHandle, &confidence);

    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Failed to get face quality confidence, error code: %d", ret);
        return nullptr;
    }

    // Create Java FaceQualityConfidence object
    jclass confidenceClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceQualityConfidence");
    jobject confidenceObj = env->AllocObject(confidenceClass);

    // Set num field
    jfieldID numField = env->GetFieldID(confidenceClass, "num", "I");
    env->SetIntField(confidenceObj, numField, confidence.num);

    // Set confidence array field
    jfieldID confidenceField = env->GetFieldID(confidenceClass, "confidence", "[F");
    jfloatArray confidenceArray = env->NewFloatArray(confidence.num);
    float *confidencePtr = confidence.confidence;
    env->SetFloatArrayRegion(confidenceArray, 0, confidence.num, confidencePtr);
    env->SetObjectField(confidenceObj, confidenceField, confidenceArray);

    return confidenceObj;
}

/**
 * @brief Get the face mask confidence.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The session object.
 * @return The face mask confidence object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_GetFaceMaskConfidence)(JNIEnv *env, jobject thiz, jobject session) {
    // Get session handle
    jclass sessionClass = env->FindClass("com/insightface/sdk/inspireface/base/Session");
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, handleField);

    // Create native confidence struct
    HFFaceMaskConfidence confidence;
    HResult ret = HFGetFaceMaskConfidence((HFSession)sessionHandle, &confidence);

    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Failed to get face mask confidence, error code: %d", ret);
        return nullptr;
    }

    // Create Java FaceMaskConfidence object
    jclass confidenceClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceMaskConfidence");
    jobject confidenceObj = env->AllocObject(confidenceClass);

    // Set num field
    jfieldID numField = env->GetFieldID(confidenceClass, "num", "I");
    env->SetIntField(confidenceObj, numField, confidence.num);

    // Set confidence array field
    jfieldID confidenceField = env->GetFieldID(confidenceClass, "confidence", "[F");
    jfloatArray confidenceArray = env->NewFloatArray(confidence.num);
    float *confidencePtr = confidence.confidence;
    env->SetFloatArrayRegion(confidenceArray, 0, confidence.num, confidencePtr);
    env->SetObjectField(confidenceObj, confidenceField, confidenceArray);

    return confidenceObj;
}

/**
 * @brief Get the face interaction state result.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The session object.
 * @return The face interaction state result object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_GetFaceInteractionStateResult)(JNIEnv *env, jobject thiz, jobject session) {
    // Get session handle
    jclass sessionClass = env->FindClass("com/insightface/sdk/inspireface/base/Session");
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, handleField);

    // Create native state struct
    HFFaceInteractionState state;
    HResult ret = HFGetFaceInteractionStateResult((HFSession)sessionHandle, &state);

    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Failed to get face interaction state result, error code: %d", ret);
        return nullptr;
    }

    // Create Java FaceInteractionState object
    jclass stateClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceInteractionState");
    jobject stateObj = env->AllocObject(stateClass);

    // Set num field
    jfieldID numField = env->GetFieldID(stateClass, "num", "I");
    env->SetIntField(stateObj, numField, state.num);

    // Set leftEyeStatusConfidence array field
    jfieldID leftEyeField = env->GetFieldID(stateClass, "leftEyeStatusConfidence", "[F");
    jfloatArray leftEyeArray = env->NewFloatArray(state.num);
    float *leftEyePtr = state.leftEyeStatusConfidence;
    env->SetFloatArrayRegion(leftEyeArray, 0, state.num, leftEyePtr);
    env->SetObjectField(stateObj, leftEyeField, leftEyeArray);

    // Set rightEyeStatusConfidence array field
    jfieldID rightEyeField = env->GetFieldID(stateClass, "rightEyeStatusConfidence", "[F");
    jfloatArray rightEyeArray = env->NewFloatArray(state.num);
    float *rightEyePtr = state.rightEyeStatusConfidence;
    env->SetFloatArrayRegion(rightEyeArray, 0, state.num, rightEyePtr);
    env->SetObjectField(stateObj, rightEyeField, rightEyeArray);

    return stateObj;
}

/**
 * @brief Get the face interaction actions result.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The session object.
 * @return The face interaction actions result object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_GetFaceInteractionActionsResult)(JNIEnv *env, jobject thiz, jobject session) {
    // Get session handle
    jclass sessionClass = env->FindClass("com/insightface/sdk/inspireface/base/Session");
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, handleField);

    // Create native actions struct
    HFFaceInteractionsActions actions;
    HResult ret = HFGetFaceInteractionActionsResult((HFSession)sessionHandle, &actions);

    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Failed to get face interaction actions result, error code: %d", ret);
        return nullptr;
    }

    // Create Java FaceInteractionsActions object
    jclass actionsClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceInteractionsActions");
    jobject actionsObj = env->AllocObject(actionsClass);

    // Set num field
    jfieldID numField = env->GetFieldID(actionsClass, "num", "I");
    env->SetIntField(actionsObj, numField, actions.num);

    // Set normal array field
    jfieldID normalField = env->GetFieldID(actionsClass, "normal", "[I");
    jintArray normalArray = env->NewIntArray(actions.num);
    int *normalPtr = actions.normal;
    env->SetIntArrayRegion(normalArray, 0, actions.num, normalPtr);
    env->SetObjectField(actionsObj, normalField, normalArray);

    // Set shake array field
    jfieldID shakeField = env->GetFieldID(actionsClass, "shake", "[I");
    jintArray shakeArray = env->NewIntArray(actions.num);
    int *shakePtr = actions.shake;
    env->SetIntArrayRegion(shakeArray, 0, actions.num, shakePtr);
    env->SetObjectField(actionsObj, shakeField, shakeArray);

    // Set jawOpen array field
    jfieldID jawOpenField = env->GetFieldID(actionsClass, "jawOpen", "[I");
    jintArray jawOpenArray = env->NewIntArray(actions.num);
    int *jawOpenPtr = actions.jawOpen;
    env->SetIntArrayRegion(jawOpenArray, 0, actions.num, jawOpenPtr);
    env->SetObjectField(actionsObj, jawOpenField, jawOpenArray);

    // Set headRaise array field
    jfieldID headRaiseField = env->GetFieldID(actionsClass, "headRaise", "[I");
    jintArray headRaiseArray = env->NewIntArray(actions.num);
    int *headRaisePtr = actions.headRaise;
    env->SetIntArrayRegion(headRaiseArray, 0, actions.num, headRaisePtr);
    env->SetObjectField(actionsObj, headRaiseField, headRaiseArray);

    // Set blink array field
    jfieldID blinkField = env->GetFieldID(actionsClass, "blink", "[I");
    jintArray blinkArray = env->NewIntArray(actions.num);
    int *blinkPtr = actions.blink;
    env->SetIntArrayRegion(blinkArray, 0, actions.num, blinkPtr);
    env->SetObjectField(actionsObj, blinkField, blinkArray);

    return actionsObj;
}

/**
 * @brief Get the face attribute result.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The session object.
 * @return The face attribute result object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_GetFaceAttributeResult)(JNIEnv *env, jobject thiz, jobject session) {
    // Validate input parameters
    if (!env || !session) {
        INSPIRE_LOGE("Invalid input parameters");
        return nullptr;
    }

    // Get session handle
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, handleField);
    if (!sessionHandle) {
        INSPIRE_LOGE("Invalid session handle");
        return nullptr;
    }

    // Get face attribute results
    HFFaceAttributeResult results = {};
    HResult ret = HFGetFaceAttributeResult((HFSession)sessionHandle, &results);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Failed to get face attribute result, error code: %d", ret);
        return nullptr;
    }

    // Create Java FaceAttributeResult object
    jclass attributeClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceAttributeResult");
    if (!attributeClass) {
        INSPIRE_LOGE("Failed to find FaceAttributeResult class");
        return nullptr;
    }

    jmethodID constructor = env->GetMethodID(attributeClass, "<init>", "()V");
    jobject attributeObj = env->NewObject(attributeClass, constructor);
    if (!attributeObj) {
        INSPIRE_LOGE("Failed to create FaceAttributeResult object");
        return nullptr;
    }

    // Set fields
    jfieldID numField = env->GetFieldID(attributeClass, "num", "I");
    jfieldID raceField = env->GetFieldID(attributeClass, "race", "[I");
    jfieldID genderField = env->GetFieldID(attributeClass, "gender", "[I");
    jfieldID ageBracketField = env->GetFieldID(attributeClass, "ageBracket", "[I");

    if (!numField || !raceField || !genderField || !ageBracketField) {
        INSPIRE_LOGE("Failed to get field IDs");
        return nullptr;
    }

    // Set num
    env->SetIntField(attributeObj, numField, results.num);

    // Set arrays
    jintArray raceArray = env->NewIntArray(results.num);
    jintArray genderArray = env->NewIntArray(results.num);
    jintArray ageBracketArray = env->NewIntArray(results.num);

    if (!raceArray || !genderArray || !ageBracketArray) {
        INSPIRE_LOGE("Failed to create arrays");
        if (raceArray)
            env->DeleteLocalRef(raceArray);
        if (genderArray)
            env->DeleteLocalRef(genderArray);
        if (ageBracketArray)
            env->DeleteLocalRef(ageBracketArray);
        return nullptr;
    }

    env->SetIntArrayRegion(raceArray, 0, results.num, results.race);
    env->SetIntArrayRegion(genderArray, 0, results.num, results.gender);
    env->SetIntArrayRegion(ageBracketArray, 0, results.num, results.ageBracket);

    env->SetObjectField(attributeObj, raceField, raceArray);
    env->SetObjectField(attributeObj, genderField, genderArray);
    env->SetObjectField(attributeObj, ageBracketField, ageBracketArray);

    env->DeleteLocalRef(raceArray);
    env->DeleteLocalRef(genderArray);
    env->DeleteLocalRef(ageBracketArray);

    return attributeObj;
}

/**
 * @brief Get the face emotion result.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param session The session object.
 * @return The face emotion result object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_GetFaceEmotionResult)(JNIEnv *env, jobject thiz, jobject session) {
    // Validate input parameters
    if (!env || !session) {
        INSPIRE_LOGE("Invalid input parameters");
        return nullptr;
    }

    // Get session handle
    jclass sessionClass = env->GetObjectClass(session);
    jfieldID handleField = env->GetFieldID(sessionClass, "handle", "J");
    jlong sessionHandle = env->GetLongField(session, handleField);
    if (!sessionHandle) {
        INSPIRE_LOGE("Invalid session handle");
        return nullptr;
    }

    // Get face emotion results
    HFFaceEmotionResult results = {};
    HResult ret = HFGetFaceEmotionResult((HFSession)sessionHandle, &results);
    if (ret != HSUCCEED) {
        INSPIRE_LOGE("Failed to get face emotion result, error code: %d", ret);
        return nullptr;
    }

    // Create Java FaceEmotionResult object
    jclass emotionClass = env->FindClass("com/insightface/sdk/inspireface/base/FaceEmotionResult");
    if (!emotionClass) {
        INSPIRE_LOGE("Failed to find FaceEmotionResult class");
        return nullptr;
    }

    jmethodID constructor = env->GetMethodID(emotionClass, "<init>", "()V");
    jobject emotionObj = env->NewObject(emotionClass, constructor);
    if (!emotionObj) {
        INSPIRE_LOGE("Failed to create FaceEmotionResult object");
        return nullptr;
    }

    // Set fields
    jfieldID numField = env->GetFieldID(emotionClass, "num", "I");
    jfieldID emotionField = env->GetFieldID(emotionClass, "emotion", "[I");

    if (!numField || !emotionField) {
        INSPIRE_LOGE("Failed to get field IDs");
        return nullptr;
    }

    // Set num
    env->SetIntField(emotionObj, numField, results.num);

    // Set emotion array
    jintArray emotionArray = env->NewIntArray(results.num);
    if (!emotionArray) {
        INSPIRE_LOGE("Failed to create emotion array");
        return nullptr;
    }

    env->SetIntArrayRegion(emotionArray, 0, results.num, results.emotion);
    env->SetObjectField(emotionObj, emotionField, emotionArray);

    env->DeleteLocalRef(emotionArray);

    return emotionObj;
}

/**
 * @brief Query the InspireFace version.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @return The InspireFace version object.
 */
JNIEXPORT jobject INSPIRE_FACE_JNI(InspireFace_QueryInspireFaceVersion)(JNIEnv *env, jobject thiz) {
    // Get version info
    HFInspireFaceVersion versionInfo;
    HFQueryInspireFaceVersion(&versionInfo);

    // Create new InspireFaceVersion object
    jclass versionClass = env->FindClass("com/insightface/sdk/inspireface/base/InspireFaceVersion");
    if (!versionClass) {
        INSPIRE_LOGE("Failed to find InspireFaceVersion class");
        return nullptr;
    }

    jmethodID constructor = env->GetMethodID(versionClass, "<init>", "()V");
    jobject version = env->NewObject(versionClass, constructor);
    if (!version) {
        INSPIRE_LOGE("Failed to create InspireFaceVersion object");
        return nullptr;
    }

    // Get field IDs
    jfieldID majorField = env->GetFieldID(versionClass, "major", "I");
    jfieldID minorField = env->GetFieldID(versionClass, "minor", "I");
    jfieldID patchField = env->GetFieldID(versionClass, "patch", "I");

    if (!majorField || !minorField || !patchField) {
        INSPIRE_LOGE("Failed to get InspireFaceVersion field IDs");
        return nullptr;
    }

    // Set version fields
    env->SetIntField(version, majorField, versionInfo.major);
    env->SetIntField(version, minorField, versionInfo.minor);
    env->SetIntField(version, patchField, versionInfo.patch);

    return version;
}

/**
 * @brief Set the log level.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 * @param level The log level.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_SetLogLevel)(JNIEnv *env, jobject thiz, jint level) {
    HFSetLogLevel((HFLogLevel)level);
}

/**
 * @brief Disable the log.
 *
 * @param env The JNI environment.
 * @param thiz The Java object.
 */
JNIEXPORT void INSPIRE_FACE_JNI(InspireFace_LogDisable)(JNIEnv *env, jobject thiz) {
    HFLogDisable();
}

}  // extern "C"

#endif  // ANDROID
