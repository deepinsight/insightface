//
// Created by tunm on 2023/10/3.
//

#ifndef HYPERFACEREPO_INSPIREFACE_H
#define HYPERFACEREPO_INSPIREFACE_H

#include <stdint.h>
#include "intypedef.h"
#include "herror.h"

#if defined(_WIN32)
#ifdef ISF_BUILD_SHARED_LIBS
#define HYPER_CAPI_EXPORT __declspec(dllexport)
#else
#define HYPER_CAPI_EXPORT
#endif
#else
#define HYPER_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#ifdef __cplusplus
extern "C" {
#endif

#define HF_ENABLE_NONE 0x00000000              ///< Flag to enable no features.
#define HF_ENABLE_FACE_RECOGNITION 0x00000002  ///< Flag to enable face recognition feature.
#define HF_ENABLE_LIVENESS 0x00000004          ///< Flag to enable RGB liveness detection feature.
#define HF_ENABLE_IR_LIVENESS \
    0x00000008  ///< Flag to enable IR (Infrared) liveness detection feature.
#define HF_ENABLE_MASK_DETECT 0x00000010     ///< Flag to enable mask detection feature.
#define HF_ENABLE_FACE_ATTRIBUTE 0x00000020  ///< Flag to enable face attribute prediction feature.
#define HF_ENABLE_PLACEHOLDER_ 0x00000040    ///< -
#define HF_ENABLE_QUALITY 0x00000080         ///< Flag to enable face quality assessment feature.
#define HF_ENABLE_INTERACTION 0x00000100     ///< Flag to enable interaction feature.

/**
 * Camera stream format.
 * Contains several common camera stream formats available in the market.
 */
typedef enum HFImageFormat {
    HF_STREAM_RGB = 0,       ///< Image in RGB format.
    HF_STREAM_BGR = 1,       ///< Image in BGR format (Opencv Mat default).
    HF_STREAM_RGBA = 2,      ///< Image in RGB format with alpha channel.
    HF_STREAM_BGRA = 3,      ///< Image in BGR format with alpha channel.
    HF_STREAM_YUV_NV12 = 4,  ///< Image in YUV NV12 format.
    HF_STREAM_YUV_NV21 = 5,  ///< Image in YUV NV21 format.
} HFImageFormat;

/**
 * Camera picture rotation mode.
 * To accommodate the rotation of certain devices, four image rotation modes are provided.
 */
typedef enum HFRotation {
    HF_CAMERA_ROTATION_0 = 0,    ///< 0 degree rotation.
    HF_CAMERA_ROTATION_90 = 1,   ///< 90 degree rotation.
    HF_CAMERA_ROTATION_180 = 2,  ///< 180 degree rotation.
    HF_CAMERA_ROTATION_270 = 3,  ///< 270 degree rotation.
} HFRotation;

/**
 * Image Buffer Data structure.
 * Defines the structure for image data stream.
 */
typedef struct HFImageData {
    uint8_t *data;         ///< Pointer to the image data stream.
    HInt32 width;          ///< Width of the image.
    HInt32 height;         ///< Height of the image.
    HFImageFormat format;  ///< Format of the image, indicating the data stream format to be parsed.
    HFRotation rotation;   ///< Rotation angle of the image.
} HFImageData, *PHFImageData;

/**
 * @brief Create a data buffer stream instantiation object.
 *
 * This function is used to create an instance of a data buffer stream with the given image data.
 *
 * @param data Pointer to the image buffer data structure.
 * @param handle Pointer to the stream handle that will be returned.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFCreateImageStream(PHFImageData data, HFImageStream *handle);

/**
 * @brief Release the instantiated DataBuffer object.
 *
 * This function is used to release the DataBuffer object that has been previously instantiated.
 *
 * @param streamHandle Pointer to the DataBuffer handle representing the camera stream component.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFReleaseImageStream(HFImageStream streamHandle);

/************************************************************************
 * Resource Function
 ************************************************************************/

/**
 * @brief Launch InspireFace SDK
 * Start the InspireFace SDK at the initialization stage of your program, as it is global and
 * designed to be used only once. It serves as a prerequisite for other function interfaces, so it
 * is essential to ensure it is initialized before calling any other APIs.
 * @param resourcePath Initializes the path to the resource file that needs to be loaded
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFLaunchInspireFace(HPath resourcePath);

/**
 * @brief Terminate InspireFace SDK
 * Terminate the InspireFace SDK, releasing all allocated resources.
 * This should be called at the end of your program to ensure proper cleanup.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFTerminateInspireFace();

/************************************************************************
 * FaceContext
 ************************************************************************/

/**
 * @brief Struct for custom parameters in face recognition context.
 *
 * This struct holds various flags to enable or disable specific features
 * in the face recognition context, such as face recognition, liveness detection,
 * mask detection, age and gender prediction, etc.
 */
typedef struct HFSessionCustomParameter {
    HInt32 enable_recognition;           ///< Enable face recognition feature.
    HInt32 enable_liveness;              ///< Enable RGB liveness detection feature.
    HInt32 enable_ir_liveness;           ///< Enable IR liveness detection feature.
    HInt32 enable_mask_detect;           ///< Enable mask detection feature.
    HInt32 enable_face_quality;          ///< Enable face quality detection feature.
    HInt32 enable_face_attribute;        ///< Enable face attribute prediction feature.
    HInt32 enable_interaction_liveness;  ///< Enable interaction for liveness detection feature.
} HFSessionCustomParameter, *PHFSessionCustomParameter;

/**
 * @brief Enumeration for face detection modes.
 */
typedef enum HFDetectMode {
    HF_DETECT_MODE_ALWAYS_DETECT,  ///< Image detection mode, always detect, applicable to images.
    HF_DETECT_MODE_LIGHT_TRACK,    ///< Video detection mode, face tracking, applicable to video
                                   ///< streaming, front camera.
    HF_DETECT_MODE_TRACK_BY_DETECTION,  ///< Video detection mode, face tracking, applicable to high
                                        ///< resolution, monitoring, capturing
                                        //   (You need a specific option turned on at compile time
                                        //   to use it).
} HFDetectMode;

/**
 * @brief Create a session from a resource file.
 *
 * @param parameter Custom parameters for session.
 * @param detectMode Detection mode to be used.
 * @param maxDetectFaceNum Maximum number of faces to detect.
 * @param detectPixelLevel Modify the input resolution level of the detector, the larger the better,
 *          the need to input a multiple of 160, such as 160, 320, 640, the default value -1 is 320.
 * @param trackByDetectModeFPS If you are using the MODE_TRACK_BY_DETECTION tracking mode,
 *          this value is used to set the fps frame rate of your current incoming video stream,
 * which defaults to -1 at 30fps.
 * @param handle Pointer to the context handle that will be returned.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFCreateInspireFaceSession(
  HFSessionCustomParameter parameter, HFDetectMode detectMode, HInt32 maxDetectFaceNum,
  HInt32 detectPixelLevel, HInt32 trackByDetectModeFPS, HFSession *handle);

/**
 * @brief Create a session from a resource file with additional options.
 *
 * @param customOption Custom option for additional configuration.
 * @param detectMode Detection mode to be used.
 * @param maxDetectFaceNum Maximum number of faces to detect.
 * @param detectPixelLevel Modify the input resolution level of the detector, the larger the better,
 *          the need to input a multiple of 160, such as 160, 320, 640, the default value -1 is 320.
 * @param trackByDetectModeFPS If you are using the MODE_TRACK_BY_DETECTION tracking mode,
 *          this value is used to set the fps frame rate of your current incoming video stream,
 * which defaults to -1 at 30fps.
 * @param handle Pointer to the context handle that will be returned.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFCreateInspireFaceSessionOptional(
  HOption customOption, HFDetectMode detectMode, HInt32 maxDetectFaceNum, HInt32 detectPixelLevel,
  HInt32 trackByDetectModeFPS, HFSession *handle);

/**
 * @brief Release the session.
 *
 * @param handle Handle to the session to be released.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFReleaseInspireFaceSession(HFSession handle);

/**
 * @brief Struct representing a basic token for face data.
 *
 * This struct holds the size and data pointer for a basic token associated with face data.
 */
typedef struct HFFaceBasicToken {
    HInt32 size;  ///< Size of the token.
    HPVoid data;  ///< Pointer to the token data.
} HFFaceBasicToken, *PHFFaceBasicToken;

/**
 * @brief Struct for face Euler angles.
 *
 * This struct represents the Euler angles (roll, yaw, pitch) for face orientation.
 */
typedef struct HFFaceEulerAngle {
    HFloat *roll;   ///< Roll angle of the face.
    HFloat *yaw;    ///< Yaw angle of the face.
    HFloat *pitch;  ///< Pitch angle of the face.
} HFFaceEulerAngle;

/**
 * @brief Struct for holding data of multiple detected faces.
 *
 * This struct stores the data related to multiple faces detected, including the number of faces,
 * their bounding rectangles, track IDs, angles, and tokens.
 */
typedef struct HFMultipleFaceData {
    HInt32 detectedNum;        ///< Number of faces detected.
    HFaceRect *rects;          ///< Array of bounding rectangles for each face.
    HInt32 *trackIds;          ///< Array of track IDs for each face.
    HFFaceEulerAngle angles;   ///< Euler angles for each face.
    PHFFaceBasicToken tokens;  ///< Tokens associated with each face.
} HFMultipleFaceData, *PHFMultipleFaceData;

/**
 * @brief Set the track preview size in the session, it works with face detection and tracking
 * algorithms. Default preview size is 192(px).
 *
 * @param session Handle to the session.
 * @param previewSize The size of the preview for tracking.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetTrackPreviewSize(HFSession session,
                                                              HInt32 previewSize);

/**
 * @brief Set the minimum number of face pixels that the face detector can capture, and people below
 * this number will be filtered.
 *
 * @param session Handle to the session.
 * @param minSize The minimum pixel value, default value is 24.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetFilterMinimumFacePixelSize(HFSession session,
                                                                        HInt32 minSize);

/**
 * @brief Set the face detect threshold in the session.
 *
 * @param session Handle to the session.
 * @param detectMode The mode of the detection mode for tracking.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetFaceDetectThreshold(HFSession session,
                                                                 HFloat threshold);

/**
 * @brief Run face tracking in the session.
 *
 * @param session Handle to the session.
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param results Pointer to the structure where the results will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFExecuteFaceTrack(HFSession session, HFImageStream streamHandle,
                                                    PHFMultipleFaceData results);

/**
 * @brief Copies the data from a HF_FaceBasicToken to a specified buffer.
 *
 * This function copies the data pointed to by the HF_FaceBasicToken's data field
 * into a user-provided buffer. The caller is responsible for ensuring that the buffer
 * is large enough to hold the data being copied.
 *
 * @param token The HF_FaceBasicToken containing the data to be copied.
 * @param buffer The buffer where the data will be copied to.
 * @param bufferSize The size of the buffer provided by the caller. Must be large enough
 *        to hold the data pointed to by the token's data field.
 * @return HResult indicating the success or failure of the operation. Returns HSUCCEED
 *         if the operation was successful, or an error code if the buffer was too small
 *         or if any other error occurred.
 */
HYPER_CAPI_EXPORT extern HResult HFCopyFaceBasicToken(HFFaceBasicToken token, HPBuffer buffer,
                                                      HInt32 bufferSize);

/**
 * @brief Retrieves the size of the data contained in a HF_FaceBasicToken.
 *
 * This function is used to query the size of the data that a HF_FaceBasicToken is
 * expected to contain. This is useful for allocating a buffer of appropriate size
 * before copying data from a HF_FaceBasicToken.
 *
 * @param bufferSize Pointer to an integer where the size of the data will be stored.
 *        On successful completion, this will contain the size of the data in bytes.
 * @return HResult indicating the success or failure of the operation. Returns HSUCCEED
 *         if the operation was successful, or an error code if it failed.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceBasicTokenSize(HPInt32 bufferSize);

/**
 * @brief Retrieve the number of dense facial landmarks.
 * @param num Number of dense facial landmarks
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetNumOfFaceDenseLandmark(HPInt32 num);

/**
 * @brief When you pass in a valid facial token, you can retrieve a set of dense facial landmarks.
 *          The memory for the dense landmarks must be allocated by you.
 * @param singleFace Basic token representing a single face.
 * @param landmarks Pre-allocated memory address of the array for 2D floating-point coordinates.
 * @param num Number of landmark points
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceDenseLandmarkFromFaceToken(HFFaceBasicToken singleFace,
                                                                     HPoint2f *landmarks,
                                                                     HInt32 num);

/************************************************************************
 * Face Recognition
 ************************************************************************/

/**
 * @brief Struct representing a face feature.
 *
 * This struct holds the data related to a face feature, including size and actual feature data.
 */
typedef struct HFFaceFeature {
    HInt32 size;   ///< Size of the feature data.
    HPFloat data;  ///< Pointer to the feature data.
} HFFaceFeature, *PHFFaceFeature;

/**
 * @brief Extract a face feature from a given face.
 *
 * @param session Handle to the session.
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param singleFace Basic token representing a single face.
 * @param feature Pointer to the extracted face feature.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFaceFeatureExtract(HFSession session, HFImageStream streamHandle,
                                                      HFFaceBasicToken singleFace,
                                                      PHFFaceFeature feature);

/**
 * @brief Extract a face feature from a given face and copy it to the provided feature buffer.
 *
 * @param session Handle to the session.
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param singleFace Basic token representing a single face.
 * @param feature Pointer to the buffer where the extracted feature will be copied.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFaceFeatureExtractCpy(HFSession session,
                                                         HFImageStream streamHandle,
                                                         HFFaceBasicToken singleFace,
                                                         HPFloat feature);

/************************************************************************
 * Feature Hub
 ************************************************************************/

/**
 * @brief Select the search mode in the process of face recognition search,
 * and different modes will affect the execution efficiency and results
 * */
typedef enum HFSearchMode {
    HF_SEARCH_MODE_EAGER = 0,   // Eager mode: Stops when a vector meets the threshold.
    HF_SEARCH_MODE_EXHAUSTIVE,  // Exhaustive mode: Searches until the best match is found.
} HFSearchMode;

/**
 * @brief Struct for database configuration.
 *
 * This struct holds the configuration settings for using a database in the face recognition
 * context.
 */
typedef struct HFFeatureHubConfiguration {
    HInt32 featureBlockNum;  ///< The order of magnitude of face feature database is N * 512, and 20
                             ///< is recommended by default
    HInt32 enablePersistence;  ///< Flag to enable or disable the use of the database.
    HString dbPath;            ///< Path to the database file.
    float searchThreshold;     ///< Threshold for face search
    HFSearchMode searchMode;   ///< Mode of face search
} HFFeatureHubConfiguration;

/**
 * @brief A lightweight face feature vector management.
 * @details FeatureHub is a built-in global lightweight face feature vector management functionality
 * provided in the InspireFace-SDK. It supports basic face feature search, deletion, and
 * modification functions, and offers two optional data storage modes: an in-memory model and a
 * persistence model. If you have simple storage needs, you can enable it.
 *
 * @param configuration FeatureHub configuration details.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubDataEnable(HFFeatureHubConfiguration configuration);

/**
 * @brief Disable the global FeatureHub feature, and you can enable it again if needed.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubDataDisable();

/**
 * @brief Struct representing the identity of a face feature.
 *
 * This struct associates a custom identifier and a tag with a specific face feature.
 */
typedef struct HFFaceFeatureIdentity {
    HInt32 customId;         ///< Custom identifier for the face feature.
    HString tag;             ///< Tag associated with the face feature.
    PHFFaceFeature feature;  ///< Pointer to the face feature.
} HFFaceFeatureIdentity, *PHFFaceFeatureIdentity;

/**
 * Search structure for top-k mode
 * */
typedef struct HFSearchTopKResults {
    HInt32 size;         ///< The number of faces searched
    HPFloat confidence;  ///< Search confidence(it has already been filtered once by the threshold)
    HPInt32 customIds;   ///< fACE customIds
} HFSearchTopKResults, *PHFSearchTopKResults;

/**
 * @brief Set the face recognition search threshold.
 *
 * This function sets the threshold for face recognition, which determines the sensitivity
 * of the recognition process. A lower threshold may yield more matches but with less confidence.
 *
 * @param threshold The threshold value to set for face recognition (default is 0.48, suitable for
 * access control scenarios).
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubFaceSearchThresholdSetting(float threshold);

/**
 * @brief Perform a one-to-one comparison of two face features.
 *
 * @param session Handle to the session.
 * @param feature1 The first face feature for comparison.
 * @param feature2 The second face feature for comparison.
 * @param result Pointer to the floating-point value where the comparison result will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFaceComparison(HFFaceFeature feature1, HFFaceFeature feature2,
                                                  HPFloat result);

/**
 * @brief Get the length of the face feature.
 *
 * @param num Pointer to an integer where the length of the feature will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFeatureLength(HPInt32 num);

/**
 * @brief Insert a face feature identity into the features group.
 *
 * @param featureIdentity The face feature identity to be inserted.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubInsertFeature(HFFaceFeatureIdentity featureIdentity);

/**
 * @brief Search for the most similar face feature in the features group.
 *
 * @param searchFeature The face feature to be searched.
 * @param confidence Pointer to a floating-point value where the confidence level of the match will
 * be stored.
 * @param mostSimilar Pointer to the most similar face feature identity found.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubFaceSearch(HFFaceFeature searchFeature,
                                                        HPFloat confidence,
                                                        PHFFaceFeatureIdentity mostSimilar);

/**
 * @brief Search for the most similar k facial features in the feature group
 *
 * @param searchFeature The face feature to be searched.
 * @param confidence topK Maximum number of searches
 * @param PHFSearchTopKResults Output search result
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubFaceSearchTopK(HFFaceFeature searchFeature,
                                                            HInt32 topK,
                                                            PHFSearchTopKResults results);

/**
 * @brief Remove a face feature from the features group based on custom ID.
 *
 * @param customId The custom ID of the feature to be removed.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubFaceRemove(HInt32 customId);

/**
 * @brief Update a face feature identity in the features group.
 *
 * @param featureIdentity The face feature identity to be updated.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubFaceUpdate(HFFaceFeatureIdentity featureIdentity);

/**
 * @brief Retrieve a face feature identity from the features group based on custom ID.
 *
 * @param customId The custom ID of the feature.
 * @param identity Pointer to the face feature identity to be retrieved.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubGetFaceIdentity(HInt32 customId,
                                                             PHFFaceFeatureIdentity identity);

/**
 * @brief Get the count of face features in the features group.
 *
 * @param count Pointer to an integer where the count of features will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubGetFaceCount(HInt32 *count);

/**
 * @brief View the face database table.
 *
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubViewDBTable();

/************************************************************************
 * Face Pipeline
 ************************************************************************/

/**
 * @brief Process multiple faces in a pipeline.
 *
 * This function processes multiple faces detected in an image or video frame, applying
 * various face recognition and analysis features as specified in the parameters.
 *
 * @param session Handle to the session.
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param faces Pointer to the structure containing data of multiple detected faces.
 * @param parameter Custom parameters for processing the faces.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFMultipleFacePipelineProcess(HFSession session,
                                                               HFImageStream streamHandle,
                                                               PHFMultipleFaceData faces,
                                                               HFSessionCustomParameter parameter);

/**
 * @brief Process multiple faces in a pipeline with an optional custom option.
 *
 * Similar to HFMultipleFacePipelineProcess, but allows for additional custom options
 * to modify the face processing behavior.
 *
 * @param session Handle to the session.
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param faces Pointer to the structure containing data of multiple detected faces.
 * @param customOption An integer representing a custom option for processing.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFMultipleFacePipelineProcessOptional(HFSession session,
                                                                       HFImageStream streamHandle,
                                                                       PHFMultipleFaceData faces,
                                                                       HInt32 customOption);

/**
 * @brief Struct representing RGB liveness confidence.
 *
 * This struct holds the number of faces and the confidence level of liveness detection
 * for each face, using RGB analysis.
 */
typedef struct HFRGBLivenessConfidence {
    HInt32 num;          ///< Number of faces detected.
    HPFloat confidence;  ///< Confidence level of RGB liveness detection for each face.
} HFRGBLivenessConfidence, *PHFRGBLivenessConfidence;

/**
 * @brief Get the RGB liveness confidence.
 *
 * This function retrieves the confidence level of RGB liveness detection for faces detected
 * in the current context.
 *
 * @param session Handle to the session.
 * @param confidence Pointer to the structure where RGB liveness confidence data will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetRGBLivenessConfidence(HFSession session,
                                                            PHFRGBLivenessConfidence confidence);

/**
 * @brief Struct representing face mask confidence.
 *
 * This struct holds the number of faces and the confidence level of mask detection
 * for each face.
 */
typedef struct HFFaceMaskConfidence {
    HInt32 num;          ///< Number of faces detected.
    HPFloat confidence;  ///< Confidence level of mask detection for each face.
} HFFaceMaskConfidence, *PHFFaceMaskConfidence;

/**
 * @brief Get the face mask confidence.
 *
 * This function retrieves the confidence level of mask detection for faces detected
 * in the current context.
 *
 * @param session Handle to the session.
 * @param confidence Pointer to the structure where face mask confidence data will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceMaskConfidence(HFSession session,
                                                         PHFFaceMaskConfidence confidence);

/**
 * @brief Struct representing face quality predict confidence.
 *
 * This struct holds the number of faces and the confidence level of face quality predict
 * for each face.
 */
typedef struct HFFaceQualityConfidence {
    HInt32 num;          ///< Number of faces detected.
    HPFloat confidence;  ///< Confidence level of face quality predict for each face.
} HFFaceQualityConfidence, *PHFFaceQualityConfidence;

/**
 * @brief Get the face quality predict confidence.
 *
 * This function retrieves the confidence level of face quality predict for faces detected
 * in the current context.
 *
 * @param session Handle to the session.
 * @param confidence Pointer to the structure where face mask confidence data will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceQualityConfidence(HFSession session,
                                                            PHFFaceQualityConfidence confidence);

/**
 * @brief Detect the quality of a face in an image.
 *
 * This function assesses the quality of a detected face, such as its clarity and visibility.
 *
 * @param session Handle to the session.
 * @param singleFace A token representing a single face.
 * @param confidence Pointer to a floating-point value where the quality confidence will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFaceQualityDetect(HFSession session, HFFaceBasicToken singleFace,
                                                     HFloat *confidence);

/**
 * @brief Facial states in the face interaction module.
 */
typedef struct HFFaceIntereactionState {
    HInt32 num;                        ///< Number of faces detected.
    HPFloat leftEyeStatusConfidence;   ///< Left eye state: confidence close to 1 means open, close
                                       ///< to 0 means closed.
    HPFloat rightEyeStatusConfidence;  ///< Right eye state: confidence close to 1 means open, close
                                       ///< to 0 means closed.
} HFFaceIntereactionState, *PHFFaceIntereactionState;

/**
 * @brief Get the prediction results of face interaction.
 * @param session Handle to the session.
 * @param result Facial state prediction results in the face interaction module.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceIntereactionStateResult(HFSession session,
                                                                  PHFFaceIntereactionState result);

/**
 * @brief Actions detected in the face interaction module.
 */
typedef struct HFFaceIntereactionsActions {
    HInt32 num;         ///< Number of actions detected.
    HPInt32 normal;     ///< Normal actions.
    HPInt32 shake;      ///< Shake actions.
    HPInt32 jawOpen;    ///< Jaw open actions.
    HPInt32 headRiase;  ///< Head raise actions.
    HPInt32 blink;      ///< Blink actions.
} HFFaceIntereactionsActions, *PHFFaceIntereactionsActions;

/**
 * @brief Get the prediction results of face interaction actions.
 * @param session Handle to the session.
 * @param actions Facial action prediction results in the face interaction module.
 * @return HResult indicating success or failure of the function call.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceIntereactionActionsResult(
  HFSession session, PHFFaceIntereactionsActions actions);
/**
 * @brief Struct representing face attribute results.
 *
 * This struct holds the race, gender, and age bracket attributes for a detected face.
 */
typedef struct HFFaceAttributeResult {
    HInt32 num;          ///< Number of faces detected.
    HPInt32 race;        ///< Race of the detected face.
                         ///< 0: Black;
                         ///< 1: Asian;
                         ///< 2: Latino/Hispanic;
                         ///< 3: Middle Eastern;
                         ///< 4: White;
    HPInt32 gender;      ///< Gender of the detected face.
                         ///< 0: Female;
                         ///< 1: Male;
    HPInt32 ageBracket;  ///< Age bracket of the detected face.
                         ///< 0: 0-2 years old;
                         ///< 1: 3-9 years old;
                         ///< 2: 10-19 years old;
                         ///< 3: 20-29 years old;
                         ///< 4: 30-39 years old;
                         ///< 5: 40-49 years old;
                         ///< 6: 50-59 years old;
                         ///< 7: 60-69 years old;
                         ///< 8: more than 70 years old;
} HFFaceAttributeResult, *PHFFaceAttributeResult;

/**
 * @brief Get the face attribute results.
 *
 * This function retrieves the attribute results such as race, gender, and age bracket
 * for faces detected in the current context.
 *
 * @param session Handle to the session.
 * @param results Pointer to the structure where face attribute results will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceAttributeResult(HFSession session,
                                                          PHFFaceAttributeResult results);

/************************************************************************
 * System Function
 ************************************************************************/

/**
 * @brief Structure representing the version information of the InspireFace library.
 */
typedef struct HFInspireFaceVersion {
    int major;  ///< Major version number.
    int minor;  ///< Minor version number.
    int patch;  ///< Patch version number.
} HFInspireFaceVersion, *PHFInspireFaceVersion;

/**
 * @brief Function to query the version information of the InspireFace library.
 *
 * This function retrieves the version information of the InspireFace library.
 *
 * @param version Pointer to the structure where the version information will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFQueryInspireFaceVersion(PHFInspireFaceVersion version);

/**
 * @brief SDK built-in log level mode
 * */
typedef enum HFLogLevel {
    HF_LOG_NONE = 0,  // No logging, disables all log output
    HF_LOG_DEBUG,     // Debug level for detailed system information mostly useful for developers
    HF_LOG_INFO,      // Information level for general system information about operational status
    HF_LOG_WARN,      // Warning level for non-critical issues that might need attention
    HF_LOG_ERROR,     // Error level for error events that might still allow the application to
                      // continue running
    HF_LOG_FATAL  // Fatal level for severe error events that will presumably lead the application
                  // to abort
} HFLogLevel;

/**
 * @brief Set the log level built into the SDK.The default is HF LOG DEBUG
 * */
HYPER_CAPI_EXPORT extern HResult HFSetLogLevel(HFLogLevel level);

/**
 * @brief Disable the log function. Like HFSetLogLevel(HF_LOG_NONE)
 * */
HYPER_CAPI_EXPORT extern HResult HFLogDisable();

/********************************DEBUG Utils****************************************/

/**
 * @brief Display an image stream for debugging purposes.
 *
 * This function is used for debugging, allowing the visualization of the image stream
 * as it is being processed. It can be useful to understand the data being received
 * from the camera or image source.
 *
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 */
HYPER_CAPI_EXPORT extern void HFDeBugImageStreamImShow(HFImageStream streamHandle);

/**
 * @brief Decode the image from ImageStream and store it to a disk path.
 *
 * It is used to verify whether there is a problem with image codec, and can quickly perform bug
 * analysis.
 *
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param savePath The path to which the image is written.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFDeBugImageStreamDecodeSave(HFImageStream streamHandle,
                                                              HPath savePath);

/**
 * @brief Display current resource management statistics.
 *
 * This function prints statistics about the resources managed by the ResourceManager,
 * including the total number of created and released sessions and image streams, as well as
 * the count of those that have not been released yet. This can be used for debugging purposes
 * to ensure that resources are being properly managed and to identify potential resource leaks.
 *
 * @return HResult indicating the success or failure of the operation.
 *         Returns HSUCCEED if the statistics were successfully displayed,
 *         otherwise, it may return an error code if there is an issue accessing the resource
 * manager.
 */
HYPER_CAPI_EXPORT extern HResult HFDeBugShowResourceStatistics();

#ifdef __cplusplus
}
#endif

#endif  // HYPERFACEREPO_INSPIREFACE_H
