/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifndef INSPIREFACE_H
#define INSPIREFACE_H

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

#define HF_STATUS_ENABLE 1   ///< The status of the feature is enabled.
#define HF_STATUS_DISABLE 0  ///< The status of the feature is disabled.

#define HF_ENABLE_NONE 0x00000000              ///< Flag to enable no features.
#define HF_ENABLE_FACE_RECOGNITION 0x00000002  ///< Flag to enable face recognition feature.
#define HF_ENABLE_LIVENESS 0x00000004          ///< Flag to enable RGB liveness detection feature.
#define HF_ENABLE_IR_LIVENESS 0x00000008       ///< Flag to enable IR (Infrared) liveness detection feature.
#define HF_ENABLE_MASK_DETECT 0x00000010       ///< Flag to enable mask detection feature.
#define HF_ENABLE_FACE_ATTRIBUTE 0x00000020    ///< Flag to enable face attribute prediction feature.
#define HF_ENABLE_PLACEHOLDER_ 0x00000040      ///< -
#define HF_ENABLE_QUALITY 0x00000080           ///< Flag to enable face quality assessment feature.
#define HF_ENABLE_INTERACTION 0x00000100       ///< Flag to enable interaction feature.
#define HF_ENABLE_FACE_POSE 0x00000200         ///< Flag to enable face pose estimation feature.
#define HF_ENABLE_FACE_EMOTION 0x00000400      ///< Flag to enable face emotion recognition feature.

/************************************************************************
 * Image Stream Function
 *
 * ImageStream directly interacts with algorithm modules, providing image data streams for algorithm module input;
 * ImageStream provides automatic transformation to adapt camera stream rotation angles and common image encoding/decoding format conversion;
 * Camera picture rotation mode.
 * To accommodate the rotation of certain devices, four image rotation modes are provided.
 *
 * 1. ROTATION_0 (No Rotation):
 *    Original Image (w x h):               Scaled Image (w*s x h*s):
 *    A(0,0) ----------- B(w-1,0)          A(0,0) ----------- B(w*s-1,0)
 *    |                           |         |                           |
 *    |        Original           |   =>    |        Scaled             |
 *    |                           |         |                           |
 *    C(0,h-1) --------- D(w-1,h-1)        C(0,h*s-1) ---- D(w*s-1,h*s-1)
 *    Point Mapping: A->A(0,0), B->B(w*s-1,0), C->C(0,h*s-1), D->D(w*s-1,h*s-1)
 *
 * 2. ROTATION_90 (90° Counter-Clockwise):
 *    Original Image (w x h):               Rotated Image (h*s x w*s):
 *    A(0,0) ----------- B(w-1,0)          B(0,0) ----------- A(h*s-1,0)
 *    |                           |         |                           |
 *    |        Original           |   =>    |        Rotated            |
 *    |                           |         |                           |
 *    C(0,h-1) --------- D(w-1,h-1)        D(0,w*s-1) ---- C(h*s-1,w*s-1)
 *    Point Mapping: A->A(h*s-1,0), B->B(0,0), C->C(h*s-1,w*s-1), D->D(0,w*s-1)
 *
 * 3. ROTATION_180 (180° Rotation):
 *    Original Image (w x h):               Rotated Image (w*s x h*s):
 *    A(0,0) ----------- B(w-1,0)          D(0,0) ----------- C(w*s-1,0)
 *    |                           |         |                           |
 *    |        Original           |   =>    |        Rotated            |
 *    |                           |         |                           |
 *    C(0,h-1) --------- D(w-1,h-1)        B(0,h*s-1) ---- A(w*s-1,h*s-1)
 *    Point Mapping: A->A(w*s-1,h*s-1), B->B(0,h*s-1), C->C(w*s-1,0), D->D(0,0)
 *
 * 4. ROTATION_270 (270° Counter-Clockwise):
 *    Original Image (w x h):               Rotated Image (h*s x w*s):
 *    A(0,0) ----------- B(w-1,0)          D(0,0) ----------- C(h*s-1,0)
 *    |                           |         |                           |
 *    |        Original           |   =>    |        Rotated            |
 *    |                           |         |                           |
 *    C(0,h-1) --------- D(w-1,h-1)        B(0,w*s-1) ---- A(h*s-1,w*s-1)
 *    Point Mapping: A->A(h*s-1,w*s-1), B->B(0,w*s-1), C->C(h*s-1,0), D->D(0,0)
 ************************************************************************/

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
    HF_STREAM_I420 = 6,      ///< Image in I420 format.
    HF_STREAM_GRAY = 7,      ///< Image in GRAY format.
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
    HPUInt8 data;          ///< Pointer to the image data stream.
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
HYPER_CAPI_EXPORT extern HResult HFCreateImageStream(PHFImageData data, PHFImageStream handle);

/**
 * @brief Create an empty image stream instance.
 *
 * This function is used to create an instance of a data buffer stream with the given image data.
 *
 * @param handle Pointer to the stream handle that will be returned.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFCreateImageStreamEmpty(PHFImageStream handle);

/**
 * @brief Set the buffer of the image stream.
 *
 * @param handle Pointer to the stream handle.
 * @param buffer Pointer to the buffer.
 * @param width Width of the image.
 * @param height Height of the image.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFImageStreamSetBuffer(HFImageStream handle, HPUInt8 buffer, HInt32 width, HInt32 height);

/**
 * @brief Set the rotation of the image stream.
 *
 * @param handle Pointer to the stream handle.
 * @param rotation Rotation angle of the image.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFImageStreamSetRotation(HFImageStream handle, HFRotation rotation);

/**
 * @brief Set the format of the image stream.
 *
 * @param handle Pointer to the stream handle.
 * @param format Format of the image.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFImageStreamSetFormat(HFImageStream handle, HFImageFormat format);

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
 * Image Bitmap Function
 *
 * Provides a simple Bitmap interface wrapper that copies image data when creating objects and requires manual release.
 * Provides interfaces for copying, drawing, displaying (OpenCV-GUI), writing to files, and converting to/from ImageStream.
 ************************************************************************/

/**
 * @brief Struct for image bitmap data.
 */
typedef struct HFImageBitmapData {
    HPUInt8 data;     ///< Pointer to the image data.
    HInt32 width;     ///< Width of the image.
    HInt32 height;    ///< Height of the image.
    HInt32 channels;  ///< Number of channels in the image, only support 3 channels or 1 channel.
} HFImageBitmapData, *PHFImageBitmapData;

/**
 * @brief Create a image bitmap from data, default pixel format is BGR.
 *
 * @param data Pointer to the image bitmap data structure.
 * @param handle Pointer to the image bitmap handle that will be returned.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFCreateImageBitmap(PHFImageBitmapData data, PHFImageBitmap handle);

/**
 * @brief Create a image bitmap from file path, default pixel format is BGR.
 *
 * @param filePath The path to the image file.
 * @param channels The number of channels in the image, only support 3 channels or 1 channel.
 * @param handle Pointer to the image bitmap handle that will be returned.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFCreateImageBitmapFromFilePath(HPath filePath, HInt32 channels, PHFImageBitmap handle);

/**
 * @brief Copy an image bitmap.
 *
 * @param handle Pointer to the image bitmap handle.
 * @param copyHandle Pointer to the image bitmap handle that will be returned.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFImageBitmapCopy(HFImageBitmap handle, PHFImageBitmap copyHandle);

/**
 * @brief Release the image bitmap.
 *
 * @param handle Pointer to the image bitmap handle.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFReleaseImageBitmap(HFImageBitmap handle);

/**
 * @brief Create a image stream from image bitmap.
 *
 * @param handle Pointer to the image bitmap handle.
 * @param rotation The rotation angle of the image.
 * @param streamHandle Pointer to the image stream handle that will be returned.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFCreateImageStreamFromImageBitmap(HFImageBitmap handle, HFRotation rotation, PHFImageStream streamHandle);

/**
 * @brief Create a image bitmap from image stream.
 *
 * @param streamHandle Pointer to the image stream handle.
 * @param handle Pointer to the image bitmap handle that will be returned.
 * @param is_rotate Whether to rotate the image.
 * @param scale The scale of the image.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFCreateImageBitmapFromImageStreamProcess(HFImageStream streamHandle, PHFImageBitmap handle, HInt32 is_rotate,
                                                                           HFloat scale);

/**
 * @brief Write the image bitmap to a file.
 *
 * @param handle Pointer to the image bitmap handle.
 * @param filePath The path to the image file.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFImageBitmapWriteToFile(HFImageBitmap handle, HPath filePath);

/**
 * @brief Draw a rectangle on the image bitmap.
 *
 * @param handle Pointer to the image bitmap handle.
 * @param rect The rectangle to be drawn.
 * @param color The color of the rectangle.
 * @param thickness The thickness of the rectangle.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFImageBitmapDrawRect(HFImageBitmap handle, HFaceRect rect, HColor color, HInt32 thickness);

/**
 * @brief Draw a circle on the image bitmap.
 *
 * @param handle Pointer to the image bitmap handle.
 * @param point The center point of the circle.
 * @param radius The radius of the circle.
 * @param color The color of the circle.
 * @param thickness The thickness of the circle.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFImageBitmapDrawCircleF(HFImageBitmap handle, HPoint2f point, HInt32 radius, HColor color, HInt32 thickness);
HYPER_CAPI_EXPORT extern HResult HFImageBitmapDrawCircle(HFImageBitmap handle, HPoint2i point, HInt32 radius, HColor color, HInt32 thickness);

/**
 * @brief Get the data of the image bitmap.
 *
 * @param handle Pointer to the image bitmap handle.
 * @param data Pointer to the image bitmap data structure.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFImageBitmapGetData(HFImageBitmap handle, PHFImageBitmapData data);

/**
 * @brief Show the image bitmap.
 *
 * @param handle Pointer to the image bitmap handle, must rely on opencv's gui functionality
 * @param title The title of the image.
 * @param delay The delay time of the image.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFImageBitmapShow(HFImageBitmap handle, HString title, HInt32 delay);

/************************************************************************
 * Resource Function
 *
 * The resource module is a system-level module that manages the life cycle of all resources.
 * It is responsible for loading and unloading resources, and managing the memory of resources.
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
 * @brief Reload InspireFace SDK
 * Reload the InspireFace SDK, releasing all allocated resources.
 * @param resourcePath Initializes the path to the resource file that needs to be loaded
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFReloadInspireFace(HPath resourcePath);

/**
 * @brief Terminate InspireFace SDK
 * Terminate the InspireFace SDK, releasing all allocated resources.
 * This should be called at the end of your program to ensure proper cleanup.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFTerminateInspireFace();

/**
 * @brief Query InspireFace SDK launch status
 * Query the launch status of the InspireFace SDK.
 * @param status Pointer to the status variable that will be returned.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFQueryInspireFaceLaunchStatus(HPInt32 status);

/************************************************************************
 * Extended Interface Based on Third-party Hardware Devices
 *
 * According to different manufacturers' devices, manufacturers typically perform deep customization and optimization, such as neural network
 * inference computation, geometric image acceleration computation, and deeply customized device interfaces, etc. These types of functionalities are
 * usually difficult to abstract, so they are placed in extension module APIs, involving hybrid computing, heterogeneous computing, multi-device
 * computing, and other features.
 ************************************************************************/

/**
 * @brief Check whether RGA is enabled during compilation
 * @param status Pointer to the status variable that will be returned.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFQueryExpansiveHardwareRGACompileOption(HPInt32 enable);

/**
 * @brief Set the rockchip dma heap path
 * By default, we have already configured the DMA Heap address used by RGA on RK devices.
 * If you wish to customize this address, you can modify it through this API.
 * @param path The path to the rockchip dma heap
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFSetExpansiveHardwareRockchipDmaHeapPath(HPath path);

/**
 * @brief Query the rockchip dma heap path
 * @param path Pointer to a pre-allocated character array that will store the returned path.
 * The array should be at least 256 bytes in size.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFQueryExpansiveHardwareRockchipDmaHeapPath(HString path);

/**
 * @brief Enum for image processing backend.
 */
typedef enum HFImageProcessingBackend {
    HF_IMAGE_PROCESSING_CPU = 0,  ///< CPU backend(Default)
    HF_IMAGE_PROCESSING_RGA = 1,  ///< Rockchip RGA backend(Hardware support is mandatory)
} HFImageProcessingBackend;

/**
 * @brief Switch the image processing backend, must be called before HFCreateInspireFaceSession.
 * @param backend The image processing backend to be set.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFSwitchImageProcessingBackend(HFImageProcessingBackend backend);

/**
 * @brief Set the image process aligned width, must be called before HFCreateInspireFaceSession.
 * @param width The image process aligned width to be set.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFSetImageProcessAlignedWidth(HInt32 width);

/**
 * @brief Enum for Apple CoreML inference mode.
 */
typedef enum HFAppleCoreMLInferenceMode {
    HF_APPLE_COREML_INFERENCE_MODE_CPU = 0,  ///< CPU Only.
    HF_APPLE_COREML_INFERENCE_MODE_GPU = 1,  ///< GPU first.
    HF_APPLE_COREML_INFERENCE_MODE_ANE = 2,  ///< Automatic selection, ANE first.
} HFAppleCoreMLInferenceMode;

/**
 * @brief Set the Apple CoreML inference mode, must be called before HFCreateInspireFaceSession.
 * @param mode The inference mode to be set.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFSetAppleCoreMLInferenceMode(HFAppleCoreMLInferenceMode mode);

/**
 * @brief Set the CUDA device id, must be called before HFCreateInspireFaceSession.
 * @param device_id The device id to be set.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFSetCudaDeviceId(HInt32 device_id);

/**
 * @brief Get the CUDA device id, must be called after HFCreateInspireFaceSession.
 * @param device_id Pointer to the device id to be returned.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFGetCudaDeviceId(HPInt32 device_id);

/**
 * @brief Print the CUDA device information.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFPrintCudaDeviceInfo();

/**
 * @brief Get the number of CUDA devices.
 * @param num_devices Pointer to the number of CUDA devices to be returned.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFGetNumCudaDevices(HPInt32 num_devices);

/**
 * @brief Check if the CUDA device is supported.
 * @param support The support flag to be checked.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFCheckCudaDeviceSupport(HPInt32 is_support);

/************************************************************************
 * FaceSession Function
 *
 * FaceSession is responsible for all face image algorithm-related functions,
 * including face detection, face alignment, face recognition, face quality detection, face attribute prediction, etc.
 * FaceSession supports flexible configuration, allowing you to enable or disable certain functions, and also set parameters for certain functions.
 * In concurrent scenarios, multiple sessions can be created, each session can run independently without interfering with each other.
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
    HInt32 enable_detect_mode_landmark;  ///< Enable landmark detection in detection mode
    HInt32 enable_face_pose;             ///< Enable face pose estimation feature.
    HInt32 enable_face_emotion;          ///< Enable face emotion recognition feature.
} HFSessionCustomParameter, *PHFSessionCustomParameter;

/**
 * @brief Enumeration for face detection modes.
 */
typedef enum HFDetectMode {
    HF_DETECT_MODE_ALWAYS_DETECT,       ///< Image detection mode, always detect, applicable to images.
    HF_DETECT_MODE_LIGHT_TRACK,         ///< Video detection mode, face tracking, applicable to video
                                        ///< streaming, front camera.
    HF_DETECT_MODE_TRACK_BY_DETECTION,  ///< Video detection mode, face tracking, applicable to high
                                        ///< resolution, monitoring, capturing
                                        //   (You need a specific option turned on at compile time
                                        //   to use it).
} HFDetectMode;

/**
 * @brief Enum for landmark engine.
 */
typedef enum HFSessionLandmarkEngine {
    HF_LANDMARK_HYPLMV2_0_25 = 0,             ///< Hyplmkv2 0.25, default
    HF_LANDMARK_HYPLMV2_0_50 = 1,             ///< Hyplmkv2 0.50
    HF_LANDMARK_INSIGHTFACE_2D106_TRACK = 2,  ///< InsightFace 2d106 track
} HFSessionLandmarkEngine;

/**
 * @brief Global switch the landmark engine. Set it globally before creating a session.
 *  If it is changed, a new session needs to be created for it to be effective.
 * @param engine The landmark engine to be set.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFSwitchLandmarkEngine(HFSessionLandmarkEngine engine);

/**
 * @brief Enum for supported pixel levels for face detection.
 */
typedef struct HFFaceDetectPixelList {
    HInt32 pixel_level[20];
    HInt32 size;
} HFFaceDetectPixelList, *PHFFaceDetectPixelList;

/**
 * @brief Query the supported pixel levels for face detection. It must be used before starting.
 * @param pixel_levels Pointer to the array of supported pixel levels.
 * @return HResult indicating the success or failure of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFQuerySupportedPixelLevelsForFaceDetection(PHFFaceDetectPixelList pixel_levels);

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
HYPER_CAPI_EXPORT extern HResult HFCreateInspireFaceSession(HFSessionCustomParameter parameter, HFDetectMode detectMode, HInt32 maxDetectFaceNum,
                                                            HInt32 detectPixelLevel, HInt32 trackByDetectModeFPS, PHFSession handle);

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
HYPER_CAPI_EXPORT extern HResult HFCreateInspireFaceSessionOptional(HOption customOption, HFDetectMode detectMode, HInt32 maxDetectFaceNum,
                                                                    HInt32 detectPixelLevel, HInt32 trackByDetectModeFPS, PHFSession handle);

/**
 * @brief Release the session.
 *
 * @param handle Handle to the session to be released.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFReleaseInspireFaceSession(HFSession handle);

/************************************************************************
 * FaceTrack Module
 *
 * FaceTrack provides the most basic face image algorithm functions, such as face detection, tracking, landmark detection, etc.
 * FaceTrack is independent of FaceSession, and can be used independently.
 ************************************************************************/

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
    HPFloat roll;   ///< Roll angle of the face.
    HPFloat yaw;    ///< Yaw angle of the face.
    HPFloat pitch;  ///< Pitch angle of the face.
} HFFaceEulerAngle;

/**
 * @brief Struct for holding data of multiple detected faces.
 *
 * This struct stores the data related to multiple faces detected, including the number of faces,
 * their bounding rectangles, track IDs, angles, and tokens.
 */
typedef struct HFMultipleFaceData {
    HInt32 detectedNum;        ///< Number of faces detected.
    PHFaceRect rects;          ///< Array of bounding rectangles for each face.
    HPInt32 trackIds;          ///< Array of track IDs for each face.
    HPInt32 trackCounts;       ///< Array of track counts for each face.
    HPFloat detConfidence;     ///< Array of detection confidence for each face.
    HFFaceEulerAngle angles;   ///< Euler angles for each face.
    PHFFaceBasicToken tokens;  ///< Tokens associated with each face.
} HFMultipleFaceData, *PHFMultipleFaceData;

/**
 * @brief Clear the tracking face
 * @param session Handle to the session.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionClearTrackingFace(HFSession session);

/**
 * @brief Set the track lost recovery mode(only for LightTrack mode, default is false(0))
 * @param session Handle to the session.
 * @param enable The track lost recovery mode value (0: disable, 1: enable)
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetTrackLostRecoveryMode(HFSession session, HInt32 enable);

/**
 * @brief Set the light track confidence threshold(only for LightTrack mode, default is 0.1)
 * @param session Handle to the session.
 * @param value The light track confidence threshold value
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetLightTrackConfidenceThreshold(HFSession session, HFloat value);

/**
 * @brief Set the track preview size in the session, it works with face detection and tracking
 * algorithms. Default preview size is 192(px).
 *
 * @param session Handle to the session.
 * @param previewSize The size of the preview for tracking.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetTrackPreviewSize(HFSession session, HInt32 previewSize);

/**
 * @brief Get the track preview size in the session.
 * @param session Handle to the session.
 * @param previewSize The size of the preview for tracking.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionGetTrackPreviewSize(HFSession session, HPInt32 previewSize);

/**
 * @brief Set the minimum number of face pixels that the face detector can capture, and people below
 * this number will be filtered.
 *
 * @param session Handle to the session.
 * @param minSize The minimum pixel value, default value is 0.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetFilterMinimumFacePixelSize(HFSession session, HInt32 minSize);

/**
 * @brief Set the face detect threshold in the session.
 *
 * @param session Handle to the session.
 * @param detectMode The mode of the detection mode for tracking.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetFaceDetectThreshold(HFSession session, HFloat threshold);

/**
 * @brief Set the track mode smooth ratio in the session. default value is  0.05
 *
 * @param session Handle to the session.
 * @param ratio The smooth ratio value.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetTrackModeSmoothRatio(HFSession session, HFloat ratio);

/**
 * @brief Set the track mode num smooth cache frame in the session. default value is 5
 *
 * @param session Handle to the session.
 * @param num The num smooth cache frame value.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetTrackModeNumSmoothCacheFrame(HFSession session, HInt32 num);

/**
 * @brief Set the track model detect interval in the session. default value is 20
 *
 * @param session Handle to the session.
 * @param num The detect interval value.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionSetTrackModeDetectInterval(HFSession session, HInt32 num);

/**
 * @brief Run face tracking in the session.
 *
 * @param session Handle to the session.
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param results Pointer to the structure where the results will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFExecuteFaceTrack(HFSession session, HFImageStream streamHandle, PHFMultipleFaceData results);

/**
 * @brief Gets the size of the debug preview image for the last face detection in the session.
 * @param session Handle to the session.
 * @param size The size of the preview for tracking.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFSessionLastFaceDetectionGetDebugPreviewImageSize(HFSession session, HPInt32 size);

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
HYPER_CAPI_EXPORT extern HResult HFCopyFaceBasicToken(HFFaceBasicToken token, HPBuffer buffer, HInt32 bufferSize);

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
HYPER_CAPI_EXPORT extern HResult HFGetFaceDenseLandmarkFromFaceToken(HFFaceBasicToken singleFace, PHPoint2f landmarks, HInt32 num);

/**
 * @brief Get the five key points from the face token.
 * @param singleFace Basic token representing a single face.
 * @param landmarks Pre-allocated memory address of the array for 2D floating-point coordinates.
 * @param num Number of landmark points
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceFiveKeyPointsFromFaceToken(HFFaceBasicToken singleFace, PHPoint2f landmarks, HInt32 num);

/**
 * @brief Set the enable cost spend
 * @param value The enable cost spend value
 * @return HResult Status code of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFSessionSetEnableTrackCostSpend(HFSession session, HInt32 value);

/**
 * @brief Print the cost spend
 * @param session The session handle
 * @return HResult Status code of the operation.
 * */
HYPER_CAPI_EXPORT extern HResult HFSessionPrintTrackCostSpend(HFSession session);

/************************************************************************
 * Face Recognition Module
 *
 * The interface of the face recognition module depends on FaceSession,
 * providing face feature extraction, face alignment image processing, and face comparison interfaces.
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
HYPER_CAPI_EXPORT extern HResult HFFaceFeatureExtract(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace,
                                                      PHFFaceFeature feature);

/**
 * @brief Extract face features to the HFFaceFeature that has applied for memory in advance.
 * @param session Handle to the session.
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param singleFace Basic token representing a single face.
 * @param feature Pointer to the buffer where the extracted feature will be copied.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFaceFeatureExtractTo(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace,
                                                        HFFaceFeature feature);

/**
 * @brief Extract a face feature from a given face and copy it to the provided feature buffer.
 *
 * @param session Handle to the session.
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param singleFace Basic token representing a single face.
 * @param feature Pointer to the buffer where the extracted feature will be copied.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFaceFeatureExtractCpy(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace, HPFloat feature);

/**
 * @brief Create a face feature. Will allocate memory.
 * @param feature Pointer to the face feature.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFCreateFaceFeature(PHFFaceFeature feature);

/**
 * @brief Release a face feature. Only the features created through the HFCreateFaceFeature need to be processed.
 * @param feature Pointer to the face feature.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFReleaseFaceFeature(PHFFaceFeature feature);

/**
 * @brief Get the face alignment image.
 * @param session Handle to the session.
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param singleFace Basic token representing a single face.
 * @param handle Pointer to the handle that will be returned.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFaceGetFaceAlignmentImage(HFSession session, HFImageStream streamHandle, HFFaceBasicToken singleFace,
                                                             PHFImageBitmap handle);

/**
 * @brief Use the aligned face image to extract face features to the HFFaceFeature that has applied memory in advance.
 * @param session Handle to the session.
 * @param streamHandle Handle to the data buffer representing the camera stream component.
 * @param feature Pointer to the buffer where the extracted feature will be copied.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFaceFeatureExtractWithAlignmentImage(HFSession session, HFImageStream streamHandle, HFFaceFeature feature);

/************************************************************************
 * Feature Hub
 *
 * FeatureHub is a built-in global lightweight face feature vector management functionality
 * provided in the InspireFace-SDK. It supports basic face feature search, deletion, and
 * modification functions, and offers two optional data storage modes: an in-memory model and a
 * persistence model. If you have simple storage needs, you can enable it.
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
 * @brief Primary key mode for face feature management.
 */
typedef enum HFPKMode {
    HF_PK_AUTO_INCREMENT = 0,  ///< Auto-increment mode for primary key.
    HF_PK_MANUAL_INPUT,        ///< Manual input mode for primary key.
} HFPKMode;

/**
 * @brief Struct for database configuration.
 *
 * This struct holds the configuration settings for using a database in the face recognition context.
 */
typedef struct HFFeatureHubConfiguration {
    HFPKMode primaryKeyMode;    ///< Primary key mode(The id increment mode is recommended)
    HInt32 enablePersistence;   ///< Flag to enable or disable the use of the database.
    HString persistenceDbPath;  ///< Path to the database file.
    HFloat searchThreshold;     ///< Threshold for face search
    HFSearchMode searchMode;    ///< Mode of face search
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
    HFaceId id;              ///< If you use automatic assignment id mode when inserting, ignore it.
    PHFFaceFeature feature;  ///< Pointer to the face feature.
    // HString tag;                 ///< Not supported yet
} HFFaceFeatureIdentity, *PHFFaceFeatureIdentity;

/**
 * Search structure for top-k mode
 * */
typedef struct HFSearchTopKResults {
    HInt32 size;         ///< The number of faces searched
    HPFloat confidence;  ///< Search confidence(it has already been filtered once by the threshold)
    HPFaceId ids;        ///< Searched face ids
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
HYPER_CAPI_EXPORT extern HResult HFFeatureHubFaceSearchThresholdSetting(HFloat threshold);

/**
 * @brief Perform a one-to-one comparison of two face features.
 *  Result is a cosine similarity score, not a percentage similarity.
 *
 * @param session Handle to the session.
 * @param feature1 The first face feature for comparison.
 * @param feature2 The second face feature for comparison.
 * @param result Pointer to the floating-point value where the comparison result will be stored.
 *               The result is a cosine similarity score, not a percentage similarity.
 *               The score ranges from -1 to 1, where 1 indicates identical features,
 *               0 indicates orthogonal features, and -1 indicates opposite features.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFaceComparison(HFFaceFeature feature1, HFFaceFeature feature2, HPFloat result);

/**
 * @brief Get recommended cosine threshold from loaded resource.
 *  Use it to determine face similarity. Note: it's just a reference and may not be optimal for your task.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetRecommendedCosineThreshold(HPFloat threshold);

/**
 * @brief Convert cosine similarity to percentage similarity.
 *  This is a nonlinear transformation function. You can adjust curve parameters to map the similarity distribution you need.
 * @note The conversion parameters are primarily read from the Resource file configuration, as different models
 *       have different conversion parameters. The parameters provided in the Resource file are only reference
 *       values. If they do not meet your specific use case requirements, you can implement your own conversion
 *       function.
 * @param similarity The cosine similarity score.
 * @param result Pointer to the floating-point value where the percentage similarity will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFCosineSimilarityConvertToPercentage(HFloat similarity, HPFloat result);

/**
 * @brief Similarity converter configuration.
 */
typedef struct HFSimilarityConverterConfig {
    HFloat threshold;    ///< If you think that the threshold for judging the same person using cosine is some value such as 0.42,
                         // you need to convert him to a percentage of 0.6(pass), you can modify it.
    HFloat middleScore;  ///< Cosine threshold converted to a percentage reference value,
                         // usually set 0.6 or 0.5, greater than it indicates similar, pass
    HFloat steepness;    ///< Steepness of the curve, usually set 8.0
    HFloat outputMin;    ///< Minimum value of output range, usually set 0.01
    HFloat outputMax;    ///< Maximum value of output range, usually set 1.0
} HFSimilarityConverterConfig, *PHFSimilarityConverterConfig;

/**
 * @brief Update the similarity converter configuration.
 * @note The default configuration is loaded from the resource file during initialization.
 *       This function allows you to override those default settings if needed.
 * @param config The new similarity converter configuration to apply.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFUpdateCosineSimilarityConverter(HFSimilarityConverterConfig config);

/**
 * @brief Get the similarity converter configuration.
 * @param config Pointer to the similarity converter configuration to be filled.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetCosineSimilarityConverter(PHFSimilarityConverterConfig config);

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
HYPER_CAPI_EXPORT extern HResult HFFeatureHubInsertFeature(HFFaceFeatureIdentity featureIdentity, HPFaceId allocId);

/**
 * @brief Search for the most similar face feature in the features group.
 *
 * @param searchFeature The face feature to be searched.
 * @param confidence Pointer to a floating-point value where the confidence level of the match will
 * be stored.
 * @param mostSimilar Pointer to the most similar face feature identity found.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubFaceSearch(HFFaceFeature searchFeature, HPFloat confidence, PHFFaceFeatureIdentity mostSimilar);

/**
 * @brief Search for the most similar k facial features in the feature group
 *
 * @param searchFeature The face feature to be searched.
 * @param confidence topK Maximum number of searches
 * @param PHFSearchTopKResults Output search result
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubFaceSearchTopK(HFFaceFeature searchFeature, HInt32 topK, PHFSearchTopKResults results);

/**
 * @brief Remove a face feature from the features group based on custom ID.
 *
 * @param ID The custom ID of the feature to be removed.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubFaceRemove(HFaceId id);

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
HYPER_CAPI_EXPORT extern HResult HFFeatureHubGetFaceIdentity(HFaceId customId, PHFFaceFeatureIdentity identity);

/**
 * @brief Get the count of face features in the features group.
 *
 * @param count Pointer to an integer where the count of features will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubGetFaceCount(HPInt32 count);

/**
 * @brief View the face database table.
 *
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubViewDBTable();

/**
 * @brief Struct representing the existing ids in the database.
 */
typedef struct HFFeatureHubExistingIds {
    HInt32 size;   ///< The number of ids
    HPFaceId ids;  ///< The ids
} HFFeatureHubExistingIds, *PHFFeatureHubExistingIds;

/**
 * @brief Get all ids in the database.
 * @param ids Output parameter to store the ids.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFFeatureHubGetExistingIds(PHFFeatureHubExistingIds ids);

/************************************************************************
 * Face Pipeline Module
 *
 * FacePipeline depends on FaceSession, providing extended business for face image algorithms,
 * supporting some face attributes, such as face mask detection, face quality detection, face attribute prediction, etc.
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
HYPER_CAPI_EXPORT extern HResult HFMultipleFacePipelineProcess(HFSession session, HFImageStream streamHandle, PHFMultipleFaceData faces,
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
HYPER_CAPI_EXPORT extern HResult HFMultipleFacePipelineProcessOptional(HFSession session, HFImageStream streamHandle, PHFMultipleFaceData faces,
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
HYPER_CAPI_EXPORT extern HResult HFGetRGBLivenessConfidence(HFSession session, PHFRGBLivenessConfidence confidence);

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
HYPER_CAPI_EXPORT extern HResult HFGetFaceMaskConfidence(HFSession session, PHFFaceMaskConfidence confidence);

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
HYPER_CAPI_EXPORT extern HResult HFGetFaceQualityConfidence(HFSession session, PHFFaceQualityConfidence confidence);

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
HYPER_CAPI_EXPORT extern HResult HFFaceQualityDetect(HFSession session, HFFaceBasicToken singleFace, HPFloat confidence);

/**
 * @brief Facial states in the face interaction module.
 */
typedef struct HFFaceInteractionState {
    HInt32 num;                        ///< Number of faces detected.
    HPFloat leftEyeStatusConfidence;   ///< Left eye state: confidence close to 1 means open, close
                                       ///< to 0 means closed.
    HPFloat rightEyeStatusConfidence;  ///< Right eye state: confidence close to 1 means open, close
                                       ///< to 0 means closed.
} HFFaceInteractionState, *PHFFaceInteractionState;

/**
 * @brief Get the prediction results of face interaction.
 * @param session Handle to the session.
 * @param result Facial state prediction results in the face interaction module.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceInteractionStateResult(HFSession session, PHFFaceInteractionState result);

/**
 * @brief Actions detected in the face interaction module.
 */
typedef struct HFFaceInteractionsActions {
    HInt32 num;         ///< Number of actions detected.
    HPInt32 normal;     ///< Normal actions.
    HPInt32 shake;      ///< Shake actions.
    HPInt32 jawOpen;    ///< Jaw open actions.
    HPInt32 headRaise;  ///< Head raise actions.
    HPInt32 blink;      ///< Blink actions.
} HFFaceInteractionsActions, *PHFFaceInteractionsActions;

/**
 * @brief Get the prediction results of face interaction actions.
 * @param session Handle to the session.
 * @param actions Facial action prediction results in the face interaction module.
 * @return HResult indicating success or failure of the function call.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceInteractionActionsResult(HFSession session, PHFFaceInteractionsActions actions);

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
HYPER_CAPI_EXPORT extern HResult HFGetFaceAttributeResult(HFSession session, PHFFaceAttributeResult results);

/**
 * @brief Struct representing face emotion results.
 */
typedef struct HFFaceEmotionResult {
    HInt32 num;       ///< Number of faces detected.
    HPInt32 emotion;  ///< Emotion of the detected face.
                      ///< 0: Neutral;
                      ///< 1: Happy;
                      ///< 2: Sad;
                      ///< 3: Surprise;
                      ///< 4: Fear;
                      ///< 5: Disgust;
                      ///< 6: Anger;
} HFFaceEmotionResult, *PHFFaceEmotionResult;

/**
 * @brief Get the face emotion result.
 * @param session Handle to the session.
 * @param result Pointer to the structure where face emotion results will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFGetFaceEmotionResult(HFSession session, PHFFaceEmotionResult result);

/************************************************************************
 * System Function
 ************************************************************************/

/**
 * @brief Structure representing the version information of the InspireFace library.
 */
typedef struct HFInspireFaceVersion {
    HInt32 major;  ///< Major version number.
    HInt32 minor;  ///< Minor version number.
    HInt32 patch;  ///< Patch version number.
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
 * @brief Struct representing the extended information of the InspireFace library.
 */
typedef struct HFInspireFaceExtendedInformation {
    HChar information[256];
    // TODO: Add more information
} HFInspireFaceExtendedInformation, *PHFInspireFaceExtendedInformation;

/**
 * @brief Get the extended information of the InspireFace library.
 *
 * This function retrieves the extended information of the InspireFace library.
 */
HYPER_CAPI_EXPORT extern HResult HFQueryInspireFaceExtendedInformation(PHFInspireFaceExtendedInformation information);

/**
 * @brief SDK built-in log level mode
 * */
typedef enum HFLogLevel {
    HF_LOG_NONE = 0,  // No logging, disables all log output
    HF_LOG_DEBUG,     // Debug level for detailed system information mostly useful for developers
    HF_LOG_INFO,      // Information level for general system information about operational status
    HF_LOG_WARN,      // Warning level for non-critical issues that might need attention
    HF_LOG_ERROR,     // Error level for error events that might still allow the application to continue running
    HF_LOG_FATAL      // Fatal level for severe error events that will presumably lead the application to abort
} HFLogLevel;

/**
 * @brief Set the log level built into the SDK.The default is HF LOG DEBUG
 * */
HYPER_CAPI_EXPORT extern HResult HFSetLogLevel(HFLogLevel level);

/**
 * @brief Disable the log function. Like HFSetLogLevel(HF_LOG_NONE)
 * */
HYPER_CAPI_EXPORT extern HResult HFLogDisable();

/**
 * @brief Print the log.
 * @param level The log level.
 * @param format The log format.
 * @param ... The log arguments.
 * @warning The maximum buffer size for log messages is 1024 bytes. Messages longer than this will be truncated.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFLogPrint(HFLogLevel level, HFormat format, ...);

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
HYPER_CAPI_EXPORT extern HResult HFDeBugImageStreamDecodeSave(HFImageStream streamHandle, HPath savePath);

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

/**
 * @brief Get the count of unreleased sessions.
 *
 * This function retrieves the count of sessions that have not been released yet.
 *
 * @param count Pointer to an integer where the count of unreleased sessions will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFDeBugGetUnreleasedSessionsCount(HPInt32 count);

/**
 * @brief Get the list of unreleased sessions.
 *
 * This function retrieves the list of sessions that have not been released yet.
 *
 * @param sessions Pointer to an array where the unreleased sessions will be stored.
 * @param count The number of sessions to retrieve.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFDeBugGetUnreleasedSessions(PHFSession sessions, HInt32 count);

/**
 * @brief Get the count of unreleased image streams.
 *
 * This function retrieves the count of image streams that have not been released yet.
 *
 * @param count Pointer to an integer where the count of unreleased image streams will be stored.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFDeBugGetUnreleasedStreamsCount(HPInt32 count);

/**
 * @brief Get the list of unreleased image streams.
 *
 * This function retrieves the list of image streams that have not been released yet.
 *
 * @param streams Pointer to an array where the unreleased image streams will be stored.
 * @param count The number of image streams to retrieve.
 * @return HResult indicating the success or failure of the operation.
 */
HYPER_CAPI_EXPORT extern HResult HFDeBugGetUnreleasedStreams(PHFImageStream streams, HInt32 count);

#ifdef __cplusplus
}
#endif

#endif  // INSPIREFACE_H
