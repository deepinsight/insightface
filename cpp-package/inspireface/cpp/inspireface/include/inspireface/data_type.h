/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FACE_DATATYPE_H
#define INSPIRE_FACE_DATATYPE_H

#include <inspirecv/inspirecv.h>

#include <cstdint>
#if defined(_WIN32) && (defined(_DEBUG) || defined(DEBUG))
#define _CRTDBG_MAP_ALLOC
#include "crtdbg.h"
#endif

#ifndef INSPIRE_API
#define INSPIRE_API
#endif

#if defined(_WIN32)
#ifdef ISF_BUILD_SHARED_LIBS
#define INSPIRE_API_EXPORT __declspec(dllexport)
#else
#define INSPIRE_API_EXPORT
#endif
#else
#define INSPIRE_API_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace inspire {

/**
 * @defgroup DataType Definitions
 * @brief Defines various data types used in the HyperFace project.
 * @{
 */

#if !defined(int64)
/** @typedef int64
 *  @brief 64-bit integer type.
 */
typedef int64_t int64;
#endif

#if !defined(uint64)
/** @typedef uint64
 *  @brief 64-bit unsigned integer type.
 */
typedef uint64_t uint64;
#endif

#if !defined(int32)
/** @typedef int32
 *  @brief 32-bit integer type.
 */
typedef int32_t int32;
#endif

#if !defined(uint32)
/** @typedef uint32
 *  @brief 32-bit unsigned integer type.
 */
typedef uint32_t uint32;
#endif

#if !defined(int8)
/** @typedef int8
 *  @brief 8-bit integer type.
 */
typedef int8_t int8;
#endif

#if !defined(uint8)
/** @typedef uint8
 *  @brief 8-bit unsigned integer type.
 */
typedef uint8_t uint8;
#endif

/** @typedef ByteArray
 *  @brief Type definition for a byte array (vector of chars).
 */
typedef std::vector<char> ByteArray;

/** @typedef Point2i
 *  @brief 2D coordinate point with integer precision.
 */
typedef inspirecv::Point2i Point2i;

/** @typedef Point2f
 *  @brief 2D coordinate point with float precision.
 */
typedef inspirecv::Point2f Point2f;

/** @typedef PointsList2i
 *  @brief List of 2D coordinate points with integer precision.
 */
typedef std::vector<Point2i> PointsList2i;

/** @typedef PointsList2f
 *  @brief List of 2D coordinate points with float precision.
 */
typedef std::vector<Point2f> PointsList2f;

/** @typedef Contours2i
 *  @brief Contours represented as a list of 2D integer points.
 */
typedef std::vector<PointsList2i> Contours2i;

/** @typedef Contours2f
 *  @brief Contours represented as a list of 2D float points.
 */
typedef std::vector<PointsList2f> Contours2f;

/** @typedef Textures2i
 *  @brief Texture lines represented as integer contours.
 */
typedef Contours2i Textures2i;

/** @typedef AnyTensorFp32
 *  @brief Generic tensor representation using a vector of floats.
 */
typedef std::vector<float> AnyTensorFp32;

/** @typedef ImageBitmap
 *  @brief Image bitmap representation.
 */
typedef inspirecv::Image ImageBitmap;

/** @typedef Rectangle
 *  @brief Rectangle representation using integer values.
 */
typedef inspirecv::Rect<int> Rectangle;

/** @typedef Size
 *  @brief Size representation using integer values.
 */
typedef inspirecv::Size<int> Size;

/** @typedef Embedded
 *  @brief Dense vector for feature embedding.
 */
typedef std::vector<float> Embedded;

/** @typedef EmbeddedList
 *  @brief List of dense vectors for feature embedding.
 */
typedef std::vector<Embedded> EmbeddedList;

/** @typedef String
 *  @brief String type definition.
 */
typedef std::string String;

/** @typedef IndexList
 *  @brief List of indices.
 */
typedef std::vector<int> IndexList;

/**
 * @enum DetectMode
 * @brief Enumeration for different detection modes.
 */
enum DetectModuleMode {
    DETECT_MODE_ALWAYS_DETECT = 0,  ///< Detection mode: Always detect
    DETECT_MODE_LIGHT_TRACK,        ///< Detection mode: Light face track
    DETECT_MODE_TRACK_BY_DETECT,    ///< Detection mode: Tracking by detection
};

/**
 * @struct CustomPipelineParameter
 * @brief Structure to hold custom parameters for the face detection and processing pipeline.
 *
 * Includes options for enabling various features such as recognition, liveness detection, and quality assessment.
 */
typedef struct CustomPipelineParameter {
    bool enable_recognition = false;           ///< Enable face recognition feature
    bool enable_liveness = false;              ///< Enable RGB liveness detection feature
    bool enable_ir_liveness = false;           ///< Enable IR (Infrared) liveness detection feature
    bool enable_mask_detect = false;           ///< Enable mask detection feature
    bool enable_face_attribute = false;        ///< Enable face attribute prediction feature
    bool enable_face_quality = false;          ///< Enable face quality assessment feature
    bool enable_interaction_liveness = false;  ///< Enable interactive liveness detection feature
    bool enable_face_pose = false;             ///< Enable face pose estimation feature
    bool enable_face_emotion = false;          ///< Enable face emotion recognition feature
} ContextCustomParameter;

/** @struct FaceLoc
 *  @brief Struct representing standardized face landmarks for detection.
 *
 *  Contains coordinates for the face, detection score, and landmarks.
 */
typedef struct FaceLoc {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float lmk[10];
} FaceLoc;

/** @typedef FaceLocList
 *  @brief List of FaceLoc structures.
 */
typedef std::vector<FaceLoc> FaceLocList;

/** @struct FaceBasicData
 *  @brief Struct for basic face data.
 *
 *  Contains the size of the data and a pointer to the data.
 */
typedef struct FaceBasicData {
    int32_t dataSize;
    void* data;
} FaceBasicData;

/** @struct FaceFeatureEntity
 *  @brief Struct for face feature data.
 *
 *  Contains the size of the feature data and a pointer to the feature array.
 */
typedef struct FaceFeatureEntity {
    int32_t dataSize;
    float* data;
} FaceFeaturePtr;

// Search for most similar vectors
struct FaceSearchResult {
    int64_t id;
    double similarity;
    std::vector<float> feature;
};

/** @struct FaceEmbedding
 *  @brief Struct for face embedding data.
 *
 *  Contains the isNormal flag and the embedding vector.
 */
struct FaceEmbedding {
    int32_t isNormal;
    float norm;
    Embedded embedding;
};

/** @struct FaceInteractionState
 *  @brief Struct for face interaction state data.
 *
 *  Contains the confidence scores for face interaction.
 */
struct FaceInteractionState {
    float left_eye_status_confidence;
    float right_eye_status_confidence;
};

/** @struct FaceInteractionAction
 *  @brief Struct for face interaction action data.
 *
 *  Contains the actions for face interaction.
 */
struct FaceInteractionAction {
    int32_t normal;     ///< Normal action.
    int32_t shake;      ///< Shake action.
    int32_t jawOpen;    ///< Jaw open action.
    int32_t headRaise;  ///< Head raise action.
    int32_t blink;      ///< Blink action.
};

/** @struct FaceAttributeResult
 *  @brief Struct for face attribute result data.
 *
 *  Contains the results for face attribute.
 */
struct FaceAttributeResult {
    int32_t race;        ///< Race of the detected face.
                         ///< 0: Black;
                         ///< 1: Asian;
                         ///< 2: Latino/Hispanic;
                         ///< 3: Middle Eastern;
                         ///< 4: White;
    int32_t gender;      ///< Gender of the detected face.
                         ///< 0: Female;
                         ///< 1: Male;
    int32_t ageBracket;  ///< Age bracket of the detected face.
                         ///< 0: 0-2 years old;
                         ///< 1: 3-9 years old;
                         ///< 2: 10-19 years old;
                         ///< 3: 20-29 years old;
                         ///< 4: 30-39 years old;
                         ///< 5: 40-49 years old;
                         ///< 6: 50-59 years old;
                         ///< 7: 60-69 years old;
                         ///< 8: more than 70 years old;
};

/** @struct FaceEmotionResult
 *  @brief Struct for face emotion result data.
 *
 *  Contains the results for face emotion.
 */
struct FaceEmotionResult {
    int32_t emotion;     ///< Emotion of the detected face.
                         ///< 0: Neutral;
                         ///< 1: Happy;
                         ///< 2: Sad;
                         ///< 3: Surprise;
                         ///< 4: Fear;
                         ///< 5: Disgust;
                         ///< 6: Anger;
};

/** @} */

}  // namespace inspire

#endif  // INSPIRE_FACE_DATATYPE_H
