//
// Created by tunm on 2023/5/5.
//
#pragma once
#ifndef HYPERFACE_DATATYPE_H
#define HYPERFACE_DATATYPE_H

#include <cstdint>
#if defined(_WIN32) && (defined(_DEBUG) || defined(DEBUG))
#define _CRTDBG_MAP_ALLOC
#include "crtdbg.h"
#endif

#ifndef INSPIRE_API
#define INSPIRE_API
#endif

#include <opencv2/opencv.hpp>

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
typedef cv::Point Point2i;

/** @typedef Point2f
*  @brief 2D coordinate point with float precision.
*/
typedef cv::Point2f Point2f;

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

/** @typedef Matrix
*  @brief Generic matrix representation.
*/
typedef cv::Mat Matrix;

/** @typedef Rectangle
*  @brief Rectangle representation using integer values.
*/
typedef cv::Rect_<int> Rectangle;

/** @typedef Size
*  @brief Size representation using integer values.
*/
typedef cv::Size_<int> Size;

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
    float *data;
} FaceFeaturePtr;

/** @} */

}  // namespace inspire

#endif //HYPERFACE_DATATYPE_H
