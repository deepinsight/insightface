/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

// Include guard to prevent double inclusion of this header file
#pragma once
#ifndef INSPIRE_FACE_FACEDATATYPE_H
#define INSPIRE_FACE_FACEDATATYPE_H

#include "data_type.h"

namespace inspire {

/**
 * Struct to represent 3D angles of a face.
 */
typedef struct Face3DAngle {
    float roll;   ///< Roll angle
    float yaw;    ///< Yaw angle
    float pitch;  ///< Pitch angle
} Face3DAngle;

/**
 * Struct to represent the rectangle coordinates of a face.
 */
typedef struct FaceRect {
    int x;       ///< X-coordinate of the top-left corner
    int y;       ///< Y-coordinate of the top-left corner
    int width;   ///< Width of the rectangle
    int height;  ///< Height of the rectangle
} FaceRect;

/**
 * Struct to represent 2D point coordinates.
 */
typedef struct Point2F {
    float x;  ///< X-coordinate
    float y;  ///< Y-coordinate
} HPoint;

/**
 * Struct to represent a 2D transformation matrix.
 */
typedef struct TransMatrix {
    double m00;  ///< Element (0,0) of the matrix
    double m01;  ///< Element (0,1) of the matrix
    double m10;  ///< Element (1,0) of the matrix
    double m11;  ///< Element (1,1) of the matrix
    double tx;   ///< Translation in the X-axis
    double ty;   ///< Translation in the Y-axis
} TransMatrix;

/**
 * Struct to represent basic face data.
 */
typedef struct FaceTrackWrap {
    int trackState;                ///< Track state
    int inGroupIndex;              ///< Index within a group
    int trackId;                   ///< Track ID
    int trackCount;                ///< Track count
    FaceRect rect;                 ///< Face rectangle
    TransMatrix trans;             ///< Transformation matrix
    Point2F keyPoints[5];          ///< Key points (e.g., landmarks)
    Face3DAngle face3DAngle;       ///< 3D face angles
    float quality[5];              ///< Quality values for key points
    Point2F densityLandmark[106];  ///< Face density landmark
    int densityLandmarkEnable;     ///< Density landmark enable
} FaceTrackWrap;

}  // namespace inspire

#endif  // INSPIRE_FACE_FACEDATATYPE_H
