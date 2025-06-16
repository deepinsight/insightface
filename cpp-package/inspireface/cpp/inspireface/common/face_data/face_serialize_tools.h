/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifndef INSPIRE_FACE_SERIALIZE_TOOLS_H
#define INSPIRE_FACE_SERIALIZE_TOOLS_H

#include "face_wrapper.h"
#include "../face_info/face_object_internal.h"
#include "herror.h"
#include "data_type.h"
#include "track_module/landmark/all.h"
#include <log.h>

// Define the namespace "inspire" for encapsulation
namespace inspire {

/**
 * @brief Print the transformation matrix.
 * @param matrix The transformation matrix to print.
 */
inline void PrintTransformMatrix(const TransMatrix& matrix) {
    INSPIRE_LOGI("Transformation Matrix:");
    INSPIRE_LOGI("a: %f\tb: %f\ttx: %f", matrix.m00, matrix.m01, matrix.tx);
    INSPIRE_LOGI("c: %f\td: %f\tty: %f", matrix.m10, matrix.m11, matrix.ty);
}

/**
 * @brief Print FaceTrackWrap structure.
 * @param data The FaceTrackWrap structure to print.
 */
inline void INSPIRE_API PrintHyperFaceDataDetail(const FaceTrackWrap& data) {
    INSPIRE_LOGI("Track State: %d", data.trackState);
    INSPIRE_LOGI("In Group Index: %d", data.inGroupIndex);
    INSPIRE_LOGI("Track ID: %d", data.trackId);
    INSPIRE_LOGI("Track Count: %d", data.trackCount);

    INSPIRE_LOGI("Face Rectangle:");
    INSPIRE_LOGI("x: %f\ty: %f\twidth: %f\theight: %f", data.rect.x, data.rect.y, data.rect.width, data.rect.height);

    PrintTransformMatrix(data.trans);
}

/**
 * @brief Convert a FaceObject to FaceTrackWrap.
 * @param obj The FaceObject to convert.
 * @param group_index The group index.
 * @return The converted FaceTrackWrap structure.
 */
inline FaceTrackWrap INSPIRE_API FaceObjectInternalToHyperFaceData(const FaceObjectInternal& obj, int group_index = -1) {
    FaceTrackWrap data;
    // Face rect
    data.rect.x = obj.bbox_.GetX();
    data.rect.y = obj.bbox_.GetY();
    data.rect.width = obj.bbox_.GetWidth();
    data.rect.height = obj.bbox_.GetHeight();
    // Trans matrix
    data.trans.m00 = obj.getTransMatrix().Get(0, 0);
    data.trans.m01 = obj.getTransMatrix().Get(0, 1);
    data.trans.m10 = obj.getTransMatrix().Get(1, 0);
    data.trans.m11 = obj.getTransMatrix().Get(1, 1);
    data.trans.tx = obj.getTransMatrix().Get(0, 2);
    data.trans.ty = obj.getTransMatrix().Get(1, 2);
    // KetPoints five
    if (!obj.high_result.lmk.empty()) {
        for (int i = 0; i < obj.high_result.lmk.size(); ++i) {
            data.keyPoints[i].x = obj.high_result.lmk[i].GetX();
            data.keyPoints[i].y = obj.high_result.lmk[i].GetY();
        }
        for (int i = 0; i < 5; ++i) {
            data.quality[i] = obj.high_result.lmk_quality[i];
        }
        //        LOGD("HIGHT");
    } else {
        for (int i = 0; i < obj.keyPointFive.size(); ++i) {
            data.keyPoints[i].x = obj.keyPointFive[i].GetX();
            data.keyPoints[i].y = obj.keyPointFive[i].GetY();
        }
        for (int i = 0; i < 5; ++i) {
            data.quality[i] = -1.0f;
        }
    }
    // Basic data
    data.inGroupIndex = group_index;
    data.trackCount = obj.tracking_count_;
    data.trackId = obj.GetTrackingId();
    data.trackState = obj.TrackingState();
    // Face 3D Angle
    data.face3DAngle.pitch = obj.high_result.pitch;
    data.face3DAngle.roll = obj.high_result.roll;
    data.face3DAngle.yaw = obj.high_result.yaw;
    // Density Landmark
    if (!obj.landmark_smooth_aux_.empty()) {
        data.densityLandmarkEnable = 1;
        const auto& lmk = obj.landmark_smooth_aux_.back();
        for (size_t i = 0; i < FaceLandmarkAdapt::NUM_OF_LANDMARK; i++) {
            data.densityLandmark[i].x = lmk[i].GetX();
            data.densityLandmark[i].y = lmk[i].GetY();
        }
    } else {
        data.densityLandmarkEnable = 0;
    }

    return data;
}

/**
 * @brief Convert a TransMatrix to a cv::Mat.
 * @param trans The TransMatrix to convert.
 * @return The converted cv::Mat.
 */
inline inspirecv::TransformMatrix INSPIRE_API TransformMatrixToInternalMatrix(const TransMatrix& trans) {
    inspirecv::TransformMatrix mat = inspirecv::TransformMatrix::Create(trans.m00, trans.m01, trans.tx, trans.m10, trans.m11, trans.ty);
    return mat;
}

/**
 * @brief Convert a FaceRect to cv::Rect.
 * @param faceRect The FaceRect to convert.
 * @return The converted cv::Rect.
 */
inline inspirecv::Rect2i INSPIRE_API FaceRectToInternalRect(const FaceRect& faceRect) {
    return inspirecv::Rect2i(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
}

/**
 * @brief Convert a Point2F to cv::Point2f.
 * @param point The Point2F to convert.
 * @return The converted cv::Point2f.
 */
inline inspirecv::Point2f INSPIRE_API HPointToInternalPoint2f(const Point2F& point) {
    return inspirecv::Point2f(point.x, point.y);
}

/**
 * @brief Serialize FaceTrackWrap to a byte stream.
 * @param data The FaceTrackWrap to serialize.
 * @param byteArray The output byte stream.
 * @return The result code.
 */
inline int32_t INSPIRE_API RunSerializeHyperFaceData(const FaceTrackWrap& data, ByteArray& byteArray) {
    byteArray.reserve(sizeof(data));

    // Serialize the FaceTrackWrap structure itself
    const char* dataBytes = reinterpret_cast<const char*>(&data);
    byteArray.insert(byteArray.end(), dataBytes, dataBytes + sizeof(data));

    return HSUCCEED;
}

/**
 * @brief Deserialize a byte stream to FaceTrackWrap.
 * @param byteArray The input byte stream.
 * @param data The output FaceTrackWrap structure.
 * @return The result code.
 */
inline int32_t INSPIRE_API RunDeserializeHyperFaceData(const ByteArray& byteArray, FaceTrackWrap& data) {
    // Check if the byte stream size is sufficient
    if (byteArray.size() >= sizeof(data)) {
        // Copy data from the byte stream to the FaceTrackWrap structure
        std::memcpy(&data, byteArray.data(), sizeof(data));
    } else {
        INSPIRE_LOGE("The byte stream size is insufficient to restore FaceTrackWrap");
        return HERR_SESS_FACE_DATA_ERROR;
    }

    return HSUCCEED;
}

/**
 * @brief Deserialize a byte stream to FaceTrackWrap.
 * @param byteArray The input byte stream as a character array.
 * @param byteCount The size of the byte stream.
 * @param data The output FaceTrackWrap structure.
 * @return The result code.
 */
inline int32_t INSPIRE_API RunDeserializeHyperFaceData(const char* byteArray, size_t byteCount, FaceTrackWrap& data) {
    // Check if the byte stream size is sufficient
    if (byteCount >= sizeof(data)) {
        // Copy data from the byte stream to the FaceTrackWrap structure
        std::memcpy(&data, byteArray, sizeof(data));
    } else {
        INSPIRE_LOGE("The byte stream size is insufficient to restore FaceTrackWrap");
        return HERR_SESS_FACE_DATA_ERROR;
    }

    return HSUCCEED;
}

}  // namespace inspire
#endif  // INSPIRE_FACE_SERIALIZE_TOOLS_H
