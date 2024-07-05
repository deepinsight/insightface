//
// Created by tunm on 2023/9/17.
//

#ifndef HYPERFACEREPO_DATATOOLS_H
#define HYPERFACEREPO_DATATOOLS_H
#include "opencv2/opencv.hpp"
#include "face_data_type.h"
#include "../face_info/face_object.h"
#include "herror.h"
#include "data_type.h"

// Define the namespace "inspire" for encapsulation
namespace inspire {

/**
 * @brief Print the transformation matrix.
 * @param matrix The transformation matrix to print.
 */
inline void PrintTransMatrix(const TransMatrix& matrix) {
    std::cout << "Transformation Matrix:" << std::endl;
    std::cout << "m00: " << matrix.m00 << "\t";
    std::cout << "m01: " << matrix.m01 << "\t";
    std::cout << "tx: " << matrix.tx << std::endl;

    std::cout << "m10: " << matrix.m10 << "\t";
    std::cout << "m11: " << matrix.m11 << "\t";
    std::cout << "ty: " << matrix.ty << std::endl;
}

/**
 * @brief Print HyperFaceData structure.
 * @param data The HyperFaceData structure to print.
 */
inline void INSPIRE_API PrintHyperFaceData(const HyperFaceData& data) {
    std::cout << "Track State: " << data.trackState << std::endl;
    std::cout << "In Group Index: " << data.inGroupIndex << std::endl;
    std::cout << "Track ID: " << data.trackId << std::endl;
    std::cout << "Track Count: " << data.trackCount << std::endl;

    std::cout << "Face Rectangle:" << std::endl;
    std::cout << "x: " << data.rect.x << "\t";
    std::cout << "y: " << data.rect.y << "\t";
    std::cout << "width: " << data.rect.width << "\t";
    std::cout << "height: " << data.rect.height << std::endl;

    PrintTransMatrix(data.trans);

}

/**
 * @brief Convert a FaceObject to HyperFaceData.
 * @param obj The FaceObject to convert.
 * @param group_index The group index.
 * @return The converted HyperFaceData structure.
 */
inline HyperFaceData INSPIRE_API FaceObjectToHyperFaceData(const FaceObject& obj, int group_index = -1) {
    HyperFaceData data;
    // Face rect
    data.rect.x = obj.bbox_.x;
    data.rect.y = obj.bbox_.y;
    data.rect.width = obj.bbox_.width;
    data.rect.height = obj.bbox_.height;
    // Trans matrix
    data.trans.m00 = obj.getTransMatrix().at<double>(0, 0);
    data.trans.m01 = obj.getTransMatrix().at<double>(0, 1);
    data.trans.m10 = obj.getTransMatrix().at<double>(1, 0);
    data.trans.m11 = obj.getTransMatrix().at<double>(1, 1);
    data.trans.tx = obj.getTransMatrix().at<double>(0, 2);
    data.trans.ty = obj.getTransMatrix().at<double>(1, 2);
    // KetPoints five
    if (!obj.high_result.lmk.empty()) {
        for (int i = 0; i < obj.high_result.lmk.size(); ++i) {
            data.keyPoints[i].x = obj.high_result.lmk[i].x;
            data.keyPoints[i].y = obj.high_result.lmk[i].y;
        }
        for (int i = 0; i < 5; ++i) {
            data.quality[i] = obj.high_result.lmk_quality[i];
        }
//        LOGD("HIGHT");
    } else {
        for (int i = 0; i < obj.keyPointFive.size(); ++i) {
            data.keyPoints[i].x = obj.keyPointFive[i].x;
            data.keyPoints[i].y = obj.keyPointFive[i].y;
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
    

    const auto &lmk = obj.landmark_smooth_aux_.back();
    for (size_t i = 0; i < lmk.size(); i++)
    {
        data.densityLandmark[i].x = lmk[i].x;
        data.densityLandmark[i].y = lmk[i].y;
    }
    

    return data;
}

/**
 * @brief Convert a TransMatrix to a cv::Mat.
 * @param trans The TransMatrix to convert.
 * @return The converted cv::Mat.
 */
inline cv::Mat INSPIRE_API TransMatrixToMat(const TransMatrix& trans) {
    cv::Mat mat(2, 3, CV_64F);
    mat.at<double>(0, 0) = trans.m00;
    mat.at<double>(0, 1) = trans.m01;
    mat.at<double>(1, 0) = trans.m10;
    mat.at<double>(1, 1) = trans.m11;
    mat.at<double>(0, 2) = trans.tx;
    mat.at<double>(1, 2) = trans.ty;
    return mat;
}

/**
 * @brief Convert a FaceRect to cv::Rect.
 * @param faceRect The FaceRect to convert.
 * @return The converted cv::Rect.
 */
inline cv::Rect INSPIRE_API FaceRectToRect(const FaceRect& faceRect) {
    return {faceRect.x, faceRect.y, faceRect.width, faceRect.height};
}

/**
 * @brief Convert a Point2F to cv::Point2f.
 * @param point The Point2F to convert.
 * @return The converted cv::Point2f.
 */
inline cv::Point2f INSPIRE_API HPointToPoint2f(const Point2F& point) {
    return {point.x, point.y};
}

/**
 * @brief Serialize HyperFaceData to a byte stream.
 * @param data The HyperFaceData to serialize.
 * @param byteArray The output byte stream.
 * @return The result code.
 */
inline int32_t INSPIRE_API SerializeHyperFaceData(const HyperFaceData& data, ByteArray& byteArray) {
    byteArray.reserve(sizeof(data));

    // Serialize the HyperFaceData structure itself
    const char* dataBytes = reinterpret_cast<const char*>(&data);
    byteArray.insert(byteArray.end(), dataBytes, dataBytes + sizeof(data));

    return HSUCCEED;
}

/**
 * @brief Deserialize a byte stream to HyperFaceData.
 * @param byteArray The input byte stream.
 * @param data The output HyperFaceData structure.
 * @return The result code.
 */
inline int32_t INSPIRE_API DeserializeHyperFaceData(const ByteArray& byteArray, HyperFaceData &data) {
    // Check if the byte stream size is sufficient
    if (byteArray.size() >= sizeof(data)) {
        // Copy data from the byte stream to the HyperFaceData structure
        std::memcpy(&data, byteArray.data(), sizeof(data));
    } else {
        INSPIRE_LOGE("The byte stream size is insufficient to restore HyperFaceData");
        return HERR_SESS_FACE_DATA_ERROR;
    }

    return HSUCCEED;
}

/**
 * @brief Deserialize a byte stream to HyperFaceData.
 * @param byteArray The input byte stream as a character array.
 * @param byteCount The size of the byte stream.
 * @param data The output HyperFaceData structure.
 * @return The result code.
 */
inline int32_t INSPIRE_API DeserializeHyperFaceData(const char* byteArray, size_t byteCount, HyperFaceData& data) {
    // Check if the byte stream size is sufficient
    if (byteCount >= sizeof(data)) {
        // Copy data from the byte stream to the HyperFaceData structure
        std::memcpy(&data, byteArray, sizeof(data));
    } else {
        INSPIRE_LOGE("The byte stream size is insufficient to restore HyperFaceData");
        return HERR_SESS_FACE_DATA_ERROR;
    }

    return HSUCCEED;
}

}   // namespace hyper
#endif //HYPERFACEREPO_DATATOOLS_H
