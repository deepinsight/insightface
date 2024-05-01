//
// Created by Tunm-Air13 on 2024/4/12.
//
#pragma once
#ifndef INSPIREFACE_FACE_FEATURE_EXTRACTION_H
#define INSPIREFACE_FACE_FEATURE_EXTRACTION_H
#include <mutex>
#include "extract/extract.h"
#include "common/face_info/face_object.h"
#include "common/face_data/data_tools.h"
#include "middleware/camera_stream/camera_stream.h"
#include "middleware/model_archive/inspire_archive.h"

namespace inspire {

class FeatureExtraction {
public:
    /**
     * @brief Constructor for FaceRecognition class.
     *
     * @param archive Model active instance for model loading.
     * @param enable_recognition Whether face recognition is enabled.
     * @param core Type of matrix core to use for feature extraction.
     * @param feature_block_num Number of feature blocks to use.
     */
    FeatureExtraction(InspireArchive &archive, bool enable_recognition);

    /**
     * @brief Example Query the module loading status
     * @return Status code
     * */
    int32_t QueryStatus() const;

    /**
     * @brief Extracts a facial feature from an image and stores it in the provided 'embedded'.
     *
     * @param image CameraStream instance containing the image.
     * @param face FaceObject representing the detected face.
     * @param embedded Output parameter to store the extracted facial feature.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t FaceExtract(CameraStream &image, const FaceObject& face, Embedded &embedded);

    /**
     * @brief Extracts a facial feature from an image and stores it in the provided 'embedded'.
     *
     * @param image CameraStream instance containing the image.
     * @param face HyperFaceData representing the detected face.
     * @param embedded Output parameter to store the extracted facial feature.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t FaceExtract(CameraStream &image, const HyperFaceData& face, Embedded &embedded);

    /**
     * @brief Gets the Extract instance associated with this FaceRecognition.
     *
     * @return const std::shared_ptr<Extract>& Pointer to the Extract instance.
     */
    const std::shared_ptr<Extract> &getMExtract() const;


private:
    /**
     * @brief Initializes the interaction with the Extract model.
     *
     * @param model Pointer to the loaded model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitExtractInteraction(InspireModel& model);

private:

    std::shared_ptr<Extract> m_extract_; ///< Pointer to the Extract instance.

    int32_t m_status_code_;     ///< Status code
};

}   // namespace inspire

#endif //INSPIREFACE_FACE_FEATURE_EXTRACTION_H
