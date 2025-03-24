/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FACE_FEATURE_EXTRACTION_MODULE_H
#define INSPIRE_FACE_FEATURE_EXTRACTION_MODULE_H
#include <mutex>
#include "extract/extract_adapt.h"
#include "common/face_info/face_object_internal.h"
#include "common/face_data/face_data_type.h"
#include "middleware/model_archive/inspire_archive.h"
#include "middleware/inspirecv_image_process.h"

namespace inspire {

class INSPIRE_API FeatureExtractionModule {
public:
    /**
     * @brief Constructor for FaceRecognition class.
     *
     * @param archive Model active instance for model loading.
     * @param enable_recognition Whether face recognition is enabled.
     * @param core Type of matrix core to use for feature extraction.
     * @param feature_block_num Number of feature blocks to use.
     */
    FeatureExtractionModule(InspireArchive &archive, bool enable_recognition);

    /**
     * @brief Example Query the module loading status
     * @return Status code
     * */
    int32_t QueryStatus() const;

    /**
     * @brief Extracts a facial feature from an image and stores it in the provided 'embedded'.
     *
     * @param processor inspirecv::InspireImageProcess instance containing the image.
     * @param face FaceObject representing the detected face.
     * @param embedded Output parameter to store the extracted facial feature.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t FaceExtract(inspirecv::InspireImageProcess &processor, const FaceObjectInternal &face, Embedded &embedded, float &norm,
                        bool normalize = false);

    /**
     * @brief Extracts a facial feature from an image and stores it in the provided 'embedded'.
     *
     * @param processor inspirecv::InspireImageProcess instance containing the image.
     * @param face HyperFaceData representing the detected face.
     * @param embedded Output parameter to store the extracted facial feature.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t FaceExtract(inspirecv::InspireImageProcess &processor, const HyperFaceData &face, Embedded &embedded, float &norm, bool normalize = true);

    /**
     * @brief Gets the Extract instance associated with this FaceRecognition.
     *
     * @return const std::shared_ptr<Extract>& Pointer to the Extract instance.
     */
    const std::shared_ptr<ExtractAdapt> &getMExtract() const;

private:
    /**
     * @brief Initializes the interaction with the Extract model.
     *
     * @param model Pointer to the loaded model.
     * @return int32_t Status code indicating success (0) or failure.
     */
    int32_t InitExtractInteraction(InspireModel &model);

private:
    std::shared_ptr<ExtractAdapt> m_extract_;  ///< Pointer to the Extract instance.

    int32_t m_status_code_;  ///< Status code
};

}  // namespace inspire

#endif  // INSPIRE_FACE_FEATURE_EXTRACTION_MODULE_H
