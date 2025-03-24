/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIREFACE_EXTRACT_ADAPT_H
#define INSPIREFACE_EXTRACT_ADAPT_H
#include "data_type.h"
#include "middleware/any_net_adapter.h"

namespace inspire {

/**
 * @class Extract
 * @brief Class for extracting features from faces, inheriting from AnyNet.
 *
 * This class specializes in processing face images to extract embedded facial features,
 * which can be used for further analysis like face recognition or verification.
 */
class INSPIRE_API ExtractAdapt : public AnyNetAdapter {
public:
    /**
     * @brief Constructor for the Extract class.
     */
    ExtractAdapt();

    /**
     * @brief Operator to process an affine-transformed face image and return the extracted features.
     * @param bgr_affine Affine-transformed face image in BGR format.
     * @param norm The L2 norm of a vector.
     * @param normalize Whether the obtained features are normalized.
     * @return Embedded Vector of extracted features.
     */
    Embedded operator()(const inspirecv::Image& bgr_affine, float& norm, bool normalize = true);

    /**
     * @brief Gets the facial features from an affine-transformed face image.
     * @param bgr_affine Affine-transformed face image in BGR format.
     * @return Embedded Vector of extracted facial features.
     */
    Embedded GetFaceFeature(const inspirecv::Image& bgr_affine);
};

}  // namespace inspire

#endif  // INSPIREFACE_EXTRACT_ADAPT_H
