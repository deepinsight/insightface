/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FACE_TRACK_MODULE_FACE_DETECT_RNET_ADAPT_H
#define INSPIRE_FACE_TRACK_MODULE_FACE_DETECT_RNET_ADAPT_H
#include "data_type.h"
#include "middleware/any_net_adapter.h"

namespace inspire {

/**
 * @class RNet
 * @brief Class representing the RNet (Refinement Network), inheriting from AnyNet.
 *
 * This class is used for refining face detection results, typically as part of a cascaded
 * network system for facial recognition or detection tasks.
 */
class INSPIRE_API RNetAdapt : public AnyNetAdapter {
public:
    /**
     * @brief Constructor for the RNet class.
     */
    RNetAdapt();

    /**
     * @brief Operator to process an affine-transformed face image and return a score indicating the quality of the refinement.
     * @param bgr_affine Affine-transformed face image in BGR format.
     * @return float Score representing the quality or confidence of the refinement.
     */
    float operator()(const inspirecv::Image& bgr_affine);
};

}  //  namespace inspire

#endif  // INSPIRE_FACE_TRACK_MODULE_FACE_DETECT_RNET_ADAPT_H
