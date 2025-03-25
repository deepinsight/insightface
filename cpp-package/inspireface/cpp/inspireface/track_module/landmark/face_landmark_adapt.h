/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FACE_TRACK_MODULE_LANDMARK_FACE_LANDMARK_ADAPT_H
#define INSPIRE_FACE_TRACK_MODULE_LANDMARK_FACE_LANDMARK_ADAPT_H
#include "../../data_type.h"
#include "middleware/any_net_adapter.h"

namespace inspire {

/**
 * @class FaceLandmark
 * @brief Class for facial landmark detection, inheriting from AnyNet.
 *
 * This class specializes in detecting facial landmarks from images using neural network models.
 */
class INSPIRE_API FaceLandmarkAdapt : public AnyNetAdapter {
public:
    /**
     * @brief Operator to process an affine-transformed face image and return facial landmarks.
     * @param bgr_affine Affine-transformed face image in BGR format.
     * @return std::vector<float> Vector containing the coordinates of facial landmarks.
     */
    std::vector<float> operator()(const inspirecv::Image& bgr_affine);

    /**
     * @brief Constructor for the FaceLandmark class.
     * @param input_size The size of the input image for the neural network.
     */
    explicit FaceLandmarkAdapt(int input_size = 112);

    /**
     * @brief Gets the input size for the neural network model.
     * @return int The input size.
     */
    int getInputSize() const;

public:
    const static int LEFT_EYE_CENTER = 67;      ///< Landmark index for the center of the left eye.
    const static int RIGHT_EYE_CENTER = 68;     ///< Landmark index for the center of the right eye.
    const static int NOSE_CORNER = 100;         ///< Landmark index for the tip of the nose.
    const static int MOUTH_LEFT_CORNER = 104;   ///< Landmark index for the left corner of the mouth.
    const static int MOUTH_RIGHT_CORNER = 105;  ///< Landmark index for the right corner of the mouth.
    const static int MOUTH_LOWER = 84;          ///< Landmark index for the lower corner of the mouth.
    const static int MOUTH_UPPER = 87;          ///< Landmark index for the upper corner of the mouth.

    const static int NUM_OF_LANDMARK = 106;  ///< Total number of landmarks detected.

private:
    const int m_input_size_;  ///< The input size for the neural network model.
};

}  //  namespace inspire

#endif  // INSPIRE_FACE_TRACK_MODULE_LANDMARK_FACE_LANDMARK_ADAPT_H
