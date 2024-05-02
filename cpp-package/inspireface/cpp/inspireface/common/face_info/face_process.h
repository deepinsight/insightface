//
// Created by tunm on 2023/9/12.
//

// Include guard to prevent double inclusion of this header file
#pragma once
#ifndef HYPERFACEREPO_FACEPROCESS_H
#define HYPERFACEREPO_FACEPROCESS_H

// Include the necessary header file "data_type.h"
#include "data_type.h"

// Define the namespace "inspire" for encapsulation
namespace inspire {

/**
 * Enumeration to represent different mask information.
 */
typedef enum MaskInfo {
    UNKNOWN_MASK = -1,  ///< Unknown mask status
    UNMASKED = 0,       ///< No mask
    MASKED = 1,         ///< Wearing a mask
} MaskInfo;

/**
* Enumeration to represent different RGB liveness information.
*/
typedef enum RGBLivenessInfo {
    UNKNOWN_RGB_LIVENESS = -1,  ///< Unknown RGB liveness status
    LIVENESS_FAKE = 0,          ///< Fake liveness
    LIVENESS_REAL = 1,          ///< Real liveness
} RGBLivenessInfo;

/**
* Class definition for FaceProcess.
*/
class INSPIRE_API FaceProcess {
public:
    /**
     * Member variable to store mask information, initialized to UNKNOWN_MASK.
     */
    MaskInfo maskInfo = UNKNOWN_MASK;

    /**
     * Member variable to store RGB liveness information, initialized to UNKNOWN_RGB_LIVENESS.
     */
    RGBLivenessInfo rgbLivenessInfo = UNKNOWN_RGB_LIVENESS;

};

} // namespace hyper

#endif //HYPERFACEREPO_FACEPROCESS_H
