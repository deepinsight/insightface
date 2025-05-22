/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIREFACE_ORDER_HYPERLANDMARK_H
#define INSPIREFACE_ORDER_HYPERLANDMARK_H
#include <iostream>
#include <vector>

namespace inspire {

// HyperLandmarkV2 left eye contour points sequence of dense facial landmarks.
const std::vector<int> HLMK_LEFT_EYE_POINTS_INDEX = {51, 52, 53, 54, 55, 56, 57, 58};

// HyperLandmarkV2 right eye contour points sequence of dense facial landmarks.
const std::vector<int> HLMK_RIGHT_EYE_POINTS_INDEX = {59, 60, 61, 62, 63, 64, 65, 66};

}  // namespace inspire

#endif  // INSPIREFACE_ORDER_HYPERLANDMARK_H
