/**
 * Created by Jingyu Yan
 * @date 2025-04-26
 */
#pragma once
#ifndef INSPIRE_FACE_TRACK_MODULE_LANDMARK_TOOLS_H
#define INSPIRE_FACE_TRACK_MODULE_LANDMARK_TOOLS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <inspirecv/inspirecv.h>

namespace inspire {

// Generate crop scales
inline std::vector<float> GenerateCropScales(float start_scale, int N) {
    std::vector<float> result;
    if (N <= 0)
        return result;

    result.push_back(start_scale);

    float delta_step = 0.02f;  // Initial step size
    int direction = -1;        // First decrease
    int expand_count = 1;      // Current expansion count

    for (int i = 1; i < N; ++i) {
        // The step size increases slightly every two expansions, e.g. 0.02 -> 0.04 -> 0.06
        float current_delta = delta_step * (static_cast<float>((expand_count + 1) / 2));

        float new_scale = start_scale + direction * current_delta;

        // Keep 5 decimal places
        new_scale = std::round(new_scale * 100000.0f) / 100000.0f;

        result.push_back(new_scale);

        direction *= -1;  // Alternating positive and negative
        expand_count++;
    }

    return result;
}

inline inspirecv::TransformMatrix ScaleAffineMatrixPreserveCenter(const inspirecv::TransformMatrix& affine, float scale, int output_size = 112) {
    // The center point in the output image is (cx, cy).
    float cx = output_size / 2.0f;
    float cy = output_size / 2.0f;

    // 1. Obtain the position of the current center point in the original image (inverse affine transformation)
    inspirecv::TransformMatrix inv_affine = affine.GetInverse();
    float center_x = inv_affine.Get(0, 0) * cx + inv_affine.Get(0, 1) * cy + inv_affine.Get(0, 2);
    float center_y = inv_affine.Get(1, 0) * cx + inv_affine.Get(1, 1) * cy + inv_affine.Get(1, 2);

    // 2. Create a new scaling matrix (note that the scale value should be reduced → to "expand the cropping area")
    float inv_scale = 1.0f / scale;
    float new_a11 = affine.Get(0, 0) * inv_scale;
    float new_a12 = affine.Get(0, 1) * inv_scale;
    float new_a21 = affine.Get(1, 0) * inv_scale;
    float new_a22 = affine.Get(1, 1) * inv_scale;

    // 3. Calculate the new offset to ensure that the center of the scaled image is still mapped to (cx, cy)
    float new_b1 = cx - (new_a11 * center_x + new_a12 * center_y);
    float new_b2 = cy - (new_a21 * center_x + new_a22 * center_y);

    return inspirecv::TransformMatrix::Create(new_a11, new_a12, new_b1, new_a21, new_a22, new_b2);
}

inline std::vector<inspirecv::Point2f> MultiFrameLandmarkMean(const std::vector<std::vector<inspirecv::Point2f>>& points) {
    std::vector<inspirecv::Point2f> mean_points;

    if (points.empty()) {
        return mean_points;
    }

    if (points.size() == 1) {
        return points[0];
    }

    size_t num_frames = points.size();
    size_t num_points = points[0].size();

    // Initialize to 0
    mean_points.resize(num_points, inspirecv::Point2f(0.0f, 0.0f));

    // Accumulate the coordinates of each point
    for (const auto& frame : points) {
        if (frame.size() != num_points)
            continue;  // skip invalid frame size

        for (size_t i = 0; i < num_points; ++i) {
            mean_points[i].SetX(mean_points[i].GetX() + frame[i].GetX());
            mean_points[i].SetY(mean_points[i].GetY() + frame[i].GetY());
        }
    }

    // Calculate the average
    for (auto& pt : mean_points) {
        pt.SetX(pt.GetX() / static_cast<float>(num_frames));
        pt.SetY(pt.GetY() / static_cast<float>(num_frames));
    }

    return mean_points;
}

inline std::vector<inspirecv::Point2f> LandmarkCropped(const std::vector<inspirecv::Point2f>& points, int output_size = 192) {
    inspirecv::Rect2f rect = inspirecv::MinBoundingRect(points);

    // Calculate the scale
    float scale = output_size / std::max(rect.GetWidth(), rect.GetHeight());

    // Create the result vector and scale each point
    std::vector<inspirecv::Point2f> result;
    result.reserve(points.size());
    for (const auto& pt : points) {
        // Translate to the origin (subtract the top-left corner coordinates), then scale
        result.push_back(inspirecv::Point2f((pt.GetX() - rect.GetX()) * scale, (pt.GetY() - rect.GetY()) * scale));
    }

    return result;
}

}  // namespace inspire

#endif  // INSPIRE_FACE_TRACK_MODULE_LANDMARK_TOOLS_H
