/**
 * Created by Jingyu Yan
 * @date 2025-03-14
 */
#pragma once
#ifndef INSPIREFACE_TEST_CHECK_
#define INSPIREFACE_TEST_CHECK_

#include <cstdint>  // for uint8_t
#include <limits>   // for std::numeric_limits
#include <sstream>

#define REQUIRE_EQ_IMAGE(a, b, h, w, c)                                                                                                \
    do {                                                                                                                               \
        double eps = 0.01;                                                                                                             \
        double mse = CalculateImageMSE(a, b, h, w, c);                                                                                 \
        REQUIRE(mse <= eps);                                                                                                           \
        if (mse > eps) {                                                                                                               \
            std::stringstream ss;                                                                                                      \
            ss << "Image comparison failed! MSE: " << mse << " (threshold: " << eps << "), dimensions: " << h << "x" << w << "x" << c; \
            INFO(ss.str());                                                                                                            \
        }                                                                                                                              \
    } while (0)

#define REQUIRE_EQ_IMAGE_WITH_EPS(a, b, h, w, c, eps)                                                                                  \
    do {                                                                                                                               \
        double mse = CalculateImageMSE(a, b, h, w, c);                                                                                 \
        REQUIRE(mse <= eps);                                                                                                           \
        if (mse > eps) {                                                                                                               \
            std::stringstream ss;                                                                                                      \
            ss << "Image comparison failed! MSE: " << mse << " (threshold: " << eps << "), dimensions: " << h << "x" << w << "x" << c; \
            INFO(ss.str());                                                                                                            \
        }                                                                                                                              \
    } while (0)

inline double CalculateImageMSE(const uint8_t* a, const uint8_t* b, int h, int w, int c) {
    if (a == nullptr || b == nullptr || h <= 0 || w <= 0 || c <= 0) {
        return std::numeric_limits<double>::infinity();
    }

    double sum_squared_diff = 0.0;
    size_t total_pixels = static_cast<size_t>(h) * w * c;
    const double normalize_factor = 255.0;

    for (size_t i = 0; i < total_pixels; ++i) {
        double a_normalized = static_cast<double>(a[i]) / normalize_factor;
        double b_normalized = static_cast<double>(b[i]) / normalize_factor;

        double diff = a_normalized - b_normalized;
        sum_squared_diff += diff * diff;
    }

    return sum_squared_diff / total_pixels;
}

#endif  // INSPIREFACE_TEST_CHECK_