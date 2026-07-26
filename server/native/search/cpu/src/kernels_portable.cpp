#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ifs_cpu {

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16u) & 1u);
    bits += rounding_bias;
    return static_cast<std::uint16_t>(bits >> 16u);
}

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::int8_t quantize_int8(float value, std::uint32_t scale) {
    // Keep the multiplication in FP32 so the CPU encoder has the same
    // half-integer behavior as the CUDA kernel and NumPy reference backend.
    const float scaled = value * static_cast<float>(scale);
    if (scaled >= 127.0f) {
        return 127;
    }
    if (scaled <= -128.0f) {
        return -128;
    }
    const long rounded = std::lroundf(scaled);
    return static_cast<std::int8_t>(std::clamp<long>(rounded, -128, 127));
}

float dot_f32_portable(const float *left, const float *right) {
    float accumulator = 0.0f;
    for (std::size_t index = 0; index < kDimension; ++index) {
        accumulator += left[index] * right[index];
    }
    return accumulator;
}

float dot_bf16_portable(const std::uint16_t *left, const std::uint16_t *right) {
    float accumulator = 0.0f;
    for (std::size_t index = 0; index < kDimension; ++index) {
        accumulator += bf16_to_float(left[index]) * bf16_to_float(right[index]);
    }
    return accumulator;
}

std::int32_t dot_int8_portable(
    const std::int8_t *left,
    const std::int8_t *right,
    std::int32_t /*query_sum*/) {
    std::int32_t accumulator = 0;
    for (std::size_t index = 0; index < kDimension; ++index) {
        accumulator += static_cast<std::int32_t>(left[index]) *
                       static_cast<std::int32_t>(right[index]);
    }
    return accumulator;
}

const Dispatch &dispatch() {
    static const Dispatch selected = make_dispatch();
    return selected;
}

}  // namespace ifs_cpu
