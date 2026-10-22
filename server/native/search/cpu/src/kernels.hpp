#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ifs_cpu {

constexpr std::size_t kDimension = 512;

using DotF32 = float (*)(const float *, const float *);
using DotBF16 = float (*)(const std::uint16_t *, const std::uint16_t *);
using DotInt8 = std::int32_t (*)(const std::int8_t *, const std::int8_t *, std::int32_t);

struct Dispatch {
    DotF32 fp32;
    DotBF16 bf16;
    DotInt8 int8;
    const char *fp32_name;
    const char *bf16_name;
    const char *int8_name;
    std::string features;
};

float dot_f32_portable(const float *left, const float *right);
float dot_bf16_portable(const std::uint16_t *left, const std::uint16_t *right);
std::int32_t dot_int8_portable(
    const std::int8_t *left,
    const std::int8_t *right,
    std::int32_t query_sum);

std::uint16_t float_to_bf16(float value);
float bf16_to_float(std::uint16_t value);
std::int8_t quantize_int8(float value, std::uint32_t scale);

Dispatch make_dispatch();
const Dispatch &dispatch();

}  // namespace ifs_cpu
