#include "kernels.hpp"

#include <array>
#include <cstdint>
#include <sstream>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#  define IFS_CPU_X86_64 1
#else
#  define IFS_CPU_X86_64 0
#endif

#if IFS_CPU_X86_64 && (defined(__GNUC__) || defined(__clang__))
#  include <cpuid.h>
#  include <immintrin.h>
#  define IFS_CPU_X86_TARGETS 1
#  define IFS_TARGET(value) __attribute__((target(value), noinline))
#else
#  define IFS_CPU_X86_TARGETS 0
#endif

namespace ifs_cpu {
namespace {

struct CpuFeatures {
    bool x86_64 = false;
    bool os_avx = false;
    bool os_avx512 = false;
    bool fma = false;
    bool avx2 = false;
    bool avx512f = false;
    bool avx_vnni = false;
    bool avx512_vnni = false;
    bool avx512_bf16 = false;
};

#if IFS_CPU_X86_TARGETS

std::uint64_t read_xcr0() {
    std::uint32_t low = 0;
    std::uint32_t high = 0;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return (static_cast<std::uint64_t>(high) << 32u) | low;
}

CpuFeatures detect_features() {
    CpuFeatures features;
    features.x86_64 = true;
    const unsigned int max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf < 1) {
        return features;
    }

    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    const bool cpu_avx = (ecx & bit_AVX) != 0;
    const bool osxsave = (ecx & bit_OSXSAVE) != 0;
    features.fma = (ecx & bit_FMA) != 0;
    if (cpu_avx && osxsave) {
        const std::uint64_t xcr0 = read_xcr0();
        features.os_avx = (xcr0 & 0x6u) == 0x6u;
        features.os_avx512 = features.os_avx && (xcr0 & 0xe6u) == 0xe6u;
    }

    if (max_leaf < 7) {
        return features;
    }
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    const unsigned int max_subleaf = eax;
    features.avx2 = features.os_avx && (ebx & bit_AVX2) != 0;
    features.avx512f = features.os_avx512 && (ebx & bit_AVX512F) != 0;
    features.avx512_vnni = features.avx512f && (ecx & (1u << 11u)) != 0;

    if (max_subleaf >= 1) {
        __cpuid_count(7, 1, eax, ebx, ecx, edx);
        features.avx_vnni = features.avx2 && (eax & (1u << 4u)) != 0;
        features.avx512_bf16 = features.avx512f && (eax & (1u << 5u)) != 0;
    }
    return features;
}

IFS_TARGET("avx2,fma")
float dot_f32_avx2_fma(const float *left, const float *right) {
    __m256 accumulator = _mm256_setzero_ps();
    for (std::size_t index = 0; index < kDimension; index += 8) {
        const __m256 a = _mm256_loadu_ps(left + index);
        const __m256 b = _mm256_loadu_ps(right + index);
        accumulator = _mm256_fmadd_ps(a, b, accumulator);
    }
    alignas(32) std::array<float, 8> lanes{};
    _mm256_store_ps(lanes.data(), accumulator);
    float result = 0.0f;
    for (float value : lanes) {
        result += value;
    }
    return result;
}

IFS_TARGET("avx512f")
float dot_f32_avx512f(const float *left, const float *right) {
    __m512 accumulator = _mm512_setzero_ps();
    for (std::size_t index = 0; index < kDimension; index += 16) {
        const __m512 a = _mm512_loadu_ps(left + index);
        const __m512 b = _mm512_loadu_ps(right + index);
        accumulator = _mm512_add_ps(accumulator, _mm512_mul_ps(a, b));
    }
    alignas(64) std::array<float, 16> lanes{};
    _mm512_store_ps(lanes.data(), accumulator);
    float result = 0.0f;
    for (float value : lanes) {
        result += value;
    }
    return result;
}

IFS_TARGET("avx2,fma")
float dot_bf16_avx2_fma(const std::uint16_t *left, const std::uint16_t *right) {
    __m256 accumulator = _mm256_setzero_ps();
    for (std::size_t index = 0; index < kDimension; index += 8) {
        const __m128i a16 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(left + index));
        const __m128i b16 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(right + index));
        const __m256 a = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(a16), 16));
        const __m256 b = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(b16), 16));
        accumulator = _mm256_fmadd_ps(a, b, accumulator);
    }
    alignas(32) std::array<float, 8> lanes{};
    _mm256_store_ps(lanes.data(), accumulator);
    float result = 0.0f;
    for (float value : lanes) {
        result += value;
    }
    return result;
}

IFS_TARGET("avx512bf16,avx512f")
float dot_bf16_avx512_bf16(
    const std::uint16_t *left,
    const std::uint16_t *right) {
    __m512 accumulator = _mm512_setzero_ps();
    for (std::size_t index = 0; index < kDimension; index += 32) {
        const __m512bh a = (__m512bh)_mm512_loadu_si512(
            reinterpret_cast<const void *>(left + index));
        const __m512bh b = (__m512bh)_mm512_loadu_si512(
            reinterpret_cast<const void *>(right + index));
        accumulator = _mm512_dpbf16_ps(accumulator, a, b);
    }
    alignas(64) std::array<float, 16> lanes{};
    _mm512_store_ps(lanes.data(), accumulator);
    float result = 0.0f;
    for (float value : lanes) {
        result += value;
    }
    return result;
}

IFS_TARGET("avx2")
std::int32_t dot_int8_avx2(
    const std::int8_t *left,
    const std::int8_t *right,
    std::int32_t /*query_sum*/) {
    __m256i accumulator = _mm256_setzero_si256();
    for (std::size_t index = 0; index < kDimension; index += 32) {
        for (std::size_t half = 0; half < 32; half += 16) {
            const __m128i a8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(left + index + half));
            const __m128i b8 = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(right + index + half));
            const __m256i a16 = _mm256_cvtepi8_epi16(a8);
            const __m256i b16 = _mm256_cvtepi8_epi16(b8);
            accumulator = _mm256_add_epi32(
                accumulator, _mm256_madd_epi16(a16, b16));
        }
    }
    alignas(32) std::array<std::int32_t, 8> lanes{};
    _mm256_store_si256(reinterpret_cast<__m256i *>(lanes.data()), accumulator);
    std::int32_t result = 0;
    for (std::int32_t value : lanes) {
        result += value;
    }
    return result;
}

IFS_TARGET("avxvnni,avx2")
std::int32_t dot_int8_avx_vnni(
    const std::int8_t *left,
    const std::int8_t *right,
    std::int32_t query_sum) {
    const __m256i sign = _mm256_set1_epi8(static_cast<char>(0x80));
    __m256i accumulator = _mm256_setzero_si256();
    for (std::size_t index = 0; index < kDimension; index += 32) {
        const __m256i signed_left = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(left + index));
        const __m256i signed_right = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(right + index));
        const __m256i unsigned_left = _mm256_xor_si256(signed_left, sign);
        accumulator = _mm256_dpbusd_epi32(
            accumulator, unsigned_left, signed_right);
    }
    alignas(32) std::array<std::int32_t, 8> lanes{};
    _mm256_store_si256(reinterpret_cast<__m256i *>(lanes.data()), accumulator);
    std::int32_t result = 0;
    for (std::int32_t value : lanes) {
        result += value;
    }
    return result - 128 * query_sum;
}

IFS_TARGET("avx512vnni,avx512f")
std::int32_t dot_int8_avx512_vnni(
    const std::int8_t *left,
    const std::int8_t *right,
    std::int32_t query_sum) {
    const __m512i sign = _mm512_set1_epi8(static_cast<char>(0x80));
    __m512i accumulator = _mm512_setzero_si512();
    for (std::size_t index = 0; index < kDimension; index += 64) {
        const __m512i signed_left = _mm512_loadu_si512(
            reinterpret_cast<const void *>(left + index));
        const __m512i signed_right = _mm512_loadu_si512(
            reinterpret_cast<const void *>(right + index));
        const __m512i unsigned_left = _mm512_xor_si512(signed_left, sign);
        accumulator = _mm512_dpbusd_epi32(
            accumulator, unsigned_left, signed_right);
    }
    alignas(64) std::array<std::int32_t, 16> lanes{};
    _mm512_store_si512(reinterpret_cast<void *>(lanes.data()), accumulator);
    std::int32_t result = 0;
    for (std::int32_t value : lanes) {
        result += value;
    }
    return result - 128 * query_sum;
}

#else

CpuFeatures detect_features() {
    CpuFeatures features;
    features.x86_64 = IFS_CPU_X86_64 != 0;
    return features;
}

#endif

std::string feature_string(const CpuFeatures &features) {
    std::ostringstream output;
    output << "arch=" << (features.x86_64 ? "x86_64" : "non-x86_64")
           << ";os_avx=" << features.os_avx
           << ";os_avx512=" << features.os_avx512
           << ";fma=" << features.fma
           << ";avx2=" << features.avx2
           << ";avx512f=" << features.avx512f
           << ";avx_vnni=" << features.avx_vnni
           << ";avx512_vnni=" << features.avx512_vnni
           << ";avx512_bf16=" << features.avx512_bf16;
    return output.str();
}

}  // namespace

Dispatch make_dispatch() {
    const CpuFeatures features = detect_features();
    Dispatch selected{
        dot_f32_portable,
        dot_bf16_portable,
        dot_int8_portable,
        "portable-fp32",
        "portable-bf16-fp32-accumulate",
        "portable-int8-int32-accumulate",
        feature_string(features),
    };

#if IFS_CPU_X86_TARGETS
    if (features.avx2 && features.fma) {
        selected.fp32 = dot_f32_avx2_fma;
        selected.fp32_name = "avx2-fma-fp32";
        selected.bf16 = dot_bf16_avx2_fma;
        selected.bf16_name = "avx2-fma-bf16-fp32-accumulate";
    }
    if (features.avx512f) {
        selected.fp32 = dot_f32_avx512f;
        selected.fp32_name = "avx512f-fp32";
    }
    if (features.avx512_bf16) {
        selected.bf16 = dot_bf16_avx512_bf16;
        selected.bf16_name = "avx512-bf16-fp32-accumulate";
    }
    if (features.avx2) {
        selected.int8 = dot_int8_avx2;
        selected.int8_name = "avx2-int8-int32-accumulate";
    }
    if (features.avx_vnni) {
        selected.int8 = dot_int8_avx_vnni;
        selected.int8_name = "avx-vnni-signed-int8-int32-accumulate";
    }
    if (features.avx512_vnni) {
        selected.int8 = dot_int8_avx512_vnni;
        selected.int8_name = "avx512-vnni-signed-int8-int32-accumulate";
    }
#endif
    return selected;
}

}  // namespace ifs_cpu
