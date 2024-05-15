#include <cstdint>

#if defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#elif (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__ARM_NEON__)
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#pragma message("USE SSE")
#endif

#if defined(__GNUC__) &&                                                       \
    (defined(__x86_64__) || defined(__i386__) || defined(_MSC_VER))
inline float simd_dot(const float *x, const float *y, const long &len) {
//#pragma message("USE SSE")
  float inner_prod = 0.0f;
  __m128 X, Y, Z;                // 128-bit values
  __m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
  float temp[4];

  long i;
  for (i = 0; i + 4 < len; i += 4) {
    X = _mm_loadu_ps(x + i); // load chunk of 4 floats
    Y = _mm_loadu_ps(y + i);
    Z = _mm_mul_ps(X, Y);
    acc = _mm_add_ps(acc, Z);
  }
  _mm_storeu_ps(&temp[0], acc); // store acc into an array
  inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

  // add the remaining values
  for (; i < len; ++i) {
    inner_prod += x[i] * y[i];
  }
  return inner_prod;
}
#else
inline float simd_dot(const float *x, const float *y, const long &len) {
//#pragma message("USE NEON")
  float inner_prod = 0.0f;
  float32x4_t X, Y, Z;                 // 128-bit values
  float32x4_t acc = vdupq_n_f32(0.0f); // set to (0, 0, 0, 0)
  long i;
  for (i = 0; i + 4 < len; i += 4) {
    X = vld1q_f32(x + i); // load chunk of 4 floats
    Y = vld1q_f32(y + i);
    Z = vmulq_f32(X, Y);
    acc = vaddq_f32(acc, Z);
  }
  inner_prod = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) +
               vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);
  for (; i < len; ++i) {
    inner_prod += x[i] * y[i];
  }
  return inner_prod;
}
#endif
