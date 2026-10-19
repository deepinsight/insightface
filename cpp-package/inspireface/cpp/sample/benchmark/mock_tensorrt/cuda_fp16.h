#ifndef INSPIREFACE_SAMPLE_FAKE_CUDA_FP16_H
#define INSPIREFACE_SAMPLE_FAKE_CUDA_FP16_H

#include <cstdint>

struct half {
    uint16_t bits;
};

inline float __half2float(half value) {
    return static_cast<float>(value.bits);
}

#endif
