#ifndef INFERENCE_WRAPPER_LOG_
#define INFERENCE_WRAPPER_LOG_

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>

#if defined(ANDROID) || defined(__ANDROID__)
#define CV_COLOR_IS_RGB
#include <android/log.h>
#define INFERENCE_WRAPPER_LOG_NDK_TAG "Inspireface-Native-Inference"
#define INFERENCE_WRAPPER_LOG_PRINT_(...) __android_log_print(ANDROID_LOG_INFO, INFERENCE_WRAPPER_LOG_NDK_TAG, __VA_ARGS__)
#else
#define INFERENCE_WRAPPER_LOG_PRINT_(...) printf(__VA_ARGS__)
#endif

#define INFERENCE_WRAPPER_LOG_PRINT(INFERENCE_WRAPPER_LOG_PRINT_TAG, ...)                     \
    do {                                                                                      \
        INFERENCE_WRAPPER_LOG_PRINT_("[" INFERENCE_WRAPPER_LOG_PRINT_TAG "][%d] ", __LINE__); \
        INFERENCE_WRAPPER_LOG_PRINT_(__VA_ARGS__);                                            \
    } while (0);

#define INFERENCE_WRAPPER_LOG_PRINT_E(INFERENCE_WRAPPER_LOG_PRINT_TAG, ...)                        \
    do {                                                                                           \
        INFERENCE_WRAPPER_LOG_PRINT_("[ERR: " INFERENCE_WRAPPER_LOG_PRINT_TAG "][%d] ", __LINE__); \
        INFERENCE_WRAPPER_LOG_PRINT_(__VA_ARGS__);                                                 \
    } while (0);

#endif
