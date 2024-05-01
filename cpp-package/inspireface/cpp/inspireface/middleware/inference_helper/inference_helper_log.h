/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef INFERENCE_HELPER_LOG_
#define INFERENCE_HELPER_LOG_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>


#if defined(ANDROID) || defined(__ANDROID__)
#define CV_COLOR_IS_RGB
#include <android/log.h>
#define INFERENCE_HELPER_LOG_NDK_TAG "HyperLPR3-Native-Inference"
#define INFERENCE_HELPER_LOG_PRINT_(...) __android_log_print(ANDROID_LOG_INFO, INFERENCE_HELPER_LOG_NDK_TAG, __VA_ARGS__)
#else
#define INFERENCE_HELPER_LOG_PRINT_(...) printf(__VA_ARGS__)
#endif

#define INFERENCE_HELPER_LOG_PRINT(INFERENCE_HELPER_LOG_PRINT_TAG, ...) do { \
    INFERENCE_HELPER_LOG_PRINT_("[" INFERENCE_HELPER_LOG_PRINT_TAG "][%d] ", __LINE__); \
    INFERENCE_HELPER_LOG_PRINT_(__VA_ARGS__); \
} while(0);

#define INFERENCE_HELPER_LOG_PRINT_E(INFERENCE_HELPER_LOG_PRINT_TAG, ...) do { \
    INFERENCE_HELPER_LOG_PRINT_("[ERR: " INFERENCE_HELPER_LOG_PRINT_TAG "][%d] ", __LINE__); \
    INFERENCE_HELPER_LOG_PRINT_(__VA_ARGS__); \
} while(0);

#endif
