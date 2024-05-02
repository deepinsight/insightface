# Overview of CMake Option for Compilation

Here are the translation details for the compilation parameters as per your requirement:

| **Parameter** | **Default Value** | **Description** |
| --- | --- | --- |
| THIRD_PARTY_DIR | 3rdparty | Path for required third-party libraries |
| SANITIZE_ADDRESS | OFF | Enable AddressSanitizer for memory error detection |
| SANITIZE_LEAK | OFF | Enable LeakSanitizer to detect memory leaks |
| ENABLE_RKNN | OFF | Enable RKNN for Rockchip embedded devices |
| RK_DEVICE_TYPE | RV1109RV1126 | Target device model for Rockchip (currently supports only RV1109 and RV1126) |
| BUILD_LINUX_ARM7 | OFF | Compile for ARM7 architecture |
| BUILD_LINUX_AARCH64 | OFF | Compile for AARCH64 architecture |
| BUILD_WITH_TEST | OFF | Compile test case programs |
| BUILD_WITH_SAMPLE | ON | Compile sample programs |
| LINUX_FETCH_MNN | OFF | Use fetch feature to download MNN source code online for compilation |
| BUILD_SHARED_LIBS | ON | Compile shared libraries |
| ENABLE_BENCHMARK | OFF | Enable Benchmark tests for test cases |
| ENABLE_USE_LFW_DATA | OFF | Enable using LFW data for test cases |
| ENABLE_TEST_EVALUATION | OFF | Enable evaluation functionality for test cases, must be used together with ENABLE_USE_LFW_DATA |
| GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA | OFF | Enable global MNN_CUDA inference mode, requires device support for CUDA |
| LINUX_MNN_CUDA | "" | Specific MNN library path, requires pre-compiled MNN library supporting MNN_CUDA, only effective when GLOBAL_INFERENCE_BACKEND_USE_MNN_CUDA is enabled |
