#include <inspireface/include/inspireface/inspireface.hpp>

#include "settings/test_settings.h"

TEST_CASE("C++ CUDA helpers validate output pointers", "[cpp_api][contract][cuda][boundary]") {
    CHECK(inspire::GetCudaDeviceCount(nullptr) == HERR_INVALID_PARAM);
    CHECK(inspire::CheckCudaUsability(nullptr) == HERR_INVALID_PARAM);
}

TEST_CASE("C++ CUDA helpers report the build capability coherently", "[cpp_api][platform][cuda]") {
    int32_t device_count = -1;
    int32_t supported = -1;
    const int32_t count_status = inspire::GetCudaDeviceCount(&device_count);
    const int32_t support_status = inspire::CheckCudaUsability(&supported);
#ifdef ISF_ENABLE_TENSORRT
    CHECK((count_status == HSUCCEED || count_status == HERR_DEVICE_CUDA_UNKNOWN_ERROR));
    CHECK((support_status == HSUCCEED || support_status == HERR_DEVICE_CUDA_NOT_SUPPORT ||
           support_status == HERR_DEVICE_CUDA_UNKNOWN_ERROR));
    if (count_status == HSUCCEED) {
        CHECK(device_count >= 0);
    }
#else
    CHECK(count_status == HERR_DEVICE_CUDA_NOT_SUPPORT);
    CHECK(support_status == HERR_DEVICE_CUDA_NOT_SUPPORT);
    CHECK(device_count == 0);
    CHECK(supported == 0);
#endif
    const int32_t detail_status = inspire::_PrintCudaDeviceInfo();
    const int32_t print_status = inspire::PrintCudaDeviceInfo();
    CHECK((detail_status == HSUCCEED || detail_status == HERR_DEVICE_CUDA_NOT_SUPPORT ||
           detail_status == HERR_DEVICE_CUDA_UNKNOWN_ERROR));
    CHECK((print_status == HSUCCEED || print_status == HERR_DEVICE_CUDA_NOT_SUPPORT ||
           print_status == HERR_DEVICE_CUDA_UNKNOWN_ERROR));
}
