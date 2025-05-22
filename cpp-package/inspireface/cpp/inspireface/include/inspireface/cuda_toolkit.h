#ifndef INSPIRE_CUDA_TOOLKIT_H
#define INSPIRE_CUDA_TOOLKIT_H

#include "data_type.h"

namespace inspire {

// Get the number of CUDA devices
int32_t INSPIRE_API_EXPORT GetCudaDeviceCount(int32_t *device_count);

// Check the availability of CUDA
int32_t INSPIRE_API_EXPORT CheckCudaUsability(int32_t *is_support);

// Internal function, print detailed information of CUDA devices
int32_t INSPIRE_API_EXPORT _PrintCudaDeviceInfo();

// Wrapper function to print CUDA device information
int32_t INSPIRE_API_EXPORT PrintCudaDeviceInfo();

}  // namespace inspire

#endif  // INSPIRE_CUDA_TOOLKIT_H