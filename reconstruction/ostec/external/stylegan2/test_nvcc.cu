// Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/stylegan2/license.html

#include <cstdio>

void checkCudaError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

__global__ void cudaKernel(void)
{
    printf("GPU says hello.\n");
}

int main(void)
{
    printf("CPU says hello.\n");
    checkCudaError(cudaLaunchKernel((void*)cudaKernel, 1, 1, NULL, 0, NULL));
    checkCudaError(cudaDeviceSynchronize());
    return 0;
}
