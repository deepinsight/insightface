#!/bin/bash

# 函数：获取CUDA和Ubuntu版本标签
# 如果CUDA_TAG环境变量已设置，则使用该值
# 否则自动检测CUDA和Ubuntu版本并生成标签
# 格式: cudaXX_ubuntuXX.XX
# 如果检测不到某个版本，则用"none"代替
get_cuda_ubuntu_tag() {
    # 如果CUDA_TAG已设置，则直接返回
    if [ -n "${CUDA_TAG}" ]; then
        echo "${CUDA_TAG}"
        return 0
    fi
    
    # 获取CUDA版本
    CUDA_VERSION="_none"
    if command -v nvcc &> /dev/null; then
        # 尝试从nvcc获取版本
        CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d',' -f1 | tr -d '.')
        if [ -z "${CUDA_VERSION}" ]; then
            CUDA_VERSION="_none"
        else
            CUDA_VERSION="${CUDA_VERSION}"
        fi
    elif [ -f "/usr/local/cuda/version.txt" ]; then
        # 尝试从CUDA安装目录获取版本
        CUDA_VERSION=$(cat /usr/local/cuda/version.txt 2>/dev/null | grep "CUDA Version" | awk '{print $3}' | tr -d '.')
        if [ -z "${CUDA_VERSION}" ]; then
            CUDA_VERSION="_none"
        fi
    elif [ -d "/usr/local/cuda" ] && ls -l /usr/local/cuda 2>/dev/null | grep -q "cuda-"; then
        # 尝试从符号链接获取版本
        CUDA_LINK=$(ls -l /usr/local/cuda 2>/dev/null | grep -o "cuda-[0-9.]*" | head -n 1)
        CUDA_VERSION=$(echo "${CUDA_LINK}" | cut -d'-' -f2 | tr -d '.')
        if [ -z "${CUDA_VERSION}" ]; then
            CUDA_VERSION="_none"
        fi
    fi
    
    # 获取Ubuntu版本
    UBUNTU_VERSION="_none"
    if [ -f "/etc/os-release" ]; then
        # 检查是否是Ubuntu
        if grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
            UBUNTU_VERSION=$(grep "VERSION_ID" /etc/os-release 2>/dev/null | cut -d'"' -f2)
            if [ -z "${UBUNTU_VERSION}" ]; then
                UBUNTU_VERSION="_none"
            fi
        fi
    elif [ -f "/etc/lsb-release" ]; then
        # 尝试从lsb-release获取版本
        if grep -q "Ubuntu" /etc/lsb-release 2>/dev/null; then
            UBUNTU_VERSION=$(grep "DISTRIB_RELEASE" /etc/lsb-release 2>/dev/null | cut -d'=' -f2)
            if [ -z "${UBUNTU_VERSION}" ]; then
                UBUNTU_VERSION="_none"
            fi
        fi
    fi
    
    # 生成并返回标签
    echo "cuda${CUDA_VERSION}_ubuntu${UBUNTU_VERSION}"
}

# 使用示例
CUDA_TAG=$(get_cuda_ubuntu_tag)
echo "Generated tag: ${CUDA_TAG}"