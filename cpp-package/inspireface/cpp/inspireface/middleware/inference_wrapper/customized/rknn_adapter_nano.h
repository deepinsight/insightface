//
// Created by Tunm-Air13 on 2023/11/3.
//

#ifndef SLEEPMONITORING_RKNN_ADAPTER_NANO_H
#define SLEEPMONITORING_RKNN_ADAPTER_NANO_H

#include "log.h"
#include "memory"
#include <cstdio>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <sys/time.h>
#endif
#include "rknn_api.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include "log.h"

inline std::vector<float> softmax(const std::vector<float> &input) {
    std::vector<float> output;
    output.reserve(input.size());

    if (input.empty()) {
        return output;
    }

    float max = *std::max_element(input.begin(), input.end());
    float sum = 0.0;

    for (float val : input) {
        sum += std::exp(val - max);
    }

    for (float val : input) {
        output.push_back(std::exp(val - max) / sum);
    }

    return output;
}

inline int NC1HWC2_int8_to_NCHW_float(const int8_t *src, float *dst, int *dims, int channel, int h, int w, int zp, float scale) {
    int batch = dims[0];
    int C1 = dims[1];
    int C2 = dims[4];
    int hw_src = dims[2] * dims[3];
    int hw_dst = h * w;
    for (int i = 0; i < batch; i++) {
        const int8_t *batch_src = src + static_cast<size_t>(i) * C1 * hw_src * C2;
        float *batch_dst = dst + static_cast<size_t>(i) * channel * hw_dst;
        for (int c = 0; c < channel; ++c) {
            int plane = c / C2;
            const int8_t *src_c = plane * hw_src * C2 + batch_src;
            int offset = c % C2;
            for (int cur_h = 0; cur_h < h; ++cur_h)
                for (int cur_w = 0; cur_w < w; ++cur_w) {
                    int cur_hw = cur_h * w + cur_w;
                    batch_dst[c * hw_dst + cur_h * w + cur_w] = (src_c[C2 * cur_hw + offset] - zp) * scale;  // int8-->float
                }
        }
    }

    return 0;
}

static void dump_tensor_attr(rknn_tensor_attr *attr) {
    if (attr == nullptr) {
        return;
    }
    char dims[128] = {0};
    const uint32_t dimension_count = std::min<uint32_t>(attr->n_dims, RKNN_MAX_DIMS);
    size_t used = 0;
    for (uint32_t i = 0; i < dimension_count && used < sizeof(dims); ++i) {
        const int written = std::snprintf(dims + used, sizeof(dims) - used, "%u%s", attr->dims[i], (i + 1 == dimension_count) ? "" : ", ");
        if (written < 0 || static_cast<size_t>(written) >= sizeof(dims) - used) {
            dims[sizeof(dims) - 1] = '\0';
            break;
        }
        used += static_cast<size_t>(written);
    }
    INSPIRE_LOGD(
      "  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
      "zp=%d, scale=%f",
      attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
      get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

class RKNNAdapterNano {
public:
    RKNNAdapterNano(const RKNNAdapterNano &) = delete;
    RKNNAdapterNano &operator=(const RKNNAdapterNano &) = delete;
    RKNNAdapterNano() = default;

    int32_t Initialize(void *model_data, unsigned int model_size) {
        Release();
        if (model_data == nullptr || model_size == 0) {
            INSPIRE_LOGE("RKNN model buffer is empty");
            return -1;
        }
        int ret = rknn_init(&m_rk_ctx_, model_data, model_size, 0, NULL);
        if (ret < 0) {
            INSPIRE_LOGE("rknn_init fail! ret = %d", ret);
            Release();
            return -1;
        }

        // Get sdk and driver version
        rknn_sdk_version sdk_ver{};
        ret = rknn_query(m_rk_ctx_, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
        if (ret != RKNN_SUCC) {
            INSPIRE_LOGE("rknn_query fail! ret = %d", ret);
            Release();
            return -1;
        }
        INSPIRE_LOGD("rknn_api/rknnrt version: %s, driver version: %s", sdk_ver.api_version, sdk_ver.drv_version);

        // Get Model Input Output Info
        ret = rknn_query(m_rk_ctx_, RKNN_QUERY_IN_OUT_NUM, &m_rk_io_num_, sizeof(m_rk_io_num_));
        if (ret != RKNN_SUCC || m_rk_io_num_.n_input == 0 || m_rk_io_num_.n_output == 0) {
            INSPIRE_LOGE("rknn_query fail! ret = %d", ret);
            Release();
            return -1;
        }
        INSPIRE_LOGD("model input num: %d, output num: %d", m_rk_io_num_.n_input, m_rk_io_num_.n_output);

        INSPIRE_LOGD("input tensors:");
        m_input_attrs_.resize(m_rk_io_num_.n_input);
        for (uint32_t i = 0; i < m_rk_io_num_.n_input; i++) {
            memset(&m_input_attrs_[i], 0, sizeof(m_input_attrs_[i]));
            m_input_attrs_[i].index = i;
            // query info
            ret = rknn_query(m_rk_ctx_, RKNN_QUERY_INPUT_ATTR, &(m_input_attrs_[i]), sizeof(rknn_tensor_attr));
            if (ret < 0 || !IsValidTensorAttr(m_input_attrs_[i])) {
                INSPIRE_LOGE("rknn_init error! ret = %d", ret);
                Release();
                return -1;
            }
            dump_tensor_attr(&m_input_attrs_[i]);
        }

        INSPIRE_LOGD("output tensors:");
        m_output_attrs_.resize(m_rk_io_num_.n_output);
        for (uint32_t i = 0; i < m_rk_io_num_.n_output; i++) {
            memset(&m_output_attrs_[i], 0, sizeof(m_output_attrs_[i]));
            m_output_attrs_[i].index = i;
            // query info
            ret = rknn_query(m_rk_ctx_, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(m_output_attrs_[i]), sizeof(rknn_tensor_attr));
            if (ret != RKNN_SUCC || !IsValidTensorAttr(m_output_attrs_[i])) {
                INSPIRE_LOGE("rknn_query fail! ret = %d", ret);
                Release();
                return -1;
            }
            dump_tensor_attr(&m_output_attrs_[i]);
        }

        // Get custom string
        rknn_custom_string custom_string{};
        ret = rknn_query(m_rk_ctx_, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
        if (ret != RKNN_SUCC) {
            INSPIRE_LOGE("rknn_query fail! ret = %d", ret);
            Release();
            return -1;
        }
        INSPIRE_LOGD("custom string: %s", custom_string.string);

        // Create input tensor memory
        m_input_mems_.resize(m_rk_io_num_.n_input);
        for (uint32_t i = 0; i < m_rk_io_num_.n_input; ++i) {
            m_input_mems_[i] = rknn_create_mem(m_rk_ctx_, m_input_attrs_[i].size_with_stride);
            if (m_input_mems_[i] == nullptr || m_input_mems_[i]->virt_addr == nullptr ||
                m_input_mems_[i]->size < m_input_attrs_[i].size_with_stride) {
                INSPIRE_LOGE("Failed to allocate RKNN input memory %u", i);
                Release();
                return -1;
            }
        }

        // Create output tensor memory
        m_output_mems_.resize(m_rk_io_num_.n_output);
        for (uint32_t i = 0; i < m_rk_io_num_.n_output; ++i) {
            m_output_mems_[i] = rknn_create_mem(m_rk_ctx_, m_output_attrs_[i].size_with_stride);
            if (m_output_mems_[i] == nullptr || m_output_mems_[i]->virt_addr == nullptr ||
                m_output_mems_[i]->size < m_output_attrs_[i].size_with_stride) {
                INSPIRE_LOGE("Failed to allocate RKNN output memory %u", i);
                Release();
                return -1;
            }
        }

        INSPIRE_LOGD("output origin tensors:");
        m_orig_output_attrs_.resize(m_rk_io_num_.n_output);
        for (uint32_t i = 0; i < m_rk_io_num_.n_output; i++) {
            memset(&m_orig_output_attrs_[i], 0, sizeof(m_orig_output_attrs_[i]));
            m_orig_output_attrs_[i].index = i;
            // query info
            ret = rknn_query(m_rk_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(m_orig_output_attrs_[i]), sizeof(rknn_tensor_attr));
            if (ret != RKNN_SUCC || !IsValidTensorAttr(m_orig_output_attrs_[i])) {
                INSPIRE_LOGE("rknn_query fail! ret = %d", ret);
                Release();
                return -1;
            }
            dump_tensor_attr(&m_orig_output_attrs_[i]);
        }

        run_ = true;
        return 0;
    }

    int32_t SetInputData(const int index, const void *data, int width, int height, int channel,
                         rknn_tensor_type type = RKNN_TENSOR_UINT8, rknn_tensor_format format = RKNN_TENSOR_NHWC) {
        if (!run_ || index < 0 || static_cast<size_t>(index) >= m_input_mems_.size() ||
            static_cast<size_t>(index) >= m_input_attrs_.size() || data == nullptr || width <= 0 || height <= 0 || channel <= 0) {
            INSPIRE_LOGE("Invalid RKNN input metadata");
            return -1;
        }

        const size_t element_bytes = TensorTypeBytes(type);
        size_t source_row_bytes = 0;
        size_t source_size = 0;
        const uint32_t configured_stride = m_input_attrs_[index].w_stride;
        const size_t destination_width = configured_stride == 0 ? static_cast<size_t>(width) : configured_stride;
        size_t destination_row_bytes = 0;
        size_t destination_size = 0;
        if (element_bytes == 0 || !CheckedMultiply(static_cast<size_t>(width), static_cast<size_t>(channel), source_row_bytes) ||
            !CheckedMultiply(source_row_bytes, element_bytes, source_row_bytes) ||
            !CheckedMultiply(source_row_bytes, static_cast<size_t>(height), source_size) ||
            !CheckedMultiply(destination_width, static_cast<size_t>(channel), destination_row_bytes) ||
            !CheckedMultiply(destination_row_bytes, element_bytes, destination_row_bytes) ||
            !CheckedMultiply(destination_row_bytes, static_cast<size_t>(height), destination_size) ||
            destination_width < static_cast<size_t>(width) || m_input_mems_[index] == nullptr ||
            m_input_mems_[index]->virt_addr == nullptr || destination_size > m_input_mems_[index]->size) {
            INSPIRE_LOGE("RKNN input dimensions or storage are invalid");
            return -1;
        }

        m_input_attrs_[index].type = type;
        m_input_attrs_[index].fmt = format;
        const auto *source = static_cast<const uint8_t *>(data);
        auto *destination = static_cast<uint8_t *>(m_input_mems_[index]->virt_addr);
        if (source_row_bytes == destination_row_bytes) {
            std::memcpy(destination, source, source_size);
        } else if (format == RKNN_TENSOR_NHWC) {
            for (int row = 0; row < height; ++row) {
                std::memcpy(destination + static_cast<size_t>(row) * destination_row_bytes,
                            source + static_cast<size_t>(row) * source_row_bytes, source_row_bytes);
            }
        } else if (format == RKNN_TENSOR_NCHW) {
            const size_t source_channel_bytes = source_row_bytes / static_cast<size_t>(channel);
            const size_t destination_channel_bytes = destination_row_bytes / static_cast<size_t>(channel);
            for (int plane = 0; plane < channel; ++plane) {
                for (int row = 0; row < height; ++row) {
                    const size_t source_offset = (static_cast<size_t>(plane) * height + row) * source_channel_bytes;
                    const size_t destination_offset = (static_cast<size_t>(plane) * height + row) * destination_channel_bytes;
                    std::memcpy(destination + destination_offset, source + source_offset, source_channel_bytes);
                }
            }
        } else {
            INSPIRE_LOGE("Unsupported RKNN input tensor format");
            return -1;
        }

        const auto ret = rknn_set_io_mem(m_rk_ctx_, m_input_mems_[index], &m_input_attrs_[index]);
        if (ret < 0) {
            INSPIRE_LOGE("rknn_set_io_mem fail! ret = %d", ret);
            return -1;
        }

        return 0;
    }

    int32_t RunSession(bool use_raw_output = false) {
        if (!run_ || m_output_mems_.size() != m_output_attrs_.size() ||
            m_output_attrs_.size() != m_orig_output_attrs_.size()) {
            return -1;
        }
        // Set output tensor memory
        for (uint32_t i = 0; i < m_rk_io_num_.n_output; ++i) {
            // set output memory and attribute
            auto ret = rknn_set_io_mem(m_rk_ctx_, m_output_mems_[i], &m_output_attrs_[i]);
            if (ret < 0) {
                INSPIRE_LOGE("rknn_set_io_mem fail! ret = %d", ret);
                return -1;
            }
        }

        auto ret = rknn_run(m_rk_ctx_, NULL);
        if (ret < 0) {
            printf("rknn run error %d\n", ret);
            return -1;
        }

        if (use_raw_output) {
            m_output_nchw_.resize(m_rk_io_num_.n_output);
            for (uint32_t i = 0; i < m_rk_io_num_.n_output; ++i) {
                const size_t num_elements = m_orig_output_attrs_[i].n_elems;
                if (num_elements == 0 || m_output_mems_[i] == nullptr || m_output_mems_[i]->virt_addr == nullptr) {
                    return -1;
                }
                m_output_nchw_[i].resize(num_elements);
                if (m_output_attrs_[i].fmt == RKNN_TENSOR_NC1HWC2) {
                    if (m_output_attrs_[i].type != RKNN_TENSOR_INT8 || m_output_attrs_[i].n_dims < 5 ||
                        m_output_attrs_[i].dims[0] == 0 || m_output_attrs_[i].dims[1] == 0 ||
                        m_output_attrs_[i].dims[2] == 0 || m_output_attrs_[i].dims[3] == 0 || m_output_attrs_[i].dims[4] == 0) {
                        return -1;
                    }
                    size_t source_element_count = 1;
                    for (uint32_t dimension = 0; dimension < 5; ++dimension) {
                        if (!CheckedMultiply(source_element_count, m_output_attrs_[i].dims[dimension], source_element_count)) {
                            return -1;
                        }
                    }
                    if (source_element_count > m_output_mems_[i]->size || m_orig_output_attrs_[i].n_dims < 2) {
                        return -1;
                    }
                    int channel = m_orig_output_attrs_[i].dims[1];
                    int h = m_orig_output_attrs_[i].n_dims > 2 ? m_orig_output_attrs_[i].dims[2] : 1;
                    int w = m_orig_output_attrs_[i].n_dims > 3 ? m_orig_output_attrs_[i].dims[3] : 1;
                    size_t converted_element_count = m_output_attrs_[i].dims[0];
                    if (channel <= 0 || h <= 0 || w <= 0 ||
                        !CheckedMultiply(converted_element_count, static_cast<size_t>(channel), converted_element_count) ||
                        !CheckedMultiply(converted_element_count, static_cast<size_t>(h), converted_element_count) ||
                        !CheckedMultiply(converted_element_count, static_cast<size_t>(w), converted_element_count) ||
                        converted_element_count != num_elements) {
                        return -1;
                    }
                    int zp = m_output_attrs_[i].zp;
                    float scale = m_output_attrs_[i].scale;
                    NC1HWC2_int8_to_NCHW_float((int8_t *)m_output_mems_[i]->virt_addr, m_output_nchw_[i].data(), (int *)m_output_attrs_[i].dims,
                                               channel, h, w, zp, scale);
                } else {
                    if (m_output_attrs_[i].n_elems != num_elements ||
                        !CopyOutputToFloat(m_output_attrs_[i], *m_output_mems_[i], m_output_nchw_[i])) {
                        return -1;
                    }
                }
            }
        }

        return 0;
    }

    std::vector<float> &GetOutputData(size_t index) {
        return m_output_nchw_[index];
    }

    rknn_tensor_mem *GetOutputRawData(size_t index) {
        return m_output_mems_[index];
    }

    std::vector<rknn_tensor_attr> &GetOutputAttrs() {
        return m_output_attrs_;
    }

    const float *GetOutputDataPtr(const int index) {
        if (index < 0 || static_cast<size_t>(index) >= m_output_nchw_.size() || m_output_nchw_[index].empty()) {
            return nullptr;
        }
        return m_output_nchw_[index].data();
    }

    std::vector<unsigned long> GetOutputTensorSize(const int &index) {
        if (index < 0 || static_cast<size_t>(index) >= m_output_attrs_.size() ||
            static_cast<size_t>(index) >= m_orig_output_attrs_.size()) {
            return {};
        }
        const auto &attribute = m_output_attrs_[index].fmt == RKNN_TENSOR_NC1HWC2
                                  ? m_orig_output_attrs_[index]
                                  : m_output_attrs_[index];
        std::vector<unsigned long> dims(attribute.dims, attribute.dims + attribute.n_dims);
        return dims;
    }

    ~RKNNAdapterNano() {
        Release();
    }

    void Release() {
        for (auto *memory : m_input_mems_) {
            if (memory != nullptr) {
                rknn_destroy_mem(m_rk_ctx_, memory);
            }
        }
        for (auto *memory : m_output_mems_) {
            if (memory != nullptr) {
                rknn_destroy_mem(m_rk_ctx_, memory);
            }
        }
        if (m_rk_ctx_ != 0) {
            rknn_destroy(m_rk_ctx_);
        }
        m_rk_ctx_ = 0;
        m_rk_io_num_ = {};
        m_input_attrs_.clear();
        m_output_attrs_.clear();
        m_orig_output_attrs_.clear();
        m_input_mems_.clear();
        m_output_mems_.clear();
        m_output_nchw_.clear();
        run_ = false;
    }

private:
    static bool CheckedMultiply(size_t left, size_t right, size_t &result) {
        if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
            return false;
        }
        result = left * right;
        return true;
    }

    static size_t TensorTypeBytes(rknn_tensor_type type) {
        switch (type) {
            case RKNN_TENSOR_FLOAT32:
            case RKNN_TENSOR_INT32:
            case RKNN_TENSOR_UINT32:
                return 4;
            case RKNN_TENSOR_FLOAT16:
            case RKNN_TENSOR_INT16:
            case RKNN_TENSOR_UINT16:
                return 2;
            case RKNN_TENSOR_INT8:
            case RKNN_TENSOR_UINT8:
            case RKNN_TENSOR_BOOL:
                return 1;
            case RKNN_TENSOR_INT64:
                return 8;
            default:
                return 0;
        }
    }

    static bool IsValidTensorAttr(const rknn_tensor_attr &attribute) {
        if (attribute.n_dims == 0 || attribute.n_dims > RKNN_MAX_DIMS || attribute.n_elems == 0 ||
            attribute.size_with_stride == 0) {
            return false;
        }
        for (uint32_t index = 0; index < attribute.n_dims; ++index) {
            if (attribute.dims[index] == 0) {
                return false;
            }
        }
        return true;
    }

    static bool CopyOutputToFloat(const rknn_tensor_attr &attribute, const rknn_tensor_mem &memory,
                                  std::vector<float> &destination) {
        const size_t element_bytes = TensorTypeBytes(attribute.type);
        if (element_bytes == 0 || memory.virt_addr == nullptr || destination.size() != attribute.n_elems) {
            return false;
        }

        size_t row_count = 1;
        size_t logical_row_elements = attribute.n_elems;
        size_t storage_row_elements = logical_row_elements;
        if (attribute.n_dims == 4 && attribute.w_stride != 0) {
            if (attribute.fmt == RKNN_TENSOR_NHWC) {
                row_count = static_cast<size_t>(attribute.dims[0]) * attribute.dims[1];
                logical_row_elements = static_cast<size_t>(attribute.dims[2]) * attribute.dims[3];
                storage_row_elements = static_cast<size_t>(attribute.w_stride) * attribute.dims[3];
            } else if (attribute.fmt == RKNN_TENSOR_NCHW) {
                row_count = static_cast<size_t>(attribute.dims[0]) * attribute.dims[1] * attribute.dims[2];
                logical_row_elements = attribute.dims[3];
                storage_row_elements = attribute.w_stride;
            }
        }

        size_t logical_elements = 0;
        size_t storage_elements = 0;
        size_t storage_bytes = 0;
        if (storage_row_elements < logical_row_elements ||
            !CheckedMultiply(row_count, logical_row_elements, logical_elements) || logical_elements != destination.size() ||
            !CheckedMultiply(row_count, storage_row_elements, storage_elements) ||
            !CheckedMultiply(storage_elements, element_bytes, storage_bytes) || storage_bytes > memory.size) {
            return false;
        }

        const auto *source = static_cast<const uint8_t *>(memory.virt_addr);
        size_t destination_index = 0;
        for (size_t row = 0; row < row_count; ++row) {
            const size_t source_row_offset = row * storage_row_elements;
            for (size_t column = 0; column < logical_row_elements; ++column) {
                const size_t source_index = source_row_offset + column;
                switch (attribute.type) {
                    case RKNN_TENSOR_FLOAT32: {
                        float value = 0.0f;
                        std::memcpy(&value, source + source_index * sizeof(float), sizeof(float));
                        destination[destination_index++] = value;
                        break;
                    }
                    case RKNN_TENSOR_INT8:
                        destination[destination_index++] =
                          (reinterpret_cast<const int8_t *>(source)[source_index] - attribute.zp) * attribute.scale;
                        break;
                    case RKNN_TENSOR_UINT8:
                        destination[destination_index++] =
                          (static_cast<int32_t>(source[source_index]) - attribute.zp) * attribute.scale;
                        break;
                    default:
                        return false;
                }
            }
        }
        return destination_index == destination.size();
    }

    rknn_context m_rk_ctx_{};

    rknn_input_output_num m_rk_io_num_{};
    std::vector<rknn_tensor_attr> m_input_attrs_;
    std::vector<rknn_tensor_attr> m_output_attrs_;
    std::vector<rknn_tensor_attr> m_orig_output_attrs_;

    std::vector<rknn_tensor_mem *> m_input_mems_;
    std::vector<rknn_tensor_mem *> m_output_mems_;

    std::vector<std::vector<float>> m_output_nchw_;
    bool run_{false};
};

#endif  // SLEEPMONITORING_RKNN_ADAPTER_NANO_H
