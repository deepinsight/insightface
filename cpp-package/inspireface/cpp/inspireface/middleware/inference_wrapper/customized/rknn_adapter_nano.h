//
// Created by Tunm-Air13 on 2023/11/3.
//

#ifndef SLEEPMONITORING_RKNN_ADAPTER_NANO_H
#define SLEEPMONITORING_RKNN_ADAPTER_NANO_H

#include "log.h"
#include "memory"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "rknn_api.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>
#include "log.h"

inline std::vector<float> softmax(const std::vector<float> &input) {
    std::vector<float> output;
    output.reserve(input.size());

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
        src = src + i * C1 * hw_src * C2;
        dst = dst + i * channel * hw_dst;
        for (int c = 0; c < channel; ++c) {
            int plane = c / C2;
            const int8_t *src_c = plane * hw_src * C2 + src;
            int offset = c % C2;
            for (int cur_h = 0; cur_h < h; ++cur_h)
                for (int cur_w = 0; cur_w < w; ++cur_w) {
                    int cur_hw = cur_h * w + cur_w;
                    dst[c * hw_dst + cur_h * w + cur_w] = (src_c[C2 * cur_hw + offset] - zp) * scale;  // int8-->float
                }
        }
    }

    return 0;
}

static void dump_tensor_attr(rknn_tensor_attr *attr) {
    char dims[128] = {0};
    for (int i = 0; i < attr->n_dims; ++i) {
        int idx = strlen(dims);
        sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
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
        int ret = rknn_init(&m_rk_ctx_, model_data, model_size, 0, NULL);
        if (ret < 0) {
            INSPIRE_LOGE("rknn_init fail! ret = %d", ret);
            return -1;
        }

        // Get sdk and driver version
        rknn_sdk_version sdk_ver;
        ret = rknn_query(m_rk_ctx_, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
        if (ret != RKNN_SUCC) {
            INSPIRE_LOGE("rknn_query fail! ret = %d", ret);
            return -1;
        }
        INSPIRE_LOGD("rknn_api/rknnrt version: %s, driver version: %s", sdk_ver.api_version, sdk_ver.drv_version);

        // Get Model Input Output Info
        ret = rknn_query(m_rk_ctx_, RKNN_QUERY_IN_OUT_NUM, &m_rk_io_num_, sizeof(m_rk_io_num_));
        if (ret != RKNN_SUCC) {
            INSPIRE_LOGE("rknn_query fail! ret = %d", ret);
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
            if (ret < 0) {
                INSPIRE_LOGE("rknn_init error! ret = %d", ret);
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
            if (ret != RKNN_SUCC) {
                INSPIRE_LOGE("rknn_query fail! ret = %d", ret);
                return -1;
            }
            dump_tensor_attr(&m_output_attrs_[i]);
        }

        // Get custom string
        rknn_custom_string custom_string;
        ret = rknn_query(m_rk_ctx_, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
        if (ret != RKNN_SUCC) {
            INSPIRE_LOGE("rknn_query fail! ret = %d", ret);
            return -1;
        }
        INSPIRE_LOGD("custom string: %s", custom_string.string);

        // Create input tensor memory
        m_input_mems_.resize(m_rk_io_num_.n_input);
        for (int i = 0; i < m_rk_io_num_.n_input; ++i) {
            m_input_mems_[i] = rknn_create_mem(m_rk_ctx_, m_input_attrs_[i].size_with_stride);
        }

        // Create output tensor memory
        m_output_mems_.resize(m_rk_io_num_.n_output);
        for (int i = 0; i < m_rk_io_num_.n_output; ++i) {
            m_output_mems_[i] = rknn_create_mem(m_rk_ctx_, m_output_attrs_[i].size_with_stride);
        }

        INSPIRE_LOGD("output origin tensors:");
        m_orig_output_attrs_.resize(m_rk_io_num_.n_output);
        for (uint32_t i = 0; i < m_rk_io_num_.n_output; i++) {
            memset(&m_orig_output_attrs_[i], 0, sizeof(m_orig_output_attrs_[i]));
            m_orig_output_attrs_[i].index = i;
            // query info
            ret = rknn_query(m_rk_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(m_orig_output_attrs_[i]), sizeof(rknn_tensor_attr));
            if (ret != RKNN_SUCC) {
                INSPIRE_LOGE("rknn_query fail! ret = %d", ret);
                return -1;
            }
            dump_tensor_attr(&m_orig_output_attrs_[i]);
        }

        run_ = true;
        return 0;
    }

    int32_t SetInputData(const int index, uint8_t *data, rknn_tensor_type type = RKNN_TENSOR_UINT8, rknn_tensor_format format = RKNN_TENSOR_NHWC) {
        if (index < m_input_mems_.size()) {
            m_input_attrs_[index].type = type;
            m_input_attrs_[index].fmt = format;
            // Copy input data to input tensor memory
            int width = m_input_attrs_[index].dims[2];
            int stride = m_input_attrs_[index].w_stride;
            if (width == stride) {
                memcpy(m_input_mems_[index]->virt_addr, data, width * m_input_attrs_[index].dims[1] * m_input_attrs_[index].dims[3]);
            } else {
                int height = m_input_attrs_[index].dims[1];
                int channel = m_input_attrs_[index].dims[3];
                // copy from src to dst with stride
                uint8_t *src_ptr = data;
                uint8_t *dst_ptr = (uint8_t *)m_input_mems_[index]->virt_addr;
                // width-channel elements
                int src_wc_elems = width * channel;
                int dst_wc_elems = stride * channel;
                for (int h = 0; h < height; ++h) {
                    memcpy(dst_ptr, src_ptr, src_wc_elems);
                    src_ptr += src_wc_elems;
                    dst_ptr += dst_wc_elems;
                }
            }
            // Set input tensor memory
            auto ret = rknn_set_io_mem(m_rk_ctx_, m_input_mems_[index], &m_input_attrs_[index]);
            if (ret < 0) {
                INSPIRE_LOGE("rknn_set_io_mem fail! ret = %d", ret);
                return -1;
            }
        } else {
            INSPIRE_LOGE("error: assert index < input size");
        }

        return 0;
    }

    int32_t RunSession(bool use_raw_output = false) {
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
                int num_elements = m_orig_output_attrs_[i].size_with_stride;
                m_output_nchw_[i].resize(num_elements);
            }

            for (uint32_t i = 0; i < m_rk_io_num_.n_output; i++) {
                if (m_output_attrs_[i].fmt == RKNN_TENSOR_NC1HWC2) {
                    int channel = m_orig_output_attrs_[i].dims[1];
                    int h = m_orig_output_attrs_[i].n_dims > 2 ? m_orig_output_attrs_[i].dims[2] : 1;
                    int w = m_orig_output_attrs_[i].n_dims > 3 ? m_orig_output_attrs_[i].dims[3] : 1;
                    int zp = m_output_attrs_[i].zp;
                    float scale = m_output_attrs_[i].scale;
                    NC1HWC2_int8_to_NCHW_float((int8_t *)m_output_mems_[i]->virt_addr, m_output_nchw_[i].data(), (int *)m_output_attrs_[i].dims,
                                               channel, h, w, zp, scale);
                } else {
                    int8_t *src = (int8_t *)m_output_mems_[i]->virt_addr;
                    float *dst = m_output_nchw_[i].data();
                    for (int index = 0; index < m_output_attrs_[i].n_elems; index++) {
                        dst[index] = (src[index] - m_output_attrs_[i].zp) * m_output_attrs_[i].scale;
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
        return (float *)(m_output_nchw_[index].data());
    }

    std::vector<unsigned long> GetOutputTensorSize(const int &index) {
        std::vector<unsigned long> dims(m_output_attrs_[index].dims, m_output_attrs_[index].dims + m_output_attrs_[index].n_dims);
        return dims;
    }

    ~RKNNAdapterNano() {
        Release();
    }

    void Release() {
        if (run_) {
            for (uint32_t i = 0; i < m_rk_io_num_.n_input; ++i) {
                rknn_destroy_mem(m_rk_ctx_, m_input_mems_[i]);
            }
            for (uint32_t i = 0; i < m_rk_io_num_.n_output; ++i) {
                rknn_destroy_mem(m_rk_ctx_, m_output_mems_[i]);
            }
            if (m_rk_ctx_) {
                rknn_destroy(m_rk_ctx_);
            }
        }
        run_ = false;
    }

private:
    rknn_context m_rk_ctx_;

    rknn_input_output_num m_rk_io_num_;
    std::vector<rknn_tensor_attr> m_input_attrs_;
    std::vector<rknn_tensor_attr> m_output_attrs_;
    std::vector<rknn_tensor_attr> m_orig_output_attrs_;

    std::vector<rknn_tensor_mem *> m_input_mems_;
    std::vector<rknn_tensor_mem *> m_output_mems_;

    std::vector<std::vector<float>> m_output_nchw_;
    bool run_;
};

#endif  // SLEEPMONITORING_RKNN_ADAPTER_NANO_H
