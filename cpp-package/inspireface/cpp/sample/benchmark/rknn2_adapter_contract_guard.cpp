#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "middleware/inference_wrapper/inference_wrapper_rknn_adapter_nano.h"

namespace {

using Clock = std::chrono::steady_clock;

struct FakeRuntimeState {
    std::vector<rknn_tensor_mem *> memories;
    int init_calls{0};
    int destroy_calls{0};
    int create_calls{0};
    int destroy_memory_calls{0};
    int run_calls{0};
    int set_io_calls{0};
    int fail_create_call{-1};
    rknn_query_cmd fail_query{RKNN_QUERY_CMD_MAX};
} runtime;

void ResetRuntime() {
    runtime = {};
}

void FillAttribute(rknn_tensor_attr &attribute, bool input, bool native_output) {
    const uint32_t index = attribute.index;
    std::memset(&attribute, 0, sizeof(attribute));
    attribute.index = index;
    attribute.qnt_type = RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;
    attribute.scale = 0.25f;
    attribute.zp = 1;
    attribute.fmt = RKNN_TENSOR_NHWC;
    if (input) {
        attribute.n_dims = 4;
        attribute.dims[0] = 1;
        attribute.dims[1] = 2;
        attribute.dims[2] = 3;
        attribute.dims[3] = 2;
        attribute.n_elems = 12;
        attribute.size = 12;
        attribute.w_stride = 4;
        attribute.size_with_stride = 16;
        attribute.type = RKNN_TENSOR_UINT8;
        std::strncpy(attribute.name, "input", sizeof(attribute.name) - 1);
        return;
    }
    if (index == 0) {
        attribute.n_dims = 4;
        attribute.dims[0] = 1;
        attribute.dims[1] = 2;
        attribute.dims[2] = 2;
        attribute.dims[3] = 2;
        attribute.n_elems = 8;
        attribute.size = 8;
        attribute.w_stride = native_output ? 3 : 2;
        attribute.size_with_stride = native_output ? 12 : 8;
        attribute.type = RKNN_TENSOR_INT8;
        std::strncpy(attribute.name, "quantized", sizeof(attribute.name) - 1);
    } else {
        attribute.n_dims = 4;
        attribute.dims[0] = 1;
        attribute.dims[1] = 1;
        attribute.dims[2] = 1;
        attribute.dims[3] = 3;
        attribute.n_elems = 3;
        attribute.size = 3 * sizeof(float);
        attribute.w_stride = 1;
        attribute.size_with_stride = 3 * sizeof(float);
        attribute.type = RKNN_TENSOR_FLOAT32;
        attribute.qnt_type = RKNN_TENSOR_QNT_NONE;
        std::strncpy(attribute.name, "floating", sizeof(attribute.name) - 1);
    }
}

double Percentile(std::vector<double> values, double ratio) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(ratio * static_cast<double>(values.size() - 1));
    return values[index];
}

bool AlmostEqual(float left, float right) {
    return std::fabs(left - right) <= 1e-6f;
}

bool TestWrapperContracts() {
    ResetRuntime();
    bool passed = true;
    std::vector<InputTensorInfo> initialization_inputs;
    std::vector<OutputTensorInfo> outputs = {
      OutputTensorInfo("quantized", TensorInfo::TensorTypeFp32, false),
      OutputTensorInfo("floating", TensorInfo::TensorTypeFp32, false),
    };

    InferenceWrapperRKNNAdapter adapter;
    passed = passed && adapter.Initialize(nullptr, 0, initialization_inputs, outputs) == InferenceWrapper::WrapperError;
    passed = passed && adapter.Initialize("unsupported.rknn", initialization_inputs, outputs) == InferenceWrapper::WrapperError;
    passed = passed && adapter.ResizeInput(initialization_inputs) == InferenceWrapper::WrapperError;
    passed = passed && adapter.PreProcess(initialization_inputs) == InferenceWrapper::WrapperError;
    std::vector<OutputTensorInfo> empty_outputs;
    passed = passed && adapter.Process(empty_outputs) == InferenceWrapper::WrapperError;

    std::array<char, 8> model = {{1, 2, 3, 4, 5, 6, 7, 8}};
    passed = passed && adapter.Initialize(model.data(), static_cast<int>(model.size()), initialization_inputs, outputs) ==
                           InferenceWrapper::WrapperOk;

    std::array<uint8_t, 12> input_pixels = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    InputTensorInfo input("input", TensorInfo::TensorTypeUint8, false);
    input.tensor_dims = {1, 2, 3, 2};
    input.data = input_pixels.data();
    passed = passed && adapter.PreProcess({input}) == InferenceWrapper::WrapperOk;
    passed = passed && runtime.memories.size() == 3;
    if (runtime.memories.size() == 3) {
        const auto *copied = static_cast<const uint8_t *>(runtime.memories[0]->virt_addr);
        passed = passed && std::equal(input_pixels.begin(), input_pixels.begin() + 6, copied);
        passed = passed && copied[6] == 0 && copied[7] == 0;
        passed = passed && std::equal(input_pixels.begin() + 6, input_pixels.end(), copied + 8);
        passed = passed && copied[14] == 0 && copied[15] == 0;
    }

    passed = passed && adapter.Process(outputs) == InferenceWrapper::WrapperOk;
    const std::vector<float> expected_quantized = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    const std::vector<float> expected_floating = {0.25f, -1.5f, 2.0f};
    passed = passed && outputs[0].tensor_dims == std::vector<int32_t>({1, 2, 2, 2});
    passed = passed && outputs[1].tensor_dims == std::vector<int32_t>({1, 1, 1, 3});
    for (size_t index = 0; index < expected_quantized.size() && outputs[0].data != nullptr; ++index) {
        passed = passed && AlmostEqual(static_cast<float *>(outputs[0].data)[index], expected_quantized[index]);
    }
    for (size_t index = 0; index < expected_floating.size() && outputs[1].data != nullptr; ++index) {
        passed = passed && AlmostEqual(static_cast<float *>(outputs[1].data)[index], expected_floating[index]);
    }

    std::vector<double> latency_us;
    latency_us.reserve(256);
    for (int iteration = 0; iteration < 256; ++iteration) {
        const auto begin = Clock::now();
        if (adapter.Process(outputs) != InferenceWrapper::WrapperOk) {
            passed = false;
            break;
        }
        const auto end = Clock::now();
        latency_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        passed = passed && AlmostEqual(static_cast<float *>(outputs[0].data)[7], 7.0f);
        passed = passed && AlmostEqual(static_cast<float *>(outputs[1].data)[1], -1.5f);
    }
    const double mean_us = latency_us.empty()
                             ? 0.0
                             : std::accumulate(latency_us.begin(), latency_us.end(), 0.0) / latency_us.size();
    const double p95_us = Percentile(latency_us, 0.95);
    passed = passed && p95_us < 1000.0;

    std::vector<OutputTensorInfo> mismatched_outputs = {
      OutputTensorInfo("quantized", TensorInfo::TensorTypeFp32, false),
    };
    passed = passed && adapter.Process(mismatched_outputs) == InferenceWrapper::WrapperError;
    passed = passed && mismatched_outputs[0].data == nullptr;
    passed = passed && adapter.Finalize() == InferenceWrapper::WrapperOk;
    passed = passed && adapter.Finalize() == InferenceWrapper::WrapperOk;
    passed = passed && runtime.destroy_memory_calls == 3 && runtime.destroy_calls == 1 && runtime.memories.empty();

    std::cout << "RKNN2_ADAPTER_ACCURACY,mean_us=" << mean_us << ",p95_us=" << p95_us
              << ",runs=" << latency_us.size() << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestFailureCleanup() {
    ResetRuntime();
    runtime.fail_create_call = 2;
    std::array<char, 4> model = {{1, 2, 3, 4}};
    std::vector<InputTensorInfo> inputs;
    std::vector<OutputTensorInfo> outputs = {
      OutputTensorInfo("quantized", TensorInfo::TensorTypeFp32, false),
      OutputTensorInfo("floating", TensorInfo::TensorTypeFp32, false),
    };
    InferenceWrapperRKNNAdapter adapter;
    const bool initialization_failed =
      adapter.Initialize(model.data(), static_cast<int>(model.size()), inputs, outputs) == InferenceWrapper::WrapperError;
    const bool cleaned = runtime.memories.empty() && runtime.destroy_memory_calls == 1 && runtime.destroy_calls == 1;
    const bool idempotent = adapter.Finalize() == InferenceWrapper::WrapperOk && runtime.destroy_calls == 1;
    const bool passed = initialization_failed && cleaned && idempotent;
    std::cout << "RKNN2_ADAPTER_FAILURE_CLEANUP,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestNumericHelpers() {
    bool passed = softmax({}).empty();
    const std::vector<float> probabilities = softmax({1000.0f, 1001.0f, 1002.0f});
    passed = passed && probabilities.size() == 3;
    passed = passed && AlmostEqual(std::accumulate(probabilities.begin(), probabilities.end(), 0.0f), 1.0f);

    const int dimensions[5] = {2, 2, 1, 2, 2};
    const std::array<int8_t, 16> source = {{1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18}};
    std::array<float, 12> destination = {};
    NC1HWC2_int8_to_NCHW_float(source.data(), destination.data(), const_cast<int *>(dimensions), 3, 1, 2, 0, 1.0f);
    const std::array<float, 12> expected = {{1, 3, 2, 4, 5, 7, 11, 13, 12, 14, 15, 17}};
    passed = passed && std::equal(destination.begin(), destination.end(), expected.begin());
    std::cout << "RKNN2_ADAPTER_NUMERIC_HELPERS,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

extern "C" {

int rknn_init(rknn_context *context, void *model, uint32_t size, uint32_t, rknn_init_extend *) {
    ++runtime.init_calls;
    if (context == nullptr || model == nullptr || size == 0) {
        return -1;
    }
    *context = 1;
    return RKNN_SUCC;
}

int rknn_destroy(rknn_context) {
    ++runtime.destroy_calls;
    return RKNN_SUCC;
}

int rknn_query(rknn_context, rknn_query_cmd command, void *information, uint32_t) {
    if (command == runtime.fail_query || information == nullptr) {
        return -1;
    }
    switch (command) {
        case RKNN_QUERY_SDK_VERSION: {
            auto *version = static_cast<rknn_sdk_version *>(information);
            std::strncpy(version->api_version, "mock-api", sizeof(version->api_version) - 1);
            std::strncpy(version->drv_version, "mock-driver", sizeof(version->drv_version) - 1);
            return RKNN_SUCC;
        }
        case RKNN_QUERY_IN_OUT_NUM: {
            auto *count = static_cast<rknn_input_output_num *>(information);
            count->n_input = 1;
            count->n_output = 2;
            return RKNN_SUCC;
        }
        case RKNN_QUERY_INPUT_ATTR:
            FillAttribute(*static_cast<rknn_tensor_attr *>(information), true, false);
            return RKNN_SUCC;
        case RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR:
            FillAttribute(*static_cast<rknn_tensor_attr *>(information), false, true);
            return RKNN_SUCC;
        case RKNN_QUERY_OUTPUT_ATTR:
            FillAttribute(*static_cast<rknn_tensor_attr *>(information), false, false);
            return RKNN_SUCC;
        case RKNN_QUERY_CUSTOM_STRING:
            std::strncpy(static_cast<rknn_custom_string *>(information)->string, "mock", 4);
            return RKNN_SUCC;
        default:
            return -1;
    }
}

rknn_tensor_mem *rknn_create_mem(rknn_context, uint32_t size) {
    ++runtime.create_calls;
    if (runtime.create_calls == runtime.fail_create_call || size == 0) {
        return nullptr;
    }
    auto *memory = static_cast<rknn_tensor_mem *>(std::calloc(1, sizeof(rknn_tensor_mem)));
    if (memory == nullptr) {
        return nullptr;
    }
    memory->virt_addr = std::calloc(1, size);
    if (memory->virt_addr == nullptr) {
        std::free(memory);
        return nullptr;
    }
    memory->size = size;
    runtime.memories.push_back(memory);
    return memory;
}

int rknn_destroy_mem(rknn_context, rknn_tensor_mem *memory) {
    if (memory == nullptr) {
        return -1;
    }
    ++runtime.destroy_memory_calls;
    const auto position = std::find(runtime.memories.begin(), runtime.memories.end(), memory);
    if (position == runtime.memories.end()) {
        return -1;
    }
    runtime.memories.erase(position);
    std::free(memory->virt_addr);
    std::free(memory);
    return RKNN_SUCC;
}

int rknn_set_io_mem(rknn_context, rknn_tensor_mem *memory, rknn_tensor_attr *) {
    ++runtime.set_io_calls;
    return memory == nullptr ? -1 : RKNN_SUCC;
}

int rknn_run(rknn_context, rknn_run_extend *) {
    ++runtime.run_calls;
    if (runtime.memories.size() != 3) {
        return -1;
    }
    const std::array<int8_t, 12> quantized = {{1, 5, 9, 13, 99, 99, 17, 21, 25, 29, 99, 99}};
    std::memcpy(runtime.memories[1]->virt_addr, quantized.data(), quantized.size());
    const std::array<float, 3> floating = {{0.25f, -1.5f, 2.0f}};
    std::memcpy(runtime.memories[2]->virt_addr, floating.data(), sizeof(floating));
    return RKNN_SUCC;
}

}  // extern "C"

int main() {
    const bool passed = TestWrapperContracts() && TestFailureCleanup() && TestNumericHelpers();
    return passed ? 0 : 1;
}
