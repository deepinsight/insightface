#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "middleware/inference_wrapper/inference_wrapper_rknn_adapter.h"

namespace {

using Clock = std::chrono::steady_clock;

struct FakeRuntimeState {
    int init_calls{0};
    int destroy_calls{0};
    int input_calls{0};
    int run_calls{0};
    int get_output_calls{0};
    int release_output_calls{0};
    bool input_contract_exact{true};
    bool output_float_requested{true};
    rknn_query_cmd fail_query{RKNN_QUERY_CMD_MAX};
    std::array<float, 2> output_a{};
    std::array<float, 3> output_b{};
} runtime;

void ResetRuntime() {
    runtime = {};
}

void FillAttribute(rknn_tensor_attr &attribute, bool input) {
    const uint32_t index = attribute.index;
    std::memset(&attribute, 0, sizeof(attribute));
    attribute.index = index;
    if (input) {
        attribute.n_dims = 4;
        attribute.dims[0] = 1;
        attribute.dims[1] = 3;
        attribute.dims[2] = 2;
        attribute.dims[3] = 2;
        attribute.n_elems = 12;
        attribute.size = 12 * sizeof(float);
        attribute.fmt = RKNN_TENSOR_NCHW;
        attribute.type = RKNN_TENSOR_FLOAT32;
        std::strncpy(attribute.name, "input", sizeof(attribute.name) - 1);
        return;
    }
    attribute.n_dims = 2;
    attribute.dims[0] = 1;
    attribute.dims[1] = index == 0 ? 2 : 3;
    attribute.n_elems = attribute.dims[1];
    attribute.size = attribute.n_elems * sizeof(float);
    attribute.fmt = RKNN_TENSOR_NCHW;
    attribute.type = RKNN_TENSOR_FLOAT32;
    attribute.qnt_type = RKNN_TENSOR_QNT_NONE;
    attribute.scale = 1.0f;
    attribute.zp = 0;
    std::strncpy(attribute.name, index == 0 ? "first" : "second", sizeof(attribute.name) - 1);
}

double Percentile(std::vector<double> values, double ratio) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(ratio * static_cast<double>(values.size() - 1))];
}

bool ExactOutputs(const std::vector<OutputTensorInfo> &outputs) {
    if (outputs.size() != 2 || outputs[0].data == nullptr || outputs[1].data == nullptr ||
        outputs[0].tensor_dims != std::vector<int32_t>({1, 2}) ||
        outputs[1].tensor_dims != std::vector<int32_t>({1, 3})) {
        return false;
    }
    const auto *first = static_cast<const float *>(outputs[0].data);
    const auto *second = static_cast<const float *>(outputs[1].data);
    return first[0] == 0.25f && first[1] == -0.5f && second[0] == 1.0f && second[1] == 2.0f && second[2] == 3.0f;
}

bool TestAccuracyLifecycleAndLatency() {
    ResetRuntime();
    bool passed = true;
    InferenceWrapperRKNNAdapter adapter;
    std::vector<InputTensorInfo> initialization_inputs;
    std::vector<OutputTensorInfo> outputs = {
      OutputTensorInfo("first", TensorInfo::TensorTypeFp32),
      OutputTensorInfo("second", TensorInfo::TensorTypeFp32),
    };
    std::vector<OutputTensorInfo> no_outputs;
    std::array<char, 8> model = {{1, 2, 3, 4, 5, 6, 7, 8}};
    passed = passed && adapter.Initialize(nullptr, 0, initialization_inputs, outputs) == InferenceWrapper::WrapperError;
    passed = passed && adapter.Initialize(model.data(), static_cast<int>(model.size()), initialization_inputs, no_outputs) ==
                           InferenceWrapper::WrapperError;
    passed = passed && adapter.Initialize("unsupported.rknn", initialization_inputs, outputs) == InferenceWrapper::WrapperError;
    passed = passed && adapter.Initialize(model.data(), static_cast<int>(model.size()), initialization_inputs, outputs) ==
                           InferenceWrapper::WrapperOk;

    std::array<float, 12> input_data = {};
    std::iota(input_data.begin(), input_data.end(), 0.0f);
    InputTensorInfo input("input", TensorInfo::TensorTypeFp32, true);
    input.tensor_dims = {1, 3, 2, 2};
    input.data = input_data.data();
    passed = passed && adapter.PreProcess({input}) == InferenceWrapper::WrapperOk;
    passed = passed && adapter.Process(outputs) == InferenceWrapper::WrapperOk;
    passed = passed && ExactOutputs(outputs) && runtime.input_contract_exact && runtime.release_output_calls == 0;

    std::vector<double> samples_us;
    samples_us.reserve(256);
    for (int iteration = 0; iteration < 256; ++iteration) {
        const auto begin = Clock::now();
        if (adapter.Process(outputs) != InferenceWrapper::WrapperOk || !ExactOutputs(outputs)) {
            passed = false;
            break;
        }
        samples_us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - begin).count());
    }
    const double mean_us = samples_us.empty() ? 0.0 : std::accumulate(samples_us.begin(), samples_us.end(), 0.0) / samples_us.size();
    const double p95_us = Percentile(samples_us, 0.95);
    passed = passed && p95_us < 1000.0;

    std::vector<OutputTensorInfo> mismatched = {OutputTensorInfo("first", TensorInfo::TensorTypeFp32)};
    passed = passed && adapter.Process(mismatched) == InferenceWrapper::WrapperError;
    passed = passed && mismatched[0].data == nullptr;
    const int releases_after_mismatch = runtime.release_output_calls;
    passed = passed && adapter.Finalize() == InferenceWrapper::WrapperOk;
    passed = passed && adapter.Finalize() == InferenceWrapper::WrapperOk;
    passed = passed && runtime.destroy_calls == 1 && runtime.release_output_calls == releases_after_mismatch;

    std::cout << "RKNN1_ADAPTER_ACCURACY,mean_us=" << mean_us << ",p95_us=" << p95_us
              << ",runs=" << samples_us.size() << ",input_bytes=48,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestFailureCleanupAndBoundaries() {
    ResetRuntime();
    runtime.fail_query = RKNN_QUERY_INPUT_ATTR;
    std::array<char, 4> model = {{1, 2, 3, 4}};
    std::vector<InputTensorInfo> inputs;
    std::vector<OutputTensorInfo> outputs = {
      OutputTensorInfo("first", TensorInfo::TensorTypeFp32),
      OutputTensorInfo("second", TensorInfo::TensorTypeFp32),
    };
    InferenceWrapperRKNNAdapter adapter;
    bool passed = adapter.Initialize(model.data(), static_cast<int>(model.size()), inputs, outputs) == InferenceWrapper::WrapperError;
    passed = passed && runtime.destroy_calls == 1;
    passed = passed && adapter.Finalize() == InferenceWrapper::WrapperOk && runtime.destroy_calls == 1;

    ResetRuntime();
    RKNNAdapter raw_adapter;
    passed = passed && raw_adapter.Initialize(reinterpret_cast<const unsigned char *>(model.data()), model.size()) == RKNN_SUCC;
    std::array<uint8_t, 4> data = {{1, 2, 3, 4}};
    passed = passed && raw_adapter.SetInputData(-1, data.data(), 1, 1, 1) == ERROR_INVALID_INPUT;
    passed = passed && raw_adapter.SetInputData(1, data.data(), 1, 1, 1) == ERROR_INVALID_INPUT;
    passed = passed && raw_adapter.SetInputData(0, nullptr, 1, 1, 1) == ERROR_INVALID_INPUT;
    passed = passed && raw_adapter.GetOutputFlow(-1) == nullptr && raw_adapter.GetOutputTensorSize(99).empty();
    raw_adapter.Release();
    passed = passed && runtime.destroy_calls == 1;

    std::cout << "RKNN1_ADAPTER_FAILURE_CLEANUP,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

extern "C" {

int rknn_init(rknn_context *context, void *model, uint32_t size, uint32_t) {
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
            FillAttribute(*static_cast<rknn_tensor_attr *>(information), true);
            return RKNN_SUCC;
        case RKNN_QUERY_OUTPUT_ATTR:
            FillAttribute(*static_cast<rknn_tensor_attr *>(information), false);
            return RKNN_SUCC;
        default:
            return -1;
    }
}

int rknn_inputs_set(rknn_context, uint32_t count, rknn_input inputs[]) {
    ++runtime.input_calls;
    if (count != 1 || inputs == nullptr) {
        runtime.input_contract_exact = false;
        return -1;
    }
    runtime.input_contract_exact = runtime.input_contract_exact && inputs[0].index == 0 && inputs[0].buf != nullptr &&
                                   inputs[0].size == 48 && inputs[0].type == RKNN_TENSOR_FLOAT32 &&
                                   inputs[0].fmt == RKNN_TENSOR_NCHW;
    return runtime.input_contract_exact ? RKNN_SUCC : -1;
}

int rknn_run(rknn_context, rknn_run_extend *) {
    ++runtime.run_calls;
    return RKNN_SUCC;
}

int rknn_outputs_get(rknn_context, uint32_t count, rknn_output outputs[], rknn_output_extend *) {
    ++runtime.get_output_calls;
    if (count != 2 || outputs == nullptr) {
        return -1;
    }
    runtime.output_float_requested = runtime.output_float_requested && outputs[0].want_float == 1 && outputs[1].want_float == 1;
    runtime.output_a = {{0.25f, -0.5f}};
    runtime.output_b = {{1.0f, 2.0f, 3.0f}};
    outputs[0].buf = runtime.output_a.data();
    outputs[0].size = sizeof(runtime.output_a);
    outputs[1].buf = runtime.output_b.data();
    outputs[1].size = sizeof(runtime.output_b);
    return runtime.output_float_requested ? RKNN_SUCC : -1;
}

int rknn_outputs_release(rknn_context, uint32_t count, rknn_output outputs[]) {
    ++runtime.release_output_calls;
    if (count != 2 || outputs == nullptr) {
        return -1;
    }
    runtime.output_a.fill(-999.0f);
    runtime.output_b.fill(-999.0f);
    outputs[0].buf = nullptr;
    outputs[1].buf = nullptr;
    return RKNN_SUCC;
}

}  // extern "C"

int main() {
    const bool passed = TestAccuracyLifecycleAndLatency() && TestFailureCleanupAndBoundaries();
    return passed ? 0 : 1;
}
