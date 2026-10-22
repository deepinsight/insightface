#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "mock_tensorrt/NvInfer.h"
#include "../../inspireface/middleware/inference_wrapper/tensorrt/tensorrt_adapter.h"

namespace fake_tensorrt {

bool dynamic_shapes = false;
bool fail_runtime = false;
bool fail_engine = false;
bool fail_context = false;
bool fail_enqueue = false;

}  // namespace fake_tensorrt

namespace {

bool fail_set_device = false;
bool fail_malloc = false;
bool fail_memcpy = false;
bool fail_synchronize = false;
int next_stream_id = 1;
int stream_create_count = 0;
int stream_destroy_count = 0;
int stream_synchronize_count = 0;
int allocation_count = 0;
int free_count = 0;
cudaStream_t last_operation_stream = nullptr;
std::set<void *> allocations;

void ResetFailures() {
    fail_set_device = false;
    fail_malloc = false;
    fail_memcpy = false;
    fail_synchronize = false;
    fake_tensorrt::dynamic_shapes = false;
    fake_tensorrt::fail_runtime = false;
    fake_tensorrt::fail_engine = false;
    fake_tensorrt::fail_context = false;
    fake_tensorrt::fail_enqueue = false;
}

void Require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void RequireOutput(const float *actual, const std::vector<float> &input) {
    Require(actual != nullptr, "output pointer must not be null");
    for (size_t index = 0; index < input.size(); ++index) {
        const float expected = input[index] * 2.0f + 1.0f;
        Require(actual[index] == expected, "output values must remain exact");
    }
}

void TestInitializationFailures() {
    char model = 1;
    Require(TensorRTAdapter().readFromBin(nullptr, 1) == TENSORRT_HFAIL, "null model data must fail");

    fail_set_device = true;
    TensorRTAdapter adapter;
    Require(adapter.readFromBin(&model, 1) == TENSORRT_HFAIL, "cudaSetDevice failure must propagate");
    fail_set_device = false;

    fake_tensorrt::fail_runtime = true;
    Require(adapter.readFromBin(&model, 1) == TENSORRT_HFAIL, "runtime creation failure must propagate");
    fake_tensorrt::fail_runtime = false;

    fake_tensorrt::fail_engine = true;
    Require(adapter.readFromBin(&model, 1) == TENSORRT_HFAIL, "engine creation failure must propagate");
    fake_tensorrt::fail_engine = false;

    fake_tensorrt::fail_context = true;
    Require(adapter.readFromBin(&model, 1) == TENSORRT_HFAIL, "context creation failure must propagate");
    fake_tensorrt::fail_context = false;

    fail_malloc = true;
    Require(adapter.readFromBin(&model, 1) == TENSORRT_HFAIL, "device allocation failure must propagate");
    fail_malloc = false;
}

void TestOwnedStreamAndErrors() {
    char model = 1;
    const int destroys_before = stream_destroy_count;
    const int syncs_before = stream_synchronize_count;
    {
        TensorRTAdapter adapter;
        Require(adapter.readFromBin(&model, 1) == TENSORRT_HSUCCEED, "static engine must initialize");
        std::vector<float> input{-2.0f, -0.25f, 0.0f, 3.5f};

        Require(adapter.setInput("input", input.data()) == TENSORRT_HSUCCEED, "input copy must succeed");
        Require(stream_synchronize_count == syncs_before, "setInput must not add a per-input synchronization");
        Require(adapter.forward() == TENSORRT_HSUCCEED, "forward must succeed");
        RequireOutput(static_cast<const float *>(adapter.getOutput("output")), input);
        Require(adapter.getOutputShapeByName("output") == std::vector<int>({1, 1, 1, 4}), "output shape must be current");

        std::vector<float> converted = adapter.getOutputAsFloat("output");
        Require(converted.size() == input.size(), "float output size must match");
        RequireOutput(converted.data(), input);

        fail_memcpy = true;
        Require(adapter.setInput("input", input.data()) == TENSORRT_HFAIL, "input copy failure must propagate");
        fail_memcpy = true;
        Require(adapter.getOutput("output") == nullptr, "output copy failure must propagate");
        fail_memcpy = false;

        fake_tensorrt::fail_enqueue = true;
        Require(adapter.forward() == TENSORRT_FORWARD_FAILED, "enqueue failure must propagate");
        fake_tensorrt::fail_enqueue = false;

        fail_synchronize = true;
        Require(adapter.forward() == TENSORRT_FORWARD_FAILED, "inference synchronization failure must propagate");
        fail_synchronize = false;

        Require(adapter.setInput("missing", input.data()) == TENSORRT_HFAIL, "unknown input must fail");
        Require(adapter.getOutput("missing") == nullptr, "unknown output must fail");
    }
    Require(stream_destroy_count == destroys_before + 1, "owned stream must be destroyed exactly once");
}

void TestBorrowedStreamOwnership() {
    char model = 1;
    cudaStream_t external = nullptr;
    Require(cudaStreamCreate(&external) == cudaSuccess, "external stream creation must succeed");
    const int destroys_before = stream_destroy_count;
    {
        TensorRTAdapter adapter;
        cudaStream_t supplied_handle = external;
        Require(adapter.setCudaStream(&supplied_handle) == TENSORRT_HSUCCEED, "external stream attachment must succeed");
        supplied_handle = nullptr;
        Require(adapter.readFromBin(&model, 1) == TENSORRT_HSUCCEED, "engine with borrowed stream must initialize");

        std::vector<float> input{1.0f, 2.0f, 4.0f, 8.0f};
        Require(adapter.setInput("input", input.data()) == TENSORRT_HSUCCEED, "borrowed stream input must succeed");
        Require(last_operation_stream == external, "adapter must copy the CUDA handle value");
        Require(adapter.forward() == TENSORRT_HSUCCEED, "borrowed stream forward must succeed");
        RequireOutput(static_cast<const float *>(adapter.getOutput("output")), input);
    }
    Require(stream_destroy_count == destroys_before, "borrowed stream must not be destroyed by adapter");
    Require(cudaStreamDestroy(external) == cudaSuccess, "caller must retain ownership of borrowed stream");

    cudaStream_t replacement = nullptr;
    Require(cudaStreamCreate(&replacement) == cudaSuccess, "replacement stream creation must succeed");
    const int switch_destroys_before = stream_destroy_count;
    {
        TensorRTAdapter adapter;
        Require(adapter.readFromBin(&model, 1) == TENSORRT_HSUCCEED, "owned stream engine must initialize");
        Require(adapter.setCudaStream(&replacement) == TENSORRT_HSUCCEED, "owned-to-borrowed switch must succeed");
        Require(stream_destroy_count == switch_destroys_before + 1, "switch must release only the owned stream");
    }
    Require(stream_destroy_count == switch_destroys_before + 1, "replacement borrowed stream must survive adapter destruction");
    Require(cudaStreamDestroy(replacement) == cudaSuccess, "caller must destroy replacement stream");
}

void TestDynamicBatchAndBufferResize() {
    char model = 1;
    fake_tensorrt::dynamic_shapes = true;
    TensorRTAdapter adapter;
    Require(adapter.readFromBin(&model, 1) == TENSORRT_HSUCCEED, "dynamic engine must defer unresolved allocations");
    Require(adapter.setBatchSize(0) == TENSORRT_HFAIL, "zero batch must fail");

    for (int batch : {2, 3}) {
        Require(adapter.setBatchSize(batch) == TENSORRT_HSUCCEED, "dynamic batch resize must succeed");
        std::vector<float> input(static_cast<size_t>(batch * 4));
        std::iota(input.begin(), input.end(), static_cast<float>(-batch));
        Require(adapter.setInput("input", input.data()) == TENSORRT_HSUCCEED, "dynamic input copy must succeed");
        Require(adapter.forward() == TENSORRT_HSUCCEED, "dynamic forward must succeed");
        RequireOutput(static_cast<const float *>(adapter.getOutput("output")), input);
        Require(adapter.getOutputShapeByName("output")[0] == batch, "dynamic output shape must update");
    }
    fake_tensorrt::dynamic_shapes = false;
}

void RunPerformanceGate() {
    char model = 1;
    TensorRTAdapter adapter;
    Require(adapter.readFromBin(&model, 1) == TENSORRT_HSUCCEED, "performance engine must initialize");
    std::vector<float> input{0.125f, -1.5f, 2.25f, 9.0f};
    constexpr int iterations = 20000;
    std::vector<double> samples;
    samples.reserve(iterations);
    const int syncs_before = stream_synchronize_count;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        input[0] = static_cast<float>(iteration % 31) / 7.0f;
        auto start = std::chrono::steady_clock::now();
        Require(adapter.setInput("input", input.data()) == TENSORRT_HSUCCEED, "performance input must succeed");
        Require(adapter.forward() == TENSORRT_HSUCCEED, "performance forward must succeed");
        RequireOutput(static_cast<const float *>(adapter.getOutput("output")), input);
        auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(samples.begin(), samples.end());
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double p50 = samples[samples.size() / 2];
    const double p95 = samples[samples.size() * 95 / 100];
    const int syncs = stream_synchronize_count - syncs_before;
    Require(syncs == iterations * 2, "each inference must use two syncs instead of the previous three");
    std::cout << "TensorRT lifecycle guard: iterations=" << iterations << " mean_us=" << mean << " p50_us=" << p50
              << " p95_us=" << p95 << " syncs_per_iteration=" << static_cast<double>(syncs) / iterations << std::endl;
}

}  // namespace

cudaError_t cudaSetDevice(int) {
    return fail_set_device ? cudaErrorUnknown : cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    if (!stream) {
        return cudaErrorUnknown;
    }
    *stream = new InspireFaceFakeCudaStream{next_stream_id++};
    ++stream_create_count;
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (!stream) {
        return cudaErrorUnknown;
    }
    delete stream;
    ++stream_destroy_count;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    last_operation_stream = stream;
    ++stream_synchronize_count;
    if (fail_synchronize) {
        fail_synchronize = false;
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

cudaError_t cudaMalloc(void **pointer, size_t size) {
    if (!pointer || size == 0 || fail_malloc) {
        fail_malloc = false;
        return cudaErrorUnknown;
    }
    *pointer = std::malloc(size);
    if (!*pointer) {
        return cudaErrorUnknown;
    }
    allocations.insert(*pointer);
    ++allocation_count;
    return cudaSuccess;
}

cudaError_t cudaFree(void *pointer) {
    if (!pointer || allocations.erase(pointer) != 1) {
        return cudaErrorUnknown;
    }
    std::free(pointer);
    ++free_count;
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *destination, const void *source, size_t size, cudaMemcpyKind, cudaStream_t stream) {
    last_operation_stream = stream;
    if (!destination || !source || fail_memcpy) {
        fail_memcpy = false;
        return cudaErrorUnknown;
    }
    std::memcpy(destination, source, size);
    return cudaSuccess;
}

const char *cudaGetErrorString(cudaError_t error) {
    return error == cudaSuccess ? "success" : "injected failure";
}

int main() {
    ResetFailures();
    TestInitializationFailures();
    TestOwnedStreamAndErrors();
    TestBorrowedStreamOwnership();
    TestDynamicBatchAndBufferResize();
    RunPerformanceGate();
    Require(allocations.empty(), "all fake device allocations must be released");
    Require(allocation_count == free_count, "device allocation/free counts must match");
    Require(stream_create_count == stream_destroy_count, "stream create/destroy counts must match after caller cleanup");
    std::cout << "TensorRT ownership, error, dynamic-shape, exact-output, and performance gates: PASS" << std::endl;
    return 0;
}
