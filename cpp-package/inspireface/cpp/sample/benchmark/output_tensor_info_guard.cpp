#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "middleware/inference_wrapper/inference_wrapper.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

void HashBytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

double Percentile(std::vector<double> samples, double percentile) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    const double position = percentile * static_cast<double>(samples.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, samples.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return samples[lower] * (1.0 - fraction) + samples[upper] * fraction;
}

template <typename Source>
bool CheckQuantizedValues(const Source* source, const float* converted, size_t size, float scale, int32_t zero_point) {
    if (source == nullptr || converted == nullptr) {
        return false;
    }
    for (size_t index = 0; index < size; ++index) {
        const float expected = (static_cast<int32_t>(source[index]) - zero_point) * scale;
        if (converted[index] != expected) {
            return false;
        }
    }
    return true;
}

bool TestSafePaths(int iterations) {
    std::vector<uint8_t> uint8_data(512);
    for (size_t index = 0; index < uint8_data.size(); ++index) {
        uint8_data[index] = static_cast<uint8_t>((index * 37 + 11) % 256);
    }
    OutputTensorInfo uint8_tensor("uint8", TensorInfo::TensorTypeUint8);
    uint8_tensor.tensor_dims = {1, 1, 1, static_cast<int32_t>(uint8_data.size())};
    uint8_tensor.quant.scale = 0.03125f;
    uint8_tensor.quant.zero_point = 117;
    uint8_tensor.data = uint8_data.data();
    float* first = uint8_tensor.GetDataAsFloat();
    const bool uint8_exact = CheckQuantizedValues(uint8_data.data(), first, uint8_data.size(),
                                                  uint8_tensor.quant.scale, uint8_tensor.quant.zero_point);

    bool pointer_stable = first != nullptr;
    std::vector<double> samples_us;
    samples_us.reserve(static_cast<size_t>(iterations));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        uint8_data[static_cast<size_t>(iteration) % uint8_data.size()] ^= static_cast<uint8_t>(iteration + 1);
        const auto begin = Clock::now();
        float* current = uint8_tensor.GetDataAsFloat();
        const auto end = Clock::now();
        pointer_stable = pointer_stable && current == first;
        samples_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }
    float* uint8_output = uint8_tensor.GetDataAsFloat();
    const bool uint8_repeat_exact = CheckQuantizedValues(uint8_data.data(), uint8_output, uint8_data.size(),
                                                         uint8_tensor.quant.scale, uint8_tensor.quant.zero_point);

    std::vector<int8_t> int8_data(257);
    for (size_t index = 0; index < int8_data.size(); ++index) {
        int8_data[index] = static_cast<int8_t>(static_cast<int>(index % 255) - 127);
    }
    OutputTensorInfo int8_tensor("int8", TensorInfo::TensorTypeInt8);
    int8_tensor.tensor_dims = {1, 1, 1, static_cast<int32_t>(int8_data.size())};
    int8_tensor.quant.scale = 0.0625f;
    int8_tensor.quant.zero_point = -3;
    int8_tensor.data = int8_data.data();
    float* int8_output = int8_tensor.GetDataAsFloat();
    const bool int8_exact = CheckQuantizedValues(int8_data.data(), int8_output, int8_data.size(),
                                                 int8_tensor.quant.scale, int8_tensor.quant.zero_point);

    std::vector<float> fp32_data = {-7.5f, -0.0f, 1.25f, 19.0f};
    OutputTensorInfo fp32_tensor("fp32", TensorInfo::TensorTypeFp32);
    fp32_tensor.tensor_dims = {1, 1, 1, static_cast<int32_t>(fp32_data.size())};
    fp32_tensor.data = fp32_data.data();
    const bool fp32_passthrough = fp32_tensor.GetDataAsFloat() == fp32_data.data();

    uint64_t digest = kFnvOffset;
    HashBytes(digest, uint8_output, uint8_data.size() * sizeof(float));
    HashBytes(digest, int8_output, int8_data.size() * sizeof(float));
    HashBytes(digest, fp32_tensor.GetDataAsFloat(), fp32_data.size() * sizeof(float));
    const double mean_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) /
                           static_cast<double>(samples_us.size());
    const double p50_us = Percentile(samples_us, 0.50);
    const double p95_us = Percentile(samples_us, 0.95);
    const bool passed = uint8_exact && uint8_repeat_exact && int8_exact && fp32_passthrough && pointer_stable;
    std::cout << std::fixed << std::setprecision(6)
              << "OUTPUT_TENSOR_SAFE,digest=0x" << std::hex << digest << std::dec
              << ",iterations=" << iterations << ",mean_us=" << mean_us << ",p50_us=" << p50_us
              << ",p95_us=" << p95_us << ",uint8=" << (uint8_repeat_exact ? "PASS" : "FAIL")
              << ",int8=" << (int8_exact ? "PASS" : "FAIL")
              << ",fp32_passthrough=" << (fp32_passthrough ? "PASS" : "FAIL")
              << ",pointer_stability=" << (pointer_stable ? "PASS" : "FAIL")
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestGrowAndShrink() {
    std::vector<uint8_t> source(8, 3);
    OutputTensorInfo tensor("dynamic", TensorInfo::TensorTypeUint8);
    tensor.quant.scale = 0.25f;
    tensor.quant.zero_point = 2;
    tensor.tensor_dims = {1, 1, 1, static_cast<int32_t>(source.size())};
    tensor.data = source.data();
    float* small = tensor.GetDataAsFloat();
    if (!CheckQuantizedValues(source.data(), small, source.size(), tensor.quant.scale, tensor.quant.zero_point)) {
        return false;
    }

    source.resize(4096);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint8_t>((index * 13) % 256);
    }
    tensor.tensor_dims.back() = static_cast<int32_t>(source.size());
    tensor.data = source.data();
    float* large = tensor.GetDataAsFloat();
    const bool grow_exact = CheckQuantizedValues(source.data(), large, source.size(), tensor.quant.scale,
                                                 tensor.quant.zero_point);

    source.resize(17);
    tensor.tensor_dims.back() = static_cast<int32_t>(source.size());
    tensor.data = source.data();
    float* shrunk = tensor.GetDataAsFloat();
    const bool shrink_exact = CheckQuantizedValues(source.data(), shrunk, source.size(), tensor.quant.scale,
                                                   tensor.quant.zero_point);
    uint64_t digest = kFnvOffset;
    HashBytes(digest, shrunk, source.size() * sizeof(float));
    const bool passed = grow_exact && shrink_exact;
    std::cout << "OUTPUT_TENSOR_DYNAMIC,digest=0x" << std::hex << digest << std::dec
              << ",grow=" << (grow_exact ? "PASS" : "FAIL")
              << ",shrink=" << (shrink_exact ? "PASS" : "FAIL")
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

OutputTensorInfo MakeConvertedTensor(std::vector<int8_t>& source) {
    OutputTensorInfo tensor("owned-cache", TensorInfo::TensorTypeInt8);
    tensor.tensor_dims = {1, 1, 1, static_cast<int32_t>(source.size())};
    tensor.quant.scale = 0.125f;
    tensor.quant.zero_point = -7;
    tensor.data = source.data();
    tensor.GetDataAsFloat();
    return tensor;
}

bool TestCopy() {
    std::vector<int8_t> source(64, 11);
    OutputTensorInfo original = MakeConvertedTensor(source);
    OutputTensorInfo copied(original);
    float* original_data = original.GetDataAsFloat();
    float* copied_data = copied.GetDataAsFloat();
    const bool passed = original_data != nullptr && copied_data != nullptr && original_data != copied_data &&
                        CheckQuantizedValues(source.data(), original_data, source.size(), original.quant.scale,
                                             original.quant.zero_point) &&
                        CheckQuantizedValues(source.data(), copied_data, source.size(), copied.quant.scale,
                                             copied.quant.zero_point);
    std::cout << "OUTPUT_TENSOR_COPY,independent=" << (original_data != copied_data ? "PASS" : "FAIL")
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestMove() {
    std::vector<int8_t> source(64, -9);
    OutputTensorInfo original = MakeConvertedTensor(source);
    OutputTensorInfo moved(std::move(original));
    float* moved_data = moved.GetDataAsFloat();
    const bool passed = CheckQuantizedValues(source.data(), moved_data, source.size(), moved.quant.scale,
                                             moved.quant.zero_point);
    std::cout << "OUTPUT_TENSOR_MOVE,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestVectorRelocation() {
    std::vector<int8_t> source(64, 23);
    std::vector<OutputTensorInfo> tensors;
    tensors.reserve(1);
    tensors.push_back(MakeConvertedTensor(source));
    for (int index = 0; index < 32; ++index) {
        tensors.emplace_back("filler", TensorInfo::TensorTypeFp32);
    }
    float* converted = tensors.front().GetDataAsFloat();
    const bool passed = CheckQuantizedValues(source.data(), converted, source.size(), tensors.front().quant.scale,
                                             tensors.front().quant.zero_point);
    std::cout << "OUTPUT_TENSOR_VECTOR_RELOCATION,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestInvalidMetadata() {
    std::vector<uint8_t> source(8, 1);
    OutputTensorInfo tensor("invalid", TensorInfo::TensorTypeUint8);
    tensor.data = source.data();
    tensor.tensor_dims = {1, 1, 1, -1};
    const bool negative_rejected = tensor.GetDataAsFloat() == nullptr;
    tensor.tensor_dims = {1, 1, 1, 0};
    const bool zero_rejected = tensor.GetDataAsFloat() == nullptr;
    tensor.tensor_dims = {std::numeric_limits<int32_t>::max(), 2};
    const bool overflow_rejected = tensor.GetDataAsFloat() == nullptr;
    tensor.tensor_type = TensorInfo::TensorTypeNone;
    tensor.tensor_dims = {1, 1, 1, 8};
    const bool type_rejected = tensor.GetDataAsFloat() == nullptr;
    tensor.tensor_type = TensorInfo::TensorTypeFp32;
    tensor.data = nullptr;
    const bool null_rejected = tensor.GetDataAsFloat() == nullptr;
    const bool passed = negative_rejected && zero_rejected && overflow_rejected && type_rejected && null_rejected;
    std::cout << "OUTPUT_TENSOR_INVALID,negative=" << (negative_rejected ? "PASS" : "FAIL")
              << ",zero=" << (zero_rejected ? "PASS" : "FAIL")
              << ",overflow=" << (overflow_rejected ? "PASS" : "FAIL")
              << ",type=" << (type_rejected ? "PASS" : "FAIL")
              << ",null=" << (null_rejected ? "PASS" : "FAIL")
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 3) {
        std::cerr << "Usage: " << argv[0] << " [safe|grow|copy|move|vector|invalid|full] [iterations]\n";
        return 2;
    }
    const std::string mode = argc >= 2 ? argv[1] : "safe";
    const int iterations = argc >= 3 ? std::max(100, std::atoi(argv[2])) : 20000;
    bool passed = false;
    if (mode == "safe") {
        passed = TestSafePaths(iterations);
    } else if (mode == "grow") {
        passed = TestGrowAndShrink();
    } else if (mode == "copy") {
        passed = TestCopy();
    } else if (mode == "move") {
        passed = TestMove();
    } else if (mode == "vector") {
        passed = TestVectorRelocation();
    } else if (mode == "invalid") {
        passed = TestInvalidMetadata();
    } else if (mode == "full") {
        passed = TestSafePaths(iterations) && TestGrowAndShrink() && TestCopy() && TestMove() &&
                 TestVectorRelocation() && TestInvalidMetadata();
    } else {
        std::cerr << "Unknown mode: " << mode << '\n';
        return 2;
    }
    std::cout << "SUMMARY,mode=" << mode << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
