#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <inspirecv/inspirecv.h>
#include <inspireface.h>
#include <launch.h>

#include "middleware/any_net_adapter.h"
#include "middleware/model_archive/inspire_archive.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
constexpr float kProbabilityTolerance = 2.0e-6f;
constexpr float kSumTolerance = 2.0e-5f;
volatile float g_benchmark_sink = 0.0f;

struct TimingSummary {
    double mean_ns = 0.0;
    double p50_ns = 0.0;
    double p95_ns = 0.0;
};

struct SyntheticCase {
    const char* name;
    std::vector<float> input;
    bool finite_input;
};

struct RealModelCase {
    const char* name;
    int threshold_index;
    float threshold;
    size_t expected_size;
};

struct RealModelResult {
    bool loaded = false;
    bool output_valid = false;
    bool probability_valid = false;
    bool decision_exact = false;
    uint64_t raw_digest = kFnvOffset;
    uint64_t legacy_digest = kFnvOffset;
    uint64_t candidate_digest = kFnvOffset;
    float max_abs_diff = 0.0f;
    double initialization_ms = 0.0;
    double inference_mean_ms = 0.0;
    size_t observed_size = 0;
    std::vector<std::vector<float>> workloads;
};

void HashBytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

template <typename T>
void HashValue(uint64_t& hash, const T& value) {
    HashBytes(hash, &value, sizeof(value));
}

void HashVector(uint64_t& hash, const std::vector<float>& values) {
    HashValue(hash, values.size());
    if (!values.empty()) {
        HashBytes(hash, values.data(), values.size() * sizeof(float));
    }
}

std::string JoinPath(const std::string& root, const std::string& relative) {
    if (root.empty()) {
        return relative;
    }
    if (root.back() == '/' || root.back() == '\\') {
        return root + relative;
    }
    return root + "/" + relative;
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

TimingSummary Summarize(const std::vector<double>& samples) {
    if (samples.empty()) {
        return {};
    }
    return {std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size()),
            Percentile(samples, 0.50), Percentile(samples, 0.95)};
}

std::vector<float> LegacySoftmax(const std::vector<float>& input) {
    std::vector<float> result;
    float sum = 0.0f;
    for (float value : input) {
        const float exponential = std::exp(value);
        result.push_back(exponential);
        sum += exponential;
    }
    for (float& value : result) {
        value /= sum;
    }
    return result;
}

bool ReferenceSoftmax(const std::vector<float>& input, std::vector<float>& output) {
    output.clear();
    if (input.empty()) {
        return true;
    }
    for (float value : input) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    const long double maximum = static_cast<long double>(*std::max_element(input.begin(), input.end()));
    std::vector<long double> exponentials(input.size());
    long double sum = 0.0L;
    for (size_t index = 0; index < input.size(); ++index) {
        exponentials[index] = std::exp(static_cast<long double>(input[index]) - maximum);
        sum += exponentials[index];
    }
    output.resize(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        output[index] = static_cast<float>(exponentials[index] / sum);
    }
    return true;
}

bool ProbabilitiesValid(const std::vector<float>& values) {
    if (values.empty()) {
        return false;
    }
    double sum = 0.0;
    for (float value : values) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            return false;
        }
        sum += value;
    }
    return std::abs(sum - 1.0) <= kSumTolerance;
}

float MaxAbsDifference(const std::vector<float>& left, const std::vector<float>& right) {
    if (left.size() != right.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float maximum = 0.0f;
    for (size_t index = 0; index < left.size(); ++index) {
        if (!std::isfinite(left[index]) || !std::isfinite(right[index])) {
            return std::numeric_limits<float>::infinity();
        }
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

size_t ArgMax(const std::vector<float>& values) {
    return static_cast<size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

bool LoadImages(const std::string& test_root, std::vector<inspirecv::Image>& images) {
    const char* paths[] = {"data/bulk/kun.jpg", "data/bulk/jntm.jpg", "data/bulk/woman.png"};
    images.clear();
    for (const char* path : paths) {
        auto image = inspirecv::Image::Create(JoinPath(test_root, path), 3);
        if (image.Empty()) {
            std::cerr << "Unable to load image: " << path << '\n';
            return false;
        }
        images.push_back(std::move(image));
    }
    return true;
}

bool RunSyntheticGate() {
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const SyntheticCase cases[] = {{"empty", {}, true},
                                   {"single", {42.0f}, true},
                                   {"balanced", {0.0f, 0.0f}, true},
                                   {"ordinary", {-1.0f, 0.0f, 1.0f, 2.0f}, true},
                                   {"large-positive", {1000.0f, 1001.0f}, true},
                                   {"large-negative", {-1000.0f, -1001.0f}, true},
                                   {"float-max", {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()}, true},
                                   {"nan", {0.0f, nan}, false},
                                   {"positive-infinity", {infinity, 0.0f}, false},
                                   {"negative-infinity", {-infinity, -infinity}, false}};

    bool passed = true;
    for (const auto& test_case : cases) {
        std::vector<float> reference;
        const bool reference_valid = ReferenceSoftmax(test_case.input, reference);
        const std::vector<float> candidate = inspire::AnyNetAdapter::Softmax(test_case.input);
        bool case_passed = reference_valid == test_case.finite_input;
        float max_abs_diff = 0.0f;
        if (test_case.finite_input) {
            max_abs_diff = MaxAbsDifference(reference, candidate);
            case_passed = case_passed && candidate.size() == reference.size() &&
                          (candidate.empty() || ProbabilitiesValid(candidate)) && max_abs_diff <= kProbabilityTolerance;
        } else {
            case_passed = case_passed && candidate.empty();
        }
        std::cout << std::fixed << std::setprecision(9) << "SOFTMAX_SYNTHETIC,name=" << test_case.name
                  << ",size=" << candidate.size() << ",max_abs_diff=" << max_abs_diff
                  << ",status=" << (case_passed ? "PASS" : "FAIL") << '\n';
        passed = case_passed && passed;
    }

    const std::vector<float> ordinary = {-1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> translated = {999.0f, 1000.0f, 1001.0f, 1002.0f};
    const auto ordinary_output = inspire::AnyNetAdapter::Softmax(ordinary);
    const auto translated_output = inspire::AnyNetAdapter::Softmax(translated);
    const float translation_diff = MaxAbsDifference(ordinary_output, translated_output);
    const bool translation_passed = translation_diff <= kProbabilityTolerance;
    std::cout << "SOFTMAX_TRANSLATION,max_abs_diff=" << translation_diff
              << ",status=" << (translation_passed ? "PASS" : "FAIL") << '\n';
    return translation_passed && passed;
}

bool DecisionMatches(const RealModelCase& model_case, const std::vector<float>& legacy,
                     const std::vector<float>& candidate) {
    if (legacy.empty() || candidate.empty() || legacy.size() != candidate.size() || ArgMax(legacy) != ArgMax(candidate)) {
        return false;
    }
    if (model_case.threshold_index >= 0) {
        const size_t index = static_cast<size_t>(model_case.threshold_index);
        if (index >= legacy.size()) {
            return false;
        }
        return (legacy[index] > model_case.threshold) == (candidate[index] > model_case.threshold);
    }
    return true;
}

RealModelResult RunRealModel(inspire::InspireArchive& archive, const RealModelCase& model_case,
                             const std::vector<inspirecv::Image>& source_images, int iterations) {
    RealModelResult result;
    inspire::InspireModel model;
    if (archive.LoadModel(model_case.name, model) != inspire::SARC_SUCCESS) {
        return result;
    }
    inspire::AnyNetAdapter net(std::string("SoftmaxGuard/") + model_case.name);
    const auto initialization_begin = Clock::now();
    result.loaded = net.LoadData(model, model.modelType) == InferenceWrapper::WrapperOk;
    const auto initialization_end = Clock::now();
    result.initialization_ms = std::chrono::duration<double, std::milli>(initialization_end - initialization_begin).count();
    if (!result.loaded || net.getMInputTensorInfoList().size() != 1) {
        return result;
    }

    const auto& input = net.getMInputTensorInfoList().front();
    std::vector<inspirecv::Image> images;
    for (const auto& source : source_images) {
        auto resized = source.Resize(input.image_info.width, input.image_info.height);
        if (resized.Empty()) {
            return result;
        }
        images.push_back(std::move(resized));
    }

    std::vector<double> inference_samples;
    bool output_valid = true;
    bool probability_valid = true;
    bool decision_exact = true;
    for (const auto& image : images) {
        inspire::AnyTensorViews warmup;
        if (net.ForwardViews(image, warmup) != InferenceWrapper::WrapperOk) {
            return result;
        }
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (const auto& image : images) {
            inspire::AnyTensorViews views;
            const auto begin = Clock::now();
            const int32_t status = net.ForwardViews(image, views);
            const auto end = Clock::now();
            inference_samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
            if (status != InferenceWrapper::WrapperOk || views.empty() || views[0].data == nullptr || views[0].size == 0) {
                output_valid = false;
                continue;
            }
            std::vector<float> raw(views[0].data, views[0].data + views[0].size);
            if (result.observed_size == 0) {
                result.observed_size = raw.size();
            } else if (result.observed_size != raw.size()) {
                output_valid = false;
            }
            const std::vector<float> legacy = LegacySoftmax(raw);
            const std::vector<float> candidate = inspire::AnyNetAdapter::Softmax(raw);
            if (model_case.expected_size != 0 && raw.size() != model_case.expected_size) {
                output_valid = false;
            }
            const float difference = MaxAbsDifference(legacy, candidate);
            result.max_abs_diff = std::max(result.max_abs_diff, difference);
            probability_valid = ProbabilitiesValid(candidate) && difference <= kProbabilityTolerance && probability_valid;
            decision_exact = DecisionMatches(model_case, legacy, candidate) && decision_exact;
            HashVector(result.raw_digest, raw);
            HashVector(result.legacy_digest, legacy);
            HashVector(result.candidate_digest, candidate);
            if (iteration == 0) {
                result.workloads.push_back(std::move(raw));
            }
        }
    }
    result.output_valid = output_valid;
    result.probability_valid = probability_valid;
    result.decision_exact = decision_exact;
    result.inference_mean_ms = inference_samples.empty()
                                   ? 0.0
                                   : std::accumulate(inference_samples.begin(), inference_samples.end(), 0.0) /
                                         static_cast<double>(inference_samples.size());
    return result;
}

template <typename SoftmaxFunction>
TimingSummary BenchmarkSoftmax(const std::vector<std::vector<float>>& workloads, int iterations,
                               SoftmaxFunction function) {
    std::vector<double> samples;
    samples.reserve(7);
    for (int round = 0; round < 7; ++round) {
        float checksum = 0.0f;
        const auto begin = Clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (const auto& workload : workloads) {
                const auto output = function(workload);
                if (!output.empty()) {
                    checksum += output[static_cast<size_t>(iteration) % output.size()];
                }
            }
        }
        const auto end = Clock::now();
        g_benchmark_sink = checksum;
        const double calls = static_cast<double>(iterations) * static_cast<double>(workloads.size());
        samples.push_back(std::chrono::duration<double, std::nano>(end - begin).count() / calls);
    }
    return Summarize(samples);
}

bool RunRealGate(inspire::InspireArchive& archive, const std::vector<inspirecv::Image>& images, int model_iterations,
                 int softmax_iterations) {
    const RealModelCase cases[] = {{"refine_net", -1, 0.0f, 2},
                                   {"mask_detect", 0, 0.95f, 2},
                                   {"rgb_anti_spoofing", 1, 0.88f, 0},
                                   {"face_emotion", -1, 0.0f, 7}};
    bool passed = true;
    std::vector<std::vector<float>> workloads;
    for (const auto& model_case : cases) {
        RealModelResult result = RunRealModel(archive, model_case, images, model_iterations);
        const bool case_passed = result.loaded && result.output_valid && result.probability_valid && result.decision_exact;
        workloads.insert(workloads.end(), result.workloads.begin(), result.workloads.end());
        std::cout << std::fixed << std::setprecision(9) << "SOFTMAX_REAL,name=" << model_case.name
                  << ",raw_digest=0x" << std::hex << result.raw_digest << ",legacy_digest=0x" << result.legacy_digest
                  << ",candidate_digest=0x" << result.candidate_digest << std::dec
                  << ",max_abs_diff=" << result.max_abs_diff << ",initialization_ms=" << result.initialization_ms
                  << ",inference_mean_ms=" << result.inference_mean_ms << ",observed_size=" << result.observed_size
                  << ",output=" << (result.output_valid ? "PASS" : "FAIL")
                  << ",probability=" << (result.probability_valid ? "PASS" : "FAIL")
                  << ",decision=" << (result.decision_exact ? "PASS" : "FAIL")
                  << ",status=" << (case_passed ? "PASS" : "FAIL") << '\n';
        passed = case_passed && passed;
    }
    if (workloads.empty()) {
        return false;
    }

    const TimingSummary legacy = BenchmarkSoftmax(workloads, softmax_iterations, LegacySoftmax);
    const TimingSummary candidate = BenchmarkSoftmax(
        workloads, softmax_iterations,
        [](const std::vector<float>& input) { return inspire::AnyNetAdapter::Softmax(input); });
    const bool latency_passed = candidate.p50_ns <= legacy.p50_ns * 1.50 + 50.0;
    std::cout << std::fixed << std::setprecision(3) << "SOFTMAX_TIMING,legacy_mean_ns=" << legacy.mean_ns
              << ",legacy_p50_ns=" << legacy.p50_ns << ",legacy_p95_ns=" << legacy.p95_ns
              << ",candidate_mean_ns=" << candidate.mean_ns << ",candidate_p50_ns=" << candidate.p50_ns
              << ",candidate_p95_ns=" << candidate.p95_ns
              << ",status=" << (latency_passed ? "PASS" : "FAIL") << '\n';
    return latency_passed && passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <pack_path> [test_res_root] [model_iterations] [softmax_iterations] [synthetic|real|full]\n";
        return 2;
    }
    const std::string pack_path = argv[1];
    const std::string test_root = argc >= 3 ? argv[2] : "test_res";
    const int model_iterations = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 10;
    const int softmax_iterations = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 100000;
    const std::string mode = argc >= 6 ? argv[5] : "full";
    if (mode != "synthetic" && mode != "real" && mode != "full") {
        std::cerr << "Mode must be synthetic, real, or full\n";
        return 2;
    }

    bool passed = true;
    if (mode == "synthetic" || mode == "full") {
        passed = RunSyntheticGate() && passed;
    }
    if (mode == "real" || mode == "full") {
        if (HFLaunchInspireFace(pack_path.c_str()) != HSUCCEED) {
            return 3;
        }
        std::vector<inspirecv::Image> images;
        passed = LoadImages(test_root, images) && passed;
        if (!images.empty()) {
            passed = RunRealGate(inspire::Launch::GetInstance()->getMArchive(), images, model_iterations,
                                 softmax_iterations) && passed;
        }
        if (HFTerminateInspireFace() != HSUCCEED) {
            passed = false;
        }
    }
    std::cout << "SUMMARY,mode=" << mode << ",images=3,model_iterations=" << model_iterations
              << ",softmax_iterations=" << softmax_iterations << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
