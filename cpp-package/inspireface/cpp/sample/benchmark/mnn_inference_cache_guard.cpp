#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
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

enum class ExpectedMode {
    kBaseline,
    kCached,
};

struct TimingSummary {
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
};

struct ModelResult {
    std::string name;
    uint64_t digest = 0;
    TimingSummary timing;
    InferenceCacheStatistics cache_statistics;
    size_t output_count = 0;
    size_t timed_calls = 0;
    bool repeatable = false;
    bool pointers_stable = false;
    bool cache_behavior_valid = false;
    bool passed = false;
};

struct PairedModelResult {
    ModelResult baseline;
    ModelResult cached;
};

struct BaselineRow {
    uint64_t digest = 0;
    TimingSummary timing;
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

void HashString(uint64_t& hash, const std::string& value) {
    HashValue(hash, value.size());
    HashBytes(hash, value.data(), value.size());
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

double Percentile(const std::vector<double>& samples, double percentile) {
    if (samples.empty()) {
        return 0.0;
    }
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double position = percentile * static_cast<double>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

double Median(const std::vector<double>& samples) {
    return Percentile(samples, 0.5);
}

TimingSummary Summarize(const std::vector<double>& samples) {
    TimingSummary summary;
    if (samples.empty()) {
        return summary;
    }
    summary.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    summary.p50_ms = Percentile(samples, 0.50);
    summary.p95_ms = Percentile(samples, 0.95);
    return summary;
}

TimingSummary MedianSummary(const std::vector<TimingSummary>& rounds) {
    std::vector<double> means;
    std::vector<double> p50s;
    std::vector<double> p95s;
    means.reserve(rounds.size());
    p50s.reserve(rounds.size());
    p95s.reserve(rounds.size());
    for (const auto& round : rounds) {
        means.push_back(round.mean_ms);
        p50s.push_back(round.p50_ms);
        p95s.push_back(round.p95_ms);
    }
    return {Median(means), Median(p50s), Median(p95s)};
}

bool HashOutputs(const inspire::AnyTensorViews& outputs, uint64_t& digest) {
    HashValue(digest, outputs.size());
    for (const auto& output : outputs) {
        if (output.name == nullptr || output.data == nullptr || output.size == 0) {
            return false;
        }
        HashString(digest, *output.name);
        HashValue(digest, output.size);
        HashBytes(digest, output.data, output.size * sizeof(float));
    }
    return true;
}

bool LoadModel(inspire::InspireArchive& archive, const std::string& model_name, inspire::InspireModel& model,
               std::unique_ptr<inspire::AnyNetAdapter>& net, bool dynamic = false) {
    if (archive.LoadModel(model_name, model) != inspire::SARC_SUCCESS) {
        std::cerr << "Unable to load model: " << model_name << '\n';
        return false;
    }
    net.reset(new inspire::AnyNetAdapter("MnnInferenceCacheGuard/" + model_name));
    if (net->LoadData(model, model.modelType, dynamic) != InferenceWrapper::WrapperOk) {
        std::cerr << "Unable to initialize model: " << model_name << '\n';
        return false;
    }
    return true;
}

bool LoadImages(const std::string& test_root, int width, int height, int channels, std::vector<inspirecv::Image>& images) {
    const char* paths[] = {"data/bulk/kun.jpg", "data/bulk/jntm.jpg", "data/bulk/woman.png"};
    images.clear();
    images.reserve(sizeof(paths) / sizeof(paths[0]));
    for (const char* path : paths) {
        auto image = inspirecv::Image::Create(JoinPath(test_root, path), channels);
        if (image.Empty()) {
            std::cerr << "Unable to load image: " << path << '\n';
            return false;
        }
        images.push_back(image.Resize(width, height));
        if (images.back().Empty() || images.back().Width() != width || images.back().Height() != height ||
            images.back().Channels() != channels) {
            return false;
        }
    }
    return true;
}

bool CacheCountsMatch(ExpectedMode mode, bool blob_input, const InferenceCacheStatistics& statistics, size_t timed_calls,
                      size_t output_count) {
    if (mode == ExpectedMode::kCached) {
        return statistics.image_process_creations == 0 && statistics.blob_host_tensor_creations == 0 &&
               statistics.output_host_tensor_creations == 0;
    }
    const uint64_t expected_image_creations = blob_input ? 0 : static_cast<uint64_t>(timed_calls);
    const uint64_t expected_blob_creations = blob_input ? static_cast<uint64_t>(timed_calls) : 0;
    const uint64_t expected_output_creations = static_cast<uint64_t>(timed_calls * output_count);
    return statistics.image_process_creations == expected_image_creations &&
           statistics.blob_host_tensor_creations == expected_blob_creations &&
           statistics.output_host_tensor_creations == expected_output_creations;
}

ModelResult RunImageModel(inspire::InspireArchive& archive, const std::string& model_name, const std::string& test_root,
                          int warmup_iterations, int timed_iterations, int rounds, ExpectedMode mode) {
    ModelResult result;
    result.name = model_name;

    inspire::InspireModel model;
    std::unique_ptr<inspire::AnyNetAdapter> net;
    if (!LoadModel(archive, model_name, model, net)) {
        return result;
    }
    net->SetInferenceCacheEnabledForTesting(mode == ExpectedMode::kCached);
    auto& inputs = net->getMInputTensorInfoList();
    if (inputs.size() != 1 || inputs[0].data_type != InputTensorInfo::DataTypeImage || inputs[0].GetWidth() <= 0 ||
        inputs[0].GetHeight() <= 0 || inputs[0].GetChannel() <= 0) {
        return result;
    }

    std::vector<inspirecv::Image> images;
    if (!LoadImages(test_root, inputs[0].GetWidth(), inputs[0].GetHeight(), inputs[0].image_info.channel, images)) {
        return result;
    }

    inspire::AnyTensorViews outputs;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        if (net->ForwardViews(images[static_cast<size_t>(iteration) % images.size()], outputs) != InferenceWrapper::WrapperOk || outputs.empty()) {
            return result;
        }
    }
    result.output_count = outputs.size();
    std::vector<const float*> stable_pointers;
    stable_pointers.reserve(outputs.size());
    for (const auto& output : outputs) {
        stable_pointers.push_back(output.data);
    }

    net->ResetInferenceCacheStatistics();
    std::vector<TimingSummary> round_timings;
    uint64_t reference_digest = 0;
    bool repeatable = true;
    bool pointers_stable = true;
    for (int round = 0; round < rounds; ++round) {
        uint64_t round_digest = kFnvOffset;
        std::vector<double> samples_ms;
        samples_ms.reserve(static_cast<size_t>(timed_iterations));
        for (int iteration = 0; iteration < timed_iterations; ++iteration) {
            const auto& image = images[static_cast<size_t>(iteration) % images.size()];
            const auto begin = Clock::now();
            const int32_t status = net->ForwardViews(image, outputs);
            const auto end = Clock::now();
            if (status != InferenceWrapper::WrapperOk || outputs.size() != result.output_count || !HashOutputs(outputs, round_digest)) {
                return result;
            }
            samples_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
            for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
                pointers_stable = pointers_stable && outputs[output_index].data == stable_pointers[output_index];
            }
        }
        if (round == 0) {
            reference_digest = round_digest;
        } else {
            repeatable = repeatable && round_digest == reference_digest;
        }
        round_timings.push_back(Summarize(samples_ms));
    }

    result.digest = reference_digest;
    result.timing = MedianSummary(round_timings);
    result.cache_statistics = net->GetInferenceCacheStatistics();
    result.timed_calls = static_cast<size_t>(timed_iterations * rounds);
    result.repeatable = repeatable;
    result.pointers_stable = pointers_stable;
    result.cache_behavior_valid = CacheCountsMatch(mode, false, result.cache_statistics, result.timed_calls, result.output_count);
    result.passed = repeatable && result.cache_behavior_valid && (mode == ExpectedMode::kBaseline || pointers_stable);
    return result;
}

ModelResult RunBlobModel(inspire::InspireArchive& archive, bool is_nchw, int warmup_iterations, int timed_iterations, int rounds,
                         ExpectedMode mode) {
    ModelResult result;
    result.name = is_nchw ? "face_detect_160_blob_nchw" : "face_detect_160_blob_nhwc";

    inspire::InspireModel model;
    std::unique_ptr<inspire::AnyNetAdapter> net;
    if (!LoadModel(archive, "face_detect_160", model, net)) {
        return result;
    }
    net->SetInferenceCacheEnabledForTesting(mode == ExpectedMode::kCached);
    auto& inputs = net->getMInputTensorInfoList();
    if (inputs.size() != 1 || inputs[0].GetElementNum() <= 0) {
        return result;
    }
    inputs[0].data_type = is_nchw ? InputTensorInfo::DataTypeBlobNchw : InputTensorInfo::DataTypeBlobNhwc;
    std::vector<float> blob(static_cast<size_t>(inputs[0].GetElementNum()));
    for (size_t index = 0; index < blob.size(); ++index) {
        blob[index] = static_cast<float>(static_cast<int>(index % 509) - 254) / 127.0f;
    }
    inputs[0].data = blob.data();

    inspire::AnyTensorViews outputs;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        if (net->ForwardViews(outputs) != InferenceWrapper::WrapperOk || outputs.empty()) {
            return result;
        }
    }
    result.output_count = outputs.size();
    std::vector<const float*> stable_pointers;
    stable_pointers.reserve(outputs.size());
    for (const auto& output : outputs) {
        stable_pointers.push_back(output.data);
    }

    net->ResetInferenceCacheStatistics();
    std::vector<TimingSummary> round_timings;
    uint64_t reference_digest = 0;
    bool repeatable = true;
    bool pointers_stable = true;
    for (int round = 0; round < rounds; ++round) {
        uint64_t round_digest = kFnvOffset;
        std::vector<double> samples_ms;
        samples_ms.reserve(static_cast<size_t>(timed_iterations));
        for (int iteration = 0; iteration < timed_iterations; ++iteration) {
            const auto begin = Clock::now();
            const int32_t status = net->ForwardViews(outputs);
            const auto end = Clock::now();
            if (status != InferenceWrapper::WrapperOk || outputs.size() != result.output_count || !HashOutputs(outputs, round_digest)) {
                return result;
            }
            samples_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
            for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
                pointers_stable = pointers_stable && outputs[output_index].data == stable_pointers[output_index];
            }
        }
        if (round == 0) {
            reference_digest = round_digest;
        } else {
            repeatable = repeatable && round_digest == reference_digest;
        }
        round_timings.push_back(Summarize(samples_ms));
    }

    result.digest = reference_digest;
    result.timing = MedianSummary(round_timings);
    result.cache_statistics = net->GetInferenceCacheStatistics();
    result.timed_calls = static_cast<size_t>(timed_iterations * rounds);
    result.repeatable = repeatable;
    result.pointers_stable = pointers_stable;
    result.cache_behavior_valid = CacheCountsMatch(mode, true, result.cache_statistics, result.timed_calls, result.output_count);
    result.passed = repeatable && result.cache_behavior_valid && (mode == ExpectedMode::kBaseline || pointers_stable);
    return result;
}

PairedModelResult RunPairedImageModel(inspire::InspireArchive& archive, const std::string& model_name,
                                      const std::string& test_root, int warmup_iterations, int timed_iterations, int rounds) {
    PairedModelResult pair;
    pair.baseline.name = model_name;
    pair.cached.name = model_name;

    inspire::InspireModel baseline_model;
    inspire::InspireModel cached_model;
    std::unique_ptr<inspire::AnyNetAdapter> baseline_net;
    std::unique_ptr<inspire::AnyNetAdapter> cached_net;
    if (!LoadModel(archive, model_name, baseline_model, baseline_net) ||
        !LoadModel(archive, model_name, cached_model, cached_net)) {
        return pair;
    }
    baseline_net->SetInferenceCacheEnabledForTesting(false);
    cached_net->SetInferenceCacheEnabledForTesting(true);
    const auto& baseline_inputs = baseline_net->getMInputTensorInfoList();
    const auto& cached_inputs = cached_net->getMInputTensorInfoList();
    if (baseline_inputs.size() != 1 || cached_inputs.size() != 1 ||
        baseline_inputs[0].data_type != InputTensorInfo::DataTypeImage ||
        cached_inputs[0].data_type != InputTensorInfo::DataTypeImage ||
        baseline_inputs[0].GetWidth() != cached_inputs[0].GetWidth() ||
        baseline_inputs[0].GetHeight() != cached_inputs[0].GetHeight() ||
        baseline_inputs[0].image_info.channel != cached_inputs[0].image_info.channel) {
        return pair;
    }

    std::vector<inspirecv::Image> images;
    if (!LoadImages(test_root, baseline_inputs[0].GetWidth(), baseline_inputs[0].GetHeight(),
                    baseline_inputs[0].image_info.channel, images)) {
        return pair;
    }

    inspire::AnyTensorViews baseline_outputs;
    inspire::AnyTensorViews cached_outputs;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        const auto& image = images[static_cast<size_t>(iteration) % images.size()];
        const bool cached_first = (iteration % 2) != 0;
        const int32_t first_status = cached_first ? cached_net->ForwardViews(image, cached_outputs)
                                                  : baseline_net->ForwardViews(image, baseline_outputs);
        const int32_t second_status = cached_first ? baseline_net->ForwardViews(image, baseline_outputs)
                                                   : cached_net->ForwardViews(image, cached_outputs);
        if (first_status != InferenceWrapper::WrapperOk || second_status != InferenceWrapper::WrapperOk ||
            baseline_outputs.empty() || baseline_outputs.size() != cached_outputs.size()) {
            return pair;
        }
    }

    pair.baseline.output_count = baseline_outputs.size();
    pair.cached.output_count = cached_outputs.size();
    std::vector<const float*> baseline_pointers;
    std::vector<const float*> cached_pointers;
    for (size_t index = 0; index < baseline_outputs.size(); ++index) {
        baseline_pointers.push_back(baseline_outputs[index].data);
        cached_pointers.push_back(cached_outputs[index].data);
    }
    baseline_net->ResetInferenceCacheStatistics();
    cached_net->ResetInferenceCacheStatistics();

    std::vector<TimingSummary> baseline_round_timings;
    std::vector<TimingSummary> cached_round_timings;
    uint64_t baseline_reference_digest = 0;
    uint64_t cached_reference_digest = 0;
    bool baseline_repeatable = true;
    bool cached_repeatable = true;
    bool baseline_pointers_stable = true;
    bool cached_pointers_stable = true;
    for (int round = 0; round < rounds; ++round) {
        uint64_t baseline_round_digest = kFnvOffset;
        uint64_t cached_round_digest = kFnvOffset;
        std::vector<double> baseline_samples_ms;
        std::vector<double> cached_samples_ms;
        baseline_samples_ms.reserve(static_cast<size_t>(timed_iterations));
        cached_samples_ms.reserve(static_cast<size_t>(timed_iterations));
        for (int iteration = 0; iteration < timed_iterations; ++iteration) {
            const auto& image = images[static_cast<size_t>(iteration) % images.size()];
            const auto run_once = [&](inspire::AnyNetAdapter& net, inspire::AnyTensorViews& outputs, uint64_t& digest,
                                      const std::vector<const float*>& pointers, bool& pointers_stable,
                                      std::vector<double>& samples_ms) {
                const auto begin = Clock::now();
                const int32_t status = net.ForwardViews(image, outputs);
                const auto end = Clock::now();
                if (status != InferenceWrapper::WrapperOk || outputs.size() != pointers.size() ||
                    !HashOutputs(outputs, digest)) {
                    return false;
                }
                samples_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
                for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
                    pointers_stable = pointers_stable && outputs[output_index].data == pointers[output_index];
                }
                return true;
            };
            const bool cached_first = ((round * timed_iterations + iteration) % 2) != 0;
            const bool first_ok = cached_first
                    ? run_once(*cached_net, cached_outputs, cached_round_digest, cached_pointers, cached_pointers_stable,
                               cached_samples_ms)
                    : run_once(*baseline_net, baseline_outputs, baseline_round_digest, baseline_pointers,
                               baseline_pointers_stable, baseline_samples_ms);
            const bool second_ok = cached_first
                    ? run_once(*baseline_net, baseline_outputs, baseline_round_digest, baseline_pointers,
                               baseline_pointers_stable, baseline_samples_ms)
                    : run_once(*cached_net, cached_outputs, cached_round_digest, cached_pointers, cached_pointers_stable,
                               cached_samples_ms);
            if (!first_ok || !second_ok) {
                return pair;
            }
        }
        if (round == 0) {
            baseline_reference_digest = baseline_round_digest;
            cached_reference_digest = cached_round_digest;
        } else {
            baseline_repeatable = baseline_repeatable && baseline_round_digest == baseline_reference_digest;
            cached_repeatable = cached_repeatable && cached_round_digest == cached_reference_digest;
        }
        baseline_round_timings.push_back(Summarize(baseline_samples_ms));
        cached_round_timings.push_back(Summarize(cached_samples_ms));
    }

    pair.baseline.digest = baseline_reference_digest;
    pair.cached.digest = cached_reference_digest;
    pair.baseline.timing = MedianSummary(baseline_round_timings);
    pair.cached.timing = MedianSummary(cached_round_timings);
    pair.baseline.cache_statistics = baseline_net->GetInferenceCacheStatistics();
    pair.cached.cache_statistics = cached_net->GetInferenceCacheStatistics();
    pair.baseline.timed_calls = pair.cached.timed_calls = static_cast<size_t>(timed_iterations * rounds);
    pair.baseline.repeatable = baseline_repeatable;
    pair.cached.repeatable = cached_repeatable;
    pair.baseline.pointers_stable = baseline_pointers_stable;
    pair.cached.pointers_stable = cached_pointers_stable;
    pair.baseline.cache_behavior_valid = CacheCountsMatch(ExpectedMode::kBaseline, false, pair.baseline.cache_statistics,
                                                          pair.baseline.timed_calls, pair.baseline.output_count);
    pair.cached.cache_behavior_valid = CacheCountsMatch(ExpectedMode::kCached, false, pair.cached.cache_statistics,
                                                        pair.cached.timed_calls, pair.cached.output_count);
    pair.baseline.passed = baseline_repeatable && pair.baseline.cache_behavior_valid;
    pair.cached.passed = cached_repeatable && cached_pointers_stable && pair.cached.cache_behavior_valid;
    return pair;
}

PairedModelResult RunPairedBlobModel(inspire::InspireArchive& archive, bool is_nchw, int warmup_iterations,
                                     int timed_iterations, int rounds) {
    PairedModelResult pair;
    pair.baseline.name = pair.cached.name = is_nchw ? "face_detect_160_blob_nchw" : "face_detect_160_blob_nhwc";

    inspire::InspireModel baseline_model;
    inspire::InspireModel cached_model;
    std::unique_ptr<inspire::AnyNetAdapter> baseline_net;
    std::unique_ptr<inspire::AnyNetAdapter> cached_net;
    if (!LoadModel(archive, "face_detect_160", baseline_model, baseline_net) ||
        !LoadModel(archive, "face_detect_160", cached_model, cached_net)) {
        return pair;
    }
    baseline_net->SetInferenceCacheEnabledForTesting(false);
    cached_net->SetInferenceCacheEnabledForTesting(true);
    auto& baseline_inputs = baseline_net->getMInputTensorInfoList();
    auto& cached_inputs = cached_net->getMInputTensorInfoList();
    if (baseline_inputs.size() != 1 || cached_inputs.size() != 1 ||
        baseline_inputs[0].GetElementNum() <= 0 ||
        baseline_inputs[0].GetElementNum() != cached_inputs[0].GetElementNum()) {
        return pair;
    }
    const auto data_type = is_nchw ? InputTensorInfo::DataTypeBlobNchw : InputTensorInfo::DataTypeBlobNhwc;
    baseline_inputs[0].data_type = data_type;
    cached_inputs[0].data_type = data_type;
    std::vector<float> blob(static_cast<size_t>(baseline_inputs[0].GetElementNum()));
    for (size_t index = 0; index < blob.size(); ++index) {
        blob[index] = static_cast<float>(static_cast<int>(index % 509) - 254) / 127.0f;
    }
    baseline_inputs[0].data = blob.data();
    cached_inputs[0].data = blob.data();

    inspire::AnyTensorViews baseline_outputs;
    inspire::AnyTensorViews cached_outputs;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        const bool cached_first = (iteration % 2) != 0;
        const int32_t first_status = cached_first ? cached_net->ForwardViews(cached_outputs)
                                                  : baseline_net->ForwardViews(baseline_outputs);
        const int32_t second_status = cached_first ? baseline_net->ForwardViews(baseline_outputs)
                                                   : cached_net->ForwardViews(cached_outputs);
        if (first_status != InferenceWrapper::WrapperOk || second_status != InferenceWrapper::WrapperOk ||
            baseline_outputs.empty() || baseline_outputs.size() != cached_outputs.size()) {
            return pair;
        }
    }

    pair.baseline.output_count = baseline_outputs.size();
    pair.cached.output_count = cached_outputs.size();
    std::vector<const float*> baseline_pointers;
    std::vector<const float*> cached_pointers;
    for (size_t index = 0; index < baseline_outputs.size(); ++index) {
        baseline_pointers.push_back(baseline_outputs[index].data);
        cached_pointers.push_back(cached_outputs[index].data);
    }
    baseline_net->ResetInferenceCacheStatistics();
    cached_net->ResetInferenceCacheStatistics();

    std::vector<TimingSummary> baseline_round_timings;
    std::vector<TimingSummary> cached_round_timings;
    uint64_t baseline_reference_digest = 0;
    uint64_t cached_reference_digest = 0;
    bool baseline_repeatable = true;
    bool cached_repeatable = true;
    bool baseline_pointers_stable = true;
    bool cached_pointers_stable = true;
    for (int round = 0; round < rounds; ++round) {
        uint64_t baseline_round_digest = kFnvOffset;
        uint64_t cached_round_digest = kFnvOffset;
        std::vector<double> baseline_samples_ms;
        std::vector<double> cached_samples_ms;
        baseline_samples_ms.reserve(static_cast<size_t>(timed_iterations));
        cached_samples_ms.reserve(static_cast<size_t>(timed_iterations));
        for (int iteration = 0; iteration < timed_iterations; ++iteration) {
            const auto run_once = [](inspire::AnyNetAdapter& net, inspire::AnyTensorViews& outputs, uint64_t& digest,
                                     const std::vector<const float*>& pointers, bool& pointers_stable,
                                     std::vector<double>& samples_ms) {
                const auto begin = Clock::now();
                const int32_t status = net.ForwardViews(outputs);
                const auto end = Clock::now();
                if (status != InferenceWrapper::WrapperOk || outputs.size() != pointers.size() ||
                    !HashOutputs(outputs, digest)) {
                    return false;
                }
                samples_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
                for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
                    pointers_stable = pointers_stable && outputs[output_index].data == pointers[output_index];
                }
                return true;
            };
            const bool cached_first = ((round * timed_iterations + iteration) % 2) != 0;
            const bool first_ok = cached_first
                    ? run_once(*cached_net, cached_outputs, cached_round_digest, cached_pointers, cached_pointers_stable,
                               cached_samples_ms)
                    : run_once(*baseline_net, baseline_outputs, baseline_round_digest, baseline_pointers,
                               baseline_pointers_stable, baseline_samples_ms);
            const bool second_ok = cached_first
                    ? run_once(*baseline_net, baseline_outputs, baseline_round_digest, baseline_pointers,
                               baseline_pointers_stable, baseline_samples_ms)
                    : run_once(*cached_net, cached_outputs, cached_round_digest, cached_pointers, cached_pointers_stable,
                               cached_samples_ms);
            if (!first_ok || !second_ok) {
                return pair;
            }
        }
        if (round == 0) {
            baseline_reference_digest = baseline_round_digest;
            cached_reference_digest = cached_round_digest;
        } else {
            baseline_repeatable = baseline_repeatable && baseline_round_digest == baseline_reference_digest;
            cached_repeatable = cached_repeatable && cached_round_digest == cached_reference_digest;
        }
        baseline_round_timings.push_back(Summarize(baseline_samples_ms));
        cached_round_timings.push_back(Summarize(cached_samples_ms));
    }

    pair.baseline.digest = baseline_reference_digest;
    pair.cached.digest = cached_reference_digest;
    pair.baseline.timing = MedianSummary(baseline_round_timings);
    pair.cached.timing = MedianSummary(cached_round_timings);
    pair.baseline.cache_statistics = baseline_net->GetInferenceCacheStatistics();
    pair.cached.cache_statistics = cached_net->GetInferenceCacheStatistics();
    pair.baseline.timed_calls = pair.cached.timed_calls = static_cast<size_t>(timed_iterations * rounds);
    pair.baseline.repeatable = baseline_repeatable;
    pair.cached.repeatable = cached_repeatable;
    pair.baseline.pointers_stable = baseline_pointers_stable;
    pair.cached.pointers_stable = cached_pointers_stable;
    pair.baseline.cache_behavior_valid = CacheCountsMatch(ExpectedMode::kBaseline, true, pair.baseline.cache_statistics,
                                                          pair.baseline.timed_calls, pair.baseline.output_count);
    pair.cached.cache_behavior_valid = CacheCountsMatch(ExpectedMode::kCached, true, pair.cached.cache_statistics,
                                                        pair.cached.timed_calls, pair.cached.output_count);
    pair.baseline.passed = baseline_repeatable && pair.baseline.cache_behavior_valid;
    pair.cached.passed = cached_repeatable && cached_pointers_stable && pair.cached.cache_behavior_valid;
    return pair;
}

bool ForwardAndHash(inspire::AnyNetAdapter& net, const inspirecv::Image& image, uint64_t& digest) {
    inspire::AnyTensorViews outputs;
    return net.ForwardViews(image, outputs) == InferenceWrapper::WrapperOk && !outputs.empty() && HashOutputs(outputs, digest);
}

bool TestMultipleInstances(inspire::InspireArchive& archive, const std::string& test_root, ExpectedMode mode) {
    inspire::InspireModel first_model;
    inspire::InspireModel second_model;
    std::unique_ptr<inspire::AnyNetAdapter> first;
    std::unique_ptr<inspire::AnyNetAdapter> second;
    if (!LoadModel(archive, "face_detect_160", first_model, first) || !LoadModel(archive, "face_detect_160", second_model, second)) {
        return false;
    }
    first->SetInferenceCacheEnabledForTesting(mode == ExpectedMode::kCached);
    second->SetInferenceCacheEnabledForTesting(mode == ExpectedMode::kCached);
    std::vector<inspirecv::Image> images;
    if (!LoadImages(test_root, 160, 160, 3, images)) {
        return false;
    }

    uint64_t warmup_digest = kFnvOffset;
    if (!ForwardAndHash(*first, images[0], warmup_digest) || !ForwardAndHash(*second, images[0], warmup_digest)) {
        return false;
    }
    first->ResetInferenceCacheStatistics();
    second->ResetInferenceCacheStatistics();
    uint64_t first_digest = kFnvOffset;
    uint64_t second_digest = kFnvOffset;
    for (int iteration = 0; iteration < 12; ++iteration) {
        const auto& image = images[static_cast<size_t>(iteration) % images.size()];
        if (!ForwardAndHash(*first, image, first_digest) || !ForwardAndHash(*second, image, second_digest)) {
            return false;
        }
    }
    const auto first_stats = first->GetInferenceCacheStatistics();
    const auto second_stats = second->GetInferenceCacheStatistics();
    const size_t output_count = first->getMOutputTensorInfoList().size();
    const bool cache_valid = CacheCountsMatch(mode, false, first_stats, 12, output_count) &&
                             CacheCountsMatch(mode, false, second_stats, 12, output_count);
    const bool passed = first_digest == second_digest && cache_valid;
    std::cout << "MNN_MULTI_INSTANCE,digest=0x" << std::hex << first_digest << std::dec << ",cache="
              << (cache_valid ? "PASS" : "FAIL") << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestDynamicInitialization(inspire::InspireArchive& archive, const std::string& test_root) {
    inspire::InspireModel normal_model;
    inspire::InspireModel dynamic_model;
    std::unique_ptr<inspire::AnyNetAdapter> normal;
    std::unique_ptr<inspire::AnyNetAdapter> dynamic;
    if (!LoadModel(archive, "face_detect_160", normal_model, normal, false) ||
        !LoadModel(archive, "face_detect_160", dynamic_model, dynamic, true)) {
        return false;
    }
    std::vector<inspirecv::Image> images;
    if (!LoadImages(test_root, 160, 160, 3, images)) {
        return false;
    }
    uint64_t normal_digest = kFnvOffset;
    uint64_t dynamic_digest = kFnvOffset;
    for (const auto& image : images) {
        if (!ForwardAndHash(*normal, image, normal_digest) || !ForwardAndHash(*dynamic, image, dynamic_digest)) {
            return false;
        }
    }
    const bool passed = normal_digest == dynamic_digest;
    std::cout << "MNN_DYNAMIC_INITIALIZATION,digest=0x" << std::hex << normal_digest << std::dec
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestLifecycle(inspire::InspireArchive& archive, const std::string& test_root) {
    auto image = inspirecv::Image::Create(JoinPath(test_root, "data/bulk/kun.jpg"), 3);
    if (image.Empty()) {
        return false;
    }
    image = image.Resize(96, 96);
    uint64_t reference_digest = 0;
    bool passed = true;
    for (int iteration = 0; iteration < 20; ++iteration) {
        inspire::InspireModel model;
        std::unique_ptr<inspire::AnyNetAdapter> net;
        if (!LoadModel(archive, "mask_detect", model, net)) {
            return false;
        }
        uint64_t digest = kFnvOffset;
        if (!ForwardAndHash(*net, image, digest)) {
            return false;
        }
        if (iteration == 0) {
            reference_digest = digest;
        } else {
            passed = passed && digest == reference_digest;
        }
    }
    std::cout << "MNN_LIFECYCLE,instances=20,digest=0x" << std::hex << reference_digest << std::dec
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestCacheInvalidation(inspire::InspireArchive& archive, const std::string& test_root) {
    inspire::InspireModel model;
    std::unique_ptr<inspire::AnyNetAdapter> net;
    if (!LoadModel(archive, "face_detect_160", model, net)) {
        return false;
    }
    net->SetInferenceCacheEnabledForTesting(true);
    auto& inputs = net->getMInputTensorInfoList();
    if (inputs.size() != 1) {
        return false;
    }
    std::vector<inspirecv::Image> images;
    if (!LoadImages(test_root, inputs[0].GetWidth(), inputs[0].GetHeight(), inputs[0].image_info.channel, images)) {
        return false;
    }

    uint64_t original_digest = kFnvOffset;
    if (!ForwardAndHash(*net, images[0], original_digest)) {
        return false;
    }
    net->ResetInferenceCacheStatistics();
    const float original_mean = inputs[0].normalize.mean[0];
    inputs[0].normalize.mean[0] = original_mean + 40.0f;
    uint64_t changed_digest = kFnvOffset;
    if (!ForwardAndHash(*net, images[0], changed_digest)) {
        return false;
    }
    inputs[0].normalize.mean[0] = original_mean;
    uint64_t restored_digest = kFnvOffset;
    if (!ForwardAndHash(*net, images[0], restored_digest)) {
        return false;
    }
    const auto signature_stats = net->GetInferenceCacheStatistics();
    const bool signature_rebuild = signature_stats.image_process_creations == 2 &&
                                   signature_stats.blob_host_tensor_creations == 0 &&
                                   signature_stats.output_host_tensor_creations == 0;

    if (net->ResizeInferenceInputForTesting() != InferenceWrapper::WrapperOk) {
        return false;
    }
    net->ResetInferenceCacheStatistics();
    uint64_t resized_digest = kFnvOffset;
    if (!ForwardAndHash(*net, images[0], resized_digest)) {
        return false;
    }
    const auto resize_stats = net->GetInferenceCacheStatistics();
    const bool resize_rebuild = resize_stats.image_process_creations == 1 &&
                                resize_stats.blob_host_tensor_creations == 0 &&
                                resize_stats.output_host_tensor_creations == 9;
    const bool changed_output = changed_digest != original_digest;
    const bool restore_exact = restored_digest == original_digest;
    const bool resize_exact = resized_digest == original_digest;
    const bool passed = changed_output && restore_exact && resize_exact && signature_rebuild && resize_rebuild;
    std::cout << "MNN_CACHE_INVALIDATION,digest=0x" << std::hex << original_digest << std::dec
              << ",changed_output=" << (changed_output ? "PASS" : "FAIL")
              << ",restore_exact=" << (restore_exact ? "PASS" : "FAIL")
              << ",resize_exact=" << (resize_exact ? "PASS" : "FAIL")
              << ",signature_rebuild=" << (signature_rebuild ? "PASS" : "FAIL")
              << ",resize_rebuild=" << (resize_rebuild ? "PASS" : "FAIL")
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestBlobRefresh(inspire::InspireArchive& archive) {
    inspire::InspireModel baseline_model;
    inspire::InspireModel cached_model;
    std::unique_ptr<inspire::AnyNetAdapter> baseline;
    std::unique_ptr<inspire::AnyNetAdapter> cached;
    if (!LoadModel(archive, "face_detect_160", baseline_model, baseline) ||
        !LoadModel(archive, "face_detect_160", cached_model, cached)) {
        return false;
    }
    baseline->SetInferenceCacheEnabledForTesting(false);
    cached->SetInferenceCacheEnabledForTesting(true);
    auto& baseline_inputs = baseline->getMInputTensorInfoList();
    auto& cached_inputs = cached->getMInputTensorInfoList();
    if (baseline_inputs.size() != 1 || cached_inputs.size() != 1 ||
        baseline_inputs[0].GetElementNum() <= 0 ||
        baseline_inputs[0].GetElementNum() != cached_inputs[0].GetElementNum() ||
        baseline_inputs[0].tensor_dims.empty() || cached_inputs[0].tensor_dims.empty()) {
        return false;
    }
    baseline_inputs[0].data_type = InputTensorInfo::DataTypeBlobNchw;
    cached_inputs[0].data_type = InputTensorInfo::DataTypeBlobNchw;
    std::vector<float> baseline_blob(static_cast<size_t>(baseline_inputs[0].GetElementNum()));
    std::vector<float> cached_blob(baseline_blob.size());
    for (size_t index = 0; index < baseline_blob.size(); ++index) {
        baseline_blob[index] = static_cast<float>(static_cast<int>(index % 509) - 254) / 127.0f;
    }
    cached_blob = baseline_blob;
    baseline_inputs[0].data = baseline_blob.data();
    cached_inputs[0].data = cached_blob.data();

    const auto forward_and_hash = [](inspire::AnyNetAdapter& net, uint64_t& digest,
                                     inspire::AnyTensorViews& outputs) {
        return net.ForwardViews(outputs) == InferenceWrapper::WrapperOk && !outputs.empty() && HashOutputs(outputs, digest);
    };
    inspire::AnyTensorViews baseline_outputs;
    inspire::AnyTensorViews cached_outputs;
    uint64_t baseline_first_digest = kFnvOffset;
    uint64_t cached_first_digest = kFnvOffset;
    if (!forward_and_hash(*baseline, baseline_first_digest, baseline_outputs) ||
        !forward_and_hash(*cached, cached_first_digest, cached_outputs) ||
        baseline_outputs.size() != cached_outputs.size()) {
        return false;
    }
    std::vector<const float*> cached_pointers;
    for (const auto& output : cached_outputs) {
        cached_pointers.push_back(output.data);
    }
    baseline->ResetInferenceCacheStatistics();
    cached->ResetInferenceCacheStatistics();

    baseline_blob[0] = 7.25f;
    cached_blob[0] = 7.25f;
    baseline_blob[baseline_blob.size() / 2] = -3.5f;
    cached_blob[cached_blob.size() / 2] = -3.5f;
    uint64_t baseline_changed_digest = kFnvOffset;
    uint64_t cached_changed_digest = kFnvOffset;
    if (!forward_and_hash(*baseline, baseline_changed_digest, baseline_outputs) ||
        !forward_and_hash(*cached, cached_changed_digest, cached_outputs)) {
        return false;
    }
    bool pointers_stable = cached_outputs.size() == cached_pointers.size();
    for (size_t index = 0; index < cached_outputs.size() && pointers_stable; ++index) {
        pointers_stable = cached_outputs[index].data == cached_pointers[index];
    }
    const auto baseline_stats = baseline->GetInferenceCacheStatistics();
    const auto cached_stats = cached->GetInferenceCacheStatistics();
    const bool counts_valid = CacheCountsMatch(ExpectedMode::kBaseline, true, baseline_stats, 1,
                                                baseline_outputs.size()) &&
                              CacheCountsMatch(ExpectedMode::kCached, true, cached_stats, 1,
                                               cached_outputs.size());
    const auto baseline_dimensions = baseline_inputs[0].tensor_dims;
    const auto cached_dimensions = cached_inputs[0].tensor_dims;
    ++baseline_inputs[0].tensor_dims.back();
    ++cached_inputs[0].tensor_dims.back();
    inspire::AnyTensorViews invalid_outputs;
    const bool invalid_size_rejected = baseline->ForwardViews(invalid_outputs) == InferenceWrapper::WrapperError &&
                                       invalid_outputs.empty() &&
                                       cached->ForwardViews(invalid_outputs) == InferenceWrapper::WrapperError &&
                                       invalid_outputs.empty();
    baseline_inputs[0].tensor_dims = baseline_dimensions;
    cached_inputs[0].tensor_dims = cached_dimensions;
    uint64_t recovered_digest = kFnvOffset;
    const bool recovered = forward_and_hash(*cached, recovered_digest, cached_outputs) &&
                           recovered_digest == cached_changed_digest;
    const bool passed = baseline_first_digest == cached_first_digest &&
                        baseline_changed_digest == cached_changed_digest &&
                        baseline_first_digest != baseline_changed_digest && pointers_stable && counts_valid &&
                        invalid_size_rejected && recovered;
    std::cout << "MNN_BLOB_REFRESH,initial_digest=0x" << std::hex << baseline_first_digest
              << ",changed_digest=0x" << baseline_changed_digest << std::dec
              << ",pointer_stability=" << (pointers_stable ? "PASS" : "FAIL")
              << ",cache=" << (counts_valid ? "PASS" : "FAIL")
              << ",invalid_size=" << (invalid_size_rejected ? "PASS" : "FAIL")
              << ",recovery=" << (recovered ? "PASS" : "FAIL")
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

void PrintModelResult(const ModelResult& result, const char* record_name = "MNN_MODEL") {
    std::cout << std::fixed << std::setprecision(6) << record_name << ",name=" << result.name << ",digest=0x" << std::hex << result.digest
              << std::dec << ",outputs=" << result.output_count << ",calls=" << result.timed_calls
              << ",mean_ms=" << result.timing.mean_ms << ",p50_ms=" << result.timing.p50_ms << ",p95_ms=" << result.timing.p95_ms
              << ",image_process_creations=" << result.cache_statistics.image_process_creations
              << ",blob_tensor_creations=" << result.cache_statistics.blob_host_tensor_creations
              << ",output_tensor_creations=" << result.cache_statistics.output_host_tensor_creations
              << ",repeatable=" << (result.repeatable ? "PASS" : "FAIL")
              << ",pointer_stability=" << (result.pointers_stable ? "PASS" : "FAIL")
              << ",cache=" << (result.cache_behavior_valid ? "PASS" : "FAIL")
              << ",status=" << (result.passed ? "PASS" : "FAIL") << '\n';
}

ModelResult AggregateResults(const std::vector<ModelResult>& results) {
    ModelResult aggregate;
    aggregate.name = "AGGREGATE";
    aggregate.digest = kFnvOffset;
    aggregate.repeatable = true;
    aggregate.pointers_stable = true;
    aggregate.cache_behavior_valid = true;
    aggregate.passed = true;
    for (const auto& result : results) {
        HashString(aggregate.digest, result.name);
        HashValue(aggregate.digest, result.digest);
        aggregate.timing.mean_ms += result.timing.mean_ms;
        aggregate.timing.p50_ms += result.timing.p50_ms;
        aggregate.timing.p95_ms += result.timing.p95_ms;
        aggregate.cache_statistics.image_process_creations += result.cache_statistics.image_process_creations;
        aggregate.cache_statistics.blob_host_tensor_creations += result.cache_statistics.blob_host_tensor_creations;
        aggregate.cache_statistics.output_host_tensor_creations += result.cache_statistics.output_host_tensor_creations;
        aggregate.output_count += result.output_count;
        aggregate.timed_calls += result.timed_calls;
        aggregate.repeatable = aggregate.repeatable && result.repeatable;
        aggregate.pointers_stable = aggregate.pointers_stable && result.pointers_stable;
        aggregate.cache_behavior_valid = aggregate.cache_behavior_valid && result.cache_behavior_valid;
        aggregate.passed = aggregate.passed && result.passed;
    }
    return aggregate;
}

bool WriteBaseline(const std::string& path, const std::vector<ModelResult>& results, const ModelResult& aggregate) {
    std::ofstream output(path);
    if (!output) {
        return false;
    }
    output << "name,digest,mean_ms,p50_ms,p95_ms\n";
    output << std::setprecision(17);
    const auto write_row = [&output](const ModelResult& result) {
        output << result.name << ",0x" << std::hex << result.digest << std::dec << ',' << result.timing.mean_ms << ','
               << result.timing.p50_ms << ',' << result.timing.p95_ms << '\n';
    };
    for (const auto& result : results) {
        write_row(result);
    }
    write_row(aggregate);
    return static_cast<bool>(output);
}

bool ReadBaseline(const std::string& path, std::unordered_map<std::string, BaselineRow>& rows) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        std::istringstream stream(line);
        std::array<std::string, 5> fields;
        for (size_t index = 0; index < fields.size(); ++index) {
            if (!std::getline(stream, fields[index], ',')) {
                return false;
            }
        }
        try {
            BaselineRow row;
            row.digest = std::stoull(fields[1], nullptr, 0);
            row.timing.mean_ms = std::stod(fields[2]);
            row.timing.p50_ms = std::stod(fields[3]);
            row.timing.p95_ms = std::stod(fields[4]);
            rows.emplace(fields[0], row);
        } catch (const std::exception&) {
            return false;
        }
    }
    return !rows.empty();
}

bool ComparePerformanceRows(const std::vector<ModelResult>& results, const ModelResult& aggregate,
                            const std::unordered_map<std::string, BaselineRow>& baseline) {
    bool passed = true;
    const auto compare_row = [&baseline, &passed](const ModelResult& result, bool require_improvement) {
        const auto found = baseline.find(result.name);
        if (found == baseline.end()) {
            std::cerr << "Missing baseline row: " << result.name << '\n';
            passed = false;
            return;
        }
        const auto& before = found->second;
        const bool exact = result.digest == before.digest;
        const bool no_regression = result.timing.mean_ms <= before.timing.mean_ms * 1.02 &&
                                   result.timing.p50_ms <= before.timing.p50_ms * 1.02 &&
                                   result.timing.p95_ms <= before.timing.p95_ms * 1.05;
        const bool improvement = !require_improvement || result.timing.p50_ms <= before.timing.p50_ms * 0.98;
        const bool row_passed = exact && no_regression && improvement;
        passed = passed && row_passed;
        std::cout << std::fixed << std::setprecision(3) << "MNN_COMPARE,name=" << result.name
                  << ",digest=" << (exact ? "EXACT" : "CHANGED")
                  << ",mean_change_pct=" << ((result.timing.mean_ms / before.timing.mean_ms) - 1.0) * 100.0
                  << ",p50_change_pct=" << ((result.timing.p50_ms / before.timing.p50_ms) - 1.0) * 100.0
                  << ",p95_change_pct=" << ((result.timing.p95_ms / before.timing.p95_ms) - 1.0) * 100.0
                  << ",status=" << (row_passed ? "PASS" : "FAIL") << '\n';
    };
    for (const auto& result : results) {
        compare_row(result, false);
    }
    compare_row(aggregate, true);
    return passed;
}

bool ComparePerformance(const std::vector<ModelResult>& results, const ModelResult& aggregate, const std::string& baseline_path) {
    std::unordered_map<std::string, BaselineRow> baseline;
    if (!ReadBaseline(baseline_path, baseline)) {
        std::cerr << "Unable to read baseline: " << baseline_path << '\n';
        return false;
    }
    return ComparePerformanceRows(results, aggregate, baseline);
}

bool ComparePairedPerformance(const std::vector<ModelResult>& baseline_results, const ModelResult& baseline_aggregate,
                              const std::vector<ModelResult>& cached_results, const ModelResult& cached_aggregate) {
    if (baseline_results.size() != cached_results.size()) {
        return false;
    }
    bool passed = true;
    const auto compare_row = [&passed](const ModelResult& before, const ModelResult& after, bool aggregate) {
        const bool exact = before.name == after.name && before.digest == after.digest;
        // Individual sub-millisecond models need an absolute noise allowance. The aggregate remains strict because it
        // represents the end-to-end cost across every model and is much less sensitive to timer granularity.
        const double mean_limit = aggregate ? before.timing.mean_ms * 1.01
                                            : std::max(before.timing.mean_ms * 1.05, before.timing.mean_ms + 0.05);
        const double p50_limit = aggregate ? before.timing.p50_ms * 1.03
                                           : std::max(before.timing.p50_ms * 1.10, before.timing.p50_ms + 0.10);
        const double p95_limit = aggregate ? before.timing.p95_ms * 1.03
                                           : std::max(before.timing.p95_ms * 1.25, before.timing.p95_ms + 0.20);
        const bool no_regression = after.timing.mean_ms <= mean_limit && after.timing.p50_ms <= p50_limit &&
                                   after.timing.p95_ms <= p95_limit;
        const bool row_passed = exact && no_regression;
        passed = passed && row_passed;
        std::cout << std::fixed << std::setprecision(3) << "MNN_COMPARE,name=" << after.name
                  << ",digest=" << (exact ? "EXACT" : "CHANGED")
                  << ",mean_change_pct=" << ((after.timing.mean_ms / before.timing.mean_ms) - 1.0) * 100.0
                  << ",p50_change_pct=" << ((after.timing.p50_ms / before.timing.p50_ms) - 1.0) * 100.0
                  << ",p95_change_pct=" << ((after.timing.p95_ms / before.timing.p95_ms) - 1.0) * 100.0
                  << ",status=" << (row_passed ? "PASS" : "FAIL") << '\n';
    };
    for (size_t index = 0; index < baseline_results.size(); ++index) {
        compare_row(baseline_results[index], cached_results[index], false);
    }
    compare_row(baseline_aggregate, cached_aggregate, true);
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <pack_path> [test_res_root] [timed_iterations] [rounds] "
                     "[baseline|cached|cached-only|paired] [metrics_file]\n";
        return 2;
    }
    const std::string pack_path = argv[1];
    const std::string test_root = argc >= 3 ? argv[2] : "test_res";
    const int timed_iterations = argc >= 4 ? std::max(3, std::atoi(argv[3])) : 30;
    const int rounds = argc >= 5 ? std::max(2, std::atoi(argv[4])) : 5;
    const std::string mode_name = argc >= 6 ? argv[5] : "baseline";
    const std::string metrics_path = argc >= 7 ? argv[6] : std::string();
    const bool paired_mode = mode_name == "paired";
    const bool cached_only_mode = mode_name == "cached-only";
    const ExpectedMode mode = (mode_name == "cached" || cached_only_mode) ? ExpectedMode::kCached
                                                                           : ExpectedMode::kBaseline;
    if (mode_name != "baseline" && mode_name != "cached" && !cached_only_mode && !paired_mode) {
        std::cerr << "Mode must be baseline, cached, cached-only, or paired\n";
        return 2;
    }
    if (mode_name == "cached" && metrics_path.empty()) {
        std::cerr << "Cached mode requires a baseline metrics file\n";
        return 2;
    }

    if (HFLaunchInspireFace(pack_path.c_str()) != HSUCCEED) {
        return 3;
    }
    bool passed = true;
    auto& archive = inspire::Launch::GetInstance()->getMArchive();
    const std::vector<std::string> model_names = {"face_detect_160", "face_detect_320", "face_detect_640", "landmark",
                                                   "refine_net",      "pose_quality",    "feature",         "mask_detect",
                                                   "rgb_anti_spoofing", "face_attribute", "blink_predict",   "face_emotion"};
    std::vector<ModelResult> results;
    results.reserve(model_names.size() + 2);
    std::vector<ModelResult> paired_baseline_results;
    paired_baseline_results.reserve(model_names.size() + 2);
    constexpr int kWarmupIterations = 20;
    if (paired_mode) {
        const auto append_pair = [&](PairedModelResult pair) {
            PrintModelResult(pair.baseline, "MNN_PAIRED_BASELINE");
            PrintModelResult(pair.cached, "MNN_PAIRED_CACHED");
            passed = pair.baseline.passed && pair.cached.passed && passed;
            paired_baseline_results.push_back(std::move(pair.baseline));
            results.push_back(std::move(pair.cached));
        };
        for (const auto& model_name : model_names) {
            append_pair(RunPairedImageModel(archive, model_name, test_root, kWarmupIterations, timed_iterations, rounds));
        }
        append_pair(RunPairedBlobModel(archive, true, kWarmupIterations, timed_iterations, rounds));
        append_pair(RunPairedBlobModel(archive, false, kWarmupIterations, timed_iterations, rounds));
    } else {
        for (const auto& model_name : model_names) {
            results.push_back(RunImageModel(archive, model_name, test_root, kWarmupIterations, timed_iterations, rounds, mode));
            PrintModelResult(results.back());
            passed = results.back().passed && passed;
        }
        results.push_back(RunBlobModel(archive, true, kWarmupIterations, timed_iterations, rounds, mode));
        PrintModelResult(results.back());
        passed = results.back().passed && passed;
        results.push_back(RunBlobModel(archive, false, kWarmupIterations, timed_iterations, rounds, mode));
        PrintModelResult(results.back());
        passed = results.back().passed && passed;
    }

    const auto aggregate = AggregateResults(results);
    PrintModelResult(aggregate);
    passed = aggregate.passed && passed;
    ModelResult paired_baseline_aggregate;
    if (paired_mode) {
        paired_baseline_aggregate = AggregateResults(paired_baseline_results);
        PrintModelResult(paired_baseline_aggregate, "MNN_PAIRED_BASELINE_AGGREGATE");
    }
    const ExpectedMode structural_mode = paired_mode ? ExpectedMode::kCached : mode;
    passed = TestMultipleInstances(archive, test_root, structural_mode) && passed;
    passed = TestDynamicInitialization(archive, test_root) && passed;
    passed = TestCacheInvalidation(archive, test_root) && passed;
    passed = TestBlobRefresh(archive) && passed;
    passed = TestLifecycle(archive, test_root) && passed;

    if (paired_mode) {
        passed = ComparePairedPerformance(paired_baseline_results, paired_baseline_aggregate, results, aggregate) && passed;
    } else if (mode == ExpectedMode::kBaseline && !metrics_path.empty()) {
        const bool written = WriteBaseline(metrics_path, results, aggregate);
        std::cout << "MNN_BASELINE_WRITE,path=" << metrics_path << ",status=" << (written ? "PASS" : "FAIL") << '\n';
        passed = written && passed;
    } else if (mode_name == "cached") {
        passed = ComparePerformance(results, aggregate, metrics_path) && passed;
    }

    if (HFTerminateInspireFace() != HSUCCEED) {
        passed = false;
    }
    std::cout << "SUMMARY,mode=" << mode_name << ",models=" << results.size() << ",images=3,rounds=" << rounds
              << ",timed_iterations=" << timed_iterations << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
