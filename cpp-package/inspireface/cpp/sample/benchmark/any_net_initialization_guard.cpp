#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
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
constexpr double kLatencyRatioLimit = 1.50;
constexpr double kLatencyAbsoluteLimitMs = 1.00;

struct TimingSummary {
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
};

struct RunResult {
    bool loaded = false;
    bool metadata_exact = false;
    bool output_valid = false;
    uint64_t digest = 0;
    double initialization_ms = 0.0;
    TimingSummary inference;
};

struct ReloadCase {
    const char* source;
    const char* target;
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

bool LatencyMatches(const RunResult& reference, const RunResult& candidate) {
    if (!reference.output_valid || !candidate.output_valid) {
        return false;
    }
    return candidate.inference.p50_ms <= reference.inference.p50_ms * kLatencyRatioLimit + kLatencyAbsoluteLimitMs;
}

template <typename T>
T ConfigOr(const inspire::Configurable& config, const std::string& name, const T& fallback) {
    return config.has(name) ? config.get<T>(name) : fallback;
}

bool LoadImages(const std::string& test_root, std::vector<inspirecv::Image>& images) {
    const char* paths[] = {"data/bulk/kun.jpg", "data/bulk/jntm.jpg", "data/bulk/woman.png"};
    images.clear();
    images.reserve(sizeof(paths) / sizeof(paths[0]));
    for (const char* relative_path : paths) {
        auto image = inspirecv::Image::Create(JoinPath(test_root, relative_path), 3);
        if (image.Empty()) {
            std::cerr << "Unable to load image: " << relative_path << '\n';
            return false;
        }
        images.push_back(std::move(image));
    }
    return true;
}

bool MetadataMatches(inspire::InspireModel& model, inspire::AnyNetAdapter& net) {
    const auto& config = model.Config();
    const auto& inputs = net.getMInputTensorInfoList();
    const auto& outputs = net.getMOutputTensorInfoList();
    if (inputs.size() != 1) {
        return false;
    }

    const std::vector<int> input_size = ConfigOr<std::vector<int>>(config, "input_size", {320, 320});
    const std::vector<float> mean = ConfigOr<std::vector<float>>(config, "mean", {127.5f, 127.5f, 127.5f});
    const std::vector<float> norm = ConfigOr<std::vector<float>>(config, "norm", {0.0078125f, 0.0078125f, 0.0078125f});
    const std::vector<std::string> output_names = ConfigOr<std::vector<std::string>>(config, "outputs_layers", {""});
    if (input_size.size() < 2 || mean.size() < 3 || norm.size() < 3 || outputs.size() != output_names.size()) {
        return false;
    }

    const int channel = ConfigOr<int>(config, "input_channel", 3);
    const int image_channel = ConfigOr<int>(config, "input_image_channel", 3);
    const bool nchw = ConfigOr<bool>(config, "nchw", true);
    const auto& input = inputs.front();
    const std::vector<int32_t> expected_dims = nchw
                                                 ? std::vector<int32_t>{1, channel, input_size[1], input_size[0]}
                                                 : std::vector<int32_t>{1, input_size[1], input_size[0], channel};
    bool exact = input.name == ConfigOr<std::string>(config, "input_layer", "") &&
                 input.tensor_type == ConfigOr<int>(config, "input_tensor_type", TensorInfo::TensorTypeFp32) &&
                 input.data_type == ConfigOr<int>(config, "data_type", InputTensorInfo::DataTypeImage) &&
                 input.is_nchw == nchw && input.tensor_dims == expected_dims &&
                 input.image_info.width == input_size[0] && input.image_info.height == input_size[1] &&
                 input.image_info.channel == image_channel && input.image_info.crop_x == 0 &&
                 input.image_info.crop_y == 0 && input.image_info.crop_width == input_size[0] &&
                 input.image_info.crop_height == input_size[1] && input.image_info.is_bgr == nchw &&
                 input.image_info.swap_color == ConfigOr<bool>(config, "swap_color", false);
    for (size_t index = 0; index < 3; ++index) {
        exact = exact && input.normalize.mean[index] == mean[index] && input.normalize.norm[index] == norm[index];
    }
    const int output_type = ConfigOr<int>(config, "output_tensor_type", TensorInfo::TensorTypeFp32);
    for (size_t index = 0; index < outputs.size(); ++index) {
        exact = exact && outputs[index].name == output_names[index] && outputs[index].tensor_type == output_type;
    }
    return exact;
}

bool ForwardAndHash(inspire::AnyNetAdapter& net, const std::vector<inspirecv::Image>& source_images, int iterations,
                    uint64_t& digest, TimingSummary& timing) {
    const auto& inputs = net.getMInputTensorInfoList();
    if (inputs.size() != 1 || inputs.front().image_info.width <= 0 || inputs.front().image_info.height <= 0) {
        return false;
    }
    std::vector<inspirecv::Image> images;
    images.reserve(source_images.size());
    for (const auto& source : source_images) {
        auto resized = source.Resize(inputs.front().image_info.width, inputs.front().image_info.height);
        if (resized.Empty()) {
            return false;
        }
        images.push_back(std::move(resized));
    }

    for (const auto& image : images) {
        inspire::AnyTensorViews warmup_views;
        if (net.ForwardViews(image, warmup_views) != InferenceWrapper::WrapperOk || warmup_views.empty()) {
            return false;
        }
    }

    digest = kFnvOffset;
    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(iterations) * images.size());
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (const auto& image : images) {
            inspire::AnyTensorViews views;
            const auto begin = Clock::now();
            const int32_t status = net.ForwardViews(image, views);
            const auto end = Clock::now();
            samples_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
            const auto& outputs = net.getMOutputTensorInfoList();
            if (status != InferenceWrapper::WrapperOk || views.empty() || views.size() != outputs.size()) {
                return false;
            }
            HashValue(digest, views.size());
            for (size_t index = 0; index < views.size(); ++index) {
                const auto& view = views[index];
                if (view.name == nullptr || view.data == nullptr || view.size == 0 || view.name != &outputs[index].name ||
                    outputs[index].GetElementNum() <= 0 || view.size != static_cast<size_t>(outputs[index].GetElementNum())) {
                    return false;
                }
                HashString(digest, *view.name);
                HashValue(digest, view.size);
                HashBytes(digest, view.data, view.size * sizeof(float));
                HashValue(digest, outputs[index].tensor_dims.size());
                HashBytes(digest, outputs[index].tensor_dims.data(), outputs[index].tensor_dims.size() * sizeof(int32_t));
            }
        }
    }
    timing = Summarize(samples_ms);
    return true;
}

bool LoadInto(inspire::InspireArchive& archive, const std::string& model_name, inspire::AnyNetAdapter& net,
              inspire::InspireModel& model, double& initialization_ms, bool dynamic = false) {
    if (archive.LoadModel(model_name, model) != inspire::SARC_SUCCESS) {
        std::cerr << "Unable to load model: " << model_name << '\n';
        return false;
    }
    const auto begin = Clock::now();
    const int32_t status = net.LoadData(model, model.modelType, dynamic);
    const auto end = Clock::now();
    initialization_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return status == InferenceWrapper::WrapperOk;
}

RunResult RunFresh(inspire::InspireArchive& archive, const std::string& model_name,
                   const std::vector<inspirecv::Image>& images, int iterations, bool dynamic = false) {
    RunResult result;
    inspire::InspireModel model;
    inspire::AnyNetAdapter net("AnyNetInitializationGuard/fresh/" + model_name);
    result.loaded = LoadInto(archive, model_name, net, model, result.initialization_ms, dynamic);
    if (!result.loaded) {
        return result;
    }
    result.metadata_exact = MetadataMatches(model, net);
    result.output_valid = ForwardAndHash(net, images, iterations, result.digest, result.inference);
    return result;
}

RunResult RunReload(inspire::InspireArchive& archive, const ReloadCase& reload_case,
                    const std::vector<inspirecv::Image>& images, int iterations) {
    RunResult result;
    inspire::AnyNetAdapter net(std::string("AnyNetInitializationGuard/reload/") + reload_case.source + "-" + reload_case.target);
    inspire::InspireModel source_model;
    double source_initialization_ms = 0.0;
    if (!LoadInto(archive, reload_case.source, net, source_model, source_initialization_ms)) {
        return result;
    }
    inspire::InspireModel target_model;
    result.loaded = LoadInto(archive, reload_case.target, net, target_model, result.initialization_ms);
    if (!result.loaded) {
        return result;
    }
    result.metadata_exact = MetadataMatches(target_model, net);
    result.output_valid = ForwardAndHash(net, images, iterations, result.digest, result.inference);
    return result;
}

void PrintResult(const char* prefix, const std::string& name, const RunResult& result, bool exact = true,
                 bool latency = true) {
    std::cout << std::fixed << std::setprecision(6) << prefix << ",name=" << name << ",digest=0x" << std::hex
              << result.digest << std::dec << ",initialization_ms=" << result.initialization_ms
              << ",mean_ms=" << result.inference.mean_ms << ",p50_ms=" << result.inference.p50_ms
              << ",p95_ms=" << result.inference.p95_ms << ",loaded=" << (result.loaded ? "PASS" : "FAIL")
              << ",metadata=" << (result.metadata_exact ? "PASS" : "FAIL")
              << ",output=" << (result.output_valid ? "PASS" : "FAIL")
              << ",exact=" << (exact ? "PASS" : "FAIL")
              << ",latency=" << (latency ? "PASS" : "FAIL") << '\n';
}

bool RunSingleGate(inspire::InspireArchive& archive, const std::vector<inspirecv::Image>& images, int iterations) {
    const char* model_names[] = {"face_detect_160", "face_detect_320", "face_detect_640", "landmark",
                                 "feature", "mask_detect", "face_attribute", "face_emotion"};
    bool passed = true;
    for (const char* model_name : model_names) {
        const RunResult result = RunFresh(archive, model_name, images, iterations);
        const bool case_passed = result.loaded && result.metadata_exact && result.output_valid;
        PrintResult("ANYNET_SINGLE", model_name, result, case_passed);
        passed = case_passed && passed;
    }

    const RunResult normal = RunFresh(archive, "face_detect_160", images, iterations, false);
    const RunResult dynamic = RunFresh(archive, "face_detect_160", images, iterations, true);
    const bool dynamic_exact = normal.loaded && normal.metadata_exact && normal.output_valid && dynamic.loaded &&
                               dynamic.metadata_exact && dynamic.output_valid && normal.digest == dynamic.digest;
    const bool dynamic_latency = LatencyMatches(normal, dynamic);
    PrintResult("ANYNET_DYNAMIC", "face_detect_160", dynamic, dynamic_exact, dynamic_latency);
    return dynamic_exact && dynamic_latency && passed;
}

bool RunReloadGate(inspire::InspireArchive& archive, const std::vector<inspirecv::Image>& images, int iterations) {
    const ReloadCase cases[] = {{"face_detect_160", "face_detect_320"}, {"face_detect_320", "face_detect_160"},
                                {"face_detect_160", "face_detect_640"}, {"mask_detect", "face_attribute"},
                                {"feature", "landmark"},               {"face_detect_160", "face_detect_160"}};
    bool passed = true;
    for (const auto& reload_case : cases) {
        const RunResult fresh = RunFresh(archive, reload_case.target, images, iterations);
        const RunResult reloaded = RunReload(archive, reload_case, images, iterations);
        const bool exact = fresh.loaded && fresh.metadata_exact && fresh.output_valid && reloaded.loaded &&
                           reloaded.metadata_exact && reloaded.output_valid && fresh.digest == reloaded.digest;
        const bool latency = LatencyMatches(fresh, reloaded);
        PrintResult("ANYNET_RELOAD_FRESH", reload_case.target, fresh, true);
        PrintResult("ANYNET_RELOAD", std::string(reload_case.source) + "->" + reload_case.target, reloaded, exact,
                    latency);
        passed = exact && latency && passed;
    }
    return passed;
}

bool RunFailureRecoveryGate(inspire::InspireArchive& archive, const std::vector<inspirecv::Image>& images,
                            int iterations) {
    inspire::AnyNetAdapter net("AnyNetInitializationGuard/failure-recovery");
    inspire::InspireModel source_model;
    double source_initialization_ms = 0.0;
    if (!LoadInto(archive, "face_detect_160", net, source_model, source_initialization_ms)) {
        std::cout << "ANYNET_RECOVERY,rejected=FAIL,recovered=FAIL,exact=FAIL\n";
        return false;
    }

    inspire::InspireModel corrupt_model;
    if (archive.LoadModel("face_detect_320", corrupt_model) != inspire::SARC_SUCCESS) {
        std::cout << "ANYNET_RECOVERY,rejected=FAIL,recovered=FAIL,exact=FAIL\n";
        return false;
    }
    std::vector<char> corrupt_buffer(64, 0);
    corrupt_model.buffer = corrupt_buffer.data();
    corrupt_model.bufferSize = corrupt_buffer.size();
    corrupt_model.loadFilePath = 0;
    const bool rejected = net.LoadData(corrupt_model, corrupt_model.modelType) != InferenceWrapper::WrapperOk;

    RunResult recovered;
    inspire::InspireModel recovery_model;
    recovered.loaded = LoadInto(archive, "face_detect_320", net, recovery_model, recovered.initialization_ms);
    if (recovered.loaded) {
        recovered.metadata_exact = MetadataMatches(recovery_model, net);
        recovered.output_valid = ForwardAndHash(net, images, iterations, recovered.digest, recovered.inference);
    }
    const RunResult fresh = RunFresh(archive, "face_detect_320", images, iterations);
    const bool exact = rejected && recovered.loaded && recovered.metadata_exact && recovered.output_valid && fresh.loaded &&
                       fresh.metadata_exact && fresh.output_valid && recovered.digest == fresh.digest;
    const bool latency = LatencyMatches(fresh, recovered);
    PrintResult("ANYNET_RECOVERY", "corrupt-face_detect_320->face_detect_320", recovered, exact, latency);
    std::cout << "ANYNET_RECOVERY_STATE,rejected=" << (rejected ? "PASS" : "FAIL")
              << ",recovered=" << (recovered.loaded ? "PASS" : "FAIL")
              << ",exact=" << (exact ? "PASS" : "FAIL")
              << ",latency=" << (latency ? "PASS" : "FAIL") << '\n';
    return exact && latency;
}

bool RunMetadataBoundaryGate(inspire::InspireArchive& archive) {
    auto rejects = [&](const std::string& name, const std::function<void(inspire::InspireModel&)>& mutate) {
        inspire::InspireModel model;
        if (archive.LoadModel("face_detect_160", model) != inspire::SARC_SUCCESS) {
            return false;
        }
        mutate(model);
        inspire::AnyNetAdapter net("AnyNetInitializationGuard/boundary/" + name);
        return net.LoadData(model, model.modelType) == InferenceWrapper::WrapperError;
    };

    bool passed = true;
    passed = rejects("zero-norm", [](inspire::InspireModel& model) {
        model.Config().set<std::vector<float>>("norm", {1.0f, 0.0f, 1.0f});
    }) && passed;
    passed = rejects("nan-mean", [](inspire::InspireModel& model) {
        model.Config().set<std::vector<float>>("mean", {0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f});
    }) && passed;
    passed = rejects("image-channel", [](inspire::InspireModel& model) {
        model.Config().set<int>("input_image_channel", 2);
    }) && passed;
    passed = rejects("tensor-channel", [](inspire::InspireModel& model) {
        model.Config().set<int>("input_channel", 4);
    }) && passed;
    passed = rejects("shape-overflow", [](inspire::InspireModel& model) {
        model.Config().set<std::vector<int>>("input_size", {std::numeric_limits<int>::max(), 2});
    }) && passed;
    passed = rejects("threads", [](inspire::InspireModel& model) {
        model.Config().set<int>("threads", 0);
    }) && passed;
    passed = rejects("buffer-size", [](inspire::InspireModel& model) {
        model.bufferSize = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
        model.loadFilePath = 0;
    }) && passed;

    std::cout << "ANYNET_METADATA_BOUNDARY,cases=7,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool RunRuntimeBoundaryGate(inspire::InspireArchive& archive, const std::vector<inspirecv::Image>& images) {
    if (images.empty()) {
        return false;
    }

    inspire::AnyNetAdapter net("AnyNetInitializationGuard/runtime-boundary");
    inspire::AnyTensorOutputs owned_outputs{{"stale", {1.0f}}};
    std::string stale_name = "stale";
    float stale_value = 1.0f;
    inspire::AnyTensorViews views;
    views.emplace_back(&stale_name, &stale_value, 1);
    const bool unloaded_rejected =
      net.Forward(owned_outputs) == InferenceWrapper::WrapperError && owned_outputs.empty() &&
      net.ForwardViews(inspirecv::Image{}, views) == InferenceWrapper::WrapperError && views.empty();

    inspire::InspireModel model;
    double initialization_ms = 0.0;
    if (!LoadInto(archive, "face_detect_160", net, model, initialization_ms)) {
        return false;
    }
    const auto& input = net.getMInputTensorInfoList().front();
    auto valid_image = images.front().Resize(input.image_info.width, input.image_info.height);
    auto wrong_size = images.front().Resize(input.image_info.width - 1, input.image_info.height);
    auto wrong_channel = inspirecv::Image::Create(input.image_info.width, input.image_info.height, 1);
    const bool invalid_images_rejected =
      net.ForwardViews(inspirecv::Image{}, views) == InferenceWrapper::WrapperError && views.empty() &&
      net.ForwardViews(wrong_size, views) == InferenceWrapper::WrapperError && views.empty() &&
      net.ForwardViews(wrong_channel, views) == InferenceWrapper::WrapperError && views.empty();

    net.getMInputTensorInfoList().front().data = nullptr;
    views.emplace_back(&stale_name, &stale_value, 1);
    const bool null_data_rejected = net.ForwardViews(views) == InferenceWrapper::WrapperError && views.empty();
    const bool valid_recovery = !valid_image.Empty() &&
                                net.ForwardViews(valid_image, views) == InferenceWrapper::WrapperOk && !views.empty();

    inspire::InspireModel corrupt_model;
    bool failed_reload_rejected = false;
    if (archive.LoadModel("face_detect_320", corrupt_model) == inspire::SARC_SUCCESS) {
        std::vector<char> corrupt_buffer(64, 0);
        corrupt_model.buffer = corrupt_buffer.data();
        corrupt_model.bufferSize = corrupt_buffer.size();
        corrupt_model.loadFilePath = 0;
        const bool load_rejected = net.LoadData(corrupt_model, corrupt_model.modelType) == InferenceWrapper::WrapperError;
        views.emplace_back(&stale_name, &stale_value, 1);
        failed_reload_rejected = load_rejected &&
                                 net.ForwardViews(valid_image, views) == InferenceWrapper::WrapperError && views.empty();
    }

    const bool passed = unloaded_rejected && invalid_images_rejected && null_data_rejected && valid_recovery &&
                        failed_reload_rejected;
    std::cout << "ANYNET_RUNTIME_BOUNDARY,cases=8,unloaded=" << (unloaded_rejected ? "PASS" : "FAIL")
              << ",image-shape=" << (invalid_images_rejected ? "PASS" : "FAIL")
              << ",null-data=" << (null_data_rejected ? "PASS" : "FAIL")
              << ",recovery=" << (valid_recovery ? "PASS" : "FAIL")
              << ",failed-reload=" << (failed_reload_rejected ? "PASS" : "FAIL")
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <pack_path> [test_res_root] [iterations] [single|reload|full]\n";
        return 2;
    }
    const std::string pack_path = argv[1];
    const std::string test_root = argc >= 3 ? argv[2] : "test_res";
    const int iterations = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 10;
    const std::string mode = argc >= 5 ? argv[4] : "full";
    if (mode != "single" && mode != "reload" && mode != "full") {
        std::cerr << "Mode must be single, reload, or full\n";
        return 2;
    }
    if (HFLaunchInspireFace(pack_path.c_str()) != HSUCCEED) {
        return 3;
    }

    std::vector<inspirecv::Image> images;
    bool passed = LoadImages(test_root, images);
    if (passed) {
        auto& archive = inspire::Launch::GetInstance()->getMArchive();
        passed = RunMetadataBoundaryGate(archive) && passed;
        passed = RunRuntimeBoundaryGate(archive, images) && passed;
        if (mode == "single" || mode == "full") {
            passed = RunSingleGate(archive, images, iterations) && passed;
        }
        if (mode == "reload" || mode == "full") {
            passed = RunReloadGate(archive, images, iterations) && passed;
            passed = RunFailureRecoveryGate(archive, images, iterations) && passed;
        }
    }
    if (HFTerminateInspireFace() != HSUCCEED) {
        passed = false;
    }
    std::cout << "SUMMARY,mode=" << mode << ",images=3,iterations=" << iterations
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
