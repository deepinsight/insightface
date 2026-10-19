#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <inspireface/include/inspireface/inspireface.hpp>
#include <inspireface/include/inspireface/meta.h>
#include "middleware/inference_adapter/inference_adapter.h"
#include "middleware/inference_wrapper/inference_wrapper.h"
#include "middleware/thread/resource_pool.h"
#include "middleware/utils.h"

#include "settings/test_settings.h"

namespace {

class SimilarityStateReset {
public:
    SimilarityStateReset() : state_(inspire::SimilarityConverter::getInstance().getState()) {}
    ~SimilarityStateReset() {
        inspire::SimilarityConverter::getInstance().updateConfigAndRecommendedThreshold(
          state_.config, state_.recommendedCosineThreshold);
    }

private:
    inspire::SimilarityConverterState state_;
};

}  // namespace

TEST_CASE("C++ SimilarityConverter validates atomically and remains monotonic", "[cpp_api][contract][similarity_converter]") {
    SimilarityStateReset reset;
    auto& converter = inspire::SimilarityConverter::getInstance();
    const inspire::SimilarityConverterConfig replacement = {0.35, 0.55, 6.0, 0.02, 0.98};
    REQUIRE(converter.updateConfigAndRecommendedThreshold(replacement, 0.42f));

    const auto state = converter.getState();
    CHECK(state.config.threshold == Approx(replacement.threshold));
    CHECK(state.config.middleScore == Approx(replacement.middleScore));
    CHECK(state.config.steepness == Approx(replacement.steepness));
    CHECK(state.config.outputMin == Approx(replacement.outputMin));
    CHECK(state.config.outputMax == Approx(replacement.outputMax));
    CHECK(state.recommendedCosineThreshold == Approx(0.42f));

    const double low = converter.convert(-1.0f);
    const double middle = converter.convert(replacement.threshold);
    const double high = converter.convert(1.0f);
    CHECK(low < middle);
    CHECK(middle < high);
    CHECK(low >= replacement.outputMin);
    CHECK(high <= replacement.outputMax);

    auto invalid = replacement;
    invalid.steepness = 0.0;
    CHECK_FALSE(converter.updateConfig(invalid));
    invalid = replacement;
    invalid.middleScore = invalid.outputMin;
    CHECK_FALSE(converter.updateConfig(invalid));
    CHECK_FALSE(converter.setRecommendedCosineThreshold(std::numeric_limits<float>::quiet_NaN()));

    const auto after_invalid = converter.getState();
    CHECK(after_invalid.config.threshold == Approx(replacement.threshold));
    CHECK(after_invalid.config.steepness == Approx(replacement.steepness));
    CHECK(after_invalid.recommendedCosineThreshold == Approx(0.42f));

    const auto config_copy = converter.getConfig();
    CHECK(config_copy.threshold == Approx(replacement.threshold));
    CHECK(converter.getRecommendedCosineThreshold() == Approx(0.42f));
    CHECK(converter.setRecommendedCosineThreshold(0.41f));
    CHECK(converter.getRecommendedCosineThreshold() == Approx(0.41f));
    CHECK(inspire::SimilarityConverter::IsConfigValid(replacement));

    inspire::SimilarityConverter local(invalid);
    CHECK(inspire::SimilarityConverter::IsConfigValid(local.getConfig()));
    const auto destroy_instance = &inspire::SimilarityConverter::destroyInstance;
    CHECK(destroy_instance != nullptr);
}

TEST_CASE("C++ version information and SpendTimer expose coherent values", "[cpp_api][contract][utility]") {
    REQUIRE(GetInspireFaceVersionMajorStr() != nullptr);
    REQUIRE(GetInspireFaceVersionMinorStr() != nullptr);
    REQUIRE(GetInspireFaceVersionPatchStr() != nullptr);
    REQUIRE(GetInspireFaceExtendedInformation() != nullptr);
    CHECK_FALSE(std::string(GetInspireFaceVersionMajorStr()).empty());
    CHECK_FALSE(std::string(GetInspireFaceVersionMinorStr()).empty());
    CHECK_FALSE(std::string(GetInspireFaceVersionPatchStr()).empty());
    CHECK_FALSE(std::string(GetInspireFaceExtendedInformation()).empty());

    const auto& sdk = inspire::GetSDKInfo();
    CHECK(sdk.GetVersionMajorStr() == GetInspireFaceVersionMajorStr());
    CHECK(sdk.GetVersionMinorStr() == GetInspireFaceVersionMinorStr());
    CHECK(sdk.GetVersionPatchStr() == GetInspireFaceVersionPatchStr());
    CHECK(sdk.GetVersionString() == sdk.GetVersionMajorStr() + "." + sdk.GetVersionMinorStr() + "." + sdk.GetVersionPatchStr());
    CHECK_FALSE(sdk.GetFullVersionInfo().empty());

    static const std::array<const char*, static_cast<size_t>(inspire::ComponentType::COUNT)> component_names = {{
      "mnn", "inspirecv", "eigen", "sqlite", "sqlite_vec", "nlohmann_json",
      "opencv", "tensorrt", "cuda", "rknn", "rga", "coreml",
    }};
    const std::string component_versions = inspire::GetComponentVersionsString();
    CHECK(component_versions.find("inspireface=" + sdk.GetVersionString()) == 0);
    for (size_t index = 0; index < component_names.size(); ++index) {
        const auto component = static_cast<inspire::ComponentType>(index);
        CHECK(std::string(inspire::GetComponentName(component)) == component_names[index]);
        const auto component_version = inspire::GetComponentVersion(component);
        CHECK(component_versions.find(std::string(component_names[index]) + "=" + component_version.GetVersionString()) != std::string::npos);
        CHECK(component_version.IsEnabled() == (component_version.state != inspire::ComponentVersionState::DISABLED));
        CHECK(component_version.IsVersionKnown() == (component_version.state == inspire::ComponentVersionState::KNOWN));
        if (index <= static_cast<size_t>(inspire::ComponentType::NLOHMANN_JSON)) {
            CHECK(component_version.IsVersionKnown());
        }
    }
    CHECK(inspire::GetComponentName(static_cast<inspire::ComponentType>(-1)) == nullptr);
    CHECK(inspire::GetComponentName(inspire::ComponentType::COUNT) == nullptr);
    CHECK(inspire::GetDiagnosticInfo().find("\nComponents: " + component_versions) != std::string::npos);
#if !defined(ISF_ENABLE_APPLE_EXTENSION) && (!defined(TARGET_OS_IOS) || !TARGET_OS_IOS)
    CHECK(inspire::GetComponentVersion(inspire::ComponentType::COREML).state == inspire::ComponentVersionState::DISABLED);
#endif

    constexpr size_t query_thread_count = 4;
    constexpr size_t queries_per_thread = 200;
    std::array<bool, query_thread_count> query_results = {{false, false, false, false}};
    std::array<std::thread, query_thread_count> query_threads;
    const auto query_start = std::chrono::steady_clock::now();
    for (size_t thread_index = 0; thread_index < query_thread_count; ++thread_index) {
        query_threads[thread_index] = std::thread([&, thread_index]() {
            bool matched = true;
            for (size_t iteration = 0; iteration < queries_per_thread; ++iteration) {
                matched = matched && inspire::GetComponentVersionsString() == component_versions;
            }
            query_results[thread_index] = matched;
        });
    }
    for (auto& thread : query_threads) {
        thread.join();
    }
    CHECK(std::all_of(query_results.begin(), query_results.end(), [](bool matched) { return matched; }));
    CHECK(std::chrono::steady_clock::now() - query_start < std::chrono::seconds(2));

    const uint64_t before = inspire::_now();
    inspire::SpendTimer timer("contract");
    CHECK(timer.name() == "contract");
    CHECK(timer.Count() == 0);
    CHECK(timer.Average() == 0);
    CHECK(timer.Min() == 0);
    CHECK(timer.Max() == 0);

    timer.Start();
    timer.Stop();
    const uint64_t after = inspire::_now();
    CHECK(after >= before);
    CHECK(timer.Count() == 1);
    CHECK(timer.Total() == timer.Get());
    CHECK(timer.Average() == timer.Total());
    CHECK(timer.Min() == timer.Get());
    CHECK(timer.Max() == timer.Get());
    CHECK(timer.Report().find("contract") != std::string::npos);

    std::ostringstream report;
    report << timer;
    CHECK(report.str() == timer.Report());

    timer.Reset();
    CHECK(timer.Count() == 0);
    CHECK(timer.Total() == 0);
    CHECK(timer.Min() == 0);
    CHECK(timer.Max() == 0);

    inspire::SpendTimer unnamed;
    CHECK(unnamed.name().empty());
    const auto disable_timer = &inspire::SpendTimer::Disable;
    CHECK(disable_timer != nullptr);
}

TEST_CASE("C++ LogManager level changes round-trip without emitting suppressed logs", "[cpp_api][contract][log]") {
    auto* logger = inspire::LogManager::getInstance();
    REQUIRE(logger != nullptr);
    CHECK(logger == inspire::LogManager::getInstance());
    const auto original = logger->getLogLevel();

    logger->setLogLevel(inspire::ISF_LOG_NONE);
    CHECK(logger->getLogLevel() == inspire::ISF_LOG_NONE);
#ifdef ANDROID
    logger->logAndroid(inspire::ISF_LOG_INFO, "InspireFaceTest", "suppressed");
#else
    logger->logStandard(inspire::ISF_LOG_INFO, "", "", -1, "suppressed");
#endif
    logger->setLogLevel(original);
    CHECK(logger->getLogLevel() == original);

    std::vector<std::thread> workers;
    for (int worker = 0; worker < 4; ++worker) {
        workers.emplace_back([logger, worker]() {
            for (int iteration = 0; iteration < 1000; ++iteration) {
                logger->setLogLevel((iteration + worker) % 2 == 0 ? inspire::ISF_LOG_NONE : inspire::ISF_LOG_ERROR);
                (void)logger->getLogLevel();
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    logger->setLogLevel(original);
}

TEST_CASE("C++ ResourcePool keeps checked-out references stable while growing", "[cpp_api][contract][resource_pool][lifetime]") {
    inspire::parallel::ResourcePool<int> pool(1);
    pool.AddResource(7);
    {
        auto first = pool.AcquireResource();
        REQUIRE(*first == 7);
        for (int value = 1; value <= 128; ++value) {
            pool.AddResource(int(value));
        }
        CHECK(*first == 7);
        CHECK(pool.TotalCount() == 129);
        CHECK(pool.AvailableCount() == 128);
    }
    CHECK(pool.AvailableCount() == 129);

    int sum = 0;
    std::vector<std::unique_ptr<inspire::parallel::ResourcePool<int>::ResourceGuard>> guards;
    guards.reserve(pool.TotalCount());
    while (auto guard = pool.TryAcquireResource()) {
        sum += **guard;
        guards.push_back(std::move(guard));
    }
    CHECK(sum == 7 + (128 * 129) / 2);
    CHECK(pool.AvailableCount() == 0);
    guards.clear();
    CHECK(pool.AvailableCount() == 129);
}

TEST_CASE("C++ utility filters reject degenerate inputs without changing valid results", "[cpp_api][contract][utility][boundary]") {
    std::vector<float> history;
    CHECK(inspire::EmaFilter(0.5f, history, 0) == Approx(0.5f));
    CHECK(history.empty());
    CHECK(inspire::EmaFilter(0.25f, history, 3, 0.5f) == Approx(0.25f));
    CHECK(inspire::EmaFilter(0.75f, history, 3, 0.5f) == Approx(0.5f));

    std::vector<std::vector<float>> vector_history{{1.0f}};
    const std::vector<float> current{0.0f, 1.0f};
    CHECK(inspire::VectorEmaFilter(current, vector_history, 3, 0.5f) == current);
    CHECK(vector_history.size() == 1);
    CHECK(inspire::VectorEmaFilter(current, vector_history, 0, 0.5f) == current);
    CHECK(vector_history.empty());

    const inspirecv::Rect2i box(0, 0, 16, 8);
    CHECK_FALSE(inspire::isShortestSideGreaterThan<int>(box, 1, 0.0f));
    CHECK(inspire::AlignmentBoxToStrideSquareBox(box, 0).GetWidth() == 0);
    CHECK(inspire::GetNewBox(100, 100, inspirecv::Rect2i(1, 1, 0, 10), 2.0f).GetWidth() == 0);
    std::vector<inspirecv::Point2f> empty_points;
    CHECK(inspire::FixPointsMeanshape(empty_points, empty_points).empty());
}

TEST_CASE("Internal inference adapter output copies validate pointer ownership", "[cpp_api][contract][inference_adapter][boundary]") {
    XOutputData empty;
    CHECK(empty.CopyToFloatArray().empty());

    float values[] = {1.0f, 2.0f, 3.0f};
    XOutputData view;
    view.size = 3;
    view.data = values;
    CHECK(view.CopyToFloatArray() == std::vector<float>{1.0f, 2.0f, 3.0f});

    view.buffer = {4.0f, 5.0f};
    view.data = nullptr;
    CHECK(view.CopyToFloatArray() == view.buffer);

    XOutputData invalid;
    invalid.size = 1;
    CHECK_THROWS_AS(invalid.CopyToFloatArray(), std::invalid_argument);
}

TEST_CASE("Inference tensor metadata rejects invalid shapes for every data type",
          "[cpp_api][contract][inference_wrapper][boundary]") {
    TensorInfo tensor;
    CHECK(tensor.GetElementNum() == -1);
    tensor.tensor_dims = {2, 3, 4};
    CHECK(tensor.GetElementNum() == 24);
    CHECK(tensor.GetBatch() == 2);
    tensor.tensor_dims = {2, 0, 4};
    CHECK(tensor.GetElementNum() == -1);
    tensor.tensor_dims = {std::numeric_limits<int32_t>::max(), 2};
    CHECK(tensor.GetElementNum() == -1);

    float float_value = 3.5f;
    OutputTensorInfo floating("float", TensorInfo::TensorTypeFp32);
    floating.data = &float_value;
    CHECK(floating.GetDataAsFloat() == nullptr);
    floating.tensor_dims = {1};
    REQUIRE(floating.GetDataAsFloat() != nullptr);
    CHECK(floating.GetDataAsFloat()[0] == Approx(3.5f));

    std::array<uint8_t, 3> unsigned_values = {{0, 2, 4}};
    OutputTensorInfo unsigned_quantized("uint8", TensorInfo::TensorTypeUint8);
    unsigned_quantized.tensor_dims = {3};
    unsigned_quantized.data = unsigned_values.data();
    unsigned_quantized.quant = {0.5f, 2};
    const float* converted_unsigned = unsigned_quantized.GetDataAsFloat();
    REQUIRE(converted_unsigned != nullptr);
    CHECK(converted_unsigned[0] == Approx(-1.0f));
    CHECK(converted_unsigned[1] == Approx(0.0f));
    CHECK(converted_unsigned[2] == Approx(1.0f));

    std::array<int8_t, 3> signed_values = {{-2, 0, 2}};
    OutputTensorInfo signed_quantized("int8", TensorInfo::TensorTypeInt8);
    signed_quantized.tensor_dims = {1, 3};
    signed_quantized.data = signed_values.data();
    signed_quantized.quant = {0.25f, 0};
    const float* converted_signed = signed_quantized.GetDataAsFloat();
    REQUIRE(converted_signed != nullptr);
    CHECK(converted_signed[0] == Approx(-0.5f));
    CHECK(converted_signed[1] == Approx(0.0f));
    CHECK(converted_signed[2] == Approx(0.5f));

    signed_quantized.tensor_dims.clear();
    CHECK(signed_quantized.GetDataAsFloat() == nullptr);
}
