#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <inspirecv/inspirecv.h>
#include <inspireface/herror.h>

#include "image_process/nexus_processor/image_processor_general.h"
#include "middleware/any_net_adapter.h"
#include "pipeline_module/attribute/mask_predict_adapt.h"
#include "pipeline_module/liveness/rgb_anti_spoofing_adapt.h"
#include "track_module/face_detect/face_detect_adapt.h"
#include "track_module/quality/face_pose_quality_adapt.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kLatencyRatioLimit = 1.20;
constexpr double kLatencyAllowanceMs = 0.03;

struct ImageCase {
    int width;
    int height;
    std::vector<uint8_t> pixels;
};

struct TimingSummary {
    double median_ms = 0.0;
    double p95_ms = 0.0;
};

double Percentile(std::vector<double> values, double percentile) {
    std::sort(values.begin(), values.end());
    const double position = percentile * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

TimingSummary Summarize(const std::vector<double>& values) {
    return {Percentile(values, 0.50), Percentile(values, 0.95)};
}

std::vector<ImageCase> MakeImages() {
    const std::vector<std::pair<int, int>> sizes = {{37, 29}, {96, 64}, {127, 91}, {160, 160}};
    std::vector<ImageCase> images;
    images.reserve(sizes.size());
    for (size_t case_index = 0; case_index < sizes.size(); ++case_index) {
        ImageCase image{sizes[case_index].first, sizes[case_index].second, {}};
        image.pixels.resize(static_cast<size_t>(image.width) * image.height * 3);
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                const size_t offset = (static_cast<size_t>(y) * image.width + x) * 3;
                image.pixels[offset] = static_cast<uint8_t>((x * 17 + y * 3 + case_index * 29) & 0xff);
                image.pixels[offset + 1] = static_cast<uint8_t>(((x / 3 + y / 2 + case_index) & 1) ? 241 : 13);
                image.pixels[offset + 2] = static_cast<uint8_t>((x * y + case_index * 71) & 0xff);
            }
        }
        images.push_back(std::move(image));
    }
    return images;
}

bool BytesEqual(const uint8_t* actual, const inspirecv::Image& expected) {
    if (actual == nullptr || expected.Empty()) {
        return false;
    }
    const size_t bytes = static_cast<size_t>(expected.Width()) * expected.Height() * expected.Channels();
    return std::memcmp(actual, expected.Data(), bytes) == 0;
}

class FaultingProcessor final : public inspire::nexus::ImageProcessor {
public:
    explicit FaultingProcessor(int32_t operation_status = -73, int32_t done_status = 0)
    : operation_status_(operation_status), done_status_(done_status) {}

    int32_t Resize(const uint8_t*, int, int, int, uint8_t** dst_data, int, int) override {
        ++resize_calls;
        return Fail(dst_data);
    }

    int32_t SwapColor(const uint8_t*, int, int, int, uint8_t** dst_data) override {
        return Fail(dst_data);
    }

    int32_t Padding(const uint8_t*, int, int, int, int, int, int, int, uint8_t** dst_data, int& dst_width,
                    int& dst_height) override {
        dst_width = 0;
        dst_height = 0;
        return Fail(dst_data);
    }

    int32_t ResizeAndPadding(const uint8_t*, int, int, int, int, int, uint8_t** dst_data, float& scale) override {
        ++resize_and_pad_calls;
        scale = 0.0f;
        return Fail(dst_data);
    }

    int32_t MarkDone() override {
        ++mark_done_calls;
        return done_status_;
    }

    void DumpCacheStatus() const override {}
    int32_t GetAlignedWidth(int width) const override { return width; }
    void SetAlignedWidth(int) override {}

    int resize_calls = 0;
    int resize_and_pad_calls = 0;
    int mark_done_calls = 0;

private:
    int32_t Fail(uint8_t** dst_data) const {
        if (dst_data != nullptr) {
            *dst_data = nullptr;
        }
        return operation_status_;
    }

    int32_t operation_status_;
    int32_t done_status_;
};

class PreprocessProbe final : public inspire::AnyNetAdapter {
public:
    PreprocessProbe() : AnyNetAdapter("ImagePreprocessGuard") {}

    void SetProcessor(std::unique_ptr<inspire::nexus::ImageProcessor> processor) {
        m_processor_ = std::move(processor);
    }

    int32_t Resize(const inspirecv::Image& source, int width, int height, inspirecv::Image& output) {
        return ResizeImageForInference(source, width, height, output);
    }

    int32_t ResizeAndPad(const inspirecv::Image& source, int width, int height, inspirecv::Image& output, float& scale) {
        return ResizeAndPadImageForInference(source, width, height, output, scale);
    }
};

template <typename Adapter>
class AdapterProbe final : public Adapter {
public:
    using Adapter::Adapter;

    void SetProcessor(std::unique_ptr<inspire::nexus::ImageProcessor> processor) {
        this->m_processor_ = std::move(processor);
    }
};

bool TestGeneralProcessorContract() {
    inspire::nexus::GeneralImageProcessor processor;
    uint8_t* output = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(1));
    float scale = 7.0f;
    int width = 9;
    int height = 9;
    bool exact = processor.Resize(nullptr, 10, 10, 3, &output, 8, 8) != 0 && output == nullptr;
    output = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(1));
    exact = exact && processor.Resize(nullptr, 10, 10, 3, nullptr, 8, 8) != 0;
    exact = exact && processor.ResizeAndPadding(nullptr, 10, 10, 3, 8, 8, &output, scale) != 0 && output == nullptr && scale == 0.0f;
    output = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(1));
    exact = exact && processor.Padding(nullptr, 10, 10, 3, 1, 1, 1, 1, &output, width, height) != 0 && output == nullptr && width == 0 &&
            height == 0;
    uint8_t pixel[3] = {0, 0, 0};
    output = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(1));
    width = 9;
    height = 9;
    exact = exact && processor.Padding(pixel, std::numeric_limits<int>::max(), 1, 3, 0, 0, 0, 1, &output, width, height) != 0 &&
            output == nullptr && width == 0 && height == 0;
    return exact;
}

bool TestCpuAccuracy(const std::vector<ImageCase>& cases) {
    inspire::nexus::GeneralImageProcessor processor;
    for (const auto& test : cases) {
        const auto source = inspirecv::Image::Create(test.width, test.height, 3, test.pixels.data(), false);
        uint8_t* output = nullptr;
        if (processor.Resize(source.Data(), source.Width(), source.Height(), 3, &output, 96, 80) != 0 ||
            !BytesEqual(output, source.Resize(96, 80))) {
            return false;
        }
        if (processor.MarkDone() != 0) {
            return false;
        }

        output = nullptr;
        if (processor.SwapColor(source.Data(), source.Width(), source.Height(), 3, &output) != 0 ||
            !BytesEqual(output, source.SwapRB())) {
            return false;
        }

        const int top = 3;
        const int bottom = 5;
        const int left = 7;
        const int right = 9;
        int output_width = 0;
        int output_height = 0;
        output = nullptr;
        if (processor.Padding(source.Data(), source.Width(), source.Height(), 3, top, bottom, left, right, &output,
                              output_width, output_height) != 0 || output_width != source.Width() + left + right ||
            output_height != source.Height() + top + bottom ||
            !BytesEqual(output, source.Pad(top, bottom, left, right, inspirecv::Color::Black))) {
            return false;
        }

        output = nullptr;
        float scale = 0.0f;
        if (processor.ResizeAndPadding(source.Data(), source.Width(), source.Height(), 3, 160, 160, &output, scale) != 0) {
            return false;
        }
        const float expected_scale = std::min(160.0f / source.Width(), 160.0f / source.Height());
        const int resized_width = static_cast<int>(source.Width() * expected_scale);
        const int resized_height = static_cast<int>(source.Height() * expected_scale);
        const auto expected = source.Resize(resized_width, resized_height)
                                .Pad(0, 160 - resized_height, 0, 160 - resized_width, inspirecv::Color::Black);
        if (scale != expected_scale || !BytesEqual(output, expected)) {
            return false;
        }
    }
    return true;
}

bool TestFaultPropagation(const ImageCase& test) {
    const auto image = inspirecv::Image::Create(test.width, test.height, 3, test.pixels.data(), false);

    PreprocessProbe helper_probe;
    auto helper_fault = std::unique_ptr<FaultingProcessor>(new FaultingProcessor());
    FaultingProcessor* helper_fault_ptr = helper_fault.get();
    helper_probe.SetProcessor(std::move(helper_fault));
    inspirecv::Image resized;
    float scale = 1.0f;
    if (helper_probe.Resize(image, 96, 96, resized) != InferenceWrapper::WrapperError || !resized.Empty() ||
        helper_probe.ResizeAndPad(image, 160, 160, resized, scale) != InferenceWrapper::WrapperError || !resized.Empty() || scale != 0.0f ||
        helper_fault_ptr->resize_calls != 1 || helper_fault_ptr->resize_and_pad_calls != 1) {
        return false;
    }

    AdapterProbe<inspire::FaceDetectAdapt> detector;
    auto detector_fault = std::unique_ptr<FaultingProcessor>(new FaultingProcessor());
    FaultingProcessor* detector_fault_ptr = detector_fault.get();
    detector.SetProcessor(std::move(detector_fault));
    inspire::FaceLocList detections(1);
    if (detector.Detect(image, detections) != HERR_DEVICE_IMAGE_PROCESS_FAILURE || !detections.empty() ||
        detector_fault_ptr->resize_and_pad_calls != 1 || detector_fault_ptr->mark_done_calls != 1) {
        return false;
    }

    AdapterProbe<inspire::MaskPredictAdapt> mask;
    mask.SetProcessor(std::unique_ptr<FaultingProcessor>(new FaultingProcessor()));
    float score = 0.0f;
    if (mask.Predict(image, score) != HERR_DEVICE_IMAGE_PROCESS_FAILURE || !std::isnan(score)) {
        return false;
    }

    AdapterProbe<inspire::RBGAntiSpoofingAdapt> liveness(112, false);
    liveness.SetProcessor(std::unique_ptr<FaultingProcessor>(new FaultingProcessor()));
    if (liveness.Predict(image, score) != HERR_DEVICE_IMAGE_PROCESS_FAILURE || !std::isnan(score)) {
        return false;
    }

    AdapterProbe<inspire::FacePoseQualityAdapt> quality;
    quality.SetProcessor(std::unique_ptr<FaultingProcessor>(new FaultingProcessor()));
    inspire::FacePoseQualityAdaptResult quality_result;
    quality_result.lmk.resize(5);
    if (quality.Predict(image, quality_result) != HERR_DEVICE_IMAGE_PROCESS_FAILURE || !quality_result.lmk.empty() ||
        !quality_result.lmk_quality.empty()) {
        return false;
    }
    return true;
}

TimingSummary BenchmarkDirect(const ImageCase& test, int iterations) {
    inspire::nexus::GeneralImageProcessor processor;
    const auto source = inspirecv::Image::Create(test.width, test.height, 3, test.pixels.data(), false);
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        uint8_t* output = nullptr;
        const auto begin = Clock::now();
        const int32_t status = processor.Resize(source.Data(), source.Width(), source.Height(), 3, &output, 96, 96);
        const auto end = Clock::now();
        if (status != 0 || output == nullptr) {
            return {};
        }
        samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return Summarize(samples);
}

TimingSummary BenchmarkGuarded(const ImageCase& test, int iterations) {
    PreprocessProbe probe;
    probe.SetProcessor(std::unique_ptr<inspire::nexus::ImageProcessor>(new inspire::nexus::GeneralImageProcessor()));
    const auto source = inspirecv::Image::Create(test.width, test.height, 3, test.pixels.data(), false);
    std::vector<double> samples;
    samples.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        inspirecv::Image output;
        const auto begin = Clock::now();
        const int32_t status = probe.Resize(source, 96, 96, output);
        const auto end = Clock::now();
        if (status != InferenceWrapper::WrapperOk || output.Empty()) {
            return {};
        }
        samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    return Summarize(samples);
}

}  // namespace

int main() {
    const auto images = MakeImages();
    const bool contract_ok = TestGeneralProcessorContract();
    const bool accuracy_ok = TestCpuAccuracy(images);
    const bool propagation_ok = TestFaultPropagation(images.front());

    constexpr int kIterations = 500;
    const TimingSummary baseline = BenchmarkDirect(images[2], kIterations);
    const TimingSummary guarded = BenchmarkGuarded(images[2], kIterations);
    const bool timing_ok = baseline.median_ms > 0.0 && guarded.median_ms > 0.0 &&
                           guarded.median_ms <= baseline.median_ms * kLatencyRatioLimit + kLatencyAllowanceMs;

    std::cout << std::fixed << std::setprecision(6)
              << "images=" << images.size() << '\n'
              << "contract=" << (contract_ok ? "PASS" : "FAIL") << '\n'
              << "pixel_exact=" << (accuracy_ok ? "PASS" : "FAIL") << '\n'
              << "fault_propagation=" << (propagation_ok ? "PASS" : "FAIL") << '\n'
              << "baseline_direct_p50_ms=" << baseline.median_ms << '\n'
              << "baseline_direct_p95_ms=" << baseline.p95_ms << '\n'
              << "guarded_p50_ms=" << guarded.median_ms << '\n'
              << "guarded_p95_ms=" << guarded.p95_ms << '\n'
              << "latency_gate=" << (timing_ok ? "PASS" : "FAIL") << '\n';

    return contract_ok && accuracy_ok && propagation_ok && timing_ok ? 0 : 1;
}
