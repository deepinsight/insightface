#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <inspireface.h>

namespace {

using Clock = std::chrono::steady_clock;

struct StreamHandle {
    HFImageStream value = nullptr;
    ~StreamHandle() {
        if (value != nullptr) HFReleaseImageStream(value);
    }
};

struct BitmapHandle {
    HFImageBitmap value = nullptr;
    ~BitmapHandle() {
        if (value != nullptr) HFReleaseImageBitmap(value);
    }
};

struct ImageCase {
    int width;
    int height;
    std::vector<uint8_t> pixels;
};

struct Output {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> pixels;
};

struct Timing {
    double p50_us = 0.0;
    double p95_us = 0.0;
};

std::vector<ImageCase> MakeCases() {
    const std::array<std::pair<int, int>, 3> sizes = {{{8, 6}, {32, 24}, {96, 64}}};
    std::vector<ImageCase> cases;
    for (size_t case_index = 0; case_index < sizes.size(); ++case_index) {
        ImageCase test{sizes[case_index].first, sizes[case_index].second, {}};
        test.pixels.resize(static_cast<size_t>(test.width) * test.height * 3);
        for (int y = 0; y < test.height; ++y) {
            for (int x = 0; x < test.width; ++x) {
                const size_t offset = (static_cast<size_t>(y) * test.width + x) * 3;
                test.pixels[offset] = static_cast<uint8_t>((x * 31 + y * 7 + case_index * 43) & 0xff);
                test.pixels[offset + 1] = static_cast<uint8_t>((x * 3 + y * 29 + case_index * 17) & 0xff);
                test.pixels[offset + 2] = static_cast<uint8_t>((x * y + x + case_index * 61) & 0xff);
            }
        }
        cases.push_back(std::move(test));
    }
    return cases;
}

Output RotateExpected(const ImageCase& source, HFRotation rotation) {
    const bool swaps = rotation == HF_CAMERA_ROTATION_90 || rotation == HF_CAMERA_ROTATION_270;
    Output output;
    output.width = swaps ? source.height : source.width;
    output.height = swaps ? source.width : source.height;
    output.channels = 3;
    output.pixels.resize(static_cast<size_t>(output.width) * output.height * output.channels);
    for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
            int source_x = x;
            int source_y = y;
            if (rotation == HF_CAMERA_ROTATION_90) {
                source_x = source.width - 1 - y;
                source_y = x;
            } else if (rotation == HF_CAMERA_ROTATION_180) {
                source_x = source.width - 1 - x;
                source_y = source.height - 1 - y;
            } else if (rotation == HF_CAMERA_ROTATION_270) {
                source_x = y;
                source_y = source.height - 1 - x;
            }
            const size_t source_offset = (static_cast<size_t>(source_y) * source.width + source_x) * 3;
            const size_t output_offset = (static_cast<size_t>(y) * output.width + x) * 3;
            std::memcpy(output.pixels.data() + output_offset, source.pixels.data() + source_offset, 3);
        }
    }
    return output;
}

bool Process(HFImageStream stream, bool rotate, float scale, Output& output) {
    BitmapHandle bitmap;
    if (HFCreateImageBitmapFromImageStreamProcess(stream, &bitmap.value, rotate ? 1 : 0, scale) != HSUCCEED) {
        return false;
    }
    HFImageBitmapData data = {};
    if (HFImageBitmapGetData(bitmap.value, &data) != HSUCCEED || data.data == nullptr || data.width <= 0 ||
        data.height <= 0 || data.channels <= 0) {
        return false;
    }
    output.width = data.width;
    output.height = data.height;
    output.channels = data.channels;
    output.pixels.assign(data.data, data.data + static_cast<size_t>(data.width) * data.height * data.channels);
    return true;
}

bool Equal(const Output& left, const Output& right) {
    return left.width == right.width && left.height == right.height && left.channels == right.channels &&
           left.pixels == right.pixels;
}

bool CreateStreams(const ImageCase& test, HFRotation rotation, StreamHandle& raw, StreamHandle& snapshot) {
    HFImageData raw_data = {const_cast<uint8_t*>(test.pixels.data()), test.width, test.height, HF_STREAM_BGR, rotation};
    if (HFCreateImageStream(&raw_data, &raw.value) != HSUCCEED) {
        return false;
    }

    HFImageBitmapData bitmap_data = {
      const_cast<uint8_t*>(test.pixels.data()), test.width, test.height, 3};
    BitmapHandle bitmap;
    if (HFCreateImageBitmap(&bitmap_data, &bitmap.value) != HSUCCEED ||
        HFCreateImageStreamFromImageBitmap(bitmap.value, rotation, &snapshot.value) != HSUCCEED) {
        return false;
    }
    HFImageBitmapData mutable_data = {};
    if (HFImageBitmapGetData(bitmap.value, &mutable_data) != HSUCCEED) {
        return false;
    }
    std::fill(mutable_data.data,
              mutable_data.data + static_cast<size_t>(mutable_data.width) * mutable_data.height * mutable_data.channels,
              0);
    return HFReleaseImageBitmap(bitmap.value) == HSUCCEED && (bitmap.value = nullptr, true);
}

double Percentile(std::vector<double> samples, double percentile) {
    std::sort(samples.begin(), samples.end());
    const double position = percentile * static_cast<double>(samples.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, samples.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return samples[lower] * (1.0 - fraction) + samples[upper] * fraction;
}

Timing Summarize(const std::vector<double>& samples) {
    return {Percentile(samples, 0.50), Percentile(samples, 0.95)};
}

bool TimeProcess(HFImageStream stream, int iterations, std::vector<double>& samples) {
    samples.reserve(samples.size() + static_cast<size_t>(iterations));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        HFImageBitmap bitmap = nullptr;
        const auto begin = Clock::now();
        const HResult create_status = HFCreateImageBitmapFromImageStreamProcess(stream, &bitmap, 1, 0.75f);
        const auto end = Clock::now();
        if (create_status != HSUCCEED || bitmap == nullptr) {
            return false;
        }
        samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        if (HFReleaseImageBitmap(bitmap) != HSUCCEED) {
            return false;
        }
    }
    return true;
}

bool AccuracyGate(const std::vector<ImageCase>& cases) {
    const std::array<HFRotation, 4> rotations = {{HF_CAMERA_ROTATION_0, HF_CAMERA_ROTATION_90,
                                                  HF_CAMERA_ROTATION_180, HF_CAMERA_ROTATION_270}};
    bool passed = true;
    for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
        for (const HFRotation rotation : rotations) {
            StreamHandle raw;
            StreamHandle snapshot;
            Output raw_output;
            Output snapshot_output;
            Output repeated_output;
            const Output expected = RotateExpected(cases[case_index], rotation);
            const bool exact = CreateStreams(cases[case_index], rotation, raw, snapshot) &&
                               Process(raw.value, true, 1.0f, raw_output) &&
                               Process(snapshot.value, true, 1.0f, snapshot_output) &&
                               Process(snapshot.value, true, 1.0f, repeated_output) &&
                               Equal(raw_output, expected) && Equal(snapshot_output, expected) &&
                               Equal(repeated_output, expected);
            std::cout << "FRAME_PROCESS_ACCURACY,case=" << case_index << ",width=" << cases[case_index].width
                      << ",height=" << cases[case_index].height << ",rotation=" << static_cast<int>(rotation)
                      << ",exact=" << (exact ? "PASS" : "FAIL") << '\n';
            passed = exact && passed;
        }
    }
    return passed;
}

bool BoundaryGate() {
    uint8_t pixel = 0;
    HFImageStream stream = reinterpret_cast<HFImageStream>(static_cast<uintptr_t>(1));
    HFImageData oversized = {
      &pixel, std::numeric_limits<HInt32>::max(), 2, HF_STREAM_BGRA, HF_CAMERA_ROTATION_0};
    bool passed = HFCreateImageStream(&oversized, &stream) == HERR_INVALID_IMAGE_STREAM_PARAM && stream == nullptr;

    StreamHandle empty;
    passed = passed && HFCreateImageStreamEmpty(&empty.value) == HSUCCEED;
    HFImageBitmap output = reinterpret_cast<HFImageBitmap>(static_cast<uintptr_t>(1));
    passed = passed && HFCreateImageBitmapFromImageStreamProcess(empty.value, &output, 0, 1.0f) == HERR_INVALID_PARAM &&
             output == nullptr;
    std::cout << "FRAME_PROCESS_BOUNDARY,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool PerformanceGate(const ImageCase& test) {
    StreamHandle raw;
    StreamHandle snapshot;
    if (!CreateStreams(test, HF_CAMERA_ROTATION_90, raw, snapshot)) {
        return false;
    }
    std::vector<double> raw_samples;
    std::vector<double> snapshot_samples;
    for (int warmup = 0; warmup < 8; ++warmup) {
        std::vector<double> ignored;
        if (!TimeProcess(raw.value, 1, ignored) || !TimeProcess(snapshot.value, 1, ignored)) return false;
    }
    for (int block = 0; block < 8; ++block) {
        if ((block & 1) == 0) {
            if (!TimeProcess(raw.value, 20, raw_samples) || !TimeProcess(snapshot.value, 20, snapshot_samples)) return false;
        } else {
            if (!TimeProcess(snapshot.value, 20, snapshot_samples) || !TimeProcess(raw.value, 20, raw_samples)) return false;
        }
    }
    const Timing raw_timing = Summarize(raw_samples);
    const Timing snapshot_timing = Summarize(snapshot_samples);
    const bool passed = snapshot_timing.p50_us <= std::max(raw_timing.p50_us * 1.30, raw_timing.p50_us + 50.0) &&
                        snapshot_timing.p95_us <= std::max(raw_timing.p95_us * 1.50, raw_timing.p95_us + 100.0);
    std::cout << std::fixed << std::setprecision(3) << "FRAME_PROCESS_PERFORMANCE,iterations=" << raw_samples.size()
              << ",raw_p50_us=" << raw_timing.p50_us << ",raw_p95_us=" << raw_timing.p95_us
              << ",snapshot_p50_us=" << snapshot_timing.p50_us << ",snapshot_p95_us=" << snapshot_timing.p95_us
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main() {
    const auto cases = MakeCases();
    const bool accuracy = AccuracyGate(cases);
    const bool boundary = BoundaryGate();
    const bool performance = PerformanceGate(cases.back());
    const bool passed = accuracy && boundary && performance;
    std::cout << "FRAME_PROCESS_GUARD,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
