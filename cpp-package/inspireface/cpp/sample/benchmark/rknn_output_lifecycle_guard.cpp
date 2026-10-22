#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "middleware/inference_wrapper/deferred_output_release.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

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

void HashBytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

struct FakeRknnRuntime {
    int ReleaseOutputs() {
        ++release_calls;
        if (fail_release) {
            return -1;
        }
        std::fill(output.begin(), output.end(), -9999.0f);
        acquired = false;
        return 0;
    }

    int Run(size_t output_size, int seed) {
        ++run_calls;
        if (acquired) {
            invalid_run_order = true;
            return -1;
        }
        if (fail_run) {
            return -1;
        }
        output.resize(output_size);
        for (size_t index = 0; index < output.size(); ++index) {
            output[index] = static_cast<float>((seed * 131 + static_cast<int>(index) * 17) % 1009) / 1009.0f;
        }
        acquired = true;
        return 0;
    }

    std::vector<float> output;
    int release_calls{0};
    int run_calls{0};
    bool acquired{false};
    bool invalid_run_order{false};
    bool fail_release{false};
    bool fail_run{false};
};

bool RunOnce(FakeRknnRuntime& runtime,
             inference_wrapper_detail::DeferredOutputRelease& lease,
             size_t output_size,
             int seed,
             const float*& output_view,
             bool populate_success = true) {
    output_view = nullptr;
    return inference_wrapper_detail::RunWithDeferredOutputRelease(
      lease,
      [&runtime]() { return runtime.ReleaseOutputs(); },
      [&runtime, output_size, seed]() { return runtime.Run(output_size, seed); },
      [&runtime, &output_view, populate_success]() {
          output_view = runtime.output.data();
          return populate_success;
      });
}

bool TestDeferredLifetimeAndAccuracy(int iterations) {
    const std::vector<size_t> output_sizes = {1, 4, 257, 512, 2048};
    FakeRknnRuntime runtime;
    inference_wrapper_detail::DeferredOutputRelease lease;
    const float* view = nullptr;
    uint64_t digest = kFnvOffset;
    bool exact = true;
    bool zero_copy = true;
    std::vector<double> samples_us;
    samples_us.reserve(static_cast<size_t>(iterations));

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const size_t output_size = output_sizes[static_cast<size_t>(iteration) % output_sizes.size()];
        const auto begin = Clock::now();
        const bool ok = RunOnce(runtime, lease, output_size, iteration + 1, view);
        const auto end = Clock::now();
        samples_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        if (!ok || view == nullptr) {
            return false;
        }
        zero_copy = zero_copy && view == runtime.output.data();
        for (size_t index = 0; index < output_size; ++index) {
            const float expected = static_cast<float>(((iteration + 1) * 131 + static_cast<int>(index) * 17) % 1009) / 1009.0f;
            exact = exact && view[index] == expected;
        }
        HashBytes(digest, view, output_size * sizeof(float));
    }

    const bool held_until_finalize = lease.HasAcquiredOutputs() && runtime.acquired;
    const bool final_release = lease.Release([&runtime]() { return runtime.ReleaseOutputs(); });
    const double mean_us = std::accumulate(samples_us.begin(), samples_us.end(), 0.0) /
                           static_cast<double>(samples_us.size());
    const bool ordering = !runtime.invalid_run_order && runtime.run_calls == iterations && runtime.release_calls == iterations;
    const bool passed = exact && zero_copy && held_until_finalize && final_release && ordering && !lease.HasAcquiredOutputs();

    std::cout << std::fixed << std::setprecision(6)
              << "RKNN_OUTPUT_LIFETIME,digest=0x" << std::hex << digest << std::dec
              << ",iterations=" << iterations
              << ",mean_us=" << mean_us
              << ",p50_us=" << Percentile(samples_us, 0.50)
              << ",p95_us=" << Percentile(samples_us, 0.95)
              << ",exact=" << (exact ? "PASS" : "FAIL")
              << ",zero_copy=" << (zero_copy ? "PASS" : "FAIL")
              << ",ordering=" << (ordering ? "PASS" : "FAIL")
              << ",status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool TestFailureCleanup() {
    bool passed = true;

    {
        FakeRknnRuntime runtime;
        inference_wrapper_detail::DeferredOutputRelease lease;
        const float* view = nullptr;
        passed = passed && !RunOnce(runtime, lease, 16, 1, view, false);
        passed = passed && !runtime.acquired && !lease.HasAcquiredOutputs() && runtime.release_calls == 1;
    }

    {
        FakeRknnRuntime runtime;
        inference_wrapper_detail::DeferredOutputRelease lease;
        const float* view = nullptr;
        passed = passed && RunOnce(runtime, lease, 16, 1, view);
        runtime.fail_run = true;
        passed = passed && !RunOnce(runtime, lease, 16, 2, view);
        passed = passed && !runtime.acquired && !lease.HasAcquiredOutputs() && runtime.release_calls == 1;
    }

    {
        FakeRknnRuntime runtime;
        inference_wrapper_detail::DeferredOutputRelease lease;
        const float* view = nullptr;
        passed = passed && RunOnce(runtime, lease, 16, 1, view);
        runtime.fail_release = true;
        passed = passed && !RunOnce(runtime, lease, 16, 2, view);
        passed = passed && runtime.run_calls == 1 && runtime.acquired && lease.HasAcquiredOutputs();
        runtime.fail_release = false;
        passed = passed && lease.Release([&runtime]() { return runtime.ReleaseOutputs(); });
    }

    std::cout << "RKNN_OUTPUT_FAILURE_CLEANUP,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [iterations]\n";
        return 2;
    }
    const int iterations = argc == 2 ? std::max(100, std::atoi(argv[1])) : 10000;
    const bool passed = TestDeferredLifetimeAndAccuracy(iterations) && TestFailureCleanup();
    std::cout << "SUMMARY,status=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}
