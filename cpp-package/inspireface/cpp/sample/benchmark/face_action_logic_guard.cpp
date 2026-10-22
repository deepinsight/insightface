#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common/face_info/face_action_data.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    bool baseline = false;
    int iterations = 200000;
    double max_p95_us = 0.0;
};

struct Guard {
    int failures = 0;

    void Expect(bool condition, const std::string &message) {
        if (!condition) {
            ++failures;
            std::cerr << "[FAIL] " << message << '\n';
        }
    }

    void ObserveOrExpect(bool condition, bool baseline, const std::string &message) {
        if (!condition && baseline) {
            std::cout << "[KNOWN-BASELINE-DEFECT] " << message << '\n';
            return;
        }
        Expect(condition, message);
    }
};

struct Frame {
    std::vector<inspirecv::Point2f> landmarks;
    inspirecv::Vec3f euler;
    inspirecv::Vec2f eyes;
};

struct Timing {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    uint64_t checksum = 0;
};

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--baseline") {
            options.baseline = true;
        } else if (argument.find("--iterations=") == 0) {
            options.iterations = std::max(10000, std::stoi(argument.substr(std::strlen("--iterations="))));
        } else if (argument.find("--max-p95-us=") == 0) {
            options.max_p95_us = std::stod(argument.substr(std::strlen("--max-p95-us=")));
        }
    }
    return options;
}

double Percentile(std::vector<double> samples, double percentile) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    const size_t index = static_cast<size_t>(std::ceil(percentile * samples.size())) - 1;
    return samples[std::min(index, samples.size() - 1)];
}

std::vector<inspirecv::Point2f> MakeLandmarks(const inspire::SemanticIndex &semantic_index, float mouth_width, float mouth_height) {
    const int maximum_index = std::max(std::max(semantic_index.mouth_left_corner, semantic_index.mouth_right_corner),
                                       std::max(semantic_index.mouth_upper, semantic_index.mouth_lower));
    std::vector<inspirecv::Point2f> landmarks(static_cast<size_t>(maximum_index + 1), inspirecv::Point2f(0.0f, 0.0f));
    landmarks[semantic_index.mouth_left_corner] = inspirecv::Point2f(-mouth_width * 0.5f, 0.0f);
    landmarks[semantic_index.mouth_right_corner] = inspirecv::Point2f(mouth_width * 0.5f, 0.0f);
    landmarks[semantic_index.mouth_upper] = inspirecv::Point2f(0.0f, -mouth_height * 0.5f);
    landmarks[semantic_index.mouth_lower] = inspirecv::Point2f(0.0f, mouth_height * 0.5f);
    return landmarks;
}

Frame MakeFrame(const inspire::SemanticIndex &semantic_index, float mouth_width, float mouth_height, float pitch, float yaw,
                float left_eye, float right_eye) {
    Frame frame;
    frame.landmarks = MakeLandmarks(semantic_index, mouth_width, mouth_height);
    frame.euler = inspirecv::Vec3f{pitch, yaw, 0.0f};
    frame.eyes = inspirecv::Vec2f{left_eye, right_eye};
    return frame;
}

inspire::FaceActionList Feed(inspire::FaceActionPredictor &predictor, const inspire::SemanticIndex &semantic_index, const Frame &frame) {
    predictor.RecordActionFrame(frame.landmarks, frame.euler, frame.eyes);
    return predictor.AnalysisFaceAction(semantic_index);
}

inspire::FaceActionList FeedSequence(inspire::FaceActionPredictor &predictor, const inspire::SemanticIndex &semantic_index,
                                    const std::vector<Frame> &frames) {
    inspire::FaceActionList result;
    for (const auto &frame : frames) {
        result = Feed(predictor, semantic_index, frame);
    }
    return result;
}

void TestWarmupAndNormalSemantics(Guard &guard, const inspire::SemanticIndex &semantic_index) {
    inspire::FaceActionPredictor predictor(3);
    const Frame neutral = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 0.0f, 0.9f, 0.9f);

    auto result = Feed(predictor, semantic_index, neutral);
    guard.Expect(result.normal == 1, "the first warmup frame must report the existing normal state");
    guard.Expect(predictor.GetActions().size() == 1 && predictor.GetActions()[0] == inspire::ACT_NORMAL,
                 "the warmup action list must contain only ACT_NORMAL");

    result = Feed(predictor, semantic_index, neutral);
    guard.Expect(result.normal == 1, "normal must remain set until the history window is full");

    result = Feed(predictor, semantic_index, neutral);
    guard.Expect(result.normal == 0 && result.shake == 0 && result.blink == 0 && result.jawOpen == 0 && result.raiseHead == 0,
                 "a full neutral window must preserve the existing all-zero action result");
    guard.Expect(predictor.GetActions().empty(), "a full neutral window must have an empty action list");
}

void TestSingleActions(Guard &guard, const inspire::SemanticIndex &semantic_index) {
    const Frame neutral = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 0.0f, 0.9f, 0.9f);

    {
        inspire::FaceActionPredictor predictor(3);
        const Frame jaw_open = MakeFrame(semantic_index, 10.0f, 4.0f, 0.0f, 0.0f, 0.9f, 0.9f);
        const auto result = FeedSequence(predictor, semantic_index, {neutral, neutral, jaw_open});
        guard.Expect(result.jawOpen == 1 && result.blink == 0 && result.shake == 0 && result.raiseHead == 0,
                     "jaw-open detection must preserve its independent field result");
        guard.Expect(predictor.GetActions().size() == 1 && predictor.GetActions()[0] == inspire::ACT_JAW_OPEN,
                     "jaw-open action list must contain only ACT_JAW_OPEN");
    }

    {
        inspire::FaceActionPredictor predictor(4);
        const Frame closed = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 0.0f, 0.1f, 0.1f);
        const auto result = FeedSequence(predictor, semantic_index, {neutral, neutral, neutral, closed});
        guard.Expect(result.blink == 1 && result.jawOpen == 0 && result.shake == 0 && result.raiseHead == 0,
                     "blink detection must preserve its independent field result");
        guard.Expect(predictor.GetActions().size() == 1 && predictor.GetActions()[0] == inspire::ACT_BLINK,
                     "blink action list must contain only ACT_BLINK");
    }

    {
        inspire::FaceActionPredictor predictor(3);
        const Frame left = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, -7.0f, 0.9f, 0.9f);
        const Frame right = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 7.0f, 0.9f, 0.9f);
        const auto result = FeedSequence(predictor, semantic_index, {left, neutral, right});
        guard.Expect(result.shake == 1 && result.blink == 0 && result.jawOpen == 0 && result.raiseHead == 0,
                     "shake detection must preserve its independent field result");
        guard.Expect(predictor.GetActions().size() == 1 && predictor.GetActions()[0] == inspire::ACT_SHAKE,
                     "shake action list must contain only ACT_SHAKE");
    }

    {
        inspire::FaceActionPredictor predictor(3);
        const Frame raised = MakeFrame(semantic_index, 10.0f, 2.0f, 11.0f, 0.0f, 0.9f, 0.9f);
        const auto result = FeedSequence(predictor, semantic_index, {neutral, neutral, raised});
        guard.Expect(result.raiseHead == 1 && result.blink == 0 && result.jawOpen == 0 && result.shake == 0,
                     "head-raise detection must preserve its independent field result");
        guard.Expect(predictor.GetActions().size() == 1 && predictor.GetActions()[0] == inspire::ACT_RAISE_HEAD,
                     "head-raise action list must contain only ACT_RAISE_HEAD");
    }
}

void TestCombinedActionsAndDeferredReset(Guard &guard, const inspire::SemanticIndex &semantic_index, bool baseline) {
    inspire::FaceActionPredictor predictor(4);
    const Frame left = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, -7.0f, 0.9f, 0.9f);
    const Frame neutral = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 0.0f, 0.9f, 0.9f);
    const Frame combined = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 7.0f, 0.1f, 0.1f);
    const auto result = FeedSequence(predictor, semantic_index, {left, neutral, neutral, combined});

    guard.Expect(result.blink == 1, "the combined window must detect blink");
    guard.ObserveOrExpect(result.shake == 1, baseline, "blink reset must not erase shake history during the same analysis");
    guard.Expect(result.raiseHead == 0, "the combined blink and shake window must not report an unrelated head raise");

    const auto actions = predictor.GetActions();
    guard.ObserveOrExpect(actions.size() == 2, baseline, "the combined action list must retain all independently detected actions");
    if (!baseline && actions.size() == 2) {
        guard.Expect(actions[0] == inspire::ACT_BLINK && actions[1] == inspire::ACT_SHAKE,
                     "combined action ordering must remain blink followed by shake");
    }

    const auto after_reset = Feed(predictor, semantic_index, neutral);
    guard.Expect(after_reset.normal == 1 && after_reset.blink == 0 && after_reset.shake == 0 && after_reset.raiseHead == 0,
                 "the frame after a blink must begin a fresh history window");
}

void TestExplicitReset(Guard &guard, const inspire::SemanticIndex &semantic_index) {
    inspire::FaceActionPredictor predictor(4);
    const Frame left = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, -7.0f, 0.9f, 0.9f);
    const Frame right = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 7.0f, 0.9f, 0.9f);
    const Frame neutral = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 0.0f, 0.9f, 0.9f);

    Feed(predictor, semantic_index, left);
    Feed(predictor, semantic_index, right);
    predictor.Reset();
    for (int index = 0; index < 3; ++index) {
        const auto result = Feed(predictor, semantic_index, neutral);
        guard.Expect(result.normal == 1 && result.shake == 0, "explicit Reset must discard the previously accumulated shake history");
    }
    const auto full_window = Feed(predictor, semantic_index, neutral);
    guard.Expect(full_window.normal == 0 && full_window.shake == 0, "a refilled neutral window must not expose stale pre-reset history");
}

void TestDegenerateMouthGeometry(Guard &guard, const inspire::SemanticIndex &semantic_index, bool baseline) {
    inspire::FaceActionPredictor predictor(3);
    const Frame degenerate = MakeFrame(semantic_index, 0.0f, 4.0f, 0.0f, 0.0f, 0.9f, 0.9f);
    const auto result = FeedSequence(predictor, semantic_index, {degenerate, degenerate, degenerate});
    guard.ObserveOrExpect(result.jawOpen == 0, baseline, "zero-width mouth geometry must not turn division-by-zero into a jaw-open action");
}

void TestLongRunningStability(Guard &guard, const inspire::SemanticIndex &semantic_index) {
    inspire::FaceActionPredictor predictor(1);
    const Frame neutral = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 0.0f, 0.9f, 0.9f);
    for (int index = 0; index < 100000; ++index) {
        const auto result = Feed(predictor, semantic_index, neutral);
        if (result.normal != 0 || result.blink != 0 || result.shake != 0 || result.jawOpen != 0 || result.raiseHead != 0) {
            guard.Expect(false, "a long-running neutral stream must remain stable");
            return;
        }
    }
}

void TestInvalidWindowLength(Guard &guard, const inspire::SemanticIndex &semantic_index, bool baseline) {
    if (baseline) {
        std::cout << "[KNOWN-BASELINE-DEFECT] non-positive history lengths cannot be exercised safely\n";
        return;
    }

    const Frame neutral = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 0.0f, 0.9f, 0.9f);
    inspire::FaceActionPredictor zero_length(0);
    inspire::FaceActionPredictor negative_length(-10);
    const auto zero_result = Feed(zero_length, semantic_index, neutral);
    const auto negative_result = Feed(negative_length, semantic_index, neutral);
    guard.Expect(zero_result.normal == 0 && zero_result.blink == 0 && zero_result.shake == 0 && zero_result.jawOpen == 0 &&
                   zero_result.raiseHead == 0,
                 "a zero history length must be normalized to a usable one-frame window");
    guard.Expect(negative_result.normal == 0 && negative_result.blink == 0 && negative_result.shake == 0 && negative_result.jawOpen == 0 &&
                   negative_result.raiseHead == 0,
                 "a negative history length must be normalized to a usable one-frame window");
}

Timing BenchmarkPredictor(const inspire::SemanticIndex &semantic_index, int iterations, bool exercise_reset) {
    constexpr int kBatchSize = 1000;
    const int batch_count = std::max(10, iterations / kBatchSize);
    inspire::FaceActionPredictor predictor(10);
    const Frame neutral = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 0.0f, 0.9f, 0.9f);
    const Frame mixed = MakeFrame(semantic_index, 10.0f, 2.0f, 0.0f, 0.0f, 0.4f, 0.6f);
    uint64_t checksum = 0;
    std::vector<double> batch_samples;
    batch_samples.reserve(static_cast<size_t>(batch_count));

    for (int batch = 0; batch < batch_count; ++batch) {
        const auto start = Clock::now();
        for (int index = 0; index < kBatchSize; ++index) {
            const auto result = Feed(predictor, semantic_index, exercise_reset ? mixed : neutral);
            checksum += static_cast<uint64_t>(result.normal + result.shake + result.blink + result.jawOpen + result.raiseHead);
        }
        const auto end = Clock::now();
        const double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        batch_samples.push_back(elapsed_us / kBatchSize);
    }

    Timing timing;
    timing.mean_us = 0.0;
    for (const double sample : batch_samples) {
        timing.mean_us += sample;
    }
    timing.mean_us /= batch_samples.size();
    timing.p50_us = Percentile(batch_samples, 0.50);
    timing.p95_us = Percentile(batch_samples, 0.95);
    timing.checksum = checksum;
    return timing;
}

void PrintTiming(const char *name, const Timing &timing) {
    std::cout << std::fixed << std::setprecision(6) << "[TIMING] " << name << " mean_us=" << timing.mean_us
              << " p50_us=" << timing.p50_us << " p95_us=" << timing.p95_us << " checksum=" << timing.checksum << '\n';
}

}  // namespace

int main(int argc, char **argv) {
    const Options options = ParseOptions(argc, argv);
    const inspire::SemanticIndex semantic_index;
    Guard guard;

    TestWarmupAndNormalSemantics(guard, semantic_index);
    TestSingleActions(guard, semantic_index);
    TestCombinedActionsAndDeferredReset(guard, semantic_index, options.baseline);
    TestExplicitReset(guard, semantic_index);
    TestDegenerateMouthGeometry(guard, semantic_index, options.baseline);
    TestLongRunningStability(guard, semantic_index);
    TestInvalidWindowLength(guard, semantic_index, options.baseline);

    const Timing steady = BenchmarkPredictor(semantic_index, options.iterations, false);
    const Timing resetting = BenchmarkPredictor(semantic_index, options.iterations, true);
    PrintTiming("steady", steady);
    PrintTiming("resetting", resetting);
    if (options.max_p95_us > 0.0) {
        guard.Expect(steady.p95_us <= options.max_p95_us, "steady-state P95 exceeded the requested performance gate");
        guard.Expect(resetting.p95_us <= options.max_p95_us, "resetting P95 exceeded the requested performance gate");
    }

    if (guard.failures != 0) {
        std::cerr << "[RESULT] FAIL failures=" << guard.failures << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "[RESULT] PASS mode=" << (options.baseline ? "baseline" : "fixed") << '\n';
    return EXIT_SUCCESS;
}
