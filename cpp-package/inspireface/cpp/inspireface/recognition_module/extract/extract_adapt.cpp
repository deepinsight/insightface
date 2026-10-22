/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "extract_adapt.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

bool IsDebugRawFeatureEnabled() {
    const char *env = std::getenv("ISF_DEBUG_RAW_FEATURE");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

float CalcSegmentNorm(const std::vector<float> &embedding, size_t begin, size_t end) {
    float sum = 0.0f;
    for (size_t i = begin; i < end; ++i) {
        sum += embedding[i] * embedding[i];
    }
    return std::sqrt(sum);
}

float CalcSegmentCosine(const std::vector<float> &embedding, size_t lhs_begin, size_t rhs_begin, size_t length) {
    float lhs_norm = 0.0f;
    float rhs_norm = 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        const float lhs = embedding[lhs_begin + i];
        const float rhs = embedding[rhs_begin + i];
        lhs_norm += lhs * lhs;
        rhs_norm += rhs * rhs;
        dot += lhs * rhs;
    }
    if (lhs_norm <= 0.0f || rhs_norm <= 0.0f) {
        return 0.0f;
    }
    return dot / std::sqrt(lhs_norm * rhs_norm);
}

float CalcSegmentMaxAbsDiff(const std::vector<float> &embedding, size_t lhs_begin, size_t rhs_begin, size_t length) {
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        const float diff = std::fabs(embedding[lhs_begin + i] - embedding[rhs_begin + i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }
    return max_abs_diff;
}

void DumpRawFeature(const inspire::AnyTensorOutputs &outputs, const std::vector<float> &embedding, float norm) {
    INSPIRE_LOGI("[ISF_DEBUG_RAW_FEATURE] outputs=%zu primary_name=%s primary_size=%zu raw_l2_norm=%.9f",
                 outputs.size(),
                 outputs.empty() ? "<none>" : outputs[0].first.c_str(),
                 embedding.size(),
                 norm);

    for (const auto &output : outputs) {
        INSPIRE_LOGI("[ISF_DEBUG_RAW_FEATURE] tensor name=%s size=%zu", output.first.c_str(), output.second.size());
    }

    if (embedding.size() == 512) {
        constexpr size_t kChunkSize = 128;
        std::array<float, 4> chunk_norms{};
        for (size_t i = 0; i < chunk_norms.size(); ++i) {
            const size_t begin = i * kChunkSize;
            chunk_norms[i] = CalcSegmentNorm(embedding, begin, begin + kChunkSize);
        }
        INSPIRE_LOGI("[ISF_DEBUG_RAW_FEATURE] chunk_norms[128x4]=[%.9f, %.9f, %.9f, %.9f]",
                     chunk_norms[0], chunk_norms[1], chunk_norms[2], chunk_norms[3]);
        INSPIRE_LOGI("[ISF_DEBUG_RAW_FEATURE] chunk_cosine c01=%.9f c02=%.9f c03=%.9f c12=%.9f c13=%.9f c23=%.9f",
                     CalcSegmentCosine(embedding, 0, 128, kChunkSize),
                     CalcSegmentCosine(embedding, 0, 256, kChunkSize),
                     CalcSegmentCosine(embedding, 0, 384, kChunkSize),
                     CalcSegmentCosine(embedding, 128, 256, kChunkSize),
                     CalcSegmentCosine(embedding, 128, 384, kChunkSize),
                     CalcSegmentCosine(embedding, 256, 384, kChunkSize));
        INSPIRE_LOGI("[ISF_DEBUG_RAW_FEATURE] chunk_max_abs_diff d01=%.9f d02=%.9f d03=%.9f",
                     CalcSegmentMaxAbsDiff(embedding, 0, 128, kChunkSize),
                     CalcSegmentMaxAbsDiff(embedding, 0, 256, kChunkSize),
                     CalcSegmentMaxAbsDiff(embedding, 0, 384, kChunkSize));
    }

    const char *dump_path = std::getenv("ISF_DEBUG_RAW_FEATURE_PATH");
    if (dump_path == nullptr || dump_path[0] == '\0') {
        return;
    }

    FILE *fp = std::fopen(dump_path, "a");
    if (fp == nullptr) {
        INSPIRE_LOGW("[ISF_DEBUG_RAW_FEATURE] failed to open dump path: %s", dump_path);
        return;
    }

    std::fprintf(fp, "primary_name=%s size=%zu raw_l2_norm=%.9f\n",
                 outputs.empty() ? "<none>" : outputs[0].first.c_str(),
                 embedding.size(),
                 norm);
    std::fprintf(fp, "values=");
    for (size_t i = 0; i < embedding.size(); ++i) {
        std::fprintf(fp, i == 0 ? "%.9f" : ",%.9f", embedding[i]);
    }
    std::fprintf(fp, "\n");
    std::fclose(fp);
}

}  // namespace

namespace inspire {

Embedded ExtractAdapt::GetFaceFeature(const inspirecv::Image &bgr_affine) {
    AnyTensorOutputs outputs;
    if (Forward(bgr_affine, outputs) != InferenceWrapper::WrapperOk || outputs.empty()) {
        return {};
    }

    return outputs[0].second;
}

Embedded ExtractAdapt::operator()(const inspirecv::Image &bgr_affine, float &norm, bool normalize) {
    AnyTensorOutputs outputs;
    if (Forward(bgr_affine, outputs) != InferenceWrapper::WrapperOk || outputs.empty()) {
        norm = 0.0f;
        return {};
    }

    auto &embedded = outputs[0].second;
    float mse = 0.0f;
    for (const auto &one : embedded) {
        mse += one * one;
    }
    mse = sqrt(mse);
    norm = mse;

    if (IsDebugRawFeatureEnabled()) {
        DumpRawFeature(outputs, embedded, norm);
    }

    if (normalize && mse > 0.0f) {
        for (float &one : embedded) {
            one /= mse;
        }
    }

    return embedded;
}

ExtractAdapt::ExtractAdapt() : AnyNetAdapter("ExtractAdapt") {}

}  // namespace inspire
