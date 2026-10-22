#include "frame_process.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#if defined(ISF_ENABLE_INSPIRECV_TASK_PREPROCESS)
#include <inspirecv/task/task.h>
#else
#include <MNN/ImageProcess.hpp>
#endif
#include "isf_check.h"

namespace inspirecv {

namespace frame_backend {

static void SwapRedBlueChannels(uint8_t* data, int width, int height) {
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t index = 0; index < pixels; ++index) {
        std::swap(data[index * 4], data[index * 4 + 2]);
    }
}

#if defined(ISF_ENABLE_INSPIRECV_TASK_PREPROCESS)
using Matrix = task::Matrix;
using Point = task::Point;
using Config = task::StreamTask::Config;
constexpr auto kBilinear = task::BILINEAR;
constexpr auto kZero = task::ZERO;
constexpr auto kNV21 = task::YUV_NV21;
constexpr auto kNV12 = task::YUV_NV12;
constexpr auto kRGBA = task::RGBA;
constexpr auto kRGB = task::RGB;
constexpr auto kBGR = task::BGR;
constexpr auto kBGRA = task::BGRA;
constexpr auto kI420 = task::YUV_I420;
constexpr auto kGray = task::GRAY;

class Converter {
public:
    Converter() = default;
    ~Converter() { Reset(); }
    Converter(const Converter&) = delete;
    Converter& operator=(const Converter&) = delete;

    void Reset() {
        if (process_ != nullptr) {
            task::StreamTask::Destroy(process_);
            process_ = nullptr;
        }
    }

    bool Convert(const Config& config, const Matrix& matrix, const uint8_t* source,
                 int sourceWidth, int sourceHeight, uint8_t* dest,
                 int destWidth, int destHeight, int destChannels) {
        // The Task converter supports RGB->RGBA and BGR->BGRA, but not the
        // opposite channel orders. Expand in the supported order, then swap
        // red/blue in the existing output without allocating another image.
        const bool swap_red_blue =
          (config.sourceFormat == task::BGR && config.destFormat == task::RGBA) ||
          (config.sourceFormat == task::RGB && config.destFormat == task::BGRA);
        if (swap_red_blue && destChannels != 4) return false;
        if (process_ == nullptr) {
            Config task_config = config;
            if (swap_red_blue) {
                task_config.destFormat = config.sourceFormat == task::BGR ? task::BGRA : task::RGBA;
            }
            process_ = task::StreamTask::Create(task_config);
        }
        if (process_ == nullptr) return false;
        process_->SetMatrix(matrix);
        if (process_->Convert(source, sourceWidth, sourceHeight, 0, dest,
                              destWidth, destHeight, destChannels, 0,
                              halide_type_of<uint8_t>()) != SUCCESS) {
            return false;
        }
        if (swap_red_blue) {
            SwapRedBlueChannels(dest, destWidth, destHeight);
        }
        return true;
    }

private:
    task::StreamTask* process_ = nullptr;
};
#else
using Matrix = MNN::CV::Matrix;
using Point = MNN::CV::Point;
using Config = MNN::CV::ImageProcess::Config;
constexpr auto kBilinear = MNN::CV::BILINEAR;
constexpr auto kZero = MNN::CV::ZERO;
constexpr auto kNV21 = MNN::CV::YUV_NV21;
constexpr auto kNV12 = MNN::CV::YUV_NV12;
constexpr auto kRGBA = MNN::CV::RGBA;
constexpr auto kRGB = MNN::CV::RGB;
constexpr auto kBGR = MNN::CV::BGR;
constexpr auto kBGRA = MNN::CV::BGRA;
constexpr auto kI420 = MNN::CV::YUV_I420;
constexpr auto kGray = MNN::CV::GRAY;

class Converter {
public:
    void Reset() { process_.reset(); }

    bool Convert(const Config& config, const Matrix& matrix, const uint8_t* source,
                 int sourceWidth, int sourceHeight, uint8_t* dest,
                 int destWidth, int destHeight, int destChannels) {
        // MNN also lacks the crossed three-to-four-channel converters. Its
        // success status alone does not guarantee a correctly packed output.
        const bool swap_red_blue =
          (config.sourceFormat == MNN::CV::BGR && config.destFormat == MNN::CV::RGBA) ||
          (config.sourceFormat == MNN::CV::RGB && config.destFormat == MNN::CV::BGRA);
        if (swap_red_blue && destChannels != 4) return false;
        if (process_ == nullptr) {
            Config mnn_config = config;
            if (swap_red_blue) {
                mnn_config.destFormat = config.sourceFormat == MNN::CV::BGR ? MNN::CV::BGRA : MNN::CV::RGBA;
            }
            process_.reset(MNN::CV::ImageProcess::create(mnn_config));
        }
        if (process_ == nullptr) return false;
        process_->setMatrix(matrix);
        if (process_->convert(source, sourceWidth, sourceHeight, 0, dest,
                              destWidth, destHeight, destChannels, 0,
                              halide_type_of<uint8_t>()) != MNN::ErrorCode::NO_ERROR) {
            return false;
        }
        if (swap_red_blue) {
            SwapRedBlueChannels(dest, destWidth, destHeight);
        }
        return true;
    }

private:
    std::shared_ptr<MNN::CV::ImageProcess> process_;
};
#endif

}  // namespace frame_backend

class FrameProcess::Impl {
public:
    Impl()
    : buffer_(nullptr),
      height_(0),
      width_(0),
      preview_scale_(0),
      preview_size_(192),
      rotation_mode_(ROTATION_0),
      source_data_format_(NV21),
      dest_data_format_(BGR) {
        SetDataFormat(NV21);
        SetDestFormat(BGR);
        config_.filterType = frame_backend::kBilinear;
        config_.wrap = frame_backend::kZero;
    }

    Impl(const Impl& other)
    : buffer_(other.buffer_),
      height_(other.height_),
      width_(other.width_),
      preview_scale_(other.preview_scale_),
      preview_size_(other.preview_size_),
      tr_(other.tr_),
      rotation_mode_(other.rotation_mode_),
      source_data_format_(other.source_data_format_),
      dest_data_format_(other.dest_data_format_),
      config_(other.config_) {}

    Impl& operator=(const Impl& other) {
        if (this != &other) {
            converter_.Reset();
            buffer_ = other.buffer_;
            height_ = other.height_;
            width_ = other.width_;
            preview_scale_ = other.preview_scale_;
            preview_size_ = other.preview_size_;
            tr_ = other.tr_;
            rotation_mode_ = other.rotation_mode_;
            source_data_format_ = other.source_data_format_;
            dest_data_format_ = other.dest_data_format_;
            config_ = other.config_;
        }
        return *this;
    }

    bool SetDataFormat(DATA_FORMAT data_format) {
        switch (data_format) {
            case NV21: config_.sourceFormat = frame_backend::kNV21; break;
            case NV12: config_.sourceFormat = frame_backend::kNV12; break;
            case RGBA: config_.sourceFormat = frame_backend::kRGBA; break;
            case RGB: config_.sourceFormat = frame_backend::kRGB; break;
            case BGR: config_.sourceFormat = frame_backend::kBGR; break;
            case BGRA: config_.sourceFormat = frame_backend::kBGRA; break;
            case I420: config_.sourceFormat = frame_backend::kI420; break;
            case GRAY: config_.sourceFormat = frame_backend::kGray; break;
            default: return false;
        }
        source_data_format_ = data_format;
        converter_.Reset();
        return true;
    }

    bool SetDestFormat(DATA_FORMAT data_format) {
        switch (data_format) {
            case NV21: config_.destFormat = frame_backend::kNV21; break;
            case NV12: config_.destFormat = frame_backend::kNV12; break;
            case RGBA: config_.destFormat = frame_backend::kRGBA; break;
            case RGB: config_.destFormat = frame_backend::kRGB; break;
            case BGR: config_.destFormat = frame_backend::kBGR; break;
            case BGRA: config_.destFormat = frame_backend::kBGRA; break;
            case I420: config_.destFormat = frame_backend::kI420; break;
            case GRAY: config_.destFormat = frame_backend::kGray; break;
            default: return false;
        }
        dest_data_format_ = data_format;
        converter_.Reset();
        return true;
    }

    static bool IsRotationValid(ROTATION_MODE mode) {
        return mode >= ROTATION_0 && mode <= ROTATION_270;
    }

    static bool IsScaleValid(float scale) {
        return std::isfinite(scale) && scale > 0.0f;
    }

    bool IsSourceReady() const {
        if (buffer_ == nullptr || width_ <= 0 || height_ <= 0 ||
            ((source_data_format_ == NV12 || source_data_format_ == NV21 || source_data_format_ == I420) &&
             ((width_ & 1) != 0 || (height_ & 1) != 0))) {
            return false;
        }
        const uint64_t pixels = static_cast<uint64_t>(width_) * static_cast<uint64_t>(height_);
        return pixels <= static_cast<uint64_t>(std::numeric_limits<int>::max());
    }

    int DestChannels() const {
        switch (dest_data_format_) {
            case GRAY: return 1;
            case RGB:
            case BGR: return 3;
            case RGBA:
            case BGRA: return 4;
            default: return 0;
        }
    }

    static bool ScaledDimension(int source, float scale, int& output) {
        if (source <= 0 || !IsScaleValid(scale)) {
            return false;
        }
        // Preserve the historical float multiplication/rounding for valid
        // inputs while rejecting values that cannot be converted to int.
        const float scaled = static_cast<float>(source) * scale;
        if (!std::isfinite(scaled) || scaled < 1.0f ||
            scaled >= static_cast<float>(std::numeric_limits<int>::max())) {
            return false;
        }
        output = static_cast<int>(scaled);
        return output > 0;
    }

    bool CanCreateOutput(int width, int height) const {
        const int channels = DestChannels();
        if (width <= 0 || height <= 0 || channels <= 0) {
            return false;
        }
        const uint64_t elements = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * static_cast<uint64_t>(channels);
        return elements <= static_cast<uint64_t>(std::numeric_limits<int>::max());
    }

    void ResetTransform() {
        const float identity[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        tr_.set9(identity);
    }

    static bool SetPolyToPoly(frame_backend::Matrix& matrix, const float* source, const float* destination) {
        std::array<frame_backend::Point, 4> source_points;
        std::array<frame_backend::Point, 4> destination_points;
        for (size_t index = 0; index < source_points.size(); ++index) {
            source_points[index].set(source[index * 2], source[index * 2 + 1]);
            destination_points[index].set(destination[index * 2], destination[index * 2 + 1]);
        }
        return matrix.setPolyToPoly(source_points.data(), destination_points.data(), 4);
    }

    void UpdateTransformMatrix() {
        if (width_ <= 0 || height_ <= 0 || !IsScaleValid(preview_scale_) || !IsRotationValid(rotation_mode_)) {
            ResetTransform();
            return;
        }
        float srcPoints[] = {0.0f, 0.0f, 0.0f, (float)(height_ - 1), (float)(width_ - 1), 0.0f, (float)(width_ - 1), (float)(height_ - 1)};

        float dstPoints[8];
        if (rotation_mode_ == ROTATION_270) {
            float points[] = {(float)(height_ * preview_scale_ - 1),
                              0.0f,
                              0.0f,
                              0.0f,
                              (float)(height_ * preview_scale_ - 1),
                              (float)(width_ * preview_scale_ - 1),
                              0.0f,
                              (float)(width_ * preview_scale_ - 1)};
            memcpy(dstPoints, points, sizeof(points));
        } else if (rotation_mode_ == ROTATION_90) {
            float points[] = {0.0f,
                              (float)(width_ * preview_scale_ - 1),
                              (float)(height_ * preview_scale_ - 1),
                              (float)(width_ * preview_scale_ - 1),
                              0.0f,
                              0.0f,
                              (float)(height_ * preview_scale_ - 1),
                              0.0f};
            memcpy(dstPoints, points, sizeof(points));
        } else if (rotation_mode_ == ROTATION_180) {
            float points[] = {(float)(width_ * preview_scale_ - 1),
                              (float)(height_ * preview_scale_ - 1),
                              (float)(width_ * preview_scale_ - 1),
                              0.0f,
                              0.0f,
                              (float)(height_ * preview_scale_ - 1),
                              0.0f,
                              0.0f};
            memcpy(dstPoints, points, sizeof(points));
        } else {  // ROTATION_0
            float points[] = {0.0f,
                              0.0f,
                              0.0f,
                              (float)(height_ * preview_scale_ - 1),
                              (float)(width_ * preview_scale_ - 1),
                              0.0f,
                              (float)(width_ * preview_scale_ - 1),
                              (float)(height_ * preview_scale_ - 1)};
            memcpy(dstPoints, points, sizeof(points));
        }

        if (!SetPolyToPoly(tr_, dstPoints, srcPoints)) {
            ResetTransform();
        }
    }

    const uint8_t *buffer_;                 // Pointer to the data buffer.
    int height_;                            // Height of the camera stream image.
    int width_;                             // Width of the camera stream image.
    float preview_scale_;                   // Scaling factor for the preview image.
    int preview_size_;                      // Size of the preview image.
    frame_backend::Matrix tr_;              // Output-to-input transformation matrix.
    ROTATION_MODE rotation_mode_;           // Current rotation mode.
    DATA_FORMAT source_data_format_;         // Source buffer layout.
    DATA_FORMAT dest_data_format_;           // Destination image layout.
    frame_backend::Config config_;          // Image processing configuration.
    mutable frame_backend::Converter converter_;  // Reused backend preprocessing context.
};

FrameProcess FrameProcess::Create(const uint8_t *data_buffer, int height, int width, DATA_FORMAT data_format, ROTATION_MODE rotation_mode) {
    FrameProcess process;
    process.SetDataFormat(data_format);
    process.SetDataBuffer(data_buffer, height, width);
    process.SetRotationMode(rotation_mode);
    return process;
}

FrameProcess FrameProcess::Create(const inspirecv::Image &image, DATA_FORMAT data_format, ROTATION_MODE rotation_mode) {
    return Create(image.Data(), image.Height(), image.Width(), data_format, rotation_mode);
}

FrameProcess::FrameProcess() : pImpl(std::make_unique<Impl>()) {
    pImpl->UpdateTransformMatrix();
}

FrameProcess::~FrameProcess() = default;

FrameProcess::FrameProcess(const FrameProcess &other)
: pImpl(other.pImpl ? std::make_unique<Impl>(*other.pImpl) : std::make_unique<Impl>()) {}

FrameProcess::FrameProcess(FrameProcess &&other) noexcept = default;

FrameProcess &FrameProcess::operator=(const FrameProcess &other) {
    if (this != &other) {
        if (!other.pImpl) {
            pImpl = std::make_unique<Impl>();
        } else if (!pImpl) {
            pImpl = std::make_unique<Impl>(*other.pImpl);
        } else {
            *pImpl = *other.pImpl;
        }
    }
    return *this;
}

FrameProcess &FrameProcess::operator=(FrameProcess &&other) noexcept = default;

void FrameProcess::SetDataBuffer(const uint8_t *data_buffer, int height, int width) {
    if (data_buffer == nullptr || height <= 0 || width <= 0) {
        pImpl->buffer_ = nullptr;
        pImpl->height_ = 0;
        pImpl->width_ = 0;
        pImpl->preview_scale_ = 0.0f;
        pImpl->UpdateTransformMatrix();
        return;
    }
    pImpl->buffer_ = data_buffer;
    pImpl->height_ = height;
    pImpl->width_ = width;
    pImpl->preview_scale_ = pImpl->preview_size_ / static_cast<float>(std::max(height, width));
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetPreviewSize(const int size) {
    if (size <= 0) {
        return;
    }
    pImpl->preview_size_ = size;
    const int source_size = std::max(pImpl->height_, pImpl->width_);
    pImpl->preview_scale_ = source_size > 0 ? pImpl->preview_size_ / static_cast<float>(source_size) : 0.0f;
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetPreviewScale(const float scale) {
    if (!Impl::IsScaleValid(scale)) {
        return;
    }
    const int source_size = std::max(pImpl->height_, pImpl->width_);
    if (source_size <= 0 || static_cast<double>(scale) * source_size > std::numeric_limits<int>::max()) {
        return;
    }
    pImpl->preview_scale_ = scale;
    pImpl->preview_size_ = static_cast<int>(pImpl->preview_scale_ * source_size);
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetRotationMode(ROTATION_MODE mode) {
    if (!Impl::IsRotationValid(mode)) {
        return;
    }
    pImpl->rotation_mode_ = mode;
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetDataFormat(DATA_FORMAT data_format) {
    pImpl->SetDataFormat(data_format);
}

void FrameProcess::SetDestFormat(DATA_FORMAT data_format) {
    pImpl->SetDestFormat(data_format);
}

float FrameProcess::GetPreviewScale() {
    return pImpl->preview_scale_;
}

inspirecv::TransformMatrix FrameProcess::GetAffineMatrix() const {
    auto affine_matrix = inspirecv::TransformMatrix::Create();
    affine_matrix[0] = pImpl->tr_[0];
    affine_matrix[1] = pImpl->tr_[1];
    affine_matrix[2] = pImpl->tr_[2];
    affine_matrix[3] = pImpl->tr_[3];
    affine_matrix[4] = pImpl->tr_[4];
    affine_matrix[5] = pImpl->tr_[5];
    return affine_matrix;
}

int FrameProcess::GetHeight() const {
    return pImpl->height_;
}

int FrameProcess::GetWidth() const {
    return pImpl->width_;
}

ROTATION_MODE FrameProcess::getRotationMode() const {
    return pImpl->rotation_mode_;
}

inspirecv::Image FrameProcess::ExecuteImageAffineProcessing(inspirecv::TransformMatrix &affine_matrix, const int width_out,
                                                            const int height_out) const {
    if (!pImpl->IsSourceReady() || !pImpl->CanCreateOutput(width_out, height_out)) {
        return {};
    }
    const int sw = pImpl->width_;
    const int sh = pImpl->height_;
    frame_backend::Matrix tr;
    std::vector<float> tr_cv({1, 0, 0, 0, 1, 0, 0, 0, 1});
    const std::vector<float> affine_values = affine_matrix.Squeeze();
    if (affine_values.size() != 6 ||
        !std::all_of(affine_values.begin(), affine_values.end(), [](float value) { return std::isfinite(value); })) {
        return {};
    }
    memcpy(tr_cv.data(), affine_values.data(), sizeof(float) * 6);
    tr.set9(tr_cv.data());
    frame_backend::Matrix tr_inv;
    if (!tr.invert(&tr_inv)) {
        return {};
    }
    auto img_out = inspirecv::Image::Create(width_out, height_out, pImpl->DestChannels());
    if (img_out.Empty()) {
        return {};
    }
    const bool converted = pImpl->converter_.Convert(pImpl->config_, tr_inv, pImpl->buffer_, sw, sh,
                                                      const_cast<uint8_t*>(img_out.Data()), width_out, height_out,
                                                      pImpl->DestChannels());
    return converted ? std::move(img_out) : inspirecv::Image();
}

inspirecv::Image FrameProcess::ExecutePreviewImageProcessing(bool with_rotation) {
    return ExecuteImageScaleProcessing(pImpl->preview_scale_, with_rotation);
}

inspirecv::Image FrameProcess::ExecuteImageScaleProcessing(const float scale, bool with_rotation) {
    if (!pImpl->IsSourceReady() || !Impl::IsScaleValid(scale)) {
        return {};
    }
    const int sw = pImpl->width_;
    const int sh = pImpl->height_;
    const bool swaps_dimensions = with_rotation &&
                                  (pImpl->rotation_mode_ == ROTATION_90 || pImpl->rotation_mode_ == ROTATION_270);
    int scaled_width = 0;
    int scaled_height = 0;
    if (!Impl::ScaledDimension(swaps_dimensions ? sh : sw, scale, scaled_width) ||
        !Impl::ScaledDimension(swaps_dimensions ? sw : sh, scale, scaled_height) ||
        !pImpl->CanCreateOutput(scaled_width, scaled_height)) {
        return {};
    }
    frame_backend::Matrix transform;
    if (pImpl->rotation_mode_ == ROTATION_270 && with_rotation) {
        float srcPoints[] = {
          0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1),
        };
        float dstPoints[] = {
          (float)(pImpl->height_ * scale - 1), 0.0f, 0.0f, 0.0f, (float)(pImpl->height_ * scale - 1), (float)(pImpl->width_ * scale - 1), 0.0f,
          (float)(pImpl->width_ * scale - 1)};

        if (!Impl::SetPolyToPoly(transform, dstPoints, srcPoints)) {
            return {};
        }
        inspirecv::Image img_out(scaled_width, scaled_height, pImpl->DestChannels());
        const bool converted = pImpl->converter_.Convert(pImpl->config_, transform, pImpl->buffer_, sw, sh,
                                                          const_cast<uint8_t*>(img_out.Data()), scaled_width, scaled_height,
                                                          pImpl->DestChannels());
        return converted ? std::move(img_out) : inspirecv::Image();
    } else if (pImpl->rotation_mode_ == ROTATION_90 && with_rotation) {
        float srcPoints[] = {
          0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1),
        };
        float dstPoints[] = {
          0.0f,
          (float)(pImpl->width_ * scale - 1),
          (float)(pImpl->height_ * scale - 1),
          (float)(pImpl->width_ * scale - 1),
          0.0f,
          0.0f,
          (float)(pImpl->height_ * scale - 1),
          0.0f,
        };
        if (!Impl::SetPolyToPoly(transform, dstPoints, srcPoints)) {
            return {};
        }
        inspirecv::Image img_out(scaled_width, scaled_height, pImpl->DestChannels());
        const bool converted = pImpl->converter_.Convert(pImpl->config_, transform, pImpl->buffer_, sw, sh,
                                                          const_cast<uint8_t*>(img_out.Data()), scaled_width, scaled_height,
                                                          pImpl->DestChannels());
        return converted ? std::move(img_out) : inspirecv::Image();
    } else if (pImpl->rotation_mode_ == ROTATION_180 && with_rotation) {
        float srcPoints[] = {
          0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1),
        };
        float dstPoints[] = {
          (float)(pImpl->width_ * scale - 1),
          (float)(pImpl->height_ * scale - 1),
          (float)(pImpl->width_ * scale - 1),
          0.0f,
          0.0f,
          (float)(pImpl->height_ * scale - 1),
          0.0f,
          0.0f,
        };
        if (!Impl::SetPolyToPoly(transform, dstPoints, srcPoints)) {
            return {};
        }
        inspirecv::Image img_out(scaled_width, scaled_height, pImpl->DestChannels());
        const bool converted = pImpl->converter_.Convert(pImpl->config_, transform, pImpl->buffer_, sw, sh,
                                                          const_cast<uint8_t*>(img_out.Data()), scaled_width, scaled_height,
                                                          pImpl->DestChannels());
        return converted ? std::move(img_out) : inspirecv::Image();
    } else {
        float srcPoints[] = {
          0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1),
        };
        float dstPoints[] = {
          0.0f,
          0.0f,
          0.0f,
          (float)(pImpl->height_ * scale - 1),
          (float)(pImpl->width_ * scale - 1),
          0.0f,
          (float)(pImpl->width_ * scale - 1),
          (float)(pImpl->height_ * scale - 1),
        };
        if (!Impl::SetPolyToPoly(transform, dstPoints, srcPoints)) {
            return {};
        }
        inspirecv::Image img_out(scaled_width, scaled_height, pImpl->DestChannels());
        const bool converted = pImpl->converter_.Convert(pImpl->config_, transform, pImpl->buffer_, sw, sh,
                                                          const_cast<uint8_t*>(img_out.Data()), scaled_width, scaled_height,
                                                          pImpl->DestChannels());
        return converted ? std::move(img_out) : inspirecv::Image();
    }
}

inspirecv::TransformMatrix FrameProcess::GetRotationModeAffineMatrix() const {
    if (pImpl->width_ <= 0 || pImpl->height_ <= 0 || !Impl::IsRotationValid(pImpl->rotation_mode_)) {
        return inspirecv::TransformMatrix::Create();
    }
    float srcPoints[] = {0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1)};
    float dstPoints[8];
    
    if (pImpl->rotation_mode_ == ROTATION_270) {
        float points[] = {(float)(pImpl->height_ - 1),
                         0.0f,
                         0.0f,
                         0.0f,
                         (float)(pImpl->height_ - 1),
                         (float)(pImpl->width_ - 1),
                         0.0f,
                         (float)(pImpl->width_ - 1)};
        memcpy(dstPoints, points, sizeof(points));
    } else if (pImpl->rotation_mode_ == ROTATION_90) {
        float points[] = {0.0f,
                         (float)(pImpl->width_ - 1),
                         (float)(pImpl->height_ - 1),
                         (float)(pImpl->width_ - 1),
                         0.0f,
                         0.0f,
                         (float)(pImpl->height_ - 1),
                         0.0f};
        memcpy(dstPoints, points, sizeof(points));
    } else if (pImpl->rotation_mode_ == ROTATION_180) {
        float points[] = {(float)(pImpl->width_ - 1),
                         (float)(pImpl->height_ - 1),
                         (float)(pImpl->width_ - 1),
                         0.0f,
                         0.0f,
                         (float)(pImpl->height_ - 1),
                         0.0f,
                         0.0f};
        memcpy(dstPoints, points, sizeof(points));
    } else {  // ROTATION_0
        float points[] = {0.0f,
                         0.0f,
                         0.0f,
                         (float)(pImpl->height_ - 1),
                         (float)(pImpl->width_ - 1),
                         0.0f,
                         (float)(pImpl->width_ - 1),
                         (float)(pImpl->height_ - 1)};
        memcpy(dstPoints, points, sizeof(points));
    }

    frame_backend::Matrix tr;
    if (!Impl::SetPolyToPoly(tr, dstPoints, srcPoints)) {
        const float identity[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        tr.set9(identity);
    }
    
    auto affine_matrix = inspirecv::TransformMatrix::Create();
    affine_matrix[0] = tr[0];
    affine_matrix[1] = tr[1];
    affine_matrix[2] = tr[2];
    affine_matrix[3] = tr[3];
    affine_matrix[4] = tr[4];
    affine_matrix[5] = tr[5];
    
    return affine_matrix;
}

}  // namespace inspirecv
