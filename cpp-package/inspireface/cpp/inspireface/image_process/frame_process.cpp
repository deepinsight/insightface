#include "frame_process.h"
#include <vector>
#include <MNN/ImageProcess.hpp>
#include "isf_check.h"

namespace inspirecv {

class FrameProcess::Impl {
public:
    Impl() : buffer_(nullptr), height_(0), width_(0), preview_scale_(0), preview_size_(192), rotation_mode_(ROTATION_0) {
        SetDataFormat(NV21);
        SetDestFormat(BGR);
        config_.filterType = MNN::CV::BILINEAR;
        config_.wrap = MNN::CV::ZERO;
    }

    void SetDataFormat(DATA_FORMAT data_format) {
        if (data_format == NV21) {
            config_.sourceFormat = MNN::CV::YUV_NV21;
        }
        if (data_format == NV12) {
            config_.sourceFormat = MNN::CV::YUV_NV12;
        }
        if (data_format == RGBA) {
            config_.sourceFormat = MNN::CV::RGBA;
        }
        if (data_format == RGB) {
            config_.sourceFormat = MNN::CV::RGB;
        }
        if (data_format == BGR) {
            config_.sourceFormat = MNN::CV::BGR;
        }
        if (data_format == BGRA) {
            config_.sourceFormat = MNN::CV::BGRA;
        }
        if (data_format == I420) {
            config_.sourceFormat = MNN::CV::YUV_I420;
        }
        if (data_format == GRAY) {
            config_.sourceFormat = MNN::CV::GRAY;
        }
    }

    void SetDestFormat(DATA_FORMAT data_format) {
        if (data_format == NV21) {
            config_.destFormat = MNN::CV::YUV_NV21;
        }
        if (data_format == NV12) {
            config_.destFormat = MNN::CV::YUV_NV12;
        }
        if (data_format == RGBA) {
            config_.destFormat = MNN::CV::RGBA;
        }
        if (data_format == RGB) {
            config_.destFormat = MNN::CV::RGB;
        }
        if (data_format == BGR) {
            config_.destFormat = MNN::CV::BGR;
        }
        if (data_format == BGRA) {
            config_.destFormat = MNN::CV::BGRA;
        }
        if (data_format == I420) {
            config_.destFormat = MNN::CV::YUV_I420;
        }
        if (data_format == GRAY) {
            config_.destFormat = MNN::CV::GRAY;
        }
    }

    void UpdateTransformMatrix() {
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

        tr_.setPolyToPoly((MNN::CV::Point *)dstPoints, (MNN::CV::Point *)srcPoints, 4);
    }

    const uint8_t *buffer_;                 // Pointer to the data buffer.
    int height_;                            // Height of the camera stream image.
    int width_;                             // Width of the camera stream image.
    float preview_scale_;                   // Scaling factor for the preview image.
    int preview_size_;                      // Size of the preview image.
    MNN::CV::Matrix tr_;                    // Affine transformation matrix.
    ROTATION_MODE rotation_mode_;           // Current rotation mode.
    MNN::CV::ImageProcess::Config config_;  // Image processing configuration.
};

FrameProcess FrameProcess::Create(const uint8_t *data_buffer, int height, int width, DATA_FORMAT data_format, ROTATION_MODE rotation_mode) {
    FrameProcess process;
    process.SetDataBuffer(data_buffer, height, width);
    process.SetDataFormat(data_format);
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

FrameProcess::FrameProcess(const FrameProcess &other) : pImpl(std::make_unique<Impl>(*other.pImpl)) {}

FrameProcess::FrameProcess(FrameProcess &&other) noexcept = default;

FrameProcess &FrameProcess::operator=(const FrameProcess &other) {
    if (this != &other) {
        *pImpl = *other.pImpl;
    }
    return *this;
}

FrameProcess &FrameProcess::operator=(FrameProcess &&other) noexcept = default;

void FrameProcess::SetDataBuffer(const uint8_t *data_buffer, int height, int width) {
    pImpl->buffer_ = data_buffer;
    pImpl->height_ = height;
    pImpl->width_ = width;
    pImpl->preview_scale_ = pImpl->preview_size_ / static_cast<float>(std::max(height, width));
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetPreviewSize(const int size) {
    pImpl->preview_size_ = size;
    pImpl->preview_scale_ = pImpl->preview_size_ / static_cast<float>(std::max(pImpl->height_, pImpl->width_));
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetPreviewScale(const float scale) {
    pImpl->preview_scale_ = scale;
    pImpl->preview_size_ = static_cast<int>(pImpl->preview_scale_ * std::max(pImpl->height_, pImpl->width_));
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetRotationMode(ROTATION_MODE mode) {
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
    int sw = pImpl->width_;
    int sh = pImpl->height_;
    int rot_sw = sw;
    int rot_sh = sh;
    MNN::CV::Matrix tr;
    std::vector<float> tr_cv({1, 0, 0, 0, 1, 0, 0, 0, 1});
    memcpy(tr_cv.data(), affine_matrix.Squeeze().data(), sizeof(float) * 6);
    tr.set9(tr_cv.data());
    MNN::CV::Matrix tr_inv;
    tr.invert(&tr_inv);
    std::shared_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(pImpl->config_));
    process->setMatrix(tr_inv);
    auto img_out = inspirecv::Image::Create(width_out, height_out, 3);
    std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(std::vector<int>{1, height_out, width_out, 3}, (uint8_t *)img_out.Data()));
    auto ret = process->convert(pImpl->buffer_, sw, sh, 0, tensor.get());
    INSPIREFACE_CHECK_MSG(ret == MNN::ErrorCode::NO_ERROR, "ImageProcess::convert failed");
    return img_out;
}

inspirecv::Image FrameProcess::ExecutePreviewImageProcessing(bool with_rotation) {
    return ExecuteImageScaleProcessing(pImpl->preview_scale_, with_rotation);
}

inspirecv::Image FrameProcess::ExecuteImageScaleProcessing(const float scale, bool with_rotation) {
    int sw = pImpl->width_;
    int sh = pImpl->height_;
    int rot_sw = sw;
    int rot_sh = sh;
    // MNN::CV::Matrix tr;
    std::shared_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(pImpl->config_));
    if (pImpl->rotation_mode_ == ROTATION_270 && with_rotation) {
        float srcPoints[] = {
          0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1),
        };
        float dstPoints[] = {
          (float)(pImpl->height_ * scale - 1), 0.0f, 0.0f, 0.0f, (float)(pImpl->height_ * scale - 1), (float)(pImpl->width_ * scale - 1), 0.0f,
          (float)(pImpl->width_ * scale - 1)};

        pImpl->tr_.setPolyToPoly((MNN::CV::Point *)dstPoints, (MNN::CV::Point *)srcPoints, 4);
        process->setMatrix(pImpl->tr_);
        int scaled_height = static_cast<int>(pImpl->width_ * scale);
        int scaled_width = static_cast<int>(pImpl->height_ * scale);
        inspirecv::Image img_out(scaled_width, scaled_height, 3);
        std::shared_ptr<MNN::Tensor> tensor(
          MNN::Tensor::create<uint8_t>(std::vector<int>{1, scaled_height, scaled_width, 3}, (uint8_t *)img_out.Data()));
        auto ret = process->convert(pImpl->buffer_, sw, sh, 0, tensor.get());
        INSPIREFACE_CHECK_MSG(ret == MNN::ErrorCode::NO_ERROR, "ImageProcess::convert failed");
        return img_out;
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
        pImpl->tr_.setPolyToPoly((MNN::CV::Point *)dstPoints, (MNN::CV::Point *)srcPoints, 4);
        process->setMatrix(pImpl->tr_);
        int scaled_height = static_cast<int>(pImpl->width_ * scale);
        int scaled_width = static_cast<int>(pImpl->height_ * scale);
        inspirecv::Image img_out(scaled_width, scaled_height, 3);
        std::shared_ptr<MNN::Tensor> tensor(
          MNN::Tensor::create<uint8_t>(std::vector<int>{1, scaled_height, scaled_width, 3}, (uint8_t *)img_out.Data()));
        auto ret = process->convert(pImpl->buffer_, sw, sh, 0, tensor.get());
        INSPIREFACE_CHECK_MSG(ret == MNN::ErrorCode::NO_ERROR, "ImageProcess::convert failed");
        return img_out;
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
        pImpl->tr_.setPolyToPoly((MNN::CV::Point *)dstPoints, (MNN::CV::Point *)srcPoints, 4);
        process->setMatrix(pImpl->tr_);
        int scaled_height = static_cast<int>(pImpl->height_ * scale);
        int scaled_width = static_cast<int>(pImpl->width_ * scale);
        inspirecv::Image img_out(scaled_width, scaled_height, 3);
        std::shared_ptr<MNN::Tensor> tensor(
          MNN::Tensor::create<uint8_t>(std::vector<int>{1, scaled_height, scaled_width, 3}, (uint8_t *)img_out.Data()));
        auto ret = process->convert(pImpl->buffer_, sw, sh, 0, tensor.get());
        INSPIREFACE_CHECK_MSG(ret == MNN::ErrorCode::NO_ERROR, "ImageProcess::convert failed");
        return img_out;
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
        pImpl->tr_.setPolyToPoly((MNN::CV::Point *)dstPoints, (MNN::CV::Point *)srcPoints, 4);
        process->setMatrix(pImpl->tr_);
        int scaled_height = static_cast<int>(pImpl->height_ * scale);
        int scaled_width = static_cast<int>(pImpl->width_ * scale);

        inspirecv::Image img_out(scaled_width, scaled_height, 3);
        std::shared_ptr<MNN::Tensor> tensor(
          MNN::Tensor::create<uint8_t>(std::vector<int>{1, scaled_height, scaled_width, 3}, (uint8_t *)img_out.Data()));
        auto ret = process->convert(pImpl->buffer_, sw, sh, 0, tensor.get());
        INSPIREFACE_CHECK_MSG(ret == MNN::ErrorCode::NO_ERROR, "ImageProcess::convert failed");
        return img_out;
    }
}

inspirecv::TransformMatrix FrameProcess::GetRotationModeAffineMatrix() const {
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

    MNN::CV::Matrix tr;
    tr.setPolyToPoly((MNN::CV::Point *)dstPoints, (MNN::CV::Point *)srcPoints, 4);
    
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