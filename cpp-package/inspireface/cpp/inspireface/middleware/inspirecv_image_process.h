#ifndef INSPIRECV_IMAGE_PROCESS_H
#define INSPIRECV_IMAGE_PROCESS_H

#include <memory>
#include <inspirecv/inspirecv.h>
#include <MNN/ImageProcess.hpp>
#include "isf_check.h"

// using namespace inspire;
namespace inspirecv {

/**
 * @brief Enum to represent rotation modes.
 */
enum ROTATION_MODE { ROTATION_0 = 0, ROTATION_90 = 1, ROTATION_180 = 2, ROTATION_270 = 3 };

/**
 * @brief Enum to represent data formats.
 */
enum DATA_FORMAT { NV21 = 0, NV12 = 1, RGBA = 2, RGB = 3, BGR = 4, BGRA = 5 };

/**
 * @brief A class to handle camera stream and image processing.
 */
class InspireImageProcess {
public:
    static InspireImageProcess Create(const uint8_t *data_buffer, int height, int width, DATA_FORMAT data_format = BGR,
                                      ROTATION_MODE rotation_mode = ROTATION_0) {
        InspireImageProcess process;
        process.SetDataBuffer(data_buffer, height, width);
        process.SetDataFormat(data_format);
        process.SetRotationMode(rotation_mode);
        return process;
    }

    InspireImageProcess() {
        SetDataFormat(NV21);
        SetDestFormat(BGR);
        config_.filterType = MNN::CV::BILINEAR;
        config_.wrap = MNN::CV::ZERO;
        rotation_mode_ = ROTATION_0;
        preview_size_ = 192;
        UpdateTransformMatrix();
    }

    /**
     * @brief Set the data buffer, height, and width of the camera stream.
     *
     * @param data_buffer Pointer to the data buffer.
     * @param height Height of the image.
     * @param width Width of the image.
     */
    void SetDataBuffer(const uint8_t *data_buffer, int height, int width) {
        this->buffer_ = data_buffer;
        this->height_ = height;
        this->width_ = width;
        preview_scale_ = preview_size_ / static_cast<float>(std::max(height, width));
        UpdateTransformMatrix();
    }

    /**
     * @brief Set the preview size.
     *
     * @param size Preview size.
     */
    void SetPreviewSize(const int size) {
        preview_size_ = size;
        preview_scale_ = preview_size_ / static_cast<float>(std::max(this->height_, this->width_));
        UpdateTransformMatrix();
    }

    void SetPreviewScale(const float scale) {
        preview_scale_ = scale;
        preview_size_ = static_cast<int>(preview_scale_ * std::max(this->height_, this->width_));
        UpdateTransformMatrix();
    }

    /**
     * @brief Set the rotation mode.
     *
     * @param mode Rotation mode (e.g., ROTATION_0, ROTATION_90).
     */
    void SetRotationMode(ROTATION_MODE mode) {
        rotation_mode_ = mode;
        UpdateTransformMatrix();
    }

    /**
     * @brief Set the data format.
     *
     * @param data_format Data format (e.g., NV21, RGBA).
     */
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
    }

    /**
     * @brief Set the destination format.
     *
     * @param data_format Data format (e.g., NV21, RGBA).
     */
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
    }

    /**
     * @brief Get an affine-transformed image.
     *
     * @param affine_matrix Affine transformation matrix.
     * @param width_out Width of the output image.
     * @param height_out Height of the output image.
     * @return cv::Mat Affine-transformed image.
     */
    inspirecv::Image ExecuteImageAffineProcessing(inspirecv::TransformMatrix &affine_matrix, const int width_out, const int height_out) const {
        int sw = width_;
        int sh = height_;
        int rot_sw = sw;
        int rot_sh = sh;
        MNN::CV::Matrix tr;
        std::vector<float> tr_cv({1, 0, 0, 0, 1, 0, 0, 0, 1});
        memcpy(tr_cv.data(), affine_matrix.Squeeze().data(), sizeof(float) * 6);
        tr.set9(tr_cv.data());
        MNN::CV::Matrix tr_inv;
        tr.invert(&tr_inv);
        std::shared_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config_));
        process->setMatrix(tr_inv);
        auto img_out = inspirecv::Image::Create(width_out, height_out, 3);
        std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(std::vector<int>{1, height_out, width_out, 3}, (uint8_t *)img_out.Data()));
        auto ret = process->convert(buffer_, sw, sh, 0, tensor.get());
        INSPIREFACE_CHECK_MSG(ret == MNN::ErrorCode::NO_ERROR, "ImageProcess::convert failed");
        return img_out;
    }

    /**
     * @brief Get a preview image with optional rotation.
     *
     * @param with_rotation True if rotation is applied, false otherwise.
     * @return cv::Mat Preview image.
     */
    inspirecv::Image ExecutePreviewImageProcessing(bool with_rotation) {
        return ExecuteImageScaleProcessing(preview_scale_, with_rotation);
    }

    /**
     * @brief Get the preview scale.
     *
     * @return float Preview scale.
     */
    float GetPreviewScale() {
        return preview_scale_;
    }

    /**
     * @brief Execute image scale processing.
     *
     * @param scale Scale factor.
     * @param with_rotation True if rotation is applied, false otherwise.
     * @return inspirecv::Image Scaled image.
     */
    inspirecv::Image ExecuteImageScaleProcessing(const float scale, bool with_rotation) {
        int sw = width_;
        int sh = height_;
        int rot_sw = sw;
        int rot_sh = sh;
        // MNN::CV::Matrix tr;
        std::shared_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config_));
        if (rotation_mode_ == ROTATION_270 && with_rotation) {
            float srcPoints[] = {
              0.0f, 0.0f, 0.0f, (float)(height_ - 1), (float)(width_ - 1), 0.0f, (float)(width_ - 1), (float)(height_ - 1),
            };
            float dstPoints[] = {(float)(height_ * scale - 1), 0.0f, 0.0f, 0.0f, (float)(height_ * scale - 1), (float)(width_ * scale - 1), 0.0f,
                                 (float)(width_ * scale - 1)};

            tr_.setPolyToPoly((MNN::CV::Point *)dstPoints, (MNN::CV::Point *)srcPoints, 4);
            process->setMatrix(tr_);
            int scaled_height = static_cast<int>(width_ * scale);
            int scaled_width = static_cast<int>(height_ * scale);
            inspirecv::Image img_out(scaled_width, scaled_height, 3);
            std::shared_ptr<MNN::Tensor> tensor(
              MNN::Tensor::create<uint8_t>(std::vector<int>{1, scaled_height, scaled_width, 3}, (uint8_t *)img_out.Data()));
            auto ret = process->convert(buffer_, sw, sh, 0, tensor.get());
            INSPIREFACE_CHECK_MSG(ret == MNN::ErrorCode::NO_ERROR, "ImageProcess::convert failed");
            return img_out;
        } else if (rotation_mode_ == ROTATION_90 && with_rotation) {
            float srcPoints[] = {
              0.0f, 0.0f, 0.0f, (float)(height_ - 1), (float)(width_ - 1), 0.0f, (float)(width_ - 1), (float)(height_ - 1),
            };
            float dstPoints[] = {
              0.0f, (float)(width_ * scale - 1), (float)(height_ * scale - 1), (float)(width_ * scale - 1), 0.0f, 0.0f, (float)(height_ * scale - 1),
              0.0f,
            };
            tr_.setPolyToPoly((MNN::CV::Point *)dstPoints, (MNN::CV::Point *)srcPoints, 4);
            process->setMatrix(tr_);
            int scaled_height = static_cast<int>(width_ * scale);
            int scaled_width = static_cast<int>(height_ * scale);
            inspirecv::Image img_out(scaled_width, scaled_height, 3);
            std::shared_ptr<MNN::Tensor> tensor(
              MNN::Tensor::create<uint8_t>(std::vector<int>{1, scaled_height, scaled_width, 3}, (uint8_t *)img_out.Data()));
            auto ret = process->convert(buffer_, sw, sh, 0, tensor.get());
            INSPIREFACE_CHECK_MSG(ret == MNN::ErrorCode::NO_ERROR, "ImageProcess::convert failed");
            return img_out;
        } else if (rotation_mode_ == ROTATION_180 && with_rotation) {
            float srcPoints[] = {
              0.0f, 0.0f, 0.0f, (float)(height_ - 1), (float)(width_ - 1), 0.0f, (float)(width_ - 1), (float)(height_ - 1),
            };
            float dstPoints[] = {
              (float)(width_ * scale - 1),
              (float)(height_ * scale - 1),
              (float)(width_ * scale - 1),
              0.0f,
              0.0f,
              (float)(height_ * scale - 1),
              0.0f,
              0.0f,
            };
            tr_.setPolyToPoly((MNN::CV::Point *)dstPoints, (MNN::CV::Point *)srcPoints, 4);
            process->setMatrix(tr_);
            int scaled_height = static_cast<int>(height_ * scale);
            int scaled_width = static_cast<int>(width_ * scale);
            inspirecv::Image img_out(scaled_width, scaled_height, 3);
            std::shared_ptr<MNN::Tensor> tensor(
              MNN::Tensor::create<uint8_t>(std::vector<int>{1, scaled_height, scaled_width, 3}, (uint8_t *)img_out.Data()));
            auto ret = process->convert(buffer_, sw, sh, 0, tensor.get());
            INSPIREFACE_CHECK_MSG(ret == MNN::ErrorCode::NO_ERROR, "ImageProcess::convert failed");
            return img_out;
        } else {
            float srcPoints[] = {
              0.0f, 0.0f, 0.0f, (float)(height_ - 1), (float)(width_ - 1), 0.0f, (float)(width_ - 1), (float)(height_ - 1),
            };
            float dstPoints[] = {
              0.0f,
              0.0f,
              0.0f,
              (float)(height_ * scale - 1),
              (float)(width_ * scale - 1),
              0.0f,
              (float)(width_ * scale - 1),
              (float)(height_ * scale - 1),
            };
            tr_.setPolyToPoly((MNN::CV::Point *)dstPoints, (MNN::CV::Point *)srcPoints, 4);
            process->setMatrix(tr_);
            int scaled_height = static_cast<int>(height_ * scale);
            int scaled_width = static_cast<int>(width_ * scale);

            inspirecv::Image img_out(scaled_width, scaled_height, 3);
            std::shared_ptr<MNN::Tensor> tensor(
              MNN::Tensor::create<uint8_t>(std::vector<int>{1, scaled_height, scaled_width, 3}, (uint8_t *)img_out.Data()));
            auto ret = process->convert(buffer_, sw, sh, 0, tensor.get());
            INSPIREFACE_CHECK_MSG(ret == MNN::ErrorCode::NO_ERROR, "ImageProcess::convert failed");
            return img_out;
        }
    }

    inspirecv::TransformMatrix GetAffineMatrix() const {
        auto affine_matrix = inspirecv::TransformMatrix::Create();
        affine_matrix[0] = tr_[0];
        affine_matrix[1] = tr_[1];
        affine_matrix[2] = tr_[2];
        affine_matrix[3] = tr_[3];
        affine_matrix[4] = tr_[4];
        affine_matrix[5] = tr_[5];
        return affine_matrix;
    }

    /**
     * @brief Get the height of the camera stream image.
     *
     * @return int Height.
     */
    int GetHeight() const {
        return height_;
    }

    /**
     * @brief Get the width of the camera stream image.
     *
     * @return int Width.
     */
    int GetWidth() const {
        return width_;
    }

    /**
     * @brief Get the current rotation mode.
     *
     * @return ROTATION_MODE Current rotation mode.
     */
    ROTATION_MODE getRotationMode() const {
        return rotation_mode_;
    }

private:
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

private:
    const uint8_t *buffer_;                           ///< Pointer to the data buffer.
    int buffer_size_;                                 ///< Size of the data buffer.
    std::vector<float> rotation_matrix;               ///< Rotation matrix.
    int height_;                                      ///< Height of the camera stream image.
    int width_;                                       ///< Width of the camera stream image.
    float preview_scale_;                             ///< Scaling factor for the preview image.
    int preview_size_;                                ///< Size of the preview image.
    MNN::CV::Matrix tr_;                              ///< Affine transformation matrix.
    ROTATION_MODE rotation_mode_;                     ///< Current rotation mode.
    MNN::CV::ImageProcess::Config config_;            ///< Configuration for image processing.
    std::shared_ptr<MNN::CV::ImageProcess> process_;  ///< Image processing instance.
};

}  // namespace inspirecv

#endif  // INSPIRECV_IMAGE_PROCESS_H