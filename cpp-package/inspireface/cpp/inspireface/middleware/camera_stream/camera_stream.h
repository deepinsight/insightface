#ifndef CAMERA_STREAM_H
#define CAMERA_STREAM_H
#include <memory>
#include "MNN/ImageProcess.hpp"
//#include "basic_types.h"
#include "opencv2/opencv.hpp"
#include "log.h"
//
namespace inspire {

/**
 * @brief Enum to represent rotation modes.
 */
enum ROTATION_MODE {
    ROTATION_0 = 0,
    ROTATION_90 = 1,
    ROTATION_180 = 2,
    ROTATION_270 = 3
};

/**
 * @brief Enum to represent rotation modes.
 */
enum DATA_FORMAT {
    NV21 = 0, NV12 = 1, RGBA = 2, RGB = 3, BGR = 4, BGRA = 5
};

/**
 * @brief A class to handle camera stream and image processing.
 */
class CameraStream {
public:

    CameraStream() {
        config_.sourceFormat = MNN::CV::YUV_NV21;
        config_.destFormat = MNN::CV::BGR;
        // config_.filterType = MNN::CV::BICUBIC;
        config_.filterType = MNN::CV::BILINEAR;
        // config_.filterType = MNN::CV::NEAREST;
        config_.wrap = MNN::CV::ZERO;
        rotation_mode_ = ROTATION_0;
        preview_size_ = 192;
//        preview_size_ = 352;
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
        preview_scale_ = preview_size_ / static_cast<float>(std::max(height,width));
    }

    /**
     * @brief Set the preview size.
     *
     * @param size Preview size.
     */
    void SetPreviewSize(const int size)
    {
        preview_size_ = size;
        preview_scale_ = preview_size_ / static_cast<float>(std::max(this->height_,this->width_));
    }

    /**
     * @brief Set the rotation mode.
     *
     * @param mode Rotation mode (e.g., ROTATION_0, ROTATION_90).
     */
    void SetRotationMode(ROTATION_MODE mode) { rotation_mode_ = mode; }

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
     * @brief Get an affine-transformed RGB image.
     *
     * @param affine_matrix Affine transformation matrix.
     * @param width_out Width of the output image.
     * @param height_out Height of the output image.
     * @return cv::Mat Affine-transformed RGB image.
     */
    cv::Mat GetAffineRGBImage(const cv::Mat &affine_matrix, const int width_out,
                              const int height_out) const {
        int sw = width_;
        int sh = height_;
        int rot_sw = sw;
        int rot_sh = sh;
        MNN::CV::Matrix tr;
        assert(affine_matrix.rows == 2);
        assert(affine_matrix.cols == 3);
        assert(affine_matrix.type() == CV_64F);
        cv::Mat trans_matrix;
        affine_matrix.convertTo(trans_matrix, CV_32F);
        std::vector<float> tr_cv({1, 0, 0, 0, 1, 0, 0, 0, 1});
        memcpy(tr_cv.data(), trans_matrix.data, sizeof(float) * 6);
        tr.set9(tr_cv.data());
        MNN::CV::Matrix tr_inv;
        tr.invert(&tr_inv);
        std::shared_ptr<MNN::CV::ImageProcess> process(
                MNN::CV::ImageProcess::create(config_));
        process->setMatrix(tr_inv);
        cv::Mat img_out(height_out, width_out, CV_8UC3);
        std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(
                std::vector<int>{1, height_out, width_out, 3}, img_out.data));
        process->convert(buffer_, sw, sh, 0, tensor.get());
//        std::cout << std::to_string(1) << std::endl;
        return img_out;
    }

    /**
     * @brief Get a preview image with optional rotation.
     *
     * @param with_rotation True if rotation is applied, false otherwise.
     * @return cv::Mat Preview image.
     */
    cv::Mat GetPreviewImage(bool with_rotation){
        return GetScaledImage(preview_scale_  , with_rotation);
    }

    /**
     * @brief Get the preview scale.
     *
     * @return float Preview scale.
     */
    float GetPreviewScale()
    {
        return preview_scale_;
    }

    /**
     * @brief Get a scaled image with optional rotation.
     *
     * @param scale Scaling factor.
     * @param with_rotation True if rotation is applied, false otherwise.
     * @return cv::Mat Scaled image.
     */
    cv::Mat GetScaledImage(const float scale, bool with_rotation) {
        int sw = width_;
        int sh = height_;
        int rot_sw = sw;
        int rot_sh = sh;
        // MNN::CV::Matrix tr;
        std::shared_ptr<MNN::CV::ImageProcess> process(
                MNN::CV::ImageProcess::create(config_));
        if (rotation_mode_ == ROTATION_270 && with_rotation) {
            float srcPoints[] = {
                    0.0f,
                    0.0f,
                    0.0f,
                    (float)(height_ - 1),
                    (float)(width_ - 1),
                    0.0f,
                    (float)(width_ - 1),
                    (float)(height_ - 1),
            };
            float dstPoints[] = {(float)(height_ * scale - 1),
                                 0.0f,
                                 0.0f,
                                 0.0f,
                                 (float)(height_ * scale - 1),
                                 (float)(width_ * scale - 1),
                                 0.0f,
                                 (float)(width_ * scale - 1)};

            tr_.setPolyToPoly((MNN::CV::Point *)dstPoints,
                              (MNN::CV::Point *)srcPoints, 4);
            process->setMatrix(tr_);
            int scaled_height = static_cast<int>(width_ * scale);
            int scaled_width = static_cast<int>(height_ * scale);
            cv::Mat img_out(scaled_height, scaled_width, CV_8UC3);
            std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(
                    std::vector<int>{1, scaled_height, scaled_width, 3}, img_out.data));
            process->convert(buffer_, sw, sh, 0, tensor.get());
            return img_out;
        } else if (rotation_mode_ == ROTATION_90 && with_rotation) {
            float srcPoints[] = {
                    0.0f,
                    0.0f,
                    0.0f,
                    (float)(height_ - 1),
                    (float)(width_ - 1),
                    0.0f,
                    (float)(width_ - 1),
                    (float)(height_ - 1),
            };
            float dstPoints[] = {
                    0.0f,
                    (float)(width_ * scale - 1),
                    (float)(height_ * scale - 1),
                    (float)(width_ * scale - 1),
                    0.0f,
                    0.0f,
                    (float)(height_ * scale - 1),
                    0.0f,
            };
            tr_.setPolyToPoly((MNN::CV::Point *)dstPoints,
                              (MNN::CV::Point *)srcPoints, 4);
            process->setMatrix(tr_);
            int scaled_height = static_cast<int>(width_ * scale);
            int scaled_width = static_cast<int>(height_ * scale);
            cv::Mat img_out(scaled_height, scaled_width, CV_8UC3);
            std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(
                    std::vector<int>{1, scaled_height, scaled_width, 3}, img_out.data));
            process->convert(buffer_, sw, sh, 0, tensor.get());
            return img_out;
        } else if (rotation_mode_ == ROTATION_180 && with_rotation) {
            float srcPoints[] = {
                    0.0f,
                    0.0f,
                    0.0f,
                    (float)(height_ - 1),
                    (float)(width_ - 1),
                    0.0f,
                    (float)(width_ - 1),
                    (float)(height_ - 1),
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
            tr_.setPolyToPoly((MNN::CV::Point *)dstPoints,
                              (MNN::CV::Point *)srcPoints, 4);
            process->setMatrix(tr_);
            int scaled_height = static_cast<int>(height_ * scale);
            int scaled_width = static_cast<int>(width_ * scale);
            cv::Mat img_out(scaled_height, scaled_width, CV_8UC3);
            std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(
                    std::vector<int>{1, scaled_height, scaled_width, 3}, img_out.data));
            process->convert(buffer_, sw, sh, 0, tensor.get());
            return img_out;
        } else {
            float srcPoints[] = {
                    0.0f,
                    0.0f,
                    0.0f,
                    (float)(height_ - 1),
                    (float)(width_ - 1),
                    0.0f,
                    (float)(width_ - 1),
                    (float)(height_ - 1),
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
            tr_.setPolyToPoly((MNN::CV::Point *)dstPoints,
                              (MNN::CV::Point *)srcPoints, 4);
            process->setMatrix(tr_);
            int scaled_height = static_cast<int>(height_ * scale);
            int scaled_width = static_cast<int>(width_ * scale);
            cv::Mat img_out(scaled_height, scaled_width, CV_8UC3);
            std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(
                    std::vector<int>{1, scaled_height, scaled_width, 3}, img_out.data));
            auto err = process->convert(buffer_, sw, sh, 0, tensor.get());
            return img_out;
        }
    }

    /**
     * @brief Get the affine transformation matrix.
     *
     * @return cv::Mat Affine transformation matrix.
     */
    cv::Mat GetAffineMatrix() const {
        cv::Mat affine_matrix(3, 3, CV_32F);
        tr_.get9((float *)affine_matrix.data);
        cv::Mat affine = affine_matrix.rowRange(0, 2);
        cv::Mat affine_64;
        affine.convertTo(affine_64, CV_64F);
        assert(affine_64.rows == 2);
        assert(affine_64.cols == 3);
        assert(affine_64.type() == CV_64F);
        return affine_64;
    }

    /**
     * @brief Get the height of the camera stream image.
     *
     * @return int Height.
     */
    int GetHeight() const{
        return height_;
    }

    /**
     * @brief Get the width of the camera stream image.
     *
     * @return int Width.
     */
    int GetWidth() const{
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
    const uint8_t *buffer_;                 ///< Pointer to the data buffer.
    int buffer_size_;                       ///< Size of the data buffer.
    std::vector<float> rotation_matrix;      ///< Rotation matrix.
    int height_;                            ///< Height of the camera stream image.
    int width_;                             ///< Width of the camera stream image.
    float preview_scale_;                    ///< Scaling factor for the preview image.
    int preview_size_;                      ///< Size of the preview image.
    MNN::CV::Matrix tr_;                    ///< Affine transformation matrix.
    ROTATION_MODE rotation_mode_;           ///< Current rotation mode.
    MNN::CV::ImageProcess::Config config_;    ///< Configuration for image processing.
    std::shared_ptr<MNN::CV::ImageProcess> process_; ///< Image processing instance.

};

} // inspire

#endif // CAMERA_STREAM_H
