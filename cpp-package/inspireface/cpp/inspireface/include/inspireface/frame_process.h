#ifndef INSPIREFACE_FRAME_PROCESS_H
#define INSPIREFACE_FRAME_PROCESS_H

#include <memory>
#include <inspirecv/inspirecv.h>
#include "data_type.h"

namespace inspirecv {

/**
 * @brief Enum to represent rotation modes.
 */
enum ROTATION_MODE { ROTATION_0 = 0, ROTATION_90 = 1, ROTATION_180 = 2, ROTATION_270 = 3 };

/**
 * @brief Enum to represent data formats.
 */
enum DATA_FORMAT { NV21 = 0, NV12 = 1, RGBA = 2, RGB = 3, BGR = 4, BGRA = 5 , I420 = 6, GRAY = 7};

/**
 * @brief A class to handle camera stream and image processing.
 */
class INSPIRE_API_EXPORT FrameProcess {
public:
    /**
     * @brief Create a FrameProcess instance.
     *
     * @param data_buffer Pointer to the data buffer.
     * @param height Height of the image.
     * @param width Width of the image.
     * @param data_format Data format (e.g., NV21, RGBA).
     * @param rotation_mode Rotation mode (e.g., ROTATION_0, ROTATION_90).
     * @return FrameProcess instance.
     */
    static FrameProcess Create(const uint8_t* data_buffer, int height, int width, DATA_FORMAT data_format = BGR,
                               ROTATION_MODE rotation_mode = ROTATION_0);

    /**
     * @brief Create a FrameProcess instance from an inspirecv::Image.
     *
     * @param image The image to process.
     * @param data_format Data format (e.g., NV21, RGBA).
     * @param rotation_mode Rotation mode (e.g., ROTATION_0, ROTATION_90).
     * @return FrameProcess instance.
     */
    static FrameProcess Create(const inspirecv::Image& image, DATA_FORMAT data_format = BGR, ROTATION_MODE rotation_mode = ROTATION_0);

    /**
     * @brief Default constructor.
     */
    FrameProcess();

    /**
     * @brief Destructor.
     */
    ~FrameProcess();

    /**
     * @brief Copy constructor.
     */
    FrameProcess(const FrameProcess& other);

    /**
     * @brief Move constructor.
     */
    FrameProcess(FrameProcess&& other) noexcept;

    /**
     * @brief Copy assignment operator.
     */
    FrameProcess& operator=(const FrameProcess& other);

    /**
     * @brief Move assignment operator.
     */
    FrameProcess& operator=(FrameProcess&& other) noexcept;

    /**
     * @brief Set the data buffer, height, and width of the camera stream.
     *
     * @param data_buffer Pointer to the data buffer.
     * @param height Height of the image.
     * @param width Width of the image.
     */
    void SetDataBuffer(const uint8_t* data_buffer, int height, int width);

    /**
     * @brief Set the preview size.
     *
     * @param size Preview size.
     */
    void SetPreviewSize(const int size);

    /**
     * @brief Set the preview scale.
     *
     * @param scale Preview scale.
     */
    void SetPreviewScale(const float scale);

    /**
     * @brief Set the rotation mode.
     *
     * @param mode Rotation mode (e.g., ROTATION_0, ROTATION_90).
     */
    void SetRotationMode(ROTATION_MODE mode);

    /**
     * @brief Set the data format.
     *
     * @param data_format Data format (e.g., NV21, RGBA).
     */
    void SetDataFormat(DATA_FORMAT data_format);

    /**
     * @brief Set the destination format.
     *
     * @param data_format Data format (e.g., NV21, RGBA).
     */
    void SetDestFormat(DATA_FORMAT data_format);

    /**
     * @brief Get an affine-transformed image.
     *
     * @param affine_matrix Affine transformation matrix.
     * @param width_out Width of the output image.
     * @param height_out Height of the output image.
     * @return inspirecv::Image Affine-transformed image.
     */
    inspirecv::Image ExecuteImageAffineProcessing(inspirecv::TransformMatrix& affine_matrix, const int width_out, const int height_out) const;

    /**
     * @brief Get a preview image with optional rotation.
     *
     * @param with_rotation True if rotation is applied, false otherwise.
     * @return inspirecv::Image Preview image.
     */
    inspirecv::Image ExecutePreviewImageProcessing(bool with_rotation);

    /**
     * @brief Get the preview scale.
     *
     * @return float Preview scale.
     */
    float GetPreviewScale();

    /**
     * @brief Execute image scale processing.
     *
     * @param scale Scale factor.
     * @param with_rotation True if rotation is applied, false otherwise.
     * @return inspirecv::Image Scaled image.
     */
    inspirecv::Image ExecuteImageScaleProcessing(const float scale, bool with_rotation);

    /**
     * @brief Get the affine transformation matrix.
     *
     * @return inspirecv::TransformMatrix Affine transformation matrix.
     */
    inspirecv::TransformMatrix GetAffineMatrix() const;

    /**
     * @brief Get the rotation mode affine transformation matrix, scale coefficient is not included.
     *
     * @return inspirecv::TransformMatrix Rotation mode affine transformation matrix.
     */
    inspirecv::TransformMatrix GetRotationModeAffineMatrix() const;

    /**
     * @brief Get the height of the camera stream image.
     *
     * @return int Height.
     */
    int GetHeight() const;

    /**
     * @brief Get the width of the camera stream image.
     *
     * @return int Width.
     */
    int GetWidth() const;

    /**
     * @brief Get the current rotation mode.
     *
     * @return ROTATION_MODE Current rotation mode.
     */
    ROTATION_MODE getRotationMode() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

}  // namespace inspirecv

#endif  // INSPIREFACE_FRAME_PROCESS_H