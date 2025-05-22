#include <iostream>
#include <inspirecv/inspirecv.h>
#include <inspireface/inspireface.hpp>
#ifdef _WIN32
#include <direct.h>
#define CREATE_DIR(dir) _mkdir(dir)
#else
#include <sys/stat.h>
#define CREATE_DIR(dir) mkdir(dir, 0777)
#endif

int main() {
    // Make directory
    if (CREATE_DIR("cv") == -1) {
        // If the directory already exists, it is not an error
        if (errno != EEXIST) {
            std::cerr << "Error creating directory" << std::endl;
            return 1;
        }
    }

    /* Image I/O */

    // Load image from file
    // Load with 3 channels (BGR, like opencv)
    inspirecv::Image img = inspirecv::Image::Create("test_res/data/bulk/kun_cartoon_crop.jpg", 3);

    // Load image from buffer
    // uint8_t* buffer = ...;  // buffer is a pointer to the image data
    // bool is_alloc_mem = false;  // if true, will allocate memory for the image data,
    //                             // false is recommended to point to the original data to avoid copying
    // inspirecv::Image img_buffer = inspirecv::Image::Create(width, height, channel, buffer, is_alloc_mem);

    // Save image to file
    img.Write("cv/output.jpg");

    // Show image, warning: it must depend on opencv
    // img.Show("input");

    // Get pointer to image data
    const uint8_t* ptr = img.Data();

    /* Image Processing */
    // Convert to grayscale
    inspirecv::Image gray = img.ToGray();
    gray.Write("cv/gray.jpg");

    // Apply Gaussian blur
    inspirecv::Image blurred = img.GaussianBlur(3, 1.0);
    blurred.Write("cv/blurred.jpg");

    // Geometric transformations
    auto scale = 0.35;
    bool use_bilinear = true;
    inspirecv::Image resized = img.Resize(img.Width() * scale, img.Height() * scale, use_bilinear);  // Resize image
    resized.Write("cv/resized.jpg");

    // Rotate 90 degrees clockwise
    inspirecv::Image rotated = img.Rotate90();
    rotated.Write("cv/rotated.jpg");

    // Flip vertically
    inspirecv::Image flipped_vertical = img.FlipVertical();
    flipped_vertical.Write("cv/flipped_vertical.jpg");

    // Flip horizontally
    inspirecv::Image flipped_horizontal = img.FlipHorizontal();
    flipped_horizontal.Write("cv/flipped_horizontal.jpg");

    // Crop for rectangle
    inspirecv::Rect<int> rect = inspirecv::Rect<int>::Create(78, 41, 171, 171);
    inspirecv::Image cropped = img.Crop(rect);
    cropped.Write("cv/cropped.jpg");

    // Image padding
    int top = 50, bottom = 50, left = 50, right = 50;
    inspirecv::Image padded = img.Pad(top, bottom, left, right, inspirecv::Color::Black);
    padded.Write("cv/padded.jpg");

    // Swap red and blue channels
    inspirecv::Image swapped = img.SwapRB();
    swapped.Write("cv/swapped.jpg");

    // Multiply image by scale factor
    double scale_factor = 0.5;
    inspirecv::Image scaled = img.Mul(scale_factor);
    scaled.Write("cv/scaled.jpg");

    // Add value to image
    double value = -175;
    inspirecv::Image added = img.Add(value);
    added.Write("cv/added.jpg");

    // Rotate 90 degrees clockwise(also support 270 and 180)
    inspirecv::Image rotated_90 = img.Rotate90();
    rotated_90.Write("cv/rotated_90.jpg");

    // Affine transform
    /**
     * Create a transform matrix from the following matrix
     * [[a11, a12, tx],
     *  [a21, a22, ty]]
     *
     * Face crop transform matrix
     * [[0.0, -1.37626, 261.127],
     *  [1.37626, 0.0, 85.1831]]
     */
    float a11 = 0.0f;
    float a12 = -1.37626f;
    float a21 = 1.37626f;
    float a22 = 0.0f;
    float b1 = 261.127f;
    float b2 = 85.1831f;

    inspirecv::TransformMatrix trans = inspirecv::TransformMatrix::Create(a11, a12, b1, a21, a22, b2);
    int dst_width = 112;
    int dst_height = 112;
    inspirecv::Image affine = rotated_90.WarpAffine(trans, dst_width, dst_height);
    affine.Write("cv/affine.jpg");

    /* Image Draw */
    inspirecv::Image draw_img = img.Clone();

    // Draw a rectangle
    inspirecv::Rect<int> new_rect = rect.Square(1.1f);  // Square and expand the rect
    int thickness = 3;
    draw_img.DrawRect(new_rect, inspirecv::Color::Green, thickness);
    draw_img.Write("cv/draw_rect.jpg");

    // Draw a circle
    draw_img = img.Clone();
    std::vector<inspirecv::Point<int>> points = new_rect.As<int>().ToFourVertices();
    for (auto& point : points) {
        draw_img.DrawCircle(point, 1, inspirecv::Color::Red, 5);
    }
    draw_img.Write("cv/draw_circle.jpg");

    // Draw a line
    draw_img = img.Clone();
    draw_img.DrawLine(points[0], points[1], inspirecv::Color::Cyan, 2);
    draw_img.DrawLine(points[1], points[2], inspirecv::Color::Magenta, 2);
    draw_img.DrawLine(points[2], points[3], inspirecv::Color::Pink, 2);
    draw_img.DrawLine(points[3], points[0], inspirecv::Color::Yellow, 2);
    draw_img.Write("cv/draw_line.jpg");

    // Fill a rectangle
    draw_img = img.Clone();
    draw_img.Fill(new_rect, inspirecv::Color::Purple);
    draw_img.Write("cv/fill_rect.jpg");

    // Reset
    std::vector<uint8_t> gray_color(img.Width() * img.Height() * 3, 128);
    img.Reset(img.Width(), img.Height(), 3, gray_color.data());
    img.Write("cv/reset.jpg");

    /** FrameProcess */

    // BGR888 as raw data
    inspirecv::Image raw = inspirecv::Image::Create("test_res/data/bulk/kun_cartoon_crop_r90.jpg", 3);
    const uint8_t* buffer = raw.Data();

    // You can also use other image format, like NV21, NV12, RGBA, RGB, BGR, BGRA
    // const uint8_t* buffer = ...;

    // Create frame process
    auto width = raw.Width();
    auto height = raw.Height();
    auto rotation_mode = inspirecv::ROTATION_90;
    auto data_format = inspirecv::BGR;
    inspirecv::FrameProcess frame_process = inspirecv::FrameProcess::Create(buffer, height, width, data_format, rotation_mode);

    // Set preview size
    frame_process.SetPreviewSize(160);

    // Set preview scale
    // frame_process.SetPreviewScale(0.5f);

    // Get transform image
    inspirecv::Image transform_img = frame_process.ExecutePreviewImageProcessing(true);
    transform_img.Write("cv/transform_img.jpg");

    // ExecuteImageAffineProcessing

    // Face crop transform matrix
    // [[0.0, 0.726607, -61.8946],
    //  [-0.726607, 0.0, 189.737]]
    a11 = 0.0f;
    a12 = 0.726607f;
    a21 = -0.726607;
    a22 = 0.0f;
    b1 = -61.8946f;
    b2 = 189.737f;
    inspirecv::TransformMatrix affine_matrix = inspirecv::TransformMatrix::Create(a11, a12, b1, a21, a22, b2);
    dst_width = 112;
    dst_height = 112;
    inspirecv::Image affine_img = frame_process.ExecuteImageAffineProcessing(affine_matrix, dst_width, dst_height);
    affine_img.Write("cv/affine_img.jpg");

    return 0;
}
