#include <iostream>
#include <inspirecv/inspirecv.h>
#include <inspireface/inspireface.hpp>
#include <inspireface/image_process/nexus_processor/image_processor.h>

void test_resize(std::unique_ptr<inspire::nexus::ImageProcessor>& processor, int aligned_width) {
    processor->SetAlignedWidth(aligned_width);
    // Create image
    inspirecv::Image img = inspirecv::Image::Create("kun.jpg", 3);
    // Resize image
    uint8_t* dst_data = nullptr;
    int dst_width = 112;
    int dst_height = 112;
    processor->Resize(img.Data(), img.Width(), img.Height(), 3, &dst_data, dst_width, dst_height);
    inspirecv::Image dst_img(dst_width, dst_height, 3, dst_data, false);
    dst_img.Write("dst_w" + std::to_string(aligned_width) + ".jpg");
    std::cout << "Save dst image to dst_w" << aligned_width << ".jpg" << std::endl;
}

int main() {
    // Create image processor
    auto processor = inspire::nexus::ImageProcessor::Create(inspire::Launch::IMAGE_PROCESSING_RGA);
    test_resize(processor, 4);
    test_resize(processor, 16);
    // wrong aligned width
    test_resize(processor, 7);

    return 0;
}