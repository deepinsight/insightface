#include <inspirecv/inspirecv.h>
#include <inspireface/image_process/nexus_processor/image_processor.h>
#include "log.h"
#include <inspireface/include/inspireface/spend_timer.h>
#include <inspireface/include/inspireface/herror.h>

using namespace inspire;

int main() {
#if defined(ISF_ENABLE_RGA)
    std::cout << "ISF_ENABLE_RGA is open" << std::endl;
#else
    std::cout << "ISF_ENABLE_RGA is close" << std::endl;
#endif

    auto img = inspirecv::Image::Create("data/bulk/r0.jpg");
    auto processor = nexus::ImageProcessor::Create();

    uint8_t* resized_data = nullptr;
    int resized_width = 100;
    int resized_height = 100;
    inspire::SpendTimer time_spend("RGA resize");
    for (int i = 0; i < 10; i++) {
        time_spend.Start();
        auto ret = processor->Resize(img.Data(), img.Width(), img.Height(), img.Channels(), &resized_data, resized_width, resized_height);
        time_spend.Stop();
        if (ret != 0) {
            INSPIRE_LOGE("RGA resize failed: %d", ret);
            return -1;
        }
    }
    processor->DumpCacheStatus();
    std::cout << time_spend << std::endl;

    auto resized_img = inspirecv::Image::Create(resized_width, resized_height, img.Channels(), resized_data);
    resized_img.Write("save_dir/kun_resized.jpg");

    processor->MarkDone();

    uint8_t* swapped_data = nullptr;
    inspire::SpendTimer swap_time_spend("RGA swap color");
    for (int i = 0; i < 10; i++) {
        swap_time_spend.Start();
        auto ret = processor->SwapColor(resized_img.Data(), resized_img.Width(), resized_img.Height(), resized_img.Channels(), &swapped_data);
        swap_time_spend.Stop();
    }
    std::cout << swap_time_spend << std::endl;
    processor->DumpCacheStatus();

    auto swapped_img = inspirecv::Image::Create(resized_img.Width(), resized_img.Height(), resized_img.Channels(), swapped_data);
    swapped_img.Write("save_dir/kun_swapped.jpg");

    processor->MarkDone();

    // padding
    uint8_t* padded_data = nullptr;
    int top = 10;
    int bottom = 10;
    int left = 10;
    int right = 10;
    inspire::SpendTimer padding_time_spend("RGA padding");
    int padded_width = 0;
    int padded_height = 0;
    for (int i = 0; i < 10; i++) {
        padding_time_spend.Start();
        auto ret = processor->Padding(swapped_img.Data(), swapped_img.Width(), swapped_img.Height(), swapped_img.Channels(), top, bottom, left, right,
                                      &padded_data, padded_width, padded_height);
        padding_time_spend.Stop();
    }
    processor->DumpCacheStatus();
    std::cout << padding_time_spend << std::endl;

    auto padded_img = inspirecv::Image::Create(padded_width, padded_height, swapped_img.Channels(), padded_data);
    padded_img.Write("save_dir/kun_padded.jpg");

    processor->MarkDone();

    // inspirecv crop
    inspirecv::Rect2i rect(30, 30, 70, 70);
    inspire::SpendTimer inspirecv_crop_time_spend("InspireCV crop");
    inspirecv::Image inspirecv_cropped_img;
    for (int i = 0; i < 10; i++) {
        inspirecv_crop_time_spend.Start();
        inspirecv_cropped_img = padded_img.Crop(rect);
        inspirecv_crop_time_spend.Stop();
    }
    std::cout << inspirecv_crop_time_spend << std::endl;
    inspirecv_cropped_img.Write("save_dir/kun_cropped_inspirecv.jpg");
    // Padding and crop
    inspirecv::Image image = inspirecv::Image::Create("data/bulk/r90.jpg");
    uint8_t* padded_cropped_data = nullptr;
    int dst_width = 320;
    int dst_height = 320;
    float scale = 0.0f;
    inspire::SpendTimer padded_crop_time_spend("RGA padded and cropped");
    for (int i = 0; i < 10; i++) {
        padded_crop_time_spend.Start();
        auto ret = processor->ResizeAndPadding(image.Data(), image.Width(), image.Height(), image.Channels(), dst_width, dst_height,
                                               &padded_cropped_data, scale);
        padded_crop_time_spend.Stop();
    }
    processor->DumpCacheStatus();
    std::cout << padded_crop_time_spend << std::endl;

    auto padded_cropped_img = inspirecv::Image::Create(dst_width, dst_height, img.Channels(), padded_cropped_data);
    padded_cropped_img.Write("save_dir/image_padded_cropped.jpg");

    processor->MarkDone();

    // resize 2
    uint8_t* resized_data_2 = nullptr;
    int resized_width_2 = 512;
    int resized_height_2 = 512;
    inspire::SpendTimer time_spend_2("RGA resize 2");
    for (int i = 0; i < 10; i++) {
        time_spend_2.Start();
        auto ret = processor->Resize(padded_cropped_img.Data(), padded_cropped_img.Width(), padded_cropped_img.Height(),
                                     padded_cropped_img.Channels(), &resized_data_2, resized_width_2, resized_height_2);
        time_spend_2.Stop();
    }
    std::cout << time_spend_2 << std::endl;
    processor->DumpCacheStatus();

    auto resized_img_2 = inspirecv::Image::Create(resized_width_2, resized_height_2, padded_cropped_img.Channels(), resized_data_2);
    resized_img_2.Write("save_dir/image_padded_cropped_resized_2.jpg");

    processor->MarkDone();

    return 0;
}
