/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_pose_quality_adapt.h"
#include "middleware/utils.h"

namespace inspire {

FacePoseQualityAdapt::FacePoseQualityAdapt() : AnyNetAdapter("FacePoseQuality") {}

FacePoseQualityAdaptResult FacePoseQualityAdapt::operator()(const inspirecv::Image &img) {
    FacePoseQualityAdaptResult res;
    AnyTensorOutputs outputs;
    if (img.Width() != INPUT_WIDTH || img.Height() != INPUT_HEIGHT) {
        uint8_t* resized_data = nullptr;
        m_processor_->Resize(img.Data(), img.Width(), img.Height(), img.Channels(), &resized_data, INPUT_WIDTH, INPUT_HEIGHT);
        auto resized = inspirecv::Image::Create(INPUT_WIDTH, INPUT_HEIGHT, img.Channels(), resized_data, false);
        Forward(resized, outputs);
    } else {
        Forward(img, outputs);
    }
    const auto &output = outputs[0].second;
    res.pitch = output[0] * 90;
    res.yaw = output[1] * 90;
    res.roll = output[2] * 90;
    std::vector<float> quality(output.begin() + 13, output.end());
    res.lmk_quality = quality;
    std::vector<float> face_pts5(output.begin() + 3, output.begin() + 13);
    res.lmk.resize(5);
    for (int i = 0; i < 5; i++) {
        res.lmk[i].SetX((face_pts5[i * 2] + 1) * (INPUT_WIDTH / 2));
        res.lmk[i].SetY((face_pts5[i * 2 + 1] + 1) * (INPUT_HEIGHT / 2));
    }

    return res;
}

inspirecv::TransformMatrix FacePoseQualityAdapt::ComputeCropMatrix(const inspirecv::Rect2i &rect) {
    auto padding_rect = rect.Square(1.5);
    std::vector<inspirecv::Point2i> rect_pts = padding_rect.ToFourVertices();
    std::vector<inspirecv::Point2i> dst_pts = {{0, 0}, {INPUT_WIDTH, 0}, {INPUT_WIDTH, INPUT_HEIGHT}, {0, INPUT_HEIGHT}};
    inspirecv::TransformMatrix m = inspirecv::SimilarityTransformEstimate(rect_pts, dst_pts);

    return m;
}

}  // namespace inspire