/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_attribute_adapt.h"
#include "middleware/utils.h"

namespace inspire {

FaceAttributePredictAdapt::FaceAttributePredictAdapt() : AnyNetAdapter("FaceAttributePredictAdapt") {}

std::vector<int> FaceAttributePredictAdapt::operator()(const inspirecv::Image &bgr_affine) {
    AnyTensorOutputs outputs;
    if (bgr_affine.Width() != INPUT_WIDTH || bgr_affine.Height() != INPUT_HEIGHT) {
        auto resized = bgr_affine.Resize(INPUT_WIDTH, INPUT_HEIGHT);
        Forward(resized, outputs);
    } else {
        Forward(bgr_affine, outputs);
    }

    // cv::imshow("w", bgr_affine);
    // cv::waitKey(0);

    std::vector<float> &raceOut = outputs[0].second;
    std::vector<float> &genderOut = outputs[1].second;
    std::vector<float> &ageOut = outputs[2].second;

    auto raceIdx = argmax(raceOut.begin(), raceOut.end());
    auto genderIdx = argmax(genderOut.begin(), genderOut.end());
    auto ageIdx = argmax(ageOut.begin(), ageOut.end());

    std::string raceLabel = m_original_labels_[raceIdx];
    std::string simplifiedLabel = m_label_map_.at(raceLabel);
    int simplifiedRaceIdx = m_simplified_label_index_.at(simplifiedLabel);

    // std::cout << raceLabel << std::endl;
    // std::cout << simplifiedLabel << std::endl;

    return {simplifiedRaceIdx, 1 - (int)genderIdx, (int)ageIdx};
}

}  // namespace inspire