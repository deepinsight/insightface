//
// Created by Tunm-Air13 on 2023/9/8.
//

#include "face_attribute.h"
#include "middleware/utils.h"

namespace inspire {

FaceAttributePredict::FaceAttributePredict(): AnyNet("FaceAttributePredict") {}

std::vector<int> FaceAttributePredict::operator()(const Matrix& bgr_affine) {
    AnyTensorOutputs outputs;
    Forward(bgr_affine, outputs);
    // cv::imshow("w", bgr_affine);
    // cv::waitKey(0);

    std::vector<float> &raceOut = outputs[0].second;
    std::vector<float> &genderOut = outputs[1].second;
    std::vector<float> &ageOut = outputs[2].second;

    // for(int i = 0; i < raceOut.size(); i++) {
    //     std::cout << raceOut[i] << ", ";
    // }
    // std::cout << std::endl;
    
    auto raceIdx = argmax(raceOut.begin(), raceOut.end());
    auto genderIdx = argmax(genderOut.begin(), genderOut.end());
    auto ageIdx = argmax(ageOut.begin(), ageOut.end());

    std::string raceLabel = m_original_labels_[raceIdx];
    std::string simplifiedLabel = m_label_map_.at(raceLabel);
    int simplifiedRaceIdx = m_simplified_label_index_.at(simplifiedLabel);
    
    // std::cout << raceLabel << std::endl;
    // std::cout << simplifiedLabel << std::endl;

    return {simplifiedRaceIdx, 1 - (int )genderIdx, (int )ageIdx};
}

}   // namespace hyper