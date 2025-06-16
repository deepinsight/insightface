/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_detect_adapt.h"
#include "cost_time.h"
#include "spend_timer.h"

namespace inspire {

FaceDetectAdapt::FaceDetectAdapt(int input_size, float nms_threshold, float cls_threshold)
: AnyNetAdapter("FaceDetectAdapt"), m_nms_threshold_(nms_threshold), m_cls_threshold_(cls_threshold), m_input_size_(input_size) {}

FaceLocList FaceDetectAdapt::operator()(const inspirecv::Image &bgr) {
    inspire::SpendTimer time_image_process("Image process");
    time_image_process.Start();
    int ori_w = bgr.Width();
    int ori_h = bgr.Height();
    float scale;

    inspirecv::Image pad;

    uint8_t *resized_data = nullptr;
    if (ori_w == m_input_size_ && ori_h == m_input_size_) {
        scale = 1.0f;
        resized_data = (uint8_t *)bgr.Data();
    } else {
        m_processor_->ResizeAndPadding(bgr.Data(), bgr.Width(), bgr.Height(), bgr.Channels(), m_input_size_, m_input_size_, &resized_data, scale);
    }

    pad = inspirecv::Image::Create(m_input_size_, m_input_size_, bgr.Channels(), resized_data, false);

    time_image_process.Stop();
    // std::cout << time_image_process << std::endl;
    // pad.Write("pad.jpg");
    //    LOGD("Prepare");
    AnyTensorOutputs outputs;
    inspire::SpendTimer time_forward("Forward");
    time_forward.Start();
    Forward(pad, outputs);
    time_forward.Stop();
    // std::cout << time_forward << std::endl;
    //    LOGD("Forward");

    inspire::SpendTimer time_decode("Decode");
    time_decode.Start();
    std::vector<FaceLoc> results;
    std::vector<int> strides = {8, 16, 32};
    for (int i = 0; i < strides.size(); ++i) {
        const std::vector<float> &tensor_cls = outputs[i].second;
        const std::vector<float> &tensor_box = outputs[i + 3].second;
        const std::vector<float> &tensor_lmk = outputs[i + 6].second;
        _decode(tensor_cls, tensor_box, tensor_lmk, strides[i], results);
    }
    time_decode.Stop();
    // std::cout << time_decode << std::endl;

    _nms(results, m_nms_threshold_);
    std::sort(results.begin(), results.end(), [](FaceLoc a, FaceLoc b) { return (a.y2 - a.y1) * (a.x2 - a.x1) > (b.y2 - b.y1) * (b.x2 - b.x1); });
    for (auto &face : results) {
        face.x1 = face.x1 / scale;
        face.y1 = face.y1 / scale;
        face.x2 = face.x2 / scale;
        face.y2 = face.y2 / scale;
        for (int i = 0; i < 5; ++i) {
            face.lmk[i * 2 + 0] = face.lmk[i * 2 + 0] / scale;
            face.lmk[i * 2 + 1] = face.lmk[i * 2 + 1] / scale;
        }
    }
    m_processor_->MarkDone();

    return results;
}

void FaceDetectAdapt::_nms(std::vector<FaceLoc> &input_faces, float nms_threshold) {
    std::sort(input_faces.begin(), input_faces.end(), [](FaceLoc a, FaceLoc b) { return a.score > b.score; });
    std::vector<float> area(input_faces.size());
    for (int i = 0; i < int(input_faces.size()); ++i) {
        area[i] = (input_faces.at(i).x2 - input_faces.at(i).x1 + 1) * (input_faces.at(i).y2 - input_faces.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_faces.size()); ++i) {
        for (int j = i + 1; j < int(input_faces.size());) {
            float xx1 = (std::max)(input_faces[i].x1, input_faces[j].x1);
            float yy1 = (std::max)(input_faces[i].y1, input_faces[j].y1);
            float xx2 = (std::min)(input_faces[i].x2, input_faces[j].x2);
            float yy2 = (std::min)(input_faces[i].y2, input_faces[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (area[i] + area[j] - inter);
            if (ovr >= nms_threshold) {
                input_faces.erase(input_faces.begin() + j);
                area.erase(area.begin() + j);
            } else {
                j++;
            }
        }
    }
}

void FaceDetectAdapt::_generate_anchors(int stride, int input_size, int num_anchors, std::vector<float> &anchors) {
    int height = ceil(input_size / stride);
    int width = ceil(input_size / stride);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            for (int k = 0; k < num_anchors; ++k) {
                anchors.push_back(i * stride);
                anchors.push_back(j * stride);
            }
        }
    }
}

void FaceDetectAdapt::_decode(const std::vector<float> &cls_pred, const std::vector<float> &box_pred, const std::vector<float> &lmk_pred, int stride,
                              std::vector<FaceLoc> &results) {
    std::vector<float> anchors_center;
    _generate_anchors(stride, m_input_size_, 2, anchors_center);

    for (int i = 0; i < anchors_center.size() / 2; ++i) {
        if (cls_pred[i] > m_cls_threshold_) {
            FaceLoc faceInfo;
            float cx = anchors_center[i * 2 + 0];
            float cy = anchors_center[i * 2 + 1];
            float x1 = cx - box_pred[i * 4 + 0] * stride;
            float y1 = cy - box_pred[i * 4 + 1] * stride;
            float x2 = cx + box_pred[i * 4 + 2] * stride;
            float y2 = cy + box_pred[i * 4 + 3] * stride;
            faceInfo.x1 = x1;
            faceInfo.y1 = y1;
            faceInfo.x2 = x2;
            faceInfo.y2 = y2;
            faceInfo.score = cls_pred[i];
            //            if (use_kps_) {
            for (int j = 0; j < 5; ++j) {
                float px = cx + lmk_pred[i * 10 + j * 2 + 0] * stride;
                float py = cy + lmk_pred[i * 10 + j * 2 + 1] * stride;
                faceInfo.lmk[j * 2 + 0] = px;
                faceInfo.lmk[j * 2 + 1] = py;
            }
            //            }
            results.push_back(faceInfo);
        }
        std::sort(results.begin(), results.end(), SortBoxSizeAdapt);
    }
}

void FaceDetectAdapt::SetNmsThreshold(float mNmsThreshold) {
    m_nms_threshold_ = mNmsThreshold;
}

void FaceDetectAdapt::SetClsThreshold(float mClsThreshold) {
    m_cls_threshold_ = mClsThreshold;
}

bool SortBoxSizeAdapt(const FaceLoc &a, const FaceLoc &b) {
    int sq_a = (a.y2 - a.y1) * (a.x2 - a.x1);
    int sq_b = (b.y2 - b.y1) * (b.x2 - b.x1);
    return sq_a > sq_b;
}

int FaceDetectAdapt::GetInputSize() const {
    return m_input_size_;
}

}  // namespace inspire