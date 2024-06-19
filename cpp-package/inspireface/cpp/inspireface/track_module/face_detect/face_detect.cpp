//
// Created by Tunm-Air13 on 2023/5/6.
//

#include "face_detect.h"

namespace inspire {

FaceDetect::FaceDetect(int input_size, float nms_threshold, float cls_threshold):
    AnyNet("FaceDetect"),
    m_nms_threshold_(nms_threshold),
    m_cls_threshold_(cls_threshold),
    m_input_size_(input_size){

}

FaceLocList FaceDetect::operator()(const Matrix &bgr) {
    int ori_w = bgr.cols;
    int ori_h = bgr.rows;
    int w, h;
    float scale;

    cv::Mat pad;
    if (ori_w == m_input_size_ && ori_h == m_input_size_) {
        // If the input image already matches the desired size, no need to resize, just pad
        cv::copyMakeBorder(bgr, pad, 0, 0, 0, 0, cv::BORDER_CONSTANT, 0.0f);
    }

    if (ori_w > ori_h) {
        scale = static_cast<float>(m_input_size_) / ori_w;
        w = m_input_size_;
        h = ori_h * scale;
    } else {
        scale = static_cast<float>(m_input_size_) / ori_h;
        h = m_input_size_;
        w = ori_w * scale;
    }
    int wpad = m_input_size_ - w;
    int hpad = m_input_size_ - h;
    cv::Mat resized_img;
    cv::resize(bgr, resized_img, cv::Size(w, h));
    cv::copyMakeBorder(resized_img, pad, 0, hpad, 0, wpad, cv::BORDER_CONSTANT, 0.0f);

//    LOGD("Prepare");
    AnyTensorOutputs outputs;
    Forward(pad, outputs);
//    LOGD("Forward");

    std::vector<FaceLoc> results;
    std::vector<int> strides = {8, 16, 32};
    for (int i = 0; i < strides.size(); ++i) {
        const std::vector<float> &tensor_cls = outputs[i].second;
        const std::vector<float> &tensor_box = outputs[i + 3].second;
        const std::vector<float> &tensor_lmk = outputs[i + 6].second;
        _decode(tensor_cls, tensor_box, tensor_lmk, strides[i], results);
    }

    _nms(results, m_nms_threshold_);
    std::sort(results.begin(), results.end(), [](FaceLoc a, FaceLoc b) { return (a.y2 -  a.y1) * (a.x2 - a.x1) > (b.y2 -  b.y1) * (b.x2 - b.x1); });
    for (auto &face:results) {
        face.x1 = face.x1 / scale;
        face.y1 = face.y1 / scale;
        face.x2 = face.x2 / scale;
        face.y2 = face.y2 / scale;
//        if(use_kps_) {
        for (int i = 0; i < 5; ++i) {
            face.lmk[i * 2 + 0] = face.lmk[i * 2 + 0] / scale;
            face.lmk[i * 2 + 1] = face.lmk[i * 2 + 1] / scale;
        }
//        }
    }
    return results;
}

void FaceDetect::_nms(std::vector<FaceLoc> &input_faces, float nms_threshold) {
    std::sort(input_faces.begin(), input_faces.end(), [](FaceLoc a, FaceLoc b) { return a.score > b.score; });
    std::vector<float> area(input_faces.size());
    for (int i = 0; i < int(input_faces.size()); ++i) {
        area[i] =
                (input_faces.at(i).x2 - input_faces.at(i).x1 + 1) * (input_faces.at(i).y2 - input_faces.at(i).y1 + 1);
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

void FaceDetect::_generate_anchors(int stride, int input_size, int num_anchors, std::vector<float> &anchors) {
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

void FaceDetect::_decode(const std::vector<float> &cls_pred, const std::vector<float> &box_pred, const std::vector<float>& lmk_pred, int stride, std::vector<FaceLoc> &results) {
    std::vector<float> anchors_center;
    _generate_anchors(stride, m_input_size_, 2, anchors_center);
//    const float *scores = cls_pred->host<float>();
//    const float *boxes = box_pred->host<float>();
//    float *lmk;
//    if(use_kps_)
//    const float *lmk = lmk_pred->host<float>();

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
        std::sort(results.begin(), results.end(), SortBoxSize);
    }
}

    void FaceDetect::SetNmsThreshold(float mNmsThreshold) {
        m_nms_threshold_ = mNmsThreshold;
    }

    void FaceDetect::SetClsThreshold(float mClsThreshold) {
        m_cls_threshold_ = mClsThreshold;
    }

    bool SortBoxSize(const FaceLoc &a, const FaceLoc &b) {
    int sq_a = (a.y2 - a.y1) * (a.x2 - a.x1);
    int sq_b = (b.y2 - b.y1) * (b.x2 - b.x1);
    return sq_a > sq_b;
}


}   // namespace