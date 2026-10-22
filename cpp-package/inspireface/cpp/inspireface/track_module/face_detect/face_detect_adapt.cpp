/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#include "face_detect_adapt.h"
#include "cost_time.h"
#include "herror.h"
#include "spend_timer.h"

namespace inspire {

FaceDetectAdapt::FaceDetectAdapt(int input_size, float nms_threshold, float cls_threshold)
: AnyNetAdapter("FaceDetectAdapt"), m_nms_threshold_(nms_threshold), m_cls_threshold_(cls_threshold), m_input_size_(input_size) {}

FaceLocList FaceDetectAdapt::operator()(const inspirecv::Image &bgr) {
    FaceLocList results;
    Detect(bgr, results);
    return results;
}

int32_t FaceDetectAdapt::Detect(const inspirecv::Image &bgr, FaceLocList &results) {
    results.clear();
    inspire::SpendTimer time_image_process("Image process");
    time_image_process.Start();
    float scale = 0.0f;
    inspirecv::Image pad;
    if (ResizeAndPadImageForInference(bgr, m_input_size_, m_input_size_, pad, scale) != InferenceWrapper::WrapperOk) {
        m_processor_->MarkDone();
        return HERR_DEVICE_IMAGE_PROCESS_FAILURE;
    }

    time_image_process.Stop();
    // std::cout << time_image_process << std::endl;
    // pad.Write("pad.jpg");
    //    LOGD("Prepare");
    AnyTensorViews outputs;
    inspire::SpendTimer time_forward("Forward");
    time_forward.Start();
    const int32_t forward_status = ForwardViews(pad, outputs);
    time_forward.Stop();
    if (forward_status != InferenceWrapper::WrapperOk || outputs.size() < 9) {
        INSPIRE_LOGE("Face detector inference returned invalid outputs");
        m_processor_->MarkDone();
        return HERR_SESS_TRACKER_FAILURE;
    }
    // std::cout << time_forward << std::endl;
    //    LOGD("Forward");

    inspire::SpendTimer time_decode("Decode");
    time_decode.Start();
    std::vector<int> strides = {8, 16, 32};
    for (int i = 0; i < strides.size(); ++i) {
        const int anchors = (m_input_size_ / strides[i]) * (m_input_size_ / strides[i]) * 2;
        if (outputs[i].size < static_cast<size_t>(anchors) || outputs[i + 3].size < static_cast<size_t>(anchors * 4) ||
            outputs[i + 6].size < static_cast<size_t>(anchors * 10)) {
            INSPIRE_LOGE("Face detector output tensor shape mismatch at stride %d", strides[i]);
            m_processor_->MarkDone();
            results.clear();
            return HERR_SESS_TRACKER_FAILURE;
        }
        _decode(outputs[i].data, outputs[i + 3].data, outputs[i + 6].data, strides[i], results);
    }
    time_decode.Stop();
    // std::cout << time_decode << std::endl;

    _nms(results, m_nms_threshold_);
    std::sort(results.begin(), results.end(),
              [](const FaceLoc &a, const FaceLoc &b) { return (a.y2 - a.y1) * (a.x2 - a.x1) > (b.y2 - b.y1) * (b.x2 - b.x1); });
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
    if (m_processor_->MarkDone() != 0) {
        results.clear();
        return HERR_DEVICE_IMAGE_PROCESS_FAILURE;
    }
    return HSUCCEED;
}

void FaceDetectAdapt::_nms(std::vector<FaceLoc> &input_faces, float nms_threshold) {
    std::sort(input_faces.begin(), input_faces.end(), [](const FaceLoc &a, const FaceLoc &b) { return a.score > b.score; });
    FaceLocList retained_faces;
    std::vector<float> retained_areas;
    retained_faces.reserve(input_faces.size());
    retained_areas.reserve(input_faces.size());

    for (const auto &candidate : input_faces) {
        const float candidate_area = (candidate.x2 - candidate.x1 + 1) * (candidate.y2 - candidate.y1 + 1);
        bool suppressed = false;
        for (size_t i = 0; i < retained_faces.size(); ++i) {
            const auto &retained = retained_faces[i];
            float xx1 = (std::max)(retained.x1, candidate.x1);
            float yy1 = (std::max)(retained.y1, candidate.y1);
            float xx2 = (std::min)(retained.x2, candidate.x2);
            float yy2 = (std::min)(retained.y2, candidate.y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (retained_areas[i] + candidate_area - inter);
            if (ovr >= nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            retained_faces.push_back(candidate);
            retained_areas.push_back(candidate_area);
        }
    }
    input_faces.swap(retained_faces);
}

void FaceDetectAdapt::_decode(const float *cls_pred, const float *box_pred, const float *lmk_pred, int stride, std::vector<FaceLoc> &results) {
    constexpr int kNumAnchors = 2;
    const int feature_height = m_input_size_ / stride;
    const int feature_width = m_input_size_ / stride;
    int anchor_index = 0;

    for (int y = 0; y < feature_height; ++y) {
        for (int x = 0; x < feature_width; ++x) {
            const float cx = static_cast<float>(x * stride);
            const float cy = static_cast<float>(y * stride);
            for (int anchor = 0; anchor < kNumAnchors; ++anchor, ++anchor_index) {
                if (cls_pred[anchor_index] > m_cls_threshold_) {
                    FaceLoc faceInfo;
                    float x1 = cx - box_pred[anchor_index * 4 + 0] * stride;
                    float y1 = cy - box_pred[anchor_index * 4 + 1] * stride;
                    float x2 = cx + box_pred[anchor_index * 4 + 2] * stride;
                    float y2 = cy + box_pred[anchor_index * 4 + 3] * stride;
                    faceInfo.x1 = x1;
                    faceInfo.y1 = y1;
                    faceInfo.x2 = x2;
                    faceInfo.y2 = y2;
                    faceInfo.score = cls_pred[anchor_index];
                    //            if (use_kps_) {
                    for (int j = 0; j < 5; ++j) {
                        float px = cx + lmk_pred[anchor_index * 10 + j * 2 + 0] * stride;
                        float py = cy + lmk_pred[anchor_index * 10 + j * 2 + 1] * stride;
                        faceInfo.lmk[j * 2 + 0] = px;
                        faceInfo.lmk[j * 2 + 1] = py;
                    }
                    //            }
                    results.push_back(faceInfo);
                }
            }
        }
    }
}

void FaceDetectAdapt::SetNmsThreshold(float mNmsThreshold) {
    m_nms_threshold_ = mNmsThreshold;
}

void FaceDetectAdapt::SetClsThreshold(float mClsThreshold) {
    m_cls_threshold_ = mClsThreshold;
}

int FaceDetectAdapt::GetInputSize() const {
    return m_input_size_;
}

}  // namespace inspire
