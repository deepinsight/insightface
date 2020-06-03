//
// Created by Jack Yu on 23/03/2018.
// Anthor: Yong WU <https://github.com/wuyongchn/umeyama-cpp>
//

#ifndef FACE_DEMO_FACEPREPROCESS_H
#define FACE_DEMO_FACEPREPROCESS_H

#include <opencv2/opencv.hpp>

namespace FacePreprocess {
cv::Mat similarTransform(const cv::Mat& src, const cv::Mat& dst,
                         bool with_scale = true);
namespace Internal {
static cv::Mat meanRow(const cv::Mat& src);
static cv::Mat demeanRow(const cv::Mat& src, const cv::Mat& mean);
}  // namespace Internal

}  // namespace FacePreprocess
#endif  // FACE_DEMO_FACEPREPROCESS_H
