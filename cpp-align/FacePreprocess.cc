//
// Created by Jack Yu on 23/03/2018.
// Anthor: Yong WU <https://github.com/wuyongchn/umeyama-cpp>
//

#include "FacePreprocess.h"

namespace FacePreprocess {
cv::Mat similarTransform(const cv::Mat& src, const cv::Mat& dst,
                         bool with_scale) {
  /* Mat layout
   * |x1, x2, x3, x4|
   * |y1, y2, y3, y4|
   */
  const int m = src.rows;  // dimension
  const int n = src.cols;  // number of measurements
  cv::Mat src_mean, dst_mean;

  // computation of mean
  src_mean = Internal::meanRow(src);
  dst_mean = Internal::meanRow(dst);

  // demeaning of src and dst points
  cv::Mat src_demean = Internal::demeanRow(src, src_mean);
  cv::Mat dst_demean = Internal::demeanRow(dst, dst_mean);

  // Eq. (36)-(37)
  double src_var = src_demean.dot(src_demean) / n;

  // Eq. (38)
  cv::Mat sigma = dst_demean * src_demean.t() / n;
  cv::SVD svd(sigma, cv::SVD::FULL_UV);

  // initialized the resulting transformation with an identity matrix...
  cv::Mat rt = cv::Mat::eye(m + 1, m + 1, CV_32FC1);

  // Eq. (39)
  cv::Mat s = cv::Mat::ones(m, 1, CV_32FC1);
  if (cv::determinant(svd.u) * cv::determinant(svd.vt) < 0) {
    s.at<float>(m - 1, 0) = -1;
  }

  // Eq. (40) and (43)
  rt.rowRange(0, m).colRange(0, m) = svd.u * cv::Mat::diag(s) * svd.vt;

  double scale = 1.0f;
  if (with_scale) {
    // Eq. (42)
    scale = scale / src_var * svd.w.dot(s);
  }
  // Eq. (41)
  cv::Mat top_left_mXm = rt.rowRange(0, m).colRange(0, m);
  cv::Mat col = dst_mean - scale * top_left_mXm * src_mean;
  col.copyTo(rt.rowRange(0, m).colRange(m, m + 1));
  top_left_mXm *= scale;
  return rt;
}
namespace Internal {
cv::Mat meanRow(const cv::Mat& src) {
  assert(src.channels() == 1);
  cv::Mat mean = cv::Mat(src.rows, 1, CV_32FC1);
  for (int i = 0; i < src.rows; ++i) {
    cv::Mat row = src.rowRange(i, i + 1);
    cv::Scalar mean_row = cv::mean(row);
    mean.at<float>(0, i) = mean_row[0];
  }
  return mean;
}

cv::Mat demeanRow(const cv::Mat& src, const cv::Mat& mean) {
  assert(src.channels() == 1 && src.rows == mean.rows);
  cv::Mat demean = src.clone();
  for (int i = 0; i < demean.rows; ++i) {
    cv::Mat row = demean.rowRange(i, i + 1);
    cv::subtract(row, mean.at<float>(0, i), row, cv::noArray(), CV_32FC1);
  }
  return demean;
}
}  // namespace Internal

}  // namespace FacePreprocess