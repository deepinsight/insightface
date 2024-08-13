
#ifndef TRACKING_LIB_UTILS_H
#define TRACKING_LIB_UTILS_H

//#include "face_attribute.h"
#include <cmath>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace inspire {

inline bool IsDirectory(const std::string& path) {
#ifdef _WIN32
    DWORD dwAttrib = GetFileAttributes(path.c_str());
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    } else {
        return false;
    }
#endif
}


inline void EstimateHeadPose(const std::vector<cv::Point2f> &current_shape,
                             cv::Vec3f &eav) {
    // head pose estimation by linear regression.
    static int HeadPosePointIndexs[] = {94, 59, 27, 20, 69, 45, 50};
    int *estimateHeadPosePointIndexs = HeadPosePointIndexs;
    static float estimateHeadPose2dArray[] = {
            0.139791, 27.4028, 7.02636, -2.48207, 9.59384, 6.03758, 1.27402,
            10.4795, 6.20801, 1.17406, 29.1886, 1.67768, 0.306761, -103.832,
            5.66238, 4.78663, 17.8726, -15.3623, -5.20016, 9.29488, -11.2495,
            -25.1704, 10.8649, -29.4877, -5.62572, 9.0871, -12.0982, -5.19707,
            -8.25251, 13.3965, -23.6643, -13.1348, 29.4322, 67.239, 0.666896,
            1.84304, -2.83223, 4.56333, -15.885, -4.74948, -3.79454, 12.7986,
            -16.1, 1.47175, 4.03941};

    cv::Mat estimateHeadPoseMat =
            cv::Mat(15, 3, CV_32FC1, estimateHeadPose2dArray);
    if (current_shape.empty())
        return;
    static const int samplePdim = 7;
    float miny = 10000000000.0f;
    float maxy = 0.0f;
    float sumx = 0.0f;
    float sumy = 0.0f;
    for (int i = 0; i < samplePdim; i++) {
        sumx += current_shape[i].x;
        float y = current_shape[i].y;
        sumy += y;
        if (miny > y)
            miny = y;
        if (maxy < y)
            maxy = y;
    }
    float dist = maxy - miny;
    sumx = sumx / samplePdim;
    sumy = sumy / samplePdim;
    static cv::Mat tmp(1, 2 * samplePdim + 1, CV_32FC1);
    for (int i = 0; i < samplePdim; i++) {
        tmp.at<float>(i) =
                (current_shape[estimateHeadPosePointIndexs[i]].x - sumx) / dist;
        tmp.at<float>(i + samplePdim) =
                (current_shape[estimateHeadPosePointIndexs[i]].y - sumy) / dist;
    }
    tmp.at<float>(2 * samplePdim) = 1.0f;
    cv::Mat predict = tmp * estimateHeadPoseMat;
    eav[0] = predict.at<float>(0);
    eav[1] = predict.at<float>(1);
    eav[2] = predict.at<float>(2);
}

inline void MinRect(const std::vector<cv::Point2f> &landmarks, int length,
                    float *rect) {
    rect[0] = landmarks[0].x;
    rect[1] = landmarks[0].y;
    rect[2] = landmarks[0].x;
    rect[3] = landmarks[0].y;
    for (int i = 0; i < length; i++) {
        if (rect[0] > landmarks[i].x)
            rect[0] = landmarks[i].x;
        else if (rect[2] < landmarks[i].x)
            rect[2] = landmarks[i].x;

        if (rect[1] > landmarks[i].y)
            rect[1] = landmarks[i].y;
        else if (rect[3] < landmarks[i].y)
            rect[3] = landmarks[i].y;
    }
}

inline float PointDistance(const cv::Point2f &a, const cv::Point2f &b) {
    float norm = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    return sqrt(norm);
}

inline cv::Point2f MeanPoint(const std::vector<cv::Point2f> &points) {
    assert(points.size() > 0);
    cv::Point2f mean;
    for (const auto &p: points)
        mean += p;
    mean /= static_cast<int>(points.size());
    return mean;
}

inline void BestFitRect(const std::vector<cv::Point2f> &pre_landmarks, int size,
                        std::vector<cv::Point2f> &src_fit) {
    src_fit.resize(pre_landmarks.size());
    std::vector<float> mean_shape_box = {56, 56, 92, 102};
    float rect[4];
    MinRect(pre_landmarks, size, rect);
    float points_height = rect[3] - rect[1];
    float points_centerx = (rect[0] + rect[2]) / 2;
    float points_centery = (rect[1] + rect[3]) / 2;
    float meanshape_centerx = mean_shape_box[0];
    float meanshape_centery = mean_shape_box[1];
    float scaleHeight = mean_shape_box[3] / points_height;
    float scale = scaleHeight;
    for (int i = 0; i < size; i++) {
        src_fit[i].x =
                pre_landmarks[i].x * scale - points_centerx * scale + meanshape_centerx;
        src_fit[i].y =
                pre_landmarks[i].y * scale - points_centery * scale + meanshape_centery;
    }
}

inline void
SimilarityTransformEstimate(const std::vector<cv::Point2f> &src_points,
                            const std::vector<cv::Point2f> &dst_points,
                            cv::Mat &matrix) {

    assert(matrix.rows == 2);
    assert(matrix.cols == 3);
    // matrix.create(2,3,CV_64F);
    assert(src_points.size() == dst_points.size());
    cv::Point2f src_mean = MeanPoint(src_points);
    cv::Point2f dst_mean = MeanPoint(dst_points);
    // matrix.resize(6);
    float src_norm2 = 0.f;
    float sum_a = 0.f;
    float sum_b = 0.f;

    for (int i = 0; i < src_points.size(); i++) {
        cv::Point2f src_d = src_points[i] - src_mean;
        cv::Point2f dst_d = dst_points[i] - dst_mean;

        src_norm2 += src_d.x * src_d.x + src_d.y * src_d.y;
        sum_a += src_d.x * dst_d.x + src_d.y * dst_d.y;
        sum_b += src_d.x * dst_d.y - src_d.y * dst_d.x;
    }

    if (std::fabs(src_norm2) < std::numeric_limits<float>::epsilon()) {
        float a = 1.f;
        float b = 0.f;
        float tx = dst_mean.x - src_mean.x;
        float ty = dst_mean.y - src_mean.y;
        matrix.at<double>(0, 0) = a;
        matrix.at<double>(0, 1) = -b;
        matrix.at<double>(0, 2) = tx;
        matrix.at<double>(1, 0) = b;
        matrix.at<double>(1, 1) = a;
        matrix.at<double>(1, 2) = ty;
    } else {
        float a = sum_a / src_norm2;
        float b = sum_b / src_norm2;
        float tx = dst_mean.x - (a * src_mean.x - b * src_mean.y);
        float ty = dst_mean.y - (b * src_mean.x + a * src_mean.y);
        matrix.at<double>(0, 0) = a;
        matrix.at<double>(0, 1) = -b;
        matrix.at<double>(0, 2) = tx;
        matrix.at<double>(1, 0) = b;
        matrix.at<double>(1, 1) = a;
        matrix.at<double>(1, 2) = ty;
    }
}

inline void
SimilarityTransformEstimate(const std::vector<cv::Point2f> &src_points,
                            const std::vector<cv::Point2f> &dst_points,
                            std::vector<float> &matrix) {
    assert(src_points.size() == dst_points.size());
    cv::Point2f src_mean = MeanPoint(src_points);
    cv::Point2f dst_mean = MeanPoint(dst_points);
    matrix.resize(6);
    float src_norm2 = 0.f;
    float sum_a = 0.f;
    float sum_b = 0.f;

    for (int i = 0; i < src_points.size(); i++) {
        cv::Point2f src_d = src_points[i] - src_mean;
        cv::Point2f dst_d = dst_points[i] - dst_mean;

        src_norm2 += src_d.x * src_d.x + src_d.y * src_d.y;
        sum_a += src_d.x * dst_d.x + src_d.y * dst_d.y;
        sum_b += src_d.x * dst_d.y - src_d.y * dst_d.x;
    }

    if (std::fabs(src_norm2) < std::numeric_limits<float>::epsilon()) {
        float a = 1.f;
        float b = 0.f;
        float tx = dst_mean.x - src_mean.x;
        float ty = dst_mean.y - src_mean.y;
        matrix[0] = a;
        matrix[1] = -b;
        matrix[2] = tx;
        matrix[3] = b;
        matrix[4] = a;
        matrix[5] = ty;
    } else {
        float a = sum_a / src_norm2;
        float b = sum_b / src_norm2;
        float tx = dst_mean.x - (a * src_mean.x - b * src_mean.y);
        float ty = dst_mean.y - (b * src_mean.x + a * src_mean.y);
        matrix[0] = a;
        matrix[1] = -b;
        matrix[2] = tx;
        matrix[3] = b;
        matrix[4] = a;
        matrix[5] = ty;
    }
}

inline cv::Mat GetRectSquareAffine(cv::Rect rect, float win_size = 112) {
    assert(rect.height == rect.width);
    std::vector<cv::Point2f> dst_pts = {
            {0,        0},
            {win_size, 0},
            {win_size, win_size}};
    float x1 = static_cast<float>(rect.x);
    float y1 = static_cast<float>(rect.y);
    float x2 = static_cast<float>(rect.x + rect.width);
    float y2 = static_cast<float>(rect.y + rect.height);
    std::vector<cv::Point2f> src_pts = {{x1, y1},
                                        {x2, y1},
                                        {x2, y2}};
    cv::Mat m = cv::getAffineTransform(src_pts, dst_pts);
    return m;
}

inline cv::Mat SquareToSquare(cv::Rect src, cv::Rect dst,
                              float win_size = 112) {
    float src_x1 = static_cast<float>(src.x);
    float src_y1 = static_cast<float>(src.y);
    float src_x2 = static_cast<float>(src.x + src.width);
    float src_y2 = static_cast<float>(src.y + src.height);

    float dst_x1 = static_cast<float>(dst.x);
    float dst_y1 = static_cast<float>(dst.y);
    float dst_x2 = static_cast<float>(dst.x + dst.width);
    float dst_y2 = static_cast<float>(dst.y + dst.height);

    std::vector<cv::Point2f> src_pts = {
            {src_x1, src_y1},
            {src_x2, src_y1},
            {src_x2, src_y2}};
    std::vector<cv::Point2f> dst_pts = {
            {dst_x1, dst_y1},
            {dst_x2, dst_y1},
            {dst_x2, dst_y2}};
    cv::Mat m = cv::getAffineTransform(src_pts, dst_pts);
    return m;
}

inline std::vector<cv::Point2f>
ApplyTransformToPoints(const std::vector<cv::Point2f> &points,
                       const cv::Mat &matrix) {
    assert(matrix.rows == 2);
    assert(matrix.cols == 3);
    double m00 = matrix.at<double>(0, 0);
    double m01 = matrix.at<double>(0, 1);
    double m02 = matrix.at<double>(0, 2);
    double m10 = matrix.at<double>(1, 0);
    double m11 = matrix.at<double>(1, 1);
    double m12 = matrix.at<double>(1, 2);
    std::vector<cv::Point2f> out_points(points.size());
    assert(out_points.size() == points.size());
    for (int j = 0; j < points.size(); j++) {
        out_points[j].x = points[j].x * m00 + points[j].y * m01 + m02;
        out_points[j].y = points[j].x * m10 + points[j].y * m11 + m12;
    }
    return out_points;
}

inline std::vector<cv::Point2f>
FixPointsMeanshape(std::vector<cv::Point2f> &points,
                   const std::vector<cv::Point2f> &mean_shape) {

    cv::Rect bbox = cv::boundingRect(points);
    int R = std::max(bbox.height, bbox.width);
    int cx = bbox.x + bbox.width / 2;
    int cy = bbox.y + bbox.height / 2;
    cv::Rect old(cx - R / 2, cy - R / 2, R, R);

    cv::Rect mean_shape_box = cv::boundingRect(mean_shape);
    int m_R = std::max(mean_shape_box.height, mean_shape_box.width);
    int m_cx = mean_shape_box.x + mean_shape_box.width / 2;
    int m_cy = mean_shape_box.y + mean_shape_box.height / 2;
    cv::Rect _new(m_cx - m_R / 2, m_cy - m_R / 2, m_R, m_R);
    cv::Mat affine = SquareToSquare(old, _new);
    std::vector<cv::Point2f> new_pts = ApplyTransformToPoints(points, affine);
    return new_pts;
}

inline std::vector<cv::Point2f> FixPoints(std::vector<cv::Point2f> &points) {
    // RotPoints(points, -2);
    cv::Rect bbox = cv::boundingRect(points);
    int R = std::max(bbox.height, bbox.width);
    int cx = bbox.x + bbox.width / 2;
    int cy = bbox.y + bbox.height / 2;
    cv::Rect old(cx - R / 2, cy - R / 2, R, R);
    int margin = 0;
    int zx = 0;
    int zy = 5;
    int new_x1 = zx + margin;
    int new_y1 = zy + margin;
    int new_x2 = 112 + zx - margin;
    int new_y2 = 112 + zy - margin;
    cv::Rect _new(new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1);
    cv::Mat affine = SquareToSquare(old, _new);
    std::vector<cv::Point2f> new_pts = ApplyTransformToPoints(points, affine);
    return new_pts;
}

inline cv::Rect ComputeSafeRect(const cv::Rect &region, int height, int width) {
    int x1 = region.x;
    int y1 = region.y;
    int x2 = region.x + region.width;
    int y2 = region.y + region.height;
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);

    x2 = std::min(x2, width - 1);
    y2 = std::min(y2, height - 1);
    cv::Rect safe_rect(x1, y1, x2 - x1, y2 - y1);
    return safe_rect;
}

inline void Transform(const std::vector<cv::Point2f> &pre_landmarks,
                      float *src_fit, const float *meanshape, const int size,
                      std::vector<float> &rotation,
                      std::vector<float> &rotation_inv) {
    std::vector<float> src(size * 2);
    std::vector<float> dst(size * 2);

    float src_mean_x = 0, src_mean_y = 0, dst_mean_x = 0, dst_mean_y = 0;
    float tx, ty;
    float src_norm = 0, sum_a = 0, sum_b = 0;
    for (int i = 0; i < size; i++) {
        src_mean_x += pre_landmarks[i].x;
        src_mean_y += pre_landmarks[i].y;
        dst_mean_x += src_fit[2 * i];
        dst_mean_y += meanshape[2 * i + 1];
    }
    src_mean_x = src_mean_x / size;
    src_mean_y = src_mean_y / size;
    dst_mean_x = dst_mean_x / size;
    dst_mean_y = dst_mean_y / size * 1.1;
    // dst_mean_y = dst_mean_y / size * 1.1;

    for (int i = 0; i < size; i++) {
        src[2 * i] = pre_landmarks[i].x - src_mean_x;
        src[2 * i + 1] = pre_landmarks[i].y - src_mean_y;
        src_norm += src[2 * i] * src[2 * i];
        src_norm += src[2 * i + 1] * src[2 * i + 1];
        dst[2 * i] = meanshape[2 * i] - dst_mean_x;
        dst[2 * i + 1] = meanshape[2 * i + 1] - dst_mean_y;

        sum_a += src[2 * i] * dst[2 * i] + src[2 * i + 1] * dst[2 * i + 1];
        sum_b += src[2 * i] * dst[2 * i + 1] - src[2 * i + 1] * dst[2 * i];
    }
    sum_a = sum_a / src_norm;
    sum_b = sum_b / src_norm;

    tx = dst_mean_x - sum_a * src_mean_x + sum_b * src_mean_y;
    ty = dst_mean_y - sum_b * src_mean_x - sum_a * src_mean_y;

    rotation.clear();
    rotation.push_back(sum_a);
    rotation.push_back(-sum_b);
    rotation.push_back(tx);
    rotation.push_back(sum_b);
    rotation.push_back(sum_a);
    rotation.push_back(ty);

    double m[6];
    m[0] = rotation[0];
    m[1] = rotation[1];
    m[2] = rotation[2];
    m[3] = rotation[3];
    m[4] = rotation[4];
    m[5] = rotation[5];

    double D = m[0] * m[4] - m[1] * m[3];
    D = D != 0 ? 1. / D : 0;
    double A11 = m[4] * D, A22 = m[0] * D;
    m[0] = A11;
    m[1] *= -D;
    m[3] *= -D;
    m[4] = A22;
    double b1 = -m[0] * m[2] - m[1] * m[5];
    double b2 = -m[3] * m[2] - m[4] * m[5];
    m[2] = b1;
    m[5] = b2;

    rotation_inv.clear();
    rotation_inv.push_back(m[0]);
    rotation_inv.push_back(m[1]);
    rotation_inv.push_back(m[2]);
    rotation_inv.push_back(m[3]);
    rotation_inv.push_back(m[4]);
    rotation_inv.push_back(m[5]);
}

inline std::vector<cv::Point2f> Rect2Points(const cv::Rect rect) {
    float x1 = static_cast<float>(rect.x);
    float y1 = static_cast<float>(rect.y);
    float x2 = static_cast<float>(rect.x + rect.width);
    float y2 = static_cast<float>(rect.y + rect.height);
    std::vector<cv::Point2f> src_pts = {{x1, y1},
                                        {x2, y1},
                                        {x2, y2},
                                        {x1, y2}};
    return src_pts;
}

inline std::vector<cv::Point2f> Rect2Points(const cv::Rect2f rect) {
    float x1 = rect.x;
    float y1 = rect.y;
    float x2 = rect.x + rect.width;
    float y2 = rect.y + rect.height;
    std::vector<cv::Point2f> src_pts = {{x1, y1},
                                        {x2, y1},
                                        {x2, y2},
                                        {x1, y2}};
    return src_pts;
}

inline cv::Mat ScaleAffineMatrix(const cv::Mat &affine, float scale,
                                 int origin_width, int origin_height,
                                 int new_width, int new_height) {
    std::vector<cv::Point2f> origin_pts =
            Rect2Points(cv::Rect(0, 0, origin_width, origin_height));
    cv::Mat affine_inv;
    cv::invertAffineTransform(affine, affine_inv);
    std::vector<cv::Point2f> screen_pts =
            ApplyTransformToPoints(origin_pts, affine_inv);
    cv::Point2f center;
    for (auto &one: screen_pts) {
        center.x += one.x * 0.25f;
        center.y += one.y * 0.25f;
    }

    screen_pts[0].x = center.x + (screen_pts[0].x - center.x) * scale;
    screen_pts[0].y = center.y + (screen_pts[0].y - center.y) * scale;

    screen_pts[1].x = center.x + (screen_pts[1].x - center.x) * scale;
    screen_pts[1].y = center.y + (screen_pts[1].y - center.y) * scale;

    screen_pts[2].x = center.x + (screen_pts[2].x - center.x) * scale;
    screen_pts[2].y = center.y + (screen_pts[2].y - center.y) * scale;

    screen_pts[3].x = center.x + (screen_pts[3].x - center.x) * scale;
    screen_pts[3].y = center.y + (screen_pts[3].y - center.y) * scale;

    std::vector<cv::Point2f> new_pts =
            Rect2Points(cv::Rect(0, 0, new_width, new_height));
    screen_pts.pop_back();
    new_pts.pop_back();
    cv::Mat m = cv::getAffineTransform(screen_pts, new_pts);
    return m;
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

inline void RotPoints(std::vector<cv::Point2f> &pts, float angle) {
    float angle_rad = angle * 3.1415 / 180;
    float m11 = cos(angle_rad);
    float m12 = -sin(angle_rad);
    float m21 = sin(angle_rad);
    float m22 = cos(angle_rad);
    for (auto &one: pts) {
        one.x = one.x * m11 + one.y * m12;
        one.y = one.x * m21 + one.y * m22;
    }
}

inline cv::Rect flipRectWidth(const cv::Rect &rect, const cv::Size &size) {
    int x1 = rect.x;
    int y1 = rect.y;
    int x2 = rect.x + rect.width;
    int y2 = rect.y + rect.height;
    x1 = size.width - (rect.x + rect.width);
    x2 = size.width - rect.x;
//    __android_log_print(ANDROID_LOG_ERROR, "flip: ", "[[%d, %d], [%d, %d]]",x1, y1, x2, y2);


    return cv::Rect(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
}

inline std::vector<cv::Point2f> RotatePoints(const std::vector<cv::Point2f>& points, float degree,
                         const cv::Size &image_size) {
    int width = image_size.width;
    int height = image_size.height;
    float radians = degree / 180 * CV_PI;
    int heightNew = int(width * fabs(sin(radians)) + height * fabs(cos(radians)));
    int widthNew = int(height * fabs(sin(radians)) + width * fabs(cos(radians)));
    cv::Mat trans(2, 3, CV_32F);
    trans = cv::getRotationMatrix2D(cv::Point2f(width / 2, height / 2), degree, 1);
    trans.at<double>(0, 2) += (widthNew - width) / 2;
    trans.at<double>(1, 2) += (heightNew - height) / 2;
    float point3_array[points.size()][3];
    for (int i = 0; i < points.size(); ++i) {
        point3_array[i][0] = points[i].x;
        point3_array[i][1] = points[i].y;
        point3_array[i][2] = 1;
    }
    trans.convertTo(trans, CV_32F);
    cv::Mat mat_point3s(points.size(), 3, CV_32F, point3_array);
//    std::cout << mat_point3s << std::endl;
    cv::Mat trans_result = mat_point3s * trans.t();
    return cv::Mat_<cv::Point2f>(trans_result);
}

inline cv::Mat RotateRect(cv::Rect &rect, std::vector<cv::Point2f> &dst,
                          cv::Rect &trans_rect, float degree, const cv::Size &image_size) {
    int width = image_size.width;
    int height = image_size.height;
    float radians = degree / 180 * CV_PI;
    int heightNew = int(width * fabs(sin(radians)) + height * fabs(cos(radians)));
    int widthNew = int(height * fabs(sin(radians)) + width * fabs(cos(radians)));
    cv::Mat trans(2, 3, CV_32F);
    trans = cv::getRotationMatrix2D(cv::Point2f(width / 2, height / 2), degree, 1);
    trans.at<double>(0, 2) += (widthNew - width) / 2;
    trans.at<double>(1, 2) += (heightNew - height) / 2;
    float xmin = rect.x;
    float ymin = rect.y;
    float xmax = rect.x + rect.width;
    float ymax = rect.y + rect.height;
    float points[][3] = {{xmin, ymin, 1},
                         {xmax, ymin, 1},
                         {xmax, ymax, 1},
                         {xmin, ymax, 1}};
    trans.convertTo(trans, CV_32F);
    cv::Mat t_points(4, 3, CV_32F, points);
    cv::Mat trans_points = t_points * trans.t();
    dst.clear();
    for (int i = 0; i < 4; ++i) {
        float x = trans_points.at<float>(i, 0);
        float y = trans_points.at<float>(i, 1);
        dst.emplace_back(x, y);
    }
    float min_x = std::numeric_limits<float>::max(), max_x = 0;
    float min_y = std::numeric_limits<float>::max(), max_y = 0;
    for (int i = 0; i < dst.size(); ++i) {
        if (dst[i].x < min_x) {
            min_x = dst[i].x;
        }
        if (dst[i].x > max_x) {
            max_x = dst[i].x;
        }
        if (dst[i].y < min_y) {
            min_y = dst[i].y;
        }
        if (dst[i].y > max_y) {
            max_y = dst[i].y;
        }
    }
    trans_rect = cv::Rect(cv::Point2f(min_x, min_y), cv::Point2f(max_x, max_y));
//    trans_rect = flipRectWidth(trans_rect, cv::Size(widthNew, heightNew));
    return trans;
}

// Structure to hold bounding box coordinates
struct BoundingBox {
    int left_top_x;
    int left_top_y;
    int right_bottom_x;
    int right_bottom_y;
};

inline cv::Rect GetNewBox(int src_w, int src_h, cv::Rect bbox, float scale) {
    // Convert cv::Rect to BoundingBox
    BoundingBox box;
    box.left_top_x = bbox.x;
    box.left_top_y = bbox.y;
    box.right_bottom_x = bbox.x + bbox.width;
    box.right_bottom_y = bbox.y + bbox.height;

    // Compute new bounding box
    scale = std::min({static_cast<float>(src_h - 1) / bbox.height, static_cast<float>(src_w - 1) / bbox.width, scale});

    float new_width = bbox.width * scale;
    float new_height = bbox.height * scale;
    float center_x = bbox.width / 2.0f + bbox.x;
    float center_y = bbox.height / 2.0f + bbox.y;

    float left_top_x = center_x - new_width / 2.0f;
    float left_top_y = center_y - new_height / 2.0f;
    float right_bottom_x = center_x + new_width / 2.0f;
    float right_bottom_y = center_y + new_height / 2.0f;

    if (left_top_x < 0) {
        right_bottom_x -= left_top_x;
        left_top_x = 0;
    }

    if (left_top_y < 0) {
        right_bottom_y -= left_top_y;
        left_top_y = 0;
    }

    if (right_bottom_x > src_w - 1) {
        left_top_x -= right_bottom_x - src_w + 1;
        right_bottom_x = src_w - 1;
    }

    if (right_bottom_y > src_h - 1) {
        left_top_y -= right_bottom_y - src_h + 1;
        right_bottom_y = src_h - 1;
    }

    // Convert back to cv::Rect for output
    cv::Rect new_bbox(static_cast<int>(left_top_x), static_cast<int>(left_top_y),
                      static_cast<int>(right_bottom_x - left_top_x), static_cast<int>(right_bottom_y - left_top_y));
    return new_bbox;
}


template<typename T>
inline bool isShortestSideGreaterThan(const cv::Rect_<T>& rect, T value, float scale) {
    // Find the shortest edge
    T shortestSide = std::min(rect.width / scale, rect.height / scale);
    // Determines whether the shortest edge is greater than the given value
    return shortestSide > value;
}

/**
 * @brief Computes the affine transformation matrix for face cropping.
 * @param rect Rectangle representing the face in the image.
 * @return cv::Mat The computed affine transformation matrix.
 */
inline cv::Mat ComputeCropMatrix(const cv::Rect2f &rect, int width, int height) {
    float x = rect.x;
    float y = rect.y;
    float w = rect.width;
    float h = rect.height;
    float cx = x + w / 2;
    float cy = y + h / 2;
    float length = std::max(w, h) * 1.5 / 2;
    float x1 = cx - length;
    float y1 = cy - length;
    float x2 = cx + length;
    float y2 = cy + length;
    cv::Rect2f padding_rect(x1, y1, x2 - x1, y2 - y1);
    std::vector<cv::Point2f> rect_pts = Rect2Points(padding_rect);
    rect_pts.erase(rect_pts.end() - 1);
    std::vector<cv::Point2f> dst_pts = {{0, 0}, {(float )width, 0}, {(float )width, (float )height}};
    cv::Mat m = cv::getAffineTransform(rect_pts, dst_pts);

    return m;
}


// Exponential Moving Average (EMA) filter function
inline float EmaFilter(float currentProb, std::vector<float>& history, int max, float alpha = 0.2f) {
    // Add current probability to history
    history.push_back(currentProb);

    // Trim history if it exceeds max size
    if (history.size() > max) {
        history.erase(history.begin(), history.begin() + (history.size() - max));
    }

    // Compute EMA
    float ema = history[0];  // Initial value
    for (size_t i = 1; i < history.size(); ++i) {
        ema = alpha * history[i] + (1 - alpha) * ema;
    }

    return ema;
}

}   // namespace inspire

#endif
