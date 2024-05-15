//
// Created by Tunm-Air13 on 2023/5/9.
//
#pragma once
#ifndef BIGGUYSMAIN_ALIGNMENT_H
#define BIGGUYSMAIN_ALIGNMENT_H

#include "data_type.h"

namespace inspire {

/**
 * @brief Calculates the mean of rows for the given matrix.
 * @param src Source matrix.
 * @return cv::Mat Mean of rows.
 */
inline cv::Mat INSPIRE_API meanAxis0(const cv::Mat &src) {
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    cv::Mat output(1, dim, CV_32F);
    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (int j = 0; j < num; j++) {
            sum += src.at<float>(j, i);
        }
        output.at<float>(0, i) = sum / num;
    }

    return output;
}

/**
* @brief Performs element-wise subtraction between two matrices.
* @param A First matrix.
* @param B Second matrix.
* @return cv::Mat Result of the subtraction.
*/
inline cv::Mat INSPIRE_API elementwiseMinus(const cv::Mat &A, const cv::Mat &B) {
    cv::Mat output(A.rows, A.cols, A.type());

    assert(B.cols == A.cols);
    if (B.cols == A.cols) {
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < B.cols; j++) {
                output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
            }
        }
    }
    return output;
}

/**
* @brief Calculates variance across rows for the given matrix.
* @param src Source matrix.
* @return cv::Mat Variance of rows.
*/
inline cv::Mat INSPIRE_API varAxis0(const cv::Mat &src) {
    cv::Mat temp_ = elementwiseMinus(src, meanAxis0(src));
    cv::multiply(temp_, temp_, temp_);
    return meanAxis0(temp_);
}

/**
* @brief Computes the rank of a matrix.
* @param M Matrix to compute the rank of.
* @return int Rank of the matrix.
*/
inline int INSPIRE_API MatrixRank(cv::Mat M) {
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;
}

/**
* @brief Computes a similarity transformation matrix.
* @param src Source matrix of points.
* @param dst Destination matrix of points.
* @return cv::Mat Similarity transformation matrix.
*
* References:
* .. [1] "Least-squares estimation of transformation parameters between two
* point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
*/
inline cv::Mat INSPIRE_API similarTransform(cv::Mat src, cv::Mat dst) {
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = meanAxis0(src);
    cv::Mat dst_mean = meanAxis0(dst);
    cv::Mat src_demean = elementwiseMinus(src, src_mean);
    cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;
    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
    cv::SVD::compute(A, S, U, V);

    // the SVD function in opencv differ from scipy .

    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);

    } else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {
            //            s = d[dim - 1]
            //            d[dim - 1] = -1
            //            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            //            d[dim - 1] = s
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V; // np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U * twp;
            d.at<float>(dim - 1, 0) = s;
        }
    } else {
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_ * V.t(); // np.dot(np.diag(d), V.T)
        cv::Mat res = U * twp;       // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
    }
    cv::Mat var_ = varAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d, S, res);
    float scale = 1.0 / val * cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) =
            -T.rowRange(0, dim).colRange(0, dim).t();
    cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat temp2 = src_mean.t();                        // src_mean.T
    cv::Mat temp3 = temp1 * temp2; // np.dot(T[:dim, :dim], src_mean.T)
    cv::Mat temp4 = scale * temp3;
    T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
    T.rowRange(0, dim).colRange(0, dim) *= scale;
    return T;
}

/**
* @brief Generates a transformation matrix for aligning facial landmarks.
* @param pts Landmark points.
* @param size Size of the transformation.
* @param scale Scale factor for transformation.
* @return cv::Mat Transformation matrix.
*/
inline cv::Mat INSPIRE_API getTransformMatrix(std::vector<cv::Point2f> &pts, int size, float scale) {
    float src_pts[] = {30.2946, 51.6963, 65.5318, 51.5014, 48.0252,
                       71.7366, 33.5493, 92.3655, 62.7299, 92.2041};
    if (size) {
        for (int i = 0; i < 5; i++) {
            *(src_pts + 2 * i) += 8.0;
        }
    }
    int default_input_size = 224;
    if (scale > 0.0f){
        float S = scale * 2;
        float D = (2.0f - S ) / 4;
        for (int i = 0; i < 10; ++i) {
            src_pts[i] *= S;
            src_pts[i] += (default_input_size * D);
            if (i % 2 == 1){
                src_pts[i] -= 20;
            }
            float scale_def = float(size) / default_input_size;
            src_pts[i] *= scale_def;
        }
    }
    cv::Mat input_mat(5, 2, CV_32F, pts.data());
    cv::Mat src(5, 2, CV_32F);
    src.data = (uchar *)src_pts;
    cv::Mat M_temp = similarTransform(input_mat, src);
    cv::Mat M = M_temp.rowRange(0, 2);
    return M;
}

/**
* @brief Generates a transformation matrix for aligning facial landmarks with a size of 112.
* @param pts Landmark points.
* @return cv::Mat Transformation matrix.
*/
inline cv::Mat INSPIRE_API getTransformMatrix112(PointsList2f &pts) {
    float src_pts[] = {30.2946, 51.6963, 65.5318, 51.5014, 48.0252,
                       71.7366, 33.5493, 92.3655, 62.7299, 92.2041};
    cv::Size image_size(112, 112);
    cv::Mat input_mat(5, 2, CV_32F, pts.data());
    if (image_size.height) {
        for (int i = 0; i < 5; i++) {
            *(src_pts + 2 * i) += 8.0;
        }
    }
    cv::Mat src(5, 2, CV_32F);
    src.data = (uchar *)src_pts;
    cv::Mat M_temp = similarTransform(input_mat, src);
    cv::Mat M = M_temp.rowRange(0, 2);

//    std::cout << src << std::endl;
//    std::cout << input_mat << std::endl;
    return M;
}

/**
* @brief Generates a transformation matrix for dense mesh alignment with a specific size of 256.
* @param pts Landmark points.
* @return cv::Mat Transformation matrix.
*/
inline cv::Mat INSPIRE_API getTransformMatrix256specific(PointsList2f &pts) {
    int input_size = 256;
    int new_size = 144;
    cv::Mat dst_pts = (cv::Mat_<float>(5,2) << 38.2946, 51.6963,
            73.5318, 51.5014,
            56.0252, 71.7366,
            41.5493, 92.3655,
            70.7299, 92.2041);

    for (int i = 0; i < dst_pts.rows; i++) {
        dst_pts.at<float>(i, 0) += ((new_size - 112) / 2);
        dst_pts.at<float>(i, 1) += 8;
    }

    float scale = input_size / static_cast<float>(new_size);
    for (int i = 0; i < dst_pts.rows; i++) {
        dst_pts.at<float>(i, 0) *= scale;
        dst_pts.at<float>(i, 1) *= scale;
    }
    cv::Mat input_mat(5, 2, CV_32F, pts.data());
    cv::Mat src(5, 2, CV_32F);
    src.data = (uchar *) dst_pts.data;
    cv::Mat M_temp = similarTransform(input_mat, src);
    cv::Mat M = M_temp.rowRange(0, 2);

    return M;
}

/**
* @brief Generates a transformation matrix for SAFAS model alignment.
* @param pts Landmark points.
* @return cv::Mat Transformation matrix.
*/
inline cv::Mat INSPIRE_API getTransformMatrixSafas(PointsList2f &pts) {
    int input_size = 112;
    int new_size = 230;
    cv::Mat dst_pts = (cv::Mat_<float>(5,2) << 38.2946, 51.6963,
            73.5318, 51.5014,
            56.0252, 71.7366,
            41.5493, 92.3655,
            70.7299, 92.2041);

    for (int i = 0; i < dst_pts.rows; i++) {
        dst_pts.at<float>(i, 0) += ((new_size - 112) / 2);
        dst_pts.at<float>(i, 1) += 8;
    }

    float scale = input_size / static_cast<float>(new_size);
    for (int i = 0; i < dst_pts.rows; i++) {
        dst_pts.at<float>(i, 0) *= scale;
        dst_pts.at<float>(i, 1) *= scale;
    }
    cv::Mat input_mat(5, 2, CV_32F, pts.data());
    cv::Mat src(5, 2, CV_32F);
    src.data = (uchar *) dst_pts.data;
    cv::Mat M_temp = similarTransform(input_mat, src);
    cv::Mat M = M_temp.rowRange(0, 2);

    return M;
}

/**
* @brief Applies affine transformation to an image.
* @param img Original image.
* @param transformed Transformed image.
* @param transM Transformation matrix.
* @param size Target size of the transformed image.
*/
inline void INSPIRE_API toTransform(const Matrix& img, Matrix &transformed, Matrix& transM, cv::Size size) {
    cv::warpAffine(img, transformed, transM, std::move(size), cv::INTER_CUBIC);
}

}

#endif //BIGGUYSMAIN_ALIGNMENT_H