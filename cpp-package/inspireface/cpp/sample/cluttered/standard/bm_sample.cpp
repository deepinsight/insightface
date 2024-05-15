//
// Created by Tunm-Air13 on 2023/9/11.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "log.h"
#include "inspireface/feature_hub/simd.h"
//#include <Eigen/Dense>

using namespace inspire;

int main() {

    int N = 512;
    int vectorSize = 512; // Vector length
    {
        // Create an Nx512 matrix of type CV_32F and fill it with random numbers
        cv::Mat mat(N, vectorSize, CV_32F);
        cv::randu(mat, cv::Scalar(0), cv::Scalar(1));

        // Create a 512x1 CV_32F matrix and fill it with random numbers
        cv::Mat one(vectorSize, 1, CV_32F);
        cv::randu(one, cv::Scalar(0), cv::Scalar(1));

        std::cout << mat.size << std::endl;
        std::cout << one.size << std::endl;

        auto timeStart = (double) cv::getTickCount();

        cv::Mat cosineSimilarities;
        cv::gemm(mat, one, 1, cv::Mat(), 0, cosineSimilarities);

        double cost = ((double) cv::getTickCount() - timeStart) / cv::getTickFrequency() * 1000;
        INSPIRE_LOGD("Matrix COST: %f", cost);

    }

    {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));

        std::vector<std::vector<float>> matrix(N, std::vector<float>(vectorSize));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < vectorSize; ++j) {
                matrix[i][j] = static_cast<float>(std::rand()) / RAND_MAX;
            }
        }

        std::vector<float> vectorOne(vectorSize);
        for (int i = 0; i < vectorSize; ++i) {
            vectorOne[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        auto timeStart = (double) cv::getTickCount();
        // dot
        for (const auto &v: matrix) {
            simd_dot(v.data(), vectorOne.data(), vectorSize);
        }

        double cost = ((double) cv::getTickCount() - timeStart) / cv::getTickFrequency() * 1000;
        INSPIRE_LOGD("Vector COST: %f", cost);
    }

//    {
//        Eigen::initParallel();
//        Eigen::MatrixXf mat(N, vectorSize);
//        mat = Eigen::MatrixXf::Random(N, vectorSize);
//
//        std::cout << mat.rows() << " x " << mat.cols() << std::endl;
//
//
//        Eigen::VectorXf one(vectorSize);
//        one = Eigen::VectorXf::Random(vectorSize);
//
//        auto timeStart = (double) cv::getTickCount();
//        Eigen::VectorXf result = mat * one;
//
//        double cost = ((double) cv::getTickCount() - timeStart) / cv::getTickFrequency() * 1000;
//        LOGD("Eigen COST: %f", cost);
//    }

    return 0;
}