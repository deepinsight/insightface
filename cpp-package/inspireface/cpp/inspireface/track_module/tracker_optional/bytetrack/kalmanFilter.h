#pragma once

#include "dataType.h"

namespace byte_kalman {
class KalmanFilter {
public:
    static const double chi2inv95[10];
    KalmanFilter();
    KAL_DATA initiate(const DETECTBOX& measurement);
    void predict(KAL_MEAN& mean, KAL_COVA& covariance);
    KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);
    KAL_DATA update(const KAL_MEAN& mean, const KAL_COVA& covariance, const DETECTBOX& measurement);

    Eigen::Matrix<float, 1, -1> gating_distance(const KAL_MEAN& mean, const KAL_COVA& covariance, const std::vector<DETECTBOX>& measurements,
                                                bool only_position = false);

private:
    Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
    Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
    float _std_weight_position;
    float _std_weight_velocity;
};
}  // namespace byte_kalman
