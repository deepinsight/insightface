//
// Created by Tunm-Air13 on 2023/9/21.
//
#pragma once
#ifndef HYPERFACEREPO_TIMER_H
#define HYPERFACEREPO_TIMER_H
#include <opencv2/opencv.hpp>

namespace inspire {

class Timer {
public:

    Timer() {
        current_time = (double) cv::getTickCount();
    }

    double GetCostTime() const {
        return ((double) cv::getTickCount() - current_time) / cv::getTickFrequency() * 1000;
    }

    double GetCostTimeUpdate() {
        auto cost = ((double) cv::getTickCount() - current_time) / cv::getTickFrequency() * 1000;
        current_time = (double) cv::getTickCount();

        return cost;
    }

private:

    double current_time;

};

}

#endif //HYPERFACEREPO_TIMER_H
