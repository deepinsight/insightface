/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */
#pragma once
#ifndef INSPIRE_FACE_COSTMAN_H
#define INSPIRE_FACE_COSTMAN_H
#include <chrono>

namespace inspire {

class Timer {
public:
    Timer() {
        current_time = std::chrono::high_resolution_clock::now();
    }

    double GetCostTime() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - current_time);
        return duration.count() / 1000000.0;  // Convert to milliseconds
    }

    double GetCostTimeUpdate() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - current_time);
        current_time = now;
        return duration.count() / 1000000.0;  // Convert to milliseconds
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> current_time;
};

}  // namespace inspire

#endif  // INSPIRE_FACE_COSTMAN_H
