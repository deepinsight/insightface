#ifndef COST_TIME_HPP
#define COST_TIME_HPP

#include <chrono>
#include <iomanip>
#include <sstream>
#include "log.h"  // Assume log.h includes the INSPIRE_LOGI macro

// Macro definition to control whether time cost calculations are enabled
#ifdef ISF_ENABLE_COST_TIME

#define COST_TIME(id, precision) inspire::CostTime cost_time_##id(#id, __FILENAME__, __FUNCTION__, __LINE__, precision)
#define COST_TIME_SIMPLE(id) inspire::CostTime cost_time_##id(#id, __FILENAME__, __FUNCTION__, __LINE__)

#else
#define COST_TIME(id, precision) // No operation, when ISF_ENABLE_COST_TIME is not defined
#define COST_TIME_SIMPLE(id) // No operation, when ISF_ENABLE_COST_TIME is not defined

#endif

namespace inspire {

class CostTime {
public:
    // Constructor, records the start time
    explicit CostTime(const char* id, const char* filename, const char* function, int line, int precision = 5)
        : id_(id), filename_(filename), function_(function), line_(line), precision_(precision),
          start_time_(std::chrono::steady_clock::now()) {}

    // Destructor, calculates and logs the time taken
    ~CostTime() {
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end_time - start_time_;

        // Use a string stream for formatted output, controlling decimal precision
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision_) << duration.count();

        // Log the time cost
        INSPIRE_LOGI("CostTime: [%s] %s:%d %s took %s seconds", id_, filename_, line_, function_, oss.str().c_str());
    }

private:
    const char* id_;
    const char* filename_;
    const char* function_;
    int line_;
    int precision_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
};

} // namespace inspire

#endif // COST_TIME_HPP