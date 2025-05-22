#include "spend_timer.h"
#include <ostream>
#include <sstream>
#include "log.h"

#if defined(_MSC_VER)
#include <chrono>  // NOLINT
#else
#include <sys/time.h>
#endif

namespace inspire {

#if defined(_MSC_VER)

uint64_t INSPIRE_API_EXPORT _now() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

#else

uint64_t INSPIRE_API_EXPORT _now() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

#endif  // defined(_MSC_VER)

int SpendTimer::is_enable = true;

SpendTimer::SpendTimer() {
    Reset();
}

SpendTimer::SpendTimer(const std::string &name) {
    name_ = name;
    Reset();
}

void SpendTimer::Start() {
    start_ = _now();
}

void SpendTimer::Stop() {
    stop_ = _now();
    uint64_t d = stop_ - start_;
    total_ += d;
    ++count_;
    min_ = std::min(min_, d);
    max_ = std::max(max_, d);
}

void SpendTimer::Reset() {
    start_ = 0;
    stop_ = 0;
    total_ = 0;
    count_ = 0;
    min_ = UINT64_MAX;
    max_ = 0;
}

uint64_t SpendTimer::Get() const {
    return stop_ - start_;
}

uint64_t SpendTimer::Average() const {
    return count_ == 0 ? 0 : total_ / count_;
}

uint64_t SpendTimer::Total() const {
    return total_;
}

uint64_t SpendTimer::Count() const {
    return count_;
}

uint64_t SpendTimer::Min() const {
    return count_ == 0 ? 0 : min_;
}

uint64_t SpendTimer::Max() const {
    return max_;
}

const std::string &SpendTimer::name() const {
    return name_;
}

std::string SpendTimer::Report() const {
    std::stringstream ss;
    if (is_enable) {
        ss << "[Time(us) Total:" << Total() << " Ave:" << Average() << " Min:" << Min() << " Max:" << Max() << " Count:" << Count() << " " << name_
           << "]";
    } else {
        ss << "Timer Disabled.";
    }
    return ss.str();
}

void SpendTimer::Disable() {
    is_enable = false;
}

std::ostream &operator<<(std::ostream &os, const SpendTimer &timer) {
    os << timer.Report();
    return os;
}

}  // namespace inspire
