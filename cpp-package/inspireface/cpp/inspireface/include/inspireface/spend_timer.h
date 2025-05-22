#ifndef INSPIRE_FACE_TIMER_H
#define INSPIRE_FACE_TIMER_H

#include "data_type.h"

namespace inspire {

// Get the current time in microseconds.
uint64_t INSPIRE_API_EXPORT _now();

/**
 * @brief A class to measure the cost of a block of code.
 */
class INSPIRE_API_EXPORT SpendTimer {
public:
    SpendTimer();
    explicit SpendTimer(const std::string &name);

    void Start();
    void Stop();
    void Reset();

    uint64_t Get() const;
    uint64_t Average() const;
    uint64_t Total() const;
    uint64_t Count() const;
    uint64_t Min() const;
    uint64_t Max() const;
    const std::string &name() const;
    std::string Report() const;

    static void Disable();

protected:
    uint64_t start_;
    uint64_t stop_;
    uint64_t total_;
    uint64_t count_;
    uint64_t min_;
    uint64_t max_;
    std::string name_;

    static int is_enable;
};

INSPIRE_API_EXPORT std::ostream &operator<<(std::ostream &os, const SpendTimer &timer);

#define TIME_NOW inspirecv::_now()

}  // namespace inspire
#endif  // INSPIRE_FACE_TIMER_H
