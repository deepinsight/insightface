#ifndef LIMONP_BLOCKINGQUEUE_HPP
#define LIMONP_BLOCKINGQUEUE_HPP

#include <queue>
#include "Condition.hpp"

namespace limonp {
template<class T>
class BlockingQueue: NonCopyable {
 public:
  BlockingQueue()
    : mutex_(), notEmpty_(mutex_), queue_() {
  }

  void Push(const T& x) {
    MutexLockGuard lock(mutex_);
    queue_.push(x);
    notEmpty_.Notify(); // Wait morphing saves us
  }

  T Pop() {
    MutexLockGuard lock(mutex_);
    // always use a while-loop, due to spurious wakeup
    while (queue_.empty()) {
      notEmpty_.Wait();
    }
    assert(!queue_.empty());
    T front(queue_.front());
    queue_.pop();
    return front;
  }

  size_t Size() const {
    MutexLockGuard lock(mutex_);
    return queue_.size();
  }
  bool Empty() const {
    return Size() == 0;
  }

 private:
  mutable MutexLock mutex_;
  Condition         notEmpty_;
  std::queue<T>     queue_;
}; // class BlockingQueue

} // namespace limonp

#endif // LIMONP_BLOCKINGQUEUE_HPP
