#ifndef LIMONP_BOUNDED_BLOCKING_QUEUE_HPP
#define LIMONP_BOUNDED_BLOCKING_QUEUE_HPP

#include "BoundedQueue.hpp"

namespace limonp {

template<typename T>
class BoundedBlockingQueue : NonCopyable {
 public:
  explicit BoundedBlockingQueue(size_t maxSize)
    : mutex_(),
      notEmpty_(mutex_),
      notFull_(mutex_),
      queue_(maxSize) {
  }

  void Push(const T& x) {
    MutexLockGuard lock(mutex_);
    while (queue_.Full()) {
      notFull_.Wait();
    }
    assert(!queue_.Full());
    queue_.Push(x);
    notEmpty_.Notify();
  }

  T Pop() {
    MutexLockGuard lock(mutex_);
    while (queue_.Empty()) {
      notEmpty_.Wait();
    }
    assert(!queue_.Empty());
    T res = queue_.Pop();
    notFull_.Notify();
    return res;
  }

  bool Empty() const {
    MutexLockGuard lock(mutex_);
    return queue_.Empty();
  }

  bool Full() const {
    MutexLockGuard lock(mutex_);
    return queue_.Full();
  }

  size_t size() const {
    MutexLockGuard lock(mutex_);
    return queue_.size();
  }

  size_t capacity() const {
    return queue_.capacity();
  }

 private:
  mutable MutexLock          mutex_;
  Condition                  notEmpty_;
  Condition                  notFull_;
  BoundedQueue<T>  queue_;
}; // class BoundedBlockingQueue

} // namespace limonp

#endif // LIMONP_BOUNDED_BLOCKING_QUEUE_HPP
