#ifndef LIMONP_BOUNDED_QUEUE_HPP
#define LIMONP_BOUNDED_QUEUE_HPP

#include <vector>
#include <fstream>
#include <cassert>

namespace limonp {
using namespace std;
template<class T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity): capacity_(capacity), circular_buffer_(capacity) {
    head_ = 0;
    tail_ = 0;
    size_ = 0;
    assert(capacity_);
  }
  ~BoundedQueue() {
  }

  void Clear() {
    head_ = 0;
    tail_ = 0;
    size_ = 0;
  }
  bool Empty() const {
    return !size_;
  }
  bool Full() const {
    return capacity_ == size_;
  }
  size_t Size() const {
    return size_;
  }
  size_t Capacity() const {
    return capacity_;
  }

  void Push(const T& t) {
    assert(!Full());
    circular_buffer_[tail_] = t;
    tail_ = (tail_ + 1) % capacity_;
    size_ ++;
  }

  T Pop() {
    assert(!Empty());
    size_t oldPos = head_;
    head_ = (head_ + 1) % capacity_;
    size_ --;
    return circular_buffer_[oldPos];
  }

 private:
  size_t head_;
  size_t tail_;
  size_t size_;
  const size_t capacity_;
  vector<T> circular_buffer_;

}; // class BoundedQueue
} // namespace limonp

#endif
