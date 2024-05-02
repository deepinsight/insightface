#ifndef LIMONP_THREAD_POOL_HPP
#define LIMONP_THREAD_POOL_HPP

#include "Thread.hpp"
#include "BlockingQueue.hpp"
#include "BoundedBlockingQueue.hpp"
#include "Closure.hpp"

namespace limonp {

using namespace std;

//class ThreadPool;
class ThreadPool: NonCopyable {
 public:
  class Worker: public IThread {
   public:
    Worker(ThreadPool* pool): ptThreadPool_(pool) {
      assert(ptThreadPool_);
    }
    virtual ~Worker() {
    }

    virtual void Run() {
      while (true) {
        ClosureInterface* closure = ptThreadPool_->queue_.Pop();
        if (closure == NULL) {
          break;
        }
        try {
          closure->Run();
        } catch(std::exception& e) {
          XLOG(ERROR) << e.what();
        } catch(...) {
          XLOG(ERROR) << " unknown exception.";
        }
        delete closure;
      }
    }
   private:
    ThreadPool * ptThreadPool_;
  }; // class Worker

  ThreadPool(size_t thread_num)
    : threads_(thread_num), 
      queue_(thread_num) {
    assert(thread_num);
    for(size_t i = 0; i < threads_.size(); i ++) {
      threads_[i] = new Worker(this);
    }
  }
  ~ThreadPool() {
    Stop();
  }

  void Start() {
    for(size_t i = 0; i < threads_.size(); i++) {
      threads_[i]->Start();
    }
  }
  void Stop() {
    for(size_t i = 0; i < threads_.size(); i ++) {
      queue_.Push(NULL);
    }
    for(size_t i = 0; i < threads_.size(); i ++) {
      threads_[i]->Join();
      delete threads_[i];
    }
    threads_.clear();
  }

  void Add(ClosureInterface* task) {
    assert(task);
    queue_.Push(task);
  }

 private:
  friend class Worker;

  vector<IThread*> threads_;
  BoundedBlockingQueue<ClosureInterface*> queue_;
}; // class ThreadPool

} // namespace limonp

#endif // LIMONP_THREAD_POOL_HPP
