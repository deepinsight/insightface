#ifndef LIMONP_THREAD_HPP
#define LIMONP_THREAD_HPP

#include "Logging.hpp"
#include "NonCopyable.hpp"

namespace limonp {

class IThread: NonCopyable {
 public:
  IThread(): isStarted(false), isJoined(false) {
  }
  virtual ~IThread() {
    if(isStarted && !isJoined) {
      XCHECK(!pthread_detach(thread_));
    }
  };

  virtual void Run() = 0;
  void Start() {
    XCHECK(!isStarted);
    XCHECK(!pthread_create(&thread_, NULL, Worker, this));
    isStarted = true;
  }
  void Join() {
    XCHECK(!isJoined);
    XCHECK(!pthread_join(thread_, NULL));
    isJoined = true;
  }
 private:
  static void * Worker(void * data) {
    IThread * ptr = (IThread* ) data;
    ptr->Run();
    return NULL;
  }

  pthread_t thread_;
  bool isStarted;
  bool isJoined;
}; // class IThread

} // namespace limonp

#endif // LIMONP_THREAD_HPP
