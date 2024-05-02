#ifndef LIMONP_FILELOCK_HPP
#define LIMONP_FILELOCK_HPP

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <string>
#include <string.h>
#include <assert.h>

namespace limonp {

using std::string;

class FileLock {
 public:
  FileLock() : fd_(-1), ok_(true) {
  }
  ~FileLock() {
    if(fd_ > 0) {
      Close();
    }
  }
  void Open(const string& fname) {
    assert(fd_ == -1);
    fd_ = open(fname.c_str(), O_RDWR | O_CREAT, 0644);
    if(fd_ < 0) {
      ok_ = false;
      err_ = strerror(errno);
    }
  }
  void Close() {
    ::close(fd_);
  }
  void Lock() {
    if(LockOrUnlock(fd_, true) < 0) {
      ok_ = false;
      err_ = strerror(errno);
    }
  }
  void UnLock() {
    if(LockOrUnlock(fd_, false) < 0) {
      ok_ = false;
      err_ = strerror(errno);
    }
  }
  bool Ok() const {
    return ok_;
  }
  string Error() const {
    return err_;
  }
 private:
  static int LockOrUnlock(int fd, bool lock) {
    errno = 0;
    struct flock f;
    memset(&f, 0, sizeof(f));
    f.l_type = (lock ? F_WRLCK : F_UNLCK);
    f.l_whence = SEEK_SET;
    f.l_start = 0;
    f.l_len = 0;        // Lock/unlock entire file
    return fcntl(fd, F_SETLK, &f);
  }

  int fd_;
  bool ok_;
  string err_;
}; // class FileLock

}// namespace limonp

#endif // LIMONP_FILELOCK_HPP
