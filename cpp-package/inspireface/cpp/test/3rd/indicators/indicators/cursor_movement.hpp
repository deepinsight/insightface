
#ifndef INDICATORS_CURSOR_MOVEMENT
#define INDICATORS_CURSOR_MOVEMENT

#if defined(_MSC_VER)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <iostream>
#endif

namespace indicators {

#ifdef _MSC_VER

static inline void move(int x, int y) {
  auto hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  if (!hStdout)
    return;

  CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
  GetConsoleScreenBufferInfo(hStdout, &csbiInfo);

  COORD cursor;

  cursor.X = csbiInfo.dwCursorPosition.X + x;
  cursor.Y = csbiInfo.dwCursorPosition.Y + y;
  SetConsoleCursorPosition(hStdout, cursor);
}

static inline void move_up(int lines) { move(0, -lines); }
static inline void move_down(int lines) { move(0, -lines); }
static inline void move_right(int cols) { move(cols, 0); }
static inline void move_left(int cols) { move(-cols, 0); }

#else

static inline void move_up(int lines) { std::cout << "\033[" << lines << "A"; }
static inline void move_down(int lines) { std::cout << "\033[" << lines << "B"; }
static inline void move_right(int cols) { std::cout << "\033[" << cols << "C"; }
static inline void move_left(int cols) { std::cout << "\033[" << cols << "D"; }

#endif

} // namespace indicators

#endif