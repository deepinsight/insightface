#ifndef RGA_UTILS_H__
#define RGA_UTILS_H__

#include <stdlib.h>

#if defined(ISF_ENABLE_RGA)

int64_t get_cur_us();
int64_t get_cur_ms();
void draw_rgba(char *buffer, int width, int height);
void draw_YUV420(char *buffer, int width, int height);
void draw_YUV422(char *buffer, int width, int height);
void draw_gray256(char *buffer, int width, int height);
int read_image_from_fbc_file(void *buf, const char *path, int sw, int sh, int fmt, int index);
int read_image_from_file(void *buf, const char *path, int sw, int sh, int fmt, int index);
int write_image_to_fbc_file(void *buf, const char *path, int sw, int sh, int fmt, int index);
int write_image_to_file(void *buf, const char *path, int sw, int sh, int fmt, int index);

#endif  // ISF_ENABLE_RGA

#endif /* #ifndef RGA_UTILS_H__ */
