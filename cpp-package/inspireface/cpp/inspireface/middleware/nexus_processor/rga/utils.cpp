#if defined(ISF_ENABLE_RGA)

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "RgaUtils.h"

int64_t get_cur_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

int64_t get_cur_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void draw_rgba(char *buffer, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width / 4; j++) {
            buffer[(i * width * 4) + j * 4 + 0] = 0xff;  // R
            buffer[(i * width * 4) + j * 4 + 1] = 0x00;  // G
            buffer[(i * width * 4) + j * 4 + 2] = 0x00;  // B
            buffer[(i * width * 4) + j * 4 + 3] = 0xff;  // A
        }
        for (int j = width / 4; j < width / 4 * 2; j++) {
            buffer[(i * width * 4) + j * 4 + 0] = 0x00;
            buffer[(i * width * 4) + j * 4 + 1] = 0xff;
            buffer[(i * width * 4) + j * 4 + 2] = 0x00;
            buffer[(i * width * 4) + j * 4 + 3] = 0xff;
        }
        for (int j = width / 4 * 2; j < width / 4 * 3; j++) {
            buffer[(i * width * 4) + j * 4 + 0] = 0x00;
            buffer[(i * width * 4) + j * 4 + 1] = 0x00;
            buffer[(i * width * 4) + j * 4 + 2] = 0xff;
            buffer[(i * width * 4) + j * 4 + 3] = 0xff;
        }
        for (int j = width / 4 * 3; j < width; j++) {
            buffer[(i * width * 4) + j * 4 + 0] = 0xff;
            buffer[(i * width * 4) + j * 4 + 1] = 0xff;
            buffer[(i * width * 4) + j * 4 + 2] = 0xff;
            buffer[(i * width * 4) + j * 4 + 3] = 0xff;
        }
    }
}

void draw_YUV420(char *buffer, int width, int height) {
    /* Y channel */
    memset(buffer, 0xa8, width * height / 2);
    memset(buffer + width * height / 2, 0x54, width * height / 2);
    /* UV channel */
    memset(buffer + width * height, 0x80, width * height / 4);
    memset(buffer + (int)(width * height * 1.25), 0x30, width * height / 4);
}

void draw_YUV422(char *buffer, int width, int height) {
    /* Y channel */
    memset(buffer, 0xa8, width * height / 2);
    memset(buffer + width * height / 2, 0x54, width * height / 2);
    /* UV channel */
    memset(buffer + width * height, 0x80, width * height / 2);
    memset(buffer + (int)(width * height * 1.5), 0x30, width * height / 2);
}

void draw_gray256(char *buffer, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width / 4; j++) {
            buffer[(i * width * 4) + j * 4] = 0xa8;
        }
        for (int j = width / 4; j < width / 4 * 2; j++) {
            buffer[(i * width * 4) + j * 4] = 0x80;
        }
        for (int j = width / 4 * 2; j < width / 4 * 3; j++) {
            buffer[(i * width * 4) + j * 4] = 0x54;
        }
        for (int j = width / 4 * 3; j < width; j++) {
            buffer[(i * width * 4) + j * 4] = 0x30;
        }
    }
}

int read_image_from_fbc_file(void *buf, const char *path, int sw, int sh, int fmt, int index) {
    int size;
    char filePath[100];
    const char *inputFbcFilePath = "%s/in%dw%d-h%d-%s-fbc.bin";

    snprintf(filePath, 100, inputFbcFilePath, path, index, sw, sh, translate_format_str(fmt));

    FILE *file = fopen(filePath, "rb");
    if (!file) {
        fprintf(stderr, "Could not open %s\n", filePath);
        return -EINVAL;
    }

    size = sw * sh * get_bpp_from_format(fmt) * 1.5;

    fread(buf, size, 1, file);

    fclose(file);

    return 0;
}

int read_image_from_file(void *buf, const char *path, int sw, int sh, int fmt, int index) {
    int size;
    char filePath[100];
    const char *inputFilePath = "%s/in%dw%d-h%d-%s.bin";

    snprintf(filePath, 100, inputFilePath, path, index, sw, sh, translate_format_str(fmt));

    FILE *file = fopen(filePath, "rb");
    if (!file) {
        fprintf(stderr, "Could not open %s\n", filePath);
        return -EINVAL;
    }

    size = sw * sh * get_bpp_from_format(fmt);

    fread(buf, size, 1, file);

    fclose(file);

    return 0;
}

int write_image_to_fbc_file(void *buf, const char *path, int sw, int sh, int fmt, int index) {
    int size;
    char filePath[100];
    const char *outputFbcFilePath = "%s/out%dw%d-h%d-%s-fbc.bin";

    snprintf(filePath, 100, outputFbcFilePath, path, index, sw, sh, translate_format_str(fmt));

    FILE *file = fopen(filePath, "wb+");
    if (!file) {
        fprintf(stderr, "Could not open %s\n", filePath);
        return false;
    } else {
        fprintf(stderr, "open %s and write ok\n", filePath);
    }

    size = sw * sh * get_bpp_from_format(fmt) * 1.5;

    fwrite(buf, size, 1, file);

    fclose(file);

    return 0;
}

int write_image_to_file(void *buf, const char *path, int sw, int sh, int fmt, int index) {
    int size;
    char filePath[100];
    const char *outputFilePath = "%s/out%dw%d-h%d-%s.bin";

    snprintf(filePath, 100, outputFilePath, path, index, sw, sh, translate_format_str(fmt));

    FILE *file = fopen(filePath, "wb+");
    if (!file) {
        fprintf(stderr, "Could not open %s\n", filePath);
        return false;
    } else {
        fprintf(stderr, "open %s and write ok\n", filePath);
    }

    size = sw * sh * get_bpp_from_format(fmt);

    fwrite(buf, size, 1, file);

    fclose(file);

    return 0;
}

#endif  // ISF_ENABLE_RGA