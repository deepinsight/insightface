//
// Created by Jack YU on 2020/6/11.
//

#ifndef ZEUSEESTRACKING_LIB_TEST_HELPER_H
#define ZEUSEESTRACKING_LIB_TEST_HELPER_H

#include <dirent.h>
#include <sys/stat.h>

#include "middleware/camera_stream/camera_stream.h"

namespace TestUtils {

inline uint8_t *rgb2nv21(const cv::Mat &Img) {
  if (Img.empty()) {
    exit(0);
  }
  int cols = Img.cols;
  int rows = Img.rows;

  int Yindex = 0;
  int UVindex = rows * cols;

  unsigned char *yuvbuff =
      new unsigned char[static_cast<int>(1.5 * rows * cols)];

  cv::Mat NV21(rows + rows / 2, cols, CV_8UC1);
  cv::Mat OpencvYUV;
  cv::Mat OpencvImg;
  cv::cvtColor(Img, OpencvYUV, cv::COLOR_BGR2YUV_YV12);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      uchar *YPointer = NV21.ptr<uchar>(i);
      int B = Img.at<cv::Vec3b>(i, j)[0];
      int G = Img.at<cv::Vec3b>(i, j)[1];
      int R = Img.at<cv::Vec3b>(i, j)[2];
      int Y = (77 * R + 150 * G + 29 * B) >> 8;
      YPointer[j] = Y;
      yuvbuff[Yindex++] = (Y < 0) ? 0 : ((Y > 255) ? 255 : Y);
      uchar *UVPointer = NV21.ptr<uchar>(rows + i / 2);
      if (i % 2 == 0 && (j) % 2 == 0) {
        int U = ((-44 * R - 87 * G + 131 * B) >> 8) + 128;
        int V = ((131 * R - 110 * G - 21 * B) >> 8) + 128;
        UVPointer[j] = V;
        UVPointer[j + 1] = U;
        yuvbuff[UVindex++] = (V < 0) ? 0 : ((V > 255) ? 255 : V);
        yuvbuff[UVindex++] = (U < 0) ? 0 : ((U > 255) ? 255 : U);
      }
    }
  }
  return yuvbuff;
}

inline void rotate(const cv::Mat &image, cv::Mat &out,
                   inspire::ROTATION_MODE mode) {
  if (mode == inspire::ROTATION_90) {
    cv::transpose(image, out);
    cv::flip(out, out, 2);
  } else if (mode == inspire::ROTATION_180) {
    cv::flip(out, out, -1);
  } else if (mode == inspire::ROTATION_270) {
    cv::transpose(image, out);
    cv::flip(out, out, 0);
  }
}


    void GetFilesInDirectory(std::vector<std::string> &out, const std::string &directory)
    {
#ifdef WINDOWS
        HANDLE dir;
    WIN32_FIND_DATA file_data;

    if ((dir = FindFirstFile((directory + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
        return; /* No files found */

    do {
        const string file_name = file_data.cFileName;
        const string full_file_name = directory + "/" + file_name;
        const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

        if (file_name[0] == '.')
            continue;

        if (is_directory)
            continue;

        out.push_back(full_file_name);
    } while (FindNextFile(dir, &file_data));

    FindClose(dir);
#else
        DIR *dir;
        class dirent *ent;
        class stat st;

        dir = opendir(directory.c_str());
        while ((ent = readdir(dir)) != NULL) {
            const std::string file_name = ent->d_name;
            const std::string full_file_name = directory + "/" + file_name;

            if (file_name[0] == '.')
                continue;

            if (stat(full_file_name.c_str(), &st) == -1)
                continue;

            const bool is_directory = (st.st_mode & S_IFDIR) != 0;

            if (is_directory)
                continue;

            out.push_back(full_file_name);
        }
        closedir(dir);
#endif
    }

} // namespace TestUtils

#endif // ZEUSEESTRACKING_LIB_TEST_HELPER_H
