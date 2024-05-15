//
// Created by YH-Mac on 2020/9/12.
//

#ifndef LMKTRACKING_LIB_TEST_FOLDER_HELPER_H
#define LMKTRACKING_LIB_TEST_FOLDER_HELPER_H

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stack>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <vector>
#include <sstream>
#include "sys/types.h"
#include "sys/stat.h"
#include <chrono>
//using namespace std;
#define MODE (S_IRWXU | S_IRWXG | S_IRWXO)

using namespace std::chrono;
class Timer
{
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic()
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true)
    {
        double diff = duration_cast<milliseconds>(high_resolution_clock::now() - tictoc_stack.top()).count();
        if(msg.size() > 0){
            if (flag)
                printf("%s time elapsed: %f ms\n", msg.c_str(), diff);
        }
        if(msg == "ir")
          return diff/3;

        tictoc_stack.pop();
        return diff;
    }
    void reset()
    {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};

std::string dePrefix(std::string file_name, std::string prefix){
    int n = file_name.find_last_of(prefix);
    std::string de = file_name.substr(n + 1, file_name.size());
    return de;
}

std::string deSuffix(std::string file_name, std::string suffix){
    int n = file_name.find_last_of(suffix);
    std::string de = file_name.substr(0, n);
    return de;
}

bool check_folder(const std::string folder_path, bool is_create) {
    int res = access(folder_path.c_str(), 0);
//    std::cout << "res " << res << std::endl;
    if (res != -1) {
        std::cout << folder_path << " exists." << std::endl;
        return true;
    } else if(is_create){
        std::cout << folder_path << " not exists." << std::endl;
        int mkdir_res = mkdir(folder_path.c_str(), MODE);
        if (mkdir_res != -1) {
            std::cout << "mkdir successful." << std::endl;
        } else {
            std::cout << "mkdir fail." << std::endl;
        }
        return true;
    } else{
        std::cout << folder_path << " not exists." << std::endl;
        return false;
    }
}

std::vector<float> read_feat_txt(const std::string feat_path){
    std::ifstream infile;
    infile.open(feat_path);
    if (!infile)
        std::cout << "file error: " << feat_path << std::endl;
    std::vector<float> feat;
    float t1;
    while (infile >> t1) {
        feat.push_back(t1);
    }
    infile.close();
    return feat;
}

void extract_feat_to_txt(const std::string feat_path, std::vector<float> feat) {
    std::ofstream txt(feat_path);
    for (int i = 0; i < feat.size(); ++i) {
        txt << feat[i] << " ";
    }
    txt.close();
//    std::cout << "export feature to " << feat_path << std::endl;
}

bool decide_ext(const char *gname, const char *nsuff)
// gname: name of the given file
// nsuff: suffix  you need
{
    char dot = '.';
    char suff[10] = {0};
    int c;
    int j = 0;
    c = strlen(gname);
//    std::cout<< c << '\n';
    for (int i = 0; i < c; i++) {
        if (gname[i] == dot)
            j = i;
    }
    int k = j;
    j = c - j - 1;
    for (int i = 0; i < j; i++) {
        suff[i] = gname[k + i + 1];
    }
    if (0 == strcmp(suff, nsuff))
        return true;
    else
        return false;
}


std::vector<std::string>
readFileListSuffix(const char *basePath, const char *suffix, std::vector<std::string>& file_names, bool recursive) {
    DIR *dir;
    struct dirent *ptr;
    char base[1000];
    std::vector<std::string> file_list;
    if ((dir = opendir(basePath)) == NULL) {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8) {    ///file
//            printf("d_name:%s/%s\n", basePath, ptr->d_name);
            if (decide_ext(ptr->d_name, suffix)) {
                std::stringstream ss;
                ss << basePath << "/" << ptr->d_name;
                std::string path = ss.str();
                file_names.push_back(ptr->d_name);
                file_list.push_back(path);
            }
        } else if (ptr->d_type == 10) {    ///link file
//            printf("d_name:%s/%s\n", basePath, ptr->d_name);
        } else if (ptr->d_type == 4 and recursive)    ///dir
        {
            memset(base, '\0', sizeof(base));
            strcpy(base, basePath);
            strcat(base, "/");
            strcat(base, ptr->d_name);
            readFileListSuffix(base, suffix, file_names, recursive);
        }
    }
    closedir(dir);
    return file_list;
}

#endif //LMKTRACKING_LIB_TEST_FOLDER_HELPER_H
