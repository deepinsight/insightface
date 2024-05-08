//
// Created by Tunm-Air13 on 2023/9/12.
//

#ifndef HYPERFACEREPO_TEST_HELP_H
#define HYPERFACEREPO_TEST_HELP_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include "inspireface/c_api/inspireface.h"
#include "opencv2/opencv.hpp"
#include <iomanip>
#include "test_tools.h"
#include <random>

using namespace indicators;

typedef std::vector<std::pair<std::string, std::string>> FaceImageDataList;

inline void Split(const std::string& src, std::vector<std::string>& res, const std::string& pattern, size_t maxsplit = std::string::npos) {
    res.clear();
    size_t Start = 0;
    size_t end = 0;
    std::string sub;
    while(Start < src.size()) {
        end = src.find_first_of(pattern, Start);
        if(std::string::npos == end || res.size() >= maxsplit) {
            sub = src.substr(Start);
            res.push_back(sub);
            return;
        }
        sub = src.substr(Start, end - Start);
        res.push_back(sub);
        Start = end + 1;
    }
    return;
}

inline std::vector<std::string> Split(const std::string& src, const std::string& pattern, size_t maxsplit = std::string::npos) {
    std::vector<std::string> res;
    Split(src, res, pattern, maxsplit);
    return res;
}

inline bool EndsWith(const std::string& str, const std::string& suffix) {
    if(suffix.length() > str.length()) {
        return false;
    }
    return 0 == str.compare(str.length() -  suffix.length(), suffix.length(), suffix);
}

inline std::string PathJoin(const std::string& path1, const std::string& path2) {
    if(EndsWith(path1, "/")) {
        return path1 + path2;
    }
    return path1 + "/" + path2;
}

inline FaceImageDataList LoadLFWFunneledValidData(const std::string &dir, const std::string &txtPath){
    FaceImageDataList list;
    std::ifstream file(txtPath);
    std::string line;

    while (std::getline(file, line)) {
        std::vector<std::string> parts;
        Split(line, parts, "/");
        if (parts.size() >= 2) {
            std::string name = parts[0];
            std::string fullPath = dir + "/" + line;

            list.push_back({name, fullPath});
        }
    }

    return list;
}

inline bool ImportLFWFunneledValidData(HFSession handle, FaceImageDataList& data, size_t importNum) {
    auto dataSize = data.size();
    std::string title = "Import " + std::to_string(importNum) + " face data...";
    // Hide cursor
    show_console_cursor(false);
    BlockProgressBar bar{
            option::BarWidth{60},
            option::Start{"["},
            option::End{"]"},
            option::PostfixText{title},
            option::ForegroundColor{Color::white}  ,
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };

    auto progress = 0.0f;
    for (size_t i = 0; i < importNum; ++i) {
        bar.set_progress(progress);
        size_t index = i % dataSize;
        // Data processing
        auto item = data[index];
        cv::Mat image = cv::imread(item.second);
        HFImageData imageData = {0};
        imageData.data = image.data;
        imageData.height = image.rows;
        imageData.width = image.cols;
        imageData.format = HF_STREAM_BGR;
        imageData.rotation = HF_CAMERA_ROTATION_0;
        HFImageStream imgHandle;
        auto ret = HFCreateImageStream(&imageData, &imgHandle);
        if (ret != HSUCCEED || image.empty()) {
            std::cerr << "Error image: " << std::to_string(ret)  << " , " << item.second << std::endl;
            return false;
        }
        // Face tracked
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(handle, imgHandle, &multipleFaceData);

        if (ret != HSUCCEED) {
            std::cerr << "Error Track: " << std::to_string(ret)  << " , " << item.second << std::endl;
            return false;
        }

        if (multipleFaceData.detectedNum == 0) {
            std::cerr << "Not Detected face: " << item.second << std::endl;
            return false;
        }

        // Extract face feature
        HFFaceFeature feature = {0};
        ret = HFFaceFeatureExtract(handle, imgHandle, multipleFaceData.tokens[0], &feature);
        if (ret != HSUCCEED) {
            std::cerr << "Error extract: " << std::to_string(ret)  << " , " << item.second << std::endl;
            return false;
        }
        char *newTagName = new char[item.first.size() + 1];
        std::strcpy(newTagName, item.first.c_str());
        HFFaceFeatureIdentity identity = {0};
        identity.customId = i;
        identity.tag = newTagName;
        identity.feature = &feature;
        ret = HFFeatureHubInsertFeature(identity);
        if (ret != HSUCCEED) {
            std::cerr << "Error insert feature: " << std::to_string(ret)  << " , " << item.second << std::endl;
            return false;
        }

        delete[] newTagName;
        HFReleaseImageStream(imgHandle);
        // Update progress
        progress = 100.0f * (float)(i + 1) / importNum;
    }
    bar.set_progress(100.0f);
    // Show cursor
    show_console_cursor(true);
    std::cout << "\033[0m\n"; // ANSI resets the color code

    return true;
}

inline double CalculateOverlap(const HFaceRect& box1, const HFaceRect& box2) {
    // Calculate the coordinates of the intersection rectangle
    HInt32 x_overlap = std::max(0, std::min(box1.x + box1.width, box2.x + box2.width) - std::max(box1.x, box2.x));
    HInt32 y_overlap = std::max(0, std::min(box1.y + box1.height, box2.y + box2.height) - std::max(box1.y, box2.y));

    // Calculate the area of the intersection
    HInt32 overlap_area = x_overlap * y_overlap;

    // Calculate the area of each rectangle
    HInt32 box1_area = box1.width * box1.height;
    HInt32 box2_area = box2.width * box2.height;

    // Calculate the total area
    HInt32 total_area = box1_area + box2_area - overlap_area;

    // Calculate the overlap ratio
    double overlap_ratio = total_area > 0 ? static_cast<double>(overlap_area) / total_area : 0;

    return overlap_ratio;
}

inline std::vector<std::string> generateFilenames(const std::string& templateStr, int start, int end) {
    std::vector<std::string> filenames;
    for (int i = start; i <= end; ++i) {
        std::ostringstream oss;
        oss << "frame-" << std::setw(4) << std::setfill('0') << i << ".jpg";
        filenames.push_back(oss.str());
    }
    return filenames;
}

inline bool FindMostSimilarScoreFromTwoPic(HFSession handle, const std::string& img1, const std::string& img2, float& mostSimilar){
    mostSimilar = -1.0f;
    std::vector<std::vector<std::vector<float>>> features(2);
    std::vector<std::string> images = {img1, img2};
    for (int i = 0; i < 2; ++i) {
        HFImageStream img;
//        auto ret = ReadImageToImageStream(images[i].c_str(), img);
        auto cvMat = cv::imread(images[i]);
        auto ret = CVImageToImageStream(cvMat, img);
        if (ret != 0) {
            std::cerr << "Image is not found: " << ret << std::endl;
            return false;
        }
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(handle, img, &multipleFaceData);
        if (ret != 0) {
            std::cerr << "Error track: " << ret << std::endl;
            HFReleaseImageStream(img);
            return false;
        }
        HInt32 featureNum;
        HFGetFeatureLength(&featureNum);
        for (int j = 0; j < multipleFaceData.detectedNum; ++j) {
            std::vector<float> feature(featureNum, 0.0f);
            ret = HFFaceFeatureExtractCpy(handle, img, multipleFaceData.tokens[j], feature.data());
            if (ret != 0) {
                std::cerr << "Error extract: " << ret << std::endl;
                HFReleaseImageStream(img);
                return false;
            }
            features[i].push_back(feature);
        }
        HFReleaseImageStream(img);
    }

    if (features[0].empty() || features[1].empty()) {
//        std::cerr << "Not detected " << std::endl;
        return false;
    }
    auto &features1 = features[0];
    auto &features2 = features[1];
    for (auto &feat1: features1) {
        for (auto &feat2: features2) {
            float comp;
            HFFaceFeature faceFeature1 = {0};
            faceFeature1.size = feat1.size();
            faceFeature1.data = feat1.data();
            HFFaceFeature faceFeature2 = {0};
            faceFeature2.size = feat2.size();
            faceFeature2.data = feat2.data();

            HFFaceComparison(faceFeature1, faceFeature2, &comp);
            if (comp > mostSimilar) {
                mostSimilar = comp;
            }
        }
    }

    return true;
}

inline std::vector<std::vector<std::string>> ReadPairs(const std::string& pairs_filename) {
    std::vector<std::vector<std::string>> pairs;
    std::ifstream file(pairs_filename); // Open the file
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << pairs_filename << std::endl;
        return pairs; // If the file cannot be opened, an empty list is returned
    }

    std::getline(file, line); // Skip the first line
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<std::string> pair;
        std::string element;

        while (iss >> element) {
            pair.push_back(element);
        }
        if (!pair.empty()) {
            pairs.push_back(pair);
        }
    }

    return pairs;
}

inline std::string zfill(int number, int width) {
    std::ostringstream oss;
    // Set padding to '0' and define the string width
    oss << std::setfill('0') << std::setw(width) << number;
    return oss.str();
}

inline std::pair<float, float> FindBestThreshold(const std::vector<float>& similarities, const std::vector<int>& labels) {
    std::vector<float> thresholds;
    for (float i = 0.0f; i < 1.0f; i += 0.01f) {
        thresholds.push_back(i);
    }

    float best_threshold = 0.0f;
    float best_accuracy = 0.0f;

    for (auto& threshold : thresholds) {
        std::vector<int> predictions;
        for (auto& similarity : similarities) {
            predictions.push_back(similarity > threshold ? 1 : 0);
        }

        int correct = 0;
        for (size_t i = 0; i < labels.size(); ++i) {
            if (predictions[i] == labels[i]) {
                ++correct;
            }
        }

        float accuracy = static_cast<float>(correct) / static_cast<float>(labels.size());

        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            best_threshold = threshold;
        }
    }

    return {best_threshold, best_accuracy};
}

/** Generate random eigenvectors of the specified length */
inline std::vector<float> GenerateRandomFeature(size_t length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<float> featureVector(length);
    float norm = 0.0;

    for (float &value : featureVector) {
        value = dis(gen);
        norm += value * value;
    }

    norm = std::sqrt(norm);

    if (norm > 0) {
        for (float &value : featureVector) {
            value /= norm;
        }
    }

    return featureVector;
}

inline std::vector<float> SimulateSimilarVector(const std::vector<float>& original) {
    std::vector<float> similar(original.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 0.02);

    for (size_t i = 0; i < original.size(); ++i) {
        similar[i] = original[i] + dis(gen);
    }

    float norm = 0.0f;
    for (auto& value : similar) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (auto& value : similar) {
            value /= norm;
        }
    }

    return similar;
}

inline std::vector<int> GenerateRandomNumbers(int n, int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(min, max);

    std::vector<int> numbers;
    numbers.reserve(n);

    for (int i = 0; i < n; ++i) {
        numbers.push_back(distrib(gen));
    }

    return numbers;
}


#endif //HYPERFACEREPO_TEST_HELP_H
