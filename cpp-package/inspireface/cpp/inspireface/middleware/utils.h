
#ifndef INSPIRE_FACE_UTILS_H
#define INSPIRE_FACE_UTILS_H

#include <cmath>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <inspirecv/inspirecv.h>

namespace inspire {


inline bool IsDirectory(const std::string& path) {
#ifdef _WIN32
    DWORD dwAttrib = GetFileAttributes(path.c_str());
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    } else {
        return false;
    }
#endif
}


inline inspirecv::TransformMatrix SquareToSquare(inspirecv::Rect2f src, inspirecv::Rect2f dst,
                              float win_size = 112) {
    float src_x1 = static_cast<float>(src.GetX());
    float src_y1 = static_cast<float>(src.GetY());
    float src_x2 = static_cast<float>(src.GetX() + src.GetWidth());
    float src_y2 = static_cast<float>(src.GetY() + src.GetHeight());

    float dst_x1 = static_cast<float>(dst.GetX());
    float dst_y1 = static_cast<float>(dst.GetY());
    float dst_x2 = static_cast<float>(dst.GetX() + dst.GetWidth());
    float dst_y2 = static_cast<float>(dst.GetY() + dst.GetHeight());

    std::vector<inspirecv::Point2f> src_pts = {
            {src_x1, src_y1},
            {src_x2, src_y1},
            {src_x2, src_y2},
            {src_x1, src_y2}};
    std::vector<inspirecv::Point2f> dst_pts = {
            {dst_x1, dst_y1},
            {dst_x2, dst_y1},
            {dst_x2, dst_y2},
            {dst_x1, dst_y2}};
    inspirecv::TransformMatrix m = inspirecv::SimilarityTransformEstimate(src_pts, dst_pts);
    return m;
}

inline std::vector<inspirecv::Point2f>
FixPointsMeanshape(std::vector<inspirecv::Point2f> &points,
                   const std::vector<inspirecv::Point2f> &mean_shape) {

    inspirecv::Rect2f bbox = inspirecv::MinBoundingRect(points);
    int R = std::max(bbox.GetHeight(), bbox.GetWidth());
    int cx = bbox.GetX() + bbox.GetWidth() / 2;
    int cy = bbox.GetY() + bbox.GetHeight() / 2;
    inspirecv::Rect2f old(cx - R / 2, cy - R / 2, R, R);

    inspirecv::Rect2f mean_shape_box = inspirecv::MinBoundingRect(mean_shape);
    int m_R = std::max(mean_shape_box.GetHeight(), mean_shape_box.GetWidth());
    int m_cx = mean_shape_box.GetX() + mean_shape_box.GetWidth() / 2;
    int m_cy = mean_shape_box.GetY() + mean_shape_box.GetHeight() / 2;
    inspirecv::Rect2f _new(m_cx - m_R / 2, m_cy - m_R / 2, m_R, m_R);
    inspirecv::TransformMatrix affine = SquareToSquare(old, _new);
    std::vector<inspirecv::Point2f> new_pts = ApplyTransformToPoints(points, affine);
    return new_pts;
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}


// Structure to hold bounding box coordinates
struct BoundingBox {
    int left_top_x;
    int left_top_y;
    int right_bottom_x;
    int right_bottom_y;
};

inline inspirecv::Rect2i AlignmentBoxToStrideSquareBox(const inspirecv::Rect2i &bbox, int stride) {
    // 1. Convert xywh to cxcywh (center point coordinates and width/height)
    int center_x = bbox.GetX() + bbox.GetWidth() / 2;
    int center_y = bbox.GetY() + bbox.GetHeight() / 2;
    int width = bbox.GetWidth();
    int height = bbox.GetHeight();
    
    // 2. Get the shortest side of the width and height
    int min_side = std::min(width, height);
    
    // 3. Align the shortest side to the stride
    int aligned_side = (min_side / stride) * stride;
    
    // 4. Create a square box (keep the center point unchanged)
    int half_side = aligned_side / 2;
    int new_x = center_x - half_side;
    int new_y = center_y - half_side;
    
    // 5. Convert cxcywh back to xywh
    return inspirecv::Rect2i(new_x, new_y, aligned_side, aligned_side);
}

inline inspirecv::Rect2i GetNewBox(int src_w, int src_h, inspirecv::Rect2i bbox, float scale) {
    // Convert cv::Rect to BoundingBox
    BoundingBox box;
    box.left_top_x = bbox.GetX();
    box.left_top_y = bbox.GetY();
    box.right_bottom_x = bbox.GetX() + bbox.GetWidth();
    box.right_bottom_y = bbox.GetY() + bbox.GetHeight();

    // Compute new bounding box
    scale = std::min({static_cast<float>(src_h - 1) / bbox.GetHeight(), static_cast<float>(src_w - 1) / bbox.GetWidth(), scale});

    float new_width = bbox.GetWidth() * scale;
    float new_height = bbox.GetHeight() * scale;
    float center_x = bbox.GetWidth() / 2.0f + bbox.GetX();
    float center_y = bbox.GetHeight() / 2.0f + bbox.GetY();

    float left_top_x = center_x - new_width / 2.0f;
    float left_top_y = center_y - new_height / 2.0f;
    float right_bottom_x = center_x + new_width / 2.0f;
    float right_bottom_y = center_y + new_height / 2.0f;

    if (left_top_x < 0) {
        right_bottom_x -= left_top_x;
        left_top_x = 0;
    }

    if (left_top_y < 0) {
        right_bottom_y -= left_top_y;
        left_top_y = 0;
    }

    if (right_bottom_x > src_w - 1) {
        left_top_x -= right_bottom_x - src_w + 1;
        right_bottom_x = src_w - 1;
    }

    if (right_bottom_y > src_h - 1) {
        left_top_y -= right_bottom_y - src_h + 1;
        right_bottom_y = src_h - 1;
    }

    // Convert back to cv::Rect for output
    inspirecv::Rect2i new_bbox(static_cast<int>(left_top_x), static_cast<int>(left_top_y),
                      static_cast<int>(right_bottom_x - left_top_x), static_cast<int>(right_bottom_y - left_top_y));
    return new_bbox;
}


template<typename T>
inline bool isShortestSideGreaterThan(const inspirecv::Rect<T>& rect, T value, float scale) {
    // Find the shortest edge
    T shortestSide = std::min(static_cast<float>(rect.GetWidth()) / scale, static_cast<float>(rect.GetHeight()) / scale);
    // Determines whether the shortest edge is greater than the given value
    return shortestSide > value;
}

// Exponential Moving Average (EMA) filter function
inline float EmaFilter(float currentProb, std::vector<float> &history, int max, float alpha = 0.2f) {
    // Add current probability to history
    history.push_back(currentProb);

    // Trim history if it exceeds max size
    if (history.size() > max) {
        history.erase(history.begin(), history.begin() + (history.size() - max));
    }

    // Compute EMA
    float ema = history[0];  // Initial value
    for (size_t i = 1; i < history.size(); ++i) {
        ema = alpha * history[i] + (1 - alpha) * ema;
    }

    return ema;
}


// Vector EMA filter function
inline std::vector<float> VectorEmaFilter(const std::vector<float>& currentProbs, 
                                         std::vector<std::vector<float>>& history, 
                                         int max, 
                                         float alpha = 0.2f) {
    // Add current probability vector to history
    history.push_back(currentProbs);
    
    // Trim history if it exceeds max size
    if (history.size() > max) {
        history.erase(history.begin(), history.begin() + (history.size() - max));
    }
    
    // If only one sample, return it directly
    if (history.size() == 1) {
        return history[0];
    }
    
    // Compute EMA for each dimension
    std::vector<float> ema = history[0];  // Initial values
    
    for (size_t i = 1; i < history.size(); ++i) {
        for (size_t j = 0; j < ema.size(); ++j) {
            ema[j] = alpha * history[i][j] + (1 - alpha) * ema[j];
        }
    }
    
    return ema;
}

}  // namespace inspire

#endif  // INSPIRE_FACE_UTILS_H
