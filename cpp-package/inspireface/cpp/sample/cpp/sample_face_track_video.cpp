#include <iostream>
#include "c_api/intypedef.h"
#include "opencv2/opencv.hpp"
#include "inspireface/c_api/inspireface.h"
#include <unordered_map>
#include <functional>

void drawMode(cv::Mat& frame, HFDetectMode mode) {
    std::string modeText;
    switch (mode) {
        case HF_DETECT_MODE_ALWAYS_DETECT:
            modeText = "Mode: Image Detection";
            break;
        case HF_DETECT_MODE_LIGHT_TRACK:
            modeText = "Mode: Video Detection";
            break;
        case HF_DETECT_MODE_TRACK_BY_DETECTION:
            modeText = "Mode: Track by Detection";
            break;
        default:
            modeText = "Mode: Unknown";
            break;
    }
    cv::putText(frame, modeText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(90, 100, 255), 2);
}

cv::Scalar generateColor(int id) {
    int maxID = 100;
    id = id % maxID;

    int hue = (id * 360 / maxID) % 360; 
    int saturation = 255; 
    int value = 200;

    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, saturation, value));
    cv::Mat rgb;
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);

    cv::Vec3b rgbColor = rgb.at<cv::Vec3b>(0, 0);
    return cv::Scalar(rgbColor[0], rgbColor[1], rgbColor[2]);
}


int main(int argc, char* argv[]) {
    // Check whether the number of parameters is correct
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <pack_path> <video_path>\n";
        return 1;
    }

    auto packPath = argv[1];
    auto videoPath = argv[2];

    std::cout << "Pack file Path: " << packPath << std::endl;
    std::cout << "Video file Path: " << videoPath << std::endl;

    HResult ret;
    // The resource file must be loaded before it can be used
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        std::cout << "Load Resource error: " << ret << std::endl;
        return ret;
    }

    // Enable the functions in the pipeline: mask detection, live detection, and face quality detection
    HOption option = HF_ENABLE_QUALITY | HF_ENABLE_MASK_DETECT | HF_ENABLE_INTERACTION;
    // Video or frame sequence mode uses VIDEO-MODE, which is face detection with tracking
    HFDetectMode detMode = HF_DETECT_MODE_TRACK_BY_DETECTION;
    // Maximum number of faces detected
    HInt32 maxDetectNum = 20;
    // Face detection image input level
    HInt32 detectPixelLevel = 640;
    // fps in tracking-by-detection mode
    HInt32 trackByDetectFps = 20;
    HFSession session = {0};
    // Handle of the current face SDK algorithm context
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, trackByDetectFps, &session);
    if (ret != HSUCCEED) {
        std::cout << "Create FaceContext error: " << ret << std::endl;
        return ret;
    }

    HFSessionSetTrackPreviewSize(session, detectPixelLevel);
    HFSessionSetFilterMinimumFacePixelSize(session, 0);

    // Open the video file
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cout << "The source entered is not a video or read error." << std::endl;
        return 1;
    }

    // Get the video properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    cv::Size frame_size(frame_width, frame_height);

    // Define the codec and create VideoWriter object
    cv::VideoWriter outputVideo("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame_size, true);
    if (!outputVideo.isOpened()) {
        std::cerr << "Could not open the output video for write: output_video.avi\n";
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        // Prepare an image parameter structure for configuration
        HFImageData imageParam = {0};
        imageParam.data = frame.data;       // Data buffer
        imageParam.width = frame.cols;      // Target view width
        imageParam.height = frame.rows;      // Target view width
        imageParam.rotation = HF_CAMERA_ROTATION_0;      // Data source rotate
        imageParam.format = HF_STREAM_BGR;      // Data source format

        // Create an image data stream
        HFImageStream imageHandle = {0};
        ret = HFCreateImageStream(&imageParam, &imageHandle);
        if (ret != HSUCCEED) {
            std::cout << "Create ImageStream error: " << ret << std::endl;
            return ret;
        }

        // Execute HF_FaceContextRunFaceTrack captures face information in an image
        double time = (double) cv::getTickCount();
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
        time = ((double) cv::getTickCount() - time) / cv::getTickFrequency();
        std::cout << "use time：" << time << "秒\n";
        if (ret != HSUCCEED) {
            std::cout << "Execute HFExecuteFaceTrack error: " << ret << std::endl;
            return ret;
        }

        // Print the number of faces detected
        auto faceNum = multipleFaceData.detectedNum;
        std::cout << "Num of face: " << faceNum << std::endl;

        // Copy a new image to draw
        cv::Mat draw = frame.clone();

        // Draw detection mode on the frame
        drawMode(draw, detMode);
        if (faceNum > 0) {
            ret = HFMultipleFacePipelineProcessOptional(session, imageHandle, &multipleFaceData, option);
            if (ret != HSUCCEED)
            {   
                std::cout << "HFMultipleFacePipelineProcessOptional error: " << ret << std::endl;
                return ret;
            }
            HFFaceIntereactionState result;
            ret = HFGetFaceIntereactionStateResult(session, &result);
             if (ret != HSUCCEED)
            {   
                std::cout << "HFGetFaceIntereactionStateResult error: " << ret << std::endl;
                return ret;
            }
            std::cout << "Left eye status: " << result.leftEyeStatusConfidence[0] << std::endl;
            std::cout << "Righ eye status: " << result.rightEyeStatusConfidence[0] << std::endl;

        }
        
        for (int index = 0; index < faceNum; ++index) {
            // std::cout << "========================================" << std::endl;
            // std::cout << "Process face index: " << index << std::endl;
            // Print FaceID, In VIDEO-MODE it is fixed, but it may be lost
            auto trackId = multipleFaceData.trackIds[index];

            // Use OpenCV's Rect to receive face bounding boxes
            auto rect = cv::Rect(multipleFaceData.rects[index].x, multipleFaceData.rects[index].y,
                                 multipleFaceData.rects[index].width, multipleFaceData.rects[index].height);
            cv::rectangle(draw, rect, generateColor(trackId), 3);

            // std::cout << "FaceID: " << trackId << std::endl;

            // Print Head euler angle, It can often be used to judge the quality of a face by the Angle of the head
            // std::cout << "Roll: " << multipleFaceData.angles.roll[index]
            //           << ", Yaw: " << multipleFaceData.angles.yaw[index]
            //           << ", Pitch: " << multipleFaceData.angles.pitch[index] << std::endl;

            // Add TrackID to the drawing
            cv::putText(draw, "ID: " + std::to_string(trackId), cv::Point(rect.x, rect.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, generateColor(trackId), 2);

            HInt32 numOfLmk;
            HFGetNumOfFaceDenseLandmark(&numOfLmk);
            HPoint2f denseLandmarkPoints[numOfLmk];
            ret = HFGetFaceDenseLandmarkFromFaceToken(multipleFaceData.tokens[index], denseLandmarkPoints, numOfLmk);
            if (ret != HSUCCEED) {
                std::cerr << "HFGetFaceDenseLandmarkFromFaceToken error!!" << std::endl;
                return -1;
            }
            for (size_t i = 0; i < numOfLmk; i++) {
                cv::Point2f p(denseLandmarkPoints[i].x, denseLandmarkPoints[i].y);
                cv::circle(draw, p, 0, generateColor(trackId), 2);
            }
        }
        
        cv::imshow("w", draw);
        cv::waitKey(1);

        // Write the frame into the file
        outputVideo.write(draw);

        ret = HFReleaseImageStream(imageHandle);
        if (ret != HSUCCEED) {
            printf("Release image stream error: %lu\n", ret);
        }
    }

    // Release the VideoCapture and VideoWriter objects
    cap.release();
    outputVideo.release();

    // The memory must be freed at the end of the program
    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        printf("Release session error: %lu\n", ret);
        return ret;
    }

    return 0;
}
