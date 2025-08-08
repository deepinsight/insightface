/**
 * Created by Jingyu Yan
 * @date 2025-06-29
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inspireface.h>

uint8_t* read_binary_file(const char* filename, size_t* file_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    uint8_t* buffer = (uint8_t*)malloc(*file_size);
    if (!buffer) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    size_t bytes_read = fread(buffer, 1, *file_size, file);
    fclose(file);
    
    if (bytes_read != *file_size) {
        printf("Error: File read incomplete. Expected %zu bytes, got %zu bytes\n", 
               *file_size, bytes_read);
        free(buffer);
        return NULL;
    }
    
    return buffer;
}

void free_binary_data(uint8_t* data) {
    if (data) {
        free(data);
    }
}

int main() {
    HResult ret;
    ret = HFLaunchInspireFace("test_res/pack/Pikachu");
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Load Resource error: %d", ret);
        return ret;
    }
    HOption option = HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_MASK_DETECT | HF_ENABLE_QUALITY;
    HFDetectMode detMode = HF_DETECT_MODE_ALWAYS_DETECT;
    HInt32 maxDetectNum = 1;
    HInt32 detectPixelLevel = 320;
    HFSession session = {0};
    ret = HFCreateInspireFaceSessionOptional(option, detMode, maxDetectNum, detectPixelLevel, -1, &session);
    if (ret != HSUCCEED)
    {
        HFLogPrint(HF_LOG_ERROR, "Create FaceContext error: %d", ret);
        return ret;
    }

    // HFImageBitmap image;
    HFImageStream imageHandle;
    /* Load a image */
    // ret = HFCreateImageBitmapFromFilePath("/Users/tunm/Downloads/Desktop/outputbw.jpg", 1, &image);
    // if (ret != HSUCCEED) {
    //     HFLogPrint(HF_LOG_ERROR, "The source entered is not a picture or read error.");
    //     return ret;
    // }
    // /* Prepare an image parameter structure for configuration */
    // ret = HFCreateImageStreamFromImageBitmap(image, HF_CAMERA_ROTATION_0, &imageHandle);
    // if (ret != HSUCCEED) {
    //     HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
    //     return ret;
    // }


    size_t file_size;
    uint8_t* buffer = read_binary_file("/Users/tunm/Downloads/Desktop/outputbw.byte", &file_size);
    if (buffer == NULL) {
        HFLogPrint(HF_LOG_ERROR, "Read file error.");
        return -1;
    }
    HFImageData imageData;
    imageData.data = buffer;
    imageData.width = 640;
    imageData.height = 480;
    imageData.format = HF_STREAM_GRAY;
    imageData.rotation = HF_CAMERA_ROTATION_0;

    ret = HFCreateImageStream(&imageData, &imageHandle);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Create ImageStream error: %d", ret);
        return ret;
    }

    HFDeBugImageStreamDecodeSave(imageHandle, "2.jpg");

    HFMultipleFaceData multipleFaceData;
    ret = HFExecuteFaceTrack(session, imageHandle, &multipleFaceData);
    if (ret != HSUCCEED) {
        HFLogPrint(HF_LOG_ERROR, "Detect Face error: %d", ret);
        return ret;
    }
    HFLogPrint(HF_LOG_INFO, "Face num: %d", multipleFaceData.detectedNum);


    return 0;
}