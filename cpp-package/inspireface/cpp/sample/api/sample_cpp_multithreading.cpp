/**
 * Created by Jingyu Yan
 * @date 2025-07-13
 * 
 * Multi-threading example: One thread performs face tracking and stores tokens,
 * another thread monitors tokens and extracts face features.
 */
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <memory>
#include <inspireface.h>

// Thread-safe token storage using C++ containers
class ThreadSafeTokenStorage {
private:
    std::vector<HFFaceBasicToken> tokens_;
    mutable std::mutex mutex_;  // Make mutex mutable for const member functions
    std::condition_variable cv_;
    std::atomic<bool> stop_flag_{false};

public:
    // Add a token to the storage
    void addToken(const HFFaceBasicToken& token) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Copy token data
        HInt32 tokenSize;
        HFGetFaceBasicTokenSize(&tokenSize);
        char* tokenBuffer = new char[tokenSize];
        HFCopyFaceBasicToken(token, tokenBuffer, tokenSize);
        
        // Create copied token
        HFFaceBasicToken copiedToken;
        copiedToken.size = tokenSize;
        copiedToken.data = tokenBuffer;
        
        tokens_.push_back(copiedToken);
        cv_.notify_one();
    }
    
    // Get the last token and remove it
    bool getLastToken(HFFaceBasicToken& token) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (tokens_.empty()) {
            return false;
        }
        
        token = tokens_.back();
        tokens_.pop_back();
        return true;
    }
    
    // Wait for a token with timeout
    bool waitForToken(HFFaceBasicToken& token, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                         [this] { return !tokens_.empty() || stop_flag_.load(); })) {
            if (!tokens_.empty()) {
                token = tokens_.back();
                tokens_.pop_back();
                return true;
            }
        }
        return false;
    }
    
    // Check if there are tokens available
    bool hasTokens() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !tokens_.empty();
    }
    
    // Get token count
    size_t getTokenCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tokens_.size();
    }
    
    // Stop the threads
    void stop() {
        stop_flag_.store(true);
        cv_.notify_all();
    }
    
    // Check if should stop
    bool shouldStop() const {
        return stop_flag_.load();
    }
    
    // Cleanup allocated memory
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& token : tokens_) {
            if (token.data != nullptr) {
                delete[] static_cast<char*>(token.data);
            }
        }
        tokens_.clear();
    }
    
    ~ThreadSafeTokenStorage() {
        cleanup();
    }
};

// Face tracking thread function
void faceTrackingThread(ThreadSafeTokenStorage& tokenStorage, 
                       HFSession session,
                       const std::string& imagePath,
                       int maxFrames = 100) {
    std::cout << "Face tracking thread started" << std::endl;
    
    // Load image using C API
    HFImageBitmap imageBitmap = {0};
    HResult ret = HFCreateImageBitmapFromFilePath(imagePath.c_str(), 3, &imageBitmap);
    if (ret != HSUCCEED) {
        std::cerr << "Failed to create image bitmap: " << ret << std::endl;
        return;
    }
    
    // Create image stream
    HFImageStream stream;
    ret = HFCreateImageStreamFromImageBitmap(imageBitmap, HF_CAMERA_ROTATION_0, &stream);
    if (ret != HSUCCEED) {
        std::cerr << "Failed to create image stream: " << ret << std::endl;
        HFReleaseImageBitmap(imageBitmap);
        return;
    }
    
    int frameCount = 0;
    while (!tokenStorage.shouldStop() && frameCount < maxFrames) {
        // Perform face detection and tracking using C API
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, stream, &multipleFaceData);
        
        if (ret == HSUCCEED && multipleFaceData.detectedNum > 0) {
            // Add tokens to storage
            for (int i = 0; i < multipleFaceData.detectedNum; i++) {
                tokenStorage.addToken(multipleFaceData.tokens[i]);
                
                std::cout << "Frame " << frameCount << ": Added token for face " 
                          << multipleFaceData.trackIds[i] << std::endl;
            }
        }
        
        frameCount++;
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate frame processing
    }
    
    // Cleanup
    HFReleaseImageStream(stream);
    HFReleaseImageBitmap(imageBitmap);
    
    std::cout << "Face tracking thread finished after " << frameCount << " frames" << std::endl;
}

// Face feature extraction thread function
void featureExtractionThread(ThreadSafeTokenStorage& tokenStorage,
                           HFSession session,
                           const std::string& imagePath) {
    std::cout << "Feature extraction thread started" << std::endl;
    
    // Load image using C API (same as tracking thread)
    HFImageBitmap imageBitmap = {0};
    HResult ret = HFCreateImageBitmapFromFilePath(imagePath.c_str(), 3, &imageBitmap);
    if (ret != HSUCCEED) {
        std::cerr << "Failed to create image bitmap: " << ret << std::endl;
        return;
    }
    
    // Create image stream
    HFImageStream stream;
    ret = HFCreateImageStreamFromImageBitmap(imageBitmap, HF_CAMERA_ROTATION_0, &stream);
    if (ret != HSUCCEED) {
        std::cerr << "Failed to create image stream: " << ret << std::endl;
        HFReleaseImageBitmap(imageBitmap);
        return;
    }
    
    int extractedCount = 0;
    while (!tokenStorage.shouldStop()) {
        HFFaceBasicToken token;
        
        // Wait for token with timeout
        if (tokenStorage.waitForToken(token, 2000)) {
            // Extract face feature using C API
            HFFaceFeature feature;
            ret = HFCreateFaceFeature(&feature);
            if (ret == HSUCCEED) {
                ret = HFFaceFeatureExtractTo(session, stream, token, feature);
                if (ret == HSUCCEED) {
                    extractedCount++;
                    std::cout << "Extracted feature " << extractedCount 
                              << " (feature size: " << feature.size << ")" << std::endl;
                    
                    // Print first few feature values as example
                    std::cout << "Feature values: ";
                    for (int i = 0; i < std::min(5, feature.size); i++) {
                        std::cout << feature.data[i] << " ";
                    }
                    std::cout << "..." << std::endl;
                } else {
                    std::cerr << "Feature extraction failed with error: " << ret << std::endl;
                }
                
                HFReleaseFaceFeature(&feature);
            }
            
            // Clean up token memory
            if (token.data != nullptr) {
                delete[] static_cast<char*>(token.data);
            }
        } else {
            std::cout << "No tokens available, waiting..." << std::endl;
        }
    }
    
    // Cleanup
    HFReleaseImageStream(stream);
    HFReleaseImageBitmap(imageBitmap);
    
    std::cout << "Feature extraction thread finished, extracted " 
              << extractedCount << " features" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    
    std::string modelPath = argv[1];
    std::string imagePath = argv[2];
    
    // Initialize InspireFace using C API
    HResult ret = HFLaunchInspireFace(modelPath.c_str());
    if (ret != HSUCCEED) {
        std::cerr << "Failed to launch InspireFace: " << ret << std::endl;
        return -1;
    }
    
    // Create session using C API
    HOption option = HF_ENABLE_FACE_RECOGNITION;
    HFSession session;
    ret = HFCreateInspireFaceSessionOptional(option, HF_DETECT_MODE_LIGHT_TRACK, 10, -1, -1, &session);
    if (ret != HSUCCEED) {
        std::cerr << "Failed to create session: " << ret << std::endl;
        return -1;
    }
    
    // Set session parameters using C API
    HFSessionSetTrackPreviewSize(session, 640);
    
    // Create thread-safe token storage
    ThreadSafeTokenStorage tokenStorage;
    
    // Start face tracking thread
    std::thread trackingThread(faceTrackingThread, std::ref(tokenStorage), session, imagePath, 50);
    
    // Start feature extraction thread
    std::thread extractionThread(featureExtractionThread, std::ref(tokenStorage), session, imagePath);
    
    // Main thread: monitor and print statistics
    std::cout << "Main thread: Monitoring token storage..." << std::endl;
    for (int i = 0; i < 30; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        size_t tokenCount = tokenStorage.getTokenCount();
        std::cout << "Token storage status: " << tokenCount << " tokens available" << std::endl;
    }
    
    // Stop threads
    std::cout << "Stopping threads..." << std::endl;
    tokenStorage.stop();
    
    // Wait for threads to finish
    trackingThread.join();
    extractionThread.join();
    
    // Cleanup using C API
    HFReleaseInspireFaceSession(session);
    
    std::cout << "All threads finished successfully" << std::endl;
    
    return 0;
}
