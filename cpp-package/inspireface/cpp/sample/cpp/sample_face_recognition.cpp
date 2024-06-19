#include <iostream>
#include "opencv2/opencv.hpp"
#include "inspireface/c_api/inspireface.h"

int main(int argc, char* argv[]) {
    // Check if the correct number of parameters was provided
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <pack_path>\n";
        return 1;
    }

    auto packPath = argv[1]; // Path to the resource pack

    std::string testDir = "test_res/"; // Directory containing test resources

    HResult ret;

    // Load resource file, necessary before using any functionality
    ret = HFLaunchInspireFace(packPath);
    if (ret != HSUCCEED) {
        std::cout << "Load Resource error: " << ret << std::endl;
        return ret;
    }

    // Configuration for the feature database
    HFFeatureHubConfiguration featureHubConfiguration;
    featureHubConfiguration.featureBlockNum = 10; // Number of feature blocks
    featureHubConfiguration.enablePersistence = 0; // Persistence not enabled, use in-memory database
    featureHubConfiguration.dbPath = ""; // Database path (not used here)
    featureHubConfiguration.searchMode = HF_SEARCH_MODE_EAGER; // Search mode configuration
    featureHubConfiguration.searchThreshold = 0.48f; // Threshold for search operations

    // Enable the global feature database
    ret = HFFeatureHubDataEnable(featureHubConfiguration);
    if (ret != HSUCCEED) {
        std::cout << "An exception occurred while starting FeatureHub: " << ret << std::endl;
        return ret;
    }

    // Prepare a list of face photos for testing
    std::vector<std::string> photos = {
            testDir  + "data/bulk/Nathalie_Baye_0002.jpg",
            testDir  + "data/bulk/jntm.jpg",
            testDir  + "data/bulk/woman.png",
            testDir  + "data/bulk/Rob_Lowe_0001.jpg",
    };
    std::vector<std::string> names = {
            "Nathalie Baye",
            "JNTM",
            "Woman",
            "Rob Lowe",
    };
    assert(photos.size() == names.size()); // Ensure each photo has a corresponding name

    // Create a session for face recognition
    HOption option = HF_ENABLE_FACE_RECOGNITION;
    HFSession session;
    ret = HFCreateInspireFaceSessionOptional(option, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &session);
    if (ret != HSUCCEED) {
        std::cout << "Create session error: " << ret << std::endl;
        return ret;
    }

    // Process each photo, extract features, and add them to the database
    for (int i = 0; i < photos.size(); ++i) {
        std::cout << "===============================" << std::endl;
        // Load the image from the specified file path
        const auto& path = photos[i];
        const auto& name = names[i];
        auto image = cv::imread(path);
        if (image.empty()) {
            std::cout << "The image is empty: " << path << ret << std::endl;
            return ret;
        }

        // Prepare image data for processing
        HFImageData imageData = {0};
        imageData.data = image.data; // Pointer to the image data
        imageData.format = HF_STREAM_BGR; // Image format (BGR in this case)
        imageData.height = image.rows; // Image height
        imageData.width = image.cols; // Image width
        imageData.rotation = HF_CAMERA_ROTATION_0; // Image rotation
        HFImageStream stream;
        ret = HFCreateImageStream(&imageData, &stream); // Create an image stream for processing
        if (ret != HSUCCEED) {
            std::cout << "Create stream error: " << ret << std::endl;
            return ret;
        }

        // Execute face tracking on the image
        HFMultipleFaceData multipleFaceData = {0};
        ret = HFExecuteFaceTrack(session, stream, &multipleFaceData); // Track faces in the image
        if (ret != HSUCCEED) {
            std::cout << "Run face track error: " << ret << std::endl;
            return ret;
        }
        if (multipleFaceData.detectedNum == 0) { // Check if any faces were detected
            std::cout << "No face was detected: " << path << ret << std::endl;
            return ret;
        }

        // Extract facial features from the first detected face
        HFFaceFeature feature = {0};
        ret = HFFaceFeatureExtract(session, stream, multipleFaceData.tokens[0], &feature); // Extract features
        if (ret != HSUCCEED) {
            std::cout << "Extract feature error: " << ret << std::endl;
            return ret;
        }

        // Assign a name to the detected face and insert it into the feature hub
        char* cstr = new char[name.size() + 1]; // Dynamically allocate memory for the name
        strcpy(cstr, name.c_str()); // Copy the name into the allocated memory
        HFFaceFeatureIdentity identity = {0};
        identity.feature = &feature; // Assign the extracted feature
        identity.customId = i; // Custom identifier for the face
        identity.tag = cstr; // Tag the feature with the name
        ret = HFFeatureHubInsertFeature(identity); // Insert the feature into the hub
        if (ret != HSUCCEED) {
            std::cout << "Feature insertion into FeatureHub failed: " << ret << std::endl;
            return ret;
        }

        delete[] cstr; // Clean up the dynamically allocated memory

        std::cout << "Insert feature to FeatureHub: " << name << std::endl;
        ret = HFReleaseImageStream(stream); // Release the image stream
        if (ret != HSUCCEED) {
            std::cout << "Release stream failed: " << ret << std::endl;
            return ret;
        }
    }

    HInt32 count;
    ret = HFFeatureHubGetFaceCount(&count);
    assert(count == photos.size());
    std::cout << "\nInserted data: " << count << std::endl;

    // Process a query image and search for similar faces in the database
    auto query = cv::imread(testDir + "data/bulk/kun.jpg");
    if (query.empty()) {
        std::cout << "The query image is empty: " << ret << std::endl;
        return ret;
    }
    HFImageData imageData = {0};
    imageData.data = query.data;
    imageData.format = HF_STREAM_BGR;
    imageData.height = query.rows;
    imageData.width = query.cols;
    imageData.rotation = HF_CAMERA_ROTATION_0;
    HFImageStream stream;
    ret = HFCreateImageStream(&imageData, &stream);
    if (ret != HSUCCEED) {
        std::cout << "Create stream error: " << ret << std::endl;
        return ret;
    }

    HFMultipleFaceData multipleFaceData = {0};
    ret = HFExecuteFaceTrack(session, stream, &multipleFaceData);
    if (ret != HSUCCEED) {
        std::cout << "Run face track error: " << ret << std::endl;
        return ret;
    }
    if (multipleFaceData.detectedNum == 0) {
        std::cout << "No face was detected from target image: " << ret << std::endl;
        return ret;
    }

    // Initialize the feature structure to store extracted face features
    HFFaceFeature feature = {0};
    // Extract facial features from the detected face using the first token
    ret = HFFaceFeatureExtract(session, stream, multipleFaceData.tokens[0], &feature);
    if (ret != HSUCCEED) {
        std::cout << "Extract feature error: " << ret << std::endl; // Print error if extraction fails
        return ret;
    }

    // Initialize the structure to store the results of the face search
    HFFaceFeatureIdentity searched = {0};
    HFloat confidence; // Variable to store the confidence level of the search result
    // Search the feature hub for a matching face feature
    ret = HFFeatureHubFaceSearch(feature, &confidence, &searched);
    if (ret != HSUCCEED) {
        std::cout << "Search face feature error: " << ret << std::endl; // Print error if search fails
        return ret;
    }
    if (searched.customId == -1) {
        std::cout << "No similar faces were found: " << std::endl; // Notify if no matching face is found
        return ret;
    }
    // Output the details of the found face, including custom ID, associated tag, and confidence level
    std::cout << "\nFound similar face: id=" << searched.customId << ", tag=" << searched.tag << ", confidence=" << confidence << std::endl;
    std::string name(searched.tag);

    // Remove feature
    ret = HFFeatureHubFaceRemove(searched.customId);
    if (ret != HSUCCEED) {
        std::cout << "Remove failed: " << ret << std::endl; // Print error if search fails
        return ret;
    }
    // Remove feature and search again
    ret = HFFeatureHubFaceSearch(feature, &confidence, &searched);
    if (ret != HSUCCEED) {
        std::cout << "Search face feature error: " << ret << std::endl; // Print error if search fails
        return ret;
    }
    if (searched.customId != -1) {
        std::cout << "Remove an exception: " << std::endl; // Notify if no matching face is found
        return ret;
    }
    std::cout << "\nSearch again confidence=" << confidence << std::endl;
    std::cout << name << " has been removed." << std::endl;

    // Clean up and close the session
    ret = HFReleaseImageStream(stream);
    if (ret != HSUCCEED) {
        std::cout << "Release stream error: " << ret << std::endl;
        return ret;
    }

    ret = HFReleaseInspireFaceSession(session);
    if (ret != HSUCCEED) {
        std::cout << "Release session error: " << ret << std::endl;
        return ret;
    }

    return ret; // Return the final result code
}
