#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <inspirecv/inspirecv.h>
#include <inspireface/inspireface.hpp>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path1> <image_path2>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path1 = argv[2];
    std::string image_path2 = argv[3];

    // Global init(only once)
    INSPIREFACE_CONTEXT->Reload(model_path);

    // Create image and frame process
    inspirecv::Image image1 = inspirecv::Image::Create(image_path1);
    inspirecv::Image image2 = inspirecv::Image::Create(image_path2);
    inspirecv::FrameProcess process1 = inspirecv::FrameProcess::Create(image1.Data(), image1.Height(), image1.Width(), inspirecv::BGR, inspirecv::ROTATION_0);
    inspirecv::FrameProcess process2 = inspirecv::FrameProcess::Create(image2.Data(), image2.Height(), image2.Width(), inspirecv::BGR, inspirecv::ROTATION_0);

    // Create session
    inspire::CustomPipelineParameter param;
    param.enable_recognition = true;

    // Create session
    std::shared_ptr<inspire::Session> session(
        inspire::Session::CreatePtr(inspire::DETECT_MODE_ALWAYS_DETECT, 1, param, 320));

    INSPIREFACE_CHECK_MSG(session != nullptr, "Session is not valid");

    // Detect and track
    std::vector<inspire::FaceTrackWrap> results1;
    std::vector<inspire::FaceTrackWrap> results2;

    // Detect and track
    session->FaceDetectAndTrack(process1, results1);
    session->FaceDetectAndTrack(process2, results2);

    INSPIREFACE_CHECK_MSG(!results1.empty() && !results2.empty(), "No face detected");

    // Get feature
    inspire::FaceEmbedding feature1;
    inspire::FaceEmbedding feature2;
    session->FaceFeatureExtract(process1, results1[0], feature1);
    session->FaceFeatureExtract(process2, results2[0], feature2);

    // Compare
    float similarity;
    INSPIREFACE_FEATURE_HUB->CosineSimilarity(feature1.embedding, feature2.embedding, similarity);
    std::cout << "cosine of similarity: " << similarity << std::endl;
    std::cout << "percentage of similarity: " << SIMILARITY_CONVERTER_RUN(similarity) << std::endl;

    std::cout << "== using alignment image ==" << std::endl;

    // Get face alignment image
    inspirecv::Image wrapped1;
    inspirecv::Image wrapped2;
    session->GetFaceAlignmentImage(process1, results1[0], wrapped1);
    session->GetFaceAlignmentImage(process2, results2[0], wrapped2);
    wrapped1.Write("wrapped1.jpg");
    wrapped2.Write("wrapped2.jpg");

    inspire::FaceEmbedding feature1_alignment;
    inspire::FaceEmbedding feature2_alignment;
    session->FaceFeatureExtractWithAlignmentImage(wrapped1, feature1_alignment);
    session->FaceFeatureExtractWithAlignmentImage(wrapped2, feature2_alignment);

    INSPIREFACE_FEATURE_HUB->CosineSimilarity(feature1_alignment.embedding, feature2_alignment.embedding, similarity);
    std::cout << "cosine of similarity: " << similarity << std::endl;
    std::cout << "percentage of similarity: " << SIMILARITY_CONVERTER_RUN(similarity) << std::endl;
    return 0;
}