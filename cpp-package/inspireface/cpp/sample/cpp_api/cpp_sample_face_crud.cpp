#include <iostream>
#include <inspireface/inspireface.hpp>

int main() {
    // Launch InspireFace
    std::string model_path = "test_res/pack/Pikachu";
    INSPIREFACE_CONTEXT->Reload(model_path);
    INSPIREFACE_CHECK_MSG(INSPIREFACE_CONTEXT->isMLoad(), "InspireFace is not loaded");

    // Enable feature hub
    std::string db_path = "case_crud.db";
    // Remove the database file if it exists
    if (std::remove(db_path.c_str()) != 0) {
        std::cerr << "Error removing database file: " << db_path << std::endl;
    }
    inspire::DatabaseConfiguration db_config;
    db_config.enable_persistence = true;
    db_config.persistence_db_path = db_path;
    db_config.search_mode = inspire::SEARCH_MODE_EXHAUSTIVE;
    db_config.recognition_threshold = 0.48f;
    db_config.primary_key_mode = inspire::AUTO_INCREMENT;
    auto ret = INSPIREFACE_FEATURE_HUB->EnableHub(db_config);
    INSPIREFACE_CHECK_MSG(ret == HSUCCEED, "EnableHub failed");

    // Create a session
    auto param = inspire::CustomPipelineParameter();
    param.enable_recognition = true;
    auto session = inspire::Session::CreatePtr(inspire::DETECT_MODE_ALWAYS_DETECT, 1, param, 320);
    INSPIREFACE_CHECK_MSG(session != nullptr, "Session is not created");

    // Prepare an image for insertion into the hub
    auto image = inspirecv::Image::Create("test_res/data/bulk/kun.jpg");
    auto image_process = inspirecv::FrameProcess::Create(image.Data(), image.Height(), image.Width(), inspirecv::BGR, inspirecv::ROTATION_0);

    // Detect and track
    std::vector<inspire::FaceTrackWrap> results;
    session->FaceDetectAndTrack(image_process, results);
    INSPIREFACE_CHECK_MSG(results.size() > 0, "No face detected");

    // Extract face feature
    inspire::FaceEmbedding feature;
    session->FaceFeatureExtract(image_process, results[0], feature);

    // Insert face feature into the hub, because the id is INSPIRE_INVALID_ID, so input id is ignored
    int64_t result_id;
    INSPIREFACE_FEATURE_HUB->FaceFeatureInsert(feature.embedding, INSPIRE_INVALID_ID, result_id);

    inspire::FaceEmbedding face_feature;
    INSPIREFACE_FEATURE_HUB->GetFaceFeature(result_id, face_feature);
    
    // Prepare a photo of the same person for the query
    auto query_image = inspirecv::Image::Create("test_res/data/bulk/jntm.jpg");
    auto query_image_process = inspirecv::FrameProcess::Create(query_image.Data(), query_image.Height(), query_image.Width(), inspirecv::BGR, inspirecv::ROTATION_0);

    // Detect and track
    std::vector<inspire::FaceTrackWrap> query_results;
    session->FaceDetectAndTrack(query_image_process, query_results);
    INSPIREFACE_CHECK_MSG(query_results.size() > 0, "No face detected");

    // Extract face feature
    inspire::FaceEmbedding query_feature;
    session->FaceFeatureExtract(query_image_process, query_results[0], query_feature);

    // Search face feature
    inspire::FaceSearchResult search_result;
    INSPIREFACE_FEATURE_HUB->SearchFaceFeature(query_feature.embedding, search_result, true);
    std::cout << "Search face feature result: " << search_result.id << std::endl;
    std::cout << "Search face feature similarity: " << search_result.similarity << std::endl;

    INSPIREFACE_CHECK_MSG(search_result.id == result_id, "Search face feature result id is not equal to the inserted id");
    
    // Update the face feature
    INSPIREFACE_FEATURE_HUB->FaceFeatureUpdate(query_feature.embedding, result_id);

    // Remove the face feature
    INSPIREFACE_FEATURE_HUB->FaceFeatureRemove(result_id);
    INSPIREFACE_CHECK_MSG(INSPIREFACE_FEATURE_HUB->GetFaceFeatureCount() == 0, "Face feature is not removed");

    
    std::cout << "Remove face feature successfully" << std::endl;

    // Query again
    INSPIREFACE_FEATURE_HUB->SearchFaceFeature(query_feature.embedding, search_result, true);
    INSPIREFACE_CHECK_MSG(search_result.id == INSPIRE_INVALID_ID, "Search face feature result id is not equal to the inserted id");
    std::cout << "Query again, search face feature result: " << search_result.id << std::endl;

    
    // Top-k query
    std::vector<inspire::FaceSearchResult> top_k_results;
    INSPIREFACE_FEATURE_HUB->SearchFaceFeatureTopK(query_feature.embedding, top_k_results, 10, true);
    std::cout << "Top-k query result: " << top_k_results.size() << std::endl;

    return 0;
}