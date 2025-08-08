import os
import inspireface as isf
import numpy as np
import os

import random

random.seed(43)

def gen_feature():
    # Generate a random vector of length 512 and normalize it
    vector = np.random.uniform(-1, 1, 512).astype(np.float32)
    normalized_vector = vector / np.linalg.norm(vector)
    return normalized_vector

def gen_similar_feature(feature):
    noise_strength = 0.3
    
    noise = np.random.uniform(-1, 1, len(feature)).astype(np.float32)
    noise = noise / np.linalg.norm(noise)  
    
    similar_vector = feature + noise_strength * noise
    
    similar_vector = similar_vector / np.linalg.norm(similar_vector)
    
    return similar_vector

def case_feature_hub():
    # Set db path
    db_path = "test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    # Configure the feature management system.
    feature_hub_config = isf.FeatureHubConfiguration(
        primary_key_mode=isf.HF_PK_AUTO_INCREMENT,
        enable_persistence=True,
        persistence_db_path=db_path,
        search_threshold=0.48,
        search_mode=isf.HF_SEARCH_MODE_EAGER,
    )
    ret = isf.feature_hub_enable(feature_hub_config)
    assert ret, "Failed to enable FeatureHub."
    
    embedding_list = []
    for i in range(10):
        # Generate a random embedding
        embedding = gen_feature()
        face_identity = isf.FaceIdentity(embedding, -1)
        # Insert the face identity into the feature hub
        ret, id = isf.feature_hub_face_insert(face_identity)
        assert ret, "Failed to insert face identity."
        print(f"Inserted face id: {id}")
        embedding_list.append(embedding)

    print(f"face count: {isf.feature_hub_get_face_count()}")
    assert isf.feature_hub_get_face_count() == 10, "Failed to insert face identity."

    # Search for the face identity
    query_embedding = embedding_list[3]
    search_result = isf.feature_hub_face_search(query_embedding)
    assert search_result, "Failed to search for face identity."
    # The auto-increment id is calculated starting from 1
    assert search_result.similar_identity.id == 3 + 1, "Failed to search for face identity."
    print(f"search confidence: {search_result.confidence}")

    # Update the face identity
    update_embedding = gen_feature()
    update_face_identity = isf.FaceIdentity(update_embedding, search_result.similar_identity.id)
    ret = isf.feature_hub_face_update(update_face_identity)
    assert ret, "Failed to update face identity."
    print(f"Updated face id: {search_result.similar_identity.id}")
    
    # Search for the face identity again
    search_result = isf.feature_hub_face_search(query_embedding)
    assert search_result, "Failed to search for face identity."
    assert search_result.similar_identity.id == -1, "Failed to update face identity."
    print(f"search confidence: {search_result.confidence}, id: {search_result.similar_identity.id}")

    # Delete the face identity
    ret = isf.feature_hub_face_remove(4)
    assert ret, "Failed to delete face identity."
    print(f"Deleted face id: {4}")
    
    # Search for the face identity again
    search_result = isf.feature_hub_face_search(embedding_list[3])
    assert search_result.similar_identity.id == -1, "Failed to delete face identity."
    print(f"search confidence: {search_result.confidence}, id: {search_result.similar_identity.id}")
    print(f"face count: {isf.feature_hub_get_face_count()}")
    assert isf.feature_hub_get_face_count() == 9, "Failed to delete face identity."


    # Top-k sample
    # Set target face id = 9
    target_face_id = 9
    # Set k = 3
    k = 4
    # Gen k similar features
    similar_features = []
    for i in range(k):
        similar_features.append(gen_similar_feature(embedding_list[target_face_id - 1]))
    # Insert the similar features into the feature hub
    expect_ids = []
    for similar_feature in similar_features:
        similar_face_identity = isf.FaceIdentity(similar_feature, -1)
        ret, id = isf.feature_hub_face_insert(similar_face_identity)
        assert ret, "Failed to insert similar face identity."
        print(f"Inserted similar face id: {id}")
        expect_ids.append(id)
    # Insert the target face id
    expect_ids.append(target_face_id)
    print(f"expect ids: {expect_ids}")
    # Search the top-k similar features
    top_k_search_result = isf.feature_hub_face_search_top_k(embedding_list[target_face_id - 1], 1000)
    assert len(top_k_search_result) == k + 1, "Failed to search for top-k similar features."
    for result in top_k_search_result:
        confidence, _id = result
        print(f"search confidence: {confidence}, id: {_id}")
        assert _id in expect_ids, "Failed to search for top-k similar features."
        print(f"search confidence: {confidence}, id: {_id}")


if __name__ == "__main__":
    case_feature_hub()