import os
import inspireface as isf
import numpy as np
import os

def case_feature_hub():
    # Gen a random feature
    gen = np.random.rand(512).astype(np.float32)
    # Set db path
    db_path = "test.db"
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
    print('T1, face count:', isf.feature_hub_get_face_count())
    for i in range(10):
        v = np.random.rand(512).astype(np.float32)
        feature = isf.FaceIdentity(v, i)
        ret, _ = isf.feature_hub_face_insert(feature)
        assert ret, "Failed to insert face feature data into FeatureHub."
    feature = isf.FaceIdentity(gen, -1)
    isf.feature_hub_face_insert(feature)
    result = isf.feature_hub_face_search(gen)
    print(f"result: {result}")
    assert os.path.exists(db_path), "FeatureHub database file not found."
    ids = isf.feature_hub_get_face_id_list()
    print(f"ids: {ids}")



if __name__ == "__main__":
    case_feature_hub()
