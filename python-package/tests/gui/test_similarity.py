import numpy as np

from insightface.gui.core.recognition import cosine_similarity, normalize_embedding, search_gallery


def test_similarity_and_topk():
    emb = normalize_embedding(np.array([3.0, 4.0], dtype=np.float32))
    assert np.allclose(np.linalg.norm(emb), 1.0)
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    gallery = [
        {"person_id": 1, "person_name": "A", "sample_id": 1, "embedding": np.array([1, 0], dtype=np.float32)},
        {"person_id": 2, "person_name": "B", "sample_id": 2, "embedding": np.array([0, 1], dtype=np.float32)},
    ]
    results = search_gallery(np.array([1, 0], dtype=np.float32), gallery, top_k=2, threshold=0.5)
    assert results[0].person_name == "A"
    assert results[0].similarity > results[1].similarity
    assert "quality_score" not in results[0].to_json_dict()
