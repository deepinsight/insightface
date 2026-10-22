import numpy as np

from insightface.gui.core.clustering import cluster_embeddings_dbscan


def test_dbscan_default_distance_threshold_groups_near_faces():
    embeddings = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.98, 0.08], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.08, 0.98], dtype=np.float32),
    ]

    labels, algorithm = cluster_embeddings_dbscan(embeddings, min_samples=2)

    assert algorithm in {"DBSCAN", "centroid fallback"}
    assert len(labels) == 4
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_dbscan_threshold_can_be_tightened_by_user():
    embeddings = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.95, 0.32], dtype=np.float32),
    ]

    loose_labels, loose_algorithm = cluster_embeddings_dbscan(
        embeddings, distance_threshold=0.28, min_samples=2
    )
    strict_labels, strict_algorithm = cluster_embeddings_dbscan(
        embeddings, distance_threshold=0.001, min_samples=2
    )

    assert loose_algorithm in {"DBSCAN", "centroid fallback"}
    assert strict_algorithm in {"DBSCAN", "centroid fallback"}
    assert loose_labels[0] == loose_labels[1]
    assert strict_labels[0] != strict_labels[1] or strict_labels[0] == -1


def test_dbscan_has_centroid_fallback_when_sklearn_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn.cluster":
            raise ImportError("missing sklearn for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    labels, algorithm = cluster_embeddings_dbscan(
        [np.array([1.0, 0.0], dtype=np.float32), np.array([0.98, 0.08], dtype=np.float32)],
        distance_threshold=0.28,
        min_samples=2,
    )

    assert algorithm == "centroid fallback"
    assert labels == [0, 0]
