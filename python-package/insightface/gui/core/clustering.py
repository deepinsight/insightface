"""Face embedding clustering helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .recognition import cosine_similarity, normalize_embedding


def cluster_embeddings(
    embeddings: Iterable[np.ndarray],
    threshold: float = 0.72,
    min_samples: int = 2,
) -> List[int]:
    labels, _ = cluster_embeddings_dbscan(
        embeddings,
        distance_threshold=max(0.01, 1.0 - float(threshold)),
        min_samples=min_samples,
    )
    return labels


def cluster_embeddings_dbscan(
    embeddings: Iterable[np.ndarray],
    distance_threshold: float = 0.28,
    min_samples: int = 2,
) -> tuple[List[int], str]:
    normalized = [normalize_embedding(embedding) for embedding in embeddings]
    vectors = [embedding for embedding in normalized if embedding is not None]
    if not vectors:
        return [], "none"
    matrix = np.vstack(vectors)
    eps = max(0.01, float(distance_threshold))
    try:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=eps, min_samples=max(1, int(min_samples)), metric="cosine").fit_predict(matrix)
        return [int(label) for label in labels], "DBSCAN"
    except Exception:
        labels: List[int] = []
        centroids: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        next_label = 0
        for vector in matrix:
            best_label = None
            best_score = -1.0
            for label, centroid in centroids.items():
                score = cosine_similarity(vector, centroid)
                if score > best_score:
                    best_label = label
                    best_score = score
            if best_label is not None and (1.0 - best_score) <= eps:
                labels.append(best_label)
                counts[best_label] += 1
                centroids[best_label] = normalize_embedding(
                    centroids[best_label] * (counts[best_label] - 1) + vector
                )
            else:
                labels.append(next_label)
                centroids[next_label] = vector
                counts[next_label] = 1
                next_label += 1
        return labels, "centroid fallback"
