"""Embedding normalization, comparison, and gallery search."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .constants import DEFAULT_THRESHOLD
from .models import SearchResult


def normalize_embedding(embedding: np.ndarray | Iterable[float] | None) -> Optional[np.ndarray]:
    if embedding is None:
        return None
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def cosine_similarity(a: np.ndarray | Iterable[float] | None, b: np.ndarray | Iterable[float] | None) -> float:
    na = normalize_embedding(a)
    nb = normalize_embedding(b)
    if na is None or nb is None or na.shape != nb.shape:
        return 0.0
    return float(np.dot(na, nb))


def compare_embeddings(a: np.ndarray, b: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    similarity = cosine_similarity(a, b)
    if similarity >= threshold:
        decision = "Same Person"
    elif similarity >= threshold - 0.05:
        decision = "Uncertain"
    else:
        decision = "Different Person"
    return {"similarity": similarity, "threshold": threshold, "decision": decision}


def search_gallery(
    query_embedding: np.ndarray,
    gallery_embeddings: Iterable[Dict[str, Any]],
    top_k: int = 5,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[SearchResult]:
    query = normalize_embedding(query_embedding)
    if query is None:
        return []
    best_by_person: Dict[Any, SearchResult] = {}
    for item in gallery_embeddings:
        emb = item.get("embedding")
        if emb is None:
            continue
        emb_norm = normalize_embedding(emb)
        if emb_norm is None or emb_norm.shape != query.shape:
            continue
        similarity = float(np.dot(query, emb_norm))
        person_id = item.get("person_id")
        status = "matched" if similarity >= threshold else "below_threshold"
        result = SearchResult(
            person_id=person_id,
            person_name=item.get("person_name") or "Unknown",
            sample_id=item.get("sample_id"),
            similarity=similarity,
            status=status,
            crop_path=item.get("crop_path"),
        )
        existing = best_by_person.get(person_id)
        if existing is None or result.similarity > existing.similarity:
            best_by_person[person_id] = result
    results = sorted(best_by_person.values(), key=lambda row: row.similarity, reverse=True)
    return results[: max(1, int(top_k))]


def aggregate_person_embeddings(face_samples: Iterable[Dict[str, Any]], method: str = "mean") -> Dict[int, np.ndarray]:
    grouped: Dict[int, List[np.ndarray]] = defaultdict(list)
    for sample in face_samples:
        person_id = sample.get("person_id")
        emb = normalize_embedding(sample.get("embedding"))
        if person_id is None or emb is None:
            continue
        grouped[int(person_id)].append(emb)
    aggregated: Dict[int, np.ndarray] = {}
    for person_id, embeddings in grouped.items():
        stack = np.vstack(embeddings)
        if method == "max":
            aggregated[person_id] = stack[0]
        else:
            aggregated[person_id] = normalize_embedding(np.mean(stack, axis=0))  # type: ignore[assignment]
    return aggregated


def identify_face(
    query_embedding: np.ndarray,
    people_gallery: Iterable[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 5,
) -> List[SearchResult]:
    results = search_gallery(query_embedding, people_gallery, top_k=top_k, threshold=threshold)
    if not results:
        return [
            SearchResult(
                person_id=None,
                person_name="Unknown",
                sample_id=None,
                similarity=0.0,
                status="unknown",
            )
        ]
    for result in results:
        if result.similarity < threshold:
            result.status = "unknown"
            if result.person_name == "":
                result.person_name = "Unknown"
    return results
