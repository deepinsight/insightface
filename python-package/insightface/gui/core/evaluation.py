"""No-code enterprise evaluation routines."""

from __future__ import annotations

import csv
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .constants import DEFAULT_LICENSE_STATUS, DEFAULT_THRESHOLD, IMAGE_EXTENSIONS
from .face_engine import FaceEngine
from .models import EvaluationResult
from .recognition import cosine_similarity
from .utils import list_images, read_image, utc_now_iso

MULTI_FACE_REQUIRE_ONE = "require_exactly_one"
MULTI_FACE_USE_LARGEST = "use_largest_face"
MULTI_FACE_USE_CENTERED_LARGEST = "use_centered_largest_face"
MULTI_FACE_SKIP = "mark_as_skip"
MULTI_FACE_POLICIES = {
    MULTI_FACE_REQUIRE_ONE,
    MULTI_FACE_USE_LARGEST,
    MULTI_FACE_USE_CENTERED_LARGEST,
    MULTI_FACE_SKIP,
}


@dataclass
class DatasetValidationResult:
    ok: bool
    mode: str
    auto_split: bool
    multi_face_policy: str
    root: str
    summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)


def hardware_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def _metrics_at_threshold(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    positives = [row for row in rows if row.get("label") == 1 and row.get("similarity") is not None]
    negatives = [row for row in rows if row.get("label") == 0 and row.get("similarity") is not None]
    tp = sum(1 for row in positives if row["similarity"] >= threshold)
    fn = sum(1 for row in positives if row["similarity"] < threshold)
    fp = sum(1 for row in negatives if row["similarity"] >= threshold)
    tn = sum(1 for row in negatives if row["similarity"] < threshold)
    total = tp + tn + fp + fn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "FAR": fp / max(1, len(negatives)),
        "FRR": fn / max(1, len(positives)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def recommend_threshold(rows: List[Dict[str, Any]]) -> Optional[float]:
    similarities = sorted({float(row["similarity"]) for row in rows if row.get("similarity") is not None})
    if not similarities:
        return None
    best_threshold = similarities[0]
    best_accuracy = -1.0
    for threshold in similarities:
        metrics = _metrics_at_threshold(rows, threshold)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_threshold = threshold
    return float(best_threshold)


def tar_at_far(rows: List[Dict[str, Any]], target_far: float) -> Any:
    positives = [row for row in rows if row.get("label") == 1 and row.get("similarity") is not None]
    negatives = [row for row in rows if row.get("label") == 0 and row.get("similarity") is not None]
    if len(negatives) < max(20, int(1 / max(target_far, 1e-9))):
        return "insufficient data"
    thresholds = sorted({float(row["similarity"]) for row in rows if row.get("similarity") is not None}, reverse=True)
    best_tar = 0.0
    for threshold in thresholds:
        far = sum(1 for row in negatives if row["similarity"] >= threshold) / max(1, len(negatives))
        if far <= target_far:
            tar = sum(1 for row in positives if row["similarity"] >= threshold) / max(1, len(positives))
            best_tar = max(best_tar, tar)
    return best_tar


def _identity_name(folder: Path) -> str:
    return folder.name


def _identity_dirs(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Identity folder root not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir())


def _auto_split_identity_images(identity_dir: Path) -> tuple[Path | None, List[Path]]:
    images = sorted(list_images(identity_dir, recursive=True))
    if not images:
        return None, []
    gallery_candidates = [path for path in images if "gallery" in path.stem.lower()]
    gallery = sorted(gallery_candidates or images)[0]
    probes = [path for path in images if path != gallery]
    return gallery, probes


def _normalize_multi_face_policy(policy: str) -> str:
    policy = (policy or MULTI_FACE_REQUIRE_ONE).strip()
    return policy if policy in MULTI_FACE_POLICIES else MULTI_FACE_REQUIRE_ONE


def _face_bbox(face) -> np.ndarray:
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        bbox = face.bbox if hasattr(face, "bbox") else [0, 0, 0, 0]
    return np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]


def _face_area(face) -> float:
    try:
        bbox = _face_bbox(face)
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    except Exception:
        return 0.0


def _largest_centered_face(faces, image_shape):
    if len(faces) == 1:
        return faces[0]
    try:
        det = np.vstack([_face_bbox(face) for face in faces])
        img_size = np.asarray(image_shape, dtype=np.float32)[0:2]
        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        img_center = img_size / 2.0
        offsets = np.vstack(
            [
                (det[:, 0] + det[:, 2]) / 2.0 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2.0 - img_center[0],
            ]
        )
        offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)
        return faces[int(np.argmax(bounding_box_size - offset_dist_squared * 2.0))]
    except Exception:
        return max(faces, key=_face_area)


def _select_face_from_faces(faces, image_shape, path: Path, policy: str, stage: str):
    del path, stage
    face_count = len(faces)
    if face_count == 0:
        raise ValueError("no face detected")
    if face_count > 1 and policy == MULTI_FACE_REQUIRE_ONE:
        raise ValueError(f"multiple faces detected ({face_count}); expected exactly one face")
    if face_count > 1 and policy == MULTI_FACE_SKIP:
        raise ValueError(f"skipped multi-face image ({face_count} faces)")
    if face_count > 1 and policy == MULTI_FACE_USE_CENTERED_LARGEST:
        return _largest_centered_face(faces, image_shape)
    if face_count > 1:
        return max(faces, key=_face_area)
    return faces[0]


def select_face_by_policy(faces, image_shape, policy: str = MULTI_FACE_REQUIRE_ONE, path: str | Path = "", stage: str = ""):
    return _select_face_from_faces(
        faces,
        image_shape,
        Path(path) if path else Path("image"),
        _normalize_multi_face_policy(policy),
        stage,
    )


def multi_face_policy_help(policy: str) -> str:
    policy = _normalize_multi_face_policy(policy)
    if policy == MULTI_FACE_REQUIRE_ONE:
        return "Multi-face policy: require exactly one face. Images with multiple detected faces fail with a clear error."
    if policy == MULTI_FACE_USE_LARGEST:
        return "Multi-face policy: use largest face. If an image has multiple faces, the largest detected face is used."
    if policy == MULTI_FACE_USE_CENTERED_LARGEST:
        return (
            "Multi-face policy: use largest centered face. If an image has multiple faces, the face with the best "
            "area-minus-center-distance score is used."
        )
    return (
        "Multi-face policy: mark as skip. Gallery images with multiple detected faces are skipped; a multi-face query "
        "stops the current run."
    )


def _detect_faces_for_image(path: Path, engine: FaceEngine):
    img = read_image(path)
    if img is None:
        raise ValueError("image read failure")
    return img, engine.detect_faces(img, source_path=str(path))


def _embedding_for_image(
    path: Path,
    engine: FaceEngine,
    cache: Dict[str, np.ndarray],
    errors: List[Dict[str, Any]],
    stage: str,
    multi_face_policy: str = MULTI_FACE_REQUIRE_ONE,
) -> Optional[np.ndarray]:
    key = str(path)
    if key in cache:
        return cache[key]
    policy = _normalize_multi_face_policy(multi_face_policy)
    try:
        image, faces = _detect_faces_for_image(path, engine)
        face = _select_face_from_faces(faces, image.shape, path, policy, stage)
        if face is None or face.normed_embedding is None:
            raise ValueError("no face or embedding")
        embedding = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
        cache[key] = embedding
        return embedding
    except Exception as exc:
        errors.append({"path": key, "error": str(exc), "stage": stage})
        return None


def _collect_verification_specs(root: Path, auto_split: bool) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    identities = _identity_dirs(root)
    specs: List[Dict[str, Any]] = []
    structure_errors: List[Dict[str, Any]] = []
    if auto_split:
        for identity_dir in identities:
            identity = _identity_name(identity_dir)
            gallery, probes = _auto_split_identity_images(identity_dir)
            if gallery is None:
                structure_errors.append({"identity": identity, "error": "identity folder has no images"})
                continue
            specs.append({"identity": identity, "path": gallery, "role": "gallery", "selected_gallery": True})
            specs.extend({"identity": identity, "path": path, "role": "probe"} for path in probes)
            if not probes:
                structure_errors.append(
                    {"identity": identity, "error": "Auto Split requires at least one probe image after the gallery image"}
                )
    else:
        for identity_dir in identities:
            identity = _identity_name(identity_dir)
            images = sorted(list_images(identity_dir, recursive=True))
            if not images:
                structure_errors.append({"identity": identity, "error": "identity folder has no images"})
            specs.extend({"identity": identity, "path": path, "role": "probe"} for path in images)
    return specs, structure_errors


def _collect_identification_specs(root: Path, auto_split: bool) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    specs: List[Dict[str, Any]] = []
    structure_errors: List[Dict[str, Any]] = []
    if auto_split:
        identities_root = _identity_root_for_auto_split(root)
        identities = _identity_dirs(identities_root)
        for identity_dir in identities:
            identity = _identity_name(identity_dir)
            gallery, probes = _auto_split_identity_images(identity_dir)
            if gallery is None:
                structure_errors.append({"identity": identity, "error": "identity folder has no images"})
                continue
            specs.append({"identity": identity, "path": gallery, "role": "gallery", "selected_gallery": True})
            specs.extend({"identity": identity, "path": path, "role": "probe"} for path in probes)
            if not probes:
                structure_errors.append(
                    {"identity": identity, "error": "Auto Split requires at least one probe image after the gallery image"}
                )
    else:
        gallery_root = root / "gallery"
        probe_root = root / "probe"
        unknown_root = root / "unknown"
        if not gallery_root.is_dir():
            structure_errors.append({"path": str(gallery_root), "error": "1:N without Auto Split requires gallery/"})
        if not probe_root.is_dir():
            structure_errors.append({"path": str(probe_root), "error": "1:N without Auto Split requires probe/"})
        if gallery_root.is_dir():
            for identity_dir in _identity_dirs(gallery_root):
                identity = _identity_name(identity_dir)
                images = sorted(list_images(identity_dir, recursive=True))
                if not images:
                    structure_errors.append({"identity": identity, "role": "gallery", "error": "gallery identity has no images"})
                specs.extend({"identity": identity, "path": path, "role": "gallery"} for path in images)
        if probe_root.is_dir():
            for identity_dir in _identity_dirs(probe_root):
                identity = _identity_name(identity_dir)
                images = sorted(list_images(identity_dir, recursive=True))
                if not images:
                    structure_errors.append({"identity": identity, "role": "probe", "error": "probe identity has no images"})
                specs.extend({"identity": identity, "path": path, "role": "probe"} for path in images)
        if unknown_root.is_dir():
            specs.extend({"identity": "Unknown", "path": path, "role": "unknown"} for path in sorted(list_images(unknown_root, recursive=True)))
    return specs, structure_errors


def _validate_image_spec(
    spec: Dict[str, Any],
    engine: FaceEngine,
    policy: str,
    mode_key: str,
) -> Dict[str, Any]:
    path = Path(spec["path"])
    row = {
        "identity": spec.get("identity", ""),
        "path": str(path),
        "role": spec.get("role", ""),
        "valid": False,
        "skipped": False,
        "face_count": 0,
    }
    try:
        image, faces = _detect_faces_for_image(path, engine)
        del image
        row["face_count"] = len(faces)
        if not faces:
            row["error"] = "no face detected"
            return row
        if len(faces) > 1:
            row["multi_face"] = True
            if policy == MULTI_FACE_REQUIRE_ONE:
                row["error"] = f"multiple faces detected ({len(faces)}); expected exactly one face"
                return row
            if policy == MULTI_FACE_SKIP:
                row["skipped"] = True
                row["warning"] = f"multi-face image skipped ({len(faces)} faces)"
                if spec.get("role") == "gallery" or spec.get("selected_gallery"):
                    row["error"] = (
                        "multi-face gallery image would be skipped, leaving this gallery sample unavailable"
                    )
                return row
            if policy == MULTI_FACE_USE_CENTERED_LARGEST:
                row["warning"] = f"multiple faces detected ({len(faces)}); largest centered face will be used"
            else:
                row["warning"] = f"multiple faces detected ({len(faces)}); largest face will be used"
        row["valid"] = True
        return row
    except Exception as exc:
        row["error"] = str(exc)
        return row


def _quick_validation_specs(specs: List[Dict[str, Any]], policy: str) -> List[Dict[str, Any]]:
    if policy == MULTI_FACE_REQUIRE_ONE:
        return specs
    gallery_specs = [spec for spec in specs if spec.get("role") == "gallery"]
    if gallery_specs:
        return gallery_specs
    representatives: List[Dict[str, Any]] = []
    seen = set()
    for spec in specs:
        identity = spec.get("identity", "")
        if identity in seen:
            continue
        seen.add(identity)
        representatives.append(spec)
    return representatives


def _validation_scope_label(policy: str, specs_to_check: List[Dict[str, Any]], specs: List[Dict[str, Any]]) -> str:
    if policy == MULTI_FACE_REQUIRE_ONE:
        return "all images"
    if len(specs_to_check) == len(specs):
        return "all gallery images"
    return "gallery or representative images only"


def validate_enterprise_dataset(
    dataset_root: str | Path,
    mode: str,
    auto_split: bool,
    engine: FaceEngine,
    multi_face_policy: str = MULTI_FACE_REQUIRE_ONE,
    progress_callback=None,
    cancel_callback=None,
) -> DatasetValidationResult:
    root = Path(dataset_root).expanduser()
    policy = _normalize_multi_face_policy(multi_face_policy)
    result = DatasetValidationResult(
        ok=False,
        mode=mode,
        auto_split=bool(auto_split),
        multi_face_policy=policy,
        root=str(root),
    )
    if not root.is_dir():
        result.errors.append({"path": str(root), "error": "dataset root not found"})
        return result

    mode_key = "1n" if mode.strip().startswith("1:N") else "1to1"
    try:
        specs, structure_errors = (
            _collect_identification_specs(root, auto_split)
            if mode_key == "1n"
            else _collect_verification_specs(root, auto_split)
        )
    except Exception as exc:
        result.errors.append({"path": str(root), "error": str(exc)})
        return result
    result.errors.extend(structure_errors)
    if not specs:
        result.errors.append({"path": str(root), "error": "no image files found for this evaluation mode"})
    if result.errors:
        result.summary.update(
            {
                "total_images": len(specs),
                "images_checked": 0,
                "valid_images": 0,
                "skipped_images": 0,
                "multi_face_images": 0,
                "validation_scope": "structure only",
                "errors": len(result.errors),
                "warnings": len(result.warnings),
            }
        )
        return result

    rows: List[Dict[str, Any]] = []
    specs_to_check = _quick_validation_specs(specs, policy)
    for index, spec in enumerate(specs_to_check):
        if cancel_callback and cancel_callback():
            result.errors.append({"error": "validation cancelled"})
            break
        row = _validate_image_spec(spec, engine, policy, mode_key)
        rows.append(row)
        if row.get("error"):
            result.errors.append({key: row[key] for key in ("identity", "path", "role", "face_count", "error") if key in row})
        elif row.get("warning"):
            result.warnings.append({key: row[key] for key in ("identity", "path", "role", "face_count", "warning") if key in row})
        if progress_callback:
            progress_callback(index + 1, max(1, len(specs_to_check)), f"Validated image {index + 1}/{len(specs_to_check)}")
        if row.get("error"):
            break

    if result.errors:
        result.summary.update(
            {
                "total_images": len(specs),
                "images_checked": len(rows),
                "valid_images": sum(1 for row in rows if row.get("valid")),
                "skipped_images": sum(1 for row in rows if row.get("skipped")),
                "multi_face_images": sum(1 for row in rows if row.get("multi_face")),
                "validation_scope": _validation_scope_label(policy, specs_to_check, specs),
                "errors": len(result.errors),
                "warnings": len(result.warnings),
            }
        )
        return result

    if policy == MULTI_FACE_REQUIRE_ONE:
        valid_specs = [
            dict(spec, **{"path": str(spec["path"])})
            for spec, row in zip(specs_to_check, rows)
            if row.get("valid")
        ]
    else:
        valid_specs = [dict(spec, **{"path": str(spec["path"])}) for spec in specs]
    valid_galleries = [spec for spec in valid_specs if spec.get("role") == "gallery"]
    valid_probes = [spec for spec in valid_specs if spec.get("role") == "probe"]

    if mode_key == "1to1":
        if auto_split:
            gallery_by_identity = {spec["identity"] for spec in valid_galleries}
            probe_by_identity = {spec["identity"] for spec in valid_probes}
            expected_identities = {spec["identity"] for spec in specs}
            for identity in sorted(expected_identities - gallery_by_identity):
                result.errors.append({"identity": identity, "role": "gallery", "error": "identity has no valid gallery image"})
            for identity in sorted(expected_identities - probe_by_identity):
                result.errors.append({"identity": identity, "role": "probe", "error": "identity has no valid probe image"})
            positive_pairs = sum(1 for probe in valid_probes for gallery in valid_galleries if probe["identity"] == gallery["identity"])
            negative_pairs = sum(1 for probe in valid_probes for gallery in valid_galleries if probe["identity"] != gallery["identity"])
            pair_count = positive_pairs + negative_pairs
        else:
            pair_count = positive_pairs = negative_pairs = 0
            for left_index, left in enumerate(valid_probes):
                for right in valid_probes[left_index + 1 :]:
                    pair_count += 1
                    if left["identity"] == right["identity"]:
                        positive_pairs += 1
                    else:
                        negative_pairs += 1
        if pair_count <= 0:
            result.errors.append({"error": "no valid 1:1 comparison pairs can be generated"})
        if positive_pairs <= 0:
            result.errors.append({"error": "no positive 1:1 pairs can be generated"})
        if negative_pairs <= 0:
            result.errors.append({"error": "no negative 1:1 pairs can be generated"})
        result.summary.update(
            {
                "valid_gallery_images": len(valid_galleries),
                "valid_probe_images": len(valid_probes),
                "comparison_pairs": pair_count,
                "positive_pairs": positive_pairs,
                "negative_pairs": negative_pairs,
            }
        )
    else:
        gallery_identities = {spec["identity"] for spec in valid_galleries}
        probe_identities = {spec["identity"] for spec in valid_probes}
        if not valid_galleries:
            result.errors.append({"role": "gallery", "error": "1:N evaluation requires at least one valid gallery image"})
        if not valid_probes:
            result.errors.append({"role": "probe", "error": "1:N evaluation requires at least one valid known probe image"})
        missing_gallery = sorted(probe_identities - gallery_identities)
        for identity in missing_gallery:
            result.errors.append({"identity": identity, "role": "gallery", "error": "probe identity has no valid gallery image"})
        result.summary.update(
            {
                "valid_gallery_images": len(valid_galleries),
                "gallery_identities": len(gallery_identities),
                "valid_known_probe_images": len(valid_probes),
                "probe_identities": len(probe_identities),
                "unknown_images": sum(1 for spec in valid_specs if spec.get("role") == "unknown"),
            }
        )

    result.summary.update(
        {
            "total_images": len(specs),
            "images_checked": len(rows),
            "valid_images": len(valid_specs),
            "skipped_images": sum(1 for row in rows if row.get("skipped")),
            "multi_face_images": sum(1 for row in rows if row.get("multi_face")),
            "validation_scope": _validation_scope_label(policy, specs_to_check, specs),
            "errors": len(result.errors),
            "warnings": len(result.warnings),
        }
    )
    result.ok = not result.errors
    return result


def _metrics_at_best_threshold(rows: List[Dict[str, Any]]) -> tuple[float, Dict[str, Any]]:
    completed = [row for row in rows if row.get("similarity") is not None]
    if not completed:
        return 0.0, {"best_accuracy": 0.0, "best_cosine_threshold": 0.0}
    positives = np.sort(
        np.asarray([float(row["similarity"]) for row in completed if row.get("label") == 1], dtype=np.float64)
    )
    negatives = np.sort(
        np.asarray([float(row["similarity"]) for row in completed if row.get("label") == 0], dtype=np.float64)
    )
    candidates = np.unique(np.asarray([float(row["similarity"]) for row in completed], dtype=np.float64))
    positive_total = int(positives.size)
    negative_total = int(negatives.size)
    tp = positive_total - np.searchsorted(positives, candidates, side="left")
    fn = positive_total - tp
    tn = np.searchsorted(negatives, candidates, side="left")
    fp = negative_total - tn
    total = positive_total + negative_total
    accuracy = (tp + tn) / max(1, total)
    best_index = int(np.argmax(accuracy))
    best_threshold = float(candidates[best_index])
    best_accuracy = float(accuracy[best_index])
    best_metrics = {
        "accuracy": best_accuracy,
        "FAR": float(fp[best_index] / max(1, negative_total)),
        "FRR": float(fn[best_index] / max(1, positive_total)),
        "TP": int(tp[best_index]),
        "TN": int(tn[best_index]),
        "FP": int(fp[best_index]),
        "FN": int(fn[best_index]),
    }
    best_metrics["best_accuracy"] = best_accuracy
    best_metrics["best_cosine_threshold"] = best_threshold
    return best_threshold, best_metrics


def _tar_far_from_scores(
    positive_scores: List[float],
    positive_total: int,
    negative_scores: List[float],
    far_targets: List[float],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "positive_trials_for_tar": int(positive_total),
        "negative_trials_for_far": len(negative_scores),
    }
    far_targets_with_budget = [
        far for far in far_targets if int(math.floor(far * len(negative_scores))) > 0
    ]
    if positive_total <= 0 or not negative_scores:
        for far in far_targets_with_budget:
            label = _far_label(far)
            metrics[f"TAR@FAR={label}"] = "insufficient data"
            metrics[f"Threshold@FAR={label}"] = "insufficient data"
        return metrics

    positives = np.sort(np.asarray(positive_scores, dtype=np.float64))
    negatives = np.sort(np.asarray(negative_scores, dtype=np.float64))
    all_scores = np.asarray(positive_scores + negative_scores, dtype=np.float64)
    max_negative = float(negatives[-1])
    candidates = np.unique(np.concatenate(([np.nextafter(max_negative, np.inf)], all_scores)))
    positive_accepts = positive_total - np.searchsorted(positives, candidates, side="left")
    false_accepts_by_threshold = len(negative_scores) - np.searchsorted(negatives, candidates, side="left")
    for far in far_targets_with_budget:
        label = _far_label(far)
        allowed_fp = int(math.floor(far * len(negative_scores)))
        valid = false_accepts_by_threshold <= allowed_fp
        if not np.any(valid):
            metrics[f"TAR@FAR={label}"] = 0.0
            metrics[f"Threshold@FAR={label}"] = float(np.nextafter(max_negative, np.inf))
            metrics[f"Achieved FAR@{label}"] = 0.0
            metrics[f"False accept budget@{label}"] = allowed_fp
            continue
        valid_positive_accepts = positive_accepts[valid]
        best_accepts = int(np.max(valid_positive_accepts))
        best_mask = valid & (positive_accepts == best_accepts)
        best_threshold = float(np.min(candidates[best_mask]))
        best_false_accepts = int(
            false_accepts_by_threshold[np.where(candidates == best_threshold)[0][0]]
        )
        best_tar = best_accepts / max(1, positive_total)
        best_far = best_false_accepts / max(1, len(negative_scores))
        metrics[f"TAR@FAR={label}"] = max(0.0, best_tar)
        metrics[f"Threshold@FAR={label}"] = best_threshold
        metrics[f"Achieved FAR@{label}"] = best_far
        metrics[f"False accept budget@{label}"] = allowed_fp
    return metrics


def _far_label(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def run_identity_verification_evaluation(
    identity_root: str | Path,
    engine: FaceEngine,
    auto_split: bool = True,
    multi_face_policy: str = MULTI_FACE_REQUIRE_ONE,
    license_status: str = DEFAULT_LICENSE_STATUS,
    progress_callback=None,
    cancel_callback=None,
) -> EvaluationResult:
    root = Path(identity_root).expanduser()
    identities = _identity_dirs(root)
    errors: List[Dict[str, Any]] = []
    cache: Dict[str, np.ndarray] = {}
    pair_specs: List[Dict[str, Any]] = []

    if auto_split:
        galleries: List[Dict[str, Any]] = []
        probes: List[Dict[str, Any]] = []
        for identity_dir in identities:
            gallery, identity_probes = _auto_split_identity_images(identity_dir)
            if gallery is None:
                continue
            identity = _identity_name(identity_dir)
            galleries.append({"identity": identity, "path": gallery})
            probes.extend({"identity": identity, "path": path} for path in identity_probes)
        for probe in probes:
            for gallery in galleries:
                pair_specs.append(
                    {
                        "image1_path": str(probe["path"]),
                        "image2_path": str(gallery["path"]),
                        "probe_identity": probe["identity"],
                        "gallery_identity": gallery["identity"],
                        "label": 1 if probe["identity"] == gallery["identity"] else 0,
                    }
                )
    else:
        images: List[Dict[str, Any]] = []
        for identity_dir in identities:
            identity = _identity_name(identity_dir)
            images.extend({"identity": identity, "path": path} for path in sorted(list_images(identity_dir, recursive=True)))
        for left_index, left in enumerate(images):
            for right in images[left_index + 1 :]:
                pair_specs.append(
                    {
                        "image1_path": str(left["path"]),
                        "image2_path": str(right["path"]),
                        "probe_identity": left["identity"],
                        "gallery_identity": right["identity"],
                        "label": 1 if left["identity"] == right["identity"] else 0,
                    }
                )

    if not pair_specs:
        raise ValueError("No verification pairs could be generated from the selected identity folders.")

    start_all = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for index, spec in enumerate(pair_specs):
        if cancel_callback and cancel_callback():
            break
        start = time.perf_counter()
        row = dict(spec)
        emb1 = _embedding_for_image(
            Path(spec["image1_path"]),
            engine,
            cache,
            errors,
            "verification",
            multi_face_policy=multi_face_policy,
        )
        emb2 = _embedding_for_image(
            Path(spec["image2_path"]),
            engine,
            cache,
            errors,
            "verification",
            multi_face_policy=multi_face_policy,
        )
        if emb1 is None or emb2 is None:
            row.update({"similarity": None, "error": "embedding unavailable"})
        else:
            row.update(
                {
                    "similarity": cosine_similarity(emb1, emb2),
                    "latency_ms": (time.perf_counter() - start) * 1000.0,
                }
            )
        rows.append(row)
        if progress_callback:
            progress_callback(index + 1, len(pair_specs), f"Processed 1:1 pair {index + 1}/{len(pair_specs)}")

    completed = [row for row in rows if row.get("similarity") is not None]
    best_threshold, best_metrics = _metrics_at_best_threshold(completed)
    positive_scores = [float(row["similarity"]) for row in completed if row.get("label") == 1]
    negative_scores = [float(row["similarity"]) for row in completed if row.get("label") == 0]
    metrics = {
        "total_pairs": len(pair_specs),
        "completed_pairs": len(completed),
        "positive_pairs": sum(1 for row in pair_specs if row["label"] == 1),
        "negative_pairs": sum(1 for row in pair_specs if row["label"] == 0),
        "failed_pairs": len(pair_specs) - len(completed),
        **best_metrics,
        **_tar_far_from_scores(positive_scores, len(positive_scores), negative_scores, [1e-6, 1e-5, 1e-4, 1e-3]),
        "average_latency_ms_per_pair": (
            float(np.mean([row["latency_ms"] for row in completed if row.get("latency_ms") is not None]))
            if completed
            else 0.0
        ),
    }
    return EvaluationResult(
        scenario="Enterprise 1:1 Verification / Identity Folders",
        model_name=engine.model_name,
        provider=", ".join(engine.requested_providers),
        threshold=best_threshold,
        dataset_summary={
            "identity_root": str(root),
            "auto_split": bool(auto_split),
            "multi_face_policy": _normalize_multi_face_policy(multi_face_policy),
            "identities": len(identities),
            "pairing_rule": (
                "probe vs selected gallery for every identity"
                if auto_split
                else "all image pairs across identity folders"
            ),
        },
        metrics=metrics,
        errors=errors,
        latency={"total_elapsed_ms": (time.perf_counter() - start_all) * 1000.0, "hardware": hardware_info()},
        license_status=license_status,
        created_at=utc_now_iso(),
        raw_results=rows,
        threshold_recommendation=best_threshold,
    )


def _identity_root_for_auto_split(root: Path) -> Path:
    identities_root = root / "identities"
    return identities_root if identities_root.is_dir() else root


def _collect_gallery_probe_auto_split(root: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Path]]:
    gallery_items: List[Dict[str, Any]] = []
    probe_items: List[Dict[str, Any]] = []
    identities_root = _identity_root_for_auto_split(root)
    for identity_dir in _identity_dirs(identities_root):
        gallery, probes = _auto_split_identity_images(identity_dir)
        if gallery is None:
            continue
        identity = _identity_name(identity_dir)
        gallery_items.append({"identity": identity, "path": gallery})
        probe_items.extend({"identity": identity, "path": path} for path in probes)
    return gallery_items, probe_items, []


def _collect_gallery_probe_structured(root: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Path]]:
    gallery_root = root / "gallery"
    probe_root = root / "probe"
    unknown_root = root / "unknown"
    if not gallery_root.is_dir() or not probe_root.is_dir():
        raise ValueError("1:N without Auto Split requires gallery/ and probe/ folders.")
    gallery_items: List[Dict[str, Any]] = []
    probe_items: List[Dict[str, Any]] = []
    for identity_dir in _identity_dirs(gallery_root):
        identity = _identity_name(identity_dir)
        gallery_items.extend({"identity": identity, "path": path} for path in sorted(list_images(identity_dir, recursive=True)))
    for identity_dir in _identity_dirs(probe_root):
        identity = _identity_name(identity_dir)
        probe_items.extend({"identity": identity, "path": path} for path in sorted(list_images(identity_dir, recursive=True)))
    unknown_items = sorted(list_images(unknown_root, recursive=True)) if unknown_root.is_dir() else []
    return gallery_items, probe_items, unknown_items


def _rank_identities(query_embedding: np.ndarray, gallery: List[Dict[str, Any]]) -> List[tuple[str, float]]:
    best_by_identity: Dict[str, float] = {}
    for sample in gallery:
        score = cosine_similarity(query_embedding, sample["embedding"])
        identity = str(sample["identity"])
        if identity not in best_by_identity or score > best_by_identity[identity]:
            best_by_identity[identity] = score
    return sorted(best_by_identity.items(), key=lambda item: item[1], reverse=True)


def run_identity_identification_evaluation(
    dataset_root: str | Path,
    engine: FaceEngine,
    auto_split: bool = False,
    multi_face_policy: str = MULTI_FACE_REQUIRE_ONE,
    license_status: str = DEFAULT_LICENSE_STATUS,
    progress_callback=None,
    cancel_callback=None,
) -> EvaluationResult:
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise ValueError(f"Dataset root not found: {root}")
    gallery_items, probe_items, unknown_items = (
        _collect_gallery_probe_auto_split(root) if auto_split else _collect_gallery_probe_structured(root)
    )
    if not gallery_items:
        raise ValueError("1:N evaluation requires at least one gallery image.")
    if not probe_items:
        raise ValueError("1:N evaluation requires at least one known probe image.")

    errors: List[Dict[str, Any]] = []
    cache: Dict[str, np.ndarray] = {}
    gallery: List[Dict[str, Any]] = []
    start_all = time.perf_counter()
    for index, item in enumerate(gallery_items):
        embedding = _embedding_for_image(
            Path(item["path"]),
            engine,
            cache,
            errors,
            "gallery",
            multi_face_policy=multi_face_policy,
        )
        if embedding is not None:
            gallery.append({"identity": item["identity"], "path": str(item["path"]), "embedding": embedding})
        if progress_callback:
            progress_callback(index + 1, len(gallery_items), f"Indexed gallery {index + 1}/{len(gallery_items)}")
    if not gallery:
        raise ValueError("No gallery embeddings could be extracted.")

    rows: List[Dict[str, Any]] = []
    known_total = len(probe_items)
    for index, item in enumerate(probe_items):
        if cancel_callback and cancel_callback():
            break
        embedding = _embedding_for_image(
            Path(item["path"]),
            engine,
            cache,
            errors,
            "probe",
            multi_face_policy=multi_face_policy,
        )
        if embedding is None:
            rows.append({"probe_path": str(item["path"]), "ground_truth": item["identity"], "error": "embedding unavailable"})
        else:
            ranked = _rank_identities(embedding, gallery)
            top1_identity, top1_score = ranked[0] if ranked else ("Unknown", 0.0)
            top5 = [identity for identity, _score in ranked[:5]]
            best_wrong = max([score for identity, score in ranked if identity != item["identity"]], default=None)
            rows.append(
                {
                    "probe_path": str(item["path"]),
                    "ground_truth": item["identity"],
                    "top1": top1_identity,
                    "top1_similarity": top1_score,
                    "top5": top5,
                    "correct_top1": top1_identity == item["identity"],
                    "best_wrong_similarity": best_wrong,
                    "label": "known",
                }
            )
        if progress_callback:
            progress_callback(index + 1, known_total, f"Processed known probe {index + 1}/{known_total}")

    for index, path in enumerate(unknown_items):
        if cancel_callback and cancel_callback():
            break
        embedding = _embedding_for_image(
            path,
            engine,
            cache,
            errors,
            "unknown",
            multi_face_policy=multi_face_policy,
        )
        if embedding is None:
            rows.append({"probe_path": str(path), "ground_truth": "Unknown", "error": "embedding unavailable"})
        else:
            ranked = _rank_identities(embedding, gallery)
            top1_identity, top1_score = ranked[0] if ranked else ("Unknown", 0.0)
            rows.append(
                {
                    "probe_path": str(path),
                    "ground_truth": "Unknown",
                    "top1": top1_identity,
                    "top1_similarity": top1_score,
                    "top5": [identity for identity, _score in ranked[:5]],
                    "correct_top1": False,
                    "label": "unknown",
                }
            )
        if progress_callback:
            progress_callback(index + 1, max(1, len(unknown_items)), f"Processed unknown probe {index + 1}/{len(unknown_items)}")

    known_rows = [row for row in rows if row.get("label") == "known" and row.get("top1_similarity") is not None]
    correct_rows = [row for row in known_rows if row.get("correct_top1")]
    positive_scores = [float(row["top1_similarity"]) for row in correct_rows]
    negative_scores = [
        float(row["top1_similarity"])
        for row in rows
        if row.get("label") == "unknown" and row.get("top1_similarity") is not None
    ]
    negative_scores.extend(
        float(row["best_wrong_similarity"])
        for row in known_rows
        if row.get("best_wrong_similarity") is not None
    )
    metrics = {
        "gallery_identities": len({item["identity"] for item in gallery}),
        "gallery_images": len(gallery_items),
        "gallery_embeddings": len(gallery),
        "known_probe_images": len(probe_items),
        "unknown_probe_images": len(unknown_items),
        "completed_known_probes": len(known_rows),
        "Top1": len(correct_rows) / max(1, len(known_rows)),
        **_tar_far_from_scores(positive_scores, len(known_rows), negative_scores, [1e-5, 1e-4, 1e-3, 1e-2]),
    }
    threshold_recommendation = None
    for key in ("Threshold@FAR=1e-3", "Threshold@FAR=1e-2", "Threshold@FAR=1e-4"):
        value = metrics.get(key)
        if isinstance(value, float):
            threshold_recommendation = value
            break
    return EvaluationResult(
        scenario="Enterprise 1:N Identification / Identity Folders",
        model_name=engine.model_name,
        provider=", ".join(engine.requested_providers),
        threshold=threshold_recommendation or DEFAULT_THRESHOLD,
        dataset_summary={
            "dataset_root": str(root),
            "auto_split": bool(auto_split),
            "multi_face_policy": _normalize_multi_face_policy(multi_face_policy),
            "structured_rule": "gallery/probe/unknown" if not auto_split else "identities auto split",
            "gallery_required": True,
        },
        metrics=metrics,
        errors=errors,
        latency={"total_elapsed_ms": (time.perf_counter() - start_all) * 1000.0, "hardware": hardware_info()},
        license_status=license_status,
        created_at=utc_now_iso(),
        raw_results=rows,
        threshold_recommendation=threshold_recommendation,
    )


def run_kyc_pairs_evaluation(
    pairs_csv: str | Path,
    engine: FaceEngine,
    threshold: float = DEFAULT_THRESHOLD,
    license_status: str = DEFAULT_LICENSE_STATUS,
    progress_callback=None,
    cancel_callback=None,
) -> EvaluationResult:
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    with Path(pairs_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        pairs = list(reader)
    start_all = time.perf_counter()
    for index, row in enumerate(pairs):
        if cancel_callback and cancel_callback():
            break
        image1_path = row.get("image1_path") or row.get("image1") or ""
        image2_path = row.get("image2_path") or row.get("image2") or ""
        label = int(row.get("label", "0"))
        item: Dict[str, Any] = {"image1_path": image1_path, "image2_path": image2_path, "label": label}
        start = time.perf_counter()
        try:
            img1 = read_image(image1_path)
            img2 = read_image(image2_path)
            if img1 is None or img2 is None:
                raise ValueError("image read failure")
            face1 = engine.detect_best_face(img1, source_path=image1_path)
            face2 = engine.detect_best_face(img2, source_path=image2_path)
            if face1 is None or face2 is None:
                raise ValueError("failed detection")
            if face1.normed_embedding is None or face2.normed_embedding is None:
                raise ValueError("embedding unavailable")
            similarity = cosine_similarity(face1.normed_embedding, face2.normed_embedding)
            item.update(
                {
                    "similarity": similarity,
                    "predicted": 1 if similarity >= threshold else 0,
                    "latency_ms": (time.perf_counter() - start) * 1000.0,
                }
            )
        except Exception as exc:
            item.update({"similarity": None, "predicted": None, "error": str(exc)})
            errors.append({"index": index, "error": str(exc), "row": dict(row)})
        rows.append(item)
        if progress_callback:
            progress_callback(index + 1, len(pairs), f"Processed pair {index + 1}/{len(pairs)}")

    completed = [row for row in rows if row.get("similarity") is not None]
    metrics = _metrics_at_threshold(completed, threshold)
    metrics.update(
        {
            "total_pairs": len(pairs),
            "positive_pairs": sum(1 for row in rows if row.get("label") == 1),
            "negative_pairs": sum(1 for row in rows if row.get("label") == 0),
            "failed_detections": len(errors),
            "TAR@FAR=1e-2": tar_at_far(completed, 1e-2),
            "TAR@FAR=1e-3": tar_at_far(completed, 1e-3),
            "TAR@FAR=1e-4": tar_at_far(completed, 1e-4),
            "average_latency_ms_per_pair": (
                float(np.mean([row["latency_ms"] for row in completed])) if completed else 0.0
            ),
        }
    )
    return EvaluationResult(
        scenario="KYC / 1:1 Verification",
        model_name=engine.model_name,
        provider=", ".join(engine.requested_providers),
        threshold=threshold,
        dataset_summary={
            "pairs_csv": str(pairs_csv),
            "total_pairs": len(pairs),
            "completed_pairs": len(completed),
        },
        metrics=metrics,
        errors=errors,
        latency={
            "total_elapsed_ms": (time.perf_counter() - start_all) * 1000.0,
            "hardware": hardware_info(),
        },
        license_status=license_status,
        created_at=utc_now_iso(),
        raw_results=rows,
        threshold_recommendation=recommend_threshold(completed),
    )


def run_identification_evaluation(
    gallery_folder: str | Path,
    probe_folder: str | Path,
    engine: FaceEngine,
    threshold: float = DEFAULT_THRESHOLD,
    ground_truth_csv: Optional[str | Path] = None,
    license_status: str = DEFAULT_LICENSE_STATUS,
    progress_callback=None,
    cancel_callback=None,
) -> EvaluationResult:
    gallery: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    gallery_root = Path(gallery_folder)
    for person_dir in sorted(path for path in gallery_root.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in person_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
            try:
                img = read_image(image_path)
                if img is None:
                    raise ValueError("image read failure")
                face = engine.detect_best_face(img, source_path=str(image_path))
                if face is None or face.normed_embedding is None:
                    raise ValueError("no face or embedding")
                gallery.append(
                    {
                        "person_id": person_dir.name,
                        "person_name": person_dir.name,
                        "sample_id": len(gallery) + 1,
                        "crop_path": "",
                        "embedding": face.normed_embedding,
                    }
                )
            except Exception as exc:
                errors.append({"path": str(image_path), "error": str(exc), "stage": "gallery"})

    ground_truth: Dict[str, str] = {}
    if ground_truth_csv:
        with Path(ground_truth_csv).open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                ground_truth[row.get("image_path", "")] = row.get("person_name", "")

    probes = list_images(probe_folder, recursive=True)
    raw: List[Dict[str, Any]] = []
    start_all = time.perf_counter()
    for index, probe in enumerate(probes):
        if cancel_callback and cancel_callback():
            break
        try:
            img = read_image(probe)
            if img is None:
                raise ValueError("image read failure")
            face = engine.detect_best_face(img, source_path=str(probe))
            if face is None or face.normed_embedding is None:
                raise ValueError("no face or embedding")
            from .recognition import search_gallery

            results = search_gallery(face.normed_embedding, gallery, top_k=5, threshold=threshold)
            truth = ground_truth.get(str(probe), ground_truth.get(probe.name, ""))
            row = {
                "probe_path": str(probe),
                "ground_truth": truth,
                "top1": results[0].person_name if results else "Unknown",
                "top1_similarity": results[0].similarity if results else 0.0,
                "top5": [result.person_name for result in results],
                "accepted": bool(results and results[0].similarity >= threshold),
            }
            raw.append(row)
        except Exception as exc:
            errors.append({"path": str(probe), "error": str(exc), "stage": "probe"})
        if progress_callback:
            progress_callback(index + 1, len(probes), f"Processed probe {index + 1}/{len(probes)}")

    truth_rows = [row for row in raw if row.get("ground_truth")]
    top1 = sum(1 for row in truth_rows if row["top1"] == row["ground_truth"]) / max(1, len(truth_rows))
    top5 = sum(1 for row in truth_rows if row["ground_truth"] in row["top5"]) / max(1, len(truth_rows))
    metrics = {
        "gallery_persons": len({row["person_name"] for row in gallery}),
        "gallery_face_samples": len(gallery),
        "probe_images": len(probes),
        "detected_probe_faces": len(raw),
        "Top-1 accuracy": top1 if truth_rows else "ground truth not provided",
        "Top-5 accuracy": top5 if truth_rows else "ground truth not provided",
        "unknown_rejection_rate": sum(1 for row in raw if not row["accepted"]) / max(1, len(raw)),
        "average_search_latency_ms": 0.0,
    }
    return EvaluationResult(
        scenario="Access Control / 1:N Identification",
        model_name=engine.model_name,
        provider=", ".join(engine.requested_providers),
        threshold=threshold,
        dataset_summary={
            "gallery_folder": str(gallery_folder),
            "probe_folder": str(probe_folder),
            "ground_truth_csv": str(ground_truth_csv or ""),
        },
        metrics=metrics,
        errors=errors,
        latency={"total_elapsed_ms": (time.perf_counter() - start_all) * 1000.0, "hardware": hardware_info()},
        license_status=license_status,
        created_at=utc_now_iso(),
        raw_results=raw,
    )
