from __future__ import annotations

import logging
import sqlite3
import uuid
from typing import Any

import numpy as np

from ..addons import LivenessUnavailable
from ..api.schemas import EmbeddingMode, ReviewMode
from ..config import DetectionProfile, Settings
from ..errors import ApiError, bad_request, conflict, not_found, unprocessable
from ..inference import FaceObservation, InferenceEngine
from ..search import (
    SearchIndexCapacityError,
    SearchIndexError,
    SearchIndexManager,
    SearchMutationCommittedError,
    profile_similarity,
)
from ..storage import (
    CollectionCapacityExceeded,
    CursorCodec,
    FaceCropStore,
    MaxFacesPerPersonExceeded,
    Repository,
    utc_now,
)
from .images import ImageData
from .presentation import bounding_box, clamp01, face_result, quality

LOGGER = logging.getLogger("insightface_server.service")
CENTER_LARGEST_DISTANCE_WEIGHT = 2.0


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


class FaceService:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        cursors: CursorCodec,
        crops: FaceCropStore,
        engine: InferenceEngine,
        search_indexes: SearchIndexManager,
    ):
        self.settings = settings
        self.repository = repository
        self.cursors = cursors
        self.crops = crops
        self.engine = engine
        self.search_indexes = search_indexes

    @staticmethod
    def _search_unavailable(exc: Exception) -> ApiError:
        details: dict[str, Any] = {}
        if isinstance(exc, SearchMutationCommittedError):
            details = {
                "write_committed": True,
                "collection_id": exc.collection_id,
                "revision": exc.revision,
            }
        return ApiError(
            "search_index_unavailable",
            "The Collection search index is unavailable and is being rebuilt.",
            503,
            details,
        )

    @staticmethod
    def _quota_error(exc: Exception) -> ApiError:
        if isinstance(exc, MaxFacesPerPersonExceeded):
            return ApiError(
                "person_face_limit_exceeded",
                "The Person has reached the configured FaceSample limit.",
                409,
                {
                    "max_faces": exc.max_faces,
                    "requested_faces": exc.requested_faces,
                },
            )
        if isinstance(exc, CollectionCapacityExceeded):
            return ApiError(
                "collection_capacity_exceeded",
                "The Collection has reached its configured search capacity.",
                409,
                {
                    "capacity_rows": exc.capacity_rows,
                    "requested_rows": exc.requested_rows,
                },
            )
        return conflict(
            "The Collection has reached its configured search capacity.",
            code="collection_capacity_exceeded",
        )

    def _delete_crop_paths(self, paths: list[str | None]) -> None:
        for path in paths:
            try:
                self.crops.delete(path)
            except OSError:
                LOGGER.warning("unable_to_remove_orphan_face_crop", exc_info=True)

    def raw_collection(self, collection_id: str) -> dict[str, Any]:
        """Return Collection metadata without enforcing the active model contract."""

        item = self.repository.get_collection(collection_id)
        if item is None:
            raise not_found("collection", collection_id)
        return item

    def collection(self, collection_id: str) -> dict[str, Any]:
        item = self.raw_collection(collection_id)
        summary = self.engine.summary
        expected = (
            summary.model_id,
            summary.model_digest,
            summary.embedding_dimension,
            summary.preprocessing_version,
        )
        actual = (
            item["model_id"],
            item["model_digest"],
            item["embedding_dimension"],
            item["preprocessing_version"],
        )
        if expected != actual:
            raise ApiError(
                "collection_model_mismatch",
                "The collection is bound to a different model; rebuild or migrate it.",
                409,
            )
        return item

    @staticmethod
    def collection_detection_profile(collection: dict[str, Any]) -> DetectionProfile:
        return DetectionProfile.from_mapping(collection.get("detection"))

    def detection_profile(self, collection_id: str | None = None) -> DetectionProfile:
        if collection_id is None:
            return self.settings.detection_profile
        return self.collection_detection_profile(self.collection(collection_id))

    def analyze(
        self,
        image: ImageData,
        *,
        embeddings: bool = True,
        detection_profile: DetectionProfile | None = None,
        single_face: bool = False,
        apply_liveness: bool = True,
    ) -> list[FaceObservation]:
        profile = detection_profile or self.settings.detection_profile
        options = {}
        enabled = "liveness" in self.settings.addons
        if enabled:
            options = {"single_face": single_face, "apply_liveness": apply_liveness}
        try:
            faces = self.engine.analyze(
                image.pixels, require_embeddings=embeddings,
                detection_profile=profile, **options,
            )
            if enabled and apply_liveness and any(face.liveness is None for face in faces):
                raise LivenessUnavailable("The configured addon returned no liveness result")
        except LivenessUnavailable as exc:
            LOGGER.exception("liveness_inference_failed")
            raise ApiError("liveness_unavailable", "Liveness inference is unavailable.", 503) from exc
        return sorted(faces, key=lambda face: -face.area)

    def liveness_rejection(self, face: FaceObservation) -> str | None:
        result = face.liveness
        if result is None or self.settings.liveness_mode != "normal":
            return None
        if result["status"] == "input_rejected":
            return "liveness_input_rejected"
        if result["is_live"] is not True:
            return "liveness_fake"
        return None

    def require_recognition(self, face: FaceObservation) -> None:
        reason = self.liveness_rejection(face)
        if reason:
            message = (
                "Input does not meet liveness requirements. Adjust face position and retry."
                if reason == "liveness_input_rejected" else "The face did not pass liveness detection."
            )
            if reason == "liveness_input_rejected" and face.liveness is not None:
                explanation = face.liveness.get("reason")
                if isinstance(explanation, str) and explanation.strip():
                    message = explanation
            raise unprocessable(message, code=reason, liveness=face.liveness)
        if face.embedding is None:
            details = {"liveness": face.liveness} if face.liveness is not None else {}
            raise unprocessable(
                "Recognition embedding is unavailable.", code="embedding_unavailable", **details,
            )

    def detect(
        self,
        image: ImageData,
        *,
        max_faces: int | None,
        detection_profile: DetectionProfile | None = None,
    ) -> list[dict[str, Any]]:
        faces = self.analyze(
            image, embeddings=False, detection_profile=detection_profile
        )
        limit = self.settings.max_detected_faces
        if max_faces is not None:
            limit = min(limit, max_faces)
        faces = faces[:limit]
        height, width = image.pixels.shape[:2]
        return [face_result(face, width, height) for face in faces]

    @staticmethod
    def _selection_score(
        face: FaceObservation,
        *,
        image_width: int,
        image_height: int,
        strategy: str,
    ) -> float:
        left, top, right, bottom = face.bbox
        if strategy == "largest":
            return face.area
        if strategy != "center_largest":
            raise ValueError(f"Unsupported single-face selection strategy: {strategy}")
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        distance_squared = (
            (center_x - float(image_width) * 0.5) ** 2
            + (center_y - float(image_height) * 0.5) ** 2
        )
        # This follows the long-standing InsightFace/FaceAnalysis selection
        # metric. Detection confidence is intentionally not part of the score.
        return face.area - CENTER_LARGEST_DISTANCE_WEIGHT * distance_squared

    def select_from_observations(
        self,
        faces: list[FaceObservation],
        image: ImageData,
        profile: DetectionProfile,
    ) -> FaceObservation:
        if not faces:
            raise ValueError("faces must not be empty")
        height, width = image.pixels.shape[:2]
        # max() performs one linear scan and, like numpy.argmax(), keeps the
        # first observation when scores are exactly equal.
        return max(
            faces,
            key=lambda face: self._selection_score(
                face,
                image_width=width,
                image_height=height,
                strategy=profile.single_face_selection,
            ),
        )

    def selected_face(
        self, image: ImageData, *, detection_profile: DetectionProfile | None = None,
        apply_liveness: bool = True,
    ) -> FaceObservation:
        profile = detection_profile or self.settings.detection_profile
        faces = self.analyze(
            image, embeddings=True, detection_profile=profile,
            single_face=True, apply_liveness=apply_liveness,
        )
        if not faces:
            raise unprocessable("No usable face was detected.", code="face_not_found")
        face = self.select_from_observations(faces, image, profile)
        self.require_recognition(face)
        return face

    def largest_face(self, image: ImageData) -> FaceObservation:
        """Backward-compatible internal alias using the system profile."""

        return self.selected_face(image, detection_profile=self.settings.detection_profile)

    def _registration_reason(
        self,
        image: ImageData,
        faces: list[FaceObservation],
        *,
        review_mode: ReviewMode,
        embedding_dimension: int,
        embedding: np.ndarray | None = None,
        use_supplied_embedding: bool = False,
        require_landmarks: bool = False,
        invalid_embedding_reason: str = "invalid_embedding",
    ) -> str | None:
        if not faces:
            return "face_not_found"
        # Off is the low-friction enrollment mode: when several faces are
        # present, analyze() has already sorted them by area and we register the
        # largest one. Standard and strict retain the unambiguous single-face
        # requirement before applying their additional review rules.
        if review_mode != "off" and len(faces) != 1:
            return "multiple_faces"
        face = faces[0]
        liveness_reason = self.liveness_rejection(face)
        if liveness_reason:
            return liveness_reason
        if require_landmarks:
            if face.landmarks is None:
                return "invalid_landmarks"
            landmarks = np.asarray(face.landmarks, dtype=np.float32)
            if landmarks.shape != (5, 2) or not np.all(np.isfinite(landmarks)):
                return "invalid_landmarks"
        # In trusted mode ``None`` means the supplied vector was invalid. Never
        # fall back to an embedding a misbehaving engine may have returned while
        # require_embeddings=False.
        candidate = embedding if use_supplied_embedding else face.embedding
        if candidate is None:
            return invalid_embedding_reason
        candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
        if (
            candidate.shape != (embedding_dimension,)
            or not np.all(np.isfinite(candidate))
            or abs(float(np.linalg.norm(candidate)) - 1.0) > 2e-4
        ):
            return invalid_embedding_reason
        if review_mode == "off":
            return None
        width = face.bbox[2] - face.bbox[0]
        height = face.bbox[3] - face.bbox[1]
        if min(width, height) < self.settings.registration_min_face_size:
            return "face_too_small"
        if face.confidence < self.settings.registration_min_score:
            return "low_detection_score"
        if any(abs(float(getattr(face, field, 0.0))) > 45.0 for field in ("yaw", "pitch", "roll")):
            return "extreme_pose"
        if quality(face)["score"] < self.settings.registration_min_quality:
            return "low_quality"
        return None

    @staticmethod
    def _trusted_embedding(value: Any, embedding_dimension: int) -> np.ndarray | None:
        """Validate and canonically normalize one trusted JSON embedding.

        JSON has no native float32 scalar type, so the public contract means the
        values must be scalar JSON numbers that are representable as a finite
        float32 vector.  The caller already trusts image/embedding identity; this
        function deliberately performs no recognition inference or identity check.
        """

        if not isinstance(value, list) or len(value) != embedding_dimension:
            return None
        if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
            return None
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                embedding = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return None
        if embedding.shape != (embedding_dimension,) or not np.all(np.isfinite(embedding)):
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 1e-12 or abs(norm - 1.0) > 2e-4:
            return None
        # Normalize even an already-valid input to remove harmless serialization
        # error before SQLite persistence and search-profile quantization.
        return (embedding / norm).astype(np.float32, copy=False)

    @staticmethod
    def _validate_embedding_mode(
        collection: dict[str, Any],
        *,
        embedding_mode: EmbeddingMode,
        external_embeddings: list[Any] | None,
        embedding_contract_id: str | None,
        image_count: int,
    ) -> None:
        if embedding_mode == "server":
            if external_embeddings is not None or embedding_contract_id is not None:
                raise bad_request(
                    "External embedding fields require embedding_mode=external_trusted.",
                    code="unexpected_external_embedding",
                )
            return
        if external_embeddings is None:
            raise bad_request(
                "external_embeddings is required for external_trusted enrollment.",
                code="missing_external_embeddings",
            )
        if len(external_embeddings) != image_count:
            raise bad_request(
                "external_embeddings must contain exactly one vector for each image.",
                code="external_embedding_count_mismatch",
                image_count=image_count,
                embedding_count=len(external_embeddings),
            )
        if not embedding_contract_id:
            raise bad_request(
                "embedding_contract_id is required for external_trusted enrollment.",
                code="missing_embedding_contract_id",
            )
        expected = str(collection["embedding_contract_id"])
        if embedding_contract_id != expected:
            raise ApiError(
                "embedding_contract_mismatch",
                "The external embedding contract does not match this Collection.",
                409,
                {
                    "expected_embedding_contract_id": expected,
                    "provided_embedding_contract_id": embedding_contract_id,
                },
            )

    def prepare_registration(
        self,
        collection: dict[str, Any],
        images: list[ImageData],
        *,
        review_mode: ReviewMode,
        embedding_mode: EmbeddingMode = "server",
        external_embeddings: list[Any] | None = None,
        embedding_contract_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._validate_embedding_mode(
            collection,
            embedding_mode=embedding_mode,
            external_embeddings=external_embeddings,
            embedding_contract_id=embedding_contract_id,
            image_count=len(images),
        )
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        detection_profile = self.collection_detection_profile(collection)
        try:
            for index, image in enumerate(images):
                trusted_embedding = None
                if embedding_mode == "external_trusted":
                    assert external_embeddings is not None
                    trusted_embedding = self._trusted_embedding(
                        external_embeddings[index],
                        int(collection["embedding_dimension"]),
                    )
                faces = self.analyze(
                    image,
                    # Trusted enrollment still runs detector, landmarks and
                    # quality but never invokes the recognizer.
                    embeddings=embedding_mode == "server",
                    detection_profile=detection_profile,
                    single_face=review_mode == "off",
                    apply_liveness=self.settings.liveness_on_registration,
                )
                if review_mode == "off":
                    faces = (
                        [
                            self.select_from_observations(
                                faces,
                                image,
                                detection_profile,
                            )
                        ]
                        if faces
                        else []
                    )
                reason = self._registration_reason(
                    image,
                    faces,
                    review_mode=review_mode,
                    embedding_dimension=int(collection["embedding_dimension"]),
                    embedding=trusted_embedding,
                    use_supplied_embedding=embedding_mode == "external_trusted",
                    require_landmarks=embedding_mode == "external_trusted",
                    invalid_embedding_reason=(
                        "invalid_external_embedding"
                        if embedding_mode == "external_trusted"
                        else "invalid_embedding"
                    ),
                )
                if reason:
                    rejected.append({
                        "index": index, "filename": image.filename, "reason": reason,
                        **({"liveness": faces[0].liveness} if len(faces) == 1 and faces[0].liveness is not None else {}),
                    })
                    continue
                face = faces[0]
                final_embedding = (
                    trusted_embedding if embedding_mode == "external_trusted" else face.embedding
                )
                assert final_embedding is not None
                face_id = str(uuid.uuid4())
                height, width = image.pixels.shape[:2]
                crop_image = (
                    self.crops.encode(image.pixels, face.bbox)
                    if bool(collection.get("save_face_crops", False))
                    else None
                )
                accepted.append(
                    {
                        "id": face_id,
                        "embedding": final_embedding,
                        "embedding_dimension": collection["embedding_dimension"],
                        "bounding_box": bounding_box(face, width, height),
                        "landmarks": (
                            face.landmarks.tolist() if face.landmarks is not None else None
                        ),
                        "detection_score": clamp01(face.confidence),
                        "quality": quality(face),
                        **({"liveness": face.liveness} if face.liveness is not None else {}),
                        "model_id": collection["model_id"],
                        "model_digest": collection["model_digest"],
                        "preprocessing_version": collection["preprocessing_version"],
                        "embedding_source": embedding_mode,
                        "embedding_contract_id": collection["embedding_contract_id"],
                        # New FaceSamples keep their optional crop in SQLite. The
                        # path field remains null while legacy path-backed crops
                        # continue to be readable and are not migrated implicitly.
                        "crop_path": None,
                        "crop_image": crop_image,
                        "crop_media_type": "image/jpeg" if crop_image is not None else None,
                        "created_at": utc_now(),
                        "_registration_index": index,
                        "_registration_filename": image.filename,
                    }
                )
        except Exception:
            for item in accepted:
                try:
                    self.crops.delete(item.get("crop_path"))
                except OSError:
                    LOGGER.warning("unable_to_remove_orphan_face_crop", exc_info=True)
            raise
        return accepted, rejected

    def _strict_registration_review(
        self,
        collection: dict[str, Any],
        person_id: str,
        faces: list[dict[str, Any]],
        existing_embeddings: list[np.ndarray],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Apply deterministic class-in versus class-out enrollment review."""

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        same_embeddings = list(existing_embeddings)
        profile = str(collection["search_profile"])
        for face in faces:
            embedding = np.asarray(face["embedding"], dtype=np.float32)
            # The first usable sample bootstraps an empty Person. Samples later in
            # this same multipart request can use it as a class-in reference.
            if not same_embeddings:
                accepted.append(face)
                same_embeddings.append(embedding)
                continue

            max_same = max(
                profile_similarity(profile, embedding, reference) for reference in same_embeddings
            )
            try:
                other = self.search_indexes.best_other_person(
                    str(collection["id"]),
                    embedding,
                    exclude_person_id=person_id,
                )
            except SearchIndexError as exc:
                raise self._search_unavailable(exc) from exc

            if other is not None and max_same <= float(other["similarity"]):
                self._delete_crop_paths([face.get("crop_path")])
                rejected.append(
                    {
                        "index": int(face["_registration_index"]),
                        "filename": str(face["_registration_filename"]),
                        "reason": "identity_similarity_conflict",
                        "same_person_similarity": float(max_same),
                        "other_person_similarity": float(other["similarity"]),
                        "other_person_id": str(other["person_id"]),
                        "matched_face_id": str(other["face_id"]),
                        **({"liveness": face["liveness"]} if face.get("liveness") is not None else {}),
                    }
                )
                continue

            accepted.append(face)
            same_embeddings.append(embedding)
        return accepted, rejected

    @staticmethod
    def public_face(face: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: face[key]
            for key in (
                "id",
                "person_id",
                "bounding_box",
                "landmarks",
                "detection_score",
                "quality",
                "liveness",
                "model_id",
                "model_digest",
                "preprocessing_version",
                "embedding_source",
                "embedding_contract_id",
                "created_at",
            )
            if key in face
        }
        result["has_crop"] = bool(
            face.get(
                "has_crop",
                face.get("crop_image") is not None or face.get("crop_path"),
            )
        )
        return result

    def create_person(
        self,
        collection_id: str,
        person: dict[str, Any],
        images: list[ImageData],
        *,
        review_mode: ReviewMode = "off",
        embedding_mode: EmbeddingMode = "server",
        external_embeddings: list[Any] | None = None,
        embedding_contract_id: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        collection = self.collection(collection_id)
        faces, rejected = self.prepare_registration(
            collection,
            images,
            review_mode=review_mode,
            embedding_mode=embedding_mode,
            external_embeddings=external_embeddings,
            embedding_contract_id=embedding_contract_id,
        )
        if not faces:
            raise unprocessable(
                "None of the uploaded images contained one registrable face.",
                code="registration_failed",
                rejected_images=rejected,
            )
        try:
            if review_mode == "strict":
                # Keep the identity decision and write under one Collection
                # lock, so another enrollment cannot change the class-out
                # maximum between review and commit.
                def strict_create():
                    reviewed, strict_rejected = self._strict_registration_review(
                        collection, str(person["id"]), faces, []
                    )
                    created_item, mutation = self.repository.create_person_with_faces_with_mutation(
                        collection_id, person, reviewed
                    )
                    return (created_item, reviewed, strict_rejected), mutation

                created, faces, strict_rejected = self.search_indexes.run_mutation(
                    collection_id,
                    strict_create,
                    # The Repository checks capacity using only rows that pass
                    # strict review; the native index has the same hard limit.
                    expected_additions=0,
                )
                rejected.extend(strict_rejected)
            else:
                created = self.search_indexes.run_mutation(
                    collection_id,
                    lambda: self.repository.create_person_with_faces_with_mutation(
                        collection_id, person, faces
                    ),
                    expected_additions=len(faces),
                )
        except SearchMutationCommittedError as exc:
            raise self._search_unavailable(exc) from exc
        except KeyError as exc:
            self._delete_crop_paths([face.get("crop_path") for face in faces])
            raise not_found("collection", collection_id) from exc
        except (CollectionCapacityExceeded, MaxFacesPerPersonExceeded) as exc:
            self._delete_crop_paths([face.get("crop_path") for face in faces])
            raise self._quota_error(exc) from exc
        except SearchIndexCapacityError as exc:
            self._delete_crop_paths([face.get("crop_path") for face in faces])
            raise self._quota_error(exc) from exc
        except SearchIndexError as exc:
            self._delete_crop_paths([face.get("crop_path") for face in faces])
            raise self._search_unavailable(exc) from exc
        except sqlite3.IntegrityError as exc:
            self._delete_crop_paths([face.get("crop_path") for face in faces])
            raise conflict(
                "The person ID or external ID already exists.", code="person_exists"
            ) from exc
        except Exception:
            self._delete_crop_paths([face.get("crop_path") for face in faces])
            raise
        return created, [self.public_face(face) for face in faces], rejected

    def add_faces(
        self,
        collection_id: str,
        person_id: str,
        images: list[ImageData],
        *,
        review_mode: ReviewMode = "off",
        embedding_mode: EmbeddingMode = "server",
        external_embeddings: list[Any] | None = None,
        embedding_contract_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        collection = self.collection(collection_id)
        if self.repository.get_person(collection_id, person_id) is None:
            raise not_found("person", person_id)
        faces, rejected = self.prepare_registration(
            collection,
            images,
            review_mode=review_mode,
            embedding_mode=embedding_mode,
            external_embeddings=external_embeddings,
            embedding_contract_id=embedding_contract_id,
        )
        if faces:
            try:
                if review_mode == "strict":

                    def strict_add():
                        if self.repository.get_person(collection_id, person_id) is None:
                            raise KeyError(person_id)
                        reviewed, strict_rejected = self._strict_registration_review(
                            collection,
                            person_id,
                            faces,
                            self.repository.list_face_embeddings(collection_id, person_id),
                        )
                        mutation = (
                            self.repository.add_faces_with_mutation(
                                collection_id, person_id, reviewed
                            )
                            if reviewed
                            else None
                        )
                        return (reviewed, strict_rejected), mutation

                    faces, strict_rejected = self.search_indexes.run_mutation(
                        collection_id,
                        strict_add,
                        expected_additions=0,
                    )
                    rejected.extend(strict_rejected)
                else:
                    self.search_indexes.run_mutation(
                        collection_id,
                        lambda: (
                            None,
                            self.repository.add_faces_with_mutation(
                                collection_id, person_id, faces
                            ),
                        ),
                        expected_additions=len(faces),
                    )
            except SearchMutationCommittedError as exc:
                raise self._search_unavailable(exc) from exc
            except KeyError as exc:
                self._delete_crop_paths([face.get("crop_path") for face in faces])
                raise not_found("person", person_id) from exc
            except (CollectionCapacityExceeded, MaxFacesPerPersonExceeded) as exc:
                self._delete_crop_paths([face.get("crop_path") for face in faces])
                raise self._quota_error(exc) from exc
            except SearchIndexCapacityError as exc:
                self._delete_crop_paths([face.get("crop_path") for face in faces])
                raise self._quota_error(exc) from exc
            except SearchIndexError as exc:
                self._delete_crop_paths([face.get("crop_path") for face in faces])
                raise self._search_unavailable(exc) from exc
            except Exception:
                self._delete_crop_paths([face.get("crop_path") for face in faces])
                raise
        return [self.public_face(face) for face in faces], rejected

    def search(
        self,
        collection_id: str,
        image: ImageData,
        *,
        limit: int,
        threshold: float | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
        collection = self.collection(collection_id)
        profile = self.collection_detection_profile(collection)
        face = self.selected_face(image, detection_profile=profile)
        assert face.embedding is not None
        selected_threshold = (
            float(collection["default_threshold"]) if threshold is None else threshold
        )
        try:
            matches = self.search_indexes.search(
                collection_id,
                face.embedding,
                limit=limit,
                threshold=selected_threshold,
            )
        except KeyError as exc:
            raise not_found("collection", collection_id) from exc
        except SearchIndexError as exc:
            raise self._search_unavailable(exc) from exc
        height, width = image.pixels.shape[:2]
        return face_result(face, width, height), matches[:limit], selected_threshold

    def search_all_faces(
        self,
        collection_id: str,
        image: ImageData,
        *,
        max_faces: int,
        threshold: float | None,
    ) -> tuple[list[dict[str, Any]], float]:
        """Detect once and search each retained face against one Collection.

        This is the shared server-side path used by RTSP recognition. It avoids
        re-detecting cropped faces and never exposes embeddings in stream state.
        """

        collection = self.collection(collection_id)
        profile = self.collection_detection_profile(collection)
        faces = self.analyze(image, embeddings=True, detection_profile=profile)
        faces = faces[: min(max_faces, self.settings.max_detected_faces)]
        selected_threshold = (
            float(collection["default_threshold"]) if threshold is None else threshold
        )
        height, width = image.pixels.shape[:2]
        results: list[dict[str, Any]] = []
        try:
            for face in faces:
                if self.liveness_rejection(face):
                    results.append({
                        "status": "liveness_blocked",
                        "face": face_result(face, width, height), "match": None,
                    })
                    continue
                self.require_recognition(face)
                assert face.embedding is not None
                matches = self.search_indexes.search(
                    collection_id,
                    face.embedding,
                    limit=1,
                    threshold=selected_threshold,
                )
                results.append(
                    {
                        "face": face_result(face, width, height),
                        "match": matches[0] if matches else None,
                    }
                )
        except KeyError as exc:
            raise not_found("collection", collection_id) from exc
        except SearchIndexError as exc:
            raise self._search_unavailable(exc) from exc
        return results, selected_threshold

    def delete_collection(self, collection_id: str, *, force: bool) -> None:
        # Deletion is an administrative cleanup operation. It intentionally
        # permits a Collection bound to an older model while enrollment,
        # inference, and search continue to enforce the active contract.
        self.raw_collection(collection_id)
        status, crop_paths = self.search_indexes.run_drop(
            collection_id,
            lambda: self._delete_collection_mutation(collection_id, force=force),
        )
        if status == "not_empty":
            raise conflict(
                "The collection is not empty; repeat with force=true.",
                code="collection_not_empty",
            )
        if status == "in_use":
            raise conflict(
                "The Collection is used by one or more Monitors; update or delete "
                "those Monitors first.",
                code="collection_in_use",
            )
        if status == "missing":
            raise not_found("collection", collection_id)
        for path in crop_paths:
            self.crops.delete(path)
        self.crops.delete_collection(collection_id)

    def _delete_collection_mutation(
        self, collection_id: str, *, force: bool
    ) -> tuple[tuple[str, list[str]], Any]:
        status, crops, mutation = self.repository.delete_collection_with_mutation(
            collection_id, force=force
        )
        return (status, crops), mutation

    def delete_person(self, collection_id: str, person_id: str) -> list[str]:
        self.collection(collection_id)

        def operation():
            deleted, crops, mutation = self.repository.delete_person_with_mutation(
                collection_id, person_id
            )
            return (deleted, crops), mutation

        try:
            deleted, crops = self.search_indexes.run_mutation(collection_id, operation)
        except SearchMutationCommittedError as exc:
            committed = exc.committed_result
            if isinstance(committed, tuple) and len(committed) == 2:
                committed_deleted, committed_crops = committed
                if committed_deleted and isinstance(committed_crops, list):
                    self._delete_crop_paths(committed_crops)
            raise self._search_unavailable(exc) from exc
        except KeyError as exc:
            raise not_found("collection", collection_id) from exc
        except SearchIndexError as exc:
            raise self._search_unavailable(exc) from exc
        if not deleted:
            raise not_found("person", person_id)
        return crops

    def delete_face(self, collection_id: str, person_id: str, face_id: str) -> str | None:
        self.collection(collection_id)

        def operation():
            deleted, crop, mutation = self.repository.delete_face_with_mutation(
                collection_id, person_id, face_id
            )
            return (deleted, crop), mutation

        try:
            deleted, crop = self.search_indexes.run_mutation(collection_id, operation)
        except SearchMutationCommittedError as exc:
            committed = exc.committed_result
            if isinstance(committed, tuple) and len(committed) == 2:
                committed_deleted, committed_crop = committed
                if committed_deleted:
                    self._delete_crop_paths([committed_crop])
            raise self._search_unavailable(exc) from exc
        except KeyError as exc:
            raise not_found("collection", collection_id) from exc
        except SearchIndexError as exc:
            raise self._search_unavailable(exc) from exc
        if not deleted:
            raise not_found("face", face_id)
        return crop

    def update_collection(self, collection_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        current = self.collection(collection_id)
        detection_columns = {
            "detector_input_sizes": "input_sizes",
            "detector_threshold": "threshold",
            "detector_nms_threshold": "nms_threshold",
            "single_face_selection": "single_face_selection",
        }

        def current_value(key: str) -> Any:
            detection_key = detection_columns.get(key)
            if detection_key is not None:
                value = current["detection"][detection_key]
                if key == "detector_input_sizes":
                    return [list(size) for size in value]
                return value
            return current.get(key)

        # Browsers and SDKs may submit a complete edit form. Only actual changes
        # should update SQLite or allocate a second index generation.
        effective_changes = {
            key: value for key, value in changes.items() if current_value(key) != value
        }
        if not effective_changes:
            return current
        if any(key in effective_changes for key in detection_columns):
            target_detection = dict(current["detection"])
            for column, detection_key in detection_columns.items():
                if column in effective_changes:
                    target_detection[detection_key] = effective_changes[column]
            profile = DetectionProfile.from_mapping(target_detection)
            try:
                self.engine.validate_detection_profile(profile)
            except ValueError as exc:
                raise bad_request(
                    str(exc), code="invalid_detection_profile"
                ) from exc
        if "capacity_rows" in effective_changes and int(effective_changes["capacity_rows"]) < int(
            current["face_count"]
        ):
            raise self._quota_error(
                CollectionCapacityExceeded(
                    capacity_rows=int(effective_changes["capacity_rows"]),
                    requested_rows=int(current["face_count"]),
                )
            )
        try:
            item = self.search_indexes.run_configuration_update(
                collection_id,
                lambda: self.repository.update_collection(collection_id, effective_changes),
                changes=effective_changes,
            )
        except (CollectionCapacityExceeded, MaxFacesPerPersonExceeded) as exc:
            raise self._quota_error(exc) from exc
        except SearchIndexError as exc:
            raise self._search_unavailable(exc) from exc
        except KeyError as exc:
            raise not_found("collection", collection_id) from exc
        if item is None:
            raise not_found("collection", collection_id)
        return item
