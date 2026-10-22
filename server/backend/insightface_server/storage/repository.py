from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ..config import DetectionProfile
from ..models.embedding_contract import embedding_contract_id_for_collection
from .database import Database

SEARCH_PROFILES = frozenset(
    {"fp32_v1", "fp16_v1", "bf16_v1", "int8_x1000_v1", "int8_x736_v1"}
)
SEARCH_LOAD_POLICIES = frozenset({"lazy", "eager"})


class CollectionCapacityExceeded(ValueError):
    def __init__(self, *, capacity_rows: int, requested_rows: int):
        self.capacity_rows = capacity_rows
        self.requested_rows = requested_rows
        super().__init__(
            f"collection capacity {capacity_rows} would be exceeded by {requested_rows} rows"
        )


class MaxFacesPerPersonExceeded(ValueError):
    def __init__(self, *, max_faces: int, requested_faces: int):
        self.max_faces = max_faces
        self.requested_faces = requested_faces
        super().__init__(
            f"person face limit {max_faces} would be exceeded by {requested_faces} faces"
        )


@dataclass(frozen=True, slots=True)
class SearchChange:
    seq: int
    collection_id: str
    revision: int
    operation: str
    vector_id: int | None
    person_numeric_id: int | None
    face_id: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class SearchMutation:
    collection_id: str
    revision: int
    changes: tuple[SearchChange, ...]

    @property
    def max_seq(self) -> int:
        return max((change.seq for change in self.changes), default=0)


@dataclass(frozen=True, slots=True)
class IndexFace:
    vector_id: int
    person_numeric_id: int
    face_id: str
    person_id: str
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class IndexFaceBatch:
    """One batch from a single consistent SQLite read snapshot."""

    revision: int
    faces: tuple[IndexFace, ...]
    max_seq: int


@dataclass(frozen=True, slots=True)
class SearchState:
    collection_id: str
    revision: int
    max_seq: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Repository:
    def __init__(self, database: Database):
        self.database = database

    @staticmethod
    def _collection_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["metadata"] = json.loads(value.pop("metadata_json"))
        defaults = DetectionProfile()
        value["detection"] = {
            "input_sizes": json.loads(
                value.pop(
                    "detector_input_sizes_json",
                    json.dumps(defaults.as_dict()["input_sizes"]),
                )
            ),
            "threshold": float(value.pop("detector_threshold", defaults.threshold)),
            "nms_threshold": float(
                value.pop("detector_nms_threshold", defaults.nms_threshold)
            ),
            "single_face_selection": value.pop(
                "single_face_selection", defaults.single_face_selection
            ),
        }
        value.setdefault("detection_revision", 1)
        value["person_count"] = int(value.get("person_count", 0))
        value["face_count"] = int(value.get("face_count", 0))
        value["save_face_crops"] = bool(value.get("save_face_crops", False))
        value["embedding_contract_id"] = embedding_contract_id_for_collection(value)
        return value

    @staticmethod
    def _person_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["metadata"] = json.loads(value.pop("metadata_json"))
        value["face_count"] = int(value.get("face_count", 0))
        value.pop("collection_id", None)
        return value

    @staticmethod
    def _monitor_row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["enabled"] = bool(value["enabled"])
        value["emit_unknown"] = bool(value.pop("emit_unknown"))
        value["preview_enabled"] = bool(value["preview_enabled"])
        return value

    @staticmethod
    def _face_row(row: sqlite3.Row, *, include_embedding: bool = True) -> dict[str, Any]:
        value = dict(row)
        value["bounding_box"] = json.loads(value.pop("bounding_box_json"))
        landmarks_json = value.pop("landmarks_json")
        value["landmarks"] = json.loads(landmarks_json) if landmarks_json else None
        value["quality"] = json.loads(value.pop("quality_json"))
        liveness_json = value.pop("liveness_json", None)
        if liveness_json is not None:
            value["liveness"] = json.loads(liveness_json)
        if include_embedding:
            value["embedding"] = np.frombuffer(
                value["embedding"], dtype=np.float32
            ).copy()
        else:
            value.pop("embedding", None)
        # Crop bytes are intentionally only exposed by get_face_crop(). Keep this
        # defensive pop for callers constructing custom SELECTs in tests.
        crop_image = value.pop("crop_image", None)
        value["has_crop"] = bool(
            value.get("has_crop", crop_image is not None or value.get("crop_path"))
        )
        value.pop("crop_path", None)
        value.pop("collection_id", None)
        return value

    @staticmethod
    def _face_columns(*, include_embedding: bool) -> str:
        columns = """
            id,collection_id,person_id,embedding_dimension,
            bounding_box_json,landmarks_json,detection_score,quality_json,liveness_json,
            model_id,model_digest,preprocessing_version,
            embedding_source,embedding_contract_id,crop_path,created_at,
            CASE WHEN crop_image IS NOT NULL OR crop_path IS NOT NULL
                 THEN 1 ELSE 0 END AS has_crop
        """
        if include_embedding:
            return "embedding," + columns
        return columns

    @staticmethod
    def _collection_query() -> str:
        return """
            SELECT c.*,
              (SELECT count(*) FROM persons p WHERE p.collection_id=c.id) AS person_count,
              (SELECT count(*) FROM face_samples f WHERE f.collection_id=c.id) AS face_count
            FROM collections c
        """

    def create_collection(self, item: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        detection = DetectionProfile()
        with self.database.write() as connection:
            connection.execute(
                """INSERT INTO collections(
                    id,name,description,default_threshold,model_id,embedding_contract_id,
                    model_digest,embedding_dimension,preprocessing_version,
                    metadata_json,created_at,updated_at,search_profile,capacity_rows,
                    max_faces_per_person,load_policy,save_face_crops,
                    detector_input_sizes_json,detector_threshold,
                    detector_nms_threshold,single_face_selection,detection_revision
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    item["id"],
                    item["name"],
                    item.get("description", ""),
                    item["default_threshold"],
                    item["model_id"],
                    embedding_contract_id_for_collection(item),
                    item["model_digest"],
                    item["embedding_dimension"],
                    item["preprocessing_version"],
                    json.dumps(item.get("metadata", {}), sort_keys=True),
                    now,
                    now,
                    item.get("search_profile", "fp32_v1"),
                    int(item.get("capacity_rows", 100000)),
                    int(item.get("max_faces_per_person", 20)),
                    item.get("load_policy", "lazy"),
                    int(bool(item.get("save_face_crops", False))),
                    json.dumps(
                        item.get(
                            "detector_input_sizes", detection.as_dict()["input_sizes"]
                        ),
                        separators=(",", ":"),
                    ),
                    float(item.get("detector_threshold", detection.threshold)),
                    float(
                        item.get("detector_nms_threshold", detection.nms_threshold)
                    ),
                    item.get(
                        "single_face_selection", detection.single_face_selection
                    ),
                    1,
                ),
            )
            row = connection.execute(
                self._collection_query() + " WHERE c.id=?", (item["id"],)
            ).fetchone()
            if row is None:  # pragma: no cover - guarded by the INSERT above
                raise RuntimeError("new Collection could not be read in its transaction")
            result = self._collection_row(row)
        return result

    def get_collection(self, collection_id: str) -> dict[str, Any] | None:
        with self.database.read() as connection:
            row = connection.execute(
                self._collection_query() + " WHERE c.id=?", (collection_id,)
            ).fetchone()
        return self._collection_row(row) if row else None

    def list_collections(self, after: str, limit: int) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                self._collection_query()
                + " WHERE c.id>? ORDER BY c.id LIMIT ?",
                (after, limit),
            ).fetchall()
        return [self._collection_row(row) for row in rows]

    def create_monitor(self, item: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self.database.write() as connection:
            connection.execute(
                """INSERT INTO monitors(
                    id,name,description,enabled,source_type,source_url_ciphertext,
                    collection_id,inference_fps,match_threshold,event_buffer_size,
                    confirm_frames,absence_timeout_seconds,cooldown_seconds,
                    emit_unknown,preview_enabled,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    item["id"],
                    item["name"],
                    item.get("description", ""),
                    int(bool(item.get("enabled", True))),
                    item.get("source_type", "rtsp"),
                    item["source_url_ciphertext"],
                    item["collection_id"],
                    float(item.get("inference_fps", 2.0)),
                    item.get("match_threshold"),
                    int(item.get("event_buffer_size", 1000)),
                    int(item.get("confirm_frames", 3)),
                    float(item.get("absence_timeout_seconds", 3.0)),
                    float(item.get("cooldown_seconds", 10.0)),
                    int(bool(item.get("emit_unknown", True))),
                    int(bool(item.get("preview_enabled", False))),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM monitors WHERE id=?", (item["id"],)
            ).fetchone()
        if row is None:  # pragma: no cover - guarded by INSERT
            raise RuntimeError("new Monitor could not be read in its transaction")
        return self._monitor_row(row)

    def get_monitor(self, monitor_id: str) -> dict[str, Any] | None:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT * FROM monitors WHERE id=?", (monitor_id,)
            ).fetchone()
        return self._monitor_row(row) if row else None

    def list_monitors(self, after: str, limit: int) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT * FROM monitors WHERE id>? ORDER BY id LIMIT ?",
                (after, limit),
            ).fetchall()
        return [self._monitor_row(row) for row in rows]

    def list_enabled_monitors(self) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT * FROM monitors WHERE enabled=1 ORDER BY id"
            ).fetchall()
        return [self._monitor_row(row) for row in rows]

    def update_monitor(
        self, monitor_id: str, changes: dict[str, Any]
    ) -> dict[str, Any] | None:
        allowed = {
            "name",
            "description",
            "enabled",
            "source_type",
            "source_url_ciphertext",
            "collection_id",
            "inference_fps",
            "match_threshold",
            "event_buffer_size",
            "confirm_frames",
            "absence_timeout_seconds",
            "cooldown_seconds",
            "emit_unknown",
            "preview_enabled",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            if key not in allowed:
                continue
            assignments.append(f"{key}=?")
            if key in {"enabled", "emit_unknown", "preview_enabled"}:
                value = int(bool(value))
            values.append(value)
        if not assignments:
            return self.get_monitor(monitor_id)
        assignments.append("updated_at=?")
        values.extend([utc_now(), monitor_id])
        with self.database.write() as connection:
            cursor = connection.execute(
                f"UPDATE monitors SET {', '.join(assignments)} WHERE id=?",
                values,
            )
            if not cursor.rowcount:
                return None
            row = connection.execute(
                "SELECT * FROM monitors WHERE id=?", (monitor_id,)
            ).fetchone()
        return self._monitor_row(row) if row else None

    def delete_monitor(self, monitor_id: str) -> bool:
        with self.database.write() as connection:
            cursor = connection.execute(
                "DELETE FROM monitors WHERE id=?", (monitor_id,)
            )
        return bool(cursor.rowcount)

    def monitor_ids_for_collection(self, collection_id: str) -> list[str]:
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT id FROM monitors WHERE collection_id=? ORDER BY id",
                (collection_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def update_collection(
        self, collection_id: str, changes: dict[str, Any]
    ) -> dict[str, Any] | None:
        allowed = {
            "name": "name",
            "description": "description",
            "default_threshold": "default_threshold",
            "metadata": "metadata_json",
            "capacity_rows": "capacity_rows",
            "max_faces_per_person": "max_faces_per_person",
            "load_policy": "load_policy",
            "save_face_crops": "save_face_crops",
            "detector_input_sizes": "detector_input_sizes_json",
            "detector_threshold": "detector_threshold",
            "detector_nms_threshold": "detector_nms_threshold",
            "single_face_selection": "single_face_selection",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, column in allowed.items():
            if key not in changes:
                continue
            assignments.append(f"{column}=?")
            value = changes[key]
            if key == "metadata":
                value = json.dumps(value, sort_keys=True)
            elif key == "detector_input_sizes":
                value = json.dumps(value, separators=(",", ":"))
            elif key == "save_face_crops":
                value = int(bool(value))
            values.append(value)
        if not assignments:
            return self.get_collection(collection_id)
        if any(key.startswith("detector_") or key == "single_face_selection" for key in changes):
            assignments.append("detection_revision=detection_revision+1")
        assignments.append("updated_at=?")
        values.extend([utc_now(), collection_id])
        result: dict[str, Any] | None = None
        with self.database.write() as connection:
            if "capacity_rows" in changes:
                face_count = int(
                    connection.execute(
                        "SELECT count(*) FROM face_samples WHERE collection_id=?",
                        (collection_id,),
                    ).fetchone()[0]
                )
                if int(changes["capacity_rows"]) < face_count:
                    raise CollectionCapacityExceeded(
                        capacity_rows=int(changes["capacity_rows"]),
                        requested_rows=face_count,
                    )
            if "max_faces_per_person" in changes:
                largest = int(
                    connection.execute(
                        """SELECT coalesce(max(face_count), 0)
                           FROM (
                             SELECT count(*) AS face_count FROM face_samples
                             WHERE collection_id=? GROUP BY person_id
                           )""",
                        (collection_id,),
                    ).fetchone()[0]
                )
                if int(changes["max_faces_per_person"]) < largest:
                    raise MaxFacesPerPersonExceeded(
                        max_faces=int(changes["max_faces_per_person"]),
                        requested_faces=largest,
                    )
            cursor = connection.execute(
                f"UPDATE collections SET {', '.join(assignments)} WHERE id=?", values
            )
            if cursor.rowcount:
                row = connection.execute(
                    self._collection_query() + " WHERE c.id=?", (collection_id,)
                ).fetchone()
                if row is None:  # pragma: no cover - guarded by the UPDATE above
                    raise RuntimeError("updated Collection disappeared in its transaction")
                result = self._collection_row(row)
        return result

    def delete_collection(
        self, collection_id: str, *, force: bool
    ) -> tuple[str, list[str]]:
        status, crops, _ = self.delete_collection_with_mutation(
            collection_id, force=force
        )
        return status, crops

    def delete_collection_with_mutation(
        self, collection_id: str, *, force: bool
    ) -> tuple[str, list[str], SearchMutation | None]:
        mutation: SearchMutation | None = None
        with self.database.write() as connection:
            exists = connection.execute(
                "SELECT 1 FROM collections WHERE id=?", (collection_id,)
            ).fetchone()
            if not exists:
                return "missing", [], None
            if connection.execute(
                "SELECT 1 FROM monitors WHERE collection_id=? LIMIT 1",
                (collection_id,),
            ).fetchone():
                return "in_use", [], None
            if not force:
                counts = connection.execute(
                    "SELECT (SELECT count(*) FROM persons WHERE collection_id=?), "
                    "(SELECT count(*) FROM face_samples WHERE collection_id=?)",
                    (collection_id, collection_id),
                ).fetchone()
                if int(counts[0]) or int(counts[1]):
                    return "not_empty", [], None
            crops = [
                row[0]
                for row in connection.execute(
                    "SELECT crop_path FROM face_samples WHERE collection_id=? "
                    "AND crop_path IS NOT NULL",
                    (collection_id,),
                )
            ]
            revision = self._next_search_revision(connection, collection_id)
            mutation = self._record_search_changes(
                connection,
                collection_id,
                revision,
                [("clear", None, None, None)],
            )
            cursor = connection.execute("DELETE FROM collections WHERE id=?", (collection_id,))
        return ("deleted" if cursor.rowcount else "missing"), crops, mutation

    def create_person_with_faces(
        self, collection_id: str, person: dict[str, Any], faces: list[dict[str, Any]]
    ) -> dict[str, Any]:
        created, _ = self.create_person_with_faces_with_mutation(
            collection_id, person, faces
        )
        return created

    def create_person_with_faces_with_mutation(
        self, collection_id: str, person: dict[str, Any], faces: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], SearchMutation | None]:
        now = utc_now()
        mutation: SearchMutation | None = None
        created: dict[str, Any] | None = None
        with self.database.write() as connection:
            self._check_face_limits(
                connection, collection_id, person_id=None, add_count=len(faces)
            )
            connection.execute(
                """INSERT INTO persons(
                    collection_id,id,name,external_id,metadata_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?)""",
                (
                    collection_id,
                    person["id"],
                    person.get("name"),
                    person.get("external_id"),
                    json.dumps(person.get("metadata", {}), sort_keys=True),
                    now,
                    now,
                ),
            )
            person_numeric_id = self._ensure_person_numeric_id(
                connection, collection_id, person["id"]
            )
            added: list[tuple[int, int, str]] = []
            for face in faces:
                vector_id = self._insert_face(
                    connection,
                    collection_id,
                    person["id"],
                    person_numeric_id,
                    face,
                )
                added.append((vector_id, person_numeric_id, face["id"]))
            if added:
                revision = self._next_search_revision(connection, collection_id)
                mutation = self._record_search_changes(
                    connection,
                    collection_id,
                    revision,
                    [("add", *entry) for entry in added],
                )
            created = self._get_person_from_connection(
                connection, collection_id, person["id"]
            )
        assert created is not None
        return created, mutation

    @staticmethod
    def _insert_face(
        connection: sqlite3.Connection,
        collection_id: str,
        person_id: str,
        person_numeric_id: int,
        face: dict[str, Any],
    ) -> int:
        connection.execute(
            """INSERT INTO face_samples(
                id,collection_id,person_id,embedding,embedding_dimension,
                bounding_box_json,landmarks_json,detection_score,quality_json,
                model_id,model_digest,preprocessing_version,
                embedding_source,embedding_contract_id,
                crop_path,crop_image,crop_media_type,created_at,liveness_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                face["id"],
                collection_id,
                person_id,
                np.asarray(face["embedding"], dtype=np.float32).tobytes(),
                int(face["embedding_dimension"]),
                json.dumps(face["bounding_box"], sort_keys=True),
                json.dumps(face.get("landmarks")),
                face["detection_score"],
                json.dumps(face["quality"], sort_keys=True),
                face["model_id"],
                face["model_digest"],
                face["preprocessing_version"],
                face.get("embedding_source", "server"),
                face.get("embedding_contract_id"),
                face.get("crop_path"),
                face.get("crop_image"),
                face.get("crop_media_type"),
                face.get("created_at", utc_now()),
                json.dumps(face["liveness"]) if face.get("liveness") is not None else None,
            ),
        )
        cursor = connection.execute(
            """INSERT INTO search_vector_ids(
                   collection_id,person_id,person_numeric_id,face_id
               ) VALUES(?,?,?,?)""",
            (collection_id, person_id, person_numeric_id, face["id"]),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("failed to allocate stable face search identifier")
        return int(cursor.lastrowid)

    def add_faces(
        self, collection_id: str, person_id: str, faces: list[dict[str, Any]]
    ) -> None:
        self.add_faces_with_mutation(collection_id, person_id, faces)

    def add_faces_with_mutation(
        self, collection_id: str, person_id: str, faces: list[dict[str, Any]]
    ) -> SearchMutation | None:
        mutation: SearchMutation | None = None
        with self.database.write() as connection:
            exists = connection.execute(
                "SELECT 1 FROM persons WHERE collection_id=? AND id=?",
                (collection_id, person_id),
            ).fetchone()
            if not exists:
                raise KeyError(person_id)
            self._check_face_limits(
                connection, collection_id, person_id=person_id, add_count=len(faces)
            )
            person_numeric_id = self._ensure_person_numeric_id(
                connection, collection_id, person_id
            )
            added: list[tuple[int, int, str]] = []
            for face in faces:
                vector_id = self._insert_face(
                    connection,
                    collection_id,
                    person_id,
                    person_numeric_id,
                    face,
                )
                added.append((vector_id, person_numeric_id, face["id"]))
            connection.execute(
                "UPDATE persons SET updated_at=? WHERE collection_id=? AND id=?",
                (utc_now(), collection_id, person_id),
            )
            if added:
                revision = self._next_search_revision(connection, collection_id)
                mutation = self._record_search_changes(
                    connection,
                    collection_id,
                    revision,
                    [("add", *entry) for entry in added],
                )
        return mutation

    def get_person(self, collection_id: str, person_id: str) -> dict[str, Any] | None:
        with self.database.read() as connection:
            return self._get_person_from_connection(connection, collection_id, person_id)

    @classmethod
    def _get_person_from_connection(
        cls, connection: sqlite3.Connection, collection_id: str, person_id: str
    ) -> dict[str, Any] | None:
        row = connection.execute(
            """SELECT p.*, count(f.id) AS face_count
               FROM persons p LEFT JOIN face_samples f
                 ON f.collection_id=p.collection_id AND f.person_id=p.id
               WHERE p.collection_id=? AND p.id=?
               GROUP BY p.collection_id,p.id""",
            (collection_id, person_id),
        ).fetchone()
        return cls._person_row(row) if row else None

    def list_persons(
        self, collection_id: str, after: str, limit: int, query: str | None = None
    ) -> list[dict[str, Any]]:
        filters = ["p.collection_id=?", "p.id>?"]
        values: list[Any] = [collection_id, after]
        if query:
            filters.append(
                "(lower(p.id) LIKE ? OR lower(coalesce(p.name,'')) LIKE ? "
                "OR lower(coalesce(p.external_id,'')) LIKE ?)"
            )
            needle = f"%{query.lower()}%"
            values.extend([needle, needle, needle])
        values.append(limit)
        with self.database.read() as connection:
            rows = connection.execute(
                f"""SELECT p.*, count(f.id) AS face_count
                    FROM persons p LEFT JOIN face_samples f
                      ON f.collection_id=p.collection_id AND f.person_id=p.id
                    WHERE {' AND '.join(filters)}
                    GROUP BY p.collection_id,p.id ORDER BY p.id LIMIT ?""",
                values,
            ).fetchall()
        return [self._person_row(row) for row in rows]

    def update_person(
        self, collection_id: str, person_id: str, changes: dict[str, Any]
    ) -> dict[str, Any] | None:
        allowed = {"name": "name", "external_id": "external_id", "metadata": "metadata_json"}
        assignments: list[str] = []
        values: list[Any] = []
        for key, column in allowed.items():
            if key not in changes:
                continue
            assignments.append(f"{column}=?")
            values.append(
                json.dumps(changes[key], sort_keys=True) if key == "metadata" else changes[key]
            )
        if not assignments:
            return self.get_person(collection_id, person_id)
        assignments.append("updated_at=?")
        values.extend([utc_now(), collection_id, person_id])
        with self.database.write() as connection:
            cursor = connection.execute(
                f"UPDATE persons SET {', '.join(assignments)} "
                "WHERE collection_id=? AND id=?",
                values,
            )
        return self.get_person(collection_id, person_id) if cursor.rowcount else None

    def delete_person(self, collection_id: str, person_id: str) -> tuple[bool, list[str]]:
        deleted, crops, _ = self.delete_person_with_mutation(collection_id, person_id)
        return deleted, crops

    def delete_person_with_mutation(
        self, collection_id: str, person_id: str
    ) -> tuple[bool, list[str], SearchMutation | None]:
        mutation: SearchMutation | None = None
        with self.database.write() as connection:
            crops = [
                row[0]
                for row in connection.execute(
                    "SELECT crop_path FROM face_samples WHERE collection_id=? "
                    "AND person_id=? AND crop_path IS NOT NULL",
                    (collection_id, person_id),
                )
            ]
            exists = connection.execute(
                "SELECT 1 FROM persons WHERE collection_id=? AND id=?",
                (collection_id, person_id),
            ).fetchone()
            if not exists:
                return False, [], None
            mappings = self._face_mappings_for_person(
                connection, collection_id, person_id
            )
            if mappings:
                revision = self._next_search_revision(connection, collection_id)
                mutation = self._record_search_changes(
                    connection,
                    collection_id,
                    revision,
                    [
                        ("delete", vector_id, person_numeric_id, face_id)
                        for vector_id, person_numeric_id, face_id in mappings
                    ],
                )
            cursor = connection.execute(
                "DELETE FROM persons WHERE collection_id=? AND id=?",
                (collection_id, person_id),
            )
        return cursor.rowcount > 0, crops, mutation

    def list_faces(
        self, collection_id: str, person_id: str, after: str, limit: int
    ) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                f"""SELECT {self._face_columns(include_embedding=False)}
                   FROM face_samples
                   WHERE collection_id=? AND person_id=? AND id>?
                   ORDER BY id LIMIT ?""",
                (collection_id, person_id, after, limit),
            ).fetchall()
        return [self._face_row(row, include_embedding=False) for row in rows]

    def list_face_embeddings(
        self, collection_id: str, person_id: str
    ) -> list[np.ndarray]:
        """Return private FP32 embeddings for one Person's enrollment review."""

        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT embedding FROM face_samples
                   WHERE collection_id=? AND person_id=? ORDER BY id""",
                (collection_id, person_id),
            ).fetchall()
        return [
            np.frombuffer(row["embedding"], dtype=np.float32).copy() for row in rows
        ]

    def get_face(
        self, collection_id: str, person_id: str, face_id: str
    ) -> dict[str, Any] | None:
        with self.database.read() as connection:
            row = connection.execute(
                f"""SELECT {self._face_columns(include_embedding=True)}
                    FROM face_samples
                    WHERE collection_id=? AND person_id=? AND id=?""",
                (collection_id, person_id, face_id),
            ).fetchone()
        return self._face_row(row) if row else None

    def get_face_crop(
        self, collection_id: str, person_id: str, face_id: str
    ) -> dict[str, Any] | None:
        """Return crop payload metadata without including it in normal face reads."""

        with self.database.read() as connection:
            row = connection.execute(
                """SELECT crop_image,crop_media_type,crop_path
                   FROM face_samples
                   WHERE collection_id=? AND person_id=? AND id=?""",
                (collection_id, person_id, face_id),
            ).fetchone()
        if row is None or (row["crop_image"] is None and row["crop_path"] is None):
            return None
        return {
            "bytes": (
                bytes(row["crop_image"]) if row["crop_image"] is not None else None
            ),
            "media_type": row["crop_media_type"] or "image/jpeg",
            "crop_path": row["crop_path"],
        }

    def delete_face(
        self, collection_id: str, person_id: str, face_id: str
    ) -> tuple[bool, str | None]:
        deleted, crop, _ = self.delete_face_with_mutation(
            collection_id, person_id, face_id
        )
        return deleted, crop

    def delete_face_with_mutation(
        self, collection_id: str, person_id: str, face_id: str
    ) -> tuple[bool, str | None, SearchMutation | None]:
        mutation: SearchMutation | None = None
        with self.database.write() as connection:
            row = connection.execute(
                """SELECT f.crop_path, v.numeric_id AS vector_id,
                          v.person_numeric_id
                   FROM face_samples f
                   JOIN search_vector_ids v ON v.face_id=f.id
                   WHERE f.collection_id=? AND f.person_id=? AND f.id=?""",
                (collection_id, person_id, face_id),
            ).fetchone()
            if not row:
                return False, None, None
            revision = self._next_search_revision(connection, collection_id)
            mutation = self._record_search_changes(
                connection,
                collection_id,
                revision,
                [
                    (
                        "delete",
                        int(row["vector_id"]),
                        int(row["person_numeric_id"]),
                        face_id,
                    )
                ],
            )
            connection.execute("DELETE FROM face_samples WHERE id=?", (face_id,))
            connection.execute(
                "UPDATE persons SET updated_at=? WHERE collection_id=? AND id=?",
                (utc_now(), collection_id, person_id),
            )
        return True, row["crop_path"], mutation

    def all_faces(self, collection_id: str) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                f"""SELECT {self._face_columns(include_embedding=True)}
                    FROM face_samples WHERE collection_id=? ORDER BY person_id,id""",
                (collection_id,),
            ).fetchall()
        return [self._face_row(row) for row in rows]

    @staticmethod
    def _ensure_person_numeric_id(
        connection: sqlite3.Connection, collection_id: str, person_id: str
    ) -> int:
        connection.execute(
            """INSERT OR IGNORE INTO search_person_ids(collection_id,person_id)
               VALUES(?,?)""",
            (collection_id, person_id),
        )
        row = connection.execute(
            """SELECT numeric_id FROM search_person_ids
               WHERE collection_id=? AND person_id=?""",
            (collection_id, person_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to allocate stable person search identifier")
        return int(row["numeric_id"])

    @staticmethod
    def _next_search_revision(
        connection: sqlite3.Connection, collection_id: str
    ) -> int:
        cursor = connection.execute(
            """UPDATE collections
               SET search_revision=search_revision+1, updated_at=?
               WHERE id=?""",
            (utc_now(), collection_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(collection_id)
        row = connection.execute(
            "SELECT search_revision FROM collections WHERE id=?", (collection_id,)
        ).fetchone()
        assert row is not None
        return int(row["search_revision"])

    @staticmethod
    def _record_search_changes(
        connection: sqlite3.Connection,
        collection_id: str,
        revision: int,
        changes: Sequence[tuple[str, int | None, int | None, str | None]],
    ) -> SearchMutation:
        recorded: list[SearchChange] = []
        created_at = utc_now()
        for operation, vector_id, person_numeric_id, face_id in changes:
            cursor = connection.execute(
                """INSERT INTO search_changes(
                       collection_id,revision,operation,vector_id,
                       person_numeric_id,face_id,created_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (
                    collection_id,
                    revision,
                    operation,
                    vector_id,
                    person_numeric_id,
                    face_id,
                    created_at,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("failed to allocate search change sequence")
            recorded.append(
                SearchChange(
                    seq=int(cursor.lastrowid),
                    collection_id=collection_id,
                    revision=revision,
                    operation=operation,
                    vector_id=vector_id,
                    person_numeric_id=person_numeric_id,
                    face_id=face_id,
                    created_at=created_at,
                )
            )
        return SearchMutation(collection_id, revision, tuple(recorded))

    @staticmethod
    def _face_mappings_for_person(
        connection: sqlite3.Connection, collection_id: str, person_id: str
    ) -> list[tuple[int, int, str]]:
        rows = connection.execute(
            """SELECT numeric_id,person_numeric_id,face_id
               FROM search_vector_ids
               WHERE collection_id=? AND person_id=? ORDER BY numeric_id""",
            (collection_id, person_id),
        ).fetchall()
        return [
            (int(row["numeric_id"]), int(row["person_numeric_id"]), row["face_id"])
            for row in rows
        ]

    @staticmethod
    def _check_face_limits(
        connection: sqlite3.Connection,
        collection_id: str,
        *,
        person_id: str | None,
        add_count: int,
    ) -> None:
        if add_count < 0:
            raise ValueError("add_count must not be negative")
        row = connection.execute(
            """SELECT capacity_rows,max_faces_per_person
               FROM collections WHERE id=?""",
            (collection_id,),
        ).fetchone()
        if row is None:
            raise KeyError(collection_id)
        collection_count = int(
            connection.execute(
                "SELECT count(*) FROM face_samples WHERE collection_id=?",
                (collection_id,),
            ).fetchone()[0]
        )
        desired_collection_count = collection_count + add_count
        capacity_rows = int(row["capacity_rows"])
        if desired_collection_count > capacity_rows:
            raise CollectionCapacityExceeded(
                capacity_rows=capacity_rows,
                requested_rows=desired_collection_count,
            )
        current_person_count = 0
        if person_id is not None:
            current_person_count = int(
                connection.execute(
                    """SELECT count(*) FROM face_samples
                       WHERE collection_id=? AND person_id=?""",
                    (collection_id, person_id),
                ).fetchone()[0]
            )
        desired_person_count = current_person_count + add_count
        max_faces = int(row["max_faces_per_person"])
        if desired_person_count > max_faces:
            raise MaxFacesPerPersonExceeded(
                max_faces=max_faces,
                requested_faces=desired_person_count,
            )

    @staticmethod
    def _search_change_row(row: sqlite3.Row) -> SearchChange:
        return SearchChange(
            seq=int(row["seq"]),
            collection_id=row["collection_id"],
            revision=int(row["revision"]),
            operation=row["operation"],
            vector_id=int(row["vector_id"]) if row["vector_id"] is not None else None,
            person_numeric_id=(
                int(row["person_numeric_id"])
                if row["person_numeric_id"] is not None
                else None
            ),
            face_id=row["face_id"],
            created_at=row["created_at"],
        )

    def get_search_revision(self, collection_id: str) -> int | None:
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT search_revision FROM collections WHERE id=?", (collection_id,)
            ).fetchone()
        return int(row["search_revision"]) if row else None

    def get_search_state(self, collection_id: str) -> SearchState | None:
        with self.database.read() as connection:
            row = connection.execute(
                """SELECT c.search_revision,
                          coalesce((SELECT max(s.seq) FROM search_changes s
                                    WHERE s.collection_id=c.id), 0) AS max_seq
                   FROM collections c WHERE c.id=?""",
                (collection_id,),
            ).fetchone()
        if row is None:
            return None
        return SearchState(collection_id, int(row["search_revision"]), int(row["max_seq"]))

    def get_search_changes(
        self, collection_id: str, after_seq: int = 0, limit: int = 1000
    ) -> list[SearchChange]:
        if after_seq < 0:
            raise ValueError("after_seq must not be negative")
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT * FROM search_changes
                   WHERE collection_id=? AND seq>?
                   ORDER BY seq LIMIT ?""",
                (collection_id, after_seq, limit),
            ).fetchall()
        return [self._search_change_row(row) for row in rows]

    def acknowledge_search_changes(
        self, collection_id: str, through_seq: int
    ) -> int:
        """Delete durable index events through a globally monotonic sequence."""

        if through_seq < 0:
            raise ValueError("through_seq must not be negative")
        with self.database.write() as connection:
            cursor = connection.execute(
                """DELETE FROM search_changes
                   WHERE collection_id=? AND seq<=?""",
                (collection_id, through_seq),
            )
        return int(cursor.rowcount)

    def reset_search_changes_before_full_rebuild(self) -> int:
        """Discard stale events when no in-memory generation survives startup."""

        with self.database.write() as connection:
            cursor = connection.execute("DELETE FROM search_changes")
        return int(cursor.rowcount)

    def iter_index_faces(
        self, collection_id: str, batch_size: int = 1000
    ) -> Iterator[IndexFaceBatch]:
        """Stream an index build from one WAL snapshot without materializing it."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        with self.database.read() as connection:
            # Explicit BEGIN holds one stable read snapshot while WAL writers continue.
            connection.execute("BEGIN")
            collection = connection.execute(
                """SELECT c.search_revision,
                          coalesce((SELECT max(s.seq) FROM search_changes s
                                    WHERE s.collection_id=c.id), 0) AS max_seq
                   FROM collections c WHERE c.id=?""",
                (collection_id,),
            ).fetchone()
            if collection is None:
                raise KeyError(collection_id)
            revision = int(collection["search_revision"])
            max_seq = int(collection["max_seq"])
            cursor = connection.execute(
                """SELECT v.numeric_id AS vector_id,v.person_numeric_id,
                          v.face_id,v.person_id,f.embedding
                   FROM search_vector_ids v
                   JOIN face_samples f ON f.id=v.face_id
                   WHERE v.collection_id=? ORDER BY v.numeric_id""",
                (collection_id,),
            )
            emitted = False
            while rows := cursor.fetchmany(batch_size):
                emitted = True
                yield IndexFaceBatch(
                    revision,
                    tuple(
                        IndexFace(
                            vector_id=int(row["vector_id"]),
                            person_numeric_id=int(row["person_numeric_id"]),
                            face_id=row["face_id"],
                            person_id=row["person_id"],
                            embedding=np.frombuffer(row["embedding"], dtype=np.float32).copy(),
                        )
                        for row in rows
                    ),
                    max_seq,
                )
            if not emitted:
                yield IndexFaceBatch(revision, (), max_seq)

    def get_index_faces_by_vector_ids(
        self, collection_id: str, vector_ids: Sequence[int]
    ) -> dict[int, IndexFace]:
        result: dict[int, IndexFace] = {}
        unique_ids = list(dict.fromkeys(int(value) for value in vector_ids))
        with self.database.read() as connection:
            for offset in range(0, len(unique_ids), 500):
                chunk = unique_ids[offset : offset + 500]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""SELECT v.numeric_id AS vector_id,v.person_numeric_id,
                               v.face_id,v.person_id,f.embedding
                        FROM search_vector_ids v
                        JOIN face_samples f ON f.id=v.face_id
                        WHERE v.collection_id=?
                          AND v.numeric_id IN ({placeholders})""",
                    [collection_id, *chunk],
                ).fetchall()
                for row in rows:
                    vector_id = int(row["vector_id"])
                    result[vector_id] = IndexFace(
                        vector_id=vector_id,
                        person_numeric_id=int(row["person_numeric_id"]),
                        face_id=row["face_id"],
                        person_id=row["person_id"],
                        embedding=np.frombuffer(row["embedding"], dtype=np.float32).copy(),
                    )
        return result

    def get_index_face_mappings(
        self,
        collection_id: str,
        vector_ids: Sequence[int],
        *,
        include_embedding: bool = False,
    ) -> dict[int, dict[str, Any]]:
        """Hydrate native results in batches, avoiding one query per Person."""

        result: dict[int, dict[str, Any]] = {}
        unique_ids = list(dict.fromkeys(int(value) for value in vector_ids))
        with self.database.read() as connection:
            for offset in range(0, len(unique_ids), 400):
                chunk = unique_ids[offset : offset + 400]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""SELECT v.numeric_id AS vector_id,v.person_numeric_id,
                               v.face_id,v.person_id,
                               p.name,p.external_id,p.metadata_json,
                               p.created_at AS person_created_at,
                               p.updated_at AS person_updated_at,
                               (SELECT count(*) FROM face_samples pf
                                WHERE pf.collection_id=p.collection_id
                                  AND pf.person_id=p.id) AS face_count,
                               f.embedding
                        FROM search_vector_ids v
                        JOIN persons p ON p.collection_id=v.collection_id
                                      AND p.id=v.person_id
                        JOIN face_samples f ON f.id=v.face_id
                        WHERE v.collection_id=?
                          AND v.numeric_id IN ({placeholders})""",
                    [collection_id, *chunk],
                ).fetchall()
                for row in rows:
                    vector_id = int(row["vector_id"])
                    value: dict[str, Any] = {
                        "vector_id": vector_id,
                        "person_numeric_id": int(row["person_numeric_id"]),
                        "face_id": row["face_id"],
                        "person_id": row["person_id"],
                        "person": {
                            "id": row["person_id"],
                            "name": row["name"],
                            "external_id": row["external_id"],
                            "metadata": json.loads(row["metadata_json"]),
                            "face_count": int(row["face_count"]),
                            "created_at": row["person_created_at"],
                            "updated_at": row["person_updated_at"],
                        },
                    }
                    if include_embedding:
                        value["embedding"] = np.frombuffer(
                            row["embedding"], dtype=np.float32
                        ).copy()
                    result[vector_id] = value
        return result

    def stats(self) -> dict[str, int]:
        with self.database.read() as connection:
            collection_count = connection.execute("SELECT count(*) FROM collections").fetchone()[0]
            person_count = connection.execute("SELECT count(*) FROM persons").fetchone()[0]
            face_count = connection.execute("SELECT count(*) FROM face_samples").fetchone()[0]
        return {
            "collection_count": int(collection_count),
            "person_count": int(person_count),
            "face_count": int(face_count),
        }

    def sync_api_key(self, raw_key: str) -> str:
        """Make ``raw_key`` the only active key, retaining inactive history.

        The comparison and any replacement happen inside one SQLite write
        transaction. Repeating the same startup key is a true no-op: its row,
        salt, digest, and creation timestamp remain unchanged.
        """

        if not raw_key:
            raise ValueError("API key must not be empty")
        with self.database.write() as connection:
            rows = connection.execute(
                "SELECT id,salt,digest FROM api_keys WHERE active=1 ORDER BY created_at,id"
            ).fetchall()
            matching_id: str | None = None
            for row in rows:
                candidate = hashlib.scrypt(
                    raw_key.encode(), salt=row["salt"], n=2**14, r=8, p=1, dklen=32
                )
                if hmac.compare_digest(candidate, row["digest"]):
                    matching_id = str(row["id"])
                    break
            if matching_id is not None and len(rows) == 1:
                return "unchanged"
            if matching_id is not None:
                connection.execute(
                    "UPDATE api_keys SET active=0 WHERE active=1 AND id<>?",
                    (matching_id,),
                )
                return "rotated"

            salt = os.urandom(16)
            digest = hashlib.scrypt(
                raw_key.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32
            )
            result = "created" if not rows else "rotated"
            connection.execute("UPDATE api_keys SET active=0 WHERE active=1")
            connection.execute(
                "INSERT INTO api_keys(id,label,salt,digest,active,created_at) "
                "VALUES(?,?,?,?,1,?)",
                (str(uuid.uuid4()), "startup", salt, digest, utc_now()),
            )
            return result

    def has_api_keys(self) -> bool:
        with self.database.read() as connection:
            return bool(
                connection.execute("SELECT 1 FROM api_keys WHERE active=1 LIMIT 1").fetchone()
            )

    def verify_api_key(self, raw_key: str) -> bool:
        with self.database.read() as connection:
            rows = connection.execute(
                "SELECT salt,digest FROM api_keys WHERE active=1"
            ).fetchall()
        for row in rows:
            candidate = hashlib.scrypt(
                raw_key.encode(), salt=row["salt"], n=2**14, r=8, p=1, dklen=32
            )
            if hmac.compare_digest(candidate, row["digest"]):
                return True
        return False
