"""SQLite storage for local people, media, embeddings, and reports."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

from .logging import get_logger
from .constants import DEFAULT_THRESHOLD
from .recognition import search_gallery
from .utils import safe_json_dumps, utc_now_iso

LOGGER = get_logger("storage")
ALBUM_DIRECTORIES_SETTING = "album.directories"
ALBUM_RESULTS_SETTING = "album.results"


def embedding_to_blob(embedding: np.ndarray | Iterable[float] | None) -> tuple[Optional[bytes], int]:
    if embedding is None:
        return None, 0
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return arr.tobytes(), int(arr.shape[0])


def blob_to_embedding(blob: bytes | memoryview | None, dim: int | None = None) -> Optional[np.ndarray]:
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if dim and arr.shape[0] != dim:
        return None
    return arr


class Storage:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.database_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            LOGGER.exception("Database operation failed")
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    display_name TEXT,
                    notes TEXT,
                    tags TEXT,
                    cover_face_sample_id INTEGER,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS face_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    source_image_path TEXT,
                    crop_path TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    bbox_json TEXT,
                    kps_json TEXT,
                    det_score REAL,
                    quality_score REAL,
                    model_name TEXT,
                    provider TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS media_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE,
                    media_type TEXT,
                    width INTEGER,
                    height INTEGER,
                    duration_ms INTEGER,
                    file_size INTEGER,
                    mtime REAL,
                    sha256 TEXT,
                    thumbnail BLOB,
                    thumbnail_mime TEXT,
                    processed_at TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS media_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id INTEGER,
                    frame_index INTEGER,
                    timestamp_ms INTEGER,
                    crop_path TEXT,
                    thumbnail BLOB,
                    thumbnail_mime TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    model_name TEXT,
                    bbox_json TEXT,
                    kps_json TEXT,
                    det_score REAL,
                    quality_score REAL,
                    assigned_person_id INTEGER,
                    predicted_person_id INTEGER,
                    similarity REAL,
                    cluster_id INTEGER,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS media_analysis (
                    media_id INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    processed_at TEXT,
                    PRIMARY KEY (media_id, model_name)
                );

                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    cover_media_face_id INTEGER,
                    notes TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT,
                    source_path TEXT,
                    query_face_sample_id INTEGER,
                    predicted_person_id INTEGER,
                    similarity REAL,
                    threshold REAL,
                    status TEXT,
                    payload_json TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario TEXT,
                    model_name TEXT,
                    provider TEXT,
                    threshold REAL,
                    dataset_summary_json TEXT,
                    metrics_json TEXT,
                    hardware_json TEXT,
                    report_path TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
                """
            )
            self._ensure_column(conn, "media_items", "thumbnail", "BLOB")
            self._ensure_column(conn, "media_items", "thumbnail_mime", "TEXT")
            self._ensure_column(conn, "media_faces", "thumbnail", "BLOB")
            self._ensure_column(conn, "media_faces", "thumbnail_mime", "TEXT")
            self._ensure_column(conn, "media_faces", "model_name", "TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_faces_model_media "
                "ON media_faces(model_name, media_id)"
            )

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(row["name"]) for row in rows}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def add_person(
        self,
        name: str,
        display_name: Optional[str] = None,
        notes: str = "",
        tags: str = "",
    ) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO people (name, display_name, notes, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, display_name or name, notes, tags, now, now),
            )
            return int(cur.lastrowid)

    def update_person(self, person_id: int, **fields: Any) -> None:
        allowed = {"name", "display_name", "notes", "tags", "cover_face_sample_id"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = utc_now_iso()
        clause = ", ".join(f"{key}=?" for key in updates)
        values = list(updates.values()) + [person_id]
        with self.connect() as conn:
            conn.execute(f"UPDATE people SET {clause} WHERE id=?", values)

    def delete_person(self, person_id: int, delete_samples: bool = False) -> None:
        with self.connect() as conn:
            if delete_samples:
                conn.execute("DELETE FROM face_samples WHERE person_id=?", (person_id,))
            else:
                conn.execute("UPDATE face_samples SET person_id=NULL WHERE person_id=?", (person_id,))
            conn.execute("UPDATE media_faces SET assigned_person_id=NULL WHERE assigned_person_id=?", (person_id,))
            conn.execute("DELETE FROM people WHERE id=?", (person_id,))

    def merge_people(self, source_person_id: int, target_person_id: int) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "UPDATE face_samples SET person_id=?, updated_at=? WHERE person_id=?",
                (target_person_id, now, source_person_id),
            )
            conn.execute(
                "UPDATE media_faces SET assigned_person_id=? WHERE assigned_person_id=?",
                (target_person_id, source_person_id),
            )
            conn.execute("DELETE FROM people WHERE id=?", (source_person_id,))

    def add_face_sample(
        self,
        person_id: Optional[int],
        embedding: np.ndarray | Iterable[float] | None,
        source_image_path: str = "",
        crop_path: str = "",
        bbox: Optional[Iterable[float]] = None,
        kps: Optional[Iterable[Iterable[float]]] = None,
        det_score: float = 0.0,
        quality_score: Optional[float] = None,
        model_name: str = "",
        provider: str = "",
    ) -> int:
        blob, dim = embedding_to_blob(embedding)
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO face_samples (
                    person_id, source_image_path, crop_path, embedding, embedding_dim,
                    bbox_json, kps_json, det_score, quality_score, model_name, provider,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    source_image_path,
                    crop_path,
                    blob,
                    dim,
                    safe_json_dumps(list(bbox) if bbox is not None else None),
                    safe_json_dumps(list(kps) if kps is not None else None),
                    det_score,
                    quality_score,
                    model_name,
                    provider,
                    now,
                    now,
                ),
            )
            sample_id = int(cur.lastrowid)
            if person_id is not None:
                conn.execute(
                    """
                    UPDATE people
                    SET cover_face_sample_id=COALESCE(cover_face_sample_id, ?), updated_at=?
                    WHERE id=?
                    """,
                    (sample_id, now, person_id),
                )
            return sample_id

    def delete_face_sample(self, sample_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM face_samples WHERE id=?", (sample_id,))

    def list_people(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*,
                       COUNT(fs.id) AS sample_count,
                       AVG(fs.quality_score) AS avg_quality,
                       COALESCE(
                           (
                               SELECT fs_cover.crop_path
                               FROM face_samples fs_cover
                               WHERE fs_cover.id = p.cover_face_sample_id
                                 AND COALESCE(fs_cover.crop_path, '') <> ''
                               LIMIT 1
                           ),
                           (
                               SELECT fs2.crop_path
                               FROM face_samples fs2
                               WHERE fs2.person_id = p.id AND COALESCE(fs2.crop_path, '') <> ''
                               ORDER BY fs2.id ASC
                               LIMIT 1
                           )
                       ) AS cover_crop_path
                FROM people p
                LEFT JOIN face_samples fs ON fs.person_id = p.id
                GROUP BY p.id
                ORDER BY LOWER(COALESCE(p.display_name, p.name))
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def list_face_samples(self, person_id: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM face_samples"
        params: tuple[Any, ...] = ()
        if person_id is not None:
            query += " WHERE person_id=?"
            params = (person_id,)
        query += " ORDER BY created_at DESC, id DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._decode_sample(row) for row in rows]

    def _decode_sample(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["embedding"] = blob_to_embedding(data.get("embedding"), data.get("embedding_dim"))
        for key in ("bbox_json", "kps_json"):
            try:
                data[key.replace("_json", "")] = json.loads(data[key]) if data.get(key) else None
            except Exception:
                data[key.replace("_json", "")] = None
        return data

    def add_media_item(
        self,
        path: str,
        media_type: str = "image",
        width: Optional[int] = None,
        height: Optional[int] = None,
        duration_ms: Optional[int] = None,
        file_size: Optional[int] = None,
        mtime: Optional[float] = None,
        sha256: Optional[str] = None,
        thumbnail: Optional[bytes] = None,
        thumbnail_mime: str = "image/webp",
        processed_at: Optional[str] = None,
    ) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO media_items (
                    path, media_type, width, height, duration_ms, file_size,
                    mtime, sha256, thumbnail, thumbnail_mime, processed_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path,
                    media_type,
                    width,
                    height,
                    duration_ms,
                    file_size,
                    mtime,
                    sha256,
                    thumbnail,
                    thumbnail_mime if thumbnail else "",
                    processed_at,
                    now,
                ),
            )
            row = conn.execute("SELECT id FROM media_items WHERE path=?", (path,)).fetchone()
            return int(row["id"])

    def add_media_face(
        self,
        media_id: int,
        embedding: np.ndarray | Iterable[float] | None,
        crop_path: str = "",
        thumbnail: Optional[bytes] = None,
        thumbnail_mime: str = "image/webp",
        bbox: Optional[Iterable[float]] = None,
        kps: Optional[Iterable[Iterable[float]]] = None,
        det_score: float = 0.0,
        quality_score: Optional[float] = None,
        frame_index: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
        assigned_person_id: Optional[int] = None,
        predicted_person_id: Optional[int] = None,
        similarity: Optional[float] = None,
        cluster_id: Optional[int] = None,
        status: str = "unknown",
        model_name: str = "",
    ) -> int:
        blob, dim = embedding_to_blob(embedding)
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO media_faces (
                    media_id, frame_index, timestamp_ms, crop_path, thumbnail, thumbnail_mime, embedding, embedding_dim,
                    model_name, bbox_json, kps_json, det_score, quality_score, assigned_person_id,
                    predicted_person_id, similarity, cluster_id, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    media_id,
                    frame_index,
                    timestamp_ms,
                    crop_path,
                    thumbnail,
                    thumbnail_mime if thumbnail else "",
                    blob,
                    dim,
                    str(model_name or ""),
                    safe_json_dumps(list(bbox) if bbox is not None else None),
                    safe_json_dumps(list(kps) if kps is not None else None),
                    det_score,
                    quality_score,
                    assigned_person_id,
                    predicted_person_id,
                    similarity,
                    cluster_id,
                    status,
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def mark_media_item_processed(self, media_id: int, model_name: str) -> None:
        """Record that a media item was analyzed, including zero-face results."""

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO media_analysis (media_id, model_name, processed_at)
                VALUES (?, ?, ?)
                ON CONFLICT(media_id, model_name)
                DO UPDATE SET processed_at=excluded.processed_at
                """,
                (int(media_id), str(model_name), utc_now_iso()),
            )

    def load_all_gallery_embeddings(
        self,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = """
                SELECT fs.id AS sample_id, fs.person_id, fs.crop_path, fs.embedding,
                       fs.embedding_dim, fs.model_name,
                       COALESCE(p.display_name, p.name, 'Unknown') AS person_name
                FROM face_samples fs
                LEFT JOIN people p ON p.id = fs.person_id
                WHERE fs.embedding IS NOT NULL AND fs.person_id IS NOT NULL
                """
        params: tuple[Any, ...] = ()
        if model_name is not None:
            query += " AND fs.model_name=?"
            params = (str(model_name),)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            gallery = []
            for row in rows:
                item = dict(row)
                item["embedding"] = blob_to_embedding(item.get("embedding"), item.get("embedding_dim"))
                gallery.append(item)
            return gallery

    def search_embeddings(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: Optional[str] = None,
    ):
        return search_gallery(
            query_embedding,
            self.load_all_gallery_embeddings(model_name=model_name),
            top_k=top_k,
            threshold=threshold,
        )

    def log_recognition(
        self,
        source_type: str,
        source_path: str,
        predicted_person_id: Optional[int],
        similarity: float,
        threshold: float,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        query_face_sample_id: Optional[int] = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO recognition_logs (
                    source_type, source_path, query_face_sample_id, predicted_person_id,
                    similarity, threshold, status, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_type,
                    source_path,
                    query_face_sample_id,
                    predicted_person_id,
                    similarity,
                    threshold,
                    status,
                    safe_json_dumps(payload or {}),
                    utc_now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def save_evaluation_run(
        self,
        scenario: str,
        model_name: str,
        provider: str,
        threshold: float,
        dataset_summary: Dict[str, Any],
        metrics: Dict[str, Any],
        hardware: Dict[str, Any],
        report_path: str = "",
        created_at: Optional[str] = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO evaluation_runs (
                    scenario, model_name, provider, threshold, dataset_summary_json,
                    metrics_json, hardware_json, report_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario,
                    model_name,
                    provider,
                    threshold,
                    safe_json_dumps(dataset_summary),
                    safe_json_dumps(metrics),
                    safe_json_dumps(hardware),
                    report_path,
                    created_at or utc_now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def list_evaluation_runs(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM evaluation_runs ORDER BY created_at DESC, id DESC").fetchall()
            result = []
            for row in rows:
                item = dict(row)
                for key in ("dataset_summary_json", "metrics_json", "hardware_json"):
                    try:
                        item[key.replace("_json", "")] = json.loads(item.get(key) or "{}")
                    except Exception:
                        item[key.replace("_json", "")] = {}
                result.append(item)
            return result

    def delete_evaluation_run(self, run_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM evaluation_runs WHERE id=?", (run_id,))

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, str(value), utc_now_iso()),
            )

    def delete_setting(self, key: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM app_settings WHERE key=?", (key,))

    def list_album_directories(self) -> List[str]:
        raw = self.get_setting(ALBUM_DIRECTORIES_SETTING, "[]")
        try:
            values = json.loads(raw or "[]")
        except Exception:
            return []
        if not isinstance(values, list):
            return []
        directories = []
        seen = set()
        for value in values:
            path = str(value)
            if path and path not in seen:
                directories.append(path)
                seen.add(path)
        return directories

    def save_album_directories(self, directories: Iterable[str]) -> None:
        values = []
        seen = set()
        for directory in directories:
            path = str(directory)
            if path and path not in seen:
                values.append(path)
                seen.add(path)
        self.set_setting(ALBUM_DIRECTORIES_SETTING, json.dumps(values, ensure_ascii=False))

    def save_album_results(
        self,
        clusters: Iterable[Dict[str, Any]],
        cluster_items: Dict[int, List[Dict[str, Any]]],
        algorithm: str,
        cluster_threshold: Optional[float] = None,
        min_samples: int = 2,
        min_face_size: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> None:
        now = utc_now_iso()
        serializable_clusters: List[Dict[str, Any]] = []
        assignments: List[tuple[int, int]] = []
        for cluster in clusters:
            cluster_id = int(cluster["id"])
            items = cluster_items.get(cluster_id, [])
            face_ids = [int(item["id"]) for item in items if item.get("id") is not None]
            for face_id in face_ids:
                assignments.append((cluster_id, face_id))
            data = {
                "id": cluster_id,
                "label": cluster.get("label"),
                "name": cluster.get("name", f"Album Person {cluster_id}"),
                "source": cluster.get("source", "album"),
                "face_count": int(cluster.get("face_count") or len(face_ids)),
                "photo_count": int(cluster.get("photo_count") or 0),
                "avg_quality": float(cluster.get("avg_quality") or 0.0),
                "thumbnail_path": cluster.get("thumbnail_path") or "",
                "thumbnail_face_id": cluster.get("thumbnail_face_id"),
                "photos": list(cluster.get("photos") or []),
                "face_ids": face_ids,
            }
            serializable_clusters.append(data)
        payload = {
            "version": 1,
            "model_name": str(model_name) if model_name is not None else None,
            "algorithm": algorithm,
            "cluster_threshold": float(cluster_threshold) if cluster_threshold is not None else None,
            "min_samples": int(min_samples),
            "min_face_size": int(min_face_size) if min_face_size is not None else None,
            "clusters": serializable_clusters,
            "updated_at": now,
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (
                    self._album_results_setting(model_name),
                    json.dumps(payload, ensure_ascii=False),
                    now,
                ),
            )
            if model_name is None:
                conn.execute(
                    "UPDATE media_faces SET cluster_id=NULL, updated_at=?",
                    (now,),
                )
                conn.executemany(
                    "UPDATE media_faces SET cluster_id=?, status='clustered', updated_at=? WHERE id=?",
                    [
                        (cluster_id, now, face_id)
                        for cluster_id, face_id in assignments
                    ],
                )
            else:
                normalized_model_name = str(model_name)
                conn.execute(
                    "UPDATE media_faces SET cluster_id=NULL, updated_at=? "
                    "WHERE model_name=?",
                    (now, normalized_model_name),
                )
                conn.executemany(
                    "UPDATE media_faces SET cluster_id=?, status='clustered', updated_at=? "
                    "WHERE id=? AND model_name=?",
                    [
                        (cluster_id, now, face_id, normalized_model_name)
                        for cluster_id, face_id in assignments
                    ],
                )

    @staticmethod
    def _album_results_setting(model_name: Optional[str]) -> str:
        if model_name is None:
            return ALBUM_RESULTS_SETTING
        return f"{ALBUM_RESULTS_SETTING}.{model_name}"

    def load_album_results(
        self,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw = self.get_setting(self._album_results_setting(model_name), "{}")
        try:
            data = json.loads(raw or "{}")
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        if model_name is not None and data.get("model_name") != str(model_name):
            return {}
        return data

    def clear_album_results(self, model_name: Optional[str] = None) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM app_settings WHERE key=?",
                (self._album_results_setting(model_name),),
            )
            if model_name is None:
                conn.execute(
                    "UPDATE media_faces SET cluster_id=NULL, updated_at=?",
                    (now,),
                )
            else:
                conn.execute(
                    "UPDATE media_faces SET cluster_id=NULL, updated_at=? "
                    "WHERE model_name=?",
                    (now, str(model_name)),
                )

    def delete_media_items_by_paths(self, paths: Iterable[str]) -> int:
        values = [str(path) for path in paths]
        if not values:
            return 0
        deleted = 0
        with self.connect() as conn:
            for index in range(0, len(values), 900):
                chunk = values[index : index + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(f"SELECT id FROM media_items WHERE path IN ({placeholders})", chunk).fetchall()
                media_ids = [int(row["id"]) for row in rows]
                if not media_ids:
                    continue
                id_placeholders = ",".join("?" for _ in media_ids)
                conn.execute(f"DELETE FROM media_faces WHERE media_id IN ({id_placeholders})", media_ids)
                conn.execute(
                    f"DELETE FROM media_analysis WHERE media_id IN ({id_placeholders})",
                    media_ids,
                )
                conn.execute(f"DELETE FROM media_items WHERE id IN ({id_placeholders})", media_ids)
                deleted += len(media_ids)
        return deleted

    def delete_media_faces_by_paths(
        self,
        paths: Iterable[str],
        *,
        model_name: str,
    ) -> int:
        """Delete one model's indexed faces without touching other model data."""

        values = [str(path) for path in paths]
        if not values:
            return 0
        affected_paths: set[str] = set()
        with self.connect() as conn:
            for index in range(0, len(values), 899):
                chunk = values[index : index + 899]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT mi.path
                    FROM media_items mi
                    WHERE mi.path IN ({placeholders})
                      AND (
                          EXISTS (
                              SELECT 1 FROM media_analysis ma
                              WHERE ma.media_id=mi.id AND ma.model_name=?
                          )
                          OR EXISTS (
                              SELECT 1 FROM media_faces mf
                              WHERE mf.media_id=mi.id AND mf.model_name=?
                          )
                      )
                    """,
                    [*chunk, str(model_name), str(model_name)],
                ).fetchall()
                affected_paths.update(str(row["path"]) for row in rows)
                conn.execute(
                    f"""
                    DELETE FROM media_faces
                    WHERE model_name=? AND media_id IN (
                        SELECT id FROM media_items WHERE path IN ({placeholders})
                    )
                    """,
                    [str(model_name), *chunk],
                )
                conn.execute(
                    f"""
                    DELETE FROM media_analysis
                    WHERE model_name=? AND media_id IN (
                        SELECT id FROM media_items WHERE path IN ({placeholders})
                    )
                    """,
                    [str(model_name), *chunk],
                )
        return len(affected_paths)

    def existing_media_paths(
        self,
        paths: Iterable[str],
        model_name: Optional[str] = None,
    ) -> set[str]:
        values = [str(path) for path in paths]
        if not values:
            return set()
        existing: set[str] = set()
        with self.connect() as conn:
            for index in range(0, len(values), 900):
                chunk = values[index : index + 900]
                placeholders = ",".join("?" for _ in chunk)
                if model_name is None:
                    rows = conn.execute(
                        f"SELECT path FROM media_items WHERE path IN ({placeholders})",
                        chunk,
                    ).fetchall()
                else:
                    rows = conn.execute(
                        f"""
                        SELECT mi.path
                        FROM media_items mi
                        WHERE mi.path IN ({placeholders})
                          AND (
                              EXISTS (
                                  SELECT 1 FROM media_analysis ma
                                  WHERE ma.media_id=mi.id AND ma.model_name=?
                              )
                              OR EXISTS (
                                  SELECT 1 FROM media_faces mf
                                  WHERE mf.media_id=mi.id
                                    AND mf.embedding IS NOT NULL
                                    AND mf.model_name=?
                              )
                          )
                        """,
                        [*chunk, str(model_name), str(model_name)],
                    ).fetchall()
                existing.update(str(row["path"]) for row in rows)
        return existing

    def list_media_faces(
        self,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = """
                SELECT mf.*, mi.path AS media_path, mi.media_type, mi.width, mi.height,
                       mi.thumbnail AS media_thumbnail,
                       mi.thumbnail_mime AS media_thumbnail_mime
                FROM media_faces mf
                JOIN media_items mi ON mi.id = mf.media_id
                WHERE mf.embedding IS NOT NULL
                """
        params: tuple[Any, ...] = ()
        if model_name is not None:
            query += " AND mf.model_name=?"
            params = (str(model_name),)
        query += " ORDER BY mf.created_at DESC, mf.id DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["embedding"] = blob_to_embedding(item.get("embedding"), item.get("embedding_dim"))
                for key in ("bbox_json", "kps_json"):
                    try:
                        item[key.replace("_json", "")] = json.loads(item[key]) if item.get(key) else None
                    except Exception:
                        item[key.replace("_json", "")] = None
                result.append(item)
            return result

    def counts(self) -> Dict[str, int]:
        with self.connect() as conn:
            return {
                "people": int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]),
                "face_samples": int(conn.execute("SELECT COUNT(*) FROM face_samples").fetchone()[0]),
                "media_items": int(conn.execute("SELECT COUNT(*) FROM media_items").fetchone()[0]),
                "media_faces": int(conn.execute("SELECT COUNT(*) FROM media_faces").fetchone()[0]),
                "evaluation_runs": int(conn.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0]),
            }
