import numpy as np
import sqlite3

from insightface.gui.core.storage import Storage
from insightface.gui.core.utils import encode_webp_thumbnail


def test_storage_people_samples_and_search(tmp_path):
    db = tmp_path / "test.db"
    storage = Storage(db)
    person_id = storage.add_person("Alice")
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sample_id = storage.add_face_sample(person_id, emb, source_image_path="a.jpg", det_score=0.9)
    people = storage.list_people()
    assert people[0]["name"] == "Alice"
    assert people[0]["sample_count"] == 1
    samples = storage.list_face_samples(person_id)
    assert samples[0]["id"] == sample_id
    assert np.allclose(samples[0]["embedding"], emb)
    results = storage.search_embeddings(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1, threshold=0.5)
    assert results[0].person_id == person_id
    assert results[0].status == "matched"


def test_gallery_search_isolates_embeddings_by_model_package(tmp_path):
    storage = Storage(tmp_path / "test.db")
    raccoon_person = storage.add_person("Raccoon Person")
    buffalo_person = storage.add_person("Buffalo Person")
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    storage.add_face_sample(
        raccoon_person,
        embedding,
        model_name="raccoon_s",
    )
    storage.add_face_sample(
        buffalo_person,
        embedding,
        model_name="buffalo_l",
    )

    raccoon_gallery = storage.load_all_gallery_embeddings(
        model_name="raccoon_s"
    )
    assert [item["person_id"] for item in raccoon_gallery] == [raccoon_person]
    assert raccoon_gallery[0]["model_name"] == "raccoon_s"

    results = storage.search_embeddings(
        embedding,
        top_k=5,
        threshold=0.5,
        model_name="buffalo_l",
    )
    assert [result.person_id for result in results] == [buffalo_person]


def test_album_directories_and_results_persist(tmp_path):
    db = tmp_path / "test.db"
    storage = Storage(db)
    album_dir = tmp_path / "album"
    album_dir.mkdir()
    image_path = album_dir / "a.jpg"
    image_path.write_bytes(b"placeholder")
    photo_thumb = b"photo-webp"
    face_thumb = b"face-webp"

    storage.save_album_directories([str(album_dir)])
    assert storage.list_album_directories() == [str(album_dir)]

    media_id = storage.add_media_item(str(image_path), "image", thumbnail=photo_thumb)
    face_id = storage.add_media_face(
        media_id,
        np.array([1.0, 0.0], dtype=np.float32),
        thumbnail=face_thumb,
    )
    cluster = {
        "id": 1,
        "label": 0,
        "name": "Album Person 1",
        "source": "album",
        "face_count": 1,
        "photo_count": 1,
        "avg_quality": 0.0,
        "thumbnail_face_id": face_id,
        "thumbnail_path": "",
        "photos": [str(image_path)],
    }
    storage.save_album_results(
        [cluster],
        {1: [{"id": face_id, "media_path": str(image_path)}]},
        "DBSCAN",
        cluster_threshold=0.28,
        min_samples=2,
        min_face_size=80,
    )

    results = storage.load_album_results()
    assert results["algorithm"] == "DBSCAN"
    assert results["cluster_threshold"] == 0.28
    assert "duplicate_threshold" not in results
    assert results["min_face_size"] == 80
    assert results["clusters"][0]["thumbnail_face_id"] == face_id
    assert results["clusters"][0]["face_ids"] == [face_id]
    face = storage.list_media_faces()[0]
    assert face["cluster_id"] == 1
    assert face["thumbnail"] == face_thumb
    assert face["thumbnail_mime"] == "image/webp"
    assert face["media_thumbnail"] == photo_thumb
    assert face["media_thumbnail_mime"] == "image/webp"

    storage.clear_album_results()
    assert storage.load_album_results() == {}
    assert storage.list_media_faces()[0]["cluster_id"] is None


def test_media_faces_schema_migrates_without_rewriting_legacy_rows(tmp_path):
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE media_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER,
                embedding BLOB,
                embedding_dim INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO media_faces (media_id, embedding, embedding_dim) VALUES (?, ?, ?)",
            (7, np.array([1.0, 0.0], dtype=np.float32).tobytes(), 2),
        )

    Storage(db)

    with sqlite3.connect(db) as conn:
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(media_faces)")
        }
        row = conn.execute(
            "SELECT media_id, model_name FROM media_faces WHERE id=1"
        ).fetchone()
        media_analysis_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='media_analysis'"
        ).fetchone()
    assert "model_name" in columns
    assert row == (7, None)
    assert media_analysis_exists == (1,)


def test_album_storage_isolates_index_and_saved_results_by_model(tmp_path):
    storage = Storage(tmp_path / "test.db")
    first_path = str(tmp_path / "first.jpg")
    second_path = str(tmp_path / "second.jpg")
    no_face_path = str(tmp_path / "no-face.jpg")
    first_media_id = storage.add_media_item(first_path, "image")
    second_media_id = storage.add_media_item(second_path, "image")
    no_face_media_id = storage.add_media_item(no_face_path, "image")
    old_face_id = storage.add_media_face(
        first_media_id,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    raccoon_face_id = storage.add_media_face(
        first_media_id,
        np.array([1.0, 0.0], dtype=np.float32),
        model_name="raccoon_s",
    )
    buffalo_face_id = storage.add_media_face(
        second_media_id,
        np.array([0.0, 1.0], dtype=np.float32),
        model_name="buffalo_l",
    )
    storage.mark_media_item_processed(no_face_media_id, "raccoon_s")

    assert storage.existing_media_paths(
        [first_path, second_path, no_face_path], model_name="raccoon_s"
    ) == {first_path, no_face_path}
    assert storage.existing_media_paths(
        [first_path, second_path], model_name="buffalo_l"
    ) == {second_path}
    assert storage.existing_media_paths(
        [first_path, second_path], model_name="missing_model"
    ) == set()
    assert {
        face["id"] for face in storage.list_media_faces(model_name="raccoon_s")
    } == {raccoon_face_id}
    assert old_face_id not in {
        face["id"] for face in storage.list_media_faces(model_name="raccoon_s")
    }

    raccoon_cluster = {
        "id": 1,
        "name": "Raccoon cluster",
        "face_count": 1,
        "photo_count": 1,
    }
    buffalo_cluster = {
        "id": 1,
        "name": "Buffalo cluster",
        "face_count": 1,
        "photo_count": 1,
    }
    storage.save_album_results(
        [raccoon_cluster],
        {1: [{"id": raccoon_face_id}]},
        "DBSCAN",
        model_name="raccoon_s",
    )
    storage.save_album_results(
        [buffalo_cluster],
        {1: [{"id": buffalo_face_id}]},
        "DBSCAN",
        model_name="buffalo_l",
    )

    assert storage.load_album_results("raccoon_s")["clusters"][0]["name"] == (
        "Raccoon cluster"
    )
    assert storage.load_album_results("buffalo_l")["clusters"][0]["name"] == (
        "Buffalo cluster"
    )
    assert storage.load_album_results("missing_model") == {}

    storage.clear_album_results(model_name="raccoon_s")

    assert storage.load_album_results("raccoon_s") == {}
    assert storage.load_album_results("buffalo_l")["clusters"]
    faces = {face["id"]: face for face in storage.list_media_faces()}
    assert faces[raccoon_face_id]["cluster_id"] is None
    assert faces[buffalo_face_id]["cluster_id"] == 1


def test_deleting_album_faces_for_one_model_preserves_other_models(tmp_path):
    storage = Storage(tmp_path / "test.db")
    path = str(tmp_path / "shared.jpg")
    media_id = storage.add_media_item(path, "image")
    raccoon_face_id = storage.add_media_face(
        media_id,
        np.array([1.0, 0.0], dtype=np.float32),
        model_name="raccoon_s",
    )
    buffalo_face_id = storage.add_media_face(
        media_id,
        np.array([0.0, 1.0], dtype=np.float32),
        model_name="buffalo_l",
    )

    assert storage.delete_media_faces_by_paths(
        [path], model_name="raccoon_s"
    ) == 1

    face_ids = {face["id"] for face in storage.list_media_faces()}
    assert raccoon_face_id not in face_ids
    assert buffalo_face_id in face_ids
    assert storage.existing_media_paths([path], model_name="raccoon_s") == set()
    assert storage.existing_media_paths([path], model_name="buffalo_l") == {path}


def test_webp_thumbnail_encoder_outputs_small_webp_bytes():
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    image[:, :, 1] = 200

    payload = encode_webp_thumbnail(image, max_side=120, quality=35)

    assert payload is not None
    assert payload[:4] == b"RIFF"
    assert b"WEBP" in payload[:16]
