from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest
from insightface_server.errors import ApiError
from insightface_server.models import (
    EMBEDDING_CONTRACT_PREFIX,
    embedding_contract_id,
)
from insightface_server.storage import (
    CollectionCapacityExceeded,
    CursorCodec,
    Database,
    FaceCropStore,
    MaxFacesPerPersonExceeded,
    Repository,
)


def collection_item(collection_id: str = "employees") -> dict[str, object]:
    return {
        "id": collection_id,
        "name": "Employees",
        "description": "",
        "default_threshold": 0.68,
        "metadata": {},
        "model_id": "recognition",
        "model_digest": "a" * 64,
        "embedding_dimension": 4,
        "preprocessing_version": "1",
    }


def face_item(face_id: str, embedding: np.ndarray | None = None) -> dict[str, object]:
    return {
        "id": face_id,
        "embedding": (
            embedding
            if embedding is not None
            else np.asarray([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
        ),
        "embedding_dimension": 4,
        "bounding_box": {"pixels": {"x": 1, "y": 2, "width": 3, "height": 4}},
        "landmarks": None,
        "detection_score": 0.9,
        "quality": {"score": 0.8},
        "model_id": "recognition",
        "model_digest": "a" * 64,
        "preprocessing_version": "1",
        "crop_path": None,
        "created_at": "2026-01-01T00:00:00.000Z",
    }


@pytest.fixture
def repository(tmp_path: Path) -> Repository:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    return Repository(database)


def test_cursor_secret_persists_and_tokens_are_scoped(tmp_path: Path) -> None:
    path = tmp_path / "cursor.key"
    first = CursorCodec(path)
    token = first.encode("people", "person-9")

    assert CursorCodec(path).decode(token, "people") == "person-9"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ApiError, match="cursor is invalid") as wrong_scope:
        first.decode(token, "collections")
    assert wrong_scope.value.code == "invalid_cursor"
    with pytest.raises(ApiError) as tampered:
        first.decode(token + "A", "people")
    assert tampered.value.code == "invalid_cursor"


def test_api_key_sync_creates_noops_and_atomically_rotates(repository: Repository) -> None:
    assert repository.sync_api_key("secret-key") == "created"
    with repository.database.read() as connection:
        original = connection.execute(
            "SELECT id,salt,digest,active,created_at FROM api_keys"
        ).fetchone()

    assert original is not None
    assert repository.sync_api_key("secret-key") == "unchanged"
    with repository.database.read() as connection:
        unchanged = connection.execute(
            "SELECT id,salt,digest,active,created_at FROM api_keys"
        ).fetchone()
    assert unchanged is not None
    assert tuple(unchanged) == tuple(original)

    assert repository.sync_api_key("replacement-key") == "rotated"
    assert repository.has_api_keys()
    assert repository.verify_api_key("replacement-key")
    assert not repository.verify_api_key("secret-key")
    assert not repository.verify_api_key("wrong")
    with repository.database.read() as connection:
        rows = connection.execute(
            "SELECT label,salt,digest,active FROM api_keys ORDER BY active,id"
        ).fetchall()
    assert len(rows) == 2
    assert [row["active"] for row in rows] == [0, 1]
    assert rows[1]["label"] == "startup"
    for row in rows:
        assert row["salt"] not in {b"secret-key", b"replacement-key"}
        assert row["digest"] not in {b"secret-key", b"replacement-key"}


def test_api_key_sync_rejects_an_empty_key(repository: Repository) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        repository.sync_api_key("")
    assert not repository.has_api_keys()


def test_repository_round_trips_embedding_and_optional_landmarks(
    repository: Repository,
) -> None:
    repository.create_collection(collection_item())
    embedding = np.asarray([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    repository.create_person_with_faces(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [
            {
                "id": "face-1",
                "embedding": embedding,
                "embedding_dimension": 4,
                "bounding_box": {"pixels": {"x": 1, "y": 2, "width": 3, "height": 4}},
                "landmarks": None,
                "detection_score": 0.9,
                "quality": {"score": 0.8},
                "model_id": "recognition",
                "model_digest": "a" * 64,
                "preprocessing_version": "1",
                "crop_path": None,
                "created_at": "2026-01-01T00:00:00.000Z",
            }
        ],
    )

    private = repository.get_face("employees", "alice", "face-1")
    public = repository.list_faces("employees", "alice", "", 10)[0]
    enrollment_embeddings = repository.list_face_embeddings("employees", "alice")

    assert private is not None
    np.testing.assert_array_equal(private["embedding"], embedding)
    assert private["landmarks"] is None
    assert private["embedding_source"] == "server"
    assert private["embedding_contract_id"] is None
    assert "landmarks_json" not in private
    assert "embedding" not in public
    assert "collection_id" not in public
    assert len(enrollment_embeddings) == 1
    np.testing.assert_array_equal(enrollment_embeddings[0], embedding)
    enrollment_embeddings[0][0] = 0.0
    np.testing.assert_array_equal(
        repository.list_face_embeddings("employees", "alice")[0], embedding
    )


def test_embedding_contract_id_is_stable_and_collection_derived(
    repository: Repository,
) -> None:
    contract_id = embedding_contract_id(
        model_id="recognition",
        model_digest="A" * 64,
        embedding_dimension=4,
        preprocessing_version="1",
    )
    created = repository.create_collection(collection_item())

    assert contract_id == (
        "ifsemb-v2-sha256:"
        "6727ec66a67682836ffe84461721f25bec728b126f47fe4c235bfb5af2ef1d9d"
    )
    assert contract_id.startswith(EMBEDDING_CONTRACT_PREFIX)
    assert created["embedding_contract_id"] == contract_id
    assert repository.get_collection("employees")["embedding_contract_id"] == contract_id  # type: ignore[index]
    listed = repository.list_collections("", 10)
    assert listed[0]["embedding_contract_id"] == contract_id


def test_repository_persists_external_trusted_embedding_provenance(
    repository: Repository,
) -> None:
    collection = repository.create_collection(collection_item())
    face = face_item("trusted-face")
    face.update(
        embedding_source="external_trusted",
        embedding_contract_id=collection["embedding_contract_id"],
    )
    repository.create_person_with_faces(
        "employees", {"id": "alice", "metadata": {}}, [face]
    )

    private = repository.get_face("employees", "alice", "trusted-face")
    public = repository.list_faces("employees", "alice", "", 10)[0]
    indexed = repository.all_faces("employees")[0]
    for stored in (private, public, indexed):
        assert stored is not None
        assert stored["embedding_source"] == "external_trusted"
        assert stored["embedding_contract_id"] == collection["embedding_contract_id"]
    assert "embedding" not in public


def test_migration_is_applied_once_and_ignores_non_migration_files(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_first.sql").write_text(
        "CREATE TABLE values_table(id INTEGER PRIMARY KEY);", encoding="utf-8"
    )
    (migrations / "README.sql").write_text("this is not SQL", encoding="utf-8")
    database = Database(tmp_path / "database.db", migrations)

    database.initialize()
    database.initialize()

    assert database.status()["migration_count"] == 1
    with database.read() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM schema_migrations WHERE version='0001_first.sql'"
            ).fetchone()[0]
            == 1
        )


def test_failed_migration_rolls_back_schema_and_can_be_repaired(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    migration = migrations / "0001_atomic.sql"
    migration.write_text(
        "CREATE TABLE should_rollback(id INTEGER); BROKEN SQL;", encoding="utf-8"
    )
    database = Database(tmp_path / "atomic.db", migrations)

    with pytest.raises(sqlite3.DatabaseError):
        database.initialize()
    with database.read() as connection:
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='should_rollback'"
        ).fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 0

    migration.write_text(
        "CREATE TABLE should_rollback(id INTEGER);", encoding="utf-8"
    )
    database.initialize()
    assert database.status()["migration_count"] == 1


def test_repository_enforces_external_id_uniqueness(repository: Repository) -> None:
    repository.create_collection(collection_item())
    repository.create_person_with_faces(
        "employees",
        {"id": "one", "external_id": "same", "metadata": {}},
        [],
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_person_with_faces(
            "employees",
            {"id": "two", "external_id": "same", "metadata": {}},
            [],
        )


def test_native_search_migration_backfills_legacy_rows(tmp_path: Path) -> None:
    source_migrations = Path(__file__).resolve().parents[2] / "migrations"
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    shutil.copy(source_migrations / "0001_initial.sql", migrations)
    database = Database(tmp_path / "legacy.db", migrations)
    database.initialize()

    now = "2026-01-01T00:00:00.000Z"
    embedding = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with database.write() as connection:
        connection.execute(
            """INSERT INTO collections(
                   id,name,description,default_threshold,model_id,model_version,
                   model_digest,embedding_dimension,preprocessing_version,
                   metadata_json,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("legacy", "Legacy", "", 0.3, "recognition", "1", "a" * 64, 4, "1", "{}", now, now),
        )
        connection.execute(
            """INSERT INTO persons(
                   collection_id,id,name,external_id,metadata_json,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?)""",
            ("legacy", "alice", "Alice", None, "{}", now, now),
        )
        connection.execute(
            """INSERT INTO face_samples(
                   id,collection_id,person_id,embedding,embedding_dimension,
                   bounding_box_json,landmarks_json,detection_score,quality_json,
                   model_id,model_version,model_digest,preprocessing_version,
                   crop_path,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "legacy-face",
                "legacy",
                "alice",
                embedding.tobytes(),
                4,
                "{}",
                None,
                0.9,
                "{}",
                "recognition",
                "1",
                "a" * 64,
                "1",
                None,
                now,
            ),
        )

    shutil.copy(source_migrations / "0002_native_search.sql", migrations)
    database.initialize()
    repository = Repository(database)

    collection = repository.get_collection("legacy")
    assert collection is not None
    assert collection["search_profile"] == "fp32_v1"
    assert collection["capacity_rows"] == 100000
    assert collection["max_faces_per_person"] == 20
    assert collection["load_policy"] == "lazy"
    assert collection["search_revision"] == 0
    batch = next(repository.iter_index_faces("legacy", batch_size=1))
    assert batch.revision == 0
    assert batch.max_seq == 0
    assert len(batch.faces) == 1
    assert batch.faces[0].vector_id > 0
    assert batch.faces[0].person_numeric_id > 0
    np.testing.assert_array_equal(batch.faces[0].embedding, embedding)


def test_int8_x736_migration_preserves_x1000_collection_and_child_rows(
    tmp_path: Path,
) -> None:
    source_migrations = Path(__file__).resolve().parents[2] / "migrations"
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    shutil.copy(source_migrations / "0001_initial.sql", migrations)
    shutil.copy(source_migrations / "0002_native_search.sql", migrations)
    # Repository always targets the current storage schema; stage migration 0004
    # through 0006, 0008 and 0009 here while withholding 0003 to exercise its x1000
    # backfill.
    shutil.copy(source_migrations / "0004_collection_face_crops.sql", migrations)
    shutil.copy(source_migrations / "0005_external_trusted_embeddings.sql", migrations)
    shutil.copy(source_migrations / "0006_collection_detection_profiles.sql", migrations)
    shutil.copy(source_migrations / "0008_liveness.sql", migrations)
    shutil.copy(source_migrations / "0009_model_identity.sql", migrations)
    database = Database(tmp_path / "x1000.db", migrations)
    database.initialize()
    repository = Repository(database)
    item = collection_item("legacy-int8")
    item.update(
        search_profile="int8_x1000_v1",
        capacity_rows=10,
        max_faces_per_person=5,
        load_policy="eager",
    )
    repository.create_collection(item)
    repository.create_person_with_faces(
        "legacy-int8",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [face_item("legacy-int8-face")],
    )
    before = next(repository.iter_index_faces("legacy-int8", batch_size=10)).faces[0]

    shutil.copy(source_migrations / "0003_int8_x736.sql", migrations)
    shutil.copy(source_migrations / "0004_collection_face_crops.sql", migrations)
    shutil.copy(source_migrations / "0005_external_trusted_embeddings.sql", migrations)
    database.initialize()
    migrated = Repository(database)

    collection = migrated.get_collection("legacy-int8")
    assert collection is not None
    assert collection["search_profile"] == "int8_x1000_v1"
    assert collection["person_count"] == 1
    assert collection["face_count"] == 1
    assert migrated.get_person("legacy-int8", "alice") is not None
    face = migrated.get_face("legacy-int8", "alice", "legacy-int8-face")
    assert face is not None
    after = next(migrated.iter_index_faces("legacy-int8", batch_size=10)).faces[0]
    assert after.vector_id == before.vector_id
    assert after.person_numeric_id == before.person_numeric_id
    np.testing.assert_array_equal(after.embedding, before.embedding)
    assert database.status()["migration_count"] == 8
    with database.read() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

    new_item = collection_item("new-int8")
    new_item["search_profile"] = "int8_x736_v1"
    assert migrated.create_collection(new_item)["search_profile"] == "int8_x736_v1"


def test_collection_update_returns_from_its_write_transaction(
    repository: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository.create_collection(collection_item())
    original_get_collection = repository.get_collection

    def forbidden_post_commit_read(collection_id: str):
        raise AssertionError(f"unexpected post-transaction read for {collection_id}")

    monkeypatch.setattr(repository, "get_collection", forbidden_post_commit_read)
    updated = repository.update_collection(
        "employees", {"name": "Updated", "capacity_rows": 12}
    )

    assert updated is not None
    assert updated["name"] == "Updated"
    assert updated["capacity_rows"] == 12
    monkeypatch.setattr(repository, "get_collection", original_get_collection)
    persisted = repository.get_collection("employees")
    assert persisted is not None
    assert persisted["name"] == "Updated"
    assert persisted["capacity_rows"] == 12


def test_face_mutation_is_durable_and_immediately_replayable(repository: Repository) -> None:
    item = collection_item()
    item.update(
        search_profile="bf16_v1",
        capacity_rows=10,
        max_faces_per_person=4,
        load_policy="eager",
    )
    repository.create_collection(item)
    person, mutation = repository.create_person_with_faces_with_mutation(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {"team": "infra"}},
        [face_item("face-1"), face_item("face-2")],
    )

    assert person["face_count"] == 2
    assert mutation is not None
    assert mutation.revision == 1
    assert mutation.max_seq > 0
    assert [change.operation for change in mutation.changes] == ["add", "add"]
    assert all(change.revision == 1 for change in mutation.changes)
    assert mutation.changes[0].seq < mutation.changes[1].seq
    assert len({change.vector_id for change in mutation.changes}) == 2
    assert len({change.person_numeric_id for change in mutation.changes}) == 1

    state = repository.get_search_state("employees")
    assert state is not None
    assert state.revision == mutation.revision
    assert state.max_seq == mutation.max_seq
    batches = list(repository.iter_index_faces("employees", batch_size=1))
    assert {batch.revision for batch in batches} == {mutation.revision}
    assert {batch.max_seq for batch in batches} == {mutation.max_seq}
    assert repository.get_search_changes("employees") == list(mutation.changes)

    vector_ids = [int(change.vector_id) for change in mutation.changes if change.vector_id]
    native_rows = repository.get_index_faces_by_vector_ids("employees", vector_ids)
    assert set(native_rows) == set(vector_ids)
    hydrated = repository.get_index_face_mappings(
        "employees", list(reversed(vector_ids)), include_embedding=True
    )
    assert set(hydrated) == set(vector_ids)
    assert {value["person"]["id"] for value in hydrated.values()} == {"alice"}
    assert all("embedding" in value for value in hydrated.values())

    next_mutation = repository.add_faces_with_mutation(
        "employees", "alice", [face_item("face-3")]
    )
    assert next_mutation is not None
    assert next_mutation.revision == 2
    assert next_mutation.max_seq > mutation.max_seq
    assert repository.get_search_changes(
        "employees", after_seq=mutation.max_seq
    ) == list(next_mutation.changes)


def test_search_change_acknowledgement_is_revision_bounded(repository: Repository) -> None:
    repository.create_collection(collection_item())
    _, first = repository.create_person_with_faces_with_mutation(
        "employees",
        {"id": "alice", "metadata": {}},
        [face_item("face-1")],
    )
    second = repository.add_faces_with_mutation(
        "employees", "alice", [face_item("face-2")]
    )
    assert first is not None and second is not None

    assert repository.acknowledge_search_changes("employees", first.max_seq) == 1
    remaining = repository.get_search_changes("employees")
    assert [change.revision for change in remaining] == [second.revision]
    assert repository.acknowledge_search_changes("employees", first.max_seq) == 0
    assert repository.reset_search_changes_before_full_rebuild() == 1
    assert repository.get_search_changes("employees") == []

    with pytest.raises(ValueError, match="must not be negative"):
        repository.acknowledge_search_changes("employees", -1)


def test_search_change_sequence_prevents_same_collection_id_aba_ack(
    repository: Repository,
) -> None:
    repository.create_collection(collection_item())
    _, added = repository.create_person_with_faces_with_mutation(
        "employees",
        {"id": "old-person", "metadata": {}},
        [face_item("old-face")],
    )
    assert added is not None
    status, _, dropped = repository.delete_collection_with_mutation(
        "employees", force=True
    )
    assert status == "deleted" and dropped is not None

    repository.create_collection(collection_item())
    _, recreated = repository.create_person_with_faces_with_mutation(
        "employees",
        {"id": "new-person", "metadata": {}},
        [face_item("new-face")],
    )
    assert recreated is not None and recreated.revision == 1
    assert recreated.max_seq > dropped.max_seq

    repository.acknowledge_search_changes("employees", dropped.max_seq)

    assert repository.get_search_changes("employees") == list(recreated.changes)


def test_face_limits_are_checked_inside_the_write_transaction(repository: Repository) -> None:
    item = collection_item()
    item.update(capacity_rows=2, max_faces_per_person=1)
    repository.create_collection(item)
    repository.create_person_with_faces_with_mutation(
        "employees", {"id": "alice", "metadata": {}}, [face_item("face-1")]
    )

    with pytest.raises(MaxFacesPerPersonExceeded):
        repository.add_faces_with_mutation(
            "employees", "alice", [face_item("face-2")]
        )
    assert repository.get_person("employees", "alice")["face_count"] == 1  # type: ignore[index]
    assert repository.get_search_revision("employees") == 1
    assert len(repository.get_search_changes("employees")) == 1

    repository.create_person_with_faces_with_mutation(
        "employees", {"id": "bob", "metadata": {}}, [face_item("face-2")]
    )
    with pytest.raises(CollectionCapacityExceeded):
        repository.create_person_with_faces_with_mutation(
            "employees", {"id": "carol", "metadata": {}}, [face_item("face-3")]
        )
    assert repository.get_person("employees", "carol") is None
    assert repository.get_search_revision("employees") == 2

    with pytest.raises(CollectionCapacityExceeded):
        repository.update_collection("employees", {"capacity_rows": 1})
    with pytest.raises(MaxFacesPerPersonExceeded):
        repository.update_collection("employees", {"max_faces_per_person": 0})
    collection = repository.get_collection("employees")
    assert collection is not None
    assert collection["capacity_rows"] == 2
    assert collection["max_faces_per_person"] == 1


def test_cascading_deletes_produce_durable_search_changes(repository: Repository) -> None:
    repository.create_collection(collection_item())
    _, added = repository.create_person_with_faces_with_mutation(
        "employees",
        {"id": "alice", "metadata": {}},
        [face_item("face-1"), face_item("face-2"), face_item("face-3")],
    )
    assert added is not None
    add_ids = [change.vector_id for change in added.changes]

    deleted, _, face_delete = repository.delete_face_with_mutation(
        "employees", "alice", "face-1"
    )
    assert deleted and face_delete is not None
    assert face_delete.revision == 2
    assert face_delete.changes[0].operation == "delete"
    assert face_delete.changes[0].vector_id == add_ids[0]

    deleted, _, person_delete = repository.delete_person_with_mutation(
        "employees", "alice"
    )
    assert deleted and person_delete is not None
    assert person_delete.revision == 3
    assert [change.vector_id for change in person_delete.changes] == add_ids[1:]
    assert repository.get_index_faces_by_vector_ids(
        "employees", [int(value) for value in add_ids if value is not None]
    ) == {}

    status, _, collection_delete = repository.delete_collection_with_mutation(
        "employees", force=True
    )
    assert status == "deleted"
    assert collection_delete is not None
    assert collection_delete.revision == 4
    assert [change.operation for change in collection_delete.changes] == ["clear"]
    assert repository.get_collection("employees") is None
    assert repository.get_search_revision("employees") is None
    changes = repository.get_search_changes("employees")
    assert [change.operation for change in changes] == [
        "add",
        "add",
        "add",
        "delete",
        "delete",
        "delete",
        "clear",
    ]


def test_collection_delete_uses_one_clear_event_for_all_faces(repository: Repository) -> None:
    repository.create_collection(collection_item("large"))
    _, added = repository.create_person_with_faces_with_mutation(
        "large",
        {"id": "alice", "metadata": {}},
        [face_item("large-face-1"), face_item("large-face-2")],
    )
    assert added is not None

    status, _, deleted = repository.delete_collection_with_mutation("large", force=True)

    assert status == "deleted"
    assert deleted is not None
    assert deleted.revision == 2
    assert [change.operation for change in deleted.changes] == ["clear"]
    assert [change.operation for change in repository.get_search_changes("large")] == [
        "add",
        "add",
        "clear",
    ]


def test_collection_crop_storage_defaults_off_and_can_be_updated(
    repository: Repository,
) -> None:
    created = repository.create_collection(collection_item())

    assert created["save_face_crops"] is False
    updated = repository.update_collection("employees", {"save_face_crops": True})
    assert updated is not None
    assert updated["save_face_crops"] is True
    with repository.database.read() as connection:
        stored = connection.execute(
            "SELECT save_face_crops FROM collections WHERE id='employees'"
        ).fetchone()[0]
    assert stored == 1


def test_crop_blob_is_only_loaded_by_dedicated_repository_method(
    repository: Repository,
) -> None:
    item = collection_item()
    item["save_face_crops"] = True
    repository.create_collection(item)
    crop_bytes = b"\xff\xd8private-jpeg\xff\xd9"
    face = face_item("face-crop")
    face.update(crop_image=crop_bytes, crop_media_type="image/jpeg")
    repository.create_person_with_faces(
        "employees", {"id": "alice", "metadata": {}}, [face]
    )

    private = repository.get_face("employees", "alice", "face-crop")
    listed = repository.list_faces("employees", "alice", "", 10)[0]
    loaded_for_index = repository.all_faces("employees")[0]
    crop = repository.get_face_crop("employees", "alice", "face-crop")

    for ordinary in (private, listed, loaded_for_index):
        assert ordinary is not None
        assert ordinary["has_crop"] is True
        assert "crop_image" not in ordinary
        assert "crop_media_type" not in ordinary
        assert "crop_path" not in ordinary
    assert crop == {
        "bytes": crop_bytes,
        "media_type": "image/jpeg",
        "crop_path": None,
    }


def test_get_face_crop_supports_legacy_disk_paths(repository: Repository) -> None:
    repository.create_collection(collection_item())
    face = face_item("legacy-crop")
    face["crop_path"] = "face-crops/legacy/legacy-crop.jpg"
    repository.create_person_with_faces(
        "employees", {"id": "alice", "metadata": {}}, [face]
    )

    public = repository.list_faces("employees", "alice", "", 10)[0]
    crop = repository.get_face_crop("employees", "alice", "legacy-crop")

    assert public["has_crop"] is True
    assert "crop_path" not in public
    assert crop == {
        "bytes": None,
        "media_type": "image/jpeg",
        "crop_path": "face-crops/legacy/legacy-crop.jpg",
    }


def test_collection_face_crop_migration_preserves_legacy_rows(tmp_path: Path) -> None:
    source_migrations = Path(__file__).resolve().parents[2] / "migrations"
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    for name in (
        "0001_initial.sql",
        "0002_native_search.sql",
        "0003_int8_x736.sql",
    ):
        shutil.copy(source_migrations / name, migrations)
    database = Database(tmp_path / "legacy-crops.db", migrations)
    database.initialize()
    now = "2026-01-01T00:00:00.000Z"
    embedding = np.asarray([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    with database.write() as connection:
        connection.execute(
            """INSERT INTO collections(
                   id,name,description,default_threshold,model_id,model_version,
                   model_digest,embedding_dimension,preprocessing_version,
                   metadata_json,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "employees",
                "Employees",
                "",
                0.68,
                "recognition",
                "1",
                "a" * 64,
                4,
                "1",
                "{}",
                now,
                now,
            ),
        )
        connection.execute(
            """INSERT INTO persons(
                   collection_id,id,name,external_id,metadata_json,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?)""",
            ("employees", "alice", "Alice", None, "{}", now, now),
        )
        connection.execute(
            """INSERT INTO face_samples(
                   id,collection_id,person_id,embedding,embedding_dimension,
                   bounding_box_json,landmarks_json,detection_score,quality_json,
                   model_id,model_version,model_digest,preprocessing_version,
                   crop_path,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "face-1",
                "employees",
                "alice",
                embedding.tobytes(),
                4,
                "{}",
                None,
                0.9,
                "{}",
                "recognition",
                "1",
                "a" * 64,
                "1",
                None,
                now,
            ),
        )

    shutil.copy(source_migrations / "0004_collection_face_crops.sql", migrations)
    shutil.copy(source_migrations / "0005_external_trusted_embeddings.sql", migrations)
    shutil.copy(source_migrations / "0008_liveness.sql", migrations)
    database.initialize()
    migrated = Repository(database)

    collection = migrated.get_collection("employees")
    face = migrated.get_face("employees", "alice", "face-1")
    assert collection is not None and collection["save_face_crops"] is False
    assert face is not None and face["has_crop"] is False
    assert face["embedding_source"] == "server"
    assert face["embedding_contract_id"] is None
    assert migrated.get_face_crop("employees", "alice", "face-1") is None
    assert "liveness" not in face
    assert database.status()["migration_count"] == 6
    with database.read() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_face_crop_store_encodes_and_safely_reads_legacy_jpeg(tmp_path: Path) -> None:
    pixels = np.zeros((40, 50, 3), dtype=np.uint8)
    pixels[5:35, 10:45] = (20, 120, 240)
    encoded = FaceCropStore.encode(pixels, (10.0, 5.0, 45.0, 35.0))

    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (112, 112, 3)

    store = FaceCropStore(tmp_path, enabled=True)
    path = store.save("employees", "face-1", pixels, (10.0, 5.0, 45.0, 35.0))
    assert path is not None
    assert store.read(path) == encoded
    assert store.read("face-crops/missing.jpg") is None
    with pytest.raises(RuntimeError, match="outside managed storage"):
        store.read("../secret")
