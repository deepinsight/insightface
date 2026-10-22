from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from insightface_server.search import ReferenceSearchBackend, SearchIndexManager
from insightface_server.search.base import SearchIndexError
from insightface_server.search.reference import ReferenceSearchIndex, profile_similarity
from insightface_server.storage import Database, Repository


def unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def collection_item() -> dict[str, object]:
    return {
        "id": "employees",
        "name": "Employees",
        "description": "",
        "default_threshold": 0.3,
        "metadata": {},
        "model_id": "recognition",
        "model_digest": "a" * 64,
        "embedding_dimension": 4,
        "preprocessing_version": "1",
        "search_profile": "fp32_v1",
        "capacity_rows": 4,
        "max_faces_per_person": 3,
        "load_policy": "lazy",
    }


def face(face_id: str, embedding: np.ndarray) -> dict[str, object]:
    return {
        "id": face_id,
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


class SelectivelyFailingBackend(ReferenceSearchBackend):
    """Reference backend with deterministic allocation failures for rollback tests."""

    def __init__(self, *failed_capacities: int) -> None:
        self.failed_capacities = set(failed_capacities)

    def create_index(self, *, profile: str, dimension: int, capacity_rows: int):
        if capacity_rows in self.failed_capacities:
            raise SearchIndexError(f"injected allocation failure for {capacity_rows}")
        return super().create_index(
            profile=profile,
            dimension=dimension,
            capacity_rows=capacity_rows,
        )


class CountingReferenceIndex(ReferenceSearchIndex):
    def __init__(
        self,
        owner: CountingReferenceBackend,
        handle_id: int,
        *,
        profile: str,
        dimension: int,
        capacity_rows: int,
    ) -> None:
        self.owner = owner
        self.handle_id = handle_id
        super().__init__(
            profile=profile,
            dimension=dimension,
            capacity_rows=capacity_rows,
        )

    def close(self) -> None:
        if not self._closed:
            self.owner.closed_handles.add(self.handle_id)
            self.owner.active_handles.remove(self.handle_id)
        super().close()


class CountingReferenceBackend(ReferenceSearchBackend):
    def __init__(self) -> None:
        self.created_handles = 0
        self.active_handles: set[int] = set()
        self.closed_handles: set[int] = set()

    def create_index(self, *, profile: str, dimension: int, capacity_rows: int):
        self.created_handles += 1
        handle_id = self.created_handles
        self.active_handles.add(handle_id)
        return CountingReferenceIndex(
            self,
            handle_id,
            profile=profile,
            dimension=dimension,
            capacity_rows=capacity_rows,
        )


class ConcurrentReadReferenceIndex(ReferenceSearchIndex):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.barrier = threading.Barrier(2)
        self.concurrent_probe_enabled = False

    def _scores(self, query: np.ndarray, slots: np.ndarray) -> np.ndarray:
        if self.concurrent_probe_enabled:
            self.barrier.wait(timeout=5)
        return super()._scores(query, slots)


class ConcurrentReadReferenceBackend(ReferenceSearchBackend):
    def __init__(self) -> None:
        self.index: ConcurrentReadReferenceIndex | None = None

    def create_index(self, *, profile: str, dimension: int, capacity_rows: int):
        self.index = ConcurrentReadReferenceIndex(
            profile=profile,
            dimension=dimension,
            capacity_rows=capacity_rows,
        )
        return self.index


def test_same_collection_searches_use_one_generation_concurrently(tmp_path: Path) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    repository.create_collection(collection_item())
    repository.create_person_with_faces(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [face("alice-1", unit([1, 0, 0, 0]))],
    )
    backend = ConcurrentReadReferenceBackend()
    manager = SearchIndexManager(repository, backend)
    manager.ensure_ready("employees")
    assert backend.index is not None
    backend.index.concurrent_probe_enabled = True

    def search() -> list[dict[str, object]]:
        return manager.search(
            "employees", unit([1, 0, 0, 0]), limit=1, threshold=0.0
        )

    failures: list[BaseException] = []
    results: list[list[dict[str, object]]] = []

    def run() -> None:
        try:
            results.append(search())
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    second.start()
    first.join(timeout=6)
    second.join(timeout=6)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert len(results) == 2
    assert all(result[0]["person"]["id"] == "alice" for result in results)
    manager.close()


def test_manager_builds_groups_and_applies_add_delete_mutations(tmp_path: Path) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    repository.create_collection(collection_item())
    repository.create_person_with_faces(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [face("alice-1", unit([1, 0, 0, 0]))],
    )
    repository.create_person_with_faces(
        "employees",
        {"id": "bob", "name": "Bob", "metadata": {}},
        [face("bob-1", unit([0, 1, 0, 0]))],
    )
    manager = SearchIndexManager(repository, ReferenceSearchBackend(), build_batch_rows=1)

    initial = manager.search(
        "employees", unit([1, 0, 0, 0]), limit=5, threshold=0.0
    )
    assert [item["person"]["id"] for item in initial] == ["alice", "bob"]
    assert initial[0]["similarity"] == 1.0
    assert initial[1]["similarity"] == 0.0
    assert repository.get_search_changes("employees") == []

    def add():
        mutation = repository.add_faces_with_mutation(
            "employees", "bob", [face("bob-2", unit([0.8, 0.6, 0, 0]))]
        )
        return None, mutation

    manager.run_mutation("employees", add, expected_additions=1)
    assert repository.get_search_changes("employees") == []
    updated = manager.search(
        "employees", unit([1, 0, 0, 0]), limit=5, threshold=0.0
    )
    assert [item["person"]["id"] for item in updated] == ["alice", "bob"]
    assert updated[1]["matched_face_id"] == "bob-2"
    assert updated[1]["similarity"] == pytest.approx(0.8)

    def delete():
        deleted, crop, mutation = repository.delete_face_with_mutation(
            "employees", "alice", "alice-1"
        )
        return (deleted, crop), mutation

    assert manager.run_mutation("employees", delete) == (True, None)
    assert repository.get_search_changes("employees") == []
    after_delete = manager.search(
        "employees", unit([1, 0, 0, 0]), limit=5, threshold=0.0
    )
    assert [item["person"]["id"] for item in after_delete] == ["bob"]
    assert manager.runtime_summary()["collections"][0]["applied_revision"] == 4
    manager.close()


def test_best_other_person_excludes_target_and_returns_profile_score(
    tmp_path: Path,
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    repository.create_collection(collection_item())
    repository.create_person_with_faces(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [face("alice-1", unit([1, 0, 0, 0]))],
    )
    repository.create_person_with_faces(
        "employees",
        {"id": "bob", "name": "Bob", "metadata": {}},
        [face("bob-1", unit([0.8, 0.6, 0, 0]))],
    )
    manager = SearchIndexManager(repository, ReferenceSearchBackend())

    other = manager.best_other_person(
        "employees", unit([1, 0, 0, 0]), exclude_person_id="alice"
    )
    assert other == {
        "person_id": "bob",
        "face_id": "bob-1",
        "similarity": pytest.approx(0.8),
    }
    assert manager.best_other_person(
        "employees", unit([1, 0, 0, 0]), exclude_person_id="nobody"
    ) == {
        "person_id": "alice",
        "face_id": "alice-1",
        "similarity": 1.0,
    }
    manager.close()


def test_best_other_person_preserves_unclipped_int8_review_score(
    tmp_path: Path,
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    values = {
        **collection_item(),
        "embedding_dimension": 512,
        "search_profile": "int8_x736_v1",
    }
    repository.create_collection(values)
    embedding = unit([1.0] * 512)
    sample = {**face("alice-1", embedding), "embedding_dimension": 512}
    repository.create_person_with_faces(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [sample],
    )
    manager = SearchIndexManager(repository, ReferenceSearchBackend())

    other = manager.best_other_person(
        "employees", embedding, exclude_person_id="nobody"
    )

    assert other is not None
    assert other["similarity"] == pytest.approx(
        profile_similarity("int8_x736_v1", embedding, embedding)
    )
    assert other["similarity"] > 1.0
    manager.close()


def test_startup_discards_pre_restart_events_for_lazy_full_rebuild(
    tmp_path: Path,
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    repository.create_collection(collection_item())
    repository.create_person_with_faces_with_mutation(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [face("alice-1", unit([1, 0, 0, 0]))],
    )
    assert len(repository.get_search_changes("employees")) == 1
    manager = SearchIndexManager(repository, ReferenceSearchBackend())

    manager.startup()

    assert repository.get_search_changes("employees") == []
    assert manager.runtime_summary()["collections"] == []
    matches = manager.search(
        "employees", unit([1, 0, 0, 0]), limit=1, threshold=0.3
    )
    assert matches[0]["person"]["id"] == "alice"
    manager.close()


def test_outbox_acknowledgement_failure_retains_event_without_failing_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    repository.create_collection(collection_item())
    repository.create_person_with_faces_with_mutation(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [face("alice-1", unit([1, 0, 0, 0]))],
    )

    def fail_acknowledgement(_collection_id: str, _revision: int) -> int:
        raise OSError("injected acknowledgement failure")

    acknowledge = repository.acknowledge_search_changes
    monkeypatch.setattr(repository, "acknowledge_search_changes", fail_acknowledgement)
    manager = SearchIndexManager(repository, ReferenceSearchBackend())

    matches = manager.search(
        "employees", unit([1, 0, 0, 0]), limit=1, threshold=0.3
    )

    assert matches[0]["person"]["id"] == "alice"
    assert len(repository.get_search_changes("employees")) == 1

    monkeypatch.setattr(repository, "acknowledge_search_changes", acknowledge)
    retry = manager.search(
        "employees", unit([1, 0, 0, 0]), limit=1, threshold=0.3
    )
    assert retry[0]["person"]["id"] == "alice"
    assert repository.get_search_changes("employees") == []
    manager.close()


def test_manager_compares_threshold_against_unrounded_raw_cosine(
    tmp_path: Path,
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    repository.create_collection(collection_item())
    query = unit([1, 0, 0, 0])
    just_below = np.asarray(
        [0.2999996, np.sqrt(1.0 - 0.2999996**2), 0, 0], dtype=np.float32
    )
    repository.create_person_with_faces(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [face("alice-1", just_below)],
    )
    manager = SearchIndexManager(repository, ReferenceSearchBackend())

    assert round(float(np.dot(query, just_below)), 6) == 0.3
    assert manager.search("employees", query, limit=1, threshold=0.3) == []
    manager.close()


def test_eager_create_allocation_failure_does_not_commit_collection(
    tmp_path: Path,
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    values = {**collection_item(), "load_policy": "eager"}
    manager = SearchIndexManager(repository, SelectivelyFailingBackend(4))

    with pytest.raises(SearchIndexError, match="injected allocation failure"):
        manager.run_create(values, lambda: repository.create_collection(values))

    assert repository.get_collection("employees") is None
    assert manager.runtime_summary()["collections"] == []
    manager.close()


def test_capacity_patch_staging_failure_preserves_old_config_and_generation(
    tmp_path: Path,
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    repository.create_collection(collection_item())
    repository.create_person_with_faces(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [face("alice-1", unit([1, 0, 0, 0]))],
    )
    backend = SelectivelyFailingBackend()
    manager = SearchIndexManager(repository, backend)
    assert manager.search(
        "employees", unit([1, 0, 0, 0]), limit=1, threshold=0.0
    )[0]["person"]["id"] == "alice"
    original_runtime = manager.runtime_summary()["collections"][0]
    backend.failed_capacities.add(8)
    update_called = False

    def update():
        nonlocal update_called
        update_called = True
        return repository.update_collection("employees", {"capacity_rows": 8})

    with pytest.raises(SearchIndexError, match="injected allocation failure"):
        manager.run_configuration_update(
            "employees", update, changes={"capacity_rows": 8}
        )

    persisted = repository.get_collection("employees")
    assert persisted is not None
    assert persisted["capacity_rows"] == 4
    assert update_called is False
    runtime = manager.runtime_summary()["collections"][0]
    assert runtime["state"] == "ready"
    assert runtime["capacity_rows"] == 4
    assert runtime["generation"] == original_runtime["generation"]
    assert manager.search(
        "employees", unit([1, 0, 0, 0]), limit=1, threshold=0.0
    )[0]["matched_face_id"] == "alice-1"
    manager.close()


def test_load_policy_transitions_materialize_and_release_index(tmp_path: Path) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    values = collection_item()
    manager = SearchIndexManager(repository, ReferenceSearchBackend())
    manager.run_create(values, lambda: repository.create_collection(values))

    initial = manager.runtime_summary()["collections"][0]
    assert initial["state"] == "unloaded"

    eager = manager.run_configuration_update(
        "employees",
        lambda: repository.update_collection("employees", {"load_policy": "eager"}),
        changes={"load_policy": "eager"},
    )
    assert eager is not None and eager["load_policy"] == "eager"
    ready = manager.runtime_summary()["collections"][0]
    assert ready["state"] == "ready"
    assert ready["capacity_rows"] == 4

    lazy = manager.run_configuration_update(
        "employees",
        lambda: repository.update_collection("employees", {"load_policy": "lazy"}),
        changes={"load_policy": "lazy"},
    )
    assert lazy is not None and lazy["load_policy"] == "lazy"
    unloaded = manager.runtime_summary()["collections"][0]
    assert unloaded["state"] == "unloaded"
    assert "capacity_rows" not in unloaded
    manager.close()


@pytest.mark.parametrize("load_policy", ["eager", "lazy"])
def test_unchanged_search_configuration_keeps_loaded_generation(
    tmp_path: Path, load_policy: str
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    values = {**collection_item(), "load_policy": load_policy}
    repository.create_collection(values)
    backend = CountingReferenceBackend()
    manager = SearchIndexManager(repository, backend)
    manager.ensure_ready("employees")
    before = manager.runtime_summary()["collections"][0]

    unchanged = {
        "capacity_rows": values["capacity_rows"],
        "max_faces_per_person": values["max_faces_per_person"],
        "load_policy": values["load_policy"],
    }
    updated = manager.run_configuration_update(
        "employees",
        lambda: repository.update_collection("employees", unchanged),
        changes=unchanged,
    )

    assert updated is not None
    assert backend.created_handles == 1
    assert backend.closed_handles == set()
    assert backend.active_handles == {1}
    after = manager.runtime_summary()["collections"][0]
    assert after["state"] == "ready"
    assert after["generation"] == before["generation"]
    assert after["capacity_rows"] == before["capacity_rows"]
    manager.close()
    assert backend.active_handles == set()
    assert backend.closed_handles == {1}


def test_collection_drop_serializes_with_search_and_returns_missing_not_index_error(
    tmp_path: Path,
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    repository.create_collection(collection_item())
    repository.create_person_with_faces(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [face("alice-1", unit([1, 0, 0, 0]))],
    )
    manager = SearchIndexManager(repository, ReferenceSearchBackend())
    manager.ensure_ready("employees")
    delete_entered = threading.Event()
    allow_delete = threading.Event()
    failures: list[BaseException] = []
    search_outcome: list[str] = []

    def drop_operation():
        delete_entered.set()
        assert allow_delete.wait(timeout=5)
        status, crops, mutation = repository.delete_collection_with_mutation(
            "employees", force=True
        )
        return (status, crops), mutation

    def drop() -> None:
        try:
            manager.run_drop("employees", drop_operation)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def search() -> None:
        try:
            manager.search(
                "employees", unit([1, 0, 0, 0]), limit=1, threshold=0.0
            )
            search_outcome.append("served-before-delete")
        except KeyError:
            # FaceService maps this expected post-delete result to HTTP 404.
            search_outcome.append("collection-not-found")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    drop_thread = threading.Thread(target=drop)
    search_thread = threading.Thread(target=search)
    drop_thread.start()
    assert delete_entered.wait(timeout=5)
    search_thread.start()
    allow_delete.set()
    drop_thread.join(timeout=5)
    search_thread.join(timeout=5)

    assert not drop_thread.is_alive()
    assert not search_thread.is_alive()
    assert failures == []
    assert search_outcome == ["collection-not-found"]
    manager.close()


def test_collection_drop_finishes_after_derived_index_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    repository.create_collection(collection_item())
    repository.create_person_with_faces(
        "employees",
        {"id": "alice", "name": "Alice", "metadata": {}},
        [face("alice-1", unit([1, 0, 0, 0]))],
    )
    manager = SearchIndexManager(repository, ReferenceSearchBackend())
    manager.ensure_ready("employees")
    index = manager._slots["employees"].index
    assert index is not None

    def fail_close() -> None:
        raise OSError("injected close failure")

    monkeypatch.setattr(index, "close", fail_close)

    def drop_operation():
        status, crops, mutation = repository.delete_collection_with_mutation(
            "employees", force=True
        )
        return (status, crops), mutation

    result = manager.run_drop("employees", drop_operation)

    assert result == ("deleted", [])
    assert repository.get_collection("employees") is None
    assert repository.get_search_changes("employees") == []
    assert manager.runtime_summary()["collections"] == []
    manager.close()


def test_concurrent_drop_and_same_id_eager_recreate_registers_only_new_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "server.db", migrations)
    database.initialize()
    repository = Repository(database)
    values = {**collection_item(), "load_policy": "eager"}
    repository.create_collection(values)
    backend = CountingReferenceBackend()
    manager = SearchIndexManager(repository, backend)
    manager.ensure_ready("employees")

    drop_entered = threading.Event()
    allow_drop = threading.Event()
    creator_captured_slot = threading.Event()
    failures: list[BaseException] = []
    outcomes: list[str] = []
    original_slot = manager._slot

    def observed_slot(collection_id: str):
        slot = original_slot(collection_id)
        if threading.current_thread().name == "creator":
            creator_captured_slot.set()
        return slot

    monkeypatch.setattr(manager, "_slot", observed_slot)

    def drop_operation():
        drop_entered.set()
        assert allow_drop.wait(timeout=5)
        status, crops, mutation = repository.delete_collection_with_mutation(
            "employees", force=True
        )
        return (status, crops), mutation

    def drop() -> None:
        try:
            manager.run_drop("employees", drop_operation)
            outcomes.append("dropped")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def recreate() -> None:
        try:
            manager.run_create(values, lambda: repository.create_collection(values))
            outcomes.append("recreated")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    drop_thread = threading.Thread(target=drop, name="dropper")
    create_thread = threading.Thread(target=recreate, name="creator")
    drop_thread.start()
    assert drop_entered.wait(timeout=5)
    create_thread.start()
    assert creator_captured_slot.wait(timeout=5)
    allow_drop.set()
    drop_thread.join(timeout=5)
    create_thread.join(timeout=5)

    assert not drop_thread.is_alive()
    assert not create_thread.is_alive()
    assert failures == []
    assert sorted(outcomes) == ["dropped", "recreated"]
    assert backend.created_handles == 2
    assert backend.closed_handles == {1}
    assert backend.active_handles == {2}
    runtime = manager.runtime_summary()["collections"]
    assert len(runtime) == 1
    assert runtime[0]["collection_id"] == "employees"
    assert runtime[0]["state"] == "ready"

    def add_person():
        created, mutation = repository.create_person_with_faces_with_mutation(
            "employees",
            {"id": "alice", "name": "Alice", "metadata": {}},
            [face("alice-1", unit([1, 0, 0, 0]))],
        )
        return created, mutation

    manager.run_mutation("employees", add_person, expected_additions=1)
    matches = manager.search(
        "employees", unit([1, 0, 0, 0]), limit=1, threshold=0.3
    )
    assert matches[0]["person"]["id"] == "alice"
    assert matches[0]["similarity"] == pytest.approx(1.0)
    assert backend.created_handles == 2
    assert backend.active_handles == {2}

    manager.close()
    assert backend.active_handles == set()
    assert backend.closed_handles == {1, 2}
