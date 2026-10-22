from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

import numpy as np

from ..storage import Repository, SearchMutation
from .base import (
    IndexRecord,
    MutableSearchIndex,
    SearchBackend,
    SearchIndexCapacityError,
    SearchIndexError,
)
from .reference import profile_similarity
from .synchronization import ReadWriteLock

LOGGER = logging.getLogger("insightface_server.search")
T = TypeVar("T")


class CollectionIndexState(StrEnum):
    UNLOADED = "unloaded"
    BUILDING = "building"
    READY = "ready"
    DIRTY = "dirty"
    REBUILDING = "rebuilding"
    FAILED = "failed"
    DRAINING = "draining"
    CLOSED = "closed"


class SearchMutationCommittedError(SearchIndexError):
    """SQLite committed, but its derived search generation could not catch up."""

    def __init__(
        self,
        collection_id: str,
        revision: int,
        detail: str,
        *,
        committed_result: Any = None,
    ):
        self.collection_id = collection_id
        self.revision = revision
        self.committed_result = committed_result
        super().__init__(
            f"search mutation committed at revision {revision}, but indexing failed: {detail}"
        )


@dataclass(slots=True)
class _CollectionSlot:
    collection_id: str
    lock: ReadWriteLock = field(default_factory=ReadWriteLock)
    state: CollectionIndexState = CollectionIndexState.UNLOADED
    index: MutableSearchIndex | None = None
    applied_revision: int = 0
    generation: int = 0
    last_error: str | None = None


class SearchIndexManager:
    """Own one disposable exact-search generation per active Collection."""

    def __init__(
        self,
        repository: Repository,
        backend: SearchBackend,
        *,
        build_batch_rows: int = 4096,
    ) -> None:
        if build_batch_rows <= 0:
            raise ValueError("build_batch_rows must be positive")
        self.repository = repository
        self.backend = backend
        self.build_batch_rows = build_batch_rows
        self._slots: dict[str, _CollectionSlot] = {}
        self._lock = threading.RLock()
        self._closed = False

    def _slot(self, collection_id: str) -> _CollectionSlot:
        with self._lock:
            if self._closed:
                raise SearchIndexError("search index manager is closed")
            return self._slots.setdefault(collection_id, _CollectionSlot(collection_id))

    @contextmanager
    def _locked_slot(self, collection_id: str) -> Iterator[_CollectionSlot]:
        """Lock the currently registered slot, retrying across delete/recreate."""

        while True:
            slot = self._slot(collection_id)
            slot.lock.acquire_write()
            with self._lock:
                registered = self._slots.get(collection_id) is slot
            if registered:
                break
            slot.lock.release_write()
        try:
            yield slot
        finally:
            slot.lock.release_write()

    @contextmanager
    def _ready_read_slot(
        self, collection_id: str
    ) -> Iterator[tuple[_CollectionSlot, MutableSearchIndex]]:
        """Yield one stable, current generation while permitting peer readers."""

        while True:
            with self._locked_slot(collection_id) as slot:
                index = self._ensure_ready_locked(slot)
                state = self.repository.get_search_state(collection_id)
                if state is None:
                    raise KeyError(collection_id)
                if state.revision != slot.applied_revision:
                    slot.state = CollectionIndexState.DIRTY
                    index = self._ensure_ready_locked(slot)
                    state = self.repository.get_search_state(collection_id)
                    if state is None:
                        raise KeyError(collection_id)
                if state.max_seq:
                    self._acknowledge(collection_id, state.max_seq)
                expected_revision = slot.applied_revision
                expected_generation = slot.generation

            slot.lock.acquire_read()
            try:
                with self._lock:
                    registered = self._slots.get(collection_id) is slot
                stable = (
                    registered
                    and slot.index is index
                    and slot.state == CollectionIndexState.READY
                    and slot.applied_revision == expected_revision
                    and slot.generation == expected_generation
                )
                if not stable:
                    continue
                yield slot, index
                return
            finally:
                slot.lock.release_read()

    def _mark_generation_dirty(
        self, collection_id: str, expected_index: MutableSearchIndex
    ) -> None:
        with self._locked_slot(collection_id) as slot:
            if slot.index is expected_index:
                slot.state = CollectionIndexState.DIRTY

    def startup(self) -> None:
        # Search generations are process-local. After a restart every eager
        # Collection is rebuilt from an authoritative SQLite snapshot and lazy
        # Collections do the same on first use, so pre-restart outbox rows are
        # no longer needed for recovery.
        reset = self.repository.reset_search_changes_before_full_rebuild()
        if reset:
            LOGGER.info("discarded_stale_search_changes count=%s", reset)
        after = ""
        while True:
            collections = self.repository.list_collections(after, 100)
            if not collections:
                break
            for collection in collections:
                if collection.get("load_policy") == "eager":
                    self.ensure_ready(collection["id"])
            after = collections[-1]["id"]
            if len(collections) < 100:
                break

    def validate_configuration(self, *, profile: str, dimension: int) -> None:
        summary = self.backend.runtime_summary()
        supported = summary.get("profiles")
        if isinstance(supported, list) and profile not in supported:
            raise SearchIndexError(
                f"search backend {self.backend.name} does not support profile {profile}"
            )
        if self.backend.name.startswith("native_") and dimension != 512:
            raise SearchIndexError("native search requires 512-dimensional embeddings")

    def run_create(
        self,
        collection: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Preallocate an eager empty generation before committing its row."""

        collection_id = str(collection["id"])
        with self._locked_slot(collection_id) as slot:
            # Let SQLite produce the authoritative duplicate error without
            # disturbing an already-active generation.
            if self.repository.get_collection(collection_id) is not None:
                return operation()
            self.validate_configuration(
                profile=str(collection["search_profile"]),
                dimension=int(collection["embedding_dimension"]),
            )
            staging: MutableSearchIndex | None = None
            try:
                if collection.get("load_policy") == "eager":
                    slot.state = CollectionIndexState.BUILDING
                    staging = self._new_index(collection)
                    stats = staging.stats()
                    if stats.live_rows != 0:
                        raise SearchIndexError("a new Collection index is not empty")
                result = operation()
            except Exception:
                if staging is not None:
                    staging.close()
                slot.state = CollectionIndexState.UNLOADED
                with self._lock:
                    if self._slots.get(collection_id) is slot:
                        self._slots.pop(collection_id, None)
                raise

            if staging is not None:
                slot.index = staging
                slot.applied_revision = int(result["search_revision"])
                slot.generation += 1
                slot.state = CollectionIndexState.READY
                slot.last_error = None
            else:
                slot.state = CollectionIndexState.UNLOADED
            return result

    def collection_created(self, collection: dict[str, Any]) -> None:
        self.validate_configuration(
            profile=str(collection["search_profile"]),
            dimension=int(collection["embedding_dimension"]),
        )
        if collection.get("load_policy") == "eager":
            self.ensure_ready(str(collection["id"]))

    @staticmethod
    def _record(value: Any) -> IndexRecord:
        return IndexRecord(
            vector_id=int(value.vector_id),
            person_numeric_id=int(value.person_numeric_id),
            embedding=value.embedding,
        )

    def _new_index(self, collection: dict[str, Any]) -> MutableSearchIndex:
        try:
            return self.backend.create_index(
                profile=str(collection["search_profile"]),
                dimension=int(collection["embedding_dimension"]),
                capacity_rows=int(collection["capacity_rows"]),
            )
        except SearchIndexError:
            raise
        except Exception as exc:
            raise SearchIndexError(
                f"search backend could not allocate the configured index: {exc}"
            ) from exc

    def _acknowledge(self, collection_id: str, through_seq: int) -> None:
        """Best-effort outbox cleanup after the index reaches a durable barrier."""

        try:
            self.repository.acknowledge_search_changes(collection_id, through_seq)
        except Exception:
            # The authoritative rows and current generation are already
            # consistent. Retaining an event is safe and a later successful
            # acknowledgement (or process restart) will remove it.
            LOGGER.warning(
                "unable_to_acknowledge_search_changes collection=%s through_seq=%s",
                collection_id,
                through_seq,
                exc_info=True,
            )

    def _create_staging(
        self, collection: dict[str, Any]
    ) -> tuple[MutableSearchIndex, int, int, int]:
        """Build a disposable generation without changing the active slot."""

        collection_id = str(collection["id"])
        staging = self._new_index(collection)
        try:
            source_revision: int | None = None
            source_max_seq: int | None = None
            loaded_rows = 0
            for batch in self.repository.iter_index_faces(
                collection_id, self.build_batch_rows
            ):
                if source_revision is None:
                    source_revision = int(batch.revision)
                    source_max_seq = int(batch.max_seq)
                elif source_revision != int(batch.revision):
                    raise SearchIndexError("SQLite index snapshot revision changed")
                elif source_max_seq != int(batch.max_seq):
                    raise SearchIndexError("SQLite index snapshot sequence changed")
                records = [self._record(face) for face in batch.faces]
                staging.add_batch(records)
                loaded_rows += len(records)
            if source_revision is None:
                raise SearchIndexError("repository returned no index snapshot")
            assert source_max_seq is not None
            committed_revision = self.repository.get_search_revision(collection_id)
            if committed_revision is None:
                raise KeyError(collection_id)
            if committed_revision != source_revision:
                raise SearchIndexError(
                    "Collection changed while its index was building; retry is required"
                )
            stats = staging.stats()
            if stats.live_rows != loaded_rows:
                raise SearchIndexError(
                    f"index build count mismatch: loaded={loaded_rows} "
                    f"native={stats.live_rows}"
                )
            return staging, source_revision, source_max_seq, loaded_rows
        except (KeyError, SearchIndexError):
            staging.close()
            raise
        except Exception as exc:
            staging.close()
            raise SearchIndexError(f"search index build failed: {exc}") from exc

    def _build_locked(
        self, slot: _CollectionSlot, collection: dict[str, Any]
    ) -> MutableSearchIndex:
        slot.state = (
            CollectionIndexState.REBUILDING
            if slot.index is not None
            else CollectionIndexState.BUILDING
        )
        staging: MutableSearchIndex | None = None
        try:
            staging, source_revision, source_max_seq, loaded_rows = self._create_staging(
                collection
            )
            stats = staging.stats()
            old = slot.index
            slot.index = staging
            slot.applied_revision = source_revision
            slot.generation += 1
            slot.state = CollectionIndexState.READY
            slot.last_error = None
            self._acknowledge(slot.collection_id, source_max_seq)
            if old is not None:
                try:
                    old.close()
                except Exception:
                    # The new generation is already authoritative. A best-effort
                    # close failure must not roll the slot back to FAILED.
                    LOGGER.warning(
                        "unable_to_close_retired_search_generation collection=%s",
                        slot.collection_id,
                        exc_info=True,
                    )
            LOGGER.info(
                "search_index_ready collection=%s generation=%s revision=%s rows=%s "
                "capacity=%s profile=%s backend=%s",
                slot.collection_id,
                slot.generation,
                slot.applied_revision,
                stats.live_rows,
                stats.capacity_rows,
                stats.profile,
                self.backend.name,
            )
            return slot.index
        except Exception as exc:
            if staging is not None and slot.index is not staging:
                try:
                    staging.close()
                except Exception:
                    LOGGER.warning(
                        "unable_to_close_failed_search_generation collection=%s",
                        slot.collection_id,
                        exc_info=True,
                    )
            slot.last_error = f"{type(exc).__name__}: {exc}"
            slot.state = CollectionIndexState.FAILED
            raise

    def _ensure_ready_locked(
        self, slot: _CollectionSlot, collection: dict[str, Any] | None = None
    ) -> MutableSearchIndex:
        item = collection or self.repository.get_collection(slot.collection_id)
        if item is None:
            raise KeyError(slot.collection_id)
        committed = int(item["search_revision"])
        if (
            slot.state == CollectionIndexState.READY
            and slot.index is not None
            and slot.applied_revision == committed
        ):
            return slot.index
        if slot.index is not None:
            slot.state = CollectionIndexState.DIRTY
        return self._build_locked(slot, item)

    def ensure_ready(self, collection_id: str) -> None:
        with self._locked_slot(collection_id) as slot:
            self._ensure_ready_locked(slot)

    def _apply_mutation_locked(
        self, slot: _CollectionSlot, mutation: SearchMutation
    ) -> None:
        if mutation.collection_id != slot.collection_id:
            raise SearchIndexError("search mutation Collection does not match its index")
        if slot.index is None or slot.state != CollectionIndexState.READY:
            self._ensure_ready_locked(slot)
            return
        if mutation.revision != slot.applied_revision + 1:
            slot.state = CollectionIndexState.DIRTY
            self._ensure_ready_locked(slot)
            return
        additions = [
            int(change.vector_id)
            for change in mutation.changes
            if change.operation == "add" and change.vector_id is not None
        ]
        deletions = [
            int(change.vector_id)
            for change in mutation.changes
            if change.operation == "delete" and change.vector_id is not None
        ]
        try:
            if additions:
                faces = self.repository.get_index_faces_by_vector_ids(
                    slot.collection_id, additions
                )
                if set(faces) != set(additions):
                    raise SearchIndexError("an added FaceSample is missing from SQLite")
                slot.index.add_batch([self._record(faces[value]) for value in additions])
            if deletions:
                slot.index.remove_batch(deletions)
            slot.applied_revision = mutation.revision
            current = self.repository.get_search_revision(slot.collection_id)
            if current != slot.applied_revision:
                raise SearchIndexError("index revision did not reach the SQLite barrier")
            self._acknowledge(slot.collection_id, mutation.max_seq)
        except Exception:
            slot.state = CollectionIndexState.DIRTY
            # A failed native mutation may have partially changed the handle.
            # Never reuse it; rebuild deterministically from SQLite.
            if slot.index is not None:
                slot.index.close()
                slot.index = None
            self._ensure_ready_locked(slot)

    def run_mutation(
        self,
        collection_id: str,
        operation: Callable[[], tuple[T, SearchMutation | None]],
        *,
        expected_additions: int = 0,
    ) -> T:
        with self._locked_slot(collection_id) as slot:
            index = self._ensure_ready_locked(slot)
            if expected_additions:
                stats = index.stats()
                if stats.live_rows + expected_additions > stats.capacity_rows:
                    raise SearchIndexCapacityError(
                        f"Collection {collection_id} capacity {stats.capacity_rows} "
                        "would be exceeded"
                    )
            result, mutation = operation()
            if mutation is None:
                return result
            try:
                self._apply_mutation_locked(slot, mutation)
            except Exception as exc:
                raise SearchMutationCommittedError(
                    collection_id,
                    mutation.revision,
                    str(exc),
                    committed_result=result,
                ) from exc
            return result

    def run_drop(
        self,
        collection_id: str,
        operation: Callable[[], tuple[T, SearchMutation | None]],
    ) -> T:
        with self._locked_slot(collection_id) as slot:
            result, mutation = operation()
            if mutation is None:
                return result
            self._acknowledge(collection_id, mutation.max_seq)
            if slot.index is not None:
                slot.state = CollectionIndexState.DRAINING
                try:
                    slot.index.close()
                except Exception:
                    # SQLite deletion is already committed. Native handles are
                    # derived state, so close failure must not block slot/crop
                    # cleanup or turn a durable delete into an ambiguous error.
                    LOGGER.warning(
                        "unable_to_close_dropped_search_generation collection=%s",
                        collection_id,
                        exc_info=True,
                    )
                slot.index = None
            slot.state = CollectionIndexState.CLOSED
            slot.applied_revision = 0
            # Remove while still holding slot.lock. A waiter that captured this
            # slot will fail _locked_slot's identity check and retry with the
            # replacement slot before recreating the same Collection ID.
            with self._lock:
                if self._slots.get(collection_id) is slot:
                    self._slots.pop(collection_id, None)
        return result

    def run_configuration_update(
        self,
        collection_id: str,
        operation: Callable[[], T],
        *,
        changes: dict[str, Any],
    ) -> T:
        with self._locked_slot(collection_id) as slot:
            current = self.repository.get_collection(collection_id)
            if current is None:
                raise KeyError(collection_id)
            effective_changes = {
                key: value
                for key, value in changes.items()
                if current.get(key) != value
            }
            target = {**current, **effective_changes}
            target_policy = str(target["load_policy"])
            explicitly_unload = (
                "load_policy" in effective_changes and target_policy == "lazy"
            )
            needs_staging = not explicitly_unload and (
                (target_policy == "eager" and slot.index is None)
                or ("capacity_rows" in effective_changes and slot.index is not None)
            )
            previous_state = slot.state
            staging: MutableSearchIndex | None = None
            staging_revision = 0
            staging_max_seq = 0
            if needs_staging:
                slot.state = (
                    CollectionIndexState.REBUILDING
                    if slot.index is not None
                    else CollectionIndexState.BUILDING
                )
                try:
                    staging, staging_revision, staging_max_seq, _ = self._create_staging(
                        target
                    )
                except Exception as exc:
                    slot.state = previous_state
                    slot.last_error = f"{type(exc).__name__}: {exc}"
                    raise

            try:
                result = operation()
            except Exception:
                if staging is not None:
                    staging.close()
                slot.state = previous_state
                raise

            if explicitly_unload:
                if slot.index is not None:
                    slot.index.close()
                    slot.index = None
                slot.state = CollectionIndexState.UNLOADED
                slot.applied_revision = 0
                slot.last_error = None
            elif staging is not None:
                old = slot.index
                slot.index = staging
                slot.applied_revision = staging_revision
                slot.generation += 1
                slot.state = CollectionIndexState.READY
                slot.last_error = None
                self._acknowledge(collection_id, staging_max_seq)
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        LOGGER.warning(
                            "unable_to_close_retired_search_generation collection=%s",
                            collection_id,
                            exc_info=True,
                        )
            return result

    def search(
        self,
        collection_id: str,
        query: np.ndarray,
        *,
        limit: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        stale_index: MutableSearchIndex | None = None
        stale_message: str | None = None
        with self._ready_read_slot(collection_id) as (slot, index):
            hits = index.search_persons(query, limit)
            mappings = self.repository.get_index_face_mappings(
                collection_id, [hit.vector_id for hit in hits]
            )
            if len(mappings) != len(hits):
                stale_index = index
                stale_message = "native result is missing from SQLite hydration"
            matches: list[dict[str, Any]] = []
            for hit in hits if stale_message is None else []:
                item = mappings[hit.vector_id]
                if int(item["person_numeric_id"]) != hit.person_numeric_id:
                    stale_index = index
                    stale_message = "native Person mapping disagrees with SQLite"
                    break
                # Match on the unrounded raw cosine. Rounding before this
                # comparison can flip decisions immediately below a threshold.
                cosine = float(np.clip(hit.cosine, -1.0, 1.0))
                if cosine < threshold:
                    continue
                matches.append(
                    {
                        "person": item["person"],
                        "similarity": cosine,
                        "matched_face_id": item["face_id"],
                    }
                )
        if stale_index is not None:
            self._mark_generation_dirty(collection_id, stale_index)
            raise SearchIndexError(stale_message or "search generation is inconsistent")
        matches.sort(key=lambda value: (-value["similarity"], value["person"]["id"]))
        return matches[:limit]

    def best_other_person(
        self,
        collection_id: str,
        query: np.ndarray,
        *,
        exclude_person_id: str,
    ) -> dict[str, Any] | None:
        """Return the strongest exact match outside ``exclude_person_id``.

        Strict enrollment review uses the Collection's configured in-memory
        profile, so its class-out score has exactly the same quantization and
        accumulation semantics as normal search. ``search_persons`` already
        groups FaceSamples by Person; requesting two hits is sufficient because
        at most one grouped hit can belong to the excluded Person.
        """

        stale_index: MutableSearchIndex | None = None
        stale_message: str | None = None
        result: dict[str, Any] | None = None
        with self._ready_read_slot(collection_id) as (slot, index):
            hits = index.search_persons(query, 2)
            mappings = self.repository.get_index_face_mappings(
                collection_id,
                [hit.vector_id for hit in hits],
                include_embedding=True,
            )
            if len(mappings) != len(hits):
                stale_index = index
                stale_message = "native result is missing from SQLite hydration"

            for hit in hits if stale_message is None else []:
                item = mappings[hit.vector_id]
                if int(item["person_numeric_id"]) != hit.person_numeric_id:
                    stale_index = index
                    stale_message = "native Person mapping disagrees with SQLite"
                    break
                person = item["person"]
                if str(person["id"]) == exclude_person_id:
                    continue
                result = {
                    "person_id": str(person["id"]),
                    "face_id": str(item["face_id"]),
                    # Public search results are clipped to cosine's range. The
                    # strict reviewer needs the profile's raw accumulator score
                    # on both sides of `same > other`, including rare INT8
                    # values slightly above 1 after quantization.
                    "similarity": profile_similarity(
                        index.profile, query, item["embedding"]
                    ),
                }
                break
        if stale_index is not None:
            self._mark_generation_dirty(collection_id, stale_index)
            raise SearchIndexError(stale_message or "search generation is inconsistent")
        return result

    def runtime_summary(self) -> dict[str, Any]:
        with self._lock:
            slots = list(self._slots.values())
        collections: list[dict[str, Any]] = []
        for slot in slots:
            with slot.lock.read():
                value: dict[str, Any] = {
                    "collection_id": slot.collection_id,
                    "state": slot.state.value,
                    "generation": slot.generation,
                    "applied_revision": slot.applied_revision,
                }
                if slot.last_error:
                    value["last_error"] = slot.last_error
                if slot.index is not None:
                    stats = slot.index.stats()
                    value.update(
                        {
                            "profile": stats.profile,
                            "live_rows": stats.live_rows,
                            "physical_rows": stats.physical_rows,
                            "capacity_rows": stats.capacity_rows,
                            "tombstone_rows": stats.tombstone_rows,
                            "reallocations": stats.reallocations,
                        }
                    )
                collections.append(value)
        return {
            **self.backend.runtime_summary(),
            "collections": sorted(collections, key=lambda value: value["collection_id"]),
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            slots = list(self._slots.values())
        for slot in slots:
            with slot.lock.write():
                slot.state = CollectionIndexState.DRAINING
                if slot.index is not None:
                    slot.index.close()
                    slot.index = None
                slot.state = CollectionIndexState.CLOSED
        self.backend.close()
