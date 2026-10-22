from .base import (
    SEARCH_PROFILES,
    IndexRecord,
    IndexStats,
    MutableSearchIndex,
    PersonHit,
    SearchBackend,
    SearchIndexCapacityError,
    SearchIndexError,
    SearchIndexStateError,
)
from .factory import NativeSearchBackend, ReferenceSearchBackend, create_search_backend
from .manager import (
    CollectionIndexState,
    SearchIndexManager,
    SearchMutationCommittedError,
)
from .native import NativeCapabilities, NativeSearchIndex, NativeSearchLibrary
from .reference import ReferenceSearchIndex, profile_similarity

__all__ = [
    "SEARCH_PROFILES",
    "IndexRecord",
    "IndexStats",
    "MutableSearchIndex",
    "NativeSearchBackend",
    "NativeCapabilities",
    "NativeSearchIndex",
    "NativeSearchLibrary",
    "PersonHit",
    "ReferenceSearchIndex",
    "profile_similarity",
    "ReferenceSearchBackend",
    "SearchBackend",
    "SearchIndexManager",
    "SearchIndexCapacityError",
    "SearchIndexError",
    "SearchIndexStateError",
    "SearchMutationCommittedError",
    "CollectionIndexState",
    "create_search_backend",
]
