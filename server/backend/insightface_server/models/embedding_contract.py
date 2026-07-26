from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

EMBEDDING_CONTRACT_PREFIX = "ifsemb-v1-sha256:"


def embedding_contract_id(
    *,
    model_id: str,
    model_version: str,
    model_digest: str,
    embedding_dimension: int,
    preprocessing_version: str,
) -> str:
    """Return the stable identity of one Collection embedding contract.

    The ordered JSON tuple and prefix are deliberately versioned. A future change
    to alignment, preprocessing identity, or canonicalization must use a new
    prefix instead of changing the meaning of existing identifiers.
    """

    canonical_tuple = (
        model_id,
        model_version,
        model_digest.lower(),
        int(embedding_dimension),
        preprocessing_version,
    )
    canonical_bytes = json.dumps(
        canonical_tuple,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    return EMBEDDING_CONTRACT_PREFIX + digest


def embedding_contract_id_for_collection(collection: Mapping[str, Any]) -> str:
    """Derive an embedding contract identifier from Collection model fields."""

    return embedding_contract_id(
        model_id=str(collection["model_id"]),
        model_version=str(collection["model_version"]),
        model_digest=str(collection["model_digest"]),
        embedding_dimension=int(collection["embedding_dimension"]),
        preprocessing_version=str(collection["preprocessing_version"]),
    )
