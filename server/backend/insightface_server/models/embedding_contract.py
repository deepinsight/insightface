from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

EMBEDDING_CONTRACT_PREFIX = "ifsemb-v2-sha256:"


def _contract_digest(values: tuple[object, ...], prefix: str) -> str:
    canonical_bytes = json.dumps(
        values, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return prefix + hashlib.sha256(canonical_bytes).hexdigest()


def legacy_embedding_contract_id(
    model_id: str,
    model_version: str,
    model_digest: str,
    embedding_dimension: int,
    preprocessing_version: str,
) -> str:
    """Freeze an existing V1 identifier before migration drops model_version."""
    return _contract_digest(
        (model_id, model_version, model_digest.lower(), int(embedding_dimension), preprocessing_version),
        "ifsemb-v1-sha256:",
    )


def embedding_contract_id(
    *,
    model_id: str,
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
        model_digest.lower(),
        int(embedding_dimension),
        preprocessing_version,
    )
    return _contract_digest(canonical_tuple, EMBEDDING_CONTRACT_PREFIX)


def embedding_contract_id_for_collection(collection: Mapping[str, Any]) -> str:
    """Reuse a persisted identifier, or derive one for a new Collection."""

    if collection.get("embedding_contract_id"):
        return str(collection["embedding_contract_id"])

    return embedding_contract_id(
        model_id=str(collection["model_id"]),
        model_digest=str(collection["model_digest"]),
        embedding_dimension=int(collection["embedding_dimension"]),
        preprocessing_version=str(collection["preprocessing_version"]),
    )
