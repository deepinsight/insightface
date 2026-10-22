from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path

from ..errors import bad_request


def _encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _encode(decoded) != value:
        raise ValueError("non-canonical base64")
    return decoded


class CursorCodec:
    def __init__(self, secret_path: Path):
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        created = os.urandom(32)
        try:
            descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(created)
            self._secret = created
        except FileExistsError:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(secret_path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                self._secret = stream.read()
        secret_path.chmod(0o600)
        if len(self._secret) < 32:
            raise RuntimeError("Cursor secret must contain at least 32 bytes")

    def encode(self, scope: str, after: str) -> str:
        payload = json.dumps(
            {"v": 1, "scope": scope, "after": after},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        signature = hmac.new(self._secret, payload, hashlib.sha256).digest()[:16]
        return f"{_encode(payload)}.{_encode(signature)}"

    def decode(self, token: str | None, scope: str) -> str:
        if not token:
            return ""
        try:
            if len(token) > 2048:
                raise ValueError("token too long")
            encoded_payload, encoded_signature = token.split(".", 1)
            payload = _decode(encoded_payload)
            signature = _decode(encoded_signature)
            expected = hmac.new(self._secret, payload, hashlib.sha256).digest()[:16]
            value = json.loads(payload)
            if not hmac.compare_digest(signature, expected):
                raise ValueError("signature")
            if value != {"v": 1, "scope": scope, "after": value.get("after")}:
                raise ValueError("scope")
            after = value["after"]
            if not isinstance(after, str):
                raise ValueError("after")
            return after
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise bad_request("The cursor is invalid.", code="invalid_cursor") from exc
