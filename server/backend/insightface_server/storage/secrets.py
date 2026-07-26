from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _encode(decoded) != value:
        raise ValueError("non-canonical base64")
    return decoded


class SecretCodec:
    """Encrypt recoverable local secrets with a key stored beside the database."""

    def __init__(self, secret_path: Path):
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        created = os.urandom(32)
        try:
            descriptor = os.open(
                secret_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(created)
            self._secret = created
        except FileExistsError:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(secret_path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                self._secret = stream.read()
        secret_path.chmod(0o600)
        if len(self._secret) != 32:
            raise RuntimeError("Monitor credential key must contain exactly 32 bytes")
        self._cipher = AESGCM(self._secret)

    def encrypt(self, value: str, *, scope: str) -> str:
        nonce = os.urandom(12)
        ciphertext = self._cipher.encrypt(
            nonce,
            value.encode("utf-8"),
            scope.encode("utf-8"),
        )
        return f"v1.{_encode(nonce + ciphertext)}"

    def decrypt(self, value: str, *, scope: str) -> str:
        try:
            version, encoded = value.split(".", 1)
            if version != "v1":
                raise ValueError("unsupported version")
            payload = _decode(encoded)
            if len(payload) < 13:
                raise ValueError("truncated value")
            plaintext = self._cipher.decrypt(
                payload[:12],
                payload[12:],
                scope.encode("utf-8"),
            )
            return plaintext.decode("utf-8")
        except Exception as exc:
            raise RuntimeError("Stored Monitor credentials could not be decrypted") from exc
