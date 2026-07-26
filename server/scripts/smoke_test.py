#!/usr/bin/env python3
"""Small dependency-free health and collection API smoke test."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request(
    base: str,
    path: str,
    key: str | None,
    payload: dict | None = None,
    *,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    headers = {"accept": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = None
    if payload is not None:
        if body is not None:
            raise ValueError("payload and body are mutually exclusive")
        headers["content-type"] = "application/json"
        data = json.dumps(payload).encode()
    elif body is not None:
        if content_type is None:
            raise ValueError("content_type is required with body")
        headers["content-type"] = content_type
        data = body
    call = urllib.request.Request(base.rstrip("/") + path, headers=headers, data=data)
    try:
        with urllib.request.urlopen(call, timeout=timeout) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    return json.loads(response_body) if response_body else {}


def multipart_image(
    path: Path,
    *,
    fields: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    boundary = f"insightface-release-{secrets.token_hex(16)}"
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts = []
    for name, value in sorted((fields or {}).items()):
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="image"; '
            'filename="release-smoke-image"\r\n'
            f"Content-Type: {media_type}\r\n\r\n"
        ).encode()
        + path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def require_single_detected_face(response: dict[str, Any]) -> None:
    faces = response.get("faces")
    if not isinstance(faces, list):
        raise SystemExit("the release smoke detect response has no face list")
    if len(faces) != 1:
        raise SystemExit(
            f"the release smoke image must detect exactly one face; found {len(faces)}"
        )


def inference_report(response: dict[str, Any]) -> dict[str, Any]:
    faces = response.get("faces")
    if not isinstance(faces, list) or not faces:
        raise SystemExit("the release smoke image did not produce a face")
    if len(faces) != 1:
        raise SystemExit(
            f"the release smoke image must produce exactly one face; found {len(faces)}"
        )
    face = faces[0]
    embedding = face.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise SystemExit("the release smoke image did not produce an embedding")
    values = [float(value) for value in embedding]
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("the release smoke embedding contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, abs_tol=1.0e-5):
        raise SystemExit(f"the release smoke embedding is not normalized: {norm}")
    return {
        "face_count": len(faces),
        "bbox": face["bbox"],
        "detection_score": face["detection_score"],
        "embedding_dimension": len(values),
        "embedding_norm": norm,
        "embedding": values,
    }


def redacted_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return log-safe evidence without face geometry or feature material."""
    printable = json.loads(json.dumps(report))
    if "inference" in printable:
        inference = printable["inference"]
        printable["inference"] = {
            "face_count": inference["face_count"],
            "embedding_dimension": inference["embedding_dimension"],
            "embedding_norm": inference["embedding_norm"],
        }
    return printable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.getenv("INSIGHTFACE_API_KEY"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    health = request(args.base_url, "/v1/health", None)
    system = request(args.base_url, "/v1/system", args.api_key)
    report: dict[str, Any] = {"health": health, "system": system}
    if args.image is not None:
        if not args.image.is_file():
            raise SystemExit(f"release smoke image does not exist: {args.image}")
        detect_body, detect_content_type = multipart_image(
            args.image,
            fields={"max_faces": "2"},
        )
        detection = request(
            args.base_url,
            "/v1/detect",
            args.api_key,
            body=detect_body,
            content_type=detect_content_type,
            timeout=65.0,
        )
        require_single_detected_face(detection)
        body, content_type = multipart_image(args.image)
        response = request(
            args.base_url,
            "/v1/embeddings",
            args.api_key,
            body=body,
            content_type=content_type,
            timeout=65.0,
        )
        report["inference"] = inference_report(response)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.output.chmod(0o600)
    print(json.dumps(redacted_report(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
