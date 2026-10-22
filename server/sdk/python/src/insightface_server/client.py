"""Synchronous client for the InsightFace Server v1 REST API."""

from __future__ import annotations

import json
import math
import mimetypes
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Dict,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import quote

import httpx

from .exceptions import (
    AuthenticationError,
    ConflictError,
    InsightFaceServerError,
    NotFoundError,
    PayloadTooLargeError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    TransportError,
    ValidationError,
)
from .results import (
    ApiResult,
    CollectionPage,
    CollectionResult,
    CompareResult,
    DetectResult,
    EmbeddingsResult,
    FacePage,
    FaceRegistrationResult,
    HealthResult,
    ModelsResult,
    MonitorEventPage,
    MonitorPage,
    MonitorResult,
    MonitorStateResult,
    PersonPage,
    PersonRegistrationResult,
    PersonResult,
    SearchResult,
    SystemResult,
)

ImageInput = Union[str, Path, bytes, bytearray, memoryview, BinaryIO]
Upload = Tuple[str, bytes, str]
ResultT = TypeVar("ResultT", bound=ApiResult)
SearchProfile = Literal[
    "fp32_v1", "fp16_v1", "bf16_v1", "int8_x736_v1", "int8_x1000_v1"
]
SearchLoadPolicy = Literal["eager", "lazy"]
ReviewMode = Literal["off", "standard", "strict"]
EmbeddingMode = Literal["server", "external_trusted"]
SingleFaceSelection = Literal["largest", "center_largest"]
ExternalEmbedding = Iterable[float]
_UNSET = object()


class Client:
    """A small synchronous client for InsightFace Server.

    Args:
        base_url: Server origin, for example ``http://localhost:8080``.
        api_key: Optional bearer token. Omit only when server authentication is
            explicitly disabled for development.
        timeout: Overall HTTP timeout in seconds or an ``httpx.Timeout``.
        transport: Optional httpx transport, primarily useful for tests.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = 65.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        normalized = base_url.rstrip("/")
        if not normalized:
            raise ValueError("base_url must not be empty")
        headers = {
            "Accept": "application/json",
            "User-Agent": "insightface-server-python/0.3.1",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            base_url=normalized,
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def health(self) -> HealthResult:
        return self._request("GET", "/v1/health", result_type=HealthResult)

    def system(self) -> SystemResult:
        return self._request("GET", "/v1/system", result_type=SystemResult)

    def models(self) -> ModelsResult:
        return self._request("GET", "/v1/models", result_type=ModelsResult)

    def create_monitor(
        self,
        monitor_id: str,
        *,
        name: str,
        rtsp_url: str,
        collection: str,
        description: str = "",
        enabled: bool = True,
        inference_fps: float = 2.0,
        match_threshold: Optional[float] = None,
        event_buffer_size: int = 1000,
        confirm_frames: int = 3,
        absence_timeout_seconds: float = 3.0,
        cooldown_seconds: float = 10.0,
        emit_unknown: bool = True,
        preview_enabled: bool = False,
    ) -> MonitorResult:
        return self._request(
            "POST",
            "/v1/monitors",
            json={
                "id": monitor_id,
                "name": name,
                "description": description,
                "enabled": enabled,
                "source": {"type": "rtsp", "url": rtsp_url},
                "collection_id": collection,
                "inference_fps": inference_fps,
                "match_threshold": match_threshold,
                "event_buffer_size": event_buffer_size,
                "event_policy": {
                    "confirm_frames": confirm_frames,
                    "absence_timeout_seconds": absence_timeout_seconds,
                    "cooldown_seconds": cooldown_seconds,
                    "emit_unknown": emit_unknown,
                },
                "preview_enabled": preview_enabled,
            },
            result_type=MonitorResult,
        )

    def list_monitors(
        self,
        *,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> MonitorPage:
        return self._request(
            "GET",
            "/v1/monitors",
            params=self._without_none(limit=limit, cursor=cursor),
            result_type=MonitorPage,
        )

    def get_monitor(self, monitor_id: str) -> MonitorResult:
        return self._request(
            "GET",
            self._monitor_path(monitor_id),
            result_type=MonitorResult,
        )

    def update_monitor(
        self,
        monitor_id: str,
        *,
        name: object = _UNSET,
        description: object = _UNSET,
        enabled: object = _UNSET,
        rtsp_url: object = _UNSET,
        collection: object = _UNSET,
        inference_fps: object = _UNSET,
        match_threshold: object = _UNSET,
        event_buffer_size: object = _UNSET,
        confirm_frames: object = _UNSET,
        absence_timeout_seconds: object = _UNSET,
        cooldown_seconds: object = _UNSET,
        emit_unknown: object = _UNSET,
        preview_enabled: object = _UNSET,
    ) -> MonitorResult:
        payload = {
            key: value
            for key, value in {
                "name": name,
                "description": description,
                "enabled": enabled,
                "collection_id": collection,
                "inference_fps": inference_fps,
                "match_threshold": match_threshold,
                "event_buffer_size": event_buffer_size,
                "preview_enabled": preview_enabled,
            }.items()
            if value is not _UNSET
        }
        if rtsp_url is not _UNSET:
            payload["source"] = {"type": "rtsp", "url": rtsp_url}
        policy = {
            key: value
            for key, value in {
                "confirm_frames": confirm_frames,
                "absence_timeout_seconds": absence_timeout_seconds,
                "cooldown_seconds": cooldown_seconds,
                "emit_unknown": emit_unknown,
            }.items()
            if value is not _UNSET
        }
        if policy:
            payload["event_policy"] = policy
        if not payload:
            raise ValueError("at least one field must be supplied")
        return self._request(
            "PATCH",
            self._monitor_path(monitor_id),
            json=payload,
            result_type=MonitorResult,
        )

    def delete_monitor(self, monitor_id: str) -> ApiResult:
        return self._request("DELETE", self._monitor_path(monitor_id))

    def monitor_state(self, monitor_id: str) -> MonitorStateResult:
        return self._request(
            "GET",
            f"{self._monitor_path(monitor_id)}/state",
            result_type=MonitorStateResult,
        )

    def monitor_events(
        self,
        monitor_id: str,
        *,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> MonitorEventPage:
        return self._request(
            "GET",
            f"{self._monitor_path(monitor_id)}/events",
            params=self._without_none(limit=limit, cursor=cursor),
            result_type=MonitorEventPage,
        )

    def detect(
        self,
        image: ImageInput,
        *,
        max_faces: Optional[int] = None,
        collection: Optional[str] = None,
    ) -> DetectResult:
        data = self._without_none(max_faces=max_faces, collection_id=collection)
        return self._request(
            "POST",
            "/v1/detect",
            data=data,
            files={"image": self._prepare_image(image, "image.jpg")},
            result_type=DetectResult,
        )

    def compare(
        self,
        source: ImageInput,
        target: ImageInput,
        *,
        threshold: Optional[float] = None,
        collection: Optional[str] = None,
    ) -> CompareResult:
        return self._request(
            "POST",
            "/v1/compare",
            data=self._without_none(threshold=threshold, collection_id=collection),
            files={
                "source": self._prepare_image(source, "source.jpg"),
                "target": self._prepare_image(target, "target.jpg"),
            },
            result_type=CompareResult,
        )

    def embeddings(
        self,
        image: ImageInput,
        *,
        collection: Optional[str] = None,
    ) -> EmbeddingsResult:
        return self._request(
            "POST",
            "/v1/embeddings",
            data=self._without_none(collection_id=collection),
            files={"image": self._prepare_image(image, "image.jpg")},
            result_type=EmbeddingsResult,
        )

    def create_collection(
        self,
        collection_id: str,
        *,
        name: str,
        description: str = "",
        threshold: float = 0.4,
        metadata: Optional[Mapping[str, Any]] = None,
        save_face_crops: Optional[bool] = None,
        search_profile: Optional[SearchProfile] = None,
        capacity_rows: Optional[int] = None,
        max_faces_per_person: Optional[int] = None,
        load_policy: Optional[SearchLoadPolicy] = None,
        detector_input_sizes: Optional[Iterable[Tuple[int, int]]] = None,
        detector_threshold: Optional[float] = None,
        detector_nms_threshold: Optional[float] = None,
        single_face_selection: Optional[SingleFaceSelection] = None,
    ) -> CollectionResult:
        payload = {
            "id": collection_id,
            "name": name,
            "description": description,
            "threshold": threshold,
            "metadata": dict(metadata or {}),
        }
        if save_face_crops is not None:
            payload["save_face_crops"] = save_face_crops
        search = self._without_none(
            profile=search_profile,
            capacity_rows=capacity_rows,
            max_faces_per_person=max_faces_per_person,
            load_policy=load_policy,
        )
        if search:
            payload["search"] = search
        detection = self._without_none(
            input_sizes=(
                [list(size) for size in detector_input_sizes]
                if detector_input_sizes is not None
                else None
            ),
            threshold=detector_threshold,
            nms_threshold=detector_nms_threshold,
            single_face_selection=single_face_selection,
        )
        if detection:
            payload["detection"] = detection
        return self._request(
            "POST", "/v1/collections", json=payload, result_type=CollectionResult
        )

    def list_collections(
        self, *, limit: int = 50, cursor: Optional[str] = None
    ) -> CollectionPage:
        return self._request(
            "GET",
            "/v1/collections",
            params=self._without_none(limit=limit, cursor=cursor),
            result_type=CollectionPage,
        )

    def get_collection(self, collection_id: str) -> CollectionResult:
        return self._request(
            "GET", self._collection_path(collection_id), result_type=CollectionResult
        )

    def update_collection(
        self,
        collection_id: str,
        *,
        name: object = _UNSET,
        description: object = _UNSET,
        threshold: object = _UNSET,
        metadata: object = _UNSET,
        save_face_crops: object = _UNSET,
        capacity_rows: object = _UNSET,
        max_faces_per_person: object = _UNSET,
        load_policy: object = _UNSET,
        detector_input_sizes: object = _UNSET,
        detector_threshold: object = _UNSET,
        detector_nms_threshold: object = _UNSET,
        single_face_selection: object = _UNSET,
    ) -> CollectionResult:
        payload = {
            key: value
            for key, value in {
                "name": name,
                "description": description,
                "threshold": threshold,
                "metadata": metadata,
                "save_face_crops": save_face_crops,
            }.items()
            if value is not _UNSET
        }
        search = {
            key: value
            for key, value in {
                "capacity_rows": capacity_rows,
                "max_faces_per_person": max_faces_per_person,
                "load_policy": load_policy,
            }.items()
            if value is not _UNSET
        }
        if search:
            payload["search"] = search
        detection = {
            key: value
            for key, value in {
                "input_sizes": detector_input_sizes,
                "threshold": detector_threshold,
                "nms_threshold": detector_nms_threshold,
                "single_face_selection": single_face_selection,
            }.items()
            if value is not _UNSET
        }
        if detection:
            if (
                "input_sizes" in detection
                and detection["input_sizes"] is not None
            ):
                input_sizes = cast(
                    Iterable[Tuple[int, int]], detection["input_sizes"]
                )
                detection["input_sizes"] = [
                    list(size) for size in input_sizes
                ]
            payload["detection"] = detection
        if not payload:
            raise ValueError("at least one field must be supplied")
        return self._request(
            "PATCH",
            self._collection_path(collection_id),
            json=payload,
            result_type=CollectionResult,
        )

    def delete_collection(self, collection_id: str, *, force: bool = False) -> ApiResult:
        return self._request(
            "DELETE",
            self._collection_path(collection_id),
            params={"force": str(force).lower()},
        )

    def create_person(
        self,
        collection: str,
        *,
        images: Iterable[ImageInput],
        person_id: Optional[str] = None,
        name: Optional[str] = None,
        external_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        review_mode: ReviewMode = "off",
        external_embeddings: Optional[Iterable[ExternalEmbedding]] = None,
        embedding_contract_id: Optional[str] = None,
    ) -> PersonRegistrationResult:
        uploads = [
            self._prepare_image(image, f"image-{index}.jpg")
            for index, image in enumerate(images)
        ]
        if not uploads:
            raise ValueError("images must contain at least one image")
        data = self._without_none(id=person_id, name=name, external_id=external_id)
        data["metadata"] = json.dumps(
            dict(metadata or {}), ensure_ascii=False, separators=(",", ":")
        )
        data.update(
            self._enrollment_fields(
                image_count=len(uploads),
                review_mode=review_mode,
                external_embeddings=external_embeddings,
                embedding_contract_id=embedding_contract_id,
            )
        )
        files = [("images", upload) for upload in uploads]
        return self._request(
            "POST",
            f"{self._collection_path(collection)}/persons",
            data=data,
            files=files,
            result_type=PersonRegistrationResult,
        )

    def add_person(
        self,
        collection: str,
        *,
        images: Iterable[ImageInput],
        person_id: Optional[str] = None,
        name: Optional[str] = None,
        external_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        review_mode: ReviewMode = "off",
        external_embeddings: Optional[Iterable[ExternalEmbedding]] = None,
        embedding_contract_id: Optional[str] = None,
    ) -> PersonRegistrationResult:
        """Alias for :meth:`create_person` matching the common SDK workflow."""

        return self.create_person(
            collection,
            images=images,
            person_id=person_id,
            name=name,
            external_id=external_id,
            metadata=metadata,
            review_mode=review_mode,
            external_embeddings=external_embeddings,
            embedding_contract_id=embedding_contract_id,
        )

    def list_persons(
        self,
        collection: str,
        *,
        limit: int = 50,
        cursor: Optional[str] = None,
        search: Optional[str] = None,
    ) -> PersonPage:
        return self._request(
            "GET",
            f"{self._collection_path(collection)}/persons",
            params=self._without_none(limit=limit, cursor=cursor, search=search),
            result_type=PersonPage,
        )

    def get_person(self, collection: str, person_id: str) -> PersonResult:
        return self._request(
            "GET", self._person_path(collection, person_id), result_type=PersonResult
        )

    def update_person(
        self,
        collection: str,
        person_id: str,
        *,
        name: object = _UNSET,
        external_id: object = _UNSET,
        metadata: object = _UNSET,
    ) -> PersonResult:
        payload = self._patch_payload(name=name, external_id=external_id, metadata=metadata)
        return self._request(
            "PATCH",
            self._person_path(collection, person_id),
            json=payload,
            result_type=PersonResult,
        )

    def delete_person(self, collection: str, person_id: str) -> ApiResult:
        return self._request("DELETE", self._person_path(collection, person_id))

    def list_faces(
        self,
        collection: str,
        person_id: str,
        *,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> FacePage:
        return self._request(
            "GET",
            f"{self._person_path(collection, person_id)}/faces",
            params=self._without_none(limit=limit, cursor=cursor),
            result_type=FacePage,
        )

    def add_faces(
        self,
        collection: str,
        person_id: str,
        images: Iterable[ImageInput],
        *,
        review_mode: ReviewMode = "off",
        external_embeddings: Optional[Iterable[ExternalEmbedding]] = None,
        embedding_contract_id: Optional[str] = None,
    ) -> FaceRegistrationResult:
        """Register one or more additional samples for an existing person."""

        uploads = [
            self._prepare_image(image, f"image-{index}.jpg")
            for index, image in enumerate(images)
        ]
        if not uploads:
            raise ValueError("images must contain at least one image")
        data = self._enrollment_fields(
            image_count=len(uploads),
            review_mode=review_mode,
            external_embeddings=external_embeddings,
            embedding_contract_id=embedding_contract_id,
        )
        return self._request(
            "POST",
            f"{self._person_path(collection, person_id)}/faces",
            data=data,
            files=[("images", upload) for upload in uploads],
            result_type=FaceRegistrationResult,
        )

    def delete_face(self, collection: str, person_id: str, face_id: str) -> ApiResult:
        return self._request(
            "DELETE",
            f"{self._person_path(collection, person_id)}/faces/{quote(face_id, safe='')}",
        )

    def get_face_crop(self, collection: str, person_id: str, face_id: str) -> bytes:
        """Download a stored 112x112 bounding-box face crop as JPEG bytes.

        A crop exists only when the Collection had ``save_face_crops`` enabled
        when this FaceSample was registered. Original uploads are never
        returned by this endpoint.
        """

        path = (
            f"{self._person_path(collection, person_id)}/faces/"
            f"{quote(face_id, safe='')}/image"
        )
        try:
            response = self._http.get(path, headers={"Accept": "image/jpeg"})
        except httpx.HTTPError as exc:
            raise TransportError(
                "Unable to complete the request.", code="transport_error"
            ) from exc
        if not response.is_success:
            self._raise_api_error(response)
        media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if media_type != "image/jpeg" or not response.content:
            raise ServerError(
                "Server returned an invalid face crop response.",
                code="invalid_response",
                status_code=response.status_code,
                request_id=response.headers.get("x-request-id"),
            )
        return bytes(response.content)

    def search(
        self,
        collection: str,
        image: ImageInput,
        *,
        limit: int = 5,
        threshold: Optional[float] = None,
    ) -> SearchResult:
        data = self._without_none(limit=limit, threshold=threshold)
        return self._request(
            "POST",
            f"{self._collection_path(collection)}/search",
            data=data,
            files={"image": self._prepare_image(image, "image.jpg")},
            result_type=SearchResult,
        )

    @staticmethod
    def _collection_path(collection_id: str) -> str:
        return f"/v1/collections/{quote(collection_id, safe='')}"

    @staticmethod
    def _monitor_path(monitor_id: str) -> str:
        return f"/v1/monitors/{quote(monitor_id, safe='')}"

    @classmethod
    def _person_path(cls, collection: str, person_id: str) -> str:
        return f"{cls._collection_path(collection)}/persons/{quote(person_id, safe='')}"

    @staticmethod
    def _without_none(**values: Any) -> Dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}

    @staticmethod
    def _enrollment_fields(
        *,
        image_count: int,
        review_mode: ReviewMode,
        external_embeddings: Optional[Iterable[ExternalEmbedding]],
        embedding_contract_id: Optional[str],
    ) -> Dict[str, str]:
        """Build multipart fields for server or trusted-external enrollment.

        Conversion uses only Python's iteration protocol, so lists, tuples and
        NumPy arrays work without making NumPy an SDK dependency.
        """

        fields: Dict[str, str] = {"review_mode": review_mode}
        if external_embeddings is None:
            if embedding_contract_id is not None:
                raise ValueError(
                    "embedding_contract_id requires external_embeddings"
                )
            fields["embedding_mode"] = "server"
            return fields

        contract_id = str(embedding_contract_id or "").strip()
        if not contract_id:
            raise ValueError(
                "embedding_contract_id is required with external_embeddings"
            )

        vectors = []
        for index, vector in enumerate(external_embeddings):
            try:
                values = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"external_embeddings[{index}] must be a numeric vector"
                ) from exc
            if not values:
                raise ValueError(
                    f"external_embeddings[{index}] must not be empty"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"external_embeddings[{index}] contains NaN or infinity"
                )
            norm = math.sqrt(sum(value * value for value in values))
            if not math.isfinite(norm) or abs(norm - 1.0) > 2e-4:
                raise ValueError(
                    f"external_embeddings[{index}] must be L2-normalized "
                    "within 0.0002 of unit norm"
                )
            vectors.append(values)

        if len(vectors) != image_count:
            raise ValueError(
                "external_embeddings count must equal images count "
                f"({len(vectors)} != {image_count})"
            )

        fields.update(
            embedding_mode="external_trusted",
            embedding_contract_id=contract_id,
            external_embeddings=json.dumps(vectors, separators=(",", ":")),
        )
        return fields

    @staticmethod
    def _patch_payload(**values: object) -> Dict[str, Any]:
        payload = {key: value for key, value in values.items() if value is not _UNSET}
        if not payload:
            raise ValueError("at least one field must be supplied")
        return payload

    @staticmethod
    def _prepare_image(image: ImageInput, fallback_name: str) -> Upload:
        filename = fallback_name
        if isinstance(image, (str, Path)):
            path = Path(image)
            filename = path.name
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"cannot read image file: {path}") from exc
        elif isinstance(image, (bytes, bytearray, memoryview)):
            content = bytes(image)
        elif hasattr(image, "read"):
            stream = image
            stream_name = getattr(stream, "name", None)
            if isinstance(stream_name, str):
                filename = Path(stream_name).name
            old_position: Optional[int] = None
            if hasattr(stream, "tell"):
                try:
                    old_position = stream.tell()
                except (OSError, ValueError):
                    # Non-seekable streams are still valid image inputs.
                    old_position = None
            try:
                content = stream.read()
            except (OSError, ValueError) as exc:
                raise ValueError("cannot read image stream") from exc
            if old_position is not None and hasattr(stream, "seek"):
                try:
                    stream.seek(old_position)
                except (OSError, ValueError):
                    pass
            if not isinstance(content, (bytes, bytearray, memoryview)):
                raise TypeError("image file-like object must return bytes")
            content = bytes(content)
        else:
            raise TypeError("image must be a path, bytes, or binary file-like object")
        if not content:
            raise ValueError("image must not be empty")
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return filename, content, content_type

    def _request(
        self,
        method: str,
        path: str,
        *,
        result_type: Type[ResultT] = ApiResult,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> ResultT:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise TransportError(
                "Unable to complete the request.", code="transport_error"
            ) from exc
        if not response.is_success:
            self._raise_api_error(response)
        body: Dict[str, Any]
        if response.status_code == 204 or not response.content:
            body = {}
        else:
            try:
                decoded = response.json()
            except ValueError as exc:
                raise ServerError(
                    "Server returned an invalid JSON response.",
                    code="invalid_response",
                    status_code=response.status_code,
                    request_id=response.headers.get("x-request-id"),
                ) from exc
            if not isinstance(decoded, dict):
                raise ServerError(
                    "Server returned an unexpected JSON response.",
                    code="invalid_response",
                    status_code=response.status_code,
                    request_id=response.headers.get("x-request-id"),
                )
            body = decoded
        request_id = response.headers.get("x-request-id")
        if not request_id:
            candidate = body.get("request_id")
            request_id = candidate if isinstance(candidate, str) else None
        return result_type(body, response.status_code, request_id)

    @staticmethod
    def _raise_api_error(response: httpx.Response) -> None:
        request_id = response.headers.get("x-request-id")
        code = "http_error"
        message = f"Server returned HTTP {response.status_code}."
        details: Mapping[str, Any] = {}
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            candidate_request_id = body.get("request_id")
            if not request_id and isinstance(candidate_request_id, str):
                request_id = candidate_request_id
            error = body.get("error")
            if isinstance(error, dict):
                if isinstance(error.get("code"), str):
                    code = error["code"]
                if isinstance(error.get("message"), str):
                    message = error["message"]
                if isinstance(error.get("details"), dict):
                    details = error["details"]
        exception_type: Type[InsightFaceServerError]
        exception_type = {
            400: ValidationError,
            401: AuthenticationError,
            403: AuthenticationError,
            404: NotFoundError,
            409: ConflictError,
            413: PayloadTooLargeError,
            422: ValidationError,
            429: RateLimitError,
            503: ServiceUnavailableError,
        }.get(response.status_code, ServerError)
        raise exception_type(
            message,
            code=code,
            status_code=response.status_code,
            request_id=request_id,
            details=details,
        )
