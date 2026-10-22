from __future__ import annotations

import importlib.util
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import httpx
import pytest


def load_sdk() -> tuple[ModuleType, ModuleType]:
    """Load the SDK under a private name so backend tests can share a process.

    The server application and installable client intentionally both use the
    public package name ``insightface_server``, but they are never installed in
    the same runtime environment. Using a private alias here prevents SDK test
    collection from shadowing the backend package in an all-tests pytest run.
    """

    package_name = "_insightface_server_sdk_under_test"
    package_dir = Path(__file__).parents[2] / "sdk" / "python" / "src" / "insightface_server"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the SDK package for tests")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    results = sys.modules[f"{package_name}.results"]
    return package, results


sdk, sdk_results = load_sdk()
Client = sdk.Client
AuthenticationError = sdk.AuthenticationError
ConflictError = sdk.ConflictError
NotFoundError = sdk.NotFoundError
PayloadTooLargeError = sdk.PayloadTooLargeError
RateLimitError = sdk.RateLimitError
ServerError = sdk.ServerError
ServiceUnavailableError = sdk.ServiceUnavailableError
TransportError = sdk.TransportError
ValidationError = sdk.ValidationError
CollectionPage = sdk_results.CollectionPage
CompareResult = sdk_results.CompareResult
DetectResult = sdk_results.DetectResult
FaceRegistrationResult = sdk_results.FaceRegistrationResult
MonitorEventPage = sdk_results.MonitorEventPage
MonitorPage = sdk_results.MonitorPage
PersonRegistrationResult = sdk_results.PersonRegistrationResult
SearchResult = sdk_results.SearchResult

Handler = Callable[[httpx.Request], httpx.Response]


def client_for(handler: Handler, *, api_key: str | None = "secret") -> Client:
    return Client(
        "http://server.test/",
        api_key=api_key,
        transport=httpx.MockTransport(handler),
    )


def response(
    status: int,
    payload: object | None = None,
    *,
    request_id: str = "req-123",
) -> httpx.Response:
    headers = {"x-request-id": request_id}
    if payload is None:
        return httpx.Response(status, headers=headers)
    return httpx.Response(status, headers=headers, json=payload)


def test_health_sends_auth_and_exposes_request_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == httpx.URL("http://server.test/v1/health")
        assert request.headers["authorization"] == "Bearer secret"
        assert request.headers["user-agent"] == "insightface-server-python/0.3.1"
        return response(200, {"status": "ok"})

    with client_for(handler) as client:
        result = client.health()

    assert result["status"] == "ok"
    assert result.status_code == 200
    assert result.request_id == "req-123"
    assert result.to_dict() == {"status": "ok"}


def test_auth_header_is_optional_for_development_mode() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return response(200, {"status": "ok"})

    with client_for(handler, api_key=None) as client:
        client.health()


def test_default_timeout_exceeds_the_server_request_deadline() -> None:
    with client_for(lambda _request: response(200, {"status": "ok"})) as client:
        assert client._http.timeout.connect == 65.0
        assert client._http.timeout.read == 65.0
        assert client._http.timeout.write == 65.0
        assert client._http.timeout.pool == 65.0


def test_detect_accepts_path_and_multipart_fields(tmp_path: Path) -> None:
    image = tmp_path / "portrait.png"
    image.write_bytes(b"fake-png-content")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/detect"
        content_type = request.headers["content-type"]
        assert content_type.startswith("multipart/form-data; boundary=")
        body = request.content
        assert b'name="image"; filename="portrait.png"' in body
        assert b"fake-png-content" in body
        assert b'name="max_faces"' in body and b"\r\n3\r\n" in body
        assert b'name="collection_id"' in body and b"\r\nemployees\r\n" in body
        return response(
            200,
            {"faces": [{"detection_score": 0.99}], "processing_ms": 4.5},
        )

    with client_for(handler) as client:
        result = client.detect(image, max_faces=3, collection="employees")

    assert isinstance(result, DetectResult)
    assert result.faces[0]["detection_score"] == 0.99
    assert result.processing_ms == 4.5


def test_compare_accepts_bytes_and_file_like_without_moving_stream() -> None:
    stream = io.BytesIO(b"target-bytes")
    stream.name = "target.jpeg"  # type: ignore[attr-defined]
    stream.seek(2)

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        assert b'name="source"; filename="source.jpg"' in body
        assert b"source-bytes" in body
        assert b'name="target"; filename="target.jpeg"' in body
        assert b"rget-bytes" in body
        assert b'name="threshold"' in body and b"\r\n0.68\r\n" in body
        return response(
            200,
            {"matched": True, "similarity": 0.82, "threshold": 0.68},
        )

    with client_for(handler) as client:
        result = client.compare(b"source-bytes", stream, threshold=0.68)

    assert isinstance(result, CompareResult)
    assert result.matched is True
    assert result.similarity == 0.82
    assert stream.tell() == 2


def test_detect_accepts_non_seekable_binary_stream() -> None:
    class NonSeekable:
        def tell(self) -> int:
            raise OSError("not seekable")

        def read(self) -> bytes:
            return b"stream-image"

    def handler(request: httpx.Request) -> httpx.Response:
        assert b"stream-image" in request.content
        return response(200, {"faces": [], "processing_ms": 1.0})

    with client_for(handler) as client:
        result = client.detect(NonSeekable())  # type: ignore[arg-type]

    assert result.faces == []


def test_embeddings_and_models_use_expected_routes() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/v1/embeddings":
            assert b'name="collection_id"' in request.content
            assert b"\r\nemployees\r\n" in request.content
            return response(200, {"faces": [{"embedding": [0.0, 1.0]}]})
        return response(200, {"models": []})

    with client_for(handler) as client:
        assert len(client.embeddings(b"image", collection="employees").faces) == 1
        assert client.models()["models"] == []
        client.system()

    assert seen == ["/v1/embeddings", "/v1/models", "/v1/system"]


def test_collection_crud_serialization_and_pagination() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return response(204)
        if request.method == "GET" and request.url.path == "/v1/collections":
            return response(200, {"collections": [{"id": "employees"}], "next_cursor": "n"})
        return response(201 if request.method == "POST" else 200, {"id": "employees"})

    with client_for(handler) as client:
        created = client.create_collection(
            "employees",
            name="Company Employees",
            description="staff",
            threshold=0.68,
            metadata={"site": "A"},
            save_face_crops=True,
            search_profile="bf16_v1",
            capacity_rows=200_000,
            max_faces_per_person=25,
            load_policy="eager",
            detector_input_sizes=[(96, 96), (512, 512)],
            detector_threshold=0.5,
            detector_nms_threshold=0.4,
            single_face_selection="center_largest",
        )
        page = client.list_collections(limit=10, cursor="opaque")
        fetched = client.get_collection("employees")
        updated = client.update_collection(
            "employees",
            threshold=0.7,
            save_face_crops=False,
            capacity_rows=300_000,
            max_faces_per_person=30,
            load_policy="lazy",
            detector_threshold=0.45,
            single_face_selection="largest",
        )
        deleted = client.delete_collection("employees", force=True)

    assert created.collection["id"] == "employees"
    assert fetched.collection["id"] == "employees"
    assert updated.collection["id"] == "employees"
    assert isinstance(page, CollectionPage)
    assert page.collections[0]["id"] == "employees"
    assert page.next_cursor == "n"
    assert deleted.status_code == 204 and deleted.to_dict() == {}
    assert json.loads(requests[0].content) == {
        "id": "employees",
        "name": "Company Employees",
        "description": "staff",
        "threshold": 0.68,
        "metadata": {"site": "A"},
        "save_face_crops": True,
        "detection": {
            "input_sizes": [[96, 96], [512, 512]],
            "threshold": 0.5,
            "nms_threshold": 0.4,
            "single_face_selection": "center_largest",
        },
        "search": {
            "profile": "bf16_v1",
            "capacity_rows": 200_000,
            "max_faces_per_person": 25,
            "load_policy": "eager",
        },
    }
    assert dict(requests[1].url.params) == {"limit": "10", "cursor": "opaque"}
    assert json.loads(requests[3].content) == {
        "threshold": 0.7,
        "save_face_crops": False,
        "detection": {
            "threshold": 0.45,
            "single_face_selection": "largest",
        },
        "search": {
            "capacity_rows": 300_000,
            "max_faces_per_person": 30,
            "load_policy": "lazy",
        },
    }
    assert dict(requests[4].url.params) == {"force": "true"}


def test_collection_crop_default_and_authenticated_download() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            payload = json.loads(request.content)
            assert "save_face_crops" not in payload
            assert payload["threshold"] == 0.4
            return response(201, {"id": "employees", "save_face_crops": False})
        assert request.headers["authorization"] == "Bearer secret"
        assert request.headers["accept"] == "image/jpeg"
        return httpx.Response(
            200,
            headers={"content-type": "image/jpeg", "x-request-id": "crop-request"},
            content=b"jpeg-crop-bytes",
        )

    with client_for(handler) as client:
        client.create_collection("employees", name="Employees")
        crop = client.get_face_crop("team a", "alice/b", "face/1")

    assert crop == b"jpeg-crop-bytes"
    assert requests[1].url.raw_path == (
        b"/v1/collections/team%20a/persons/alice%2Fb/faces/face%2F1/image"
    )


def test_monitor_crud_state_and_event_cursor() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return response(204)
        if request.url.path.endswith("/state"):
            return response(200, {"state": {"status": "running"}})
        if request.url.path.endswith("/events"):
            return response(
                200,
                {
                    "events": [],
                    "next_cursor": "next",
                    "has_more": False,
                    "truncated": True,
                    "stream_reset": False,
                },
            )
        if request.url.path == "/v1/monitors" and request.method == "GET":
            return response(
                200,
                {"monitors": [{"id": "front-gate"}], "next_cursor": "page"},
            )
        return response(201 if request.method == "POST" else 200, {
            "monitor": {"id": "front-gate", "preview_enabled": False}
        })

    with client_for(handler) as client:
        created = client.create_monitor(
            "front-gate",
            name="Front gate",
            rtsp_url="rtsp://camera.test/live",
            collection="employees",
            inference_fps=2.5,
            match_threshold=0.45,
        )
        page = client.list_monitors(limit=10, cursor="list-cursor")
        client.get_monitor("front-gate")
        client.update_monitor(
            "front-gate",
            preview_enabled=True,
            confirm_frames=4,
        )
        state_result = client.monitor_state("front-gate")
        events = client.monitor_events(
            "front-gate",
            limit=25,
            cursor="event-cursor",
        )
        deleted = client.delete_monitor("front-gate")

    assert created.monitor["id"] == "front-gate"
    assert isinstance(page, MonitorPage)
    assert page.next_cursor == "page"
    assert state_result.state["status"] == "running"
    assert isinstance(events, MonitorEventPage)
    assert events.next_cursor == "next"
    assert events.has_more is False
    assert events.truncated is True
    assert events.stream_reset is False
    assert deleted.status_code == 204
    assert json.loads(requests[0].content) == {
        "id": "front-gate",
        "name": "Front gate",
        "description": "",
        "enabled": True,
        "source": {"type": "rtsp", "url": "rtsp://camera.test/live"},
        "collection_id": "employees",
        "inference_fps": 2.5,
        "match_threshold": 0.45,
        "event_buffer_size": 1000,
        "event_policy": {
            "confirm_frames": 3,
            "absence_timeout_seconds": 3.0,
            "cooldown_seconds": 10.0,
            "emit_unknown": True,
        },
        "preview_enabled": False,
    }
    assert dict(requests[1].url.params) == {
        "limit": "10",
        "cursor": "list-cursor",
    }
    assert json.loads(requests[3].content) == {
        "preview_enabled": True,
        "event_policy": {"confirm_frames": 4},
    }
    assert dict(requests[5].url.params) == {
        "limit": "25",
        "cursor": "event-cursor",
    }


def test_monitor_patch_preserves_explicit_false_without_defaulting_other_policy_fields() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response(200, {"monitor": {"id": "front-gate"}})

    with client_for(handler) as client:
        client.update_monitor("front-gate", emit_unknown=False)

    assert json.loads(requests[0].content) == {
        "event_policy": {"emit_unknown": False},
    }


def test_person_registration_sends_repeated_images_and_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/collections/employees/persons"
        body = request.content
        assert body.count(b'name="images";') == 2
        assert b"first-image" in body and b"second-image" in body
        assert b'name="id"' in body and b"employee-001" in body
        assert b'name="external_id"' in body and b"HR-1001" in body
        assert b'{"department":"sales"}' in body
        assert b'name="review_mode"' in body and b"\r\nstrict\r\n" in body
        return response(
            201,
            {
                "person": {"id": "employee-001", "face_count": 1},
                "faces": [{"id": "face-1"}],
                "rejected_images": [{"index": 1, "reason": "multiple_faces"}],
            },
        )

    with client_for(handler) as client:
        result = client.add_person(
            "employees",
            person_id="employee-001",
            name="Alice",
            external_id="HR-1001",
            metadata={"department": "sales"},
            images=[b"first-image", io.BytesIO(b"second-image")],
            review_mode="strict",
        )

    assert isinstance(result, PersonRegistrationResult)
    assert result.person["id"] == "employee-001"
    assert result.faces[0]["id"] == "face-1"
    assert result.rejected_images[0]["reason"] == "multiple_faces"


def test_enrollment_review_mode_defaults_to_off() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert b'name="review_mode"' in request.content
        assert b"\r\noff\r\n" in request.content
        assert b'name="embedding_mode"' in request.content
        assert b"\r\nserver\r\n" in request.content
        return response(
            201,
            {"person": {"id": "alice"}, "faces": [], "rejected_images": []},
        )

    with client_for(handler) as client:
        client.create_person("employees", images=[b"image"])


def test_external_trusted_registration_serializes_vectors_and_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response(
            201,
            {"person": {"id": "alice"}, "faces": [], "rejected_images": []},
        )

    class FloatLike:
        def __init__(self, value: float) -> None:
            self.value = value

        def __float__(self) -> float:
            return self.value

    with client_for(handler) as client:
        client.add_person(
            "employees",
            person_id="alice",
            images=[b"first", b"second"],
            external_embeddings=[
                [FloatLike(1.0), FloatLike(0.0)],
                (value for value in (0.0, 1.0)),
            ],
            embedding_contract_id="contract-v1",
            review_mode="strict",
        )
        client.add_faces(
            "employees",
            "alice",
            [b"third"],
            external_embeddings=[[1.0, 0.0]],
            embedding_contract_id="contract-v1",
        )

    for request in requests:
        body = request.content
        assert b'name="embedding_mode"' in body
        assert b"\r\nexternal_trusted\r\n" in body
        assert b'name="embedding_contract_id"' in body
        assert b"\r\ncontract-v1\r\n" in body
    assert b'[[1.0,0.0],[0.0,1.0]]' in requests[0].content
    assert b'[[1.0,0.0]]' in requests[1].content


@pytest.mark.parametrize(
    ("external_embeddings", "contract_id", "message"),
    [
        ([[1.0, 0.0]], None, "embedding_contract_id is required"),
        (None, "contract-v1", "requires external_embeddings"),
        ([[1.0, 0.0]], "contract-v1", "count must equal images count"),
        ([[float("nan")], [1.0]], "contract-v1", "NaN or infinity"),
        ([[] , [1.0]], "contract-v1", "must not be empty"),
        ([[0.5, 0.5], [1.0, 0.0]], "contract-v1", "L2-normalized"),
    ],
)
def test_external_trusted_registration_rejects_invalid_client_input(
    external_embeddings: object,
    contract_id: str | None,
    message: str,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called")

    with client_for(handler) as client, pytest.raises(ValueError, match=message):
        client.create_person(
            "employees",
            images=[b"first", b"second"],
            external_embeddings=external_embeddings,  # type: ignore[arg-type]
            embedding_contract_id=contract_id,
        )


def test_person_face_crud_and_search_routes_are_encoded() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith("/search"):
            return response(200, {"matches": [], "threshold": 0.68})
        if request.method == "DELETE":
            return response(204)
        if path.endswith("/faces") and request.method == "GET":
            return response(200, {"faces": [], "next_cursor": None})
        if path.endswith("/faces") and request.method == "POST":
            return response(200, {"faces": [], "rejected_images": []})
        if request.method == "GET" and path.endswith("/persons"):
            return response(200, {"persons": []})
        return response(200, {"id": "alice/b"})

    with client_for(handler) as client:
        client.list_persons("team a", limit=7, search="alice")
        client.get_person("team a", "alice/b")
        client.update_person("team a", "alice/b", name="Alice")
        added = client.add_faces(
            "team a", "alice/b", [b"face"], review_mode="standard"
        )
        client.list_faces("team a", "alice/b", limit=3)
        client.delete_face("team a", "alice/b", "face/1")
        client.delete_person("team a", "alice/b")
        search = client.search("team a", b"query", limit=2, threshold=0.68)

    assert isinstance(search, SearchResult)
    assert isinstance(added, FaceRegistrationResult)
    assert added.faces == [] and added.rejected_images == []
    assert search.matches == []
    assert search.threshold == 0.68
    assert dict(requests[0].url.params) == {"limit": "7", "search": "alice"}
    assert requests[1].url.raw_path == b"/v1/collections/team%20a/persons/alice%2Fb"
    assert requests[5].url.raw_path.endswith(b"/faces/face%2F1")
    assert b'name="review_mode"' in requests[3].content
    assert b"\r\nstandard\r\n" in requests[3].content
    assert b'name="limit"' in requests[7].content
    assert b'name="face_selection"' not in requests[7].content


def test_empty_images_and_empty_patch_fail_before_http() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called")

    with client_for(handler) as client:
        with pytest.raises(ValueError, match="at least one image"):
            client.create_person("employees", images=[])
        with pytest.raises(ValueError, match="at least one field"):
            client.update_person("employees", "alice")
        with pytest.raises(ValueError, match="must not be empty"):
            client.detect(b"")
        with pytest.raises(TypeError, match="path, bytes"):
            client.detect(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("status", "exception_type"),
    [
        (400, ValidationError),
        (401, AuthenticationError),
        (404, NotFoundError),
        (409, ConflictError),
        (413, PayloadTooLargeError),
        (422, ValidationError),
        (429, RateLimitError),
        (500, ServerError),
        (503, ServiceUnavailableError),
    ],
)
def test_api_errors_are_typed(
    status: int, exception_type: type[Exception]
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return response(
            status,
            {
                "error": {
                    "code": "face_not_found",
                    "message": "No usable face was detected.",
                    "details": {"image": "source"},
                },
                "request_id": "body-request",
            },
            request_id="header-request",
        )

    with client_for(handler) as client:
        with pytest.raises(exception_type) as captured:
            client.detect(b"image")

    error = captured.value
    assert error.code == "face_not_found"
    assert error.status_code == status
    assert error.request_id == "header-request"
    assert error.details == {"image": "source"}
    assert "header-request" in str(error)


def test_transport_and_invalid_success_response_are_safe() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret low-level reason", request=request)

    with client_for(unavailable) as client:
        with pytest.raises(TransportError, match="Unable to complete") as captured:
            client.health()
    assert "secret low-level reason" not in str(captured.value)

    def invalid(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-id": "bad-json"},
            content=b"not-json",
        )

    with client_for(invalid) as client:
        with pytest.raises(ServerError) as invalid_response:
            client.health()
    assert invalid_response.value.code == "invalid_response"
    assert invalid_response.value.request_id == "bad-json"


@pytest.mark.parametrize("code,liveness", [
    ("liveness_fake", {"status": "ok", "is_live": False, "live_score": 0.1}),
    ("liveness_input_rejected", {"status": "input_rejected", "is_live": None, "live_score": None}),
    ("liveness_input_rejected", {
        "status": "input_rejected", "is_live": None, "live_score": None,
        "reason": (
            "Insufficient image area around the face for liveness detection. Move the face "
            "toward the center, step back from the camera, or use a less tightly cropped image."
        ),
    }),
])
def test_liveness_results_and_rejection_details_are_preserved(code, liveness) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/detect":
            return response(200, {"faces": [{"liveness": liveness}], "processing_ms": 1.0})
        return response(422, {"error": {"code": code, "message": "Liveness rejected", "details": {"side": "target", "liveness": liveness}}})

    with client_for(handler) as client:
        assert client.detect(b"image").faces[0]["liveness"] == liveness
        with pytest.raises(ValidationError) as captured:
            client.compare(b"source", b"target")
    assert captured.value.code == code
    assert captured.value.details == {"side": "target", "liveness": liveness}


def test_liveness_type_keeps_core_fields_required_and_reason_optional() -> None:
    assert sdk_results.LivenessResult.__required_keys__ == {"status", "is_live", "live_score"}
    assert sdk_results.LivenessResult.__optional_keys__ == {"reason"}
