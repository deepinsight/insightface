from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import sqlite3
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .addon_management import LivenessManager, require_management_request
from .api.auth import ApiKeyAuthenticator
from .api.responses import (
    CollectionPageResponse,
    CollectionResponse,
    CompareResponse,
    DetectResponse,
    EmbeddingsResponse,
    ErrorEnvelope,
    FacePageResponse,
    FaceRegistrationResponse,
    HealthResponse,
    LivenessManagementResponse,
    ModelsResponse,
    MonitorEventPageResponse,
    MonitorPageResponse,
    MonitorResponse,
    MonitorStateResponse,
    PersonPageResponse,
    PersonRegistrationResponse,
    PersonResponse,
    SearchResponse,
    SystemResponse,
)
from .api.schemas import (
    CollectionCreate,
    CollectionPatch,
    EmbeddingMode,
    MonitorCreate,
    MonitorPatch,
    PersonPatch,
    ReviewMode,
    validate_id,
)
from .config import DetectionProfile, Settings
from .errors import ApiError, bad_request, conflict, not_found, unprocessable
from .inference import create_engine
from .request_context import REQUEST_DEADLINE, remaining_seconds
from .search import SearchIndexError, SearchIndexManager, create_search_backend
from .services import (
    FaceService,
    ImageData,
    ImageLoader,
    MonitorAlreadyExistsError,
    MonitorLimitError,
    MonitorManager,
    MonitorNotFoundError,
    MonitorPreviewDisabledError,
    MonitorUnavailableError,
)
from .services.presentation import face_result
from .storage import (
    CursorCodec,
    Database,
    FaceCropStore,
    Repository,
    SecretCodec,
)

LOGGER = logging.getLogger("insightface_server")
PACKAGE_DIR = Path(__file__).resolve().parent
SERVER_DIR = PACKAGE_DIR.parents[1]
MAINTAINER_GUIDE = SERVER_DIR / "docs" / "maintainer-guide.md"
DOCUMENTS: dict[str, dict[str, Path]] = {
    "en": {
        "api": SERVER_DIR / "docs" / "api.md",
        "user-guide": SERVER_DIR / "docs" / "user-guide.md",
        "maintainer": MAINTAINER_GUIDE,
    },
    "zh": {
        "api": SERVER_DIR / "docs" / "api.zh-CN.md",
        "user-guide": SERVER_DIR / "docs" / "user-guide.zh-CN.md",
        "maintainer": MAINTAINER_GUIDE,
    },
    **{
        language: {
            "api": SERVER_DIR / "docs" / f"api.{language}.md",
            "user-guide": SERVER_DIR / "docs" / f"user-guide.{language}.md",
            "maintainer": MAINTAINER_GUIDE,
        }
        for language in ("ja", "de", "es", "fr", "ru", "pt", "ko")
    },
}


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def _json(request: Request, content: dict[str, Any], *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={**content, "request_id": _request_id(request)},
    )


def _metadata(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise bad_request("metadata must be valid JSON.", code="invalid_metadata") from exc
    if not isinstance(value, dict):
        raise bad_request("metadata must be a JSON object.", code="invalid_metadata")
    return value


def _enrollment_embeddings(
    *,
    embedding_mode: EmbeddingMode,
    external_embeddings: str | None,
    embedding_contract_id: str | None,
    image_count: int,
) -> list[Any] | None:
    """Validate multipart-level trusted embedding fields without logging values."""

    if embedding_mode == "server":
        if external_embeddings is not None or embedding_contract_id is not None:
            raise bad_request(
                "External embedding fields require embedding_mode=external_trusted.",
                code="unexpected_external_embedding",
            )
        return None
    if external_embeddings is None:
        raise bad_request(
            "external_embeddings is required for external_trusted enrollment.",
            code="missing_external_embeddings",
        )
    if embedding_contract_id is None or not embedding_contract_id.strip():
        raise bad_request(
            "embedding_contract_id is required for external_trusted enrollment.",
            code="missing_embedding_contract_id",
        )
    try:
        values = json.loads(external_embeddings)
    except (json.JSONDecodeError, TypeError) as exc:
        raise bad_request(
            "external_embeddings must be a JSON array of vectors.",
            code="invalid_external_embeddings",
        ) from exc
    if not isinstance(values, list):
        raise bad_request(
            "external_embeddings must be a JSON array of vectors.",
            code="invalid_external_embeddings",
        )
    if len(values) != image_count:
        raise bad_request(
            "external_embeddings must contain exactly one vector for each image.",
            code="external_embedding_count_mismatch",
            image_count=image_count,
            embedding_count=len(values),
        )
    return values


def _path_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
    }


def _cpu_name() -> str:
    value = platform.processor()
    if value:
        return value
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


async def _blocking(function, *args, **kwargs):
    """Run blocking model/storage work without delaying timeout responses."""

    work = asyncio.to_thread(partial(function, *args, **kwargs))
    remaining = remaining_seconds()
    if remaining is None:
        return await work
    try:
        return await asyncio.wait_for(work, timeout=max(0.001, remaining - 0.05))
    except TimeoutError as exc:
        raise ApiError("request_timeout", "The request timed out.", 503) from exc


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings.from_env()
    recent_errors: deque[dict[str, Any]] = deque(maxlen=20)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=getattr(logging, configured.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        database = Database(
            configured.data_dir / "insightface-server.db", SERVER_DIR / "migrations"
        )
        search_indexes: SearchIndexManager | None = None
        monitors: MonitorManager | None = None
        liveness_manager: LivenessManager | None = None
        engine = None
        app.state.ready = False
        database.acquire_process_lock()
        try:
            database.initialize()
            repository = Repository(database)
            authenticator = ApiKeyAuthenticator(configured, repository)
            authenticator.initialize()
            engine = create_engine(configured)
            engine.startup()
            search_backend = create_search_backend(configured)
            search_indexes = SearchIndexManager(
                repository,
                search_backend,
                build_batch_rows=configured.search_build_batch_rows,
            )
            search_indexes.startup()
            image_loader = ImageLoader(configured)
            cursors = CursorCodec(configured.data_dir / "cursor.key")
            service = FaceService(
                configured,
                repository,
                cursors,
                FaceCropStore(configured.data_dir, enabled=configured.save_face_crops),
                engine,
                search_indexes,
            )
            monitors = MonitorManager(
                service,
                repository,
                cursors,
                SecretCodec(configured.data_dir / "monitor-credentials.key"),
                max_monitors=configured.rtsp_max_streams,
                max_faces=configured.max_detected_faces,
                preview_fps=configured.rtsp_preview_fps,
                jpeg_quality=configured.rtsp_jpeg_quality,
                open_timeout_seconds=configured.rtsp_open_timeout_seconds,
                read_timeout_seconds=configured.rtsp_read_timeout_seconds,
                reconnect_delay_seconds=configured.rtsp_reconnect_delay_seconds,
            )
            monitors.startup()
            app.state.settings = configured
            app.state.database = database
            app.state.repository = repository
            app.state.authenticator = authenticator
            app.state.engine = engine
            app.state.search_indexes = search_indexes
            app.state.image_loader = image_loader
            app.state.service = service
            app.state.monitors = monitors
            liveness_manager = LivenessManager(configured, enabled="liveness" in configured.addons)
            app.state.liveness_manager = liveness_manager
            app.state.ready = True
            LOGGER.info(
                "server_ready version=%s provider=%s inference_max_concurrency=%s "
                "model=%s runtime=%s search=%s",
                __version__,
                engine.summary.provider,
                configured.effective_inference_max_concurrency,
                json.dumps(engine.summary.as_dict(), sort_keys=True),
                json.dumps(engine.runtime_summary(), sort_keys=True),
                json.dumps(search_indexes.runtime_summary(), sort_keys=True),
            )
            LOGGER.info(
                "web_ui_%s",
                "disabled" if configured.web_ui_disabled else "enabled",
            )
            yield
        finally:
            app.state.ready = False
            if liveness_manager is not None:
                await liveness_manager.close()
            if monitors is not None:
                monitors.close()
            if search_indexes is not None:
                search_indexes.close()
            if engine is not None:
                engine.close()
            database.release_process_lock()

    app = FastAPI(
        title="InsightFace Server",
        version=__version__,
        description=(
            "Simple self-hosted face detection, optional liveness checks, comparison, registration, and search. "
            "Image upload endpoints accept JPEG, PNG, WebP, and BMP. "
            "Liveness is disabled by default in server.toml, including in legacy configurations that omit "
            "inference.addons. Use the Web UI to download and enable liveness for the next "
            "startup, or set inference.addons = [\"liveness\"] and install the addon manually. "
            "Set inference.addons = [] to disable it. "
            "Setting addons.auto_download = [\"liveness\"] includes it when running "
            "models install <package>, even when the base package is cached. Mode, threshold, "
            "compare scope, and registration policy are server configuration, not request "
            "parameters. When evaluated, liveness has "
            "status, is_live, and live_score. An input_rejected status means the "
            "input could not be evaluated, with both values null and an optional reason "
            "containing English retry guidance. Older results may omit reason; successful "
            "and fake evaluations have only the three core fields. A missing liveness "
            "field means no evaluation was performed. In normal mode, failed or rejected "
            "liveness blocks recognition; observe mode continues recognition. "
            "Registration skips liveness by default. Install the selected addon before "
            "starting the Server; startup does not download models."
        ),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    if configured.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(configured.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.monotonic()
        deadline_token = REQUEST_DEADLINE.set(started + configured.request_timeout_seconds)
        try:
            content_length = int(request.headers.get("content-length", "0"))
        except ValueError:
            content_length = 0
        if content_length > configured.max_request_bytes:
            response: Response = JSONResponse(
                status_code=413,
                content=ApiError("request_too_large", "The request body is too large.", 413).body(
                    request_id
                ),
            )
        else:
            try:
                chunks: list[bytes] = []
                received = 0
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > configured.max_request_bytes:
                        raise ApiError("request_too_large", "The request body is too large.", 413)
                    chunks.append(chunk)
                # Function middleware receives Starlette's cached Request. Setting
                # its bounded body lets the downstream multipart parser replay it.
                request._body = b"".join(chunks)  # noqa: SLF001
                remaining = configured.request_timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError
                response = await asyncio.wait_for(call_next(request), timeout=remaining)
            except ApiError as request_error:
                recent_errors.append(
                    {
                        "code": request_error.code,
                        "path": request.url.path,
                        "request_id": request_id,
                    }
                )
                response = JSONResponse(
                    status_code=request_error.status_code,
                    content=request_error.body(request_id),
                )
            except TimeoutError:
                timeout_error = ApiError("request_timeout", "The request timed out.", 503)
                recent_errors.append(
                    {
                        "code": timeout_error.code,
                        "path": request.url.path,
                        "request_id": request_id,
                    }
                )
                response = JSONResponse(status_code=503, content=timeout_error.body(request_id))
        response.headers["x-request-id"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        REQUEST_DEADLINE.reset(deadline_token)
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        recent_errors.append(
            {"code": exc.code, "path": request.url.path, "request_id": _request_id(request)}
        )
        response = JSONResponse(status_code=exc.status_code, content=exc.body(_request_id(request)))
        if exc.status_code == 401:
            response.headers["WWW-Authenticate"] = "Bearer"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {"msg": "invalid request"}
        error = bad_request(f"Request validation failed: {first.get('msg', 'invalid request')}")
        return JSONResponse(status_code=400, content=error.body(_request_id(request)))

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        code = {
            404: "route_not_found",
            405: "method_not_allowed",
            413: "request_too_large",
        }.get(exc.status_code, "http_error")
        error = ApiError(code, str(exc.detail), exc.status_code)
        return JSONResponse(status_code=exc.status_code, content=error.body(_request_id(request)))

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        LOGGER.exception(
            "unhandled_request_error request_id=%s path=%s type=%s",
            _request_id(request),
            request.url.path,
            type(exc).__name__,
        )
        recent_errors.append(
            {"code": "internal_error", "path": request.url.path, "request_id": _request_id(request)}
        )
        error = ApiError("internal_error", "The server could not complete the request.", 500)
        return JSONResponse(status_code=500, content=error.body(_request_id(request)))

    async def authorize(request: Request) -> None:
        await _blocking(app.state.authenticator.require, request)

    async def load_images(
        uploads: list[UploadFile],
    ) -> tuple[list[ImageData], list[int], list[dict[str, Any]]]:
        loaded: list[ImageData] = []
        original_indices: list[int] = []
        rejected: list[dict[str, Any]] = []
        for index, upload in enumerate(uploads):
            try:
                loaded.append(await app.state.image_loader.from_upload(upload))
                original_indices.append(index)
            except ApiError as exc:
                rejected.append(
                    {
                        "index": index,
                        "filename": upload.filename or "upload",
                        "reason": exc.code,
                    }
                )
        return loaded, original_indices, rejected

    def remap_rejections(
        rejected: list[dict[str, Any]], original_indices: list[int]
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in rejected:
            value = dict(item)
            relative = int(value["index"])
            if 0 <= relative < len(original_indices):
                value["index"] = original_indices[relative]
            result.append(value)
        return result

    @app.get(
        "/v1/health",
        tags=["system"],
        response_model=HealthResponse,
        responses={
            503: {
                "model": HealthResponse,
                "description": "The process is running but is not ready.",
            }
        },
    )
    async def health(request: Request):
        ready = bool(getattr(app.state, "ready", False))
        if ready:
            try:
                status = await _blocking(app.state.database.status)
                ready = status.get("quick_check") == "ok"
            except Exception:
                ready = False
        return _json(
            request,
            {
                "status": "ready" if ready else "not_ready",
                "version": __version__,
                "auth_enabled": configured.auth_enabled,
            },
            status_code=200 if ready else 503,
        )

    @app.get("/v1/system", tags=["system"], response_model=SystemResponse)
    async def system(request: Request):
        await authorize(request)
        runtime = app.state.engine.runtime_summary()
        return _json(
            request,
            {
                "server_version": __version__,
                "os": platform.platform(),
                "architecture": platform.machine(),
                "cpu": {"model": _cpu_name(), "logical_cores": os.cpu_count()},
                "runtime": runtime,
                "execution_provider": app.state.engine.summary.provider,
                "model": app.state.engine.summary.as_dict(),
                "database": app.state.database.status(),
                "data": _path_status(configured.data_dir),
                "models": _path_status(configured.models_dir),
                "stats": app.state.repository.stats(),
                "search": app.state.search_indexes.runtime_summary(),
                "api_key": {
                    "authentication_enabled": configured.auth_enabled,
                    "configured": app.state.repository.has_api_keys(),
                },
                "safe_config": {
                    "max_image_bytes": configured.max_image_bytes,
                    "max_image_pixels": configured.max_image_pixels,
                    "max_registration_images": configured.max_registration_images,
                    "max_request_bytes": configured.max_request_bytes,
                    "request_timeout_seconds": configured.request_timeout_seconds,
                    "inference_max_concurrency": (
                        configured.effective_inference_max_concurrency
                    ),
                    "addons": list(configured.addons),
                    "auto_download_addons": list(configured.auto_download_addons),
                    "liveness_mode": configured.liveness_mode,
                    "liveness_threshold": configured.liveness_threshold,
                    "liveness_compare_scope": configured.liveness_compare_scope,
                    "liveness_on_registration": configured.liveness_on_registration,
                    "save_face_crops": configured.save_face_crops,
                    "default_similarity_threshold": configured.default_threshold,
                    "default_search_profile": configured.default_search_profile,
                    "default_search_capacity_rows": configured.default_search_capacity_rows,
                    "max_search_capacity_rows": configured.max_search_capacity_rows,
                    "default_max_faces_per_person": configured.default_max_faces_per_person,
                    "default_search_load_policy": configured.default_search_load_policy,
                    "search_backend": configured.search_backend,
                    "search_device_id": configured.search_device_id,
                    "search_topk_mode": configured.search_topk_mode,
                    "detector_input_sizes": [
                        list(size) for size in configured.detector_input_sizes
                    ],
                    "detection": configured.detection_profile.as_dict(),
                    "max_detected_faces": configured.max_detected_faces,
                    "rtsp_max_streams": configured.rtsp_max_streams,
                    "rtsp_preview_fps": configured.rtsp_preview_fps,
                    "web_ui_disabled": configured.web_ui_disabled,
                    "config_file": (
                        str(configured.config_file) if configured.config_file else None
                    ),
                },
                "recent_errors": list(recent_errors)[-10:],
            },
        )

    @app.get("/v1/models", tags=["system"], response_model=ModelsResponse)
    async def models(request: Request):
        await authorize(request)
        return _json(
            request,
            {
                "models": [dict(model) for model in app.state.engine.summary.models],
                **({"addons": [dict(addon) for addon in app.state.engine.summary.addons]}
                   if app.state.engine.summary.addons else {}),
                "execution_provider": app.state.engine.summary.provider,
                "license": app.state.engine.summary.license,
            },
        )

    @app.get("/v1/addons/liveness", tags=["system"], response_model=LivenessManagementResponse)
    async def liveness_status(request: Request):
        """Inspect active liveness, the verified local model, and saved restart settings.

        This is a management endpoint, not a standalone liveness inference API.
        Read-only or single-file config mounts report can_enable=false with an
        actionable unavailable_reason. Checking status never downloads a model.
        """
        await authorize(request)
        return _json(request, await _blocking(app.state.liveness_manager.status))

    @app.post(
        "/v1/addons/liveness/enable", tags=["system"],
        response_model=LivenessManagementResponse, status_code=202,
        openapi_extra={"requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {
                "type": "object", "additionalProperties": False,
            }}},
        }},
    )
    async def enable_liveness(request: Request):
        """Download and verify liveness, then save its next-startup configuration.

        Send an empty JSON object. The background job continues after the request
        or browser tab closes; poll GET /v1/addons/liveness for completion. A
        verified cached model is reused. Only after successful verification does
        the job add liveness to inference.addons and addons.auto_download in the
        configured server.toml, preserving other values and comments. Current
        inference does not change: the operator must manually restart Server.
        Repeated requests share the active job. Auth follows other /v1 APIs;
        browser requests must originate from this Server or an allowed CORS origin.
        """
        await authorize(request)
        require_management_request(request, configured.cors_origins)
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise bad_request("Provide an empty JSON object.", code="invalid_addon_request") from None
        if not isinstance(body, dict) or body:
            raise bad_request("Provide an empty JSON object.", code="invalid_addon_request")
        return _json(request, await app.state.liveness_manager.enable(), status_code=202)

    @app.post("/v1/detect", tags=["faces"], response_model=DetectResponse)
    async def detect(
        request: Request,
        image: UploadFile = File(...),
        max_faces: int | None = Form(default=None, ge=1, le=100),
        collection_id: str | None = Form(default=None),
        min_score: float | None = Form(default=None, ge=0.0, le=1.0, deprecated=True),
    ):
        """Detect faces without generating recognition embeddings.

        When the liveness addon is enabled, faces[].liveness contains status,
        is_live, and live_score. HTTP 200 also includes fake faces (status=ok,
        is_live=false) and unsuitable input (status=input_rejected, is_live=null,
        live_score=null); neither is a detection error. An omitted liveness field
        means no evaluation. Rejected input may include liveness.reason with English
        retry guidance; older results may omit it. Successful and fake evaluations
        have only the three core fields. Model execution failure returns HTTP 503
        liveness_unavailable. There is no separate liveness endpoint.
        """
        await authorize(request)
        if min_score is not None:
            raise bad_request(
                "min_score is configured by the system or Collection detection profile.",
                code="request_detection_override_not_supported",
            )
        started = time.perf_counter()
        loaded = await app.state.image_loader.from_upload(image)
        profile = await _blocking(app.state.service.detection_profile, collection_id)
        faces = await _blocking(
            app.state.service.detect,
            loaded,
            max_faces=max_faces,
            detection_profile=profile,
        )
        return _json(
            request,
            {"faces": faces, "processing_ms": round((time.perf_counter() - started) * 1000, 3)},
        )

    @app.post("/v1/compare", tags=["faces"], response_model=CompareResponse)
    async def compare(
        request: Request,
        source: UploadFile = File(...),
        target: UploadFile = File(...),
        threshold: float | None = Form(default=None, ge=0.0, le=1.0),
        collection_id: str | None = Form(default=None),
    ):
        """Compare the selected face in each image.

        Configured liveness_compare_scope selects both, source, or target for
        liveness evaluation. In normal mode, a fake or input rejection stops
        recognition and returns HTTP 422 liveness_fake or liveness_input_rejected;
        error.details includes liveness and side (source or target). In observe
        mode, comparison continues and each evaluated face includes liveness.
        For input_rejected, optional liveness.reason provides English retry guidance;
        when present, it is also used as error.message.
        Unevaluated sides omit liveness. Runtime faults return HTTP 503
        liveness_unavailable. These policies cannot be overridden per request.
        """
        await authorize(request)
        started = time.perf_counter()
        source_image, target_image = await asyncio.gather(
            app.state.image_loader.from_upload(source),
            app.state.image_loader.from_upload(target),
        )
        profile = await _blocking(app.state.service.detection_profile, collection_id)
        comparison_faces = []
        for side, loaded_image in (("source", source_image), ("target", target_image)):
            options = {}
            if "liveness" in configured.addons:
                options["apply_liveness"] = configured.liveness_compare_scope in ("both", side)
            try:
                comparison_faces.append(await _blocking(
                    app.state.service.selected_face, loaded_image,
                    detection_profile=profile, **options,
                ))
            except ApiError as exc:
                if exc.code.startswith("liveness_"):
                    exc.details["side"] = side
                raise
        source_face, target_face = comparison_faces
        assert source_face.embedding is not None and target_face.embedding is not None
        from .services.core import similarity

        score = similarity(source_face.embedding, target_face.embedding)
        selected_threshold = configured.default_threshold if threshold is None else threshold
        source_height, source_width = source_image.pixels.shape[:2]
        target_height, target_width = target_image.pixels.shape[:2]
        return _json(
            request,
            {
                "matched": score >= selected_threshold,
                "similarity": score,
                "threshold": selected_threshold,
                "source_face": face_result(source_face, source_width, source_height),
                "target_face": face_result(target_face, target_width, target_height),
                "processing_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )

    @app.post("/v1/embeddings", tags=["faces"], response_model=EmbeddingsResponse)
    async def embeddings(
        request: Request,
        image: UploadFile = File(...),
        collection_id: str | None = Form(default=None),
        face_selection: str | None = Form(default=None, deprecated=True),
    ):
        """Generate the selected face's recognition embedding.

        With liveness enabled in normal mode, fake faces and unsuitable input
        stop before recognition: HTTP 422 liveness_fake or liveness_input_rejected,
        with the result in error.details.liveness. For input_rejected, optional
        liveness.reason provides English retry guidance and is used as error.message.
        Observe mode continues recognition and returns faces[].liveness beside the embedding.
        Without evaluation, liveness is omitted. Runtime faults return HTTP 503
        liveness_unavailable in either mode.
        """
        await authorize(request)
        started = time.perf_counter()
        if face_selection is not None:
            raise bad_request(
                "face_selection is configured by the system or Collection detection profile.",
                code="request_detection_override_not_supported",
            )
        loaded = await app.state.image_loader.from_upload(image)
        profile = await _blocking(app.state.service.detection_profile, collection_id)
        selected = await _blocking(
            app.state.service.selected_face,
            loaded,
            detection_profile=profile,
        )
        faces = [selected]
        height, width = loaded.pixels.shape[:2]
        values = []
        for face in faces:
            if face.embedding is None:
                continue
            values.append(
                {
                    **face_result(face, width, height),
                    "embedding": [round(float(value), 8) for value in face.embedding],
                }
            )
        return _json(
            request,
            {
                "faces": values,
                "model": app.state.engine.summary.as_dict(),
                "processing_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )

    @app.post(
        "/v1/collections",
        tags=["collections"],
        status_code=201,
        response_model=CollectionResponse,
    )
    async def create_collection(request: Request, payload: CollectionCreate):
        await authorize(request)
        summary = app.state.engine.summary
        profile = (
            configured.default_search_profile
            if payload.search.profile is None
            else payload.search.profile
        )
        try:
            app.state.search_indexes.validate_configuration(
                profile=profile, dimension=summary.embedding_dimension
            )
        except SearchIndexError as exc:
            raise bad_request(str(exc), code="unsupported_search_profile") from exc
        try:
            detection_profile = DetectionProfile.from_mapping(
                payload.detection.model_dump(exclude_none=True),
                defaults=configured.detection_profile,
            )
            app.state.engine.validate_detection_profile(detection_profile)
            capacity_rows = (
                configured.default_search_capacity_rows
                if payload.search.capacity_rows is None
                else payload.search.capacity_rows
            )
            if capacity_rows > configured.max_search_capacity_rows:
                raise bad_request(
                    "capacity_rows exceeds this deployment's configured maximum.",
                    code="search_capacity_too_large",
                    max_capacity_rows=configured.max_search_capacity_rows,
                )
            values = {
                "id": payload.id,
                "name": payload.name,
                "description": payload.description,
                "default_threshold": (
                    configured.default_threshold if payload.threshold is None else payload.threshold
                ),
                "metadata": payload.metadata,
                "save_face_crops": (
                    configured.save_face_crops
                    if payload.save_face_crops is None
                    else payload.save_face_crops
                ),
                "model_id": summary.model_id,
                "model_digest": summary.model_digest,
                "embedding_dimension": summary.embedding_dimension,
                "preprocessing_version": summary.preprocessing_version,
                "search_profile": profile,
                "capacity_rows": capacity_rows,
                "max_faces_per_person": (
                    configured.default_max_faces_per_person
                    if payload.search.max_faces_per_person is None
                    else payload.search.max_faces_per_person
                ),
                "load_policy": (
                    "eager"
                    if payload.id == "_default" and payload.search.load_policy is None
                    else (
                        configured.default_search_load_policy
                        if payload.search.load_policy is None
                        else payload.search.load_policy
                    )
                ),
                "detector_input_sizes": [
                    list(size) for size in detection_profile.input_sizes
                ],
                "detector_threshold": detection_profile.threshold,
                "detector_nms_threshold": detection_profile.nms_threshold,
                "single_face_selection": detection_profile.single_face_selection,
            }
            item = await _blocking(
                app.state.search_indexes.run_create,
                values,
                lambda: app.state.repository.create_collection(values),
            )
        except sqlite3.IntegrityError as exc:
            raise conflict("The collection already exists.", code="collection_exists") from exc
        except ValueError as exc:
            raise bad_request(str(exc), code="invalid_detection_profile") from exc
        except SearchIndexError as exc:
            raise ApiError(
                "search_index_unavailable",
                "The Collection search index could not be allocated.",
                503,
                {"reason": str(exc)},
            ) from exc
        return _json(request, {"collection": item}, status_code=201)

    @app.get(
        "/v1/collections",
        tags=["collections"],
        response_model=CollectionPageResponse,
    )
    async def list_collections(
        request: Request,
        limit: int = Query(default=50, ge=1, le=100),
        cursor: str | None = Query(default=None),
    ):
        await authorize(request)
        after = app.state.service.cursors.decode(cursor, "collections")
        rows = app.state.repository.list_collections(after, limit + 1)
        page, more = rows[:limit], len(rows) > limit
        next_cursor = (
            app.state.service.cursors.encode("collections", page[-1]["id"])
            if more and page
            else None
        )
        return _json(request, {"collections": page, "next_cursor": next_cursor})

    @app.get(
        "/v1/collections/{collection_id}",
        tags=["collections"],
        response_model=CollectionResponse,
    )
    async def get_collection(request: Request, collection_id: str):
        await authorize(request)
        return _json(request, {"collection": app.state.service.collection(collection_id)})

    @app.patch(
        "/v1/collections/{collection_id}",
        tags=["collections"],
        response_model=CollectionResponse,
    )
    async def patch_collection(request: Request, collection_id: str, payload: CollectionPatch):
        await authorize(request)
        app.state.service.collection(collection_id)
        changes = payload.model_dump(exclude_unset=True)
        if "threshold" in changes:
            changes["default_threshold"] = changes.pop("threshold")
        search_changes = changes.pop("search", None)
        if search_changes:
            changes.update(search_changes)
        detection_changes = changes.pop("detection", None)
        if detection_changes:
            mapping = {
                "input_sizes": "detector_input_sizes",
                "threshold": "detector_threshold",
                "nms_threshold": "detector_nms_threshold",
                "single_face_selection": "single_face_selection",
            }
            for name, value in detection_changes.items():
                changes[mapping[name]] = value
        if (
            "capacity_rows" in changes
            and int(changes["capacity_rows"]) > configured.max_search_capacity_rows
        ):
            raise bad_request(
                "capacity_rows exceeds this deployment's configured maximum.",
                code="search_capacity_too_large",
                max_capacity_rows=configured.max_search_capacity_rows,
            )
        item = await _blocking(app.state.service.update_collection, collection_id, changes)
        return _json(request, {"collection": item})

    @app.delete("/v1/collections/{collection_id}", tags=["collections"], status_code=204)
    async def delete_collection(
        request: Request, collection_id: str, force: bool = Query(default=False)
    ):
        await authorize(request)
        await _blocking(app.state.service.delete_collection, collection_id, force=force)
        return Response(status_code=204)

    @app.post(
        "/v1/collections/{collection_id}/persons",
        tags=["persons"],
        status_code=201,
        response_model=PersonRegistrationResponse,
    )
    async def create_person(
        request: Request,
        collection_id: str,
        images: list[UploadFile] = File(...),
        id: str | None = Form(default=None),
        name: str | None = Form(default=None, max_length=200),
        external_id: str | None = Form(default=None, max_length=200),
        metadata: str = Form(default="{}"),
        review_mode: ReviewMode = Form(default="off"),
        embedding_mode: EmbeddingMode = Form(default="server"),
        external_embeddings: str | None = Form(default=None),
        embedding_contract_id: str | None = Form(default=None, max_length=256),
    ):
        """Register a person from one or more images, allowing partial success.

        Liveness is skipped by default (liveness_on_registration=false), and new
        samples omit liveness. When enabled in server.toml, normal mode rejects
        fake or unsuitable images with reason liveness_fake or
        liveness_input_rejected and a separate liveness result in rejected_images.
        Rejected input may include liveness.reason with English retry guidance;
        older results may omit it. This is separate from the rejection category reason.
        Observe mode continues normal registration checks and stores the result
        with accepted samples. review_mode=off and external_trusted embeddings do
        not bypass enabled liveness. HTTP 201 may contain accepted faces and
        rejected_images; if all images are rejected, HTTP 422 registration_failed
        contains error.details.rejected_images. Model faults return HTTP 503
        liveness_unavailable, not a fake verdict.
        """
        await authorize(request)
        try:
            person_id = validate_id(id) if id else str(uuid.uuid4())
        except ValueError as exc:
            raise bad_request(str(exc), code="invalid_person_id") from exc
        if not images:
            raise bad_request("At least one image is required.")
        if len(images) > configured.max_registration_images:
            raise bad_request(
                f"At most {configured.max_registration_images} images may be registered at once.",
                code="too_many_images",
            )
        supplied_embeddings = _enrollment_embeddings(
            embedding_mode=embedding_mode,
            external_embeddings=external_embeddings,
            embedding_contract_id=embedding_contract_id,
            image_count=len(images),
        )
        if embedding_mode == "external_trusted":
            collection = await _blocking(app.state.service.collection, collection_id)
            expected_contract = str(collection["embedding_contract_id"])
            if embedding_contract_id != expected_contract:
                raise ApiError(
                    "embedding_contract_mismatch",
                    "The external embedding contract does not match this Collection.",
                    409,
                    {
                        "expected_embedding_contract_id": expected_contract,
                        "provided_embedding_contract_id": embedding_contract_id,
                    },
                )
        loaded, indices, invalid = await load_images(images)
        if not loaded:
            raise unprocessable(
                "None of the uploaded images were valid.",
                code="registration_failed",
                rejected_images=invalid,
            )
        try:
            loaded_embeddings = (
                [supplied_embeddings[index] for index in indices]
                if supplied_embeddings is not None
                else None
            )
            person, faces, rejected = await _blocking(
                app.state.service.create_person,
                collection_id,
                {
                    "id": person_id,
                    "name": name,
                    "external_id": external_id,
                    "metadata": _metadata(metadata),
                },
                loaded,
                review_mode=review_mode,
                embedding_mode=embedding_mode,
                external_embeddings=loaded_embeddings,
                embedding_contract_id=embedding_contract_id,
            )
        except ApiError as exc:
            if exc.code == "registration_failed" and invalid:
                exc.details["rejected_images"] = invalid + remap_rejections(
                    list(exc.details.get("rejected_images", [])), indices
                )
            raise
        return _json(
            request,
            {
                "person": person,
                "faces": faces,
                "rejected_images": invalid + remap_rejections(rejected, indices),
            },
            status_code=201,
        )

    @app.get(
        "/v1/collections/{collection_id}/persons",
        tags=["persons"],
        response_model=PersonPageResponse,
    )
    async def list_persons(
        request: Request,
        collection_id: str,
        limit: int = Query(default=50, ge=1, le=100),
        cursor: str | None = Query(default=None),
        search: str | None = Query(default=None, max_length=200),
    ):
        await authorize(request)
        app.state.service.collection(collection_id)
        scope = f"persons:{collection_id}:{search or ''}"
        after = app.state.service.cursors.decode(cursor, scope)
        rows = app.state.repository.list_persons(collection_id, after, limit + 1, search)
        page, more = rows[:limit], len(rows) > limit
        next_cursor = (
            app.state.service.cursors.encode(scope, page[-1]["id"]) if more and page else None
        )
        return _json(request, {"persons": page, "next_cursor": next_cursor})

    @app.get(
        "/v1/collections/{collection_id}/persons/{person_id}",
        tags=["persons"],
        response_model=PersonResponse,
    )
    async def get_person(request: Request, collection_id: str, person_id: str):
        await authorize(request)
        app.state.service.collection(collection_id)
        person = app.state.repository.get_person(collection_id, person_id)
        if person is None:
            raise not_found("person", person_id)
        return _json(request, {"person": person})

    @app.patch(
        "/v1/collections/{collection_id}/persons/{person_id}",
        tags=["persons"],
        response_model=PersonResponse,
    )
    async def patch_person(
        request: Request, collection_id: str, person_id: str, payload: PersonPatch
    ):
        await authorize(request)
        app.state.service.collection(collection_id)
        try:
            person = app.state.repository.update_person(
                collection_id, person_id, payload.model_dump(exclude_unset=True)
            )
        except sqlite3.IntegrityError as exc:
            raise conflict("The external ID already exists.", code="external_id_exists") from exc
        if person is None:
            raise not_found("person", person_id)
        return _json(request, {"person": person})

    @app.delete(
        "/v1/collections/{collection_id}/persons/{person_id}",
        tags=["persons"],
        status_code=204,
    )
    async def delete_person(request: Request, collection_id: str, person_id: str):
        await authorize(request)
        app.state.service.collection(collection_id)
        crops = await _blocking(app.state.service.delete_person, collection_id, person_id)
        for crop in crops:
            app.state.service.crops.delete(crop)
        return Response(status_code=204)

    @app.post(
        "/v1/collections/{collection_id}/persons/{person_id}/faces",
        tags=["persons"],
        status_code=201,
        response_model=FaceRegistrationResponse,
    )
    async def add_faces(
        request: Request,
        collection_id: str,
        person_id: str,
        images: list[UploadFile] = File(...),
        review_mode: ReviewMode = Form(default="off"),
        embedding_mode: EmbeddingMode = Form(default="server"),
        external_embeddings: str | None = Form(default=None),
        embedding_contract_id: str | None = Form(default=None, max_length=256),
    ):
        """Add images to an existing person, allowing partial success.

        Liveness is skipped by default (liveness_on_registration=false). When
        enabled, normal mode rejects fake or unsuitable images with reason
        liveness_fake or liveness_input_rejected and a separate liveness result
        in rejected_images. Observe mode continues other registration checks.
        For input_rejected, optional liveness.reason contains English retry guidance,
        separate from the rejection category reason; older results may omit it.
        review_mode=off and external_trusted embeddings do not bypass enabled
        liveness. HTTP 201 can contain an empty faces list and only rejected_images.
        Accepted samples retain their evaluated liveness result; unevaluated
        samples omit it. Model faults return HTTP 503 liveness_unavailable.
        """
        await authorize(request)
        if len(images) > configured.max_registration_images:
            raise bad_request(
                f"At most {configured.max_registration_images} images may be registered at once.",
                code="too_many_images",
            )
        supplied_embeddings = _enrollment_embeddings(
            embedding_mode=embedding_mode,
            external_embeddings=external_embeddings,
            embedding_contract_id=embedding_contract_id,
            image_count=len(images),
        )
        if embedding_mode == "external_trusted":
            collection = await _blocking(app.state.service.collection, collection_id)
            expected_contract = str(collection["embedding_contract_id"])
            if embedding_contract_id != expected_contract:
                raise ApiError(
                    "embedding_contract_mismatch",
                    "The external embedding contract does not match this Collection.",
                    409,
                    {
                        "expected_embedding_contract_id": expected_contract,
                        "provided_embedding_contract_id": embedding_contract_id,
                    },
                )
        loaded, indices, invalid = await load_images(images)
        loaded_embeddings = (
            [supplied_embeddings[index] for index in indices]
            if supplied_embeddings is not None
            else None
        )
        faces, rejected = await _blocking(
            app.state.service.add_faces,
            collection_id,
            person_id,
            loaded,
            review_mode=review_mode,
            embedding_mode=embedding_mode,
            external_embeddings=loaded_embeddings,
            embedding_contract_id=embedding_contract_id,
        )
        return _json(
            request,
            {
                "faces": faces,
                "rejected_images": invalid + remap_rejections(rejected, indices),
            },
            status_code=201,
        )

    @app.get(
        "/v1/collections/{collection_id}/persons/{person_id}/faces",
        tags=["persons"],
        response_model=FacePageResponse,
    )
    async def list_faces(
        request: Request,
        collection_id: str,
        person_id: str,
        limit: int = Query(default=50, ge=1, le=100),
        cursor: str | None = Query(default=None),
    ):
        await authorize(request)
        app.state.service.collection(collection_id)
        if app.state.repository.get_person(collection_id, person_id) is None:
            raise not_found("person", person_id)
        scope = f"faces:{collection_id}:{person_id}"
        after = app.state.service.cursors.decode(cursor, scope)
        rows = app.state.repository.list_faces(collection_id, person_id, after, limit + 1)
        page, more = rows[:limit], len(rows) > limit
        next_cursor = (
            app.state.service.cursors.encode(scope, page[-1]["id"]) if more and page else None
        )
        return _json(
            request,
            {
                "faces": [app.state.service.public_face(item) for item in page],
                "next_cursor": next_cursor,
            },
        )

    @app.get(
        "/v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image",
        tags=["persons"],
        response_class=Response,
        responses={
            200: {
                "description": "Saved 112 x 112 face crop",
                "content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    async def get_face_image(request: Request, collection_id: str, person_id: str, face_id: str):
        await authorize(request)
        app.state.service.collection(collection_id)
        face = await _blocking(
            app.state.repository.get_face_crop, collection_id, person_id, face_id
        )
        if face is None:
            existing = await _blocking(
                app.state.repository.get_face, collection_id, person_id, face_id
            )
            if existing is None:
                raise not_found("face", face_id)
            raise ApiError(
                "face_image_not_found",
                "The FaceSample has no saved image.",
                404,
            )

        image = face.get("bytes")
        if image is None and face.get("crop_path"):
            try:
                image = await _blocking(app.state.service.crops.read, str(face["crop_path"]))
            except (OSError, RuntimeError):
                image = None
        if image is None:
            raise ApiError(
                "face_image_not_found",
                "The FaceSample has no saved image.",
                404,
            )
        return Response(
            content=bytes(image),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.delete(
        "/v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}",
        tags=["persons"],
        status_code=204,
    )
    async def delete_face(request: Request, collection_id: str, person_id: str, face_id: str):
        await authorize(request)
        app.state.service.collection(collection_id)
        crop = await _blocking(app.state.service.delete_face, collection_id, person_id, face_id)
        app.state.service.crops.delete(crop)
        return Response(status_code=204)

    @app.post(
        "/v1/collections/{collection_id}/search",
        tags=["search"],
        response_model=SearchResponse,
    )
    async def search(
        request: Request,
        collection_id: str,
        image: UploadFile = File(...),
        limit: int = Form(default=5, ge=1, le=100),
        threshold: float | None = Form(default=None, ge=0.0, le=1.0),
        face_selection: str | None = Form(default=None, deprecated=True),
    ):
        """Search the Collection using the selected face.

        With liveness enabled in normal mode, HTTP 422 liveness_fake or
        liveness_input_rejected stops recognition and index search; the result is
        in error.details.liveness. This is distinct from a valid query with no
        identity matches. For input_rejected, optional liveness.reason provides
        English retry guidance and is used as error.message. Observe mode continues
        search and includes liveness in searched_face. Without evaluation, liveness is omitted. Model faults
        return HTTP 503 liveness_unavailable in either mode.
        """
        await authorize(request)
        if face_selection is not None:
            raise bad_request(
                "face_selection is configured by the Collection detection profile.",
                code="request_detection_override_not_supported",
            )
        started = time.perf_counter()
        loaded = await app.state.image_loader.from_upload(image)
        searched_face, matches, selected_threshold = await _blocking(
            app.state.service.search,
            collection_id,
            loaded,
            limit=limit,
            threshold=threshold,
        )
        return _json(
            request,
            {
                "searched_face": searched_face,
                "matches": matches,
                "threshold": selected_threshold,
                "processing_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )

    def monitor_values(payload: MonitorCreate | MonitorPatch) -> dict[str, Any]:
        values = payload.model_dump(exclude_unset=True)
        source = values.pop("source", None)
        if source is not None:
            values["source_type"] = source["type"]
            values["url"] = source["url"]
        policy = values.pop("event_policy", None)
        if policy is not None:
            values.update(policy)
        return values

    @app.post(
        "/v1/monitors",
        tags=["monitors"],
        status_code=201,
        response_model=MonitorResponse,
        responses={
            429: {
                "model": ErrorEnvelope,
                "description": "The enabled Monitor capacity is exhausted.",
            }
        },
    )
    async def create_monitor(request: Request, payload: MonitorCreate):
        await authorize(request)
        try:
            monitor = await _blocking(
                app.state.monitors.create,
                monitor_values(payload),
            )
        except MonitorAlreadyExistsError as exc:
            raise conflict(
                "The Monitor already exists.",
                code="monitor_exists",
            ) from exc
        except MonitorLimitError as exc:
            raise ApiError(
                "monitor_limit_exceeded",
                "The maximum number of enabled Monitors is already running.",
                429,
                {"max_monitors": configured.rtsp_max_streams},
            ) from exc
        return _json(request, {"monitor": monitor}, status_code=201)

    @app.get(
        "/v1/monitors",
        tags=["monitors"],
        response_model=MonitorPageResponse,
    )
    async def list_monitors(
        request: Request,
        limit: int = Query(default=50, ge=1, le=100),
        cursor: str | None = Query(default=None),
    ):
        await authorize(request)
        after = app.state.service.cursors.decode(cursor, "monitors")
        rows = await _blocking(app.state.monitors.list, after, limit + 1)
        page, more = rows[:limit], len(rows) > limit
        next_cursor = (
            app.state.service.cursors.encode("monitors", page[-1]["id"])
            if more and page
            else None
        )
        return _json(
            request,
            {"monitors": page, "next_cursor": next_cursor},
        )

    @app.get(
        "/v1/monitors/{monitor_id}",
        tags=["monitors"],
        response_model=MonitorResponse,
    )
    async def get_monitor(request: Request, monitor_id: str):
        await authorize(request)
        try:
            monitor = await _blocking(app.state.monitors.get, monitor_id)
        except MonitorNotFoundError as exc:
            raise not_found("monitor", monitor_id) from exc
        return _json(request, {"monitor": monitor})

    @app.patch(
        "/v1/monitors/{monitor_id}",
        tags=["monitors"],
        response_model=MonitorResponse,
        responses={
            429: {
                "model": ErrorEnvelope,
                "description": "The enabled Monitor capacity is exhausted.",
            }
        },
    )
    async def patch_monitor(
        request: Request,
        monitor_id: str,
        payload: MonitorPatch,
    ):
        await authorize(request)
        try:
            monitor = await _blocking(
                app.state.monitors.update,
                monitor_id,
                monitor_values(payload),
            )
        except MonitorNotFoundError as exc:
            raise not_found("monitor", monitor_id) from exc
        except MonitorLimitError as exc:
            raise ApiError(
                "monitor_limit_exceeded",
                "The maximum number of enabled Monitors is already running.",
                429,
                {"max_monitors": configured.rtsp_max_streams},
            ) from exc
        return _json(request, {"monitor": monitor})

    @app.delete(
        "/v1/monitors/{monitor_id}",
        tags=["monitors"],
        status_code=204,
    )
    async def delete_monitor(request: Request, monitor_id: str):
        await authorize(request)
        try:
            await _blocking(app.state.monitors.delete, monitor_id)
        except MonitorNotFoundError as exc:
            raise not_found("monitor", monitor_id) from exc
        return Response(status_code=204)

    @app.get(
        "/v1/monitors/{monitor_id}/state",
        tags=["monitors"],
        response_model=MonitorStateResponse,
    )
    async def monitor_state(request: Request, monitor_id: str):
        """Read the current RTSP detection and recognition state.

        Evaluated faces include the same liveness result as image endpoints:
        status, is_live, and live_score, plus optional English reason for input_rejected.
        Older results may omit reason. In normal mode, blocked faces have status=liveness_blocked and
        contribute to liveness_blocked_faces, not unknown_faces. They do not run
        identity search or emit person_enter events. Observe mode continues
        recognition and identity events. Runtime liveness failures are inference
        errors and must not be interpreted as fake faces.
        """
        await authorize(request)
        try:
            state = await _blocking(app.state.monitors.state, monitor_id)
        except MonitorNotFoundError as exc:
            raise not_found("monitor", monitor_id) from exc
        return _json(request, {"state": state})

    @app.get(
        "/v1/monitors/{monitor_id}/events",
        tags=["monitors"],
        response_model=MonitorEventPageResponse,
    )
    async def monitor_events(
        request: Request,
        monitor_id: str,
        cursor: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        await authorize(request)
        try:
            page = await _blocking(
                app.state.monitors.events,
                monitor_id,
                cursor=cursor,
                limit=limit,
            )
        except MonitorNotFoundError as exc:
            raise not_found("monitor", monitor_id) from exc
        return _json(request, page)

    @app.get(
        "/v1/monitors/{monitor_id}/preview.mjpeg",
        tags=["monitors"],
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "Live Motion JPEG preview stream.",
                "content": {
                    "multipart/x-mixed-replace": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def monitor_preview(request: Request, monitor_id: str):
        await authorize(request)
        try:
            stream = app.state.monitors.preview(monitor_id)
        except MonitorNotFoundError as exc:
            raise not_found("monitor", monitor_id) from exc
        except MonitorPreviewDisabledError as exc:
            raise conflict(
                "Web preview is disabled for this Monitor.",
                code="preview_disabled",
            ) from exc
        except MonitorUnavailableError as exc:
            raise ApiError(
                "stream_unavailable",
                "The Monitor does not currently have an RTSP frame to preview.",
                503,
            ) from exc
        return StreamingResponse(
            stream,
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    if not configured.web_ui_disabled:
        frontend = SERVER_DIR / "frontend"
        app.mount("/assets", StaticFiles(directory=frontend), name="assets")
        app.mount(
            "/guide-images",
            StaticFiles(directory=SERVER_DIR / "docs" / "images"),
            name="guide-images",
        )

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(frontend / "index.html")

        @app.get("/docs", include_in_schema=False)
        async def docs():
            return FileResponse(frontend / "openapi.html")

        @app.get(
            "/guide-content/{locale_code}/{document}.md",
            include_in_schema=False,
        )
        async def guide_content(locale_code: str, document: str):
            """Serve only versioned Markdown bundled in this image."""

            path = DOCUMENTS.get(locale_code, {}).get(document)
            if path is None or not path.is_file():
                raise StarletteHTTPException(
                    status_code=404, detail="Documentation not found"
                )
            return FileResponse(
                path,
                media_type="text/markdown; charset=utf-8",
                headers={
                    "Cache-Control": "public, max-age=300",
                    "X-Content-Type-Options": "nosniff",
                },
            )

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})["bearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
        }
        components.setdefault("schemas", {})["ErrorEnvelope"] = {
            "type": "object",
            "required": ["error", "request_id"],
            "properties": {
                "error": {
                    "type": "object",
                    "required": ["code", "message", "details"],
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "details": {"type": "object"},
                    },
                },
                "request_id": {"type": "string", "format": "uuid"},
            },
        }
        error_response = {
            "description": "Error",
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/ErrorEnvelope"}}
            },
        }
        for path, operations in schema.get("paths", {}).items():
            for operation in operations.values():
                if not isinstance(operation, dict) or "responses" not in operation:
                    continue
                if path.startswith("/v1/") and path != "/v1/health":
                    operation["security"] = [{"bearerAuth": []}]
                operation["responses"].pop("422", None)
                for status in ("400", "401", "404", "409", "413", "422", "500", "503"):
                    operation["responses"].setdefault(status, error_response)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    return app


app = create_app()
