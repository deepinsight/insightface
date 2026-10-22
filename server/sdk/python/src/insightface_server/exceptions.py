"""Typed errors raised by the InsightFace Server client."""

from __future__ import annotations

from typing import Any, Mapping, Optional


class InsightFaceServerError(Exception):
    """Base class for API and client errors.

    Attributes are deliberately safe to log: response bodies containing image
    or embedding data are not retained.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "client_error",
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.request_id = request_id
        self.details = dict(details or {})

    def __str__(self) -> str:
        suffix = f" (request_id={self.request_id})" if self.request_id else ""
        return f"{self.code}: {self.message}{suffix}"


class TransportError(InsightFaceServerError):
    """The server could not be reached or the request timed out."""


class AuthenticationError(InsightFaceServerError):
    """The API key is missing or invalid."""


class NotFoundError(InsightFaceServerError):
    """The requested resource does not exist."""


class ConflictError(InsightFaceServerError):
    """The requested change conflicts with current server state."""


class PayloadTooLargeError(InsightFaceServerError):
    """An uploaded image or request body exceeds a server limit."""


class ValidationError(InsightFaceServerError):
    """The request or image cannot be processed as supplied."""


class RateLimitError(InsightFaceServerError):
    """The server rejected the request due to rate limiting."""


class ServiceUnavailableError(InsightFaceServerError):
    """The model, GPU, or another required service is unavailable."""


class ServerError(InsightFaceServerError):
    """The server returned an unexpected error."""
