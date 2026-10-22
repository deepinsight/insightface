from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ApiError(Exception):
    code: str
    message: str
    status_code: int = 400
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message

    def body(self, request_id: str) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
            "request_id": request_id,
        }


def bad_request(message: str, *, code: str = "invalid_request", **details: Any) -> ApiError:
    return ApiError(code, message, 400, details)


def not_found(resource: str, identifier: str) -> ApiError:
    return ApiError(
        f"{resource}_not_found",
        f"{resource.replace('_', ' ').title()} '{identifier}' was not found.",
        404,
    )


def conflict(message: str, *, code: str = "resource_conflict") -> ApiError:
    return ApiError(code, message, 409)


def unprocessable(message: str, *, code: str, **details: Any) -> ApiError:
    return ApiError(code, message, 422, details)
