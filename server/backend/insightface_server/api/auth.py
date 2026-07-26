from __future__ import annotations

import logging

from fastapi import Request

from ..config import Settings
from ..errors import ApiError
from ..storage import Repository

LOGGER = logging.getLogger("insightface_server")


class ApiKeyAuthenticator:
    def __init__(self, settings: Settings, repository: Repository):
        self.settings = settings
        self.repository = repository

    def initialize(self) -> None:
        if not self.settings.auth_enabled:
            return
        if self.settings.startup_api_key:
            result = self.repository.sync_api_key(self.settings.startup_api_key)
            if result != "unchanged":
                LOGGER.info("api_key_%s", result)
        if not self.repository.has_api_keys():
            raise RuntimeError(
                "Authentication is enabled but no API key exists; set INSIGHTFACE_API_KEY "
                "when starting the server"
            )

    def require(self, request: Request) -> None:
        if not self.settings.auth_enabled:
            return
        authorization = request.headers.get("authorization", "")
        scheme, separator, key = authorization.partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not key:
            raise ApiError("unauthorized", "A valid Bearer API key is required.", 401)
        if not self.repository.verify_api_key(key):
            raise ApiError("unauthorized", "A valid Bearer API key is required.", 401)
