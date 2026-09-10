"""OpenAI-shaped error envelope.

A drop-in proxy has to fail the way the thing it replaces fails, otherwise
client SDK error handling breaks. The official clients read
``{"error": {"message", "type", "param", "code"}}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


class CacheLLMError(HTTPException):
    def __init__(
        self,
        status_code: int,
        message: str,
        err_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.message = message
        self.err_type = err_type
        self.code = code
        self.param = param

    def envelope(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.err_type,
                "param": self.param,
                "code": self.code,
            }
        }


class AuthError(CacheLLMError):
    def __init__(self, message: str = "Invalid API key provided.") -> None:
        super().__init__(401, message, "invalid_request_error", code="invalid_api_key")


class UpstreamError(CacheLLMError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(status_code, message, "api_error", code="upstream_error")


class ModelNotFoundError(CacheLLMError):
    def __init__(self, model: str) -> None:
        super().__init__(
            404,
            f"The model `{model}` does not exist or you do not have access to it.",
            "invalid_request_error",
            code="model_not_found",
            param="model",
        )


class CacheMissError(CacheLLMError):
    """Raised when a client sent ``X-Cache-Control: only-if-cached`` and missed."""

    def __init__(self) -> None:
        super().__init__(
            504,
            "No cached response above the similarity threshold and the request "
            "was sent with X-Cache-Control: only-if-cached.",
            "api_error",
            code="cache_miss",
        )
