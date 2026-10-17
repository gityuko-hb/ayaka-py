"""Protocol error taxonomy, independent of devices and model residency."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from starlette.responses import JSONResponse


class ApiError(Exception):
    """One classified error, projected into HTTP, SSE, or WebSocket framing."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        code: str = "invalid_request",
        kind: str = "invalid_request_error",
        param: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message, self.status, self.code = message, status, code
        self.kind, self.param, self.retryable = kind, param, retryable

    @classmethod
    def server(cls, message: str) -> ApiError:
        return cls(message, status=500, code="backend_error", kind="server_error", retryable=True)

    def body(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "message": self.message,
            "type": self.kind,
            "code": self.code,
        }
        if self.param is not None:
            value["param"] = self.param
        return value

    def anthropic(self, request_id: str) -> dict[str, Any]:
        return {
            "type": "error",
            "error": {"type": anthropic_error_type(self.status), "message": self.message},
            "request_id": request_id,
        }

    def response(self, *, anthropic: bool = False, request_id: str | None = None) -> JSONResponse:
        request_id = request_id or "req_" + uuid4().hex
        headers = {"x-request-id": request_id}
        if self.retryable and self.status in (429, 503):
            headers["Retry-After"] = "1"
        if anthropic:
            headers["request-id"] = request_id
        payload = self.anthropic(request_id) if anthropic else {"error": self.body()}
        return JSONResponse(payload, status_code=self.status, headers=headers)


def anthropic_error_type(status: int) -> str:
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
        503: "overloaded_error",
    }.get(status, "api_error")


def classify(error: Exception) -> ApiError:
    """Adapt Ayaka errors without importing the runtime into the protocol plane."""
    if isinstance(error, ApiError):
        return error
    status = getattr(error, "status_code", 500)
    code = getattr(error, "code", "backend_error")
    kind = "invalid_request_error" if status < 500 else "server_error"
    message = str(error)
    retryable = getattr(error, "retryable", status >= 500)
    if status == 503 and retryable:
        message = "Temporarily unavailable; retry shortly. " + message
    return ApiError(
        message,
        status=status,
        code=code,
        kind=kind,
        param=getattr(error, "param", None),
        retryable=retryable,
    )
