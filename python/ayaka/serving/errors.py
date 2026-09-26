from __future__ import annotations

from ayaka.kvcache.resize import CacheRebuildRejected, ResizeRejectionReason


class ServingError(RuntimeError):
    """Base error crossing the engine/serving boundary"""

    code = "serving_error"
    status_code = 500
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.detail = detail


class InvalidRequestError(ServingError):
    code = "invalid_request"
    status_code = 400


class UnsupportedFeatureError(InvalidRequestError):
    code = "unsupported_feature"


class ModelNotFoundError(ServingError):
    code = "model_not_found"
    status_code = 404


class ContextLengthExceededError(InvalidRequestError):
    code = "context_length_exceeded"


class EngineUnavailableError(ServingError):
    code = "engine_unavailable"
    status_code = 503
    retryable = True


class RequestCancelledError(ServingError):
    code = "request_cancelled"
    status_code = 499


class DeadlineExceededError(ServingError):
    """The request deadline elapsed before or during execution."""

    code = "deadline_exceeded"
    status_code = 408


class StructuredOutputError(InvalidRequestError):
    code = "structured_output_error"


class ParserError(ServingError):
    code = "parser_error"


class UnknownParserError(ParserError):
    code = "unknown_parser"


class OverloadedError(ServingError):
    code = "rate_limit_exceeded"
    status_code = 429
    retryable = True


class AuthenticationError(ServingError):
    code = "authentication_error"
    status_code = 401


class CacheResizeRejectedError(ServingError):
    """The requested cache capacity does not fit the device budget."""

    code = "cache_resize_rejected"
    status_code = 422


class CacheResizeBusyError(ServingError):
    """The engine is serving requests; retry the resize when idle."""

    code = "cache_resize_busy"
    status_code = 409
    retryable = True


def cache_resize_error(exc: CacheRebuildRejected) -> ServingError:
    """Map a core resize rejection onto the HTTP-facing error taxonomy."""
    message = str(exc)
    detail = {
        "reason": exc.reason.value,
        "requested_pages": exc.requested_pages,
        "need_bytes": exc.need_bytes,
        "available_bytes": exc.available_bytes,
        "old_bytes": exc.old_bytes,
    }
    if exc.reason is ResizeRejectionReason.BUSY:
        return CacheResizeBusyError(message, detail=detail)
    if exc.reason is ResizeRejectionReason.INVALID_PAGES:
        return InvalidRequestError(message, detail=detail)
    if exc.reason is ResizeRejectionReason.UNSUPPORTED:
        return UnsupportedFeatureError(message, detail=detail)
    return CacheResizeRejectedError(message, detail=detail)
