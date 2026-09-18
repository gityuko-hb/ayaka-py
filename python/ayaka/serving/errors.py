from __future__ import annotations


class ServingError(RuntimeError):
    """Base error crossing the engine/serving boundary"""

    code = "serving_error"
    status_code = 500
    retryable = False

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


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
