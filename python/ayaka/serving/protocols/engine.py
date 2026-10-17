"""Bridge the protocol ports to Ayaka's existing GenerationPipeline.

The owner thread, admission, bounded GenerationHandle queue, cancellation,
grammar matcher, scheduler, and KV cleanup remain owned by Ayaka.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from ayaka.serving.events import GenerationFailed, GenerationFinished, UsageUpdate

from .error import ApiError, classify
from .source import (
    Admitted,
    ChatInput,
    Completed,
    Failed,
    GenerationRequest,
    GenerationStream,
    HostChat,
    LoadProgress,
    Output,
)


def to_spec(model: str, request: GenerationRequest):
    """Translate supported semantics; reject controls the backend cannot honor."""
    from ayaka.serving.prepare import GenerationSpec

    input, controls = request.input, request.controls
    if input.reasoning is not None or controls.reasoning_budget is not None:
        raise ApiError(
            "Ayaka has no model reasoning-profile/budget adapter yet",
            code="unsupported_feature",
            param="reasoning_effort",
        )
    if input.template_arguments:
        raise ApiError(
            "Ayaka does not yet forward chat_template_kwargs",
            code="unsupported_feature",
            param="chat_template_kwargs",
        )
    if input.output.get("type") == "grammar":
        raise ApiError(
            "raw grammar requires an Ayaka grammar backend adapter",
            code="unsupported_feature",
            param="response_format",
        )
    tools = list(input.tools)
    choice = input.tool_choice
    if isinstance(choice, dict) and choice.get("type") == "allowed_tools":
        tools = [tool for tool in tools if tool["function"]["name"] in choice["names"]]
        choice = choice["mode"]
    if tools and choice != "none" and not input.parallel_tool_calls and choice == "auto":
        raise ApiError(
            "Ayaka's automatic tool parser cannot enforce parallel_tool_calls=false",
            code="unsupported_feature",
            param="parallel_tool_calls",
        )
    messages = []
    for source in input.messages:
        message = dict(source)
        content = message.get("content")
        if isinstance(content, list):
            if any(part["type"] != "text" for part in content):
                raise ApiError(
                    "Ayaka's serving preprocessor is text-only; a vision encoder is required",
                    code="unsupported_feature",
                    param="messages",
                )
            message["content"] = "".join(part["text"] for part in content)
        messages.append(message)
    sampling = controls.sampling
    try:
        return GenerationSpec(
            model=model,
            messages=messages,
            max_tokens=controls.max_output_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k or -1,
            min_p=sampling.min_p,
            repetition_penalty=sampling.repetition_penalty,
            presence_penalty=sampling.presence_penalty,
            frequency_penalty=sampling.frequency_penalty,
            seed=sampling.seed,
            stop=list(controls.stops),
            ignore_eos=controls.ignore_eos,
            cache_prompt=controls.cache_prompt,
            timeout=controls.timeout,
            min_tokens=controls.min_tokens,
            stop_token_ids=list(controls.stop_token_ids),
            tools=tools,
            tool_choice=choice,
            response_format=None if input.output == {"type": "text"} else input.output,
        )
    except ValueError as exc:
        raise ApiError(str(exc), code="unsupported_feature") from exc


class AyakaInvocation:
    def __init__(self, models: AyakaModels, model: str, tenant: str) -> None:
        self.models, self.model, self.tenant = models, model, tenant
        self.handle = None
        self.used = False

    def observe_write(self) -> None:
        """Preserve the existing client TTFT metric at the first actual socket write."""
        handle = self.handle
        if handle is not None and not handle.first_socket_write_ns:
            handle.first_socket_write_ns = time.monotonic_ns()
            self.models.pipeline.service.stats.observe_client_ttft(handle.ingress_ns)

    async def aclose(self) -> None:
        if self.handle is not None:
            self.models.pipeline.release(self.handle)

    async def generate(self, request: GenerationRequest) -> GenerationStream:
        if self.used:
            raise ApiError.server("model invocation is single-use")
        self.used = True
        try:
            spec = to_spec(self.model, request)
            self.handle = await self.models.pipeline.prepare(spec, self.tenant)
            yield Admitted(self.handle.request.prompt_len)
            async for event in self.models.pipeline.generate_events(self.handle):
                if isinstance(event, GenerationFinished):
                    if event.finish_reason in ("cancelled", "timeout"):
                        yield Failed(
                            ApiError(
                                "generation cancelled"
                                if event.finish_reason == "cancelled"
                                else "request deadline exceeded",
                                status=499 if event.finish_reason == "cancelled" else 408,
                                code="request_cancelled"
                                if event.finish_reason == "cancelled"
                                else "deadline_exceeded",
                            )
                        )
                    else:
                        yield Completed(event.usage, event.finish_reason)
                    return
                if isinstance(event, GenerationFailed):
                    yield Failed(
                        ApiError(
                            event.message,
                            status=500,
                            code=event.code,
                            kind="server_error",
                            retryable=event.retryable,
                        )
                    )
                    return
                if isinstance(event, UsageUpdate):
                    continue
                yield Output(event)
        except Exception as exc:
            yield Failed(classify(exc))
        finally:
            await self.aclose()


class AyakaHost:
    """Use tokenizer artifacts without reserving a device-generation slot."""

    def __init__(self, models: AyakaModels, model: str) -> None:
        self.models, self.model = models, model

    def _source(self, input: ChatInput):
        spec = to_spec(self.model, GenerationRequest(input))
        return self.models.processor.render_source(spec)

    def count(self, input: ChatInput) -> int:
        from ayaka.tokenizers.contracts import EncodeOptions

        try:
            encoded = self.models.processor.tokenizer.encode(
                self._source(input), EncodeOptions(max_output_tokens=0)
            )
        except (ValueError, TypeError) as exc:
            raise ApiError(str(exc), param="messages") from exc
        except Exception as exc:
            raise classify_host_error(exc) from exc
        return len(encoded.token_ids)

    def apply_template(self, input: ChatInput) -> dict[str, Any]:
        if input.output != {"type": "text"} or (
            input.tools and input.tool_choice not in ("auto", "none")
        ):
            raise ApiError(
                "compiled grammar inspection requires an Ayaka grammar metadata adapter",
                code="unsupported_feature",
            )
        tokenizer = self.models.processor.tokenizer
        source = self._source(input)
        try:
            prompt = tokenizer.render_prompt(source)
            without_generation = tokenizer.render_prompt(
                replace(source, add_generation_prompt=False)
            )
        except (ValueError, TypeError) as exc:
            raise ApiError(str(exc), param="messages") from exc
        except Exception as exc:
            raise classify_host_error(exc) from exc
        if not prompt.startswith(without_generation):
            raise ApiError(
                "template generation prefix requires model-specific host metadata",
                code="unsupported_feature",
            )
        return {
            "prompt": prompt,
            "generation_prompt": prompt[len(without_generation) :],
            "grammar": "",
            "grammar_lazy": False,
            "grammar_triggers": [],
            "preserved_tokens": [],
            "additional_stops": [],
            "supports_thinking": False,
            "template_fingerprint": self.properties()["template_fingerprint"],
        }

    def properties(self) -> dict[str, Any]:
        from .properties import ayaka_properties

        return ayaka_properties(self.models.processor)


def classify_host_error(error: Exception) -> ApiError:
    from ayaka.tokenizers.service import TokenizerOverloaded

    if isinstance(error, TokenizerOverloaded):
        return ApiError(str(error), status=429, code="rate_limit_exceeded", retryable=True)
    return classify(error)


class AyakaModels:
    """One resident model today; a model registry can implement the same ports later."""

    def __init__(self, pipeline, processor, *, host: HostChat | None = None) -> None:
        self.pipeline, self.processor, self._host = pipeline, processor, host

    def _check(self, model: str) -> None:
        if model != self.processor.config.model:
            raise ApiError(
                f"unknown model: {model}", status=404, code="model_not_found", param="model"
            )

    async def invoke(
        self,
        model: str,
        progress: LoadProgress | None = None,
        *,
        tenant: str = "anonymous",
    ) -> AyakaInvocation:
        self._check(model)
        return AyakaInvocation(self, model, tenant)

    async def host(self, model: str) -> HostChat:
        self._check(model)
        return self._host if self._host is not None else AyakaHost(self, model)
