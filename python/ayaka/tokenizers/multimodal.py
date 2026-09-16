from __future__ import annotations

from ayaka.tokenizers.contracts import ChatInput, EncodeOptions, TextInput, TokenIdsInput
from ayaka.tokenizers.service import TokenizerOverloaded, TokenizerService


def encode_media_chat(
    service: TokenizerService,
    input: ChatInput,
    options: EncodeOptions,
    markers: tuple[tuple[str, int], ...],
    placeholder: int,
):
    """Return validated tokens and media offsets; templates must preserve each marker.

       Text is encoded separately on each side of an embedding span, so no token can
      straddle a media boundary. Template control tokens remain in their original
    positions. Markers are generated internally, never interpreted from user text.
    """
    if options.truncate:
        raise ValueError("media chat cannot truncate embedding spans")
    if type(placeholder) is not int or not 0 <= placeholder < service.model_vocab_size:
        raise ValueError("invalid media placeholder token")
    if service._weight(input) > service.config.encode_max_pending_bytes:
        raise TokenizerOverloaded("media chat input exceeds byte budget")
    with service._adapter_lock:
        rendered, _ = service._render(input)
    if len(rendered.encode("utf-8")) > service.config.encode_max_pending_bytes:
        raise TokenizerOverloaded("rendered media chat exceeds byte budget")
    tokens, starts = [], []
    cursor = 0
    for marker, count in markers:
        if not marker or type(count) is not int or count < 1 or rendered.count(marker) != 1:
            raise ValueError("chat template must preserve each media placeholder exactly once")
        end = rendered.index(marker)
        if end < cursor:
            raise ValueError("chat template reordered media placeholders")
        text = rendered[cursor:end]
        if text:
            tokens.extend(
                service.encode(TextInput(text, add_special_tokens=False), options).token_ids
            )
        starts.append(len(tokens))
        tokens.extend([placeholder] * count)
        cursor = end + len(marker)
    if rendered[cursor:]:
        tokens.extend(
            service.encode(
                TextInput(rendered[cursor:], add_special_tokens=False), options
            ).token_ids
        )
    result = service.encode(TokenIdsInput(tuple(tokens)), options)
    return result, tuple(starts)
