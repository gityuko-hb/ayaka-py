"""Host model metadata. Unknown template capabilities remain unknown."""

from __future__ import annotations

import hashlib
from typing import Any


def ayaka_properties(processor) -> dict[str, Any]:
    from ayaka.__version__ import __version__

    tokenizer = processor.tokenizer
    adapter = tokenizer.tokenizer
    template = getattr(adapter, "chat_template", None)
    template = template if isinstance(template, str) else ""
    capabilities = getattr(adapter, "template_capabilities", {})
    # A configured output parser does not prove that the model template supports
    # any tool/reasoning format. Only explicit host capability evidence is used.
    known = {
        name: capabilities.get(name)
        for name in (
            "string_content",
            "typed_content",
            "tools",
            "tool_calls",
            "parallel_tool_calls",
            "system_role",
            "preserve_reasoning",
            "object_arguments",
            "enable_thinking",
        )
    }
    return {
        "build_info": f"ayaka {__version__}",
        "general_name": processor.config.model,
        "default_generation_settings": {"n_ctx": tokenizer.max_model_len},
        "modalities": {"vision": False, "audio": False, "video": False},
        "chat_template": template,
        "template_fingerprint": hashlib.sha256(template.encode()).hexdigest(),
        "template_capabilities": known,
        "reasoning": {"default_reasoning_effort": None, "reasoning_efforts": []},
    }
