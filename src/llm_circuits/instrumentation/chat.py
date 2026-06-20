from __future__ import annotations

from typing import Any

_BOS_COUNTS: dict[str, int] = {
    "qwen3": 1,
}


def prepare_messages(
    prompt: str,
    family: str,
    *,
    system: str | None = None,
    enable_thinking: bool = True,
) -> tuple[list[dict[str, str]], int, dict[str, Any]]:
    """Build a chat-template message list and return the family-specific BOS token count.

    Parameters
    ----------
    prompt:
        The user's plain text prompt.
    family:
        Model family string (currently ``"qwen3"``).
    system:
        Optional system message prepended to the conversation.
    enable_thinking:
        Whether to allow the model's thinking/reasoning mode.  Set to
        ``False`` to force the model to answer directly (Qwen3 only).

    Returns
    -------
    messages:
        List of message dicts ready for ``tokenizer.apply_chat_template()``.
    n_bos_tokens:
        Number of leading BOS tokens for this family.
    template_kwargs:
        Extra keyword arguments to forward to ``apply_chat_template()``
        (e.g. ``enable_thinking`` for Qwen3).
    """
    if family not in _BOS_COUNTS:
        raise ValueError(
            f"Unknown model family {family!r}. Supported families: {sorted(_BOS_COUNTS)}"
        )

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    template_kwargs: dict[str, Any] = {}
    if family == "qwen3":
        template_kwargs["enable_thinking"] = enable_thinking

    return messages, _BOS_COUNTS[family], template_kwargs
