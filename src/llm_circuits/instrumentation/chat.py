from __future__ import annotations

_BOS_COUNTS: dict[str, int] = {
    "qwen3": 1,
    "gemma2": 1,
}


def prepare_messages(
    prompt: str,
    family: str,
    *,
    system: str | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Build a chat-template message list and return the family-specific BOS token count.

    Parameters
    ----------
    prompt:
        The user's plain text prompt.
    family:
        Model family string (``"qwen3"`` or ``"gemma2"``).
    system:
        Optional system message prepended to the conversation.

    Returns
    -------
    messages:
        List of message dicts ready for ``tokenizer.apply_chat_template()``.
    n_bos_tokens:
        Number of leading BOS tokens for this family.
    """
    if family not in _BOS_COUNTS:
        raise ValueError(
            f"Unknown model family {family!r}. Supported families: {sorted(_BOS_COUNTS)}"
        )

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return messages, _BOS_COUNTS[family]
