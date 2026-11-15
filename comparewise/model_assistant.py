"""Shared configuration and bounded sanitizers for model assistants."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


class ModelAssistantError(Exception):
    """A safe, user-facing model configuration or request failure."""


MAX_QUESTION_CHARS = 2000
MAX_PRIORITY_CHARS = 200
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 1500
MAX_FIELD_CHARS = 1000
MAX_FEATURES = 20
MAX_SPECIFICATIONS = 30
REQUEST_TIMEOUT = (10, 30)

_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+|data:[^\s<>\"']+", re.IGNORECASE)


def _text(value: Any, limit: int) -> str | None:
    """Keep bounded text while removing direct links, including embedded ones."""
    if not isinstance(value, str):
        return None
    # Bound before applying the regex, so even unusually large inputs stay cheap.
    return _URL.sub("[link removed]", value[:limit]).strip()


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _recent_history(history: Sequence[Mapping[str, Any]] | None, question: str) -> list[dict[str, str]]:
    if not isinstance(history, (list, tuple)):
        return []
    recent = []
    for message in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(message, Mapping) or message.get("role") not in ("user", "assistant"):
            continue
        content = _text(message.get("content"), MAX_HISTORY_CHARS)
        if content:
            recent.append({"role": message["role"], "content": content})
    # The existing UI can append the current question before invoking an assistant.
    if recent and recent[-1]["role"] == "user" and recent[-1]["content"] == question:
        recent.pop()
    return recent