"""Pull runnable Python code out of a model response."""

from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
# A fenced block; the closing fence is optional so truncated responses still yield code.
_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)


def extract_code(response: str, entry_point: str | None = None) -> str:
    """Return the code block most likely to contain the solution.

    Preference order: a block defining `entry_point`, then the longest python
    block, then the longest block of any language, then the raw text.
    """
    text = _THINK.sub("", response or "").strip()
    blocks = [(lang.lower(), body.strip()) for lang, body in _FENCE.findall(text)]
    blocks = [(lang, body) for lang, body in blocks if body]
    if not blocks:
        return text

    if entry_point:
        defining = [b for _, b in blocks if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", b)]
        if defining:
            return max(defining, key=len)

    python = [b for lang, b in blocks if lang in ("python", "py", "python3")]
    return max(python or [b for _, b in blocks], key=len)
