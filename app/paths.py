from __future__ import annotations

MAX_RELATIVE_PATH_LEN = 500


def sanitize_relative_path(raw: str) -> str:
    """Normalize and validate a client-supplied relative upload path."""
    if not raw or not raw.strip():
        raise ValueError("filename missing")

    normalized = raw.replace("\\", "/").lstrip("/")
    if "\x00" in normalized:
        raise ValueError("invalid path")

    parts = normalized.split("/")
    if not parts:
        raise ValueError("invalid path")

    for part in parts:
        if not part or part in {".", ".."}:
            raise ValueError("invalid path")

    if len(normalized) > MAX_RELATIVE_PATH_LEN:
        raise ValueError(f"path exceeds {MAX_RELATIVE_PATH_LEN} characters")

    return normalized
