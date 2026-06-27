from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

_FOLDER_RE = re.compile(r"/(?:drive/)?folders/([a-zA-Z0-9_-]+)")
_FILE_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")


@dataclass(frozen=True)
class ParsedDriveUrl:
    root_id: str
    is_single_file: bool


def parse_drive_url(url: str) -> ParsedDriveUrl:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Drive URL is required")

    parsed = urlparse(raw)
    if parsed.netloc and "google.com" not in parsed.netloc:
        raise ValueError("URL must be a Google Drive link")

    path = parsed.path or ""
    folder_match = _FOLDER_RE.search(path)
    if folder_match:
        return ParsedDriveUrl(root_id=folder_match.group(1), is_single_file=False)

    file_match = _FILE_RE.search(path)
    if file_match:
        return ParsedDriveUrl(root_id=file_match.group(1), is_single_file=True)

    query_id = parse_qs(parsed.query).get("id", [None])[0]
    if query_id:
        return ParsedDriveUrl(root_id=query_id, is_single_file=True)

    raise ValueError(
        "Unrecognized Drive URL. Use a folder link, file link, or open?id= link."
    )
