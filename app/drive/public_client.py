from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import httpx

from ..config import settings

DRIVE_API = "https://www.googleapis.com/drive/v3"
FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"

SUPPORTED_BINARY_MIMES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/html",
}

EXPORT_MIMES = {
    GOOGLE_DOC_MIME: ("text/plain", ".txt"),
}


@dataclass(frozen=True)
class DriveFileMeta:
    drive_file_id: str
    name: str
    mime_type: str
    modified_time: datetime
    md5_checksum: str | None
    size: int | None
    relative_path: str

    @property
    def is_folder(self) -> bool:
        return self.mime_type == FOLDER_MIME

    @property
    def is_supported(self) -> bool:
        if self.is_folder:
            return False
        return self.mime_type in SUPPORTED_BINARY_MIMES or self.mime_type in EXPORT_MIMES


class PublicDriveClient:
    """List and download files from publicly shared Google Drive folders."""

    def __init__(self, api_key: str | None = None, timeout: float | None = None) -> None:
        self._api_key = api_key if api_key is not None else settings.google_api_key
        self._timeout = timeout if timeout is not None else settings.drive_request_timeout_seconds
        if not self._api_key:
            raise ValueError(
                "RAG_GOOGLE_API_KEY is not set. Add it to .env to enable Drive sync."
            )

    def get_file_metadata(self, file_id: str) -> DriveFileMeta:
        data = self._get(f"/files/{file_id}", params={"fields": _FILE_FIELDS})
        return _meta_from_api(data, relative_path=data.get("name", file_id))

    def list_children(self, folder_id: str) -> list[dict]:
        files: list[dict] = []
        page_token: str | None = None
        while True:
            params = {
                "q": f"'{folder_id}' in parents and trashed=false",
                "fields": f"nextPageToken,files({_FILE_FIELDS})",
                "pageSize": "100",
            }
            if page_token:
                params["pageToken"] = page_token
            data = self._get("/files", params=params)
            files.extend(data.get("files", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return files

    def walk_folder(
        self, folder_id: str, path_prefix: str = ""
    ) -> Iterator[DriveFileMeta]:
        for item in self.list_children(folder_id):
            name = item.get("name") or item["id"]
            rel = f"{path_prefix}{name}" if not path_prefix else f"{path_prefix}/{name}"
            meta = _meta_from_api(item, relative_path=rel)
            if meta.is_folder:
                yield from self.walk_folder(meta.drive_file_id, rel)
            else:
                yield meta

    def walk_source(
        self, root_id: str, *, is_single_file: bool
    ) -> list[DriveFileMeta]:
        if is_single_file:
            root = self.get_file_metadata(root_id)
            if root.is_folder:
                return list(self.walk_folder(root.drive_file_id))
            return [root]
        return list(self.walk_folder(root_id))

    def download_file(self, meta: DriveFileMeta, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if meta.mime_type in EXPORT_MIMES:
            export_mime, suffix = EXPORT_MIMES[meta.mime_type]
            target = dest if dest.suffix else dest.with_suffix(suffix)
            self._download_export(meta.drive_file_id, export_mime, target)
            if target != dest and dest.exists():
                dest.unlink(missing_ok=True)
            if target != dest:
                target.rename(dest)
            return

        self._download_binary(meta.drive_file_id, dest)

    def _download_binary(self, file_id: str, dest: Path) -> None:
        url = f"{DRIVE_API}/files/{file_id}"
        with httpx.stream(
            "GET",
            url,
            params={"alt": "media", "key": self._api_key},
            timeout=self._timeout,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            with dest.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

    def _download_export(self, file_id: str, mime_type: str, dest: Path) -> None:
        url = f"{DRIVE_API}/files/{file_id}/export"
        with httpx.stream(
            "GET",
            url,
            params={"mimeType": mime_type, "key": self._api_key},
            timeout=self._timeout,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            with dest.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

    def _get(self, path: str, params: dict | None = None) -> dict:
        merged = dict(params or {})
        merged["key"] = self._api_key
        try:
            response = httpx.get(
                f"{DRIVE_API}{path}",
                params=merged,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {403, 404}:
                raise ValueError(
                    "Drive folder is not accessible. Ensure it is shared as "
                    "'Anyone with the link can view' and the API key is valid."
                ) from exc
            raise ValueError(f"Google Drive API error ({status}): {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise ValueError(f"Google Drive request failed: {exc}") from exc
        return response.json()


_FILE_FIELDS = "id,name,mimeType,modifiedTime,md5Checksum,size"


def _parse_drive_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _meta_from_api(data: dict, *, relative_path: str) -> DriveFileMeta:
    size_raw = data.get("size")
    return DriveFileMeta(
        drive_file_id=data["id"],
        name=data.get("name") or data["id"],
        mime_type=data.get("mimeType") or "application/octet-stream",
        modified_time=_parse_drive_time(data.get("modifiedTime")),
        md5_checksum=data.get("md5Checksum"),
        size=int(size_raw) if size_raw is not None else None,
        relative_path=relative_path,
    )
