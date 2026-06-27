from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from .. import repository
from ..config import settings
from ..indexer import Indexer
from ..paths import sanitize_relative_path
from .public_client import EXPORT_MIMES, DriveFileMeta, PublicDriveClient


@dataclass
class DriveSyncFileResult:
    relative_path: str
    drive_file_id: str
    status: str
    chunks_added: int = 0
    message: str = ""


@dataclass
class DriveSyncResult:
    listed: int = 0
    skipped: int = 0
    indexed: int = 0
    updated: int = 0
    unsupported: int = 0
    errors: int = 0
    files: list[DriveSyncFileResult] = field(default_factory=list)


class DriveSyncService:
    def __init__(self, client: PublicDriveClient | None = None) -> None:
        self._client = client or PublicDriveClient()

    def sync_source(
        self,
        session: Session,
        indexer: Indexer,
        source,
    ) -> DriveSyncResult:
        remote_files = self._client.walk_source(
            source.root_id,
            is_single_file=source.is_single_file,
        )
        result = DriveSyncResult(listed=len(remote_files))
        storage_root = settings.docs_dir / source.dataset_id / "drive" / source.id

        for meta in remote_files:
            file_result = self._sync_one(
                session, indexer, source, meta, storage_root
            )
            result.files.append(file_result)
            if file_result.status == "skipped":
                result.skipped += 1
            elif file_result.status == "indexed":
                result.indexed += 1
            elif file_result.status == "updated":
                result.updated += 1
            elif file_result.status == "unsupported":
                result.unsupported += 1
            else:
                result.errors += 1

        repository.touch_drive_source_synced(session, source)
        return result

    def _sync_one(
        self,
        session: Session,
        indexer: Indexer,
        source,
        meta: DriveFileMeta,
        storage_root: Path,
    ) -> DriveSyncFileResult:
        snapshot = repository.get_drive_file_snapshot(
            session, source.id, meta.drive_file_id
        )

        if not meta.is_supported:
            repository.upsert_drive_file_snapshot(
                session,
                snapshot_id=snapshot.id if snapshot else None,
                drive_source_id=source.id,
                drive_file_id=meta.drive_file_id,
                relative_path=meta.relative_path,
                name=meta.name,
                mime_type=meta.mime_type,
                md5_checksum=meta.md5_checksum,
                modified_time=meta.modified_time,
                size=meta.size,
                document_id=snapshot.document_id if snapshot else None,
                last_sync_status="unsupported",
            )
            return DriveSyncFileResult(
                relative_path=meta.relative_path,
                drive_file_id=meta.drive_file_id,
                status="unsupported",
                message="unsupported file type",
            )

        if snapshot is not None and _metadata_unchanged(snapshot, meta):
            repository.upsert_drive_file_snapshot(
                session,
                snapshot_id=snapshot.id,
                drive_source_id=source.id,
                drive_file_id=meta.drive_file_id,
                relative_path=meta.relative_path,
                name=meta.name,
                mime_type=meta.mime_type,
                md5_checksum=meta.md5_checksum,
                modified_time=meta.modified_time,
                size=meta.size,
                document_id=snapshot.document_id,
                last_sync_status="skipped",
            )
            return DriveSyncFileResult(
                relative_path=meta.relative_path,
                drive_file_id=meta.drive_file_id,
                status="skipped",
            )

        try:
            local_rel = local_relative_path(meta)
            local_path = storage_root / local_rel
            indexed_name = indexed_filename_for(source.root_name, meta)
            self._client.download_file(meta, local_path)

            index_result = indexer.index_file(
                session,
                source.dataset_id,
                local_path,
                indexed_name,
            )
            repository.upsert_drive_file_snapshot(
                session,
                snapshot_id=snapshot.id if snapshot else None,
                drive_source_id=source.id,
                drive_file_id=meta.drive_file_id,
                relative_path=meta.relative_path,
                name=meta.name,
                mime_type=meta.mime_type,
                md5_checksum=meta.md5_checksum,
                modified_time=meta.modified_time,
                size=meta.size,
                document_id=index_result.doc_id,
                last_sync_status=index_result.status,
            )
            return DriveSyncFileResult(
                relative_path=meta.relative_path,
                drive_file_id=meta.drive_file_id,
                status=index_result.status,
                chunks_added=index_result.chunks_added,
            )
        except Exception as exc:
            repository.upsert_drive_file_snapshot(
                session,
                snapshot_id=snapshot.id if snapshot else None,
                drive_source_id=source.id,
                drive_file_id=meta.drive_file_id,
                relative_path=meta.relative_path,
                name=meta.name,
                mime_type=meta.mime_type,
                md5_checksum=meta.md5_checksum,
                modified_time=meta.modified_time,
                size=meta.size,
                document_id=snapshot.document_id if snapshot else None,
                last_sync_status="error",
            )
            return DriveSyncFileResult(
                relative_path=meta.relative_path,
                drive_file_id=meta.drive_file_id,
                status="error",
                message=str(exc),
            )


def _metadata_unchanged(snapshot, meta: DriveFileMeta) -> bool:
    snap_ts = _as_utc_timestamp(snapshot.modified_time)
    meta_ts = _as_utc_timestamp(meta.modified_time)
    if snap_ts != meta_ts:
        return False
    if snapshot.md5_checksum and meta.md5_checksum:
        return snapshot.md5_checksum == meta.md5_checksum
    return snapshot.size == meta.size


def _as_utc_timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def local_relative_path(meta: DriveFileMeta) -> str:
    path = Path(meta.relative_path)
    if meta.mime_type in EXPORT_MIMES:
        if path.suffix.lower() not in {".txt", ".md", ".pdf", ".html", ".htm"}:
            return str(path.with_suffix(".txt"))
    return meta.relative_path


def indexed_filename_for(root_name: str, meta: DriveFileMeta) -> str:
    composed = f"{root_name}/{local_relative_path(meta)}"
    return sanitize_relative_path(composed)
