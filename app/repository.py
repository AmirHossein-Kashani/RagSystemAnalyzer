from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Dataset, Document, DriveFileSnapshot, DriveSource


def new_id(nbytes: int = 8) -> str:
    return secrets.token_hex(nbytes)


# ---------- Datasets ----------

def create_dataset(session: Session, name: str, description: Optional[str]) -> Dataset:
    dataset = Dataset(id=new_id(), name=name.strip(), description=description)
    session.add(dataset)
    session.flush()
    return dataset


def get_dataset(session: Session, dataset_id: str) -> Optional[Dataset]:
    return session.get(Dataset, dataset_id)


def get_dataset_by_name(session: Session, name: str) -> Optional[Dataset]:
    return session.scalar(select(Dataset).where(Dataset.name == name.strip()))


def list_datasets(session: Session) -> list[Dataset]:
    return list(session.scalars(select(Dataset).order_by(Dataset.created_at.desc())))


def delete_dataset(session: Session, dataset: Dataset) -> None:
    session.delete(dataset)


# ---------- Documents ----------

def get_document(session: Session, doc_id: str) -> Optional[Document]:
    return session.get(Document, doc_id)


def get_document_by_filename(
    session: Session, dataset_id: str, filename: str
) -> Optional[Document]:
    return session.scalar(
        select(Document).where(
            Document.dataset_id == dataset_id, Document.filename == filename
        )
    )


def list_documents(session: Session, dataset_id: str) -> list[Document]:
    return list(
        session.scalars(
            select(Document)
            .where(Document.dataset_id == dataset_id)
            .order_by(Document.indexed_at.desc())
        )
    )


def upsert_document(
    session: Session,
    *,
    doc_id: str,
    dataset_id: str,
    filename: str,
    source_path: str,
    file_hash: str,
    chunk_count: int,
) -> Document:
    doc = session.get(Document, doc_id)
    now = datetime.now(timezone.utc)
    if doc is None:
        doc = Document(
            id=doc_id,
            dataset_id=dataset_id,
            filename=filename,
            source_path=source_path,
            file_hash=file_hash,
            chunk_count=chunk_count,
            indexed_at=now,
        )
        session.add(doc)
    else:
        doc.source_path = source_path
        doc.file_hash = file_hash
        doc.chunk_count = chunk_count
        doc.indexed_at = now
    session.flush()
    return doc


def delete_document(session: Session, document: Document) -> None:
    session.delete(document)


# ---------- Google Drive ----------

def create_drive_source(
    session: Session,
    *,
    dataset_id: str,
    folder_url: str,
    root_id: str,
    root_name: str,
    is_single_file: bool,
) -> DriveSource:
    source = DriveSource(
        id=new_id(),
        dataset_id=dataset_id,
        folder_url=folder_url.strip(),
        root_id=root_id,
        root_name=root_name,
        is_single_file=is_single_file,
    )
    session.add(source)
    session.flush()
    return source


def get_drive_source(session: Session, source_id: str) -> Optional[DriveSource]:
    return session.get(DriveSource, source_id)


def get_drive_source_by_root(
    session: Session, dataset_id: str, root_id: str
) -> Optional[DriveSource]:
    return session.scalar(
        select(DriveSource).where(
            DriveSource.dataset_id == dataset_id,
            DriveSource.root_id == root_id,
        )
    )


def list_drive_sources(session: Session, dataset_id: str) -> list[DriveSource]:
    return list(
        session.scalars(
            select(DriveSource)
            .where(DriveSource.dataset_id == dataset_id)
            .order_by(DriveSource.created_at.desc())
        )
    )


def delete_drive_source(session: Session, source: DriveSource) -> None:
    session.delete(source)


def get_drive_file_snapshot(
    session: Session, drive_source_id: str, drive_file_id: str
) -> Optional[DriveFileSnapshot]:
    return session.scalar(
        select(DriveFileSnapshot).where(
            DriveFileSnapshot.drive_source_id == drive_source_id,
            DriveFileSnapshot.drive_file_id == drive_file_id,
        )
    )


def list_drive_file_snapshots(
    session: Session, drive_source_id: str
) -> list[DriveFileSnapshot]:
    return list(
        session.scalars(
            select(DriveFileSnapshot)
            .where(DriveFileSnapshot.drive_source_id == drive_source_id)
            .order_by(DriveFileSnapshot.relative_path)
        )
    )


def upsert_drive_file_snapshot(
    session: Session,
    *,
    snapshot_id: str | None,
    drive_source_id: str,
    drive_file_id: str,
    relative_path: str,
    name: str,
    mime_type: str,
    md5_checksum: Optional[str],
    modified_time: datetime,
    size: Optional[int],
    document_id: Optional[str] = None,
    last_sync_status: Optional[str] = None,
) -> DriveFileSnapshot:
    now = datetime.now(timezone.utc)
    snap = None
    if snapshot_id:
        snap = session.get(DriveFileSnapshot, snapshot_id)
    if snap is None:
        snap = get_drive_file_snapshot(session, drive_source_id, drive_file_id)
    if snap is None:
        snap = DriveFileSnapshot(
            id=new_id(),
            drive_source_id=drive_source_id,
            drive_file_id=drive_file_id,
            relative_path=relative_path,
            name=name,
            mime_type=mime_type,
            md5_checksum=md5_checksum,
            modified_time=modified_time,
            size=size,
            document_id=document_id,
            last_seen_at=now,
            last_sync_status=last_sync_status,
        )
        session.add(snap)
    else:
        snap.relative_path = relative_path
        snap.name = name
        snap.mime_type = mime_type
        snap.md5_checksum = md5_checksum
        snap.modified_time = modified_time
        snap.size = size
        snap.last_seen_at = now
        if document_id is not None:
            snap.document_id = document_id
        if last_sync_status is not None:
            snap.last_sync_status = last_sync_status
    session.flush()
    return snap


def touch_drive_source_synced(session: Session, source: DriveSource) -> None:
    source.last_synced_at = datetime.now(timezone.utc)
    session.flush()
