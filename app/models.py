from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(1000), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    documents: Mapped[list["Document"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("dataset_id", "filename", name="uq_document_dataset_filename"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(500))
    source_path: Mapped[str] = mapped_column(String(1000))
    file_hash: Mapped[str] = mapped_column(String(64))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    dataset: Mapped["Dataset"] = relationship(back_populates="documents")


class DriveSource(Base):
    __tablename__ = "drive_sources"
    __table_args__ = (
        UniqueConstraint("dataset_id", "root_id", name="uq_drive_source_dataset_root"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    folder_url: Mapped[str] = mapped_column(String(1000))
    root_id: Mapped[str] = mapped_column(String(128))
    root_name: Mapped[str] = mapped_column(String(500))
    is_single_file: Mapped[bool] = mapped_column(Boolean, default=False)
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    files: Mapped[list["DriveFileSnapshot"]] = relationship(
        back_populates="drive_source",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DriveFileSnapshot(Base):
    __tablename__ = "drive_files"
    __table_args__ = (
        UniqueConstraint(
            "drive_source_id", "drive_file_id", name="uq_drive_file_source_id"
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    drive_source_id: Mapped[str] = mapped_column(
        ForeignKey("drive_sources.id", ondelete="CASCADE"), index=True
    )
    drive_file_id: Mapped[str] = mapped_column(String(128))
    relative_path: Mapped[str] = mapped_column(String(500))
    name: Mapped[str] = mapped_column(String(500))
    mime_type: Mapped[str] = mapped_column(String(200))
    md5_checksum: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    modified_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    size: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    document_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), default=None
    )
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_sync_status: Mapped[Optional[str]] = mapped_column(String(32), default=None)

    drive_source: Mapped["DriveSource"] = relationship(back_populates="files")
