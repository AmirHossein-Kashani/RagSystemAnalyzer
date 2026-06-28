from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
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


class MappingPlan(Base):
    __tablename__ = "mapping_plans"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(1000), default=None)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    output_schema: Mapped[Optional[str]] = mapped_column(Text, default=None)
    prompt_template: Mapped[str] = mapped_column(Text, default="")
    # Optional stage-1 "entity layer": when entity_prompt is set, a first LLM
    # call extracts structured entities that get injected as {entities}.
    entity_prompt: Mapped[Optional[str]] = mapped_column(Text, default=None)
    entity_schema: Mapped[Optional[str]] = mapped_column(Text, default=None)
    default_top_k: Mapped[int] = mapped_column(Integer, default=5)
    temperature: Mapped[Optional[float]] = mapped_column(Float, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    datasets: Mapped[list["MappingPlanDataset"]] = relationship(
        back_populates="mapping_plan",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class MappingPlanDataset(Base):
    __tablename__ = "mapping_plan_datasets"
    __table_args__ = (
        UniqueConstraint(
            "mapping_plan_id", "dataset_id", name="uq_mapping_plan_dataset"
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    mapping_plan_id: Mapped[str] = mapped_column(
        ForeignKey("mapping_plans.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )

    mapping_plan: Mapped["MappingPlan"] = relationship(back_populates="datasets")


class PromptPreset(Base):
    """A reusable, pre-designed system prompt + output schema + template that can
    be loaded into a mapping plan. Built-in presets (e.g. LIO) ship with the app;
    users may also save their own.
    """

    __tablename__ = "prompt_presets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    key: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[Optional[str]] = mapped_column(String(1000), default=None)
    category: Mapped[Optional[str]] = mapped_column(String(60), default=None)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    output_schema: Mapped[Optional[str]] = mapped_column(Text, default=None)
    prompt_template: Mapped[str] = mapped_column(Text, default="")
    entity_prompt: Mapped[Optional[str]] = mapped_column(Text, default=None)
    entity_schema: Mapped[Optional[str]] = mapped_column(Text, default=None)
    default_top_k: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    temperature: Mapped[Optional[float]] = mapped_column(Float, default=None)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
