from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Dataset, Document


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
