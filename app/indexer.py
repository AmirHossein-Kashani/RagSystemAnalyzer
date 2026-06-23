from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from . import repository
from .chunker import chunk_text
from .config import settings
from .embedder import Embedder
from .loader import SUPPORTED_EXTENSIONS, load_text
from .schemas import IndexResult
from .store import VectorStore


class Indexer:
    """Orchestrates file -> text -> chunks -> embeddings -> Chroma + SQL row."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def index_file(
        self,
        session: Session,
        dataset_id: str,
        file_path: Path,
        original_filename: str,
    ) -> IndexResult:
        file_path = file_path.resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Not a file: {file_path}")
        suffix = Path(original_filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        doc_id = _doc_id_for(dataset_id, original_filename)
        file_hash = _hash_file(file_path)

        existing = repository.get_document(session, doc_id)
        if existing is not None and existing.file_hash == file_hash:
            return IndexResult(
                doc_id=doc_id, dataset_id=dataset_id,
                filename=original_filename, chunks_added=0, status="skipped",
            )

        text = load_text(file_path)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            return IndexResult(
                doc_id=doc_id, dataset_id=dataset_id,
                filename=original_filename, chunks_added=0, status="skipped",
            )

        embeddings = self._embedder.embed(chunks)
        indexed_at = datetime.now(timezone.utc).isoformat()
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "dataset_id": dataset_id,
                "doc_id": doc_id,
                "filename": original_filename,
                "source": str(file_path),
                "chunk_index": i,
                "file_hash": file_hash,
                "indexed_at": indexed_at,
            }
            for i in range(len(chunks))
        ]

        was_present = existing is not None
        if was_present:
            self._store.delete_by_doc_id(doc_id)

        self._store.upsert(
            ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )

        repository.upsert_document(
            session,
            doc_id=doc_id,
            dataset_id=dataset_id,
            filename=original_filename,
            source_path=str(file_path),
            file_hash=file_hash,
            chunk_count=len(chunks),
        )

        return IndexResult(
            doc_id=doc_id,
            dataset_id=dataset_id,
            filename=original_filename,
            chunks_added=len(chunks),
            status="updated" if was_present else "indexed",
        )

    def delete_document(self, session: Session, doc_id: str) -> int:
        document = repository.get_document(session, doc_id)
        if document is None:
            return 0
        removed = self._store.delete_by_doc_id(doc_id)
        repository.delete_document(session, document)
        return removed

    def delete_dataset_chunks(self, dataset_id: str) -> int:
        return self._store.delete_by_dataset_id(dataset_id)


def _doc_id_for(dataset_id: str, filename: str) -> str:
    return hashlib.sha1(f"{dataset_id}:{filename}".encode("utf-8")).hexdigest()[:16]


def _hash_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()
