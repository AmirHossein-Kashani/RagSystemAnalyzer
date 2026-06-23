from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .. import repository
from ..config import settings
from ..db import get_session
from ..deps import get_indexer
from ..indexer import Indexer
from ..schemas import DatasetCreate, DatasetOut, DocumentOut, IndexResult

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

SessionDep = Annotated[Session, Depends(get_session)]
IndexerDep = Annotated[Indexer, Depends(get_indexer)]


def _dataset_to_out(dataset) -> DatasetOut:
    return DatasetOut(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        created_at=dataset.created_at,
        document_count=len(dataset.documents),
    )


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate, session: SessionDep) -> DatasetOut:
    if repository.get_dataset_by_name(session, payload.name):
        raise HTTPException(status_code=409, detail="dataset name already exists")
    dataset = repository.create_dataset(session, payload.name, payload.description)
    return _dataset_to_out(dataset)


@router.get("", response_model=list[DatasetOut])
def list_datasets(session: SessionDep) -> list[DatasetOut]:
    return [_dataset_to_out(d) for d in repository.list_datasets(session)]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, session: SessionDep) -> DatasetOut:
    dataset = repository.get_dataset(session, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return _dataset_to_out(dataset)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(
    dataset_id: str, session: SessionDep, indexer: IndexerDep
) -> None:
    dataset = repository.get_dataset(session, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")

    indexer.delete_dataset_chunks(dataset_id)
    repository.delete_dataset(session, dataset)

    dataset_dir = settings.docs_dir / dataset_id
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir, ignore_errors=True)


# ---------- Documents within a dataset ----------

@router.get("/{dataset_id}/documents", response_model=list[DocumentOut])
def list_dataset_documents(dataset_id: str, session: SessionDep) -> list[DocumentOut]:
    dataset = repository.get_dataset(session, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return [DocumentOut.model_validate(d) for d in repository.list_documents(session, dataset_id)]


@router.post(
    "/{dataset_id}/documents/upload",
    response_model=IndexResult,
    status_code=201,
)
def upload_document(
    dataset_id: str,
    session: SessionDep,
    indexer: IndexerDep,
    file: UploadFile = File(...),
) -> IndexResult:
    dataset = repository.get_dataset(session, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename missing")

    dataset_dir = settings.docs_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / Path(file.filename).name
    target.write_bytes(file.file.read())

    try:
        return indexer.index_file(session, dataset_id, target, target.name)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{dataset_id}/documents/{doc_id}", status_code=204)
def delete_document(
    dataset_id: str, doc_id: str, session: SessionDep, indexer: IndexerDep
) -> None:
    document = repository.get_document(session, doc_id)
    if document is None or document.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="document not found")

    source = Path(document.source_path)
    indexer.delete_document(session, doc_id)
    if source.exists():
        try:
            source.unlink()
        except OSError:
            pass
