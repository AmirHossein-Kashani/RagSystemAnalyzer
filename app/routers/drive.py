from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import repository
from ..config import settings
from ..db import get_session
from ..deps import get_indexer
from ..drive import DriveSyncService, PublicDriveClient, parse_drive_url
from ..indexer import Indexer
from ..schemas import (
    DriveFileOut,
    DriveSourceCreate,
    DriveSourceOut,
    DriveSyncFileResultOut,
    DriveSyncResultOut,
)

router = APIRouter(prefix="/api/datasets", tags=["drive"])

SessionDep = Annotated[Session, Depends(get_session)]
IndexerDep = Annotated[Indexer, Depends(get_indexer)]


def _require_api_key() -> None:
    if not settings.google_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Google Drive sync is not configured. Set RAG_GOOGLE_API_KEY in .env "
                "and enable the Drive API in Google Cloud Console."
            ),
        )


def _source_to_out(source) -> DriveSourceOut:
    return DriveSourceOut(
        id=source.id,
        dataset_id=source.dataset_id,
        folder_url=source.folder_url,
        root_id=source.root_id,
        root_name=source.root_name,
        is_single_file=source.is_single_file,
        last_synced_at=source.last_synced_at,
        created_at=source.created_at,
        file_count=len(source.files),
    )


def _get_source_or_404(session: Session, dataset_id: str, source_id: str):
    dataset = repository.get_dataset(session, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    source = repository.get_drive_source(session, source_id)
    if source is None or source.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="drive source not found")
    return source


@router.post(
    "/{dataset_id}/drive/sources",
    response_model=DriveSourceOut,
    status_code=201,
)
def link_drive_source(
    dataset_id: str,
    payload: DriveSourceCreate,
    session: SessionDep,
) -> DriveSourceOut:
    _require_api_key()
    dataset = repository.get_dataset(session, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")

    try:
        parsed = parse_drive_url(payload.url)
        client = PublicDriveClient()
        root_meta = client.get_file_metadata(parsed.root_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing = repository.get_drive_source_by_root(session, dataset_id, parsed.root_id)
    if existing is not None:
        raise HTTPException(status_code=409, detail="this Drive folder is already linked")

    source = repository.create_drive_source(
        session,
        dataset_id=dataset_id,
        folder_url=payload.url.strip(),
        root_id=parsed.root_id,
        root_name=root_meta.name,
        is_single_file=parsed.is_single_file and not root_meta.is_folder,
    )
    return _source_to_out(source)


@router.get("/{dataset_id}/drive/sources", response_model=list[DriveSourceOut])
def list_drive_sources(dataset_id: str, session: SessionDep) -> list[DriveSourceOut]:
    dataset = repository.get_dataset(session, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    sources = repository.list_drive_sources(session, dataset_id)
    return [_source_to_out(s) for s in sources]


@router.get(
    "/{dataset_id}/drive/sources/{source_id}/files",
    response_model=list[DriveFileOut],
)
def list_drive_files(
    dataset_id: str, source_id: str, session: SessionDep
) -> list[DriveFileOut]:
    source = _get_source_or_404(session, dataset_id, source_id)
    return [
        DriveFileOut.model_validate(f)
        for f in repository.list_drive_file_snapshots(session, source.id)
    ]


@router.post(
    "/{dataset_id}/drive/sources/{source_id}/sync",
    response_model=DriveSyncResultOut,
)
def sync_drive_source(
    dataset_id: str,
    source_id: str,
    session: SessionDep,
    indexer: IndexerDep,
) -> DriveSyncResultOut:
    _require_api_key()
    source = _get_source_or_404(session, dataset_id, source_id)

    try:
        result = DriveSyncService().sync_source(session, indexer, source)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return DriveSyncResultOut(
        listed=result.listed,
        skipped=result.skipped,
        indexed=result.indexed,
        updated=result.updated,
        unsupported=result.unsupported,
        errors=result.errors,
        files=[
            DriveSyncFileResultOut(
                relative_path=f.relative_path,
                drive_file_id=f.drive_file_id,
                status=f.status,
                chunks_added=f.chunks_added,
                message=f.message,
            )
            for f in result.files
        ],
    )


@router.delete("/{dataset_id}/drive/sources/{source_id}", status_code=204, response_model=None)
def delete_drive_source(
    dataset_id: str, source_id: str, session: SessionDep
) -> None:
    source = _get_source_or_404(session, dataset_id, source_id)
    repository.delete_drive_source(session, source)
