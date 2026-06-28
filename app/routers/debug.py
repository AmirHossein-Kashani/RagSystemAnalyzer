from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_session
from ..debug_retrieval import retrieve_debug
from ..deps import get_embedder, get_store
from ..embedder import Embedder
from ..retrieval import lookup_dataset_or_raise
from ..schemas import DebugRetrieveRequest, DebugRetrieveResponse
from ..store import VectorStore

router = APIRouter(prefix="/api/debug", tags=["debug"])

SessionDep = Annotated[Session, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
StoreDep = Annotated[VectorStore, Depends(get_store)]


@router.post("/retrieve", response_model=DebugRetrieveResponse)
def debug_retrieve(
    req: DebugRetrieveRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    store: StoreDep,
) -> DebugRetrieveResponse:
    dataset = lookup_dataset_or_raise(session, req.dataset_id)
    return retrieve_debug(embedder, store, dataset, req.query, req.top_k)
