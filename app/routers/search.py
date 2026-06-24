from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_session
from ..deps import get_embedder, get_store
from ..embedder import Embedder
from ..retrieval import lookup_dataset_or_raise, retrieve
from ..schemas import SearchRequest, SearchResponse
from ..store import VectorStore

router = APIRouter(prefix="/api/search", tags=["search"])

SessionDep = Annotated[Session, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
StoreDep = Annotated[VectorStore, Depends(get_store)]


@router.post("", response_model=SearchResponse)
def search(
    req: SearchRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    store: StoreDep,
) -> SearchResponse:
    dataset = lookup_dataset_or_raise(session, req.dataset_id)
    return retrieve(session, embedder, store, dataset, req.query, req.top_k)
