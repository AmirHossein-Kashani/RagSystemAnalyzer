from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import repository
from ..db import get_session
from ..deps import get_embedder, get_store
from ..embedder import Embedder
from ..schemas import SearchHit, SearchReference, SearchRequest, SearchResponse
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
    dataset = repository.get_dataset(session, req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")

    [vector] = embedder.embed([req.query])
    raw_hits = store.query(vector, top_k=req.top_k, dataset_id=req.dataset_id)

    hits = [
        SearchHit(
            text=h["text"],
            score=h["score"],
            reference=SearchReference(
                dataset_id=req.dataset_id,
                dataset_name=dataset.name,
                document_id=h["metadata"]["doc_id"],
                filename=h["metadata"]["filename"],
                chunk_index=h["metadata"]["chunk_index"],
            ),
        )
        for h in raw_hits
    ]
    return SearchResponse(query=req.query, dataset_id=req.dataset_id, hits=hits)
