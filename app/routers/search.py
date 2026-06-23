from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import repository
from ..confidence import label_for, overall_verdict, score_to_confidence
from ..config import settings
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

    hits: list[SearchHit] = []
    for h in raw_hits:
        confidence = score_to_confidence(h["score"], settings.confidence_full_score)
        confidence_label = label_for(
            confidence,
            settings.confidence_high_threshold,
            settings.confidence_medium_threshold,
        )
        hits.append(
            SearchHit(
                text=h["text"],
                score=h["score"],
                confidence=confidence,
                confidence_label=confidence_label,
                reference=SearchReference(
                    dataset_id=req.dataset_id,
                    dataset_name=dataset.name,
                    document_id=h["metadata"]["doc_id"],
                    filename=h["metadata"]["filename"],
                    chunk_index=h["metadata"]["chunk_index"],
                ),
            )
        )

    top_label = hits[0].confidence_label if hits else None
    overall_label, overall_message = overall_verdict(top_label)
    overall_score = hits[0].confidence if hits else 0.0

    return SearchResponse(
        query=req.query,
        dataset_id=req.dataset_id,
        hits=hits,
        overall_confidence=overall_label,
        overall_confidence_score=overall_score,
        overall_message=overall_message,
    )
