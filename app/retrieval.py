from __future__ import annotations

from sqlalchemy.orm import Session

from . import repository
from .confidence import label_for, overall_verdict, score_to_confidence
from .config import settings
from .embedder import Embedder
from .models import Dataset
from .schemas import SearchHit, SearchReference, SearchResponse
from .store import VectorStore


def retrieve(
    session: Session,
    embedder: Embedder,
    store: VectorStore,
    dataset: Dataset,
    query: str,
    top_k: int,
) -> SearchResponse:
    """Embed the query, fetch top-k chunks from the dataset, attach confidence,
    and return a SearchResponse. Shared by /api/search and /api/ask.
    """
    [vector] = embedder.embed([query])
    raw_hits = store.query(vector, top_k=top_k, dataset_id=dataset.id)

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
                    dataset_id=dataset.id,
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
        query=query,
        dataset_id=dataset.id,
        hits=hits,
        overall_confidence=overall_label,
        overall_confidence_score=overall_score,
        overall_message=overall_message,
    )


def retrieve_multi(
    session: Session,
    embedder: Embedder,
    store: VectorStore,
    datasets: list[Dataset],
    query: str,
    top_k: int,
) -> SearchResponse:
    """Embed the query once, fetch top-k chunks across multiple datasets, attach
    confidence, merge by score, and return a SearchResponse whose hits carry their
    own per-dataset references.
    """
    name_by_id = {d.id: d.name for d in datasets}
    dataset_ids = list(name_by_id.keys())

    [vector] = embedder.embed([query])
    raw_hits = store.query_multi(vector, top_k=top_k, dataset_ids=dataset_ids)

    hits: list[SearchHit] = []
    for h in raw_hits:
        confidence = score_to_confidence(h["score"], settings.confidence_full_score)
        confidence_label = label_for(
            confidence,
            settings.confidence_high_threshold,
            settings.confidence_medium_threshold,
        )
        hit_dataset_id = h["metadata"]["dataset_id"]
        hits.append(
            SearchHit(
                text=h["text"],
                score=h["score"],
                confidence=confidence,
                confidence_label=confidence_label,
                reference=SearchReference(
                    dataset_id=hit_dataset_id,
                    dataset_name=name_by_id.get(hit_dataset_id, hit_dataset_id),
                    document_id=h["metadata"]["doc_id"],
                    filename=h["metadata"]["filename"],
                    chunk_index=h["metadata"]["chunk_index"],
                ),
            )
        )

    hits.sort(key=lambda x: x.score, reverse=True)
    hits = hits[:top_k]

    top_label = hits[0].confidence_label if hits else None
    overall_label, overall_message = overall_verdict(top_label)
    overall_score = hits[0].confidence if hits else 0.0

    return SearchResponse(
        query=query,
        dataset_id=",".join(dataset_ids),
        hits=hits,
        overall_confidence=overall_label,
        overall_confidence_score=overall_score,
        overall_message=overall_message,
    )


def lookup_dataset_or_raise(session: Session, dataset_id: str) -> Dataset:
    dataset = repository.get_dataset(session, dataset_id)
    if dataset is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="dataset not found")
    return dataset


def lookup_datasets_or_raise(session: Session, dataset_ids: list[str]) -> list[Dataset]:
    from fastapi import HTTPException

    if not dataset_ids:
        raise HTTPException(status_code=400, detail="no datasets selected")

    seen: set[str] = set()
    datasets: list[Dataset] = []
    for dataset_id in dataset_ids:
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        dataset = repository.get_dataset(session, dataset_id)
        if dataset is None:
            raise HTTPException(
                status_code=404, detail=f"dataset not found: {dataset_id}"
            )
        datasets.append(dataset)
    return datasets
