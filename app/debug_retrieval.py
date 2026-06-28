from __future__ import annotations

import numpy as np

from .confidence import label_for, overall_verdict, score_to_confidence
from .config import settings
from .embedder import Embedder
from .models import Dataset
from .schemas import (
    DebugHit,
    DebugProjection,
    DebugRetrieveResponse,
    SearchReference,
)
from .store import VectorStore


def _pca_2d(vectors: list[list[float]]) -> list[tuple[float, float]]:
    """Project vectors to 2D with PCA (SVD). Returns one (x, y) per input vector."""
    if not vectors:
        return []
    if len(vectors) == 1:
        return [(0.0, 0.0)]

    matrix = np.array(vectors, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    return [(float(x), float(y)) for x, y in coords]


def _cosine_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Pairwise cosine similarity for L2-normalized vectors (dot product)."""
    if not vectors:
        return []
    matrix = np.array(vectors, dtype=np.float64)
    sim = matrix @ matrix.T
    return [[float(v) for v in row] for row in sim]


def retrieve_debug(
    embedder: Embedder,
    store: VectorStore,
    dataset: Dataset,
    query: str,
    top_k: int,
) -> DebugRetrieveResponse:
    """Retrieve chunks with full embedding vectors and 2D projections for debug UI."""
    [query_vector] = embedder.embed([query])
    raw_hits = store.query(
        query_vector,
        top_k=top_k,
        dataset_id=dataset.id,
        include_embeddings=True,
    )

    all_vectors = [query_vector] + [h["embedding"] for h in raw_hits]
    projections = _pca_2d(all_vectors)
    query_projection = DebugProjection(x=projections[0][0], y=projections[0][1])

    hits: list[DebugHit] = []
    for i, h in enumerate(raw_hits):
        confidence = score_to_confidence(h["score"], settings.confidence_full_score)
        confidence_label = label_for(
            confidence,
            settings.confidence_high_threshold,
            settings.confidence_medium_threshold,
        )
        meta = h["metadata"]
        px, py = projections[i + 1]
        hits.append(
            DebugHit(
                id=h["id"],
                text=h["text"],
                score=h["score"],
                confidence=confidence,
                confidence_label=confidence_label,
                embedding=h["embedding"],
                reference=SearchReference(
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                    document_id=meta["doc_id"],
                    filename=meta["filename"],
                    chunk_index=meta["chunk_index"],
                ),
                metadata=meta,
                projection=DebugProjection(x=px, y=py),
            )
        )

    top_label = hits[0].confidence_label if hits else None
    overall_label, overall_message = overall_verdict(top_label)
    overall_score = hits[0].confidence if hits else 0.0

    labels = ["Query"] + [
        f"#{i + 1} {h.reference.filename[:24]}" for i, h in enumerate(hits)
    ]
    sim_matrix = _cosine_similarity_matrix(all_vectors)

    return DebugRetrieveResponse(
        query=query,
        dataset_id=dataset.id,
        embedding_model=settings.embedding_model,
        dimension=embedder.dimension,
        query_embedding=query_vector,
        query_projection=query_projection,
        hits=hits,
        overall_confidence=overall_label,
        overall_confidence_score=overall_score,
        overall_message=overall_message,
        similarity_matrix=sim_matrix,
        similarity_labels=labels,
    )
