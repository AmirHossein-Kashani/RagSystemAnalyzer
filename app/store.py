from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    """Thin wrapper over a persistent Chroma collection (cosine space).

    Every chunk's metadata carries `dataset_id` so queries can be scoped to a
    single dataset via a server-side `where` filter.
    """

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_by_doc_id(self, doc_id: str) -> int:
        existing = self._collection.get(where={"doc_id": doc_id}, include=[])
        count = len(existing.get("ids", []) or [])
        if count:
            self._collection.delete(where={"doc_id": doc_id})
        return count

    def delete_by_dataset_id(self, dataset_id: str) -> int:
        existing = self._collection.get(where={"dataset_id": dataset_id}, include=[])
        count = len(existing.get("ids", []) or [])
        if count:
            self._collection.delete(where={"dataset_id": dataset_id})
        return count

    def query(
        self,
        embedding: list[float],
        top_k: int,
        dataset_id: str,
        *,
        include_embeddings: bool = False,
    ) -> list[dict]:
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"dataset_id": dataset_id},
            include=include,
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        embs = (res.get("embeddings") or [[]])[0] if include_embeddings else []

        hits: list[dict] = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            # Cosine distance in [0, 2]; convert to similarity in [-1, 1].
            hit: dict = {
                "id": ids[i] if i < len(ids) else f"{meta.get('doc_id')}:{meta.get('chunk_index')}",
                "text": doc,
                "metadata": meta,
                "score": 1.0 - float(dist),
            }
            if include_embeddings and i < len(embs):
                hit["embedding"] = embs[i]
            hits.append(hit)
        return hits
