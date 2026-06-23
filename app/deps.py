from __future__ import annotations

from functools import lru_cache

from .config import settings
from .embedder import Embedder
from .indexer import Indexer
from .store import VectorStore


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(settings.embedding_model)


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore(settings.chroma_dir, settings.collection_name)


@lru_cache(maxsize=1)
def get_indexer() -> Indexer:
    return Indexer(get_embedder(), get_store())
