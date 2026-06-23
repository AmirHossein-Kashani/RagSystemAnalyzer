from __future__ import annotations

from functools import cached_property


class Embedder:
    """Wraps a SentenceTransformer model. CPU-friendly; vectors are L2-normalized."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, device="cpu")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())
