from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- Datasets ----------

class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=1000)


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    document_count: int = 0


# ---------- Documents ----------

class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    filename: str
    file_hash: str
    chunk_count: int
    indexed_at: datetime


class IndexResult(BaseModel):
    doc_id: str
    dataset_id: str
    filename: str
    chunks_added: int
    status: str  # "indexed" | "updated" | "skipped" | "error: ..."


# ---------- Search ----------

class SearchRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class SearchReference(BaseModel):
    """Where this chunk came from — used by the UI to render citations."""
    dataset_id: str
    dataset_name: str
    document_id: str
    filename: str
    chunk_index: int


class SearchHit(BaseModel):
    text: str
    score: float           # raw cosine similarity, kept for transparency
    confidence: float      # calibrated 0..1
    confidence_label: str  # "high" | "medium" | "low"
    reference: SearchReference


class SearchResponse(BaseModel):
    query: str
    dataset_id: str
    hits: list[SearchHit]
    overall_confidence: str         # "high" | "medium" | "low" | "none"
    overall_confidence_score: float  # top hit's calibrated confidence, 0 if no hits
    overall_message: str
