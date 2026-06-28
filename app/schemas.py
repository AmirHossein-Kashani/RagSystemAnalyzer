from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

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


# ---------- Ask (RAG) ----------

class AskRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class AskResponse(BaseModel):
    query: str
    dataset_id: str
    answer: str
    model: str
    provider: str
    search: SearchResponse  # the retrieval used as context (with references + confidence)


# ---------- LLM (direct, caller-supplied context) ----------

class LLMPassage(BaseModel):
    filename: str
    chunk_index: int = 0
    text: str = Field(..., min_length=1)


class LLMAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    passages: list[LLMPassage] = Field(..., min_length=1)
    model: Optional[str] = None         # override llm_config.json
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    system_prompt: Optional[str] = None  # override llm_config.json


class LLMChatRequest(BaseModel):
    """Direct chat with the LLM — no retrieval, no supplied context."""
    query: str = Field(..., min_length=1)
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    system_prompt: Optional[str] = None


class LLMAnswerResponse(BaseModel):
    query: str
    answer: str
    model: str
    provider: str


class LLMInfo(BaseModel):
    provider: str
    base_url: str
    model: str  # resolved (auto-discovered for Ollama if not pinned)
    default_temperature: float
    default_system_prompt: str
    max_context_chars: int


# ---------- Google Drive ----------

class DriveSourceCreate(BaseModel):
    url: str = Field(..., min_length=10, max_length=1000)


class DriveSourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    folder_url: str
    root_id: str
    root_name: str
    is_single_file: bool
    last_synced_at: Optional[datetime]
    created_at: datetime
    file_count: int = 0


class DriveFileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    drive_source_id: str
    drive_file_id: str
    relative_path: str
    name: str
    mime_type: str
    md5_checksum: Optional[str]
    modified_time: datetime
    size: Optional[int]
    document_id: Optional[str]
    last_seen_at: datetime
    last_sync_status: Optional[str]


class DriveSyncFileResultOut(BaseModel):
    relative_path: str
    drive_file_id: str
    status: str
    chunks_added: int = 0
    message: str = ""


class DriveSyncResultOut(BaseModel):
    listed: int
    skipped: int
    indexed: int
    updated: int
    unsupported: int
    errors: int
    files: list[DriveSyncFileResultOut]


# ---------- Mapping plans ----------

class MappingPlanCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=1000)
    system_prompt: str = Field("", max_length=200000)
    output_schema: Optional[dict] = None
    prompt_template: str = Field(
        "Input query:\n{query}\n\nRetrieved context:\n{context}\n",
        max_length=50000,
    )
    default_top_k: int = Field(5, ge=1, le=50)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    dataset_ids: list[str] = Field(default_factory=list)


class MappingPlanUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=1000)
    system_prompt: Optional[str] = Field(None, max_length=200000)
    output_schema: Optional[dict] = None
    prompt_template: Optional[str] = Field(None, max_length=50000)
    default_top_k: Optional[int] = Field(None, ge=1, le=50)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    dataset_ids: Optional[list[str]] = None


class MappingPlanOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    system_prompt: str
    output_schema: Optional[dict]
    prompt_template: str
    default_top_k: int
    temperature: Optional[float]
    dataset_ids: list[str]
    created_at: datetime
    updated_at: datetime


class MappingRunRequest(BaseModel):
    query: str = Field(..., min_length=1)
    dataset_ids: Optional[list[str]] = None
    top_k: Optional[int] = Field(None, ge=1, le=50)
    variables: Optional[dict] = None


class MappingRunResponse(BaseModel):
    plan_id: str
    output: Any
    valid: bool
    validation_errors: list[str]
    repaired: bool
    model: str
    provider: str
    search: SearchResponse


# ---------- Prompt presets ----------

class PromptPresetCreate(BaseModel):
    key: str = Field(..., min_length=1, max_length=80)
    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=1000)
    category: Optional[str] = Field(None, max_length=60)
    system_prompt: str = Field("", max_length=200000)
    output_schema: Optional[dict] = None
    prompt_template: str = Field("", max_length=50000)
    default_top_k: Optional[int] = Field(None, ge=1, le=50)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class PromptPresetOut(BaseModel):
    id: str
    key: str
    name: str
    description: Optional[str]
    category: Optional[str]
    system_prompt: str
    output_schema: Optional[dict]
    prompt_template: str
    default_top_k: Optional[int]
    temperature: Optional[float]
    is_builtin: bool
    created_at: datetime
    updated_at: datetime


class CreatePlanFromPresetRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    dataset_ids: list[str] = Field(default_factory=list)
