from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_session
from ..deps import get_embedder, get_store
from ..embedder import Embedder
from ..llm import (
    build_entity_synthesis_prompt,
    build_user_prompt,
    extract_entities,
    get_client,
    get_config,
    resolve_model,
)
from ..retrieval import lookup_dataset_or_raise, retrieve
from ..schemas import (
    AskRequest,
    AskResponse,
    EntityAskRequest,
    EntityAskResponse,
    EntitySearch,
)
from ..store import VectorStore

router = APIRouter(prefix="/api/ask", tags=["ask"])

SessionDep = Annotated[Session, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
StoreDep = Annotated[VectorStore, Depends(get_store)]


@router.post("", response_model=AskResponse)
def ask(
    req: AskRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    store: StoreDep,
) -> AskResponse:
    dataset = lookup_dataset_or_raise(session, req.dataset_id)
    search_result = retrieve(session, embedder, store, dataset, req.query, req.top_k)

    config = get_config()

    if not search_result.hits:
        return AskResponse(
            query=req.query,
            dataset_id=req.dataset_id,
            answer=(
                "I couldn't find any passages in this dataset to ground an answer. "
                "Upload relevant documents first, or try a different query."
            ),
            model=config.model or "(unresolved)",
            provider=config.provider,
            search=search_result,
        )

    passages = [
        {
            "filename": h.reference.filename,
            "chunk_index": h.reference.chunk_index,
            "text": h.text,
        }
        for h in search_result.hits
    ]
    user_prompt = build_user_prompt(req.query, passages, config.max_context_chars)
    temperature = req.temperature if req.temperature is not None else config.temperature

    try:
        model = resolve_model(config)
        answer = get_client().chat(
            model=model,
            system=config.system_prompt,
            user=user_prompt,
            temperature=temperature,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed ({config.provider} @ {config.base_url}): {exc}",
        )

    return AskResponse(
        query=req.query,
        dataset_id=req.dataset_id,
        answer=answer,
        model=model,
        provider=config.provider,
        search=search_result,
    )


@router.post("/entities", response_model=EntityAskResponse)
def ask_entities(
    req: EntityAskRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    store: StoreDep,
) -> EntityAskResponse:
    """Agentic RAG: let the LLM plan entity searches, retrieve per entity, then decide.

    1. Ask the LLM which entities/subjects to search for.
    2. Run a separate vector search for each (top_k = per_entity_top_k).
    3. Feed all grouped passages back to the LLM for a final decision.
    """
    dataset = lookup_dataset_or_raise(session, req.dataset_id)
    config = get_config()
    temperature = req.temperature if req.temperature is not None else config.temperature

    try:
        model = resolve_model(config)
        entities = extract_entities(
            req.query,
            model=model,
            temperature=temperature,
            max_entities=req.max_entities,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"LLM entity planning failed ({config.provider} @ {config.base_url}): {exc}"
            ),
        )

    entity_searches: list[EntitySearch] = []
    groups: list[dict] = []
    for entity in entities:
        result = retrieve(session, embedder, store, dataset, entity, req.per_entity_top_k)
        entity_searches.append(EntitySearch(entity=entity, search=result))
        groups.append(
            {
                "entity": entity,
                "passages": [
                    {
                        "filename": h.reference.filename,
                        "chunk_index": h.reference.chunk_index,
                        "text": h.text,
                    }
                    for h in result.hits
                ],
            }
        )

    if not any(g["passages"] for g in groups):
        return EntityAskResponse(
            query=req.query,
            dataset_id=req.dataset_id,
            answer=(
                "I couldn't find any passages in this dataset for the entities I "
                f"identified ({', '.join(entities)}). Upload relevant documents first, "
                "or try a different question."
            ),
            model=model,
            provider=config.provider,
            entities=entity_searches,
        )

    user_prompt = build_entity_synthesis_prompt(
        req.query, groups, config.max_context_chars
    )

    try:
        answer = get_client().chat(
            model=model,
            system=config.system_prompt,
            user=user_prompt,
            temperature=temperature,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"LLM synthesis failed ({config.provider} @ {config.base_url}): {exc}"
            ),
        )

    return EntityAskResponse(
        query=req.query,
        dataset_id=req.dataset_id,
        answer=answer,
        model=model,
        provider=config.provider,
        entities=entity_searches,
    )
