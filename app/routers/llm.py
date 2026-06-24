from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..llm import build_user_prompt, get_client, get_config, resolve_model
from ..schemas import (
    LLMAnswerRequest,
    LLMAnswerResponse,
    LLMChatRequest,
    LLMInfo,
)

router = APIRouter(prefix="/api/llm", tags=["llm"])


@router.get("/info", response_model=LLMInfo)
def info() -> LLMInfo:
    """Inspect the currently configured LLM (resolves Ollama auto-discovery)."""
    config = get_config()
    try:
        model = resolve_model(config)
    except Exception:
        model = config.model or ""
    return LLMInfo(
        provider=config.provider,
        base_url=config.base_url,
        model=model,
        default_temperature=config.temperature,
        default_system_prompt=config.system_prompt,
        max_context_chars=config.max_context_chars,
    )


@router.post("/answer", response_model=LLMAnswerResponse)
def answer(req: LLMAnswerRequest) -> LLMAnswerResponse:
    """Ask the configured LLM using caller-supplied passages as context.

    Passages are inlined in the prompt in the order received and tagged as [1], [2], ...
    Use this when you've already retrieved (or hand-curated) the context and don't want
    the server to run another vector search.
    """
    config = get_config()

    system = req.system_prompt or config.system_prompt
    temperature = (
        req.temperature if req.temperature is not None else config.temperature
    )
    user_prompt = build_user_prompt(
        req.query,
        [p.model_dump() for p in req.passages],
        config.max_context_chars,
    )

    try:
        model = req.model or resolve_model(config)
        answer_text = get_client().chat(
            model=model,
            system=system,
            user=user_prompt,
            temperature=temperature,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed ({config.provider} @ {config.base_url}): {exc}",
        )

    return LLMAnswerResponse(
        query=req.query,
        answer=answer_text,
        model=model,
        provider=config.provider,
    )


@router.post("/chat", response_model=LLMAnswerResponse)
def chat(req: LLMChatRequest) -> LLMAnswerResponse:
    """Direct chat with the configured LLM — no retrieval, no context.

    Use this for plain LLM access (general questions, testing connectivity,
    queries that don't need your documents).
    """
    config = get_config()

    system = req.system_prompt or config.system_prompt
    temperature = (
        req.temperature if req.temperature is not None else config.temperature
    )

    try:
        model = req.model or resolve_model(config)
        answer_text = get_client().chat(
            model=model,
            system=system,
            user=req.query,
            temperature=temperature,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM call failed ({config.provider} @ {config.base_url}): {exc}",
        )

    return LLMAnswerResponse(
        query=req.query,
        answer=answer_text,
        model=model,
        provider=config.provider,
    )
