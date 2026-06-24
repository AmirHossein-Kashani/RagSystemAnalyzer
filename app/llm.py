from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field


CONFIG_PATH = Path("llm_config.json")


class LLMConfig(BaseModel):
    provider: str = Field("ollama", pattern="^(ollama|openai)$")
    base_url: str
    api_key: str = "not-used"
    model: Optional[str] = None
    headers: dict[str, str] = Field(default_factory=dict)
    temperature: float = 0.2
    max_context_chars: int = 8000
    request_timeout_seconds: float = 120.0
    system_prompt: str = (
        "You are a concise, factual assistant. Answer the user's question using ONLY "
        "the supplied context passages. If the context does not contain the answer, "
        "say so explicitly instead of guessing. Cite sources as [1], [2], etc."
    )


def load_config(path: Path = CONFIG_PATH) -> LLMConfig:
    if not path.is_file():
        raise FileNotFoundError(
            f"LLM config not found at {path.resolve()}. "
            "Create it (see README) or copy from llm_config.json shipped with the project."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return LLMConfig(**raw)


@lru_cache(maxsize=1)
def get_config() -> LLMConfig:
    return load_config()


def discover_ollama_model(config: LLMConfig) -> str:
    """Hit Ollama's /api/tags to pick the first available model.

    The OpenAI-compatible endpoint lives at /v1; the discovery endpoint is /api/tags
    on the same host. We strip a trailing /v1 if present.
    """
    base = config.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/api/tags"
    with httpx.Client(timeout=15.0, headers=config.headers) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    models = [m.get("name") for m in data.get("models", []) if m.get("name")]
    if not models:
        raise RuntimeError(f"No models returned by Ollama at {url}")
    return models[0]


def resolve_model(config: LLMConfig) -> str:
    if config.model:
        return config.model
    if config.provider == "ollama":
        return discover_ollama_model(config)
    raise ValueError("`model` must be set in llm_config.json when provider is 'openai'")


class LLMClient:
    """Thin wrapper around the OpenAI SDK pointed at any OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "not-used",
            default_headers=config.headers or None,
            timeout=config.request_timeout_seconds,
        )

    def chat(self, model: str, system: str, user: str, temperature: float) -> str:
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        choice = resp.choices[0]
        return (choice.message.content or "").strip()


@lru_cache(maxsize=1)
def get_client() -> LLMClient:
    return LLMClient(get_config())


def build_user_prompt(query: str, passages: list[dict], max_chars: int) -> str:
    """Compose the user message: numbered, source-tagged passages + question.

    `passages` is a list of dicts with keys: filename, chunk_index, text.
    Truncates the concatenated context to `max_chars` to stay within model limits.
    """
    lines: list[str] = []
    used = 0
    for i, p in enumerate(passages, 1):
        header = f"[{i}] Source: {p['filename']} (chunk {p['chunk_index']})\n"
        body = p["text"].strip()
        remaining = max_chars - used - len(header)
        if remaining <= 200:  # not enough room for a meaningful chunk; stop
            break
        if len(body) > remaining:
            body = body[:remaining].rstrip() + " ..."
        block = header + body + "\n"
        lines.append(block)
        used += len(block)

    context_block = "\n".join(lines) if lines else "(no context retrieved)"
    return (
        f"Question: {query}\n\n"
        f"Context (sorted by relevance):\n{context_block}\n"
        "Answer the question using only the context above. "
        "Cite sources by their bracketed number (e.g. [1], [2]). "
        "If the answer is not in the context, say so."
    )
