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


# ---------- Entity-based (agentic) retrieval planning ----------

ENTITY_PLANNER_SYSTEM = (
    "You are a retrieval-planning assistant for a RAG system. Given a user's question, "
    "identify the distinct entities, subjects, or sub-topics that should each be searched "
    "SEPARATELY in a document collection to gather the evidence needed to answer well. "
    "Prefer concrete names/nouns over generic words, and keep each item to a short search "
    "phrase. Return ONLY a JSON array of strings — no prose, no markdown fences."
)


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = it.lower()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _coerce_entity_list(data: object) -> list[str]:
    """Pull a flat list of non-empty strings out of whatever the model returned."""
    if isinstance(data, dict):
        # Tolerate {"entities": [...]}, {"queries": [...]}, or the first list value.
        for key in ("entities", "queries", "subjects", "items", "results"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
        else:
            data = next((v for v in data.values() if isinstance(v, list)), [])
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(
                item.get("entity") or item.get("query") or item.get("name") or ""
            ).strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def parse_entity_list(raw: str) -> list[str]:
    """Best-effort parse of an LLM response into a list of search phrases.

    Handles plain JSON, ```json fenced blocks, and falls back to line/comma splitting
    when the model ignores the JSON instruction.
    """
    s = (raw or "").strip()
    if not s:
        return []

    # Strip a surrounding markdown code fence if present.
    if s.startswith("```"):
        inner = s[3:]
        if inner[:4].lower() == "json":
            inner = inner[4:]
        s = inner.rsplit("```", 1)[0].strip() if "```" in inner else inner.strip()

    # Prefer the first JSON array embedded in the text, then the whole string.
    candidates: list[str] = []
    start, end = s.find("["), s.rfind("]")
    if start != -1 and end > start:
        candidates.append(s[start : end + 1])
    candidates.append(s)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        items = _coerce_entity_list(parsed)
        if items:
            return _dedupe_preserving_order(items)

    # Fallback: treat each non-empty line as an entity, stripping bullets/numbering.
    lines: list[str] = []
    for line in s.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip().strip('",')
        if cleaned:
            lines.append(cleaned)
    # A single line that reads like a sentence (a refusal or prose) is not a list —
    # signal a parse failure so the caller can fall back to the raw query.
    if len(lines) == 1 and lines[0].endswith((".", "!", "?")) and " " in lines[0]:
        return []
    if lines:
        return _dedupe_preserving_order(lines)

    return _dedupe_preserving_order([p.strip() for p in s.split(",") if p.strip()])


def extract_entities(
    query: str,
    *,
    model: str,
    temperature: float,
    max_entities: int,
) -> list[str]:
    """Ask the LLM which entities/subjects to search for.

    Always returns at least one item: falls back to the raw query as a single entity
    if the model returns nothing parseable.
    """
    user = (
        f"Question: {query}\n\n"
        f"List up to {max_entities} entities or subjects to search for, ordered by "
        "importance, as a JSON array of short search phrases."
    )
    raw = get_client().chat(
        model=model,
        system=ENTITY_PLANNER_SYSTEM,
        user=user,
        temperature=temperature,
    )
    entities = parse_entity_list(raw) or [query]
    return entities[:max_entities]


def build_entity_synthesis_prompt(
    query: str,
    groups: list[dict],
    max_chars: int,
) -> str:
    """Compose the final synthesis prompt from per-entity evidence groups.

    `groups` is a list of dicts: {"entity": str, "passages": [{filename, chunk_index,
    text}, ...]}. Passages are numbered GLOBALLY ([1], [2], ...) across all entities so
    the model can cite unambiguously. Truncates the evidence to `max_chars`.
    """
    blocks: list[str] = []
    used = 0
    n = 0
    stop = False
    for group in groups:
        if stop:
            break
        header = f"\n### Evidence for entity: {group['entity']}\n"
        if used + len(header) > max_chars:
            break
        blocks.append(header)
        used += len(header)

        passages = group.get("passages") or []
        if not passages:
            note = "(no relevant passages found)\n"
            blocks.append(note)
            used += len(note)
            continue

        for p in passages:
            ref = f"[{n + 1}] Source: {p['filename']} (chunk {p['chunk_index']})\n"
            remaining = max_chars - used - len(ref)
            if remaining <= 200:  # not enough room for a meaningful chunk; stop
                stop = True
                break
            n += 1
            body = p["text"].strip()
            if len(body) > remaining:
                body = body[:remaining].rstrip() + " ..."
            block = ref + body + "\n"
            blocks.append(block)
            used += len(block)

    context_block = "".join(blocks).strip() or "(no context retrieved)"
    return (
        f"Question: {query}\n\n"
        "You ran a separate search for each of several entities/subjects. The evidence "
        "gathered is grouped by entity below and numbered globally:\n\n"
        f"{context_block}\n\n"
        "Using ONLY the evidence above, make a final decision and answer the question. "
        "Synthesize across the entities, note any agreements or conflicts between them, "
        "and cite the passages you use by their bracketed number (e.g. [1], [3]). "
        "If the evidence is insufficient to answer, say so explicitly."
    )
