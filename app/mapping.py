from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.orm import Session

from .config import settings
from .embedder import Embedder
from .llm import get_client, get_config, resolve_model
from .models import MappingPlan
from .retrieval import lookup_datasets_or_raise, retrieve_multi
from .schemas import SearchResponse
from .store import VectorStore
from .validation import validate_against_schema


@dataclass
class MappingResult:
    output: Any
    valid: bool
    validation_errors: list[str]
    repaired: bool
    model: str
    provider: str
    search: SearchResponse
    entities: Optional[dict] = None
    entity_error: Optional[str] = None


def _format_obj(obj: Optional[dict]) -> str:
    if not obj:
        return ""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _render_dotted(rendered: str, prefix: str, obj: Optional[dict]) -> str:
    """Substitute {prefix.field} placeholders from a dict."""
    if not obj:
        return rendered
    for field, value in obj.items():
        token = "{" + prefix + "." + str(field) + "}"
        if token in rendered:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            rendered = rendered.replace(token, str(value))
    return rendered


def _render_template(
    template: str,
    query: str,
    context: str,
    variables: Optional[dict],
    entities: Optional[dict] = None,
) -> str:
    rendered = template
    replacements = {
        "{query}": query,
        "{context}": context,
        "{variables}": _format_obj(variables),
        "{entities}": _format_obj(entities),
    }
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)

    # Support {variables.field} / {entities.field} style placeholders.
    rendered = _render_dotted(rendered, "variables", variables)
    rendered = _render_dotted(rendered, "entities", entities)
    return rendered


def _generate_json(client, model: str, system: str, user: str, temperature: float) -> dict:
    """chat_json with one retry on malformed/truncated output."""
    try:
        return client.chat_json(model=model, system=system, user=user, temperature=temperature)
    except ValueError:
        retry_user = (
            user
            + "\n\nIMPORTANT: Return ONLY a single complete, valid JSON object. "
            "Close and escape every string; do not include any text outside the JSON."
        )
        return client.chat_json(
            model=model, system=system, user=retry_user, temperature=temperature
        )


def _load_schema(raw: Optional[str], label: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"stored {label} is not valid JSON: {exc}")


def extract_entities(
    client,
    model: str,
    plan: MappingPlan,
    query: str,
    variables: Optional[dict],
    temperature: float,
) -> tuple[Optional[dict], Optional[str]]:
    """Stage 1: run the entity-extraction LLM on the raw query + variables.

    Returns (entities, error). On any failure we degrade gracefully: the mapping
    run continues with entities=None and the error is surfaced to the caller.
    """
    entity_schema = _load_schema(plan.entity_schema, "entity_schema")
    system = _build_system_prompt(plan.entity_prompt or "", entity_schema)
    user = (
        f"Input:\n{query}\n\n"
        f"Structured variables:\n{_format_obj(variables) or '(none)'}\n\n"
        "Extract the entities as instructed. Return only the JSON object."
    )
    try:
        entities = _generate_json(client, model, system, user, temperature)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        return None, f"entity extraction failed: {exc}"

    if entity_schema is not None:
        # Best-effort validation: note errors but don't block the run.
        errs = validate_against_schema(entities, entity_schema)
        if errs:
            return entities, "entities did not fully match entity_schema: " + "; ".join(errs[:3])
    return entities, None


def _build_system_prompt(base_system: str, schema: Optional[dict]) -> str:
    """When a JSON Schema is configured, append it (and a strict instruction) so
    the model sees the exact required structure, field names, and enum values."""
    if schema is None:
        return base_system
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return (
        base_system
        + "\n\nYour entire response MUST be a single JSON object that strictly "
        "validates against this JSON Schema. Use ONLY the property names and enum "
        "values it defines; do not add, rename, or invent fields, and respect every "
        "`required` list and `additionalProperties: false` constraint.\n\n"
        "JSON Schema:\n"
        + schema_text
        + "\n\nReturn only the JSON object — no markdown, comments, or extra text."
    )


def _build_context(search: SearchResponse, max_chars: int) -> str:
    lines: list[str] = []
    used = 0
    for i, hit in enumerate(search.hits, 1):
        ref = hit.reference
        header = (
            f"[{i}] Source: {ref.filename} (chunk {ref.chunk_index}, "
            f"dataset {ref.dataset_name})\n"
        )
        body = hit.text.strip()
        remaining = max_chars - used - len(header)
        if remaining <= 200:
            break
        if len(body) > remaining:
            body = body[:remaining].rstrip() + " ..."
        block = header + body + "\n"
        lines.append(block)
        used += len(block)
    return "\n".join(lines) if lines else "(no context retrieved)"


def run_mapping_plan(
    session: Session,
    embedder: Embedder,
    store: VectorStore,
    plan: MappingPlan,
    query: str,
    dataset_ids: Optional[list[str]],
    top_k: Optional[int],
    variables: Optional[dict],
) -> MappingResult:
    effective_ids = dataset_ids or [link.dataset_id for link in plan.datasets]
    datasets = lookup_datasets_or_raise(session, effective_ids)

    config = get_config()
    temperature = plan.temperature if plan.temperature is not None else config.temperature
    model = resolve_model(config)
    client = get_client()

    # Stage 1 (optional): extract structured entities from the raw input first.
    entities: Optional[dict] = None
    entity_error: Optional[str] = None
    if plan.entity_prompt:
        entities, entity_error = extract_entities(
            client, model, plan, query, variables, temperature
        )

    effective_top_k = top_k or plan.default_top_k or settings.default_top_k
    search = retrieve_multi(session, embedder, store, datasets, query, effective_top_k)

    context = _build_context(search, config.max_context_chars)
    user_prompt = _render_template(
        plan.prompt_template, query, context, variables, entities
    )

    schema = _load_schema(plan.output_schema, "output_schema")

    if schema is None:
        # Free-form text output (no schema to enforce).
        text = client.chat(
            model=model,
            system=plan.system_prompt,
            user=user_prompt,
            temperature=temperature,
        )
        return MappingResult(
            output=text,
            valid=True,
            validation_errors=[],
            repaired=False,
            model=model,
            provider=config.provider,
            search=search,
            entities=entities,
            entity_error=entity_error,
        )

    system_prompt = _build_system_prompt(plan.system_prompt, schema)

    output = _generate_json(client, model, system_prompt, user_prompt, temperature)
    errors = validate_against_schema(output, schema)
    repaired = False

    if errors:
        repair_prompt = (
            user_prompt
            + "\n\nYour previous response was INVALID against the required JSON "
            + "Schema. Previous response:\n"
            + json.dumps(output, ensure_ascii=False)
            + "\n\nValidation errors:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n\nReturn a corrected JSON object that fixes every error above and "
            + "conforms exactly to the schema. Return only the JSON object."
        )
        try:
            repaired_output = client.chat_json(
                model=model,
                system=system_prompt,
                user=repair_prompt,
                temperature=temperature,
            )
            repaired_errors = validate_against_schema(repaired_output, schema)
            repaired = True
            if not repaired_errors or len(repaired_errors) < len(errors):
                output = repaired_output
                errors = repaired_errors
        except Exception:
            # Keep the original output and errors if repair fails to parse.
            pass

    return MappingResult(
        output=output,
        valid=not errors,
        validation_errors=errors,
        repaired=repaired,
        model=model,
        provider=config.provider,
        search=search,
        entities=entities,
        entity_error=entity_error,
    )
