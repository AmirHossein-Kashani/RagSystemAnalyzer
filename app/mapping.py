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


def _format_variables(variables: Optional[dict]) -> str:
    if not variables:
        return ""
    return json.dumps(variables, ensure_ascii=False, indent=2)


def _render_template(template: str, query: str, context: str, variables: Optional[dict]) -> str:
    rendered = template
    replacements = {
        "{query}": query,
        "{context}": context,
        "{variables}": _format_variables(variables),
    }
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)

    # Support {variables.field} style placeholders for convenience.
    if variables:
        for field, value in variables.items():
            token = "{variables." + str(field) + "}"
            if token in rendered:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                rendered = rendered.replace(token, str(value))
    return rendered


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

    effective_top_k = top_k or plan.default_top_k or settings.default_top_k
    search = retrieve_multi(session, embedder, store, datasets, query, effective_top_k)

    config = get_config()
    context = _build_context(search, config.max_context_chars)
    user_prompt = _render_template(plan.prompt_template, query, context, variables)
    temperature = plan.temperature if plan.temperature is not None else config.temperature
    model = resolve_model(config)

    schema: Optional[dict] = None
    if plan.output_schema:
        try:
            schema = json.loads(plan.output_schema)
        except json.JSONDecodeError as exc:
            raise ValueError(f"stored output_schema is not valid JSON: {exc}")

    client = get_client()

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
        )

    system_prompt = _build_system_prompt(plan.system_prompt, schema)

    try:
        output = client.chat_json(
            model=model,
            system=system_prompt,
            user=user_prompt,
            temperature=temperature,
        )
    except ValueError:
        # Occasionally the model emits malformed/truncated JSON. Retry once with
        # an explicit reminder before giving up.
        retry_user = (
            user_prompt
            + "\n\nIMPORTANT: Return ONLY a single complete, valid JSON object. "
            "Close and escape every string; do not include any text outside the JSON."
        )
        output = client.chat_json(
            model=model,
            system=system_prompt,
            user=retry_user,
            temperature=temperature,
        )
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
    )
