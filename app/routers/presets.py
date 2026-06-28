from __future__ import annotations

import json
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import repository
from ..db import get_session
from ..models import MappingPlan, PromptPreset
from ..schemas import (
    CreatePlanFromPresetRequest,
    MappingPlanOut,
    PromptPresetCreate,
    PromptPresetOut,
)
from ..seeds import seed_prompt_presets
from ..validation import check_schema

router = APIRouter(prefix="/api/prompt-presets", tags=["prompt-presets"])

SessionDep = Annotated[Session, Depends(get_session)]


def _schema_dict(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _check_and_dump(schema: Optional[dict]) -> Optional[str]:
    if schema is None:
        return None
    schema_error = check_schema(schema)
    if schema_error:
        raise HTTPException(status_code=400, detail=schema_error)
    return json.dumps(schema)


def _preset_to_out(preset: PromptPreset) -> PromptPresetOut:
    return PromptPresetOut(
        id=preset.id,
        key=preset.key,
        name=preset.name,
        description=preset.description,
        category=preset.category,
        system_prompt=preset.system_prompt,
        output_schema=_schema_dict(preset.output_schema),
        prompt_template=preset.prompt_template,
        entity_prompt=preset.entity_prompt,
        entity_schema=_schema_dict(preset.entity_schema),
        default_top_k=preset.default_top_k,
        temperature=preset.temperature,
        is_builtin=preset.is_builtin,
        created_at=preset.created_at,
        updated_at=preset.updated_at,
    )


def _plan_to_out(plan: MappingPlan) -> MappingPlanOut:
    return MappingPlanOut(
        id=plan.id,
        name=plan.name,
        description=plan.description,
        system_prompt=plan.system_prompt,
        output_schema=_schema_dict(plan.output_schema),
        prompt_template=plan.prompt_template,
        entity_prompt=plan.entity_prompt,
        entity_schema=_schema_dict(plan.entity_schema),
        default_top_k=plan.default_top_k,
        temperature=plan.temperature,
        dataset_ids=[link.dataset_id for link in plan.datasets],
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


@router.post("/seed", response_model=list[PromptPresetOut])
def seed_presets(session: SessionDep) -> list[PromptPresetOut]:
    """Idempotently create/refresh the built-in prompt-preset library."""
    presets = seed_prompt_presets(session)
    return [_preset_to_out(p) for p in presets]


@router.get("", response_model=list[PromptPresetOut])
def list_presets(session: SessionDep) -> list[PromptPresetOut]:
    return [_preset_to_out(p) for p in repository.list_prompt_presets(session)]


@router.post("", response_model=PromptPresetOut, status_code=201)
def create_preset(payload: PromptPresetCreate, session: SessionDep) -> PromptPresetOut:
    if repository.get_prompt_preset_by_key(session, payload.key):
        raise HTTPException(status_code=409, detail="preset key already exists")

    schema_json = _check_and_dump(payload.output_schema)
    entity_schema_json = _check_and_dump(payload.entity_schema)

    preset = repository.create_prompt_preset(
        session,
        key=payload.key,
        name=payload.name,
        description=payload.description,
        category=payload.category,
        system_prompt=payload.system_prompt,
        output_schema=schema_json,
        prompt_template=payload.prompt_template,
        entity_prompt=payload.entity_prompt,
        entity_schema=entity_schema_json,
        default_top_k=payload.default_top_k,
        temperature=payload.temperature,
        is_builtin=False,
    )
    return _preset_to_out(preset)


@router.get("/{preset_id}", response_model=PromptPresetOut)
def get_preset(preset_id: str, session: SessionDep) -> PromptPresetOut:
    preset = repository.get_prompt_preset(session, preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="preset not found")
    return _preset_to_out(preset)


@router.delete("/{preset_id}", status_code=204)
def delete_preset(preset_id: str, session: SessionDep) -> None:
    preset = repository.get_prompt_preset(session, preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="preset not found")
    if preset.is_builtin:
        raise HTTPException(status_code=400, detail="built-in presets cannot be deleted")
    repository.delete_prompt_preset(session, preset)


@router.post("/{preset_id}/create-plan", response_model=MappingPlanOut, status_code=201)
def create_plan_from_preset(
    preset_id: str, payload: CreatePlanFromPresetRequest, session: SessionDep
) -> MappingPlanOut:
    """Create a new mapping plan pre-filled from a preset."""
    preset = repository.get_prompt_preset(session, preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="preset not found")

    base_name = (payload.name or preset.name).strip()
    name = base_name
    suffix = 2
    while repository.get_mapping_plan_by_name(session, name) is not None:
        name = f"{base_name} ({suffix})"
        suffix += 1

    plan = repository.create_mapping_plan(
        session,
        name=name,
        description=preset.description,
        system_prompt=preset.system_prompt,
        output_schema=preset.output_schema,
        prompt_template=preset.prompt_template,
        entity_prompt=preset.entity_prompt,
        entity_schema=preset.entity_schema,
        default_top_k=preset.default_top_k or 5,
        temperature=preset.temperature,
    )
    if payload.dataset_ids:
        repository.set_mapping_plan_datasets(session, plan, payload.dataset_ids)
    return _plan_to_out(plan)
