from __future__ import annotations

import json
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import repository
from ..db import get_session
from ..deps import get_embedder, get_store
from ..embedder import Embedder
from ..mapping import run_mapping_plan
from ..models import MappingPlan
from ..schemas import (
    MappingPlanCreate,
    MappingPlanOut,
    MappingPlanUpdate,
    MappingRunRequest,
    MappingRunResponse,
)
from ..seeds import seed_lio_plan
from ..store import VectorStore
from ..validation import check_schema

router = APIRouter(prefix="/api/mapping-plans", tags=["mapping"])

SessionDep = Annotated[Session, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
StoreDep = Annotated[VectorStore, Depends(get_store)]


def _parse_schema_or_400(output_schema: Optional[dict]) -> Optional[str]:
    if output_schema is None:
        return None
    schema_error = check_schema(output_schema)
    if schema_error:
        raise HTTPException(status_code=400, detail=schema_error)
    return json.dumps(output_schema)


def _plan_to_out(plan: MappingPlan) -> MappingPlanOut:
    schema: Optional[dict] = None
    if plan.output_schema:
        try:
            schema = json.loads(plan.output_schema)
        except json.JSONDecodeError:
            schema = None
    return MappingPlanOut(
        id=plan.id,
        name=plan.name,
        description=plan.description,
        system_prompt=plan.system_prompt,
        output_schema=schema,
        prompt_template=plan.prompt_template,
        default_top_k=plan.default_top_k,
        temperature=plan.temperature,
        dataset_ids=[link.dataset_id for link in plan.datasets],
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


@router.post("", response_model=MappingPlanOut, status_code=201)
def create_plan(payload: MappingPlanCreate, session: SessionDep) -> MappingPlanOut:
    if repository.get_mapping_plan_by_name(session, payload.name):
        raise HTTPException(status_code=409, detail="mapping plan name already exists")

    schema_json = _parse_schema_or_400(payload.output_schema)
    plan = repository.create_mapping_plan(
        session,
        name=payload.name,
        description=payload.description,
        system_prompt=payload.system_prompt,
        output_schema=schema_json,
        prompt_template=payload.prompt_template,
        default_top_k=payload.default_top_k,
        temperature=payload.temperature,
    )
    if payload.dataset_ids:
        repository.set_mapping_plan_datasets(session, plan, payload.dataset_ids)
    return _plan_to_out(plan)


@router.post("/seed-lio", response_model=MappingPlanOut)
def seed_lio(session: SessionDep) -> MappingPlanOut:
    """Idempotently create (or return) the built-in LIO mapping plan."""
    plan = seed_lio_plan(session)
    return _plan_to_out(plan)


@router.get("", response_model=list[MappingPlanOut])
def list_plans(session: SessionDep) -> list[MappingPlanOut]:
    return [_plan_to_out(p) for p in repository.list_mapping_plans(session)]


@router.get("/{plan_id}", response_model=MappingPlanOut)
def get_plan(plan_id: str, session: SessionDep) -> MappingPlanOut:
    plan = repository.get_mapping_plan(session, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="mapping plan not found")
    return _plan_to_out(plan)


@router.put("/{plan_id}", response_model=MappingPlanOut)
def update_plan(
    plan_id: str, payload: MappingPlanUpdate, session: SessionDep
) -> MappingPlanOut:
    plan = repository.get_mapping_plan(session, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="mapping plan not found")

    if payload.name and payload.name != plan.name:
        existing = repository.get_mapping_plan_by_name(session, payload.name)
        if existing is not None and existing.id != plan.id:
            raise HTTPException(
                status_code=409, detail="mapping plan name already exists"
            )

    fields: dict = {}
    if payload.name is not None:
        fields["name"] = payload.name.strip()
    if payload.description is not None:
        fields["description"] = payload.description
    if payload.system_prompt is not None:
        fields["system_prompt"] = payload.system_prompt
    if payload.prompt_template is not None:
        fields["prompt_template"] = payload.prompt_template
    if payload.default_top_k is not None:
        fields["default_top_k"] = payload.default_top_k
    if payload.temperature is not None:
        fields["temperature"] = payload.temperature
    if payload.output_schema is not None:
        fields["output_schema"] = _parse_schema_or_400(payload.output_schema)

    repository.update_mapping_plan(session, plan, **fields)
    if payload.dataset_ids is not None:
        repository.set_mapping_plan_datasets(session, plan, payload.dataset_ids)
    return _plan_to_out(plan)


@router.delete("/{plan_id}", status_code=204)
def delete_plan(plan_id: str, session: SessionDep) -> None:
    plan = repository.get_mapping_plan(session, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="mapping plan not found")
    repository.delete_mapping_plan(session, plan)


@router.post("/{plan_id}/run", response_model=MappingRunResponse)
def run_plan(
    plan_id: str,
    payload: MappingRunRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    store: StoreDep,
) -> MappingRunResponse:
    plan = repository.get_mapping_plan(session, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="mapping plan not found")

    try:
        result = run_mapping_plan(
            session,
            embedder,
            store,
            plan,
            query=payload.query,
            dataset_ids=payload.dataset_ids,
            top_k=payload.top_k,
            variables=payload.variables,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"mapping run failed: {exc}") from exc

    return MappingRunResponse(
        plan_id=plan.id,
        output=result.output,
        valid=result.valid,
        validation_errors=result.validation_errors,
        repaired=result.repaired,
        model=result.model,
        provider=result.provider,
        search=result.search,
    )
