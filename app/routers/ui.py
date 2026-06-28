from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .. import repository
from ..config import settings
from ..db import get_session

router = APIRouter(tags=["ui"], include_in_schema=False)

templates = Jinja2Templates(directory=str(settings.templates_dir))

SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/", response_class=HTMLResponse)
def root() -> RedirectResponse:
    return RedirectResponse(url="/datasets", status_code=302)


@router.get("/datasets", response_class=HTMLResponse)
def datasets_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "datasets.html", {})


@router.get("/datasets/{dataset_id}", response_class=HTMLResponse)
def dataset_detail_page(
    request: Request, dataset_id: str, session: SessionDep
) -> HTMLResponse:
    dataset = repository.get_dataset(session, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return templates.TemplateResponse(
        request,
        "dataset.html",
        {"dataset_id": dataset.id, "dataset_name": dataset.name},
    )


@router.get("/search", response_class=HTMLResponse)
def search_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "search.html", {})


@router.get("/mapping-plans", response_class=HTMLResponse)
def mapping_plans_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "mapping_plans.html", {})


@router.get("/mapping-plans/{plan_id}", response_class=HTMLResponse)
def mapping_plan_detail_page(
    request: Request, plan_id: str, session: SessionDep
) -> HTMLResponse:
    plan = repository.get_mapping_plan(session, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="mapping plan not found")
    return templates.TemplateResponse(
        request,
        "mapping_plan.html",
        {"plan_id": plan.id, "plan_name": plan.name},
    )


@router.get("/debug", response_class=HTMLResponse)
def debug_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "debug.html", {})
