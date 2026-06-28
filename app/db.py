from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


# `check_same_thread` is a SQLite-only quirk; ignored by other drivers.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


# Columns added after the initial schema. `create_all` only creates missing
# tables, not missing columns, so we additively ALTER existing tables. Each
# entry is (table, column, column DDL type).
_ADDED_COLUMNS: list[tuple[str, str, str]] = [
    ("mapping_plans", "entity_prompt", "TEXT"),
    ("mapping_plans", "entity_schema", "TEXT"),
    ("prompt_presets", "entity_prompt", "TEXT"),
    ("prompt_presets", "entity_schema", "TEXT"),
]


def _ensure_columns() -> None:
    """Idempotently add new nullable columns to existing tables."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, column, ddl_type in _ADDED_COLUMNS:
            if table not in existing_tables:
                continue
            cols = {c["name"] for c in inspector.get_columns(table)}
            if column not in cols:
                conn.execute(
                    text(f'ALTER TABLE {table} ADD COLUMN {column} {ddl_type}')
                )


def init_db() -> None:
    # Import models so they register with Base before create_all.
    from . import models  # noqa: F401

    Base.metadata.create_all(engine)
    _ensure_columns()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session
