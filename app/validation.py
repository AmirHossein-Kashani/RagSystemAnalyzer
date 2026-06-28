from __future__ import annotations

from typing import Any, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


def check_schema(schema: dict) -> Optional[str]:
    """Validate that `schema` is itself a well-formed JSON Schema.

    Returns an error message if the schema is invalid, else None.
    """
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return f"invalid output_schema: {exc.message}"
    except Exception as exc:
        return f"invalid output_schema: {exc}"
    return None


def validate_against_schema(obj: Any, schema: dict) -> list[str]:
    """Validate `obj` against a JSON Schema.

    Returns a list of human-readable error messages; an empty list means the
    object is valid. Schema problems are returned as a single error rather than
    raised, so callers can surface them uniformly.
    """
    schema_error = check_schema(schema)
    if schema_error:
        return [schema_error]

    validator = Draft202012Validator(schema)
    errors = []
    try:
        raw_errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
    except Exception as exc:
        return [f"invalid output_schema: {exc}"]
    for error in raw_errors:
        location = "/".join(str(p) for p in error.path) or "(root)"
        errors.append(f"{location}: {error.message}")
    return errors


# Placeholder for future plan-specific semantic validators (e.g. LIO cross-field
# rules such as "every non-UNKNOWN support_needs field needs field_reasoning").
# Registered by plan name when needed.
SEMANTIC_VALIDATORS: dict[str, Any] = {}
