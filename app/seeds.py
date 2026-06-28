from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from . import repository
from .models import MappingPlan, PromptPreset

LIO_PLAN_NAME = "LIO (Learner Interpretation Object)"
LIO_PRESET_KEY = "lio_v1"

_LEVEL = ["LOW", "MODERATE", "HIGH", "UNKNOWN"]
_SENSORY = ["LOW", "MODERATE", "HIGH", "SEEKS_INPUT", "UNKNOWN"]
_DOMAIN = ["COMMUNICATION", "MOTOR", "ATTENTION", "REGULATION", "ENGAGEMENT", "AUTONOMY"]


def _strength_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "enum": _DOMAIN},
            "description": {"type": "string", "maxLength": 200},
        },
        "required": ["domain", "description"],
        "additionalProperties": False,
    }


LIO_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "LearnerInterpretationObject",
    "type": "object",
    "required": [
        "schema_version",
        "interpretation_id",
        "learner_id",
        "assessed_at",
        "interpretation_scope",
        "learner_state",
        "support_needs",
        "strengths",
        "active_challenges",
        "session_focus",
        "field_reasoning",
        "field_confidence",
        "rationale",
    ],
    "properties": {
        "schema_version": {"const": "1.0"},
        "interpretation_id": {"type": "string"},
        "learner_id": {"type": "string"},
        "assessed_at": {"type": "string"},
        "interpretation_scope": {
            "type": "string",
            "enum": ["PRE_LESSON", "PRE_SESSION"],
        },
        "learner_state": {
            "type": "object",
            "properties": {
                "sustained_attention": {"type": "string", "enum": _LEVEL},
                "cognitive_capacity_today": {"type": "string", "enum": _LEVEL},
                "regulation_capacity_today": {"type": "string", "enum": _LEVEL},
                "motor_confidence": {"type": "string", "enum": _LEVEL},
                "communication_confidence": {"type": "string", "enum": _LEVEL},
                "sensory_tolerance": {"type": "string", "enum": _SENSORY},
            },
            "additionalProperties": False,
        },
        "support_needs": {
            "type": "object",
            "properties": {
                "attention_support_need": {"type": "string", "enum": _LEVEL},
                "cognitive_support_need": {"type": "string", "enum": _LEVEL},
                "sensory_support_need": {"type": "string", "enum": _LEVEL},
                "motor_support_need": {"type": "string", "enum": _LEVEL},
                "auditory_guidance_need": {"type": "string", "enum": _LEVEL},
                "regulation_support_need": {"type": "string", "enum": _LEVEL},
                "communication_scaffolding_need": {"type": "string", "enum": _LEVEL},
                "intervention_patience_need": {"type": "string", "enum": _LEVEL},
            },
            "additionalProperties": False,
        },
        "strengths": {
            "type": "array",
            "maxItems": 3,
            "items": _strength_schema(),
        },
        "active_challenges": {
            "type": "array",
            "maxItems": 3,
            "items": _strength_schema(),
        },
        "session_focus": {
            "type": "object",
            "required": ["primary_goal", "challenge_readiness"],
            "properties": {
                "primary_goal": {
                    "type": "string",
                    "enum": [
                        "DAILY_PRACTICE",
                        "BUILD_CONFIDENCE",
                        "TARGET_SKILL",
                        "COMMUNICATION_PREP",
                        "REGULATION_FIRST",
                        "ASSESSMENT",
                        "CAREGIVER_LED",
                    ],
                },
                "challenge_readiness": {
                    "type": "string",
                    "enum": ["RECOVERY", "AT_LEVEL", "STRETCH"],
                },
                "communication_function": {
                    "type": "string",
                    "enum": [
                        "LABEL",
                        "REQUEST",
                        "REFUSE",
                        "SOCIAL",
                        "SELF_ADVOCACY",
                        "INFORMATION",
                        "EXPRESS_PREFERENCE",
                        "NONE",
                    ],
                },
                "success_priority": {
                    "type": "string",
                    "enum": [
                        "PARTICIPATION",
                        "COMPLETION_WITH_SUPPORT",
                        "INDEPENDENT_MASTERY",
                    ],
                },
            },
            "additionalProperties": False,
        },
        "content_hints": {
            "type": "object",
            "properties": {
                "vocabulary_strategy": {
                    "type": "string",
                    "enum": [
                        "CORE_AAC",
                        "INTEREST_BASED",
                        "ERROR_TARGETED",
                        "ROUTINE_FUNCTIONAL",
                        "CUSTOM_LIST",
                    ],
                },
                "interest_topics": {"type": "array", "items": {"type": "string"}},
                "target_skill_focus": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "field_reasoning": {"type": "object"},
        "field_confidence": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": ["LOW", "MODERATE", "HIGH"],
            },
        },
        "rationale": {
            "type": "object",
            "required": ["summary"],
            "properties": {"summary": {"type": "string"}},
            "additionalProperties": False,
        },
        "evidence_used": {"type": "array"},
        "constraints_acknowledged": {"type": "array", "items": {"type": "string"}},
        "overall_confidence": {
            "type": "string",
            "enum": ["LOW", "MODERATE", "HIGH"],
        },
    },
    "additionalProperties": False,
}

LIO_SYSTEM_PROMPT = (
    "You are an educational reasoning assistant for an autism-focused learning "
    "platform. Analyze all available learner information and produce a single "
    "valid LearnerInterpretationObject (LIO).\n\n"
    "The LIO represents your INTERPRETATION of the learner's current state and "
    "educational needs. It does NOT describe how the software should behave. A "
    "separate Policy Engine translates the LIO into runtime behavior.\n\n"
    "Infer higher-level learner characteristics; do not merely repeat observations. "
    "Use the retrieved context (autism intervention literature, AAC best practices, "
    "clinical guidelines) to interpret evidence, not to replace it.\n\n"
    "You MUST NOT generate runtime settings, API calls, UI configuration, keyboard "
    "settings, timing thresholds, brightness/animation/audio settings, lesson "
    "execution parameters, or any platform-specific configuration.\n\n"
    "Rules:\n"
    "- Use UNKNOWN when evidence is insufficient; do not guess.\n"
    "- Every non-UNKNOWN support_needs field must have supporting bullets in "
    "field_reasoning.support_needs.\n"
    "- Prefer recent evidence and repeated patterns over isolated/stale observations.\n"
    "- field_confidence estimates YOUR confidence per domain, not the learner's ability.\n"
    "- strengths and active_challenges each contain 0-3 items.\n\n"
    "Return ONLY a single valid JSON object conforming to the LIO schema. Do not "
    "include explanations, markdown, code fences, or any text outside the JSON.\n\n"
    "Follow the exact structure of this minimal valid example (same field names, "
    "nesting, and enum values; fill with your own interpretation):\n"
    + json.dumps(
        {
            "schema_version": "1.0",
            "interpretation_id": "00000000-0000-4000-8000-000000000001",
            "learner_id": "learner_042",
            "assessed_at": "2026-06-27T14:30:00Z",
            "interpretation_scope": "PRE_LESSON",
            "learner_state": {
                "sustained_attention": "LOW",
                "cognitive_capacity_today": "MODERATE",
                "regulation_capacity_today": "LOW",
                "motor_confidence": "MODERATE",
                "communication_confidence": "MODERATE",
                "sensory_tolerance": "LOW",
            },
            "support_needs": {
                "attention_support_need": "HIGH",
                "cognitive_support_need": "HIGH",
                "sensory_support_need": "HIGH",
                "motor_support_need": "MODERATE",
                "auditory_guidance_need": "LOW",
                "regulation_support_need": "HIGH",
                "communication_scaffolding_need": "MODERATE",
                "intervention_patience_need": "HIGH",
            },
            "strengths": [],
            "active_challenges": [
                {
                    "domain": "ATTENTION",
                    "description": "Difficulty sustaining focus during multi-letter words",
                }
            ],
            "session_focus": {
                "primary_goal": "BUILD_CONFIDENCE",
                "challenge_readiness": "RECOVERY",
            },
            "field_reasoning": {
                "support_needs": {
                    "attention_support_need": [
                        "Frequent gaze shifts reported by caregiver",
                        "Elevated idle assists in recent history",
                    ]
                }
            },
            "field_confidence": {
                "attention": "HIGH",
                "cognitive": "MODERATE",
                "regulation": "HIGH",
                "motor": "MODERATE",
                "communication": "LOW",
                "sensory": "MODERATE",
                "session_plan": "MODERATE",
            },
            "rationale": {
                "summary": "Caregiver reports gaze shifts and prior session ended after dysregulation; infer low attention and regulation capacity today.",
            },
        },
        indent=2,
    )
)

LIO_PROMPT_TEMPLATE = (
    "Learner input and context:\n{query}\n\n"
    "Structured variables (may include learner_id, assessed_at, profile constraints):\n"
    "{variables}\n\n"
    "Retrieved knowledge (numbered sources):\n{context}\n\n"
    "Produce the LearnerInterpretationObject as specified by the system instructions. "
    "Return only the JSON object."
)


# ---------------------------------------------------------------------------
# Additional ready-made prompt presets (a small library to choose from)
# ---------------------------------------------------------------------------

GROUNDED_QA_SYSTEM_PROMPT = (
    "You are a careful question-answering assistant. Answer the user's question "
    "using ONLY the retrieved context provided. If the context does not contain "
    "the answer, say so explicitly rather than guessing. Be concise, accurate, "
    "and cite the numbered sources you relied on."
)

GROUNDED_QA_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GroundedAnswer",
    "type": "object",
    "required": ["answer", "key_points", "confidence"],
    "properties": {
        "answer": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "sources_used": {"type": "array", "items": {"type": "integer"}},
        "answer_found": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["LOW", "MODERATE", "HIGH"]},
    },
    "additionalProperties": False,
}

GROUNDED_QA_TEMPLATE = (
    "Question:\n{query}\n\n"
    "Retrieved context (numbered sources):\n{context}\n\n"
    "Answer the question grounded strictly in the context above. "
    "Return only the JSON object."
)

EXTRACTION_SYSTEM_PROMPT = (
    "You are an information extraction engine. From the retrieved context, extract "
    "structured entities and facts relevant to the user's request. Do not invent "
    "values; only extract what is supported by the context. Use null or empty "
    "arrays when information is absent."
)

EXTRACTION_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ExtractionResult",
    "type": "object",
    "required": ["entities", "summary"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "value": {"type": ["string", "number", "boolean", "null"]},
                    "evidence": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "additionalProperties": False,
}

EXTRACTION_TEMPLATE = (
    "Extraction request:\n{query}\n\n"
    "Optional field hints / variables:\n{variables}\n\n"
    "Retrieved context (numbered sources):\n{context}\n\n"
    "Extract the requested structured information. Return only the JSON object."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are a precise summarization assistant. Produce a clear, faithful summary "
    "of the retrieved context, focused on the user's request. Do not add facts "
    "that are not present in the context. Prefer plain language."
)

SUMMARY_TEMPLATE = (
    "Summarization request:\n{query}\n\n"
    "Retrieved context (numbered sources):\n{context}\n\n"
    "Write the summary."
)


@dataclass(frozen=True)
class PresetDef:
    key: str
    name: str
    description: str
    category: str
    system_prompt: str
    output_schema: Optional[dict]
    prompt_template: str
    default_top_k: Optional[int]
    temperature: Optional[float]


PROMPT_PRESETS: list[PresetDef] = [
    PresetDef(
        key=LIO_PRESET_KEY,
        name=LIO_PLAN_NAME,
        description=(
            "Maps learner information to a validated LearnerInterpretationObject "
            "(LIO v1.0) grounded in selected knowledge datasets."
        ),
        category="Autism platform",
        system_prompt=LIO_SYSTEM_PROMPT,
        output_schema=LIO_SCHEMA,
        prompt_template=LIO_PROMPT_TEMPLATE,
        default_top_k=6,
        temperature=0.1,
    ),
    PresetDef(
        key="grounded_qa_v1",
        name="Grounded Q&A (structured)",
        description=(
            "Answers a question strictly from retrieved context and returns a "
            "structured answer with key points and a confidence level."
        ),
        category="General",
        system_prompt=GROUNDED_QA_SYSTEM_PROMPT,
        output_schema=GROUNDED_QA_SCHEMA,
        prompt_template=GROUNDED_QA_TEMPLATE,
        default_top_k=5,
        temperature=0.1,
    ),
    PresetDef(
        key="extraction_v1",
        name="Structured extraction",
        description=(
            "Extracts entities and facts from retrieved context into a generic "
            "structured JSON result."
        ),
        category="General",
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        output_schema=EXTRACTION_SCHEMA,
        prompt_template=EXTRACTION_TEMPLATE,
        default_top_k=8,
        temperature=0.0,
    ),
    PresetDef(
        key="summary_v1",
        name="Document summary (free text)",
        description=(
            "Produces a faithful plain-text summary of the retrieved context. No "
            "output schema (free-form text)."
        ),
        category="General",
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        output_schema=None,
        prompt_template=SUMMARY_TEMPLATE,
        default_top_k=6,
        temperature=0.2,
    ),
]


def seed_prompt_presets(session: Session) -> list[PromptPreset]:
    """Idempotently create/refresh all built-in prompt presets."""
    presets: list[PromptPreset] = []
    for d in PROMPT_PRESETS:
        presets.append(
            repository.upsert_builtin_preset(
                session,
                key=d.key,
                name=d.name,
                description=d.description,
                category=d.category,
                system_prompt=d.system_prompt,
                output_schema=json.dumps(d.output_schema) if d.output_schema else None,
                prompt_template=d.prompt_template,
                default_top_k=d.default_top_k,
                temperature=d.temperature,
            )
        )
    return presets


def seed_lio_plan(session: Session) -> MappingPlan:
    """Create the built-in LIO mapping plan if it does not already exist.

    Also ensures the prompt-preset library is seeded so the LIO (and the other
    ready-made presets) are available to choose from.
    """
    seed_prompt_presets(session)

    existing = repository.get_mapping_plan_by_name(session, LIO_PLAN_NAME)
    if existing is not None:
        return existing

    return repository.create_mapping_plan(
        session,
        name=LIO_PLAN_NAME,
        description=(
            "Maps learner information to a validated LearnerInterpretationObject "
            "(LIO v1.0) grounded in selected knowledge datasets."
        ),
        system_prompt=LIO_SYSTEM_PROMPT,
        output_schema=json.dumps(LIO_SCHEMA),
        prompt_template=LIO_PROMPT_TEMPLATE,
        default_top_k=6,
        temperature=0.1,
    )
