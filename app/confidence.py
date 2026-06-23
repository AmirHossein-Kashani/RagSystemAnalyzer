from __future__ import annotations


def score_to_confidence(score: float, full_score: float) -> float:
    """Map a raw cosine similarity to a 0..1 confidence.

    Values at or above ``full_score`` map to 1.0; non-positive scores map to 0.0.
    Linear in between.
    """
    if full_score <= 0:
        raise ValueError("full_score must be > 0")
    if score <= 0:
        return 0.0
    return min(1.0, score / full_score)


def label_for(confidence: float, high: float, medium: float) -> str:
    if confidence >= high:
        return "high"
    if confidence >= medium:
        return "medium"
    return "low"


def overall_verdict(top_label: str | None) -> tuple[str, str]:
    """Pick an overall label + user-facing message from the top hit's label."""
    if top_label is None:
        return "none", "No matches found in this dataset."
    if top_label == "high":
        return "high", "Strong match found."
    if top_label == "medium":
        return "medium", "Possible match — review the references."
    return "low", "No strong match; these are the closest passages we have."
