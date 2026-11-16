"""Rules for reviewer feedback, in plain code: which sections a change touches, which assumption values are acceptable.

route_feedback (nodes/review.py) uses these; nothing here calls a model.
"""
from .memo_writer import SECTIONS, WRITTEN_BY_MODEL

# Sections redrafted after the valuation reruns (assumptions or peers changed): the four sections that depend on the valuation.
VALUATION_SECTIONS = ["Valuation", "Investment Thesis", "Executive Summary", "Conclusion"]
# An assumptions change also moves the projections that Financial Outlook cites; a peers change does not.
ASSUMPTION_SECTIONS = ["Valuation", "Investment Thesis", "Financial Outlook", "Executive Summary", "Conclusion"]
# A section is redrafted when one it depends on was changed. The Executive Summary is written from all the others; the Conclusion
# from the findings, valuation and risks, which the sections below discuss.
DEPENDS_ON = {"Executive Summary": set(WRITTEN_BY_MODEL) - {"Executive Summary"},
              "Conclusion": {"Historical Financial Analysis", "Investment Thesis", "Financial Outlook", "Valuation", "Risks and Catalysts"}}

# Editable assumptions: key in config.yaml -> (label, lowest, highest, unit shown). Proposed ranges: a wider value is rejected.
ASSUMPTION_RANGES = {
    "equity_risk_premium": ("Equity risk premium", 0.03, 0.08, "fraction"),
    "terminal_growth": ("Terminal growth", 0.0, 0.04, "fraction"),
    "scenario_spread_pts": ("Scenario spread (points of growth, bear and bull)", 1.0, 6.0, "points"),
    "fade_target_growth": ("Fade target growth", 0.0, 0.06, "fraction"),
}


def with_dependents(chosen: list[str]) -> list[str]:
    """The chosen sections plus the ones that depend on them, in memo order. Unknown names are dropped."""
    picked = {s for s in chosen if s in SECTIONS}
    for name, deps in DEPENDS_ON.items():
        if picked & deps:
            picked.add(name)
    return [s for s in SECTIONS if s in picked and s != "Sources and Methodology"]


def validate_assumptions(values: dict) -> tuple[dict[str, float], list[str]]:
    """(clean values for the keys that were given, error messages). Fractions are written as fractions: 0.05 for 5%."""
    clean, errors = {}, []
    for key, raw in values.items():
        if key not in ASSUMPTION_RANGES:
            errors.append(f"'{key}' cannot be edited. Editable: {', '.join(ASSUMPTION_RANGES)}.")
            continue
        label, low, high, unit = ASSUMPTION_RANGES[key]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label}: '{raw}' is not a number.")
            continue
        if not low <= value <= high:
            shown = (lambda x: f"{x:.1%}") if unit == "fraction" else (lambda x: f"{x:g} points")
            errors.append(f"{label}: {shown(value) if unit == 'fraction' else f'{value:g} points'} is outside the accepted range {shown(low)} to {shown(high)}.")
            continue
        clean[key] = value
    return clean, errors
