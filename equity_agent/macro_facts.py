"""Latest value and 12-month change of FRED series, as ledger facts. Pure function: no I/O."""
from datetime import date, datetime

from .config import SeriesDef
from .schemas import DataGap, Fact

DCF_NOTE = "; the DCF cannot be computed without the 10-year yield (its risk-free rate)"


def _year_earlier(obs: list[list], latest_date: str) -> list | None:
    """The observation on or just before the same date one year earlier (daily series skip holidays)."""
    d = date.fromisoformat(latest_date)
    target = d.replace(year=d.year - 1, day=28) if (d.month, d.day) == (2, 29) else d.replace(year=d.year - 1)
    earlier = [o for o in obs if date.fromisoformat(o[0]) <= target]
    return earlier[-1] if earlier and (target - date.fromisoformat(earlier[-1][0])).days <= 45 else None


def series_facts(series_id: str, spec: SeriesDef, obs: list[list], retrieved_at: datetime) -> tuple[list[Fact], list[DataGap]]:
    sid = series_id.lower()
    if not obs:
        return [], [DataGap(area="macro", missing=f"{spec.label} ({series_id})", reason="FRED returned no observations",
                            effect="the series is not shown in the macro analysis" + (DCF_NOTE if series_id == "DGS10" else ""))]
    latest_date, latest = obs[-1]
    source = f"FRED series {series_id}"
    facts = [Fact(id=f"macro.{sid}.value", value=latest, unit=spec.unit, label=f"{spec.label} ({series_id})", kind="reported",
                  source=source, as_of=latest_date, retrieved_at=retrieved_at, note=f"latest observation, dated {latest_date}")]
    prior = _year_earlier(obs, latest_date)
    if prior is None or (spec.change == "pct" and prior[1] == 0):
        return facts, [DataGap(area="macro", missing=f"{spec.label}: 12-month change", reason="no observation about a year before the latest one",
                               effect="the series is shown without its trend")]
    if spec.change == "pct":
        change, unit, formula = latest / prior[1] - 1, "fraction", f"latest / value on {prior[0]} - 1 (growth over 12 months)"
    else:
        change, unit, formula = latest - prior[1], spec.unit, f"latest - value on {prior[0]} (change over 12 months)"
    facts.append(Fact(id=f"macro.{sid}.change_12m", value=change, unit=unit, label=f"{spec.label}: change over 12 months ({series_id})",
                      kind="derived", source=source, as_of=latest_date, retrieved_at=retrieved_at, note=formula))
    return facts, []
