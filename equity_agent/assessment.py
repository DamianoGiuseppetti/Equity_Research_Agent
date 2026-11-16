"""analyze: one LLM call, four areas, findings checked in code.

A finding survives only if it cites at least one ledger id, every cited id exists, and every number in its statement is
supported by one of the facts it cites (the rounding rule in numbers.py). A failing finding is dropped and logged.
"""
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .ledger import Ledger
from .llm import LLMClient
from .numbers import display, unmatched_numbers
from .schemas import DataGap, Fact

AREAS = ("financial", "technical", "macro", "news")
Direction = Literal["positive", "neutral", "negative"]


class Finding(BaseModel):
    statement: str = Field(min_length=15, description="One or two sentences")
    direction: Direction = Field(description="Effect on the investment case")
    ledger_ids: list[str] = Field(description="Ids of the facts this statement rests on, copied exactly from the fact list")


class _ListsAsJson(BaseModel):
    """Haiku sometimes returns a list or object field as a JSON string ('[{"statement": ...}]'). Parse it; anything else is left
    to the normal validation, so a really malformed answer still fails."""

    @field_validator("*", mode="before")
    @classmethod
    def json_string_to_value(cls, v):
        if isinstance(v, str) and v.lstrip()[:1] in ("[", "{"):
            try:
                return json.loads(v)
            except ValueError:
                return v
        return v


class AreaAssessment(BaseModel):
    findings: list[Finding]
    limitations: list[str]


class Assessment(BaseModel):
    """The result: four parts and the cross-area notes."""

    financial: AreaAssessment
    technical: AreaAssessment
    macro: AreaAssessment
    news: AreaAssessment
    cross_area_notes: list[Finding] = Field(max_length=4)


class AreaFinding(Finding):
    area: Literal["financial", "technical", "macro", "news"]


class AreaLimitation(BaseModel):
    area: Literal["financial", "technical", "macro", "news"]
    text: str


class AssessmentCall(_ListsAsJson):
    """What the model is asked to return: flat lists with an area on each item. Haiku mixed up deeply nested objects (a part
    returned as a string, a part missing), so the four-part Assessment is built from this in code."""

    findings: list[AreaFinding] = Field(description="3 to 6 findings for each of the four areas")
    limitations: list[AreaLimitation] = Field(description="What the data does not show, per area")
    cross_area_notes: list[Finding] = Field(max_length=4, description="Links between areas, with ledger_ids")

    @model_validator(mode="after")
    def three_to_six_per_area(self) -> "AssessmentCall":
        for area in AREAS:
            n = sum(f.area == area for f in self.findings)
            if n < 3:
                raise ValueError(f"area {area} has {n} findings, it needs at least 3")
        return self

    def to_assessment(self) -> tuple[Assessment, list[dict]]:
        """The four-part Assessment, and the log of findings beyond the sixth in an area (the model often writes 7; the first 6 stay)."""
        trimmed: list[dict] = []

        def part(a: str) -> AreaAssessment:
            mine = [f for f in self.findings if f.area == a]
            trimmed.extend({"area": a, "statement": f.statement, "reason": "more than 6 findings for the area; only the first 6 are kept"} for f in mine[6:])
            return AreaAssessment(findings=[Finding(**f.model_dump(exclude={"area"})) for f in mine[:6]],
                                  limitations=[l.text for l in self.limitations if l.area == a])
        return Assessment(financial=part("financial"), technical=part("technical"), macro=part("macro"), news=part("news"),
                          cross_area_notes=self.cross_area_notes), trimmed


SYSTEM = """You are the analysis step of an equity research pipeline. You get a list of facts (each with an id), data gaps, and \
the company. Return flat lists: findings (each with its area: financial, technical, macro or news), limitations (each with its area) and \
cross_area_notes. Rules:
1. Use only the facts given. Each finding is one or two sentences, has a direction (positive, neutral or negative for the \
investment case) and lists the ledger ids of the facts it rests on, copied exactly.
2. Every number in a statement must be copied from a fact you cite, in the same form as the fact's display value: dollar \
amounts like "$332.2 billion", percentages like "8.1%", multiples like "1.4x", prices like "$347.73". Do not compute new numbers \
(no sums, differences or ratios of your own): if a number is not in a cited fact, leave it out and describe the point in words.
3. Exactly 3 to 6 findings for each of the four areas. Use the area's own facts: financial = market.income/balance/cashflow/ratio, filings.xbrl and the \
extract.* summaries of the 10-K; technical = market.tech and market.price; macro = macro.*; news = news.sentiment.*.
4. Technicals: describe trend, momentum and volatility. Never predict prices.
5. News sentiment scores are model estimates: word them as estimates ("the model-scored average"), and if news coverage is thin \
(fewer than 5 scored articles, no average fact), say so instead of drawing a conclusion.
6. No investment conclusion, rating, price target or valuation. Those come later from fixed rules.
7. limitations: what is missing or weak in the area, drawn from the data gaps and from what the facts cannot show.
8. cross_area_notes: up to 4 findings linking areas (for example rates and leverage), each with ledger ids."""

# fact id prefixes left out of the prompt: bookkeeping that carries no analysis value
SKIP_PREFIXES = ("filings.section.", "filings.tenk.", "filings.tenq.", "news.n_", "memo.", "analysis.", "valuation.")


def facts_for_prompt(ledger: Ledger, skip: tuple[str, ...] = SKIP_PREFIXES) -> str:
    """One line per fact: id | label | display value | kind | as_of."""
    lines = [f"{f.id} | {f.label} | {display(f)} | {f.kind} | {f.as_of}" for f in ledger.facts if not f.id.startswith(skip)]
    return "\n".join(lines)


def gaps_for_prompt(gaps: list[dict]) -> str:
    return "\n".join(f"- [{g['area']}] missing: {g['missing']} ({g['reason']}); effect: {g['effect']}" for g in gaps) or "none"


def check_finding(finding: Finding, ledger: Ledger) -> str | None:
    """None if the finding passes, otherwise why it is dropped."""
    if not finding.ledger_ids:
        return "no ledger ids"
    unknown = [i for i in finding.ledger_ids if i not in ledger]
    if unknown:
        return f"unknown ledger id(s) {unknown}"
    bad = unmatched_numbers(finding.statement, [ledger.get(i) for i in finding.ledger_ids])
    if bad:
        return f"number(s) {bad} not supported by the cited facts"
    return None


async def run_analysis(client: LLMClient, company: dict, ledger: Ledger, gaps: list[dict]) -> tuple[dict, list[dict]]:
    """(assessment as a dict with only the findings that passed, log of dropped findings). Raises LLMOutputError if the call
    is malformed twice: the caller lets it stop the run."""
    human = (f"Company: {company['name']} ({company['ticker']})\n\nFACTS (id | label | value | kind | as_of):\n{facts_for_prompt(ledger)}\n\n"
             f"DATA GAPS:\n{gaps_for_prompt(gaps)}")
    call = await client.structured("analyst", AssessmentCall, [("system", SYSTEM), ("human", human)], "analyze", max_tokens=8000)
    result, trimmed = call.to_assessment()
    dropped: list[dict] = list(trimmed)
    out: dict = {}
    for area in AREAS:
        part: AreaAssessment = getattr(result, area)
        kept, lost = [], 0
        for f in part.findings:
            why = check_finding(f, ledger)
            if why:
                dropped.append({"area": area, "statement": f.statement, "reason": why})
                lost += 1
            else:
                kept.append(f.model_dump())
        limits = list(part.limitations)
        if lost:      # written by code, so the memo cannot hide that the model's own findings were thinned
            limits.append(f"{lost} of {len(part.findings)} findings were dropped by the checks (unknown id or unsupported number)")
        out[area] = {"findings": kept, "limitations": limits}
    notes = []
    for f in result.cross_area_notes:
        why = check_finding(f, ledger)
        if why:
            dropped.append({"area": "cross_area", "statement": f.statement, "reason": why})
        else:
            notes.append(f.model_dump())
    out["cross_area_notes"] = notes
    return out, dropped


def thin_area_gaps(assessment: dict) -> list[DataGap]:
    return [DataGap(area="analysis", missing=f"findings for the {a} area", reason="all findings failed the checks",
                    effect=f"the memo has no analysed findings for {a}") for a in AREAS if not assessment[a]["findings"]]
