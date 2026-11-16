"""Pydantic models for facts, gaps and company resolution, plus the small graph-state TypedDict."""
import operator
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Kind = Literal["reported", "derived", "summary", "assumption", "model_estimate"]   # summary: model-written text from the 10-K
ResolutionStatus = Literal[
    "single", "ambiguous", "not_10k_filer", "no_10k_on_file", "not_found", "not_listed", "not_us_exchange",
    "financial_company",
]


class Fact(BaseModel):
    """One number used anywhere in the memo. Estimates are never written as reported facts."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    value: float | int | str
    unit: str
    label: str = Field(min_length=1)
    kind: Kind
    source: str          # provider plus URL or document id
    as_of: str           # period or date the value describes, e.g. "FY2025" or "2026-10-23"
    retrieved_at: datetime
    note: str = ""

    @field_validator("source", "as_of")
    @classmethod
    def not_blank(cls, v: str, info) -> str:
        if not v.strip():
            raise ValueError(f"{info.field_name} must not be blank: a fact without it cannot be traced")
        return v

    @field_validator("retrieved_at")
    @classmethod
    def timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("retrieved_at must carry a timezone")
        return v


class DataGap(BaseModel):
    """Something missing, kept apart from the facts. Missing data is never replaced by a plug number."""

    model_config = ConfigDict(extra="forbid")

    area: str
    missing: str
    reason: str
    effect: str          # effect on the memo

    @property
    def key(self) -> tuple[str, str]:
        return (self.area, self.missing)


class CompanyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    name: str
    cik: int
    exchange: str
    market_cap: float | None = None
    sic: str | None = None            # EDGAR SIC code as text; None when EDGAR gives none
    sic_description: str | None = None
    filer_note: str | None = None     # set when the 10-K filings are read from another CIK (a predecessor filer)


class ResolutionResult(BaseModel):
    """Outcome of resolve_company. `status` is the class of the outcome."""

    model_config = ConfigDict(extra="forbid")

    status: ResolutionStatus
    candidates: list[CompanyCandidate] = []
    message: str

    @model_validator(mode="after")
    def candidates_match_status(self) -> "ResolutionResult":
        n = len(self.candidates)
        if self.status == "single" and n != 1:
            raise ValueError("status 'single' needs exactly one candidate")
        if self.status == "ambiguous" and n < 2:
            raise ValueError("status 'ambiguous' needs two or more candidates")
        if self.status == "not_found" and n != 0:
            raise ValueError("status 'not_found' has no candidates")
        return self


# ---- graph state -------------------------------------------------------------------------
# State holds ids, file paths and small dicts only, never DataFrames. Facts live in the ledger file
# (state["ledger_path"]); the state keeps only counters. Parallel nodes write to gaps, collectors,
# fact_counts and node_log, so those have reducers.

def merge_gaps(left: list[dict], right: list[dict]) -> list[dict]:
    merged = {(g["area"], g["missing"]): g for g in (left or [])}
    merged.update({(g["area"], g["missing"]): g for g in (right or [])})
    return list(merged.values())


def merge_dicts(left: dict, right: dict) -> dict:
    return {**(left or {}), **(right or {})}


class GraphState(TypedDict, total=False):
    thread_id: str
    user_input: str
    profile: str
    model_ids: dict[str, str]
    resolution: dict                                   # ResolutionResult.model_dump()
    company: dict                                      # chosen CompanyCandidate.model_dump()
    ledger_path: str                                   # .data/runs/<thread_id>/ledger.json
    fact_counts: Annotated[dict[str, int], merge_dicts]  # owner (collector or node) -> facts it wrote
    coverage: Annotated[dict[str, dict], merge_dicts]  # area -> {"fetched": n, "expected": n}
    extract_summary: dict                              # counts of extracted and dropped items; the items are in extracts.json
    sentiment_summary: dict                            # aggregate of the article scores; the scores are ledger facts
    news_summary: dict                                 # counts, drop reasons and the path of the article records file
    sections: Annotated[dict[str, dict], merge_dicts]  # section key -> {"path", "status", "chars"}; the text is in the file
    gaps: Annotated[list[dict], merge_gaps]            # DataGap.model_dump()
    collectors: Annotated[dict[str, str], merge_dicts]  # collector name -> "ok" or an error message
    abort_reason: str                                  # set by data_gate; empty when the run continues
    assessment: dict
    valuation: dict
    memo_path: str                                     # work file under .data/runs/<thread_id>/
    check_failures: list[str]
    review_round: int
    review_notice: str                                 # shown at the next review: a rejected value, or a confirmation that is needed
    assumption_overrides: dict                         # reviewer values for the four editable assumptions (keys as in config.yaml)
    feedback: dict                                     # last Request-changes payload
    approved: bool
    export_dir: str
    export_message: str                                # what export wrote, or which file is missing
    node_log: Annotated[list[str], operator.add]       # one entry per node run, for the kill-and-resume evidence
