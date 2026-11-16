"""LLM extraction from 10-K sections: eight schemas, mechanical checks, chunking, ledger facts.

The model proposes; code decides. Every item must quote the filing (source_excerpt is checked as a verbatim substring)
and every number in its other fields must appear in that excerpt. An item that fails is retried once with the reason,
then dropped and logged. It is never repaired by editing.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .llm import LLMClient, LLMOutputError
from .schemas import DataGap, Fact

# ---- the eight schemas ---------------------------------------------------------------------------
class _Item(BaseModel):
    source_excerpt: str = Field(min_length=20, description="One to three sentences copied exactly, character for character, from the text")
    confidence: float = Field(ge=0, le=1, description="0 to 1: how sure you are the excerpt supports this item")


class BusinessSegment(_Item):
    name: str
    description: str
    amount: str | None = Field(None, description="Revenue share or amount exactly as written in the excerpt, or null if not stated")


class GeographicExposure(_Item):
    region: str
    amount: str | None = Field(None, description="Share or amount exactly as written in the excerpt, or null if not stated")


class CompetitorMention(_Item):
    name: str
    context: str


class PerformanceDriver(_Item):
    topic: str
    direction: Literal["positive", "negative", "neutral"] = Field(description="Effect on the company's results")
    quantified_effect: str | None = Field(None, description="The effect exactly as written in the excerpt, or null if not stated")


class Guidance(_Item):
    metric: str
    period: str
    statement: str


class LiquidityNote(_Item):
    kind: Literal["debt", "buyback", "dividend", "capex"]
    statement: str
    amount: str | None = Field(None, description="Amount exactly as written in the excerpt, or null if not stated")


class RiskItem(_Item):
    title: str
    category: Literal["macro", "industry", "operational", "regulatory", "legal", "financial", "technology", "other"]
    summary: str
    rank: int = Field(description="1 is the most severe. Ranks are unique and run from 1 to the number of risks returned")


class CatalystMention(_Item):
    event: str
    expected_timing: str | None = Field(None, description="Timing exactly as written in the excerpt, or null if not stated")


class BusinessFamily(BaseModel):
    segments: list[BusinessSegment] = []
    geographies: list[GeographicExposure] = []
    competitors: list[CompetitorMention] = []
    catalysts: list[CatalystMention] = []


class MdnaFamily(BaseModel):
    segments: list[BusinessSegment] = []
    geographies: list[GeographicExposure] = []
    drivers: list[PerformanceDriver] = []
    guidance: list[Guidance] = []
    liquidity: list[LiquidityNote] = []
    catalysts: list[CatalystMention] = []


class RiskFamily(BaseModel):
    risks: list[RiskItem] = []


# section key -> (family model, what to look for). One call per family per section (and per chunk).
FAMILIES: dict[str, tuple[type[BaseModel], str]] = {
    "item1": (BusinessFamily, "segments (at most 8), geographic regions (at most 8), named competitors (at most 10), and "
                              "catalysts: events the text says are expected or planned (at most 6)."),
    "item7": (MdnaFamily, "segments, geographic regions, performance drivers (what the text says moved results, with direction), "
                          "forward-looking guidance, liquidity notes (debt, buyback, dividend, capex) and catalysts. "
                          "At most 8 segments, 8 regions, 10 drivers, 6 guidance statements, 8 liquidity notes, 6 catalysts."),
    "item1a": (RiskFamily, "the 8 most severe risk factors, ranked: rank 1 is the most severe. Ranks must be unique and run from 1 "
                           "to the number of risks returned. Return fewer than 8 if fewer stand out."),
}
SECTION_LABEL = {"item1": "Item 1 (Business)", "item1a": "Item 1A (Risk Factors)", "item7": "Item 7 (MD&A)"}
# family field -> (schema key used in ledger ids, memo label). The ranking is a model estimate.
FIELD_KEYS = {"segments": "segment", "geographies": "geography", "competitors": "competitor", "drivers": "driver",
              "guidance": "guidance", "liquidity": "liquidity", "risks": "risk", "catalysts": "catalyst"}
SYSTEM = (
    "You extract facts from one section of a company's SEC Form 10-K. Rules:\n"
    "1. Extract only what the text states. Use no outside knowledge. If the text has nothing for a list, return it empty.\n"
    "2. source_excerpt must be copied from the text exactly, character for character, one to three sentences. "
    "It is checked mechanically against the text.\n"
    "3. Every number you write in any other field (amounts, percentages, years) must also appear in your source_excerpt, "
    "copied exactly as written there. If the excerpt has no number, write no number.\n"
    "4. Write names and descriptions in plain words and add nothing the excerpt does not say.\n"
    "5. Do not extract the same passage twice.\n"
    "Find: {task}"
)


# ---- mechanical checks ---------------------------------------------------------------------------
_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‑": "-", "\xa0": " "})
_ZERO_WIDTH = re.compile("[​‌‍﻿]")


def normalise(text: str) -> str:
    """Curly quotes and dashes made plain, zero-width characters removed and whitespace collapsed.

    Whitespace is collapsed because edgartools keeps paragraph breaks that a quoted sentence never has.
    """
    return re.sub(r"\s+", " ", _ZERO_WIDTH.sub("", text.translate(_PUNCT))).strip()


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Numbers as plain strings: '1,234.50' -> '1234.50'. A trailing sentence period is not part of the number."""
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def check_item(item: _Item, section_text_normalised: str) -> str | None:
    """None if the item passes, otherwise the reason it failed."""
    if normalise(item.source_excerpt) not in section_text_normalised:
        return "source_excerpt is not a verbatim substring of the section text"
    excerpt_numbers = numbers_in(item.source_excerpt)
    for name, value in item.model_dump(exclude={"source_excerpt", "confidence", "rank"}).items():
        missing = numbers_in(value) - excerpt_numbers if isinstance(value, str) else set()
        if missing:
            return f"number(s) {sorted(missing)} in field '{name}' do not appear in the source_excerpt"
    return None


def risk_ranks_valid(risks: list[RiskItem]) -> str | None:
    """Ranks must be exactly 1..n for the n risks returned: no duplicates, none out of range, none missing."""
    ranks = sorted(r.rank for r in risks)
    if ranks != list(range(1, len(risks) + 1)):
        return f"risk ranks {ranks} are not unique numbers from 1 to {len(risks)}"
    if len(risks) > 8:
        return f"{len(risks)} risks returned, the maximum is 8"
    return None


# ---- chunking ------------------------------------------------------------------------------------
def _heading_offsets(text: str, strong: bool) -> list[int]:
    """Character offsets of sub-heading lines: short, alone between blank lines, no closing punctuation.
    `strong` keeps only ALL-CAPS headings (the major parts of an MD&A)."""
    lines, offsets, pos = text.split("\n"), [], 0
    for i, line in enumerate(lines):
        s = line.strip()
        alone = (i == 0 or not lines[i - 1].strip()) and (i + 1 >= len(lines) or not lines[i + 1].strip())
        letters = "".join(c for c in s if c.isalpha())
        if (s and len(s) < 80 and alone and len(letters) >= 6 and not s.endswith((".", ",", ";", ":")) and not s.startswith("•")
                and "table of contents" not in s.lower() and (letters.isupper() or not strong)):
            offsets.append(pos)
        pos += len(line) + 1
    return offsets


def chunk_text(text: str, cap_chars: int, level: int = 0) -> list[str]:
    """Split at sub-headings into chunks of at most cap_chars. Major (ALL-CAPS) headings first, then any heading,
    then paragraph breaks, then a hard cut. Consecutive pieces are packed together while they fit."""
    if len(text) <= cap_chars:
        return [text]
    if level == 3:
        return [text[i:i + cap_chars] for i in range(0, len(text), cap_chars)]
    cuts = _heading_offsets(text, strong=level == 0) if level < 2 else [m.start() for m in re.finditer(r"\n\n", text)]
    bounds = sorted({0, *cuts, len(text)})
    pieces = [text[a:b] for a, b in zip(bounds, bounds[1:])]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > cap_chars:
            chunks.append(current)
            current = ""
        current += piece
    chunks.append(current)
    out: list[str] = []
    for c in chunks:
        out += chunk_text(c, cap_chars, level + 1) if len(c) > cap_chars else [c]
    return out


# ---- amounts -> numbers --------------------------------------------------------------------------
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def parse_amount(text: str | None) -> tuple[float, str] | None:
    """'$47.9 billion' -> (47.9e9, 'USD'); '25%' -> (0.25, 'fraction'). None if the text is not exactly one such amount."""
    if not text:
        return None
    found = _NUMBER.findall(text)
    if len(found) != 1:
        return None
    value = float(found[0].replace(",", ""))
    low = text.lower()
    scale = next((m for w, m in _SCALE.items() if re.search(rf"\b{w}\b", low)), 1.0)
    if "%" in text or "percent" in low:
        return value / 100, "fraction"
    if "$" in text or "usd" in low or "dollar" in low:
        return value * scale, "USD"
    return None


# ---- running the extraction ----------------------------------------------------------------------
@dataclass
class Filing:
    company: str
    ticker: str
    accession: str
    period: str
    retrieved_at: datetime


@dataclass
class Extraction:
    items: list[dict] = field(default_factory=list)     # surviving items: schema, section, fields, excerpt, confidence
    dropped: list[dict] = field(default_factory=list)   # every item that failed a check, with the reason
    failed_calls: list[str] = field(default_factory=list)
    n_calls: int = 0


def _evaluate(family: BaseModel, text_norm: str) -> tuple[list[tuple[str, BaseModel]], list[dict]]:
    """(kept items as (field, item)) and the failures, for one family result."""
    kept, failures = [], []
    for fname in type(family).model_fields:
        items = getattr(family, fname)
        rank_problem = risk_ranks_valid(items) if fname == "risks" and items else None
        for item in items:
            reason = rank_problem or check_item(item, text_norm)
            if reason:
                failures.append({"schema": FIELD_KEYS[fname], "reason": reason, "excerpt": item.source_excerpt[:160]})
            else:
                kept.append((fname, item))
    return kept, failures


def _retry_note(failures: list[dict]) -> str:
    lines = "\n".join(f"- {f['schema']}: {f['reason']} (excerpt began: {f['excerpt'][:80]!r})" for f in failures)
    return ("These items from your first answer failed the mechanical checks:\n" + lines +
            "\nAnswer again for the whole task. Copy excerpts exactly and keep numbers inside the excerpt. "
            "Leave out an item you cannot support exactly.")


async def extract_section(client: LLMClient, filing: Filing, section: str, text: str, cap_chars: int, out: Extraction) -> None:
    """All chunks of one section. A call that stays malformed after the wrapper's retry is logged, not fatal."""
    family_model, task = FAMILIES[section]
    chunks = chunk_text(text, cap_chars)
    for n, chunk in enumerate(chunks, 1):
        step = f"extract.{section}" + (f".chunk{n}of{len(chunks)}" if len(chunks) > 1 else "")
        head = (f"Company: {filing.company} ({filing.ticker}). Form 10-K, period {filing.period}, accession {filing.accession}. "
                f"Section: {SECTION_LABEL[section]}" + (f", part {n} of {len(chunks)}" if len(chunks) > 1 else "") + ".\n\nTEXT:\n")
        messages = [("system", SYSTEM.format(task=task)), ("human", head + chunk)]
        text_norm = normalise(chunk)
        try:
            first = await client.structured("extract", family_model, messages, step, max_tokens=8000)
            out.n_calls += 1
            kept, failures = _evaluate(first, text_norm)
            if failures:   # one retry, told exactly what failed
                for f in failures:
                    out.dropped.append({**f, "section": section, "stage": "first answer"})
                second = await client.structured("extract", family_model, messages + [("human", _retry_note(failures))],
                                                 step + ".retry", max_tokens=8000)
                out.n_calls += 1
                kept2, failures2 = _evaluate(second, text_norm)
                if section == "item1a":     # ranks only make sense as one list: keep the better list, never a mix
                    use_second = len(kept2) > len(kept) or (len(kept2) == len(kept) and len(failures2) < len(failures))
                    kept, final = (kept2, failures2) if use_second else (kept, failures)
                else:
                    seen = {normalise(i.source_excerpt) for _, i in kept}
                    kept += [(f, i) for f, i in kept2 if normalise(i.source_excerpt) not in seen]
                    final = failures2
                for f in final:
                    out.dropped.append({**f, "section": section, "stage": "after retry (dropped)"})
        except LLMOutputError as exc:
            out.failed_calls.append(f"{step}: {exc}")
            continue
        for fname, item in kept:
            out.items.append({"schema": FIELD_KEYS[fname], "section": section, "accession_number": filing.accession,
                              "fields": item.model_dump(exclude={"source_excerpt", "confidence"}),
                              "source_excerpt": item.source_excerpt, "confidence": item.confidence})


def _summary(item: dict) -> str:
    f, s = item["fields"], item["schema"]
    extra = lambda v: f" ({v})" if v else ""
    return {
        "segment": f"{f.get('name')}: {f.get('description')}" + extra(f.get("amount")),
        "geography": f"{f.get('region')}" + extra(f.get("amount")),
        "competitor": f"{f.get('name')}: {f.get('context')}",
        "driver": f"{f.get('topic')} ({f.get('direction')})" + extra(f.get("quantified_effect")),
        "guidance": f"{f.get('metric')}, {f.get('period')}: {f.get('statement')}",
        "liquidity": f"{f.get('kind')}: {f.get('statement')}" + extra(f.get("amount")),
        "risk": f"{f.get('title')} ({f.get('category')}): {f.get('summary')}",
        "catalyst": f"{f.get('event')}" + extra(f.get("expected_timing")),
    }[s]


AMOUNT_FIELD = {"segment": "amount", "geography": "amount", "driver": "quantified_effect", "liquidity": "amount"}


def to_facts(items: list[dict], filing: Filing) -> tuple[list[Fact], list[dict]]:
    """Ledger facts for the surviving items; returns (facts, items with their ledger id added).

    Each item becomes one text fact of kind `summary` (model-written text from the filing, not a figure) with the excerpt in `note`.
    A risk also gets a rank fact of kind `model_estimate`: the ranking is the model's. A stated amount that parses as one number becomes a second
    fact of kind `reported` with the numeric value, so the memo can cite the number itself.
    """
    facts, numbered, counts = [], [], {}
    for item in items:
        s = item["schema"]
        counts[s] = counts.get(s, 0) + 1
        fid = f"extract.{s}.{counts[s]}"
        src = f"SEC EDGAR filing {filing.accession} (10-K), {SECTION_LABEL[item['section']]}"
        note_excerpt = f'excerpt: "{item["source_excerpt"]}"; model confidence {item["confidence"]:.2f}'
        kind, note = "summary", f"summary of filing text written by the model, not a reported figure; {note_excerpt}"
        facts.append(Fact(id=fid, value=_summary(item), unit="text", label=f"{filing.ticker} 10-K {s} {counts[s]}", kind=kind,
                          source=src, as_of=filing.period, retrieved_at=filing.retrieved_at, note=note))
        if s == "risk":     # the ranking is the model's judgement: its own fact, kind model_estimate, so citing the risk text is not an estimate
            facts.append(Fact(id=f"{fid}.rank", value=item["fields"]["rank"], unit="rank", label=f"{filing.ticker} 10-K risk {counts[s]}: severity rank",
                              kind="model_estimate", source=src, as_of=filing.period, retrieved_at=filing.retrieved_at,
                              note="rank by severity assigned by the model among the risks it returned (1 is the most severe): a model estimate, not a reported ranking"))
        amount = item["fields"].get(AMOUNT_FIELD.get(s, ""), None)
        parsed = parse_amount(amount)
        if parsed:
            facts.append(Fact(id=f"{fid}.amount", value=parsed[0], unit=parsed[1], label=f"{filing.ticker} 10-K {s} {counts[s]}: stated amount",
                              kind="reported", source=src, as_of=filing.period, retrieved_at=filing.retrieved_at,
                              note=f'as written in the filing: "{amount}"; {note_excerpt}'))
        numbered.append({**item, "id": fid})
    return facts, numbered


def to_gaps(out: Extraction) -> list[DataGap]:
    gaps = []
    by_schema: dict[str, int] = {}
    for d in out.dropped:
        if d["stage"].startswith("after retry"):
            by_schema[d["schema"]] = by_schema.get(d["schema"], 0) + 1
    if by_schema:
        gaps.append(DataGap(area="filings", missing="extracted items that failed checks: " + ", ".join(f"{n} {s}" for s, n in sorted(by_schema.items())),
                            reason="source_excerpt not verbatim, a number missing from its excerpt, or invalid risk ranks, still failing after one retry",
                            effect="those items are left out of the memo; they are listed in extracts.json"))
    for call in out.failed_calls:
        gaps.append(DataGap(area="filings", missing=f"LLM extraction call {call.split(':')[0]}", reason=call[:300],
                            effect="nothing was extracted from that section part"))
    return gaps
