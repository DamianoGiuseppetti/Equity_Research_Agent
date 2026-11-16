"""Draft the ten memo sections. One writer call per narrative section; each call gets only the facts that section needs.

Tables and the Sources and Methodology section come from memo_tables.py. The Executive Summary is written last, from the
finished sections. The Conclusion's rating line is put in by code from the rule-based rating.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import memo_tables as mt
from .ledger import Ledger
from .llm import LLMClient
from .numbers import display
from .valuation_facts import LABEL

SECTIONS = ("Executive Summary", "Company Overview", "Industry and Competitive Analysis", "Historical Financial Analysis",
            "Investment Thesis", "Financial Outlook", "Valuation", "Risks and Catalysts", "Conclusion", "Sources and Methodology")
WRITTEN_BY_MODEL = tuple(s for s in SECTIONS if s != "Sources and Methodology")
DRAFT_ORDER = tuple(s for s in WRITTEN_BY_MODEL if s != "Executive Summary") + ("Executive Summary",)    # the summary is written last

SYSTEM = """You write one section of an equity research memo. The memo is a draft for a human reviewer. Rules:
1. Use only the FACTS, TEXT ITEMS and FINDINGS below. No outside knowledge. No investment advice, no price targets, no predictions \
of the share price. {rating_rule}
2. After every number write the id of its fact in square brackets, right after the number, copied exactly: \
"net margin of 8.1% [market.ratio.net_margin]". Write each number the way the fact's value is shown in the list \
("$332.2 billion", "8.1%", "1.4x", "$347.73"). Never compute a number of your own and never write a number that is not in the list. \
Years, fiscal-year labels and dates need no id. Every number gets its own id right after it, also when two numbers are in one \
sentence: "rose to 61.6% [id1] from 61.1% [id2]". Never write "up from 61.1% in 2024 as stated in the filing" without an id after 61.1%. \
Do not use digits for counts or ranks ("the top-ranked risk", "three segments", "rank one"). If you cannot cite a number, leave it out. \
Before you finish, re-read the text and check that every number is followed by an id.
3. Facts tagged ESTIMATE or ASSUMPTION must be worded as estimates in the same sentence ("we estimate", "the model projects", \
"assuming", "the scenario implies"). Sentiment scores and the risk ranking are model estimates too.
4. GAPS TO MENTION: for each gap listed there, write one sentence saying what is missing and what that means for this section, and \
put its tag right after it, written exactly as shown in front of the gap (nothing else inside the brackets). Use only tags that are \
listed; if none is listed, write no gap tag and do not invent limitations about missing data. The numbers inside a gap description are \
not facts: describe them in words ("a small difference") and never quote them.
5. Plain, concise prose for a professional reader; short bullet lists are fine. No heading (it is added) and no tables (code adds them). \
Do not quote news headlines.
6. {length}
"""
NO_RATING = "Do not state a rating or a valuation-implied view; the Conclusion carries it."


@dataclass(frozen=True)
class Spec:
    purpose: str
    prefixes: tuple[str, ...]            # ledger id prefixes this section may use
    areas: tuple[str, ...] = ()          # Assessment areas whose findings are given
    length: str = "About 250 to 400 words."
    extras: tuple[str, ...] = ()         # extra material: "risks_conclusion"


ANNUAL = ("market.income.", "market.cashflow.", "market.balance.", "filings.xbrl.", "market.xbrl.")
SPECS: dict[str, Spec] = {
    "Company Overview": Spec("Describe what the company does: its segments, where it operates, its size (market capitalization, price) and its sector and industry.",
                             ("market.info.", "market.price.last_close", "extract.segment.", "extract.geography."), (), "About 200 to 350 words."),
    "Industry and Competitive Analysis": Spec("Describe the industry the company is in, the named competitors and how the filing frames competition.",
                                              ("market.info.sector", "market.info.industry", "extract.competitor.", "extract.segment."), (), "About 200 to 350 words."),
    "Historical Financial Analysis": Spec("Analyse the last fiscal years: revenue, margins, earnings, cash flow, debt and liquidity, and what management says drove results. "
                                          "A financial history table and a ratio table are added after your text, so do not repeat them in full.",
                                          (*ANNUAL, "market.ratio.", "extract.driver.", "extract.liquidity."), ("financial", "technical"), "About 350 to 500 words."),
    "Investment Thesis": Spec("State the investment case in balanced terms: the strongest positive and negative points from the findings, management's drivers and guidance, "
                              "news sentiment (a model estimate) and macro context. No rating.",
                              ("extract.driver.", "extract.guidance.", "news.sentiment.", "macro.", "valuation.dcf.base.value_per_share", "market.price.last_close"),
                              ("financial", "technical", "macro", "news"), "About 300 to 450 words."),
    "Financial Outlook": Spec("Describe what a simple extrapolation of history implies for the next five years (bear, base, bull) and what management guided. "
                              "Everything projected is an estimate. A projection table is added after your text.",
                              ("valuation.dcf.", "valuation.input.", "valuation.assump.", "extract.guidance.", "market.ratio.revenue_growth"), (), "About 250 to 400 words."),
    "Valuation": Spec("Explain the valuation: the DCF scenarios and their assumptions (WACC, terminal growth, growth path, margins), the peer multiples and what they imply, "
                      "and how the last close compares. Tables of scenarios, the sensitivity grid and peer multiples are added after your text.",
                      ("valuation.", "market.price.last_close", "market.info.beta", "macro.dgs10."), (), "About 350 to 500 words."),
    "Risks and Catalysts": Spec("Discuss the main risks (ranked by the model, which is an estimate) and the catalysts. A risk table is added after your text, so discuss the top risks "
                                "by theme rather than repeating every one.", ("extract.risk.", "extract.catalyst.", "macro.vixcls.", "macro.bamlh0a0hym2."), ("financial", "macro", "news"),
                                "About 300 to 450 words."),
    "Conclusion": Spec("Conclude: summarise the case and give 3 to 5 conditions that would invalidate the thesis, drawn from the negative findings, the ranked risks and the limitations. "
                       "The valuation-implied view appears directly above your text; refer to it as 'the view above' but do not restate or change it.",
                       ("valuation.dcf.", "valuation.comps.range_", "market.price.last_close", "extract.risk."), ("financial", "technical", "macro", "news"), "About 200 to 350 words.",
                       extras=("rating",)),
    "Executive Summary": Spec("Summarise the memo in one short section: what the company is, the key financial facts, the main positives and negatives, the valuation-implied view "
                              "and its basis, and that this is a draft for review. Reuse numbers only exactly as they appear in the finished sections, with their ids.",
                              (), (), "About 150 to 220 words."),
}


@dataclass
class Context:
    company: dict
    ledger: Ledger
    assessment: dict            # {"financial": {"findings": [...], "limitations": [...]}, ..., "cross_area_notes": [...]}
    gaps: list[dict]
    gap_index: dict[str, dict]  # tag G1.. -> gap
    valuation: dict             # state["valuation"] summary: rating, basis
    drafted: dict[str, str] = field(default_factory=dict)       # finished narratives, for the Executive Summary
    allowed: dict[str, set[str]] = field(default_factory=dict)  # ids each finished section was given


def tag_gaps(gaps: list[dict]) -> dict[str, dict]:
    return {f"G{i}": g for i, g in enumerate(gaps, start=1)}


def relevant_gaps(section: str, gap_index: dict[str, dict]) -> dict[str, dict]:
    """Which gaps a section must mention. Rules by gap area and by words in what is missing."""
    def pick(g: dict) -> bool:
        area, miss = g["area"], (g["missing"] + " " + g["reason"]).lower()
        filings = area == "filings"
        extraction = filings and any(k in miss for k in ("extract", "llm"))
        if section == "Company Overview":
            return area == "company" or extraction or (filings and "item 1 (" in miss)
        if section == "Industry and Competitive Analysis":
            return extraction or (filings and "item 1 (" in miss)
        if section == "Historical Financial Analysis":
            return area == "market data" or (filings and any(k in miss for k in ("xbrl", "item 7"))) or extraction
        if section == "Investment Thesis":
            return area in ("analysis", "news")
        if section == "Financial Outlook":
            return (filings and "item 7" in miss) or (area == "valuation" and "dcf" in miss)
        if section == "Valuation":
            return area == "valuation" or (area == "macro" and "dgs10" in miss)
        if section == "Risks and Catalysts":
            return extraction or (filings and "item 1a" in miss)
        if section == "Conclusion":
            return area == "valuation" and "dcf" in miss
        return False
    return {t: g for t, g in gap_index.items() if pick(g)}


def _fact_line(f) -> str:
    tag = " [ESTIMATE]" if f.kind == "model_estimate" else " [ASSUMPTION]" if f.kind == "assumption" else ""
    return f"{f.id} | {f.label} | {display(f)} | {f.kind}{tag} | {f.as_of}"


def _findings(ctx: Context, areas: tuple[str, ...]) -> tuple[str, set[str]]:
    lines, ids = [], set()
    for a in areas:
        for f in ctx.assessment[a]["findings"]:
            lines.append(f"- ({a}, {f['direction']}) {f['statement']}  [facts: {', '.join(f['ledger_ids'])}]")
            ids.update(f["ledger_ids"])
        for lim in ctx.assessment[a]["limitations"]:
            lines.append(f"- ({a}, limitation) {lim}")
    return "\n".join(lines), ids


def build_prompt(name: str, ctx: Context) -> tuple[str, str, set[str], dict[str, dict]]:
    """(system, human, allowed ids, gaps to mention) for one section."""
    spec = SPECS[name]
    gaps = relevant_gaps(name, ctx.gap_index)
    rating_rule = NO_RATING if name != "Conclusion" else "The valuation-implied view is added by code; do not state or alter it."
    system = SYSTEM.format(length=spec.length, rating_rule=rating_rule)
    parts = [f"Company: {ctx.company['name']} ({ctx.company['ticker']})", f"SECTION: {name}", f"PURPOSE: {spec.purpose}"]
    if name == "Executive Summary":
        allowed = set().union(*ctx.allowed.values()) if ctx.allowed else set()
        parts.append("FINISHED SECTIONS (reuse numbers only as written here, with their ids):")
        parts += [f"### {n}\n{ctx.drafted[n]}" for n in WRITTEN_BY_MODEL if n in ctx.drafted]
        parts.append(f"VALUATION-IMPLIED VIEW: {ctx.valuation.get('rating') or 'not available'} (basis: {ctx.valuation.get('basis') or 'none'}); it is {LABEL}.")
        return system, "\n\n".join(parts), allowed, {}
    findings_text, finding_ids = _findings(ctx, spec.areas) if spec.areas else ("", set())
    facts = [f for f in ctx.ledger.facts if f.id.startswith(spec.prefixes) or f.id in finding_ids]
    if name in ("Historical Financial Analysis", "Valuation", "Financial Outlook"):    # keep the prompt small: skip quarterly rows and per-year projections already in tables
        facts = [f for f in facts if f.id in finding_ids or (".quarterly." not in f.id and not any(k in f.id for k in (".growth_y", ".revenue_y", ".fcf_y", "valuation.grid.", "valuation.comps.peer.")))
                 or (name == "Financial Outlook" and f.id.endswith(("revenue_y5", "fcf_y5", "growth_y1")))]
    allowed = {f.id for f in facts}
    parts.append("FACTS (id | label | value | kind | as of):\n" + "\n".join(_fact_line(f) for f in facts))
    if findings_text:
        parts.append("FINDINGS from the analysis step (restate in your own words; cite numbers from FACTS):\n" + findings_text)
    if not gaps:
        parts.append("GAPS TO MENTION: none. Write no gap tags.")
    else:
        parts.append("GAPS TO MENTION (tag: what is missing, reason, effect):\n" + "\n".join(
            f"[gap:{t}] {g['missing']}: {g['reason']}. Effect: {g['effect']}" for t, g in gaps.items()))
    return system, "\n\n".join(parts), allowed, gaps


@dataclass
class Drafted:
    text: str
    allowed: set[str]
    gap_tags: dict[str, str]     # tag -> short description, for the check


async def draft_section(client: LLMClient, name: str, ctx: Context, previous: str = "", problems: list[str] | None = None,
                        instruction: str = "") -> Drafted:
    system, human, allowed, gaps = build_prompt(name, ctx)
    if instruction:      # a reviewer's request: it can change wording and emphasis, never a number
        human += ("\n\nREVIEWER INSTRUCTION: " + instruction + "\nFollow it for wording and emphasis only. You cannot change a number or add one that is not in the list; "
                  "if the instruction asks for a different number, say in the text that this cannot be done.")
    if problems:      # redraft: show the previous text and what failed
        human += ("\n\nYOUR PREVIOUS DRAFT:\n" + previous + "\n\nIT FAILED THESE CHECKS:\n" + "\n".join(f"- {p}" for p in problems) +
                  "\n\nWrite the whole section again and fix every problem. Keep to the rules.")
    text = await client.text("writer", [("system", system), ("human", human)], f"draft.{name}" + (".redraft" if problems else ""), max_tokens=4000)
    first, _, rest = text.partition("\n")
    if first.strip(" #*:").lower() == name.lower():      # the model repeats the heading although told not to: the heading is added by code
        text = rest.strip()
    return Drafted(text, allowed, {t: f"{g['area']}: {g['missing']}" for t, g in gaps.items()})


# ---- code-built parts of each section ---------------------------------------------------------------
def lead_and_tables(name: str, ctx: Context, state: dict) -> tuple[str, str, set[str]]:
    """(lead text placed above the narrative, tables placed below it, ledger ids they use)."""
    ledger, gaps = ctx.ledger, ctx.gaps
    if name == "Historical Financial Analysis":
        t, used = mt.financial_tables(ledger)
        return "", t, used
    if name == "Financial Outlook":
        t, used = mt.outlook_table(ledger)
        return "", t, used
    if name == "Valuation":
        parts = [mt.scenario_table(ledger), mt.grid_table(ledger), mt.peer_table(ledger, gaps)]
        return "", "\n\n".join(p[0] for p in parts if p[0]), set().union(*[p[1] for p in parts])
    if name == "Risks and Catalysts":
        t, used = mt.risk_table(ledger)
        return "", t, used
    if name == "Conclusion":
        rating = ctx.valuation.get("rating")
        if rating:
            lead = (f"**Valuation-implied view: {rating}.** Basis: {ctx.valuation['basis']}. This is a {LABEL}. [valuation.rating]")
            return lead, "", {"valuation.rating"}
        return "**No valuation-implied view could be computed** (see the data gaps in Sources and Methodology). This is not investment advice.", "", set()
    if name == "Executive Summary":
        cov = "; ".join(f"{a} {c['fetched']} of {c['expected']}" for a, c in sorted(state.get("coverage", {}).items()))
        return "", f"*Data coverage (items fetched of items expected): {cov}. {len(ctx.gaps)} data gaps are listed in Sources and Methodology.*", set()
    return "", "", set()
