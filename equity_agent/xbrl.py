"""Reported figures from SEC XBRL `companyfacts` (one request per company, all tags).

Tag fallbacks: every tag is tried and the one with the most recent fiscal year
wins, because a filer may stop using a tag (Apple moved off "Revenues"). Fiscal-year discrete periods only.
"""
from dataclasses import dataclass, field
from datetime import date, datetime

from .schemas import Fact

REVENUE_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"]
NET_INCOME_TAGS = ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"]
EPS_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"]
INTEREST_TAGS = ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"]
# Total debt is built from parts that must all describe the same period end: long-term debt (noncurrent) plus current debt
NONCURRENT_DEBT_TAGS = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"]
CURRENT_DEBT_TAGS = ["DebtCurrent"]                                      # already includes current long-term debt and commercial paper
CURRENT_DEBT_PARTS = ["LongTermDebtCurrent", "ShortTermBorrowings", "CommercialPaper"]   # used when DebtCurrent is absent

FLOW = {"revenue": ("USD", REVENUE_TAGS), "net_income": ("USD", NET_INCOME_TAGS),
        "diluted_eps": ("USD/shares", EPS_TAGS), "interest_expense": ("USD", INTEREST_TAGS)}
LABELS = {"revenue": "Revenue", "net_income": "Net income", "diluted_eps": "Diluted EPS",
          "interest_expense": "Interest expense", "total_debt": "Total debt"}
HISTORY_YEARS = 4


@dataclass(frozen=True)
class XbrlValue:
    metric: str
    value: float
    unit: str
    end: str
    accession: str
    form: str
    filed: str
    kind: str                 # "reported" for one tag, "derived" for a sum of tags
    tags: tuple[str, ...]
    note: str


@dataclass
class XbrlFigures:
    entity: str = ""
    values: dict[str, dict[str, XbrlValue]] = field(default_factory=dict)   # metric -> period end -> value

    def get(self, metric: str, end: str) -> XbrlValue | None:
        return self.values.get(metric, {}).get(end)


def fact_id(metric: str, end: str) -> str:
    return f"filings.xbrl.{metric}.annual.{end}"


def _series(gaap: dict, tag: str, unit: str, duration: bool) -> dict[str, dict]:
    """period end -> the newest 10-K entry for that period. Duration facts must span one fiscal year."""
    out: dict[str, dict] = {}
    for e in gaap.get(tag, {}).get("units", {}).get(unit, []):
        if e.get("fp") != "FY" or not str(e.get("form", "")).startswith("10-K"):
            continue
        if duration:
            if "start" not in e or not 350 <= (date.fromisoformat(e["end"]) - date.fromisoformat(e["start"])).days <= 380:
                continue
        if e["end"] not in out or e["filed"] > out[e["end"]]["filed"]:
            out[e["end"]] = e
    return out


def _best_tag(gaap: dict, tags: list[str], unit: str, duration: bool) -> tuple[str, dict[str, dict]] | None:
    best = None
    for tag in tags:
        series = _series(gaap, tag, unit, duration)
        if series and (best is None or max(series) > max(best[1])):
            best = (tag, series)
    return best


def extract(facts_json: dict) -> XbrlFigures:
    gaap = facts_json.get("facts", {}).get("us-gaap", {})
    fig = XbrlFigures(entity=facts_json.get("entityName", ""))
    for metric, (unit, tags) in FLOW.items():
        best = _best_tag(gaap, tags, unit, duration=True)
        if best:
            tag, series = best
            fig.values[metric] = {
                end: XbrlValue(metric, e["val"], unit, end, e["accn"], e["form"], e["filed"], "reported", (tag,),
                               f"us-gaap:{tag}; period {e['start']} to {end}; accession {e['accn']}; filed {e['filed']}")
                for end, e in series.items()}
    fig.values["total_debt"] = _total_debt(gaap)
    return fig


def _total_debt(gaap: dict) -> dict[str, XbrlValue]:
    noncurrent = _best_tag(gaap, NONCURRENT_DEBT_TAGS, "USD", duration=False)
    if not noncurrent:
        return {}
    nc_tag, nc_series = noncurrent
    current_total = _best_tag(gaap, CURRENT_DEBT_TAGS, "USD", duration=False)
    parts = {t: _series(gaap, t, "USD", duration=False) for t in CURRENT_DEBT_PARTS}
    out: dict[str, XbrlValue] = {}
    for end, nc in nc_series.items():
        used, total = [nc_tag], nc["val"]
        if current_total and end in current_total[1]:
            used.append(current_total[0])
            total += current_total[1][end]["val"]
        else:
            found = [(t, s[end]["val"]) for t, s in parts.items() if end in s]
            if not found:
                continue       # no current-debt figure for this period end: do not present noncurrent debt as total debt
            used += [t for t, _ in found]
            total += sum(v for _, v in found)
        out[end] = XbrlValue("total_debt", total, "USD", end, nc["accn"], nc["form"], nc["filed"], "derived", tuple(used),
                             f"sum of us-gaap:{' + us-gaap:'.join(used)} at {end}; accession {nc['accn']}; filed {nc['filed']}")
    return out


def fact_from(v: XbrlValue, entity: str, retrieved_at: datetime, fid: str) -> Fact:
    return Fact(id=fid, value=v.value, unit=v.unit, label=f"{LABELS[v.metric]} ({entity}, XBRL)", kind=v.kind,
                source=f"SEC EDGAR XBRL companyfacts, accession {v.accession} ({v.form})", as_of=f"FY ended {v.end}",
                retrieved_at=retrieved_at, note=v.note)


def reported_facts(fig: XbrlFigures, retrieved_at: datetime) -> list[Fact]:
    """Revenue, net income and diluted EPS for the latest four fiscal years, with tag and accession in the note."""
    facts = []
    for metric in ("revenue", "net_income", "diluted_eps"):
        by_end = fig.values.get(metric, {})
        for end in sorted(by_end, reverse=True)[:HISTORY_YEARS]:
            facts.append(fact_from(by_end[end], fig.entity, retrieved_at, fact_id(metric, end)))
    return facts


def match_end(by_end: dict, end: str, days: int = 7) -> str | None:
    """The XBRL period end that matches a yfinance period end: the same date, or within a week.

    Week-based fiscal years end on a Saturday or Sunday while yfinance often shows a month end (Apple FY2022: SEC 2022-09-24,
    yfinance 2022-09-30, six days apart). Annual periods are a year apart, so a week cannot match the wrong year.
    """
    d = date.fromisoformat(end)
    close = [e for e in by_end if abs((date.fromisoformat(e) - d).days) <= days]
    return min(close, key=lambda e: abs((date.fromisoformat(e) - d).days)) if close else None
