"""Turn one raw market-data pull into ledger facts, data gaps and a coverage count. Pure function: no I/O."""
from dataclasses import dataclass, field
from datetime import datetime

from . import xbrl as xb
from .metrics import Input, ratios, technicals
from .schemas import DataGap, Fact

AREA = "market data"
ANNUAL_YEARS = 4          # 4 years of annual statements; quarters: all that yfinance returns

# statement -> {yfinance row name: slug}. EPS gets its own unit below.
LINES: dict[str, dict[str, str]] = {
    "income": {"Total Revenue": "total_revenue", "Gross Profit": "gross_profit", "Operating Income": "operating_income",
               "Net Income": "net_income", "Diluted EPS": "diluted_eps", "EBITDA": "ebitda",
               "Interest Expense": "interest_expense", "Pretax Income": "pretax_income", "Tax Provision": "tax_provision"},
    "balance": {"Total Debt": "total_debt", "Cash Cash Equivalents And Short Term Investments": "cash_and_st_investments",
                "Cash And Cash Equivalents": "cash", "Stockholders Equity": "stockholders_equity",
                "Invested Capital": "invested_capital", "Total Assets": "total_assets",
                "Current Liabilities": "current_liabilities"},
    "cashflow": {"Operating Cash Flow": "operating_cash_flow", "Capital Expenditure": "capital_expenditure",
                 "Free Cash Flow": "free_cash_flow"},
}
# ratio input name -> (statement, slug), tried in order
RATIO_INPUTS = {
    "revenue": [("income", "total_revenue")], "gross_profit": [("income", "gross_profit")],
    "operating_income": [("income", "operating_income")], "net_income": [("income", "net_income")],
    "interest_expense": [("income", "interest_expense")], "tax_provision": [("income", "tax_provision")],
    "pretax_income": [("income", "pretax_income")],
    "total_debt": [("balance", "total_debt")],
    "cash": [("balance", "cash_and_st_investments"), ("balance", "cash")],
    "equity": [("balance", "stockholders_equity")], "invested_capital": [("balance", "invested_capital")],
    "operating_cash_flow": [("cashflow", "operating_cash_flow")], "capital_expenditure": [("cashflow", "capital_expenditure")],
}
INFO_FIELDS = [  # (yfinance key, slug, label, unit, effect of a gap)
    ("sector", "sector", "Sector", "text", "the macro collector cannot add sector-specific series"),
    ("industry", "industry", "Industry", "text", "the industry description in the memo is thinner"),
    ("sharesOutstanding", "shares_outstanding", "Shares outstanding", "shares", "value per share cannot be computed"),
    ("beta", "beta", "Equity beta", "ratio", "the cost of equity cannot use a market beta"),
    ("marketCap", "market_cap", "Market capitalization", "USD", "WACC weights and peer comparison lose the market value"),
    ("financialCurrency", "reporting_currency", "Reporting currency", "text", "the USD-only rule cannot be checked"),
    ("exchange", "exchange", "Exchange code", "text", "the listing venue is not shown"),
]


@dataclass
class MarketResult:
    facts: list[Fact] = field(default_factory=list)
    gaps: list[DataGap] = field(default_factory=list)
    fetched: int = 0


def stmt_fact_id(stmt: str, slug: str, freq: str, end: str) -> str:
    return f"market.{stmt}.{slug}.{freq}.{end}"


# yfinance income-statement line -> XBRL metric, compared when the period ends match
RECONCILED = {"total_revenue": "revenue", "net_income": "net_income", "diluted_eps": "diluted_eps"}
# ratio input -> (statement, slug, XBRL metric) for lines that fall back to XBRL when yfinance lacks them
FALLBACK = {"interest_expense": ("income", "interest_expense", "interest_expense"),
            "total_debt": ("balance", "total_debt", "total_debt")}


def build_market_facts(raw: dict, ticker: str, retrieved_at: datetime, figures: "xb.XbrlFigures | None" = None,
                       tolerance: float = 0.005, xbrl_error: str = "") -> MarketResult:
    """Facts, gaps and coverage from one yfinance pull.

    `figures` are the SEC XBRL numbers. Where a yfinance annual figure and the XBRL figure describe the same period
    and differ by more than `tolerance`, the yfinance fact is left out, the XBRL fact (written by the filings collector)
    stands, and one gap per metric says so. Interest expense and total debt are taken from XBRL only when yfinance
    lacks them.
    """
    res = MarketResult()
    src = f"yfinance {ticker} (Yahoo Finance)"

    def add(fid, value, unit, label, kind, source, as_of, note=""):
        res.facts.append(Fact(id=fid, value=value, unit=unit, label=label, kind=kind, source=source, as_of=as_of,
                              retrieved_at=retrieved_at, note=note))

    def gap(missing, reason, effect):
        res.gaps.append(DataGap(area=AREA, missing=missing, reason=reason, effect=effect))

    # ---- info
    info = raw.get("info", {})
    for key, slug, label, unit, effect in INFO_FIELDS:
        value = info.get(key)
        if value in (None, ""):
            gap(label, f"yfinance info has no '{key}'", effect)
            continue
        add(f"market.info.{slug}", value, unit, f"{label} ({ticker})", "reported", src, retrieved_at.date().isoformat())
        res.fetched += 1
    if info.get("currency"):
        add("market.info.price_currency", info["currency"], "text", f"Price currency ({ticker})", "reported", src,
            retrieved_at.date().isoformat())
    if info.get("financialCurrency") and info["financialCurrency"] != "USD":
        return res  # data_gate aborts on a non-USD reporting currency; nothing more is worth pulling

    # ---- prices and technicals
    prices = [p for p in raw.get("prices", []) if p.get("close") is not None]
    if prices:
        last = prices[-1]
        add("market.price.last_close", last["close"], "USD", f"Last close ({ticker})", "reported", src, last["date"],
            f"daily close; {len(prices)} daily bars from {prices[0]['date']}")
        res.fetched += 1
    else:
        gap("Daily prices", "yfinance returned no price history", "no technicals, no valuation; the run is stopped by the data gate")
    for ind in technicals(prices):
        if ind.value is None:
            gap(ind.label, ind.reason_missing, "this indicator is left out of the technical analysis")
            continue
        add(f"market.tech.{ind.slug}", ind.value, ind.unit, f"{ind.label} ({ticker})", "derived",
            "computed in code from yfinance daily prices", prices[-1]["date"], ind.formula)
        res.fetched += 1

    # ---- XBRL cross-check: which yfinance annual figures disagree with the reported figures
    stm = raw.get("statements", {})
    skip: set[tuple[str, str]] = set()                 # (slug, yfinance period end) left out of the ledger
    xbrl_end_for: dict[tuple[str, str], str] = {}      # (xbrl metric, yfinance period end) -> matching XBRL period end
    if xbrl_error:
        gap("XBRL cross-check", f"SEC XBRL could not be read: {xbrl_error}",
            "yfinance revenue, net income and EPS are used unchecked")
    mismatches: dict[str, list[str]] = {}
    for slug, metric in RECONCILED.items():
        for end, lines in stm.get("income_annual", {}).items():
            row = next(r for r, s_ in LINES["income"].items() if s_ == slug)
            ref_end = xb.match_end((figures.values.get(metric, {}) if figures else {}), end)
            if ref_end is None or lines.get(row) is None:
                continue
            ref = figures.values[metric][ref_end]
            xbrl_end_for[(metric, end)] = ref_end
            diff = abs(lines[row] - ref.value) / (abs(ref.value) if ref.value else 1.0)
            if diff > tolerance:
                skip.add((slug, end))
                mismatches.setdefault(metric, []).append(
                    f"FY ended {end}: yfinance {lines[row]:,.2f} vs XBRL {ref.value:,.2f} ({diff:.2%})")
    for metric, items in mismatches.items():
        gap(f"{xb.LABELS[metric]}: yfinance and XBRL agree within {tolerance:.1%}", "; ".join(items),
            "the XBRL value is used and the yfinance figure is not in the ledger")

    # ---- statements
    written: dict[tuple[str, str, str], tuple[float, str]] = {}   # (statement, slug, end) -> (value, fact id), annual only
    annual_ends: dict[str, list[str]] = {}
    for stmt, lines in LINES.items():
        for freq, limit in (("annual", ANNUAL_YEARS), ("quarterly", None)):
            periods = stm.get(f"{stmt}_{freq}", {})
            ends = sorted(periods, reverse=True)[:limit]
            if freq == "annual":
                annual_ends[stmt] = ends
            for end in ends:
                for row, slug in lines.items():
                    value = periods[end].get(row)
                    if value is None or (freq == "annual" and (slug, end) in skip):
                        continue
                    fid = stmt_fact_id(stmt, slug, freq, end)
                    unit = "USD/share" if slug == "diluted_eps" else "USD"
                    kind_label = "FY ended" if freq == "annual" else "quarter ended"
                    add(fid, value, unit, f"{row} ({ticker}, {freq})", "reported", src, f"{kind_label} {end}")
                    if freq == "annual":
                        written[(stmt, slug, end)] = (value, fid)
        if annual_ends[stmt]:
            res.fetched += 1
        else:
            gap(f"{stmt.capitalize()} statement (annual)", "yfinance returned no annual periods",
                "ratios that need it are missing" + ("; the run is stopped by the data gate" if stmt == "income" else ""))

    # ---- ratios (latest fiscal year against the prior one)
    if not annual_ends.get("income"):
        return res
    end0 = annual_ends["income"][0]
    end1 = annual_ends["income"][1] if len(annual_ends["income"]) > 1 else None

    def book(end: str | None) -> dict[str, Input | None]:
        out: dict[str, Input | None] = {}
        for name, options in RATIO_INPUTS.items():
            out[name] = None
            for stmt, slug in options:
                if end and (stmt, slug, end) in written:
                    value, fid = written[(stmt, slug, end)]
                    out[name] = Input(value, fid)
                    break
        if end and figures:
            for name, metric in (("revenue", "revenue"), ("net_income", "net_income")):   # reported figures win
                ref_end = xbrl_end_for.get((metric, end))
                if ref_end:
                    out[name] = Input(figures.values[metric][ref_end].value, xb.fact_id(metric, ref_end))
            for name, (stmt, slug, metric) in FALLBACK.items():     # only where yfinance has nothing
                ref_end = xb.match_end(figures.values.get(metric, {}), end)
                if out[name] is None and end == end0 and ref_end:
                    v = figures.values[metric][ref_end]
                    fid = f"market.xbrl.{metric}.annual.{ref_end}"
                    add_fact = xb.fact_from(v, figures.entity, retrieved_at, fid)
                    if all(f.id != fid for f in res.facts):
                        res.facts.append(add_fact)
                    out[name] = Input(v.value, fid)
        return out

    for r in ratios(book(end0), book(end1)):
        if r.value is None:
            gap(r.label, r.reason_missing, f"{r.label.lower()} is not shown in the memo")
            continue
        add(f"market.ratio.{r.slug}", r.value, r.unit, f"{r.label} ({ticker}, FY ended {end0})", "derived",
            "computed in code from ledger facts", f"FY ended {end0}", f"{r.formula}. Inputs: {', '.join(r.inputs)}")
        res.fetched += 1
    return res
