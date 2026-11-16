"""data_gate: plain code, no LLM, no interrupt. Runs once after the collectors (the graph defers it).

Abort only when prices or the income statement are missing, or the reporting currency is not USD. Everything else
is an entry in the gap register, and the run goes on. Coverage is a count of items fetched against items expected.
"""
import json

from ..config import load_config, run_dir
from ..ledger import LedgerFile
from ..schemas import DataGap, GraphState
from .collectors import COLLECTORS

AREA_OF = {"market_data": "market data", "filings": "filings", "macro": "macro", "news": "news"}


def evaluate_gate(state: GraphState) -> str:
    """Return the abort message, or an empty string if the run may continue."""
    fact_ids = {f.id: f for f in LedgerFile(state["ledger_path"]).load().facts}
    if state.get("collectors", {}).get("market_data") != "ok":
        return "Run stopped: no price data could be collected, so there is nothing to value."
    # a missing reporting currency is a gap (the market collector records it); only a known non-USD currency stops the run
    currency = fact_ids["market.info.reporting_currency"].value if "market.info.reporting_currency" in fact_ids else None
    if currency and currency != "USD":
        return f"Run stopped: the company reports in {currency}, and only USD is supported."
    if "market.price.last_close" not in fact_ids:
        return "Run stopped: no price data could be collected, so there is nothing to value."
    # any annual income-statement figure counts: a revenue line replaced by the XBRL value is still an income statement
    if not any(i.startswith("market.income.") and ".annual." in i for i in fact_ids):
        return "Run stopped: the income statement is missing, so fundamentals cannot be analysed."
    return ""


def extra_gaps(state: GraphState) -> list[DataGap]:
    """Gaps only the gate can see: facts that should exist across collectors but do not."""
    cfg, found = load_config(), []
    ledger = LedgerFile(state["ledger_path"]).load()
    already = {(g["area"], g["missing"]) for g in state.get("gaps", [])}
    if "macro.dgs10.value" not in ledger and not any(a == "macro" and "DGS10" in m for a, m in already):
        found.append(DataGap(area="macro", missing="10-year Treasury yield (DGS10)", reason="the series is not in the ledger",
                             effect="the DCF cannot be computed without its risk-free rate; the valuation uses peer multiples only"))
    summary, sent = state.get("news_summary"), state.get("sentiment_summary")
    if sent is not None:       # sentiment ran: the average needs scored articles, and the news collector has already added its own gap
        return found
    if summary is not None and summary["with_full_text"] < cfg.news.low_coverage_min_scored:
        found.append(DataGap(area="news", missing="enough articles for a sentiment average",
                             reason=f"{summary['with_full_text']} articles with full text, fewer than the {cfg.news.low_coverage_min_scored} needed",
                             effect="the memo says news coverage is thin and shows no average sentiment"))
    return found


def complete_coverage(state: GraphState) -> dict[str, dict]:
    """Coverage entries for areas whose collector failed before it could report any."""
    cfg, have = load_config(), state.get("coverage", {})
    expected = {"market_data": cfg.expected.market_data_items, "filings": cfg.expected.filings_sections,
                "macro": len(cfg.macro.common_series), "news": cfg.news.max_articles}
    return {name: {"fetched": 0, "expected": expected[name]} for name in COLLECTORS if name not in have}


async def data_gate(state: GraphState) -> dict:
    abort = evaluate_gate(state)
    gaps = [] if abort else [g.model_dump() for g in extra_gaps(state)]
    coverage = complete_coverage(state)
    register = {"gaps": [*state.get("gaps", []), *gaps], "coverage": {**state.get("coverage", {}), **coverage},
                "collectors": state.get("collectors", {}), "abort_reason": abort}
    (run_dir(state["thread_id"]) / "gap_register.json").write_text(json.dumps(register, indent=1))
    return {"abort_reason": abort, "gaps": gaps, "coverage": coverage, "node_log": ["data_gate"]}
