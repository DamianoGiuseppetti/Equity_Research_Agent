"""Tables and the Sources and Methodology section, built by code from the ledger. The model never types these.

Each table function returns (markdown, ids used), so the Sources section can list every ledger fact the memo rests on.
"""
import json
import re
from datetime import datetime
from pathlib import Path

from .ledger import Ledger
from .numbers import display
from .valuation_facts import LABEL


def _find(ledger: Ledger, prefixes: list[str], end: str) -> str | None:
    """Id of the first prefix that has a fact for this fiscal year (period ends within 7 days count as the same year)."""
    for pre in prefixes:
        for f in ledger.facts:
            if f.id.startswith(pre):
                try:
                    if abs((datetime.fromisoformat(f.id[len(pre):]) - datetime.fromisoformat(end)).days) <= 7:
                        return f.id
                except ValueError:
                    continue
    return None


def _md(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def fiscal_ends(ledger: Ledger, n: int = 4) -> list[str]:
    ends = {f.id.rsplit(".", 1)[1] for f in ledger.facts if f.id.startswith(("filings.xbrl.revenue.annual.", "market.income.total_revenue.annual."))}
    merged: list[str] = []
    for e in sorted(ends, reverse=True):
        if not any(abs((datetime.fromisoformat(e) - datetime.fromisoformat(m)).days) <= 7 for m in merged):
            merged.append(e)
    return merged[:n]


FIN_ROWS = [("Revenue", ["filings.xbrl.revenue.annual.", "market.income.total_revenue.annual."]),
            ("Gross profit", ["market.income.gross_profit.annual."]),
            ("Operating income", ["market.income.operating_income.annual."]),
            ("Net income", ["filings.xbrl.net_income.annual.", "market.income.net_income.annual."]),
            ("Diluted EPS", ["filings.xbrl.diluted_eps.annual.", "market.income.diluted_eps.annual."]),
            ("Free cash flow", ["market.cashflow.free_cash_flow.annual."])]


def financial_tables(ledger: Ledger) -> tuple[str, set[str]]:
    ends, used = fiscal_ends(ledger), set()
    rows = []
    for label, prefixes in FIN_ROWS:
        cells = []
        for e in ends:
            fid = _find(ledger, prefixes, e)
            used.add(fid) if fid else None
            cells.append(display(ledger.get(fid)) if fid else "n/a")
        rows.append([label, *cells, "`" + prefixes[0] + "<period end>`" + (f" or `{prefixes[1]}<period end>`" if len(prefixes) > 1 else "")])
    text = "**Annual financial history** (fiscal years ending on the dates shown; built from the ledger)\n\n" + _md(["Metric", *ends, "Ledger ids"], rows)
    ratio_rows = []
    for f in ledger.facts:
        if f.id.startswith("market.ratio."):
            ratio_rows.append([f.label, display(f), f"`{f.id}`"])
            used.add(f.id)
    if ratio_rows:
        text += "\n\n**Ratios, latest fiscal year** (computed in code)\n\n" + _md(["Ratio", "Value", "Ledger id"], ratio_rows)
    return text, used


def scenario_table(ledger: Ledger) -> tuple[str, set[str]]:
    if "valuation.dcf.base.value_per_share" not in ledger:
        return "", set()
    used, rows = {"market.price.last_close"}, []
    for s in ("bear", "base", "bull"):
        ids = [f"valuation.dcf.{s}.growth_y1", f"valuation.dcf.{s}.margin", f"valuation.dcf.{s}.value_per_share", f"valuation.dcf.{s}.gap_vs_price"]
        used.update(ids)
        rows.append([s.capitalize(), *[display(ledger.get(i)) for i in ids], f"`valuation.dcf.{s}.*`"])
    price = display(ledger.get("market.price.last_close"))
    return (f"**DCF scenarios** (all values are model estimates; last close {price})\n\n" +
            _md(["Scenario", "Year-1 revenue growth (assumption)", "Free cash flow margin (assumption)", "Value per share (estimate)", "Versus last close (estimate)", "Ledger ids"], rows)), used


def grid_table(ledger: Ledger) -> tuple[str, set[str]]:
    if "valuation.grid.wacc_0" not in ledger:
        return "", set()
    used, rows = set(), []
    gs = [f"valuation.grid.g_{j}" for j in range(5)]
    used.update(gs)
    for i in range(5):
        wid = f"valuation.grid.wacc_{i}"
        used.add(wid)
        cells = []
        for j in range(5):
            cid = f"valuation.grid.cell_{i}_{j}"
            if cid in ledger:
                used.add(cid)
            cells.append(display(ledger.get(cid)) if cid in ledger else "n/m")
        rows.append([display(ledger.get(wid)), *cells])
    return ("**Sensitivity of the base-case DCF value per share** (model estimates; rows are WACC, columns are terminal growth; n/m where WACC is not above growth; "
            "ids `valuation.grid.*`)\n\n" + _md(["WACC \\ terminal growth", *[display(ledger.get(g)) for g in gs]], rows)), used


def peer_table(ledger: Ledger, gaps: list[dict]) -> tuple[str, set[str]]:
    peers = sorted({f.id.split(".")[3] for f in ledger.facts if f.id.startswith("valuation.comps.peer.")})
    if not peers:
        return "", set()
    used, rows = set(), []
    keys = ("ev_ebitda", "pe", "ev_sales")
    for t in peers:
        ids = [f"valuation.comps.peer.{t}.{k}" for k in keys]
        used.update(ids)
        rows.append([t, *[display(ledger.get(i)) for i in ids]])
    mids = [f"valuation.comps.median.{k}" for k in keys]
    used.update(mids)
    rows.append(["**Median**", *[display(ledger.get(i)) for i in mids]])
    dropped = [f"{g['missing'].removeprefix('peer ')} ({g['reason']})" for g in gaps if g["area"] == "valuation" and g["missing"].startswith("peer ") and g["missing"] != "peer multiples"]
    text = "**Peer multiples** (trailing, from yfinance; ids `valuation.comps.peer.<ticker>.<multiple>`)\n\n" + _md(["Peer", "EV/EBITDA", "P/E", "EV/Sales"], rows)
    return text + (f"\n\nProposed peers left out: {'; '.join(dropped)}." if dropped else ""), used


def outlook_table(ledger: Ledger) -> tuple[str, set[str]]:
    if "valuation.dcf.base.revenue_y1" not in ledger:
        return "", set()
    used, rows = set(), []
    for s in ("bear", "base", "bull"):
        for key, label in (("revenue", "Revenue"), ("fcf", "Free cash flow")):
            ids = [f"valuation.dcf.{s}.{key}_y{y}" for y in range(1, 6)]
            used.update(ids)
            rows.append([f"{s.capitalize()} {label.lower()}", *[display(ledger.get(i)) for i in ids]])
    return ("**Projected revenue and free cash flow** (model estimates from the DCF scenarios; ids `valuation.dcf.<scenario>.revenue_y<n>` and `fcf_y<n>`)\n\n" +
            _md(["Scenario and line", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"], rows)), used


def risk_table(ledger: Ledger) -> tuple[str, set[str]]:
    rows = []
    for f in ledger.facts:
        if re.fullmatch(r"extract\.risk\.\d+", f.id) and f"{f.id}.rank" in ledger:
            title, _, rest = str(f.value).partition(" (")
            category, _, summary = rest.partition("): ")
            rank = ledger.get(f"{f.id}.rank").value
            rows.append((rank, [str(rank), title, category, summary, f"`{f.id}`"], {f.id, f"{f.id}.rank"}))
    if not rows:
        return "", set()
    rows.sort(key=lambda r: r[0])
    return ("**Risk factors from Item 1A, as ranked by the model** (the ranking by severity is a model estimate; each risk quotes the filing in its ledger note)\n\n" +
            _md(["Rank (estimate)", "Risk", "Category", "Summary", "Ledger id"], [r[1] for r in rows])), set().union(*[r[2] for r in rows])


# ---- header and Sources and Methodology -----------------------------------------------------------
def header(state: dict, ledger: Ledger, generated_at: str = "") -> str:
    c = state["company"]
    price = f"{display(ledger.get('market.price.last_close'))} on {ledger.get('market.price.last_close').as_of}" if "market.price.last_close" in ledger else "n/a"
    return (f"# {c['name']} ({c['ticker']}): equity research memo\n\n"
            f"> **Draft for a human reviewer, not a finished report.** {LABEL.capitalize()}. Last close {price}. "
            f"Every number in the text is followed by the id of its ledger fact; estimates and assumptions are worded as such." +
            (f" Generated {generated_at}." if generated_at else "") + "\n")


def _family(source: str) -> str:
    s = source.lower()
    for key, name in (("yfinance", "yfinance (Yahoo Finance): prices, statements, company info, peer multiples"),
                      ("sec edgar", "SEC EDGAR: 10-K and 10-Q filings and XBRL companyfacts"), ("fred", "FRED (Federal Reserve Bank of St. Louis): macro series"),
                      ("finnhub", "Finnhub company-news: article list (publisher pages for full text; text stays in the local cache)"),
                      ("computed", "Computed in code from the facts above")):
        if key in s:
            return name
    return source[:80]


def sources_section(state: dict, ledger: Ledger, cited: set[str], gap_index: dict[str, dict],
                    failed: dict[str, list[str]], cost: dict | None, run_folder: Path, refs: dict[str, int] | None = None) -> str:
    lines = ["### Sources"]
    by_family: dict[str, list] = {}
    for f in ledger.facts:
        by_family.setdefault(_family(f.source), []).append(f)
    for fam, facts in sorted(by_family.items()):
        lines.append(f"- {fam}: {len(facts)} facts, retrieved {max(f.retrieved_at for f in facts).date().isoformat()}")
    for tag in ("tenk", "tenq"):
        if f"filings.{tag}.accession" in ledger:
            lines.append(f"- {ledger.get(f'filings.{tag}.accession').source}: accession {ledger.get(f'filings.{tag}.accession').value}, "
                         f"period {ledger.get(f'filings.{tag}.period').value}, {ledger.get(f'filings.{tag}.url').value}")
    sent_path = run_folder / "news" / "sentiment.json"
    if sent_path.exists():
        rows = [[a["headline"].replace("|", "/"), a["published_at"][:10], a["url"], (f"{a['scored']['label']} ({a['scored']['score']:+.2f}, model estimate)" if a["scored"] else "unscored")]
                for a in json.loads(sent_path.read_text())]
        if rows:
            lines += ["", "**News articles used for the sentiment estimate** (headline, date and URL only; Finnhub personal-use data)", "",
                      _md(["Headline", "Date", "URL", "Model-estimated effect on shareholders"], rows)]
    lines += ["", "### Model and run",
              f"- Profile `{state.get('profile')}`; models: " + ", ".join(f"{r} = `{m}`" for r, m in state.get("model_ids", {}).items()) + f"; thread `{state.get('thread_id')}`.",
              (f"- LLM cost of this run so far: ${cost['totals']['cost_usd']:.4f} over {cost['totals']['calls']} calls "
               f"({cost['totals']['input_tokens']:,} input and {cost['totals']['output_tokens']:,} output tokens)." if cost else "- LLM cost: no cost log found.")]
    assumptions = [f for f in ledger.facts if f.id.startswith("valuation.assump.")]
    lines += ["", "### Method and assumptions",
              "- Reported figures come from SEC XBRL and yfinance; a revenue, net income or EPS figure that differs by more than 0.5% between them is a data gap and the XBRL value is used.",
              "- DCF: three scenarios over five years; free cash flow = revenue x free cash flow margin; value per share = (enterprise value - net debt) / shares. Every projected figure is a model estimate.",
              "- Comparable multiples: peers are proposed by the model and validated in code (listed in the US, non-financial, EV/EBITDA, P/E and EV/Sales available); medians are applied to the company's latest fiscal-year figures.",
              f"- Rating: a fixed rule comparing the last close with the bear, base and bull values ({LABEL}).",
              "- Risk ranking, news sentiment scores and extracted summaries are produced by a language model. Quotes from the filing were checked by code as exact substrings; sentiment evidence likewise.",
              "- A number in the text matches a ledger value when the value, scaled to the unit shown and rounded to the precision shown, equals it. Years, dates and time windows are not checked."]
    lines += [f"- Assumption: {display(f)} {f.label.lower()} ({f.note}) `{f.id}`" for f in assumptions]
    lines += ["", "### Coverage"]
    lines += [f"- {a}: {c['fetched']} of {c['expected']} items fetched" for a, c in sorted(state.get("coverage", {}).items())]
    ex, se = state.get("extract_summary"), state.get("sentiment_summary")
    if ex:
        lines.append(f"- Filing extraction: {ex['items']} items kept, {ex['dropped']} dropped after failing checks, {ex['failed_calls']} model calls failed.")
    if se:
        lines.append(f"- News sentiment: {se['n_scored']} scored, {se['n_unscored']} unscored, {se['n_dropped_no_text']} dropped for missing text" + ("; coverage is thin, so no average is shown." if se["low_coverage"] else "."))
    lines += ["", "### Data gaps"]
    anchor = (lambda t: f'<a id="gap-{t}"></a>') if refs is not None else (lambda t: "")
    lines += [f"- {anchor(t)}[gap:{t}] [{g['area']}] {g['missing']}: {g['reason']}. Effect: {g['effect']}" for t, g in gap_index.items()] or ["- none"]
    lines += ["", "### Checks"]
    lines += ([f"- **{name}** failed checks: " + "; ".join(p) for name, p in failed.items()] or ["- All sections passed the memo checks."])
    def row(i):
        cells = _row(i)
        return [f'<a id="ref-{refs[i]}"></a>{refs[i]}', *cells] if refs is not None else cells

    _row = lambda i: [f"`{i}`", ledger.get(i).label.replace("|", "/"), display(ledger.get(i))[:90].replace("|", "/"), ledger.get(i).kind,
                     ledger.get(i).source[:70].replace("|", "/"), ledger.get(i).as_of]
    header_row = (["Ref"] if refs is not None else []) + ["Id", "Label", "Value", "Kind", "Source", "As of"]
    order = (lambda ids: sorted(ids, key=lambda i: refs[i])) if refs is not None else sorted
    figures = [i for i in order(cited) if i in ledger and ledger.get(i).kind != "summary"]
    summaries = [i for i in order(cited) if i in ledger and ledger.get(i).kind == "summary"]
    lines += ["", "### Ledger facts cited in this memo: figures", "", _md(header_row, [row(i) for i in figures])]
    if summaries:
        lines += ["", "### Ledger facts cited in this memo: summaries of the 10-K written by the model (not figures; each quotes the filing in its ledger note)", "",
                  _md(header_row, [row(i) for i in summaries])]
    return "\n".join(lines)
