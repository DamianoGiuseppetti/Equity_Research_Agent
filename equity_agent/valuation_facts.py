"""Valuation orchestration without I/O: read the inputs from the ledger, run the maths, return facts, gaps and a result dict.

Every projected figure is kind model_estimate and every parameter is kind assumption. A DCF that cannot be computed says
why in a data gap; it is never replaced by a plug number.
"""
import statistics
from dataclasses import dataclass
from datetime import datetime

from . import valuation as v
from .config import Valuation
from .ledger import Ledger
from .schemas import DataGap, Fact
from .xbrl import match_end

PREFIX = "valuation"
LABEL = "valuation-implied view, not investment advice"


@dataclass
class Inputs:
    price: float
    shares: float
    beta: float | None
    market_cap: float | None
    risk_free: float | None                       # fraction
    revenue: list[tuple[str, float, str]]         # (period end, value, fact id), latest first
    fcf_margins: list[tuple[str, float, str, str]]   # (period end, FCF / revenue, FCF fact id, revenue fact id), latest first
    net_debt: tuple[float, str] | None
    interest: tuple[float, str] | None
    debt: tuple[float, str] | None
    eps: tuple[float, str] | None
    ebitda: tuple[float, str] | None
    tax_provision: tuple[float, str] | None = None
    pretax_income: tuple[float, str] | None = None


def _val(ledger: Ledger, fid: str) -> float | None:
    return float(ledger.get(fid).value) if fid in ledger and not isinstance(ledger.get(fid).value, str) else None


def _latest(ledger: Ledger, prefixes: list[str]) -> tuple[float, str] | None:
    """The annual fact with the latest period end among the id prefixes (first prefix wins on the same end)."""
    best = None
    for pre in prefixes:
        for f in ledger.facts:
            if f.id.startswith(pre) and not isinstance(f.value, str):
                end = f.id[len(pre):]
                if best is None or end > best[0]:
                    best = (end, float(f.value), f.id)
    return (best[1], best[2]) if best else None


def read_inputs(ledger: Ledger) -> Inputs | None:
    price, shares = _val(ledger, "market.price.last_close"), _val(ledger, "market.info.shares_outstanding")
    if not price or not shares:
        return None
    rf = _val(ledger, "macro.dgs10.value")
    # revenue per fiscal year: the SEC value where the ledger has it, else yfinance; ends within 7 days are the same year
    xbrl = {f.id.rsplit(".", 1)[1]: f for f in ledger.facts if f.id.startswith("filings.xbrl.revenue.annual.")}
    yf = {f.id.rsplit(".", 1)[1]: f for f in ledger.facts if f.id.startswith("market.income.total_revenue.annual.")}
    revenue = {end: (end, float(f.value), f.id) for end, f in yf.items()}
    for end, f in xbrl.items():
        for old in [e for e in revenue if match_end({e: 0}, end) or e == end]:
            revenue.pop(old)
        revenue[end] = (end, float(f.value), f.id)
    rev_list = sorted(revenue.values(), reverse=True)
    margins = []
    for f in sorted((f for f in ledger.facts if f.id.startswith("market.cashflow.free_cash_flow.annual.")), key=lambda f: f.id, reverse=True):
        end = f.id.rsplit(".", 1)[1]
        year = next((r for r in rev_list if abs((datetime.fromisoformat(r[0]) - datetime.fromisoformat(end)).days) <= 7), None)
        if year and year[1]:
            margins.append((end, float(f.value) / year[1], f.id, year[2]))
    nd = _val(ledger, "market.ratio.net_debt")
    return Inputs(price, shares, _val(ledger, "market.info.beta"), _val(ledger, "market.info.market_cap"),
                  None if rf is None else rf / 100, rev_list, margins,
                  (nd, "market.ratio.net_debt") if nd is not None else None,
                  _latest(ledger, ["market.income.interest_expense.annual.", "market.xbrl.interest_expense.annual."]),
                  _latest(ledger, ["market.balance.total_debt.annual.", "market.xbrl.total_debt.annual."]),
                  _latest(ledger, ["filings.xbrl.diluted_eps.annual.", "market.income.diluted_eps.annual."]),
                  _latest(ledger, ["market.income.ebitda.annual."]),
                  _latest(ledger, ["market.income.tax_provision.annual."]), _latest(ledger, ["market.income.pretax_income.annual."]))


class Builder:
    """Collects facts and gaps with the shared source and timestamp."""

    def __init__(self, as_of: str, retrieved_at: datetime) -> None:
        self.facts: list[Fact] = []
        self.gaps: list[DataGap] = []
        self.as_of, self.retrieved_at = as_of, retrieved_at

    def add(self, slug: str, value, unit: str, label: str, kind: str, note: str, source: str = "computed in code (valuation node)") -> str:
        fid = f"{PREFIX}.{slug}"
        self.facts.append(Fact(id=fid, value=value, unit=unit, label=label, kind=kind, source=source, as_of=self.as_of,
                               retrieved_at=self.retrieved_at, note=note))
        return fid

    def gap(self, missing: str, reason: str, effect: str) -> None:
        self.gaps.append(DataGap(area="valuation", missing=missing, reason=reason, effect=effect))


def run_dcf(inp: Inputs, cfg: Valuation, cfg_fallback_pts: float, b: Builder, changed: dict | None = None) -> dict | None:
    """The DCF block of the result, or None with a gap that says why it was not computed."""
    effect = "no DCF in the memo; the valuation uses peer multiples only"
    if inp.risk_free is None:
        b.gap("DCF (10-year Treasury yield missing)", "no risk-free rate in the ledger", effect + " (DCF not computed)")
        return None
    missing = [name for name, val in (("beta", inp.beta), ("market capitalization", inp.market_cap), ("net debt", inp.net_debt)) if val is None]
    if missing:
        b.gap(f"DCF inputs: {', '.join(missing)}", "not in the ledger", effect + " (DCF not computed)")
        return None
    if len(inp.revenue) < 4 or len(inp.fcf_margins) < 3:
        b.gap("DCF history", f"needs 4 fiscal years of revenue and 3 of free cash flow; the ledger has {len(inp.revenue)} and {len(inp.fcf_margins)}",
              effect + " (DCF not computed)")
        return None
    margins = inp.fcf_margins[:3]
    avg_margin = statistics.mean(m[1] for m in margins)
    if margins[0][1] < cfg.near_zero_fcf_margin or avg_margin < cfg.near_zero_fcf_margin:
        b.gap("DCF (free cash flow negative or near zero)",
              f"latest free cash flow margin {margins[0][1]:.1%}, 3-year average {avg_margin:.1%}, below the {cfg.near_zero_fcf_margin:.0%} floor",
              effect + " (a DCF on near-zero cash flow is not meaningful)")
        return None
    rev0, rev3 = inp.revenue[0], inp.revenue[3]
    if rev3[1] <= 0:
        b.gap("DCF revenue growth", "revenue three years earlier is not positive", effect + " (DCF not computed)")
        return None
    cagr = (rev0[1] / rev3[1]) ** (1 / 3) - 1
    # cost of debt: interest expense over total debt from the ledger, else risk-free plus the configured spread (an assumption)
    kd, kd_id = None, None
    if inp.debt and inp.debt[0] > 0:
        if inp.interest and inp.interest[0]:
            kd = abs(inp.interest[0]) / inp.debt[0]
            kd_id = b.add("input.cost_of_debt", kd, "fraction", "Cost of debt", "derived",
                          f"interest expense / total debt, from {inp.interest[1]} and {inp.debt[1]}")
        else:
            kd = inp.risk_free + cfg.cost_of_debt_fallback_spread_pts / 100
            kd_id = b.add("assump.cost_of_debt_fallback", kd, "fraction", "Cost of debt (fallback)", "assumption",
                          f"the ledger has no interest expense, so cost of debt = 10-year yield (macro.dgs10.value) + {cfg.cost_of_debt_fallback_spread_pts} points "
                          f"(cost_of_debt_fallback_spread_pts in config.yaml)")
    # tax rate = tax provision / pretax income of the latest fiscal year; interest is deductible, so cost of debt is taken after tax.
    # Missing inputs or a rate outside 0 to 40% mean the pre-tax cost of debt is used, and a gap says so.
    tax, tax_id = 0.0, None
    if kd is not None:
        rate = inp.tax_provision[0] / inp.pretax_income[0] if inp.tax_provision and inp.pretax_income and inp.pretax_income[0] > 0 else None
        if rate is not None and 0 <= rate <= 0.40:
            tax = rate
            tax_id = b.add("input.tax_rate", rate, "fraction", "Effective tax rate", "derived",
                           f"tax provision / pretax income, from {inp.tax_provision[1]} and {inp.pretax_income[1]}")
        else:
            why = ("tax provision or pretax income is missing, or pretax income is not positive" if rate is None
                   else f"the effective tax rate {rate:.1%} is outside 0 to 40%")
            b.gap("tax rate for the after-tax cost of debt", why, "the WACC uses the pre-tax cost of debt, which overstates it a little")
    w = v.wacc_of(inp.risk_free, inp.beta, cfg.equity_risk_premium, kd, inp.market_cap, inp.debt[0] if inp.debt else 0.0, tax)
    if kd is not None and tax_id:
        b.add("input.cost_of_debt_after_tax", w.cost_of_debt, "fraction", "Cost of debt after tax", "derived",
              f"cost of debt ({kd_id}) x (1 - tax rate ({tax_id}))")
    if w.wacc <= cfg.terminal_growth + 0.02:
        b.gap("DCF (WACC too close to terminal growth)", f"WACC {w.wacc:.2%} against terminal growth {cfg.terminal_growth:.2%}; the grid needs 2 points of room",
              effect + " (DCF not computed)")
        return None

    # parameters (assumptions) and derived inputs
    changed = changed or {}
    by = lambda key: f"; changed by the reviewer from the default {changed[key]:g}" if key in changed else ""
    ids = {
        "erp": b.add("assump.equity_risk_premium", cfg.equity_risk_premium, "fraction", "Equity risk premium", "assumption", f"{cfg.equity_risk_premium:.1%} (config.yaml valuation.equity_risk_premium)" + by("equity_risk_premium")),
        "tg": b.add("assump.terminal_growth", cfg.terminal_growth, "fraction", "Terminal growth", "assumption", "Gordon growth in all scenarios (config.yaml valuation.terminal_growth)" + by("terminal_growth")),
        "spread": b.add("assump.scenario_spread_pts", cfg.scenario_spread_pts, "pts", "Scenario growth spread", "assumption", "bear growth = base minus this many points in every year, bull = base plus (config.yaml)" + by("scenario_spread_pts")),
        "fade": b.add("assump.fade_target_growth", cfg.fade_target_growth, "fraction", "Fade target growth", "assumption", "revenue growth fades linearly to this by year 5 (config.yaml)" + by("fade_target_growth")),
        "shift": b.add("assump.margin_shift_pts", cfg.margin_shift_pts, "pts", "Scenario margin shift", "assumption", "bear free cash flow margin = 3-year average minus this many points, bull = plus (config.yaml)"),
        "cagr": b.add("input.revenue_cagr_3y", cagr, "fraction", "Revenue CAGR, last 3 fiscal years", "derived",
                      f"(revenue {rev0[0]} / revenue {rev3[0]})^(1/3) - 1, from {rev0[2]} and {rev3[2]}"),
        "margin": b.add("input.fcf_margin_3y_avg", avg_margin, "fraction", "Free cash flow margin, 3-year average", "derived",
                        "mean of free cash flow / revenue over the last 3 fiscal years; inputs " + ", ".join(f"{m[2]} and {m[3]}" for m in margins)),
        "ke": b.add("input.cost_of_equity", w.cost_of_equity, "fraction", "Cost of equity (CAPM)", "derived",
                    "risk-free (macro.dgs10.value) + beta (market.info.beta) x equity risk premium"),
        "wacc": b.add("input.wacc", w.wacc, "fraction", "WACC", "derived",
                      f"equity weight {w.weight_equity:.1%} (market cap, market.info.market_cap) x cost of equity + debt weight {w.weight_debt:.1%} "
                      f"(book debt) x {'after-tax ' if tax_id else 'pre-tax '}cost of debt" + ("" if tax_id else " (no usable tax rate: this overstates WACC a little)")),
    }
    base_g = v.growth_path(cagr, cfg.fade_target_growth)
    scenarios = {"bear": (v.growth_path(cagr, cfg.fade_target_growth, -cfg.scenario_spread_pts), avg_margin - cfg.margin_shift_pts / 100),
                 "base": (base_g, avg_margin),
                 "bull": (v.growth_path(cagr, cfg.fade_target_growth, cfg.scenario_spread_pts), avg_margin + cfg.margin_shift_pts / 100)}
    out: dict = {"wacc": w.wacc, "scenarios": {}, "ids": ids}
    for name, (growths, margin) in scenarios.items():
        res = v.dcf(rev0[1], growths, margin, w.wacc, cfg.terminal_growth, inp.net_debt[0], inp.shares)
        b.add(f"dcf.{name}.margin", margin, "fraction", f"{name.capitalize()} free cash flow margin", "assumption",
              "3-year average margin" + ("" if name == "base" else f" {'minus' if name == 'bear' else 'plus'} {cfg.margin_shift_pts} points"))
        for r in res.rows:
            b.add(f"dcf.{name}.growth_y{r.year}", r.growth, "fraction", f"{name.capitalize()} revenue growth, year {r.year}", "assumption",
                  "3-year revenue CAGR fading linearly to the fade target by year 5" + ("" if name == "base" else f", {'minus' if name == 'bear' else 'plus'} {cfg.scenario_spread_pts} points"))
            b.add(f"dcf.{name}.revenue_y{r.year}", r.revenue, "USD", f"{name.capitalize()} projected revenue, year {r.year}", "model_estimate", "prior-year revenue x (1 + growth)")
            b.add(f"dcf.{name}.fcf_y{r.year}", r.fcf, "USD", f"{name.capitalize()} projected free cash flow, year {r.year}", "model_estimate", "projected revenue x free cash flow margin")
        for slug, label, val, note in (
                ("sum_pv_fcf", "Present value of years 1 to 5", res.sum_pv_fcf, "each year's free cash flow / (1 + WACC)^year"),
                ("terminal_value", "Terminal value", res.terminal_value, "year-5 free cash flow x (1 + g) / (WACC - g)"),
                ("pv_terminal_value", "Present value of terminal value", res.pv_terminal_value, "terminal value / (1 + WACC)^5"),
                ("enterprise_value", "Enterprise value", res.enterprise_value, "present value of years 1 to 5 + present value of terminal value"),
                ("equity_value", "Equity value", res.equity_value, f"enterprise value - net debt ({inp.net_debt[1]})")):
            b.add(f"dcf.{name}.{slug}", val, "USD", f"{name.capitalize()} {label.lower()}", "model_estimate", note)
        b.add(f"dcf.{name}.value_per_share", res.value_per_share, "USD/share", f"{name.capitalize()} DCF value per share", "model_estimate",
              f"equity value / shares outstanding (market.info.shares_outstanding)")
        b.add(f"dcf.{name}.gap_vs_price", res.value_per_share / inp.price - 1, "fraction", f"{name.capitalize()} DCF value versus last close", "model_estimate",
              "value per share / last close (market.price.last_close) - 1")
        out["scenarios"][name] = {"margin": margin, "growths": growths, "value_per_share": res.value_per_share,
                                  "gap_vs_price": res.value_per_share / inp.price - 1, "enterprise_value": res.enterprise_value,
                                  "equity_value": res.equity_value, "rows": [r.__dict__ for r in res.rows]}
    grid = v.sensitivity_grid(rev0[1], base_g, avg_margin, w.wacc, cfg.terminal_growth, inp.net_debt[0], inp.shares)
    for i, wv in enumerate(grid["waccs"]):
        b.add(f"grid.wacc_{i}", wv, "fraction", f"Sensitivity grid WACC, row {i + 1}", "assumption", "base WACC plus -2, -1, 0, +1, +2 points")
    for j, gv in enumerate(grid["growths"]):
        b.add(f"grid.g_{j}", gv, "fraction", f"Sensitivity grid terminal growth, column {j + 1}", "assumption", "configured terminal growth plus -1, -0.5, 0, +0.5, +1 points")
    for i, row in enumerate(grid["cells"]):
        for j, cell in enumerate(row):
            if cell is not None:
                b.add(f"grid.cell_{i}_{j}", cell, "USD/share", f"Base-case value per share, WACC row {i + 1}, growth column {j + 1}", "model_estimate",
                      "base-case DCF recomputed at this WACC and terminal growth")
    out["grid"] = grid
    s = out["scenarios"]
    out["rating"] = v.rating(inp.price, s["bear"]["value_per_share"], s["base"]["value_per_share"], s["bull"]["value_per_share"])
    return out


def run_comps(inp: Inputs, peers: dict, b: Builder, min_peers: int) -> dict | None:
    """The comps block, or None with a gap. `peers` = {"validated": {ticker: {"name", "ev_ebitda", "pe", "ev_sales"}}, "dropped": [...]}."""
    validated = peers["validated"]
    for d in peers["dropped"]:
        b.gap(f"peer {d['ticker']}", d["reason"], "the peer is left out of the peer medians")
    if len(validated) < min_peers:
        b.gap("peer multiples", f"{len(validated)} validated peers, fewer than the {min_peers} needed for a median", "no peer-multiple valuation in the memo")
        return None
    for t, m in validated.items():
        for key, label in v.MULTIPLES.items():
            b.facts.append(Fact(id=f"{PREFIX}.comps.peer.{t}.{key}", value=m[key], unit="times", label=f"{label} of peer {t}",
                                kind="reported", source=f"yfinance {t} (Yahoo Finance), trailing", as_of=b.as_of,
                                retrieved_at=b.retrieved_at, note=f"peer {m['name']}"))
    med = v.medians_of(validated)
    for key, label in v.MULTIPLES.items():
        b.add(f"comps.median.{key}", med[key], "times", f"Peer median {label}", "derived", f"median of {len(validated)} validated peers: {', '.join(validated)}")
    nd = inp.net_debt[0] if inp.net_debt else 0.0
    values, skipped = v.implied_values(med, inp.eps[0] if inp.eps else None, inp.ebitda[0] if inp.ebitda else None,
                                       inp.revenue[0][1] if inp.revenue else None, nd, inp.shares)
    for key, why in skipped.items():
        b.gap(f"implied value from {v.MULTIPLES[key]}", why, "that multiple is not used in the comps range")
    if not values:
        return None
    srcs = {"pe": "peer median P/E x diluted EPS", "ev_ebitda": "(peer median EV/EBITDA x EBITDA - net debt) / shares",
            "ev_sales": "(peer median EV/Sales x revenue - net debt) / shares"}
    for key, val in values.items():
        b.add(f"comps.implied.{key}", val, "USD/share", f"Value per share implied by {v.MULTIPLES[key]}", "model_estimate", srcs[key])
    low, high = min(values.values()), max(values.values())
    b.add("comps.range_low", low, "USD/share", "Peer-multiple range, low", "model_estimate", "lowest implied value per share")
    b.add("comps.range_high", high, "USD/share", "Peer-multiple range, high", "model_estimate", "highest implied value per share")
    out = {"medians": med, "implied": values, "range": [low, high], "peers": validated, "skipped": skipped}
    if len(values) >= 2:
        out["rating"] = v.rating(inp.price, low, statistics.median(values.values()), high)
    return out
