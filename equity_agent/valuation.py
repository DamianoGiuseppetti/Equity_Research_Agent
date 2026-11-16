"""Valuation maths: three-scenario DCF, WACC, sensitivity grid, comps and the rating rule. Plain Python, no LLM, no I/O.

Every function takes numbers and returns numbers, so each can be checked by hand (see the DCF test).
"""
import statistics
from dataclasses import dataclass

YEARS = 5
GRID_WACC_STEPS_PTS = (-2.0, -1.0, 0.0, 1.0, 2.0)       # rows: WACC around the base WACC
GRID_G_STEPS_PTS = (-1.0, -0.5, 0.0, 0.5, 1.0)          # columns: terminal growth around the configured one
RATINGS = ("undervalued", "modestly undervalued", "modestly overvalued", "overvalued")


# ---- growth, margins and the DCF --------------------------------------------------------------
def growth_path(g_start: float, fade_target: float, shift_pts: float = 0.0, years: int = YEARS) -> list[float]:
    """Growth for years 1..years: starts at g_start in year 1 and fades in equal steps to fade_target in the last year.
    shift_pts (percentage points) is then added to every year: -3.0 for bear, +3.0 for bull."""
    step = (fade_target - g_start) / (years - 1)
    return [g_start + step * i + shift_pts / 100 for i in range(years)]


@dataclass(frozen=True)
class YearRow:
    year: int
    growth: float
    revenue: float
    fcf: float
    pv_fcf: float


@dataclass(frozen=True)
class Dcf:
    rows: list[YearRow]
    sum_pv_fcf: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    equity_value: float
    value_per_share: float


def dcf(revenue0: float, growths: list[float], margin: float, wacc: float, terminal_growth: float,
        net_debt: float, shares: float) -> Dcf:
    """Revenue grows by the yearly rates, free cash flow is revenue x margin, each year is discounted at year-end.
    Terminal value (Gordon growth) = last FCF x (1 + g) / (WACC - g), discounted from the last year."""
    if wacc <= terminal_growth:
        raise ValueError(f"WACC {wacc:.4f} must exceed terminal growth {terminal_growth:.4f}")
    rows, revenue = [], revenue0
    for t, g in enumerate(growths, start=1):
        revenue *= 1 + g
        fcf = revenue * margin
        rows.append(YearRow(t, g, revenue, fcf, fcf / (1 + wacc) ** t))
    sum_pv = sum(r.pv_fcf for r in rows)
    tv = rows[-1].fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_tv = tv / (1 + wacc) ** len(rows)
    ev = sum_pv + pv_tv
    equity = ev - net_debt
    return Dcf(rows, sum_pv, tv, pv_tv, ev, equity, equity / shares)


@dataclass(frozen=True)
class Wacc:
    cost_of_equity: float
    cost_of_debt: float | None     # None when the company has no debt
    weight_equity: float
    weight_debt: float
    wacc: float


def wacc_of(risk_free: float, beta: float, erp: float, cost_of_debt: float | None, market_cap: float, debt: float,
            tax_rate: float = 0.0) -> Wacc:
    """CAPM cost of equity = risk-free + beta x ERP. Weights: market cap and book debt. Interest is tax deductible, so the debt
    term uses cost of debt x (1 - tax rate); with tax_rate 0 it is the pre-tax cost of debt. Wacc.cost_of_debt is the rate used."""
    ke = risk_free + beta * erp
    if debt <= 0 or cost_of_debt is None:
        return Wacc(ke, None, 1.0, 0.0, ke)
    kd = cost_of_debt * (1 - tax_rate)
    we, wd = market_cap / (market_cap + debt), debt / (market_cap + debt)
    return Wacc(ke, kd, we, wd, we * ke + wd * kd)


def sensitivity_grid(revenue0, growths, margin, wacc, terminal_growth, net_debt, shares) -> dict:
    """5 x 5 value per share: rows are WACC (base -2, -1, 0, +1, +2 points), columns are terminal growth (-1, -0.5, 0, +0.5, +1
    points). A cell where WACC is not above growth is None."""
    waccs = [wacc + s / 100 for s in GRID_WACC_STEPS_PTS]
    gs = [terminal_growth + s / 100 for s in GRID_G_STEPS_PTS]
    cells = [[dcf(revenue0, growths, margin, w, g, net_debt, shares).value_per_share if w > g else None for g in gs] for w in waccs]
    return {"waccs": waccs, "growths": gs, "cells": cells}


# ---- rating -----------------------------------------------------------------------------------
def rating(price: float, low: float, mid: float, high: float) -> str:
    """Price below low: undervalued. low up to mid: modestly undervalued. mid up to high (inclusive): modestly overvalued.
    Above high: overvalued. For the DCF low, mid, high are bear, base, bull; for comps they are the lowest, middle and highest
    implied value."""
    low, mid, high = sorted((low, mid, high))
    if price < low:
        return RATINGS[0]
    if price < mid:
        return RATINGS[1]
    if price <= high:
        return RATINGS[2]
    return RATINGS[3]


# ---- comps ------------------------------------------------------------------------------------
MULTIPLES = {"ev_ebitda": "EV/EBITDA", "pe": "P/E", "ev_sales": "EV/Sales"}


def implied_values(medians: dict[str, float], eps: float | None, ebitda: float | None, revenue: float | None,
                   net_debt: float, shares: float) -> tuple[dict[str, float], dict[str, str]]:
    """Value per share implied by each peer median. Returns (values, reasons for the multiples that cannot be applied)."""
    values, skipped = {}, {}
    if eps is not None and eps > 0:
        values["pe"] = medians["pe"] * eps
    else:
        skipped["pe"] = "the company's earnings per share are not positive"
    for key, base, name in (("ev_ebitda", ebitda, "EBITDA"), ("ev_sales", revenue, "revenue")):
        if base is not None and base > 0:
            values[key] = (medians[key] * base - net_debt) / shares
        else:
            skipped[key] = f"the company's {name} is missing or not positive"
    return values, skipped


def medians_of(peers: dict[str, dict[str, float]]) -> dict[str, float]:
    return {m: statistics.median(p[m] for p in peers.values()) for m in MULTIPLES}
