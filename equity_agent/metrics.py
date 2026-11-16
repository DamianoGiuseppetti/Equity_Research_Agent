"""Technical indicators and financial ratios. Plain pandas and arithmetic, no LLM.

Every function returns data, never raises on missing inputs: a value that cannot be computed is None and the
caller turns it into a data gap.
"""
import math
from dataclasses import dataclass

import pandas as pd

TRADING_DAYS_YEAR = 252


# ---- technicals -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Indicator:
    slug: str
    label: str
    value: float | None
    unit: str
    formula: str
    reason_missing: str = ""


def technicals(prices: list[dict]) -> list[Indicator]:
    """SMA 50 and 200, RSI 14, 52-week high and low, 1-year volatility, returns over 1, 3 and 12 months."""
    df = pd.DataFrame(prices)
    close = df["close"].astype(float) if not df.empty else pd.Series(dtype=float)
    n = len(close)

    def need(days: int, what: str) -> str:
        return "" if n >= days else f"only {n} daily prices, {what} needs {days}"

    out: list[Indicator] = []
    for window in (50, 200):
        why = need(window, f"SMA {window}")
        out.append(Indicator(f"sma_{window}", f"{window}-day simple moving average of the close",
                             None if why else float(close.tail(window).mean()), "USD",
                             f"mean of the last {window} daily closes", why))
    why = need(TRADING_DAYS_YEAR, "the 52-week range")
    out.append(Indicator("high_52w", "52-week high", None if why else float(df["high"].tail(TRADING_DAYS_YEAR).max()),
                         "USD", f"highest daily high of the last {TRADING_DAYS_YEAR} trading days", why))
    out.append(Indicator("low_52w", "52-week low", None if why else float(df["low"].tail(TRADING_DAYS_YEAR).min()),
                         "USD", f"lowest daily low of the last {TRADING_DAYS_YEAR} trading days", why))
    why = need(15, "RSI 14")
    out.append(Indicator("rsi_14", "14-day relative strength index", None if why else _rsi(close, 14), "index",
                         "Wilder RSI: 100 - 100/(1 + average gain/average loss), exponential smoothing alpha 1/14", why))
    why = need(TRADING_DAYS_YEAR + 1, "1-year volatility")
    vol = None if why else float(close.pct_change().tail(TRADING_DAYS_YEAR).std(ddof=1) * math.sqrt(TRADING_DAYS_YEAR))
    out.append(Indicator("volatility_1y", "1-year annualized volatility", vol, "fraction",
                         f"standard deviation of the last {TRADING_DAYS_YEAR} daily returns x sqrt({TRADING_DAYS_YEAR})", why))
    for slug, label, days in (("return_1m", "1-month price return", 21), ("return_3m", "3-month price return", 63),
                              ("return_12m", "12-month price return", TRADING_DAYS_YEAR)):
        why = need(days + 1, label)
        value = None if why else float(close.iloc[-1] / close.iloc[-1 - days] - 1)
        out.append(Indicator(slug, label, value, "fraction",
                             f"close today / close {days} trading days earlier - 1 (price only, dividends excluded)", why))
    return out


def _rsi(close: pd.Series, period: int) -> float:
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    return float(100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1]))


# ---- ratios ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Input:
    value: float
    fact_id: str     # ledger fact the number comes from, so the ratio's note can cite it


@dataclass(frozen=True)
class Ratio:
    slug: str
    label: str
    value: float | None
    unit: str
    formula: str
    inputs: tuple[str, ...] = ()
    reason_missing: str = ""


def _missing(slug: str, label: str, unit: str, formula: str, reason: str) -> Ratio:
    return Ratio(slug, label, None, unit, formula, (), reason)


def ratios(latest: dict[str, Input | None], prior: dict[str, Input | None]) -> list[Ratio]:
    """Ten ratios from the latest and the prior fiscal year. Each argument maps a line name to an Input or None."""

    def has(book: dict, *names: str) -> str:
        gone = [n for n in names if book.get(n) is None]
        return f"missing: {', '.join(gone)}" if gone else ""

    def build(slug, label, unit, formula, value, *inputs):
        return Ratio(slug, label, value, unit, formula, tuple(i.fact_id for i in inputs))

    out: list[Ratio] = []
    rev, rev0 = latest.get("revenue"), prior.get("revenue")
    f = "revenue / prior-year revenue - 1"
    why = has(latest, "revenue") or has(prior, "revenue")
    out.append(_missing("revenue_growth", "Revenue growth", "fraction", f, why) if why or rev0.value == 0 else
               build("revenue_growth", "Revenue growth", "fraction", f, rev.value / rev0.value - 1, rev, rev0))
    ni, ni0 = latest.get("net_income"), prior.get("net_income")
    f = "net income / prior-year net income - 1 (only when prior-year net income is positive)"
    why = has(latest, "net_income") or has(prior, "net_income")
    if not why and ni0.value <= 0:
        why = "prior-year net income is not positive, so growth is not meaningful"
    out.append(_missing("earnings_growth", "Earnings growth", "fraction", f, why) if why else
               build("earnings_growth", "Earnings growth", "fraction", f, ni.value / ni0.value - 1, ni, ni0))
    gp, op = latest.get("gross_profit"), latest.get("operating_income")
    for slug, label, num, name in (("gross_margin", "Gross margin", gp, "gross_profit"),
                                   ("operating_margin", "Operating margin", op, "operating_income"),
                                   ("net_margin", "Net margin", ni, "net_income")):
        f = f"{name.replace('_', ' ')} / revenue"
        why = has(latest, "revenue", name)
        if not why and rev.value == 0:
            why = "revenue is zero"
        out.append(_missing(slug, label, "fraction", f, why) if why else
                   build(slug, label, "fraction", f, num.value / rev.value, num, rev))
    ocf, capex = latest.get("operating_cash_flow"), latest.get("capital_expenditure")
    f = "operating cash flow + capital expenditure (capital expenditure is negative in the statement)"
    why = has(latest, "operating_cash_flow", "capital_expenditure")
    out.append(_missing("free_cash_flow", "Free cash flow", "USD", f, why) if why else
               build("free_cash_flow", "Free cash flow", "USD", f, ocf.value + capex.value, ocf, capex))
    debt, cash = latest.get("total_debt"), latest.get("cash")
    f = "total debt - cash and short-term investments"
    why = has(latest, "total_debt", "cash")
    out.append(_missing("net_debt", "Net debt", "USD", f, why) if why else
               build("net_debt", "Net debt", "USD", f, debt.value - cash.value, debt, cash))
    intr = latest.get("interest_expense")
    f = "operating income / interest expense"
    why = has(latest, "operating_income", "interest_expense")
    if not why and intr.value == 0:
        why = "interest expense is zero"
    out.append(_missing("interest_coverage", "Interest coverage", "times", f, why) if why else
               build("interest_coverage", "Interest coverage", "times", f, op.value / abs(intr.value), op, intr))
    eq, eq0 = latest.get("equity"), prior.get("equity")
    f = "net income / average of this year's and last year's stockholders' equity (this year's only if last year's is missing)"
    why = has(latest, "net_income", "equity")
    if not why:
        avg = (eq.value + eq0.value) / 2 if eq0 is not None else eq.value
        why = "average equity is not positive" if avg <= 0 else ""
    out.append(_missing("roe", "Return on equity", "fraction", f, why) if why else
               build("roe", "Return on equity", "fraction", f, ni.value / avg, ni, eq, *([eq0] if eq0 else [])))
    tax, pretax, ic = latest.get("tax_provision"), latest.get("pretax_income"), latest.get("invested_capital")
    f = "operating income x (1 - tax provision / pretax income) / invested capital"
    why = has(latest, "operating_income", "tax_provision", "pretax_income", "invested_capital")
    if not why:
        rate = tax.value / pretax.value if pretax.value > 0 else None
        if rate is None or not 0 <= rate <= 1:
            why = "effective tax rate is not meaningful (pretax income not positive or tax outside 0 to 100%)"
        elif ic.value <= 0:
            why = "invested capital is not positive"
    out.append(_missing("roic", "Return on invested capital", "fraction", f, why) if why else
               build("roic", "Return on invested capital", "fraction", f, op.value * (1 - rate) / ic.value, op, tax, pretax, ic))
    return out
