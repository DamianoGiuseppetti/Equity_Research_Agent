"""Market data provider (yfinance) behind a small interface. Returns plain Python data so it can be cached as JSON."""
import math
import warnings
from typing import Any, Protocol

# yfinance 1.7 triggers pandas deprecation warnings from worker threads, where catch_warnings does not reach
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

STATEMENT_ATTRS = {
    "income_annual": "income_stmt", "income_quarterly": "quarterly_income_stmt",
    "balance_annual": "balance_sheet", "balance_quarterly": "quarterly_balance_sheet",
    "cashflow_annual": "cashflow", "cashflow_quarterly": "quarterly_cashflow",
}
INFO_KEYS = ("sector", "industry", "sharesOutstanding", "beta", "marketCap", "currency", "financialCurrency",
             "exchange", "longName")


PEER_KEYS = ("longName", "marketCap", "enterpriseToEbitda", "trailingPE", "enterpriseToRevenue", "exchange")


class MarketDataProvider(Protocol):
    def peer_info(self, ticker: str) -> dict:
        """The PEER_KEYS that yfinance has for one ticker (trailing multiples), for the peer comparison."""
        ...

    def fetch(self, ticker: str) -> dict:
        """Return {"prices": [...], "statements": {...}, "info": {...}} for one ticker.

        prices: [{"date": "2021-10-25", "open", "high", "low", "close", "volume"}], oldest first, 5 years.
        statements: name -> {period end "2025-09-27": {line item: value}}; names are STATEMENT_ATTRS keys.
        info: the INFO_KEYS that yfinance has.
        """
        ...


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        return None if math.isnan(value) else float(value)
    except (TypeError, ValueError):
        return value


class YahooMarket:
    def peer_info(self, ticker: str) -> dict:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = yf.Ticker(ticker).info or {}
        return {k: _clean(info.get(k)) for k in PEER_KEYS}

    def fetch(self, ticker: str) -> dict:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # yfinance 1.7 triggers pandas deprecation warnings
            t = yf.Ticker(ticker)
            history = t.history(period="5y", interval="1d", auto_adjust=False)
            prices = [{"date": ts.date().isoformat(), "open": _clean(r.Open), "high": _clean(r.High),
                       "low": _clean(r.Low), "close": _clean(r.Close), "volume": _clean(r.Volume)}
                      for ts, r in history.iterrows()]
            statements: dict[str, dict] = {}
            for name, attr in STATEMENT_ATTRS.items():
                frame = getattr(t, attr)
                statements[name] = {} if frame is None or frame.empty else {
                    col.date().isoformat(): {str(row): _clean(v) for row, v in frame[col].items()}
                    for col in frame.columns}
            info = t.info or {}
        return {"prices": prices, "statements": statements, "info": {k: info.get(k) for k in INFO_KEYS}}
