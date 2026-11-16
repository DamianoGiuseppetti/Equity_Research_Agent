"""Real data sources for the resolver: the SEC (tickers file, filing list, 10-K name search) and yfinance.

Plain synchronous code. Graph nodes call it through asyncio.to_thread.
The SEC client sends a User-Agent on every request, keeps under 10 requests per second and retries a bounded number of times.
"""
import json
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

MAX_REQUESTS_PER_SECOND = 8      # the SEC limit is 10; stay clear of it
MAX_ATTEMPTS = 3                 # one try plus two retries on 429 or 5xx
BACKOFF_SECONDS = 1.0            # doubles on each retry
TICKERS_MAX_AGE_HOURS = 24       # the file was about two days old when first checked


class SecError(Exception):
    """The SEC could not be reached or answered with an error we cannot recover from."""


class RateLimiter:
    """At most max_per_second calls in any one-second window. Thread-safe."""

    def __init__(self, max_per_second: int = MAX_REQUESTS_PER_SECOND,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self._max = max_per_second
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._stamps: list[float] = []

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            self._stamps = [t for t in self._stamps if now - t < 1.0]
            if len(self._stamps) >= self._max:
                self._sleep(1.0 - (now - self._stamps[0]))
                now = self._clock()
                self._stamps = [t for t in self._stamps if now - t < 1.0]
            self._stamps.append(now)


@dataclass(frozen=True)
class TickerRow:
    cik: int
    ticker: str
    title: str


@dataclass(frozen=True)
class TickerFile:
    rows: list[TickerRow]
    fetched_at: datetime


@dataclass(frozen=True)
class FilingSummary:
    name: str
    annual_forms: frozenset[str]     # which of 10-K, 20-F, 40-F appear in the recent filing list
    sic: str | None = None           # EDGAR SIC code as text, None when EDGAR gives none
    sic_description: str | None = None

    @property
    def files_10k(self) -> bool:
        return "10-K" in self.annual_forms


@dataclass(frozen=True)
class Quote:
    ticker: str
    quote_type: str                  # EQUITY, ETF, ...
    exchange: str                    # Yahoo exchange code, e.g. NMS, NYQ, PNK
    last_price_date: date | None     # date of the newest daily bar
    market_cap: float | None


@dataclass(frozen=True)
class SearchHit:
    symbol: str
    name: str
    quote_type: str


class EdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path, transport: httpx.BaseTransport | None = None,
                 limiter: RateLimiter | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError("EDGAR_USER_AGENT must hold a name and an email, e.g. 'Your Name you@example.com'")
        self._http = httpx.Client(headers={"User-Agent": user_agent}, timeout=30.0, transport=transport)
        self._limiter = limiter or RateLimiter()
        self._sleep = sleep
        self._cache_dir = cache_dir

    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        last = "no attempt made"
        for attempt in range(MAX_ATTEMPTS):
            self._limiter.acquire()
            try:
                r = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                last = f"network error: {type(exc).__name__}"
            else:
                if r.status_code == 200:
                    return r
                last = f"HTTP {r.status_code}"
                if r.status_code != 429 and r.status_code < 500:
                    break  # a 4xx other than 429 will not get better on retry
            self._sleep(BACKOFF_SECONDS * 2 ** attempt)
        raise SecError(f"SEC request failed ({last}): {url}")

    def company_tickers(self) -> TickerFile:
        """The SEC tickers file, cached on disk with its fetch time. A stale cache is used if the SEC is down."""
        data_path = self._cache_dir / "sec_company_tickers.json"
        meta_path = self._cache_dir / "sec_company_tickers.meta.json"
        if data_path.exists() and meta_path.exists():
            fetched = datetime.fromisoformat(json.loads(meta_path.read_text())["fetched_at"])
            if (datetime.now(timezone.utc) - fetched).total_seconds() < TICKERS_MAX_AGE_HOURS * 3600:
                return TickerFile(_parse_rows(json.loads(data_path.read_text())), fetched)
        try:
            raw = self._get(TICKERS_URL).json()
        except SecError:
            if data_path.exists() and meta_path.exists():
                fetched = datetime.fromisoformat(json.loads(meta_path.read_text())["fetched_at"])
                return TickerFile(_parse_rows(json.loads(data_path.read_text())), fetched)
            raise
        fetched = datetime.now(timezone.utc)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps(raw))
        meta_path.write_text(json.dumps({"fetched_at": fetched.isoformat()}))
        return TickerFile(_parse_rows(raw), fetched)

    def filing_summary(self, cik: int) -> FilingSummary:
        j = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        forms = set(j["filings"]["recent"]["form"])
        return FilingSummary(name=j.get("name", ""), annual_forms=frozenset(forms & {"10-K", "20-F", "40-F"}),
                             sic=j.get("sic") or None, sic_description=j.get("sicDescription") or None)

    def company_facts(self, cik: int) -> dict:
        """All XBRL facts of one company in a single request (can be several MB)."""
        return self._get(COMPANY_FACTS_URL.format(cik=cik)).json()

    def search_10k_filers(self, name: str) -> list[tuple[int, str]]:
        """EDGAR full-text search for 10-K filings that mention `name`. Returns (cik, filer name) pairs.

        Used only by the yfinance fallback. The caller must still compare the filer name with the
        company it is looking for, because the search also matches filings that merely mention the name.
        """
        j = self._get(FULL_TEXT_SEARCH_URL, params={"q": f'"{name}"', "forms": "10-K"}).json()
        found: list[tuple[int, str]] = []
        for hit in j.get("hits", {}).get("hits", []):
            src = hit["_source"]
            for cik, display in zip(src.get("ciks", []), src.get("display_names", [])):
                pair = (int(cik), display.split("  (")[0].strip())
                if pair not in found:
                    found.append(pair)
        return found


def _parse_rows(raw: dict) -> list[TickerRow]:
    return [TickerRow(cik=int(v["cik_str"]), ticker=v["ticker"], title=v["title"]) for v in raw.values()]


class YahooQuotes:
    """Listing checks through yfinance. Imported lazily so offline tests never touch the network."""

    def quote(self, ticker: str) -> Quote | None:
        import yfinance as yf

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # yfinance 1.7 triggers pandas deprecation warnings
                t = yf.Ticker(ticker)
                bars = t.history(period="10d")
                if bars.empty:
                    return None
                info = t.fast_info
                return Quote(ticker=ticker, quote_type=str(info.get("quoteType") or ""),
                             exchange=str(info.get("exchange") or ""), last_price_date=bars.index[-1].date(),
                             market_cap=info.get("marketCap"))
        except Exception:  # yfinance raises many kinds of errors; for a listing check they all mean "no quote"
            return None

    def search(self, query: str) -> list[SearchHit]:
        import yfinance as yf

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                quotes = yf.Search(query, max_results=8).quotes
        except Exception:
            return []
        return [SearchHit(symbol=q["symbol"], name=q.get("longname") or q.get("shortname") or "",
                          quote_type=q.get("quoteType", "")) for q in quotes if q.get("symbol")]
