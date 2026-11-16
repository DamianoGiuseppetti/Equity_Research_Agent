"""News: Finnhub company-news (unscored articles), relevance filter, selection and full-text fetch.

No provider sentiment field is read. Article text goes to the cache folder only and is never written to a tracked file.
Finnhub's licence is personal use only: the memo shows headline, source, date and URL, never the text.
"""
import hashlib
import re
import threading
import time
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol
from urllib.parse import urljoin, urlparse

import httpx
import requests

COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
MIN_TEXT_CHARS = 500          # shorter extracted text is a paywall stub, consent page or script-only page
HOST_GAP_SECONDS = 2.0        # pause between two requests to the same site
MAX_REDIRECTS = 6
FIRST_TOKEN_STOPLIST = {"united", "general", "american", "national", "first", "global", "international", "new"}
NAME_SUFFIXES = {"inc", "corp", "corporation", "co", "company", "companies", "ltd", "plc", "sa", "spa", "the", "holdings", "group"}


class NewsError(Exception):
    """Finnhub could not be reached or refused the request."""


class NewsProvider(Protocol):
    def company_news(self, ticker: str, start: str, end: str) -> list[dict]:
        """[{"headline", "summary", "url", "source", "datetime" (epoch seconds)}], any order."""
        ...


class ArticleFetcher(Protocol):
    def fetch(self, url: str) -> dict:
        """{"status": "ok" | "fail", "chars": int, "reason": str, "text": str}. Never raises."""
        ...


class FinnhubClient:
    def __init__(self, api_key: str, sleep: Callable[[float], None] = time.sleep, get=requests.get) -> None:
        if not api_key:
            raise NewsError("FINNHUB_API_KEY is not set")
        self._key, self._sleep, self._get = api_key, sleep, get

    def company_news(self, ticker: str, start: str, end: str) -> list[dict]:
        params = {"symbol": ticker, "from": start, "to": end, "token": self._key}
        last = "no attempt made"
        for attempt in range(3):
            try:
                r = self._get(COMPANY_NEWS_URL, params=params, timeout=30)
            except requests.RequestException as exc:
                last = f"network error: {type(exc).__name__}"
            else:
                if r.status_code == 200:
                    return [{k: a.get(k) for k in ("headline", "summary", "url", "source", "datetime")} for a in r.json()]
                last = f"HTTP {r.status_code}"      # 429 means the 60 calls per minute limit was hit
                if r.status_code != 429 and r.status_code < 500:
                    break
            self._sleep(1.0 * 2 ** attempt)
        raise NewsError(f"Finnhub company-news for {ticker} failed ({last})")  # never includes the token


class HttpArticleFetcher:
    """Fetch an article page politely and extract its text with trafilatura.

    robots.txt is checked on every redirect hop, requests to one site are at least two seconds apart, and the
    User-Agent says who is asking. A page that cannot be fetched is reported with the reason, never guessed.
    """

    def __init__(self, user_agent: str, transport: httpx.BaseTransport | None = None) -> None:
        self._ua = user_agent
        self._http = httpx.Client(headers={"User-Agent": user_agent}, timeout=20.0, transport=transport)
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._host_lock: dict[str, threading.Lock] = {}
        self._last: dict[str, float] = {}
        self._guard = threading.Lock()

    def _allowed(self, url: str) -> bool:
        parts = urlparse(url)
        with self._guard:
            if parts.netloc not in self._robots:
                rp = urllib.robotparser.RobotFileParser()
                try:
                    r = self._http.get(f"{parts.scheme}://{parts.netloc}/robots.txt", follow_redirects=True, timeout=10)
                    rp.parse(r.text.splitlines() if r.status_code == 200 else [])
                except httpx.HTTPError:
                    rp = None   # robots.txt unreachable: no rules known, so allowed
                self._robots[parts.netloc] = rp
            rp = self._robots[parts.netloc]
        return True if rp is None else rp.can_fetch(self._ua, url)

    def _pace(self, host: str) -> None:
        with self._guard:
            lock = self._host_lock.setdefault(host, threading.Lock())
        with lock:
            wait = HOST_GAP_SECONDS - (time.time() - self._last.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.time()

    def fetch(self, url: str) -> dict:
        import trafilatura

        fail = lambda reason, chars=0: {"status": "fail", "chars": chars, "reason": reason, "text": ""}
        try:
            current = url
            for _ in range(MAX_REDIRECTS):
                host = urlparse(current).netloc
                if not self._allowed(current):
                    return fail(f"robots.txt disallows {host}")
                self._pace(host)
                r = self._http.get(current, follow_redirects=False)
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    current = urljoin(current, r.headers["location"])
                    continue
                if r.status_code in (401, 402):
                    return fail(f"HTTP {r.status_code} (paywall or login)")
                if r.status_code == 403:
                    return fail("HTTP 403 (blocked the request)")
                if r.status_code == 429:
                    return fail("HTTP 429 (rate limited)")
                if r.status_code != 200:
                    return fail(f"HTTP {r.status_code}")
                text = trafilatura.extract(r.text) or ""
                if len(text) < MIN_TEXT_CHARS:
                    return fail(f"extracted text too short ({len(text)} characters): paywall, consent page or script-only page", len(text))
                return {"status": "ok", "chars": len(text), "reason": "", "text": text}
            return fail("too many redirects")
        except Exception as exc:   # network errors and parser errors are data about this article, not crashes
            return fail(f"error: {type(exc).__name__}")


# ---- selection (pure functions) ------------------------------------------------------------
def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _name_phrases(company_name: str) -> list[str]:
    core = [t for t in _norm(company_name).split() if t not in NAME_SUFFIXES]
    phrases = [" ".join(core)] if core else []
    if len(core) > 1 and len(core[0]) >= 5 and core[0] not in FIRST_TOKEN_STOPLIST:
        phrases.append(core[0])
    return phrases


def mentions(article: dict, ticker: str, company_name: str) -> bool:
    """The ticker (as an upper-case word) or the company name appears in the headline or the summary."""
    text = f"{article.get('headline') or ''} {article.get('summary') or ''}"
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", text):
        return True
    flat = f" {_norm(text)} "
    return any(f" {p} " in flat for p in _name_phrases(company_name))


@dataclass
class Selection:
    selected: list[dict]
    n_fetched: int
    n_duplicates: int
    n_irrelevant: int
    n_outside_window: int


def select_articles(raw: list[dict], ticker: str, company_name: str, max_articles: int, window_start: datetime) -> Selection:
    """Dedupe by normalised headline, keep articles inside the window that mention the company, newest first, at most N."""
    seen: set[str] = set()
    kept, dup, irrelevant, old = [], 0, 0, 0
    for a in sorted((a for a in raw if a.get("headline") and a.get("url") and a.get("datetime")), key=lambda x: x["datetime"], reverse=True):
        if datetime.fromtimestamp(a["datetime"], timezone.utc) < window_start:
            old += 1
        elif _norm(a["headline"]) in seen:
            dup += 1
        elif not mentions(a, ticker, company_name):
            irrelevant += 1
        else:
            kept.append(a)
        seen.add(_norm(a["headline"]))
    return Selection(kept[:max_articles], len(raw), dup, irrelevant, old)


def url_key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:10]
