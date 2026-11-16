"""resolve_company logic: a name or ticker in, one verified company (or a clear rejection) out. No LLM.

Candidates come from the SEC tickers file. Each company (grouped by CIK, so GOOGL and GOOG count once) must
have a 10-K on file and a recent price on a US exchange. If the SEC file has no match, a yfinance search
runs as a fallback and the same checks apply.
"""
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Protocol

from .config import load_config
from .schemas import CompanyCandidate, ResolutionResult, ResolutionStatus
from .sources import FilingSummary, Quote, SearchHit, TickerFile, TickerRow

# Yahoo exchange codes for US exchanges. OTC codes (PNK, OQB, OQX) are deliberately absent: an OTC
# pink-sheet company passed a plain quote check in testing.
US_EXCHANGES = {"NMS": "Nasdaq", "NGM": "Nasdaq", "NCM": "Nasdaq", "NYQ": "NYSE", "ASE": "NYSE American",
                "PCX": "NYSE Arca", "BTS": "Cboe BZX"}
MAX_PRICE_AGE_TRADING_DAYS = 5
MAX_COMPANIES_CHECKED = 10        # bounds the number of SEC and yfinance calls for a broad name like "Bank"
MAX_TICKERS_PER_COMPANY = 3       # share classes tried when looking for a quote
NAME_SUFFIXES = {"inc", "corp", "corporation", "co", "company", "ltd", "plc", "sa", "spa", "the", "holdings", "group"}


class SecSource(Protocol):
    def company_tickers(self) -> TickerFile: ...
    def filing_summary(self, cik: int) -> FilingSummary: ...
    def search_10k_filers(self, name: str) -> list[tuple[int, str]]: ...


class QuoteSource(Protocol):
    def quote(self, ticker: str) -> Quote | None: ...
    def search(self, query: str) -> list[SearchHit]: ...


def core_tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", name.lower()) if t not in NAME_SUFFIXES]


def names_match(a: str, b: str) -> bool:
    """True if one name's core words are the start of the other's ('Meta' and 'Meta Platforms, Inc.')."""
    ta, tb = core_tokens(a), core_tokens(b)
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return long_[: len(short)] == short


def usable_sic(sic: str | None) -> int | None:
    """The SIC code as a number, or None when EDGAR gives nothing usable (missing, blank, 0, not numeric)."""
    try:
        code = int(sic or "")
    except ValueError:
        return None
    return code or None


@dataclass
class _Verdict:
    name: str
    status: ResolutionStatus | None      # None means the company passed every check
    message: str
    candidate: CompanyCandidate | None = None


def _oldest_allowed_bar(today: date) -> date:
    """Five weekdays back. Ignores exchange holidays, which only makes the check slightly more lenient."""
    day, steps = today, 0
    while steps < MAX_PRICE_AGE_TRADING_DAYS:
        day -= timedelta(days=1)
        if day.weekday() < 5:
            steps += 1
    return day


class Resolver:
    def __init__(self, sec: SecSource, quotes: QuoteSource, today: date | None = None,
                 financial_sic: set[int] | None = None, successors: dict[int, int] | None = None) -> None:
        self._sec, self._quotes = sec, quotes
        self._today = today or datetime.now(timezone.utc).date()
        self._financial_sic = financial_sic if financial_sic is not None else set(load_config().scope.financial_sic_codes)
        self._successors = successors if successors is not None else dict(load_config().scope.successor_ciks)

    # ---- public ---------------------------------------------------------------------------
    def resolve(self, query: str) -> ResolutionResult:
        q = query.strip()
        if not q:
            return self._not_found(q)
        rows = self._sec.company_tickers().rows
        groups = self._match_sec_rows(q, rows)
        if groups:
            truncated = len(groups) > MAX_COMPANIES_CHECKED
            verdicts = [self._check(cik, [r.ticker for r in rs], rs[0].title, prefer=q.upper())
                        for cik, rs in list(groups.items())[:MAX_COMPANIES_CHECKED]]
            return self._decide(q, verdicts, truncated)
        verdicts = self._fallback(q, rows)
        return self._decide(q, verdicts) if verdicts else self._not_found(q)

    # ---- matching -------------------------------------------------------------------------
    @staticmethod
    def _match_sec_rows(q: str, rows: list[TickerRow]) -> dict[int, list[TickerRow]]:
        """Exact ticker wins. Otherwise names that start with the query words. Grouped by CIK, file order kept."""
        exact = [r for r in rows if r.ticker.upper() == q.upper()]
        picked = exact or [r for r in rows if names_match(q, r.title)]
        groups: dict[int, list[TickerRow]] = {}
        for r in picked:
            groups.setdefault(r.cik, []).append(r)
        return groups

    def _fallback(self, q: str, rows: list[TickerRow]) -> list[_Verdict]:
        """yfinance search for a company the SEC file does not list. The 10-K check still applies.

        A hit counts only if its symbol is the query or its name matches the query, so an old name
        ('Facebook') that yfinance maps to a new company is not silently accepted.
        """
        hits = [h for h in self._quotes.search(q)
                if h.quote_type == "EQUITY" and (h.symbol.upper() == q.upper() or names_match(q, h.name))]
        verdicts = []
        for hit in hits[:MAX_TICKERS_PER_COMPANY]:
            cik = next((r.cik for r in rows if r.ticker.upper() == hit.symbol.upper()), None)
            if cik is None:
                cik = next((c for c, filer in self._sec.search_10k_filers(hit.name) if names_match(hit.name, filer)), None)
            if cik is None:
                verdicts.append(_Verdict(hit.name, "no_10k_on_file",
                                         f"{hit.name} ({hit.symbol}) is not in the SEC tickers file and EDGAR shows no 10-K "
                                         f"filed under that name, so it cannot be researched."))
                continue
            verdicts.append(self._check(cik, [hit.symbol], hit.name, prefer=hit.symbol.upper()))
        return verdicts

    # ---- checks ---------------------------------------------------------------------------
    def _check(self, cik: int, tickers: list[str], name: str, prefer: str) -> _Verdict:
        filer_cik, filer_note = self._successors.get(cik, cik), None
        summary = self._sec.filing_summary(filer_cik)
        shown = name if filer_cik != cik else (summary.name or name)   # a successor keeps its own name from the SEC tickers file
        if filer_cik != cik:
            filer_note = f"10-K filings are read from the predecessor filer {summary.name} (CIK {filer_cik})"
        if not summary.files_10k:
            others = sorted(summary.annual_forms)
            if others:
                return _Verdict(shown, "not_10k_filer",
                                f"{shown} files {', '.join(others)} with the SEC, not a 10-K, so it is outside this agent's "
                                f"scope (US companies that file a 10-K).")
            return _Verdict(shown, "no_10k_on_file",
                            f"{shown} has no 10-K on file with the SEC (for example a recent IPO, or a holding-company "
                            f"reorganisation whose new filer has not filed one yet), so it cannot be researched yet.")
        code = usable_sic(summary.sic)
        if code in self._financial_sic:
            return _Verdict(shown, "financial_company",
                            f"{shown} is a financial company (SIC {code}, {summary.sic_description}). Banks and insurers "
                            f"are outside this agent's scope because a discounted-cash-flow valuation does not fit them.")
        ordered = ([prefer] if prefer in tickers else []) + [t for t in tickers if t != prefer]
        quote = next((qt for t in ordered[:MAX_TICKERS_PER_COMPANY] if (qt := self._quotes.quote(t))), None)
        if quote is None or quote.last_price_date is None:
            return _Verdict(shown, "not_listed",
                            f"{shown} has no price quote in yfinance, so it looks delisted or unlisted.")
        if quote.quote_type != "EQUITY":
            return _Verdict(shown, "not_listed", f"{quote.ticker} is a {quote.quote_type or 'non-equity instrument'}, "
                                                 f"not a company share.")
        if quote.last_price_date < _oldest_allowed_bar(self._today):
            return _Verdict(shown, "not_listed",
                            f"{shown} ({quote.ticker}) last traded on {quote.last_price_date}, more than "
                            f"{MAX_PRICE_AGE_TRADING_DAYS} trading days ago: delisted or halted.")
        if quote.exchange not in US_EXCHANGES:
            return _Verdict(shown, "not_us_exchange",
                            f"{shown} ({quote.ticker}) trades on exchange code {quote.exchange or 'unknown'}, "
                            f"not on a US exchange (OTC markets are not accepted).")
        cand = CompanyCandidate(ticker=quote.ticker, name=shown, cik=filer_cik, exchange=US_EXCHANGES[quote.exchange],
                                market_cap=quote.market_cap, sic=summary.sic, sic_description=summary.sic_description,
                                filer_note=filer_note)
        return _Verdict(shown, None, "ok", cand)

    # ---- outcome --------------------------------------------------------------------------
    @staticmethod
    def _decide(q: str, verdicts: list[_Verdict], truncated: bool = False) -> ResolutionResult:
        passed = sorted((v.candidate for v in verdicts if v.candidate), key=lambda c: c.market_cap or 0, reverse=True)
        if len(passed) == 1:
            c = passed[0]
            return ResolutionResult(status="single", candidates=passed,
                                    message=f"Resolved '{q}' to {c.name} ({c.ticker}, {c.exchange}).")
        if len(passed) > 1:
            more = f" Only the first {MAX_COMPANIES_CHECKED} matches were checked." if truncated else ""
            return ResolutionResult(status="ambiguous", candidates=passed,
                                    message=f"{len(passed)} companies match '{q}'. Choose one by ticker.{more}")
        first = verdicts[0]  # the best-ranked match decides the class; all reasons go into the message
        reasons = " ".join(v.message for v in verdicts[:3])
        return ResolutionResult(status=first.status or "not_found", message=reasons)

    @staticmethod
    def _not_found(q: str) -> ResolutionResult:
        return ResolutionResult(status="not_found", message=(
            f"No company matches '{q}'. Old company names are not handled: enter the current company name or the ticker."))
