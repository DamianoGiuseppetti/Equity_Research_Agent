"""The outside world the collectors talk to, in one place, so tests can swap in fakes."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import data_dir
from .llm import LLMClient, get_llm
from .export import write_pdf
from .fred import FredClient, FredProvider
from .market import MarketDataProvider, YahooMarket
from .news import ArticleFetcher, FinnhubClient, HttpArticleFetcher, NewsProvider
from .sections import EdgarToolsSections, SectionsProvider
from .sources import EdgarClient


@dataclass
class Services:
    market: MarketDataProvider = field(default_factory=YahooMarket)
    fred: FredProvider | None = None   # built from FRED_API_KEY on first use
    sections: SectionsProvider | None = None   # edgartools, built on first use
    news: NewsProvider | None = None           # Finnhub, built from FINNHUB_API_KEY on first use
    articles: ArticleFetcher | None = None     # fetches and extracts article pages
    pdf_writer: Callable[..., Any] = write_pdf    # (markdown, header line, path); tests pass a fake
    llm_factory: Callable[..., Any] = get_llm   # role -> chat model; tests pass a fake one
    edgar: Any = None          # an EdgarClient, or a fake with the same methods; built from EDGAR_USER_AGENT on first use

    def edgar_client(self):
        if self.edgar is None:
            self.edgar = EdgarClient(os.environ.get("EDGAR_USER_AGENT", ""), data_dir())
        return self.edgar

    def fred_client(self):
        if self.fred is None:
            self.fred = FredClient(os.environ.get("FRED_API_KEY", ""))
        return self.fred

    def sections_provider(self):
        if self.sections is None:
            self.sections = EdgarToolsSections(os.environ.get("EDGAR_USER_AGENT", ""))
        return self.sections

    def news_client(self):
        if self.news is None:
            self.news = FinnhubClient(os.environ.get("FINNHUB_API_KEY", ""))
        return self.news

    def article_fetcher(self):
        if self.articles is None:
            self.articles = HttpArticleFetcher(f"EquityResearchAgent/0.1 (personal research project; {os.environ.get('EDGAR_USER_AGENT', '')})")
        return self.articles

    def llm_client(self, run_folder: Path) -> LLMClient:
        return LLMClient(run_folder, self.llm_factory)
