"""Sentiment: one LLM call per article, validators in the schema, aggregate in plain code.

Label means the effect on the company's shareholders, not the mood of the writing. An article whose answer fails
validation (after the wrapper's one retry) stays unscored and is counted. Nothing is ever filled in.
"""
import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .extraction import normalise
from .llm import LLMClient, LLMOutputError
from .schemas import Fact

Label = Literal["positive", "neutral", "negative"]
Topic = Literal["earnings", "guidance", "legal", "ma", "product", "macro", "other"]
CONCURRENCY = 4      # articles scored at once

SYSTEM = (
    "You score one news article by its effect on the shareholders of {company} ({ticker}).\n"
    "- label is the effect on this company's shareholders, not the mood of the writing. General market commentary that says "
    "nothing about {company}, or an article mainly about another company, is neutral.\n"
    "- score runs from -1 (very negative for shareholders) to 1 (very positive), consistent with the label; neutral is near 0.\n"
    "- confidence runs from 0 to 1.\n"
    "- topic is one of: earnings, guidance, legal, ma, product, macro, other.\n"
    "- evidence is a short span of 5 to 25 words copied exactly, character for character, from the ARTICLE TEXT. "
    "It is checked mechanically."
)


class SentimentScore(BaseModel):
    """The score of one article."""

    label: Label
    score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    topic: Topic
    evidence: str = Field(min_length=10)


def schema_for(article_text: str) -> type[SentimentScore]:
    """The schema with one more rule: the evidence must appear in this article's text (after the usual normalising)."""
    text_norm = normalise(article_text)

    class SentimentCheckedScore(SentimentScore):
        """The score of one article."""

        @field_validator("evidence")
        @classmethod
        def evidence_is_in_the_article(cls, v: str) -> str:
            if normalise(v) not in text_norm:
                raise ValueError("evidence is not a verbatim span of the article text")
            return v

    return SentimentCheckedScore


def first_words(text: str, n: int) -> str:
    """The first n words with their original spacing."""
    words = list(re.finditer(r"\S+", text))
    return text if len(words) <= n else text[: words[n - 1].end()]


@dataclass
class Scored:
    key: str
    headline: str
    source: str
    url: str
    published_at: str
    score: SentimentScore | None      # None means unscored


@dataclass
class SentimentResult:
    articles: list[Scored] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)


async def score_articles(client: LLMClient, company: str, ticker: str, items: list[dict], cap_words: int) -> list[Scored]:
    """items: dicts with key, headline, source, url, published_at, text. Order is kept."""
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(it: dict) -> Scored:
        text = first_words(it["text"], cap_words)
        messages = [("system", SYSTEM.format(company=company, ticker=ticker)),
                    ("human", f"Headline: {it['headline']}\nPublisher: {it['source']}\n\nARTICLE TEXT:\n{text}")]
        async with gate:
            try:
                score = await client.structured("extract", schema_for(text), messages, "sentiment.article", max_tokens=600)
            except LLMOutputError:
                score = None
        return Scored(it["key"], it["headline"], it["source"], it["url"], it["published_at"], score)

    return list(await asyncio.gather(*[one(i) for i in items]))


def aggregate(scored: list[Scored], counts: dict, window_days: int, min_scored: int) -> dict:
    """n_* counts, window, and (unless coverage is low) the mean score and the shares. Plain arithmetic."""
    ok = [s.score for s in scored if s.score is not None]
    out = {"n_fetched": counts["fetched"], "n_dropped_no_text": counts["dropped_no_text"], "n_selected": counts["selected"],
           "n_scored": len(ok), "n_unscored": len(scored) - len(ok), "window_days": window_days, "low_coverage": len(ok) < min_scored}
    if not out["low_coverage"]:
        out["mean_score"] = sum(s.score for s in ok) / len(ok)
        for label in ("positive", "neutral", "negative"):
            out[f"share_{label}"] = sum(s.label == label for s in ok) / len(ok)
    return out


def to_facts(scored: list[Scored], summary: dict, ticker: str, now: datetime, retrieved_at: datetime) -> list[Fact]:
    """Article scores and the aggregate: all kind model_estimate, sourced to Finnhub with the article URL."""
    facts = []
    for s in scored:
        if s.score is None:
            continue
        facts.append(Fact(id=f"news.sentiment.{s.key}.score", value=s.score.score, unit="score", label=f"Sentiment score: {s.headline[:80]}",
                          kind="model_estimate", source=f"Finnhub company-news, publisher {s.source or 'unknown'}; {s.url}",
                          as_of=s.published_at[:10], retrieved_at=retrieved_at,
                          note=(f"LLM score from -1 to 1 of the effect on shareholders; label {s.score.label}, topic {s.score.topic}, "
                                f"confidence {s.score.confidence:.2f}; the evidence span is kept in the local news/sentiment.json, "
                                f"not in the ledger, because article text stays out of exported files")))
    src = "computed in code from the article scores (Finnhub company-news articles, scored by the model)"
    note = f"over {summary['n_scored']} scored articles in a {summary['window_days']}-day window; every score is a model estimate"
    add = lambda slug, value, unit, label, kind="model_estimate": facts.append(Fact(
        id=f"news.sentiment.{slug}", value=value, unit=unit, label=f"{label} ({ticker})", kind=kind, source=src,
        as_of=now.date().isoformat(), retrieved_at=retrieved_at, note=note))
    add("n_scored", summary["n_scored"], "articles", "News articles scored", "derived")
    add("n_unscored", summary["n_unscored"], "articles", "News articles left unscored after a failed validation", "derived")
    if not summary["low_coverage"]:
        add("mean_score", summary["mean_score"], "score", "Mean news sentiment score")
        for label in ("positive", "neutral", "negative"):
            add(f"share_{label}", summary[f"share_{label}"], "fraction", f"Share of {label} articles")
    return facts
