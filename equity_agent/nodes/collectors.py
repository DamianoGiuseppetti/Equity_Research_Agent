"""The four parallel collectors: market data, filings (XBRL, 10-K sections and LLM extraction), macro (FRED) and news (Finnhub, article text and LLM sentiment).

Each writes its own facts to the ledger and returns data or a typed error; a failure becomes a data gap and the run goes on."""
import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone

from ..cache import cached_pull, entry_path
from ..config import load_config, run_dir
from .. import extraction, sections, sentiment, xbrl
from .. import news as news_mod
from ..llm import RunStopped
from ..macro_facts import DCF_NOTE, series_facts
from ..market_facts import build_market_facts
from ..schemas import DataGap, Fact, GraphState
from ..services import Services
from .common import write_facts

COLLECTORS = ("market_data", "filings", "macro", "news")


async def _maybe_slow(name: str) -> None:
    """EQUITY_AGENT_SLOW_COLLECTOR='news=30' makes that collector sleep 30 s, so a run can be killed mid-way."""
    area, _, seconds = os.environ.get("EQUITY_AGENT_SLOW_COLLECTOR", "").partition("=")
    if area == name and seconds:
        await asyncio.sleep(float(seconds))


def collector(name: str, area: str, effect: str):
    """Wrap a collector so it never raises into the graph.

    A failure becomes a typed error in `collectors` plus a data gap, and the run goes on. data_gate decides
    whether the loss is fatal (no prices or no income statement) or only a gap.
    """
    def decorate(fn):
        async def run(state: GraphState) -> dict:
            await _maybe_slow(name)
            try:
                update = await fn(state)
            except RunStopped:      # the cost cap or a malformed analysis must end the run, not become a gap
                raise
            except Exception as exc:  # deliberate: any collector failure must become a gap, not a crash
                reason = f"{type(exc).__name__}: {exc}"[:300]
                gap = DataGap(area=area, missing=f"all {area} data", reason=reason, effect=effect)
                return {"collectors": {name: f"error: {reason}"}, "gaps": [gap.model_dump()], "node_log": [name]}
            return {**update, "collectors": {name: "ok"}, "node_log": [name]}
        return run
    return decorate


async def _xbrl_figures(services: Services, cik: int):
    """SEC companyfacts, cached and shared: the market and filings collectors both need it and fetch it once."""
    edgar = services.edgar_client()
    ttl = load_config().cache.filings_ttl_hours
    pull = await asyncio.to_thread(cached_pull, "xbrl", str(cik), ttl, lambda: edgar.company_facts(cik))
    return xbrl.extract(pull.payload), pull.retrieved_at


def make_market_data(services: Services):
    @collector("market_data", "market data", "prices, statements, technicals and ratios are missing")
    async def market_data(state: GraphState) -> dict:
        ticker, cik = state["company"]["ticker"], state["company"]["cik"]
        cfg = load_config()
        pull = await asyncio.to_thread(cached_pull, "market", ticker, cfg.cache.market_data_ttl_hours,
                                       lambda: services.market.fetch(ticker))
        figures, error = None, ""
        try:
            figures, _ = await _xbrl_figures(services, cik)
        except Exception as exc:  # the cross-check is optional: report it as a gap and carry on with yfinance alone
            error = f"{type(exc).__name__}: {exc}"[:200]
        result = build_market_facts(pull.payload, ticker, pull.retrieved_at, figures, cfg.checks.mismatch_tolerance, error)
        return {**write_facts(state, "market", result.facts), "gaps": [g.model_dump() for g in result.gaps],
                "coverage": {"market_data": {"fetched": result.fetched, "expected": cfg.expected.market_data_items}}}

    return market_data


async def _extract(services: Services, state: GraphState, outcomes, pull):
    """LLM extraction from the usable 10-K sections (flagged sections are not read: their span may be wrong).
    Returns (state update, ledger facts, gaps)."""
    cfg = load_config()
    usable = {o.key: o.text for o in outcomes if o.usable and o.key in extraction.FAMILIES}
    if not usable:
        return {}, [], []
    tenk = pull.payload["tenk"]
    filing = extraction.Filing(state["company"]["name"], state["company"]["ticker"], tenk["accession"], tenk["period"], pull.retrieved_at)
    result = extraction.Extraction()
    client = services.llm_client(run_dir(state["thread_id"]))
    cap_chars = round(cfg.filings.section_token_cap * sections.CHARS_PER_TOKEN)
    for key, text in usable.items():
        await extraction.extract_section(client, filing, key, text, cap_chars, result)
    facts, numbered = extraction.to_facts(result.items, filing)
    (run_dir(state["thread_id"]) / "extracts.json").write_text(json.dumps(
        {"items": numbered, "dropped": result.dropped, "failed_calls": result.failed_calls}, indent=1))
    summary = {"items": len(numbered), "dropped": sum(d["stage"].startswith("after retry") for d in result.dropped),
               "calls": result.n_calls, "failed_calls": len(result.failed_calls)}
    return {"extract_summary": summary}, facts, extraction.to_gaps(result)


def make_filings(services: Services):
    @collector("filings", "filings", "reported figures and filing sections are missing")
    async def filings(state: GraphState) -> dict:
        cfg = load_config()
        cik, ticker = state["company"]["cik"], state["company"]["ticker"]
        figures, retrieved_at = await _xbrl_figures(services, cik)
        facts = xbrl.reported_facts(figures, retrieved_at)
        gaps = [DataGap(area="filings", missing=f"XBRL {xbrl.LABELS[m].lower()}", reason="no usable fiscal-year value in SEC companyfacts",
                        effect="the memo has no reported figure for it")
                for m in ("revenue", "net_income", "diluted_eps") if not figures.values.get(m)]
        update: dict = {}
        facts_extra: list = []
        try:  # a failure in the sections must not throw away the XBRL figures
            pull = await asyncio.to_thread(cached_pull, "filings", ticker, cfg.cache.filings_ttl_hours,
                                           lambda: services.sections_provider().fetch(ticker, cik))
            outcomes = sections.analyse(pull.payload, cfg.filings.min_section_chars)
            folder = run_dir(state["thread_id"]) / "sections"
            folder.mkdir(exist_ok=True)
            paths = {}
            for o in outcomes:
                if o.text:
                    (folder / f"{o.key}.txt").write_text(o.text)
                    paths[o.key] = str(folder / f"{o.key}.txt")
            sec_facts, sec_gaps = sections.to_facts_and_gaps(outcomes, pull.payload, ticker, pull.retrieved_at, paths)
            facts += sec_facts
            gaps += sec_gaps
            update = {"sections": {o.key: {"path": paths.get(o.key, ""), "status": o.status, "chars": len(o.text or "")} for o in outcomes},
                      "coverage": {"filings": {"fetched": sum(o.usable for o in outcomes if o.key != "tenq_mdna"),
                                               "expected": cfg.expected.filings_sections}}}
            try:    # the sections are already safe; an extraction failure must not hide them
                ex_update, ex_facts, ex_gaps = await _extract(services, state, outcomes, pull)
                update.update(ex_update)
                facts_extra += ex_facts
                gaps += ex_gaps
            except RunStopped:
                raise
            except Exception as exc:
                gaps.append(DataGap(area="filings", missing="LLM extraction", reason=f"{type(exc).__name__}: {exc}"[:300],
                                    effect="the memo has no extracted segments, risks, drivers or guidance"))
        except RunStopped:
            raise
        except Exception as exc:
            gaps.append(DataGap(area="filings", missing="Items 1, 1A and 7", reason=f"{type(exc).__name__}: {exc}"[:300],
                                effect="the memo has no extracts from the 10-K"))
            update = {"coverage": {"filings": {"fetched": 0, "expected": cfg.expected.filings_sections}}}
        counts = {**write_facts(state, "filings", facts)["fact_counts"],
                  **(write_facts(state, "extract", facts_extra)["fact_counts"] if facts_extra else {})}
        return {**update, "fact_counts": counts, "gaps": [g.model_dump() for g in gaps]}

    return filings


def make_macro(services: Services):
    @collector("macro", "macro", "no macro series are available")
    async def macro(state: GraphState) -> dict:
        cfg = load_config()
        ticker = state["company"]["ticker"]
        start = (datetime.now(timezone.utc).date() - timedelta(days=800)).isoformat()   # enough for 5 quarters of GDP

        async def sector_of() -> tuple[str | None, str]:
            """The sector comes from the same cached yfinance pull the market collector uses."""
            try:
                pull = await asyncio.to_thread(cached_pull, "market", ticker, cfg.cache.market_data_ttl_hours,
                                               lambda: services.market.fetch(ticker))
                sector = pull.payload["info"].get("sector")
                return sector, "" if sector else "yfinance has no sector for this company"
            except Exception as exc:
                return None, f"market data could not be read: {type(exc).__name__}: {exc}"[:200]

        async def one(series_id: str):
            spec = cfg.macro.series[series_id]
            try:
                pull = await asyncio.to_thread(cached_pull, "fred", series_id, cfg.cache.macro_ttl_hours,
                                               lambda: services.fred_client().observations(series_id, start))
            except Exception as exc:
                effect = "the series is not shown in the macro analysis" + (DCF_NOTE if series_id == "DGS10" else "")
                return [], [DataGap(area="macro", missing=f"{spec.label} ({series_id})", reason=f"{type(exc).__name__}: {exc}"[:200],
                                    effect=effect)]
            return series_facts(series_id, spec, pull.payload, pull.retrieved_at)

        # The common series do not need the sector, so they start at once; only the sector extras wait for it.
        sector_result, *common_results = await asyncio.gather(sector_of(), *[one(i) for i in cfg.macro.common_series])
        sector, why = sector_result
        gaps: list[DataGap] = []
        if not sector:
            gaps.append(DataGap(area="macro", missing="sector-specific series", reason=why, effect="only the common macro series are shown"))
        extras = [s for s in cfg.macro.sector_series.get(sector or "", []) if s not in cfg.macro.common_series]
        extra_results = await asyncio.gather(*[one(i) for i in extras])
        facts: list = []
        for f, g in [*common_results, *extra_results]:
            facts += f
            gaps += g
        expected = len(cfg.macro.common_series) + len(extras)
        fetched = len({f.id for f in facts if f.id.endswith(".value")})
        return {**write_facts(state, "macro", facts), "gaps": [g.model_dump() for g in gaps],
                "coverage": {"macro": {"fetched": fetched, "expected": expected}}}

    return macro


def published_of(article: dict) -> str:
    return datetime.fromtimestamp(article["datetime"], timezone.utc).isoformat()


def make_news(services: Services):
    @collector("news", "news", "no news is available for sentiment scoring")
    async def news(state: GraphState) -> dict:
        cfg = load_config()
        ticker, name = state["company"]["ticker"], state["company"]["name"]
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=cfg.news.window_days)
        pull = await asyncio.to_thread(cached_pull, "news-list", ticker, cfg.cache.news_ttl_hours,
                                       lambda: services.news_client().company_news(ticker, start.date().isoformat(), now.date().isoformat()))
        sel = news_mod.select_articles(pull.payload, ticker, name, cfg.news.max_articles, start)
        fetcher = services.article_fetcher()

        async def text_of(article: dict) -> dict:
            key = news_mod.url_key(article["url"])
            got = await asyncio.to_thread(cached_pull, "news-text", key, cfg.cache.news_ttl_hours, lambda: fetcher.fetch(article["url"]))
            return {"key": key, **got.payload}

        results = await asyncio.gather(*[text_of(a) for a in sel.selected])
        records, facts, reasons = [], [], {}
        for a, r in zip(sel.selected, results):
            published = datetime.fromtimestamp(a["datetime"], timezone.utc)
            ok = r["status"] == "ok"
            records.append({"headline": a["headline"], "source": a.get("source") or "", "published_at": published.isoformat(), "url": a["url"],
                            "status": r["status"], "reason": r["reason"], "chars": r["chars"],
                            "text_path": str(entry_path("news-text", r["key"])) if ok else ""})
            if ok:
                facts.append(Fact(id=f"news.article.{r['key']}", value=a["headline"], unit="text", label=f"News article ({a.get('source') or 'unknown source'})",
                                  kind="reported", source=f"Finnhub company-news, publisher {a.get('source') or 'unknown'}",
                                  as_of=published.date().isoformat(), retrieved_at=pull.retrieved_at,
                                  note=f"url {a['url']}; full text in the local cache only"))
            else:
                reason = re.sub(r"\(\d+ characters\)", "(N characters)", r["reason"])   # group "too short" reasons, keep HTTP codes
                reasons[reason] = reasons.get(reason, 0) + 1
        n_text = sum(r["status"] == "ok" for r in results)
        counts = {"fetched": sel.n_fetched, "duplicates": sel.n_duplicates, "not_mentioning_company": sel.n_irrelevant,
                  "outside_window": sel.n_outside_window, "selected": len(sel.selected), "with_full_text": n_text,
                  "dropped_no_text": len(sel.selected) - n_text}
        for k, v in counts.items():
            facts.append(Fact(id=f"news.n_{k}", value=v, unit="articles", label=f"News articles: {k.replace('_', ' ')} ({ticker})", kind="derived",
                              source="counted in code from the Finnhub company-news response", as_of=now.date().isoformat(),
                              retrieved_at=pull.retrieved_at, note=f"{cfg.news.window_days}-day window; at most {cfg.news.max_articles} newest articles that mention the company"))
        folder = run_dir(state["thread_id"]) / "news"
        folder.mkdir(exist_ok=True)
        (folder / "articles.json").write_text(json.dumps(records, indent=1))
        gaps = []
        # sentiment: one LLM call per article that has full text (the text comes from the cache and is never written elsewhere)
        to_score = [{"key": r["key"], "headline": a["headline"], "source": a.get("source") or "", "url": a["url"],
                     "published_at": published_of(a), "text": r["text"]}
                    for a, r in zip(sel.selected, results) if r["status"] == "ok"]
        scored = await sentiment.score_articles(services.llm_client(run_dir(state["thread_id"])), name, ticker, to_score, cfg.news.text_cap_words)
        sent = sentiment.aggregate(scored, counts, cfg.news.window_days, cfg.news.low_coverage_min_scored)
        facts += sentiment.to_facts(scored, sent, ticker, now, pull.retrieved_at)
        (folder / "sentiment.json").write_text(json.dumps(
            [{"url": s.url, "headline": s.headline, "published_at": s.published_at, "scored": s.score.model_dump() if s.score else None}
             for s in scored] , indent=1))
        if sent["low_coverage"]:
            gaps.append(DataGap(area="news", missing="enough articles for a sentiment average",
                                reason=f"{sent['n_scored']} articles scored, fewer than the {cfg.news.low_coverage_min_scored} needed",
                                effect="the memo says news coverage is thin and shows no average sentiment"))
        if sent["n_unscored"]:
            gaps.append(DataGap(area="news", missing=f"sentiment score for {sent['n_unscored']} of {sent['n_selected'] - sent['n_dropped_no_text']} articles with text",
                                reason="the model's answer failed validation twice (score range, label, topic or evidence not found in the text)",
                                effect="those articles are left unscored; the average rests on the others"))
        if counts["dropped_no_text"]:
            gaps.append(DataGap(area="news", missing=f"full text of {counts['dropped_no_text']} of {counts['selected']} selected articles",
                                reason="; ".join(f"{n} x {r}" for r, n in reasons.items()),
                                effect="those articles are not scored; sentiment rests on the articles with full text"))
        return {**write_facts(state, "news", facts), "gaps": [g.model_dump() for g in gaps],
                "sentiment_summary": sent, "news_summary": {**counts, "drop_reasons": reasons, "records_path": str(folder / "articles.json")},
                "coverage": {"news": {"fetched": n_text, "expected": cfg.news.max_articles}}}

    return news
