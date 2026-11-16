"""valuation node: plain Python except the peer proposal (analyst role). Banks and insurers never get here:
the resolver rejects them by SIC code before any collector runs."""
import asyncio
import json
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .. import valuation_facts as vf
from ..cache import cached_pull
from ..config import load_config, run_dir
from ..ledger import LedgerFile
from ..resolver import Resolver
from ..schemas import GraphState
from ..services import Services
from .common import write_facts


class PeerPick(BaseModel):
    ticker: str = Field(description="US-listed ticker symbol")
    name: str
    reason: str = Field(description="One sentence on why the company is comparable")


class PeerProposal(BaseModel):
    peers: list[PeerPick] = Field(min_length=4, max_length=6)


PEER_SYSTEM = ("You propose comparable listed companies for an equity valuation. Propose 4 to 6 US-listed, non-financial companies "
               "(not banks, insurers or asset managers) that file a 10-K with the SEC (not a 20-F, so no foreign private issuers) and trade on a US exchange, "
               "in the same line of business and of similar scale. Use the exact ticker "
               "symbol. Never include the company itself. Do not propose a company you are unsure exists.")


async def propose_peers(client, company: dict, sector: str, industry: str) -> list[str]:
    human = f"Company: {company['name']} ({company['ticker']}). Sector: {sector or 'unknown'}. Industry: {industry or 'unknown'}."
    result = await client.structured("analyst", PeerProposal, [("system", PEER_SYSTEM), ("human", human)], "valuation.peers", max_tokens=1000)
    return [p.ticker.strip().upper() for p in result.peers]


def validate_peer(ticker: str, subject: dict, resolver: Resolver, services: Services, ttl_hours: float) -> tuple[dict | None, str]:
    """(multiples, "") for a usable peer, or (None, reason). Blocking: run it in a thread."""
    if ticker == subject["ticker"].upper():
        return None, "it is the company itself"
    try:
        res = resolver.resolve(ticker)
    except Exception as exc:        # a network error on one peer must not stop the valuation
        return None, f"the resolver failed: {type(exc).__name__}"
    if res.status != "single":
        return None, f"not accepted by the resolver ({res.status}): {res.message[:120]}"
    cand = res.candidates[0]
    if cand.cik == subject["cik"] or cand.ticker.upper() != ticker:
        return None, f"resolves to {cand.ticker} ({cand.name}), not to the proposed ticker or it is the company itself"
    try:
        info = cached_pull("peer-info", ticker, ttl_hours, lambda: services.market.peer_info(ticker)).payload
    except Exception as exc:
        return None, f"yfinance multiples could not be read: {type(exc).__name__}"
    keys = {"ev_ebitda": "enterpriseToEbitda", "pe": "trailingPE", "ev_sales": "enterpriseToRevenue"}
    got = {k: info.get(src) for k, src in keys.items()}
    bad = [k for k, x in got.items() if not isinstance(x, (int, float)) or x <= 0]
    if bad:
        return None, f"multiple(s) missing or not positive: {', '.join(bad)}"
    return {**{k: float(x) for k, x in got.items()}, "name": cand.name}, ""


def make_valuation(services: Services, resolver: Resolver):
    async def valuation(state: GraphState) -> dict:
        cfg = load_config()
        folder = run_dir(state["thread_id"])
        ledger = LedgerFile(state["ledger_path"]).load()
        now = datetime.now(timezone.utc)
        inp = vf.read_inputs(ledger)
        b = vf.Builder(ledger.get("market.price.last_close").as_of if "market.price.last_close" in ledger else now.date().isoformat(), now)
        if inp is None:
            b.gap("valuation inputs", "last close or shares outstanding missing from the ledger", "no valuation in the memo")
            return {"valuation": {"rating": None}, "gaps": [g.model_dump() for g in b.gaps], "node_log": ["valuation"]}
        overrides = state.get("assumption_overrides") or {}
        vcfg = cfg.valuation.model_copy(update=overrides)       # the reviewer's values replace the defaults for the four editable assumptions
        dcf = vf.run_dcf(inp, vcfg, vcfg.cost_of_debt_fallback_spread_pts, b, {k: getattr(cfg.valuation, k) for k in overrides})

        # peers: the user's edited list, else the list from an earlier run of this node (kept stable), else a new proposal
        previous = json.loads((folder / "valuation.json").read_text()).get("proposed_peers") if (folder / "valuation.json").exists() else None
        fb = state.get("feedback") or {}
        if fb.get("action") == "edit_peers" and fb.get("peers"):
            proposed = [p.strip().upper() for p in fb["peers"] if p.strip()]
        elif previous:
            proposed = previous
        else:
            proposed = await propose_peers(services.llm_client(folder), state["company"],
                                           str(ledger.get("market.info.sector").value) if "market.info.sector" in ledger else "",
                                           str(ledger.get("market.info.industry").value) if "market.info.industry" in ledger else "")
        unique = list(dict.fromkeys(proposed))
        duplicates = sorted({t for t in proposed if proposed.count(t) > 1})      # a ticker listed twice counts once
        proposed = unique
        checked = await asyncio.gather(*[asyncio.to_thread(validate_peer, t, state["company"], resolver, services, cfg.cache.market_data_ttl_hours)
                                         for t in proposed])
        peers = {"validated": {t: m for t, (m, why) in zip(proposed, checked) if m},
                 "dropped": [{"ticker": t, "reason": why} for t, (m, why) in zip(proposed, checked) if not m]}
        comps = vf.run_comps(inp, peers, b, vcfg.min_peers)

        # rating: the DCF scenarios if there is a DCF, else the peer-multiple range
        rating, basis = (dcf["rating"], "DCF scenarios (bear, base, bull)") if dcf else \
            ((comps["rating"], "peer-multiple range (no DCF; see the data gaps)") if comps and "rating" in comps else (None, ""))
        if rating:
            b.add("rating", rating, "text", "Valuation-implied view", "derived",
                  f"{vf.LABEL}. Rule: last close below the lowest value is undervalued; up to the middle value modestly undervalued; "
                  f"up to the highest modestly overvalued; above it overvalued. Basis: {basis}.")
        result = {"proposed_peers": proposed, "listed_twice": duplicates, "peers": peers, "dcf": dcf, "comps": comps, "rating": rating, "basis": basis, "price": inp.price}
        (folder / "valuation.json").write_text(json.dumps(result, indent=1, default=str))
        update = {"valuation": {"path": str(folder / "valuation.json"), "rating": rating, "basis": basis, "dcf_computed": dcf is not None,
                                "comps_computed": comps is not None, "peers_validated": len(peers["validated"]), "peers_dropped": len(peers["dropped"]),
                                "peers": {"validated": list(peers["validated"]), "dropped": peers["dropped"], "proposed": proposed},
                                "assumptions": {k: getattr(vcfg, k) for k in ("equity_risk_premium", "terminal_growth", "scenario_spread_pts", "fade_target_growth")}},
                  "gaps": [g.model_dump() for g in b.gaps], "node_log": ["valuation"]}
        return {**update, **write_facts(state, "valuation", [f.model_dump(mode="json") for f in b.facts])}

    return valuation
