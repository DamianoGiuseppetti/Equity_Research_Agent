"""init_session and resolve_company."""
import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from ..config import ROLES, active_profile, model_id, run_dir
from ..resolver import Resolver, usable_sic
from ..schemas import DataGap, GraphState


async def init_session(state: GraphState, config: RunnableConfig) -> dict:
    """Record the thread id, the model profile and the model IDs it maps to. No model is called."""
    if not state.get("user_input", "").strip():
        raise ValueError("user_input is empty: pass a company name or ticker")
    return {
        "thread_id": config["configurable"]["thread_id"],
        "ledger_path": str(run_dir(config["configurable"]["thread_id"]) / "ledger.json"),
        "profile": active_profile(),
        "model_ids": {role: model_id(role) for role in ROLES},
        "node_log": ["init_session"],
    }


def make_resolve_company(resolver: Resolver):
    async def resolve_company(state: GraphState) -> dict:
        """Real resolver. If several companies match, ask the user to choose.

        On resume LangGraph runs this node again from its first line, so the resolver runs a second time.
        It only reads, so that costs a few requests and has no side effect.
        """
        result = await asyncio.to_thread(resolver.resolve, state["user_input"])
        update = {"resolution": result.model_dump(), "node_log": ["resolve_company"]}
        if result.status == "single":
            return {**update, **_company_update(result.candidates[0])}
        if result.status != "ambiguous":
            return update  # a rejection: no company, the graph ends and the CLI prints result.message
        choice = interrupt({"type": "choose_company", "message": result.message,
                            "candidates": [c.model_dump() for c in result.candidates]})
        picked = next((c for c in result.candidates if c.ticker.upper() == str(choice).strip().upper()), None)
        if picked is None:
            raise ValueError(f"{choice!r} is not one of the offered tickers: {[c.ticker for c in result.candidates]}")
        return {**update, **_company_update(picked)}

    return resolve_company


def _company_update(company) -> dict:
    """The chosen company, plus a gap if EDGAR's SIC code is unusable (the bank and insurer filter could not run)."""
    update: dict = {"company": company.model_dump()}
    if usable_sic(company.sic) is None:
        update["gaps"] = [DataGap(area="company", missing="SIC code", reason="EDGAR returned no usable SIC code",
                                  effect="could not confirm the company is not a bank or insurer").model_dump()]
    return update
