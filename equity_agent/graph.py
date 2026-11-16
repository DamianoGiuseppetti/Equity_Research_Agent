"""The research graph: resolver, four parallel collectors, data gate, analysis, valuation, memo drafting and checks, review, feedback routing and export.

Every node is real. Only the nodes that call a model (extraction and sentiment inside the collectors, analyze, valuation's peer proposal, draft_memo and the
redraft in check_memo) need a chat model, and they get it through Services, so tests pass a fake one. The graph never imports UI code."""
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from .config import data_dir
from .nodes.analysis import make_analyze
from .nodes.valuation import make_valuation
from .nodes.collectors import COLLECTORS, make_filings, make_macro, make_market_data, make_news
from .nodes.export import make_export
from .nodes.gate import data_gate
from .nodes.memo import make_check_memo, make_draft_memo
from .nodes.review import review, route_feedback
from .nodes.session import init_session, make_resolve_company
from .resolver import Resolver
from .schemas import GraphState
from .services import Services


def default_checkpoint_path() -> Path:
    return data_dir() / "checkpoints.sqlite"


@asynccontextmanager
async def open_checkpointer(path: Path | None = None):
    """AsyncSqliteSaver on a SQLite file. Tables are created now (the saver does it lazily otherwise) and the
    connection is closed on exit; an open aiosqlite connection keeps the process alive."""
    conn = await aiosqlite.connect(path or default_checkpoint_path())
    try:
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        yield saver
    finally:
        await conn.close()


def _after_resolve(state: GraphState):
    # No company means the resolver rejected the input: end here. Otherwise start all four collectors in parallel.
    return list(COLLECTORS) if state.get("company") else END


def _after_gate(state: GraphState):
    return END if state.get("abort_reason") else "analyze"


def build_graph(checkpointer, resolver: Resolver, services: Services | None = None):
    services = services or Services()
    g = StateGraph(GraphState)
    g.add_node("init_session", init_session)
    g.add_node("resolve_company", make_resolve_company(resolver))
    for name, fn in zip(COLLECTORS, (make_market_data(services), make_filings(services), make_macro(services), make_news(services))):
        g.add_node(name, fn)
    # defer=True: run once, after every collector scheduled in this step has finished. This also works when
    # feedback re-runs a single collector, which a plain join on all four would wait for forever.
    g.add_node("data_gate", data_gate, defer=True)
    g.add_node("analyze", make_analyze(services))
    g.add_node("valuation", make_valuation(services, resolver))
    g.add_node("draft_memo", make_draft_memo(services))
    g.add_node("check_memo", make_check_memo(services))
    g.add_node("review", review)
    g.add_node("route_feedback", route_feedback, destinations=("export", "draft_memo", "valuation", "review", END, *COLLECTORS))
    g.add_node("export", make_export(services))

    g.add_edge(START, "init_session")
    g.add_edge("init_session", "resolve_company")
    g.add_conditional_edges("resolve_company", _after_resolve, [*COLLECTORS, END])
    for name in COLLECTORS:
        g.add_edge(name, "data_gate")
    g.add_conditional_edges("data_gate", _after_gate, ["analyze", END])
    g.add_edge("analyze", "valuation")
    g.add_edge("valuation", "draft_memo")
    g.add_edge("draft_memo", "check_memo")
    g.add_edge("check_memo", "review")
    g.add_edge("review", "route_feedback")
    g.add_edge("export", END)
    return g.compile(checkpointer=checkpointer)
