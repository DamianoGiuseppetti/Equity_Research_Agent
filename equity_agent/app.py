"""The Gradio app: the only module that imports Gradio. The graph never imports this file.

Flow: company box, candidate confirmation, progress per node, review page (approve, request changes, assumptions, edit peers, new
data, cancel), export. Every graph segment goes through runner.run_segment, as in the CLI, so the LangSmith link is recorded and
shown for every segment, including one that stopped on an error or the cost cap.

Gradio details that each need a line below: the saver's tables are created at startup (open_checkpointer calls setup); a dropdown is
refreshed by returning a new gr.Dropdown, never gr.update(choices=...); launch() is called without show_api; a dropdown selection
opens the thread from the event's own value (a click right after a selection read an empty value once); waiting threads are read
from a small index table, not by asking the graph about every thread; the aiosqlite connection is closed on exit.
"""
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from langgraph.types import Command

from .cli import make_resolver
import os

from .config import ROLES, active_profile, data_dir, load_config, load_env_file, model_id, run_dir
from .feedback import ASSUMPTION_RANGES
from .graph import build_graph, open_checkpointer
from .memo_writer import SECTIONS
from .nodes.collectors import COLLECTORS
from .runner import Segment, return_to_review, run_segment
from .trace import TraceLinks

NODE_LABELS = {"init_session": "Session started", "resolve_company": "Company resolved", "market_data": "Market data collected",
               "filings": "10-K sections read and extracted", "macro": "Macro series collected", "news": "News collected and scored",
               "data_gate": "Data gate (gaps and coverage)", "analyze": "Analysis written", "valuation": "Valuation computed",
               "draft_memo": "Memo drafted", "check_memo": "Memo checked", "review": "Waiting for your review", "route_feedback": "Feedback routed",
               "export": "Export"}

# Typical cost of one first draft per profile, from the cost.json logs of the first-draft runs of 2026-11-05 (KO, XOM and AAPL on each
# profile): demo (Sonnet) $0.4608, $0.6272 and $0.7013; dev (Haiku) $0.0414 to $0.0531.
TYPICAL_COST = {"demo": "$0.46 to $0.70", "dev": "$0.04 to $0.05"}


def profile_banner() -> str:
    """What this process will spend: profile, model by role, the per-run cap and the typical cost of a first draft."""
    profile = active_profile()
    cap = float(os.environ.get("EQUITY_AGENT_MAX_RUN_COST_USD") or load_config().cost.max_run_cost_usd[profile])
    models = ", ".join(f"{role} = `{model_id(role)}`" for role in ROLES)
    return (f"**Model profile: `{profile}`.** {models}.  \n"
            f"Run cost cap: ${cap:.2f} per thread, revisions included. Typical cost of a first draft: {TYPICAL_COST.get(profile, 'not measured')} "
            "(from the cost logs of three first-draft runs on 2026-11-05). Each revision round adds to it.")


_state: dict = {}      # graph, saver connection and resolver, built on first use inside Gradio's event loop


# ---- startup and the thread index ----------------------------------------------------------------------
async def get_graph():
    if "graph" not in _state:
        ctx = open_checkpointer()
        saver = await ctx.__aenter__()          # creates the checkpoint tables now
        _state.update(ctx=ctx, saver=saver, conn=saver.conn, graph=build_graph(saver, make_resolver()), resolver=make_resolver())
        await _state["conn"].execute("create table if not exists app_threads (thread_id text primary key, label text, status text, updated_at text)")
        await _state["conn"].commit()
        await _backfill_index()
    return _state["graph"]


async def close_app() -> None:
    if "ctx" in _state:
        await _state.pop("ctx").__aexit__(None, None, None)      # closes the aiosqlite connection, which would otherwise keep the process alive
        _state.clear()


def cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def status_of(snap) -> str:
    v = snap.values
    interrupts = [i.value for t in snap.tasks for i in t.interrupts]
    if interrupts:
        return {"choose_company": "waiting_company", "confirm_overwrite": "waiting_overwrite"}.get(interrupts[0].get("type"), "waiting_review")
    if snap.next:
        return "stopped"
    if not v.get("company"):
        return "rejected"
    if v.get("abort_reason"):
        return "aborted"
    return "approved" if v.get("approved") else "cancelled"


async def index_thread(thread_id: str) -> None:
    snap = await _state["graph"].aget_state(cfg(thread_id))
    label = ((snap.values.get("company") or {}).get("ticker") or snap.values.get("user_input") or "?")
    await _state["conn"].execute("insert or replace into app_threads values (?, ?, ?, ?)",
                                 (thread_id, label, status_of(snap), snap.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")))
    await _state["conn"].commit()


async def _backfill_index(limit: int = 100) -> None:
    """Threads started before the index existed (for example by the CLI): add the most recent ones once."""
    cur = await _state["conn"].execute("select thread_id from checkpoints group by thread_id order by max(checkpoint_id) desc limit ?", (limit,))
    known = {r[0] for r in await (await _state["conn"].execute("select thread_id from app_threads")).fetchall()}
    for (tid,) in await cur.fetchall():
        if tid not in known:
            await index_thread(tid)


async def waiting_choices() -> list[tuple[str, str]]:
    cur = await _state["conn"].execute("select thread_id, label, status, updated_at from app_threads "
                                       "where status in ('waiting_review', 'waiting_overwrite', 'stopped') order by updated_at desc limit 50")   # updated_at is the thread's last checkpoint time
    return [(f"{label} | {status.replace('waiting_', 'waiting for ')} | {updated[:16].replace('T', ' ')} UTC | {tid[:8]}", tid) for tid, label, status, updated in await cur.fetchall()]


# ---- the screen ---------------------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Equity research agent") as demo:
        gr.Markdown("# Equity research agent\nA draft memo for a human reviewer. Valuation-implied view, not investment advice.")
        gr.Markdown(profile_banner())
        thread = gr.State("")
        with gr.Row():
            pending = gr.Dropdown(label="Threads waiting for you (resume after a restart)", choices=[], scale=4)
            open_btn = gr.Button("Open selected thread", scale=1)
            refresh_btn = gr.Button("Refresh list", scale=1)
        with gr.Group() as start_box:
            with gr.Row():
                company = gr.Textbox(label="Company name or ticker", placeholder="Coca-Cola, KO, XOM ...", scale=4)
                find_btn = gr.Button("Find company", variant="primary", scale=1)
            cand_msg = gr.Markdown("")
            cand = gr.Radio(label="Candidates (confirm one)", choices=[], visible=False)
            go_btn = gr.Button("Confirm and run", variant="primary", visible=False)
        progress = gr.Markdown("")
        status = gr.Markdown("")
        with gr.Column(visible=False) as review_col:
            notice = gr.Markdown("")
            summary = gr.Markdown("")
            with gr.Accordion("Memo (ledger ids in brackets; tables built from the ledger)", open=True):
                memo = gr.Markdown("", max_height=520)
            with gr.Accordion("Data gaps, coverage and failed checks", open=False):
                gaps_md = gr.Markdown("")
            peers_md = gr.Markdown("")
            gr.Markdown("### Your decision")
            with gr.Row():
                confirm_failed = gr.Checkbox(label="I accept the failed checks listed above (needed to approve a memo that has any)", visible=False)
                approve_btn = gr.Button("Approve and export", variant="primary")
                cancel_btn = gr.Button("Cancel (nothing is exported)")
            with gr.Accordion("Request changes to sections", open=False):
                sections = gr.CheckboxGroup(label="Sections to rewrite", choices=[s for s in SECTIONS if s != "Sources and Methodology"])
                instruction = gr.Textbox(label="Instruction (wording and emphasis only; it cannot change a number)")
                changes_btn = gr.Button("Rewrite the chosen sections")
            with gr.Accordion("Change valuation assumptions", open=False):
                gr.Markdown("Values in percent, except the spread (points of growth). WACC and other settings cannot be edited.")
                with gr.Row():
                    erp = gr.Number(label="Equity risk premium %", precision=2)
                    tg = gr.Number(label="Terminal growth %", precision=2)
                    spread = gr.Number(label="Scenario spread (points)", precision=2)
                    fade = gr.Number(label="Fade target growth %", precision=2)
                assume_btn = gr.Button("Recompute the valuation and rewrite the dependent sections")
            with gr.Accordion("Edit peers", open=False):
                peers_in = gr.Textbox(label="Peer tickers, comma separated (validated again: US-listed 10-K filers with the three multiples)")
                peers_btn = gr.Button("Recompute the comparables")
            with gr.Accordion("New data (most expensive path)", open=False):
                area = gr.Dropdown(label="Area to collect again", choices=list(COLLECTORS))
                confirm_cost = gr.Checkbox(label="I understand this collects again, re-runs the analysis and rewrites the whole memo")
                data_btn = gr.Button("Collect again")
        with gr.Column(visible=False) as done_col:
            done_md = gr.Markdown("")
            with gr.Row():
                reapprove_btn = gr.Button("Approve again (export again)", visible=False)
                yes_btn = gr.Button("Yes, overwrite the folder", visible=False, variant="primary")
                no_btn = gr.Button("No, keep it", visible=False)
                back_btn = gr.Button("Return to the review", visible=False)

        outs = [thread, pending, start_box, cand_msg, cand, go_btn, progress, status, review_col, notice, summary, memo, gaps_md, peers_md, confirm_failed,
                sections, instruction, erp, tg, spread, fade, peers_in, area, confirm_cost, done_col, done_md, reapprove_btn, yes_btn, no_btn, back_btn, company]
        C = dict(zip(["thread", "pending", "start_box", "cand_msg", "cand", "go_btn", "progress", "status", "review_col", "notice", "summary", "memo", "gaps_md",
                      "peers_md", "confirm_failed", "sections", "instruction", "erp", "tg", "spread", "fade", "peers_in", "area", "confirm_cost", "done_col",
                      "done_md", "reapprove_btn", "yes_btn", "no_btn", "back_btn", "company"], outs))

        async def view(thread_id: str, note: str = "", progress_text: str = "") -> dict:
            """Every screen update in one place: what is visible and what it shows, from the thread's stored state."""
            graph = await get_graph()
            u: dict = {C["thread"]: thread_id, C["pending"]: gr.Dropdown(choices=await waiting_choices(), value=None), C["progress"]: progress_text}
            if not thread_id:
                u.update({C["start_box"]: gr.Group(visible=True), C["review_col"]: gr.Column(visible=False), C["done_col"]: gr.Column(visible=False)})
                return u
            snap = await graph.aget_state(cfg(thread_id))
            st = status_of(snap)
            links = TraceLinks(run_dir(thread_id))
            u[C["status"]] = "\n\n".join(x for x in (note, f"`{thread_id}`  {links.text()}") if x)
            u.update({C["start_box"]: gr.Group(visible=st in ("waiting_company",)), C["review_col"]: gr.Column(visible=st == "waiting_review"),
                      C["done_col"]: gr.Column(visible=st != "waiting_review")})
            for b in ("reapprove_btn", "yes_btn", "no_btn", "back_btn"):
                u[C[b]] = gr.Button(visible=False)
            if st == "waiting_review":
                p = [i.value for t in snap.tasks for i in t.interrupts][0]
                v = snap.values
                u[C["notice"]] = f"**{p['notice']}**" if p.get("notice") else ""
                u[C["summary"]] = (f"**{(v.get('company') or {}).get('name')} ({(v.get('company') or {}).get('ticker')})** | valuation-implied view: **{p.get('rating') or 'not available'}** "
                                   f"(not investment advice) | revision round {p['round']} of 3, {p['rounds_left']} left | "
                                   f"actions: {', '.join(p['actions'])} | failed checks: {len(p['check_failures'])}")
                u[C["memo"]] = Path(p["memo_path"]).read_text()
                u[C["gaps_md"]] = _gaps_text(p)
                peers = p.get("peers") or {}
                u[C["peers_md"]] = ("**Peers used:** " + (", ".join(peers.get("validated", [])) or "none validated") +
                                    ("".join(f"\n- left out {d['ticker']}: {d['reason']}" for d in peers.get("dropped", []))))
                a = p.get("assumptions") or {}
                u.update({C["erp"]: round(a.get("equity_risk_premium", 0.05) * 100, 2), C["tg"]: round(a.get("terminal_growth", 0.025) * 100, 2),
                          C["spread"]: a.get("scenario_spread_pts", 3.0), C["fade"]: round(a.get("fade_target_growth", 0.03) * 100, 2),
                          C["peers_in"]: ", ".join(peers.get("validated", [])), C["confirm_failed"]: gr.Checkbox(value=False, visible=bool(p["check_failures"])), C["confirm_cost"]: False,
                          C["sections"]: [], C["instruction"]: ""})
            elif st == "waiting_overwrite":
                p = [i.value for t in snap.tasks for i in t.interrupts][0]
                u[C["done_md"]] = f"**{p['folder']} already exists** (thread {p.get('previous_thread')}). Overwrite it? The old text is kept as memo_previous.md."
                u[C["yes_btn"]], u[C["no_btn"]] = gr.Button(visible=True), gr.Button(visible=True)
            elif st == "waiting_company":
                u[C["done_md"]] = "The company is ambiguous: confirm one candidate above."
            else:
                v = snap.values
                if st == "approved":
                    u[C["done_md"]] = f"### Approved\n{v.get('export_message', '')}"
                    u[C["reapprove_btn"]] = gr.Button(visible=True)
                elif st == "cancelled":
                    u[C["done_md"]] = "### Cancelled\nNothing was exported."
                elif st == "rejected":
                    u[C["done_md"]] = f"### Not researched\n{(v.get('resolution') or {}).get('message', 'no result')}"
                elif st == "aborted":
                    u[C["done_md"]] = f"### Run stopped\n{v.get('abort_reason')}"
                else:       # stopped: an error or the cost cap ended a segment
                    reviewed = bool(v.get("memo_path"))
                    u[C["done_md"]] = ("### Stopped\nThe run stopped (see the message above). " + ("The memo is as it was before the change." if reviewed else "Nothing was exported."))
                    u[C["back_btn"]] = gr.Button(visible=reviewed)
            await index_thread(thread_id)
            u[C["pending"]] = gr.Dropdown(choices=await waiting_choices(), value=None)
            return u

        def _gaps_text(p: dict) -> str:
            cov = "; ".join(f"{a} {c['fetched']} of {c['expected']}" for a, c in sorted(p.get("coverage", {}).items()))
            gaps = "\n".join(f"- [{g['area']}] {g['missing']}: {g['reason']}. Effect: {g['effect']}" for g in p["gaps"]) or "- none"
            fails = "\n".join(f"- {f}" for f in p["check_failures"]) or "- none"
            return f"**Coverage (fetched of expected):** {cov}\n\n**Data gaps**\n{gaps}\n\n**Failed checks**\n{fails}"

        async def drive(thread_id: str, graph_input, what: str, ticker: str = ""):
            """Stream a segment: progress lines as nodes finish, a clock while one runs, then the screen for where the thread stopped."""
            graph = await get_graph()
            while True:
                queue: asyncio.Queue = asyncio.Queue()
                task = asyncio.create_task(run_segment(graph, graph_input, thread_id, what, lambda n, verb: queue.put_nowait(n)))
                done, started = [], time.time()
                while not task.done() or not queue.empty():
                    try:
                        done.append(NODE_LABELS.get(await asyncio.wait_for(queue.get(), 1.0), "step"))
                    except asyncio.TimeoutError:
                        pass
                    yield {C["thread"]: thread_id, C["review_col"]: gr.Column(visible=False), C["done_col"]: gr.Column(visible=False),
                           C["progress"]: "**Working** (" + f"{int(time.time() - started)} s)\n" + "\n".join(f"- done: {d}" for d in done)}
                seg: Segment = task.result()
                if seg.error:
                    yield await view(thread_id, note=f"**Stopped:** {seg.error}", progress_text="\n".join(f"- done: {d}" for d in done))
                    return
                if seg.interrupts and seg.interrupts[0].get("type") == "choose_company" and ticker:
                    graph_input, what = Command(resume=ticker), "resume"     # the user already confirmed this ticker
                    continue
                yield await view(thread_id, progress_text="\n".join(f"- done: {d}" for d in done))
                return

        async def decide(thread_id: str, payload: dict):
            async for u in drive(thread_id, Command(resume=payload), "review"):
                yield u

        # ---- handlers ------------------------------------------------------------------------------------
        async def on_load():
            await get_graph()
            return await view("")

        async def find(text: str):
            await get_graph()
            if not text.strip():
                yield {C["cand_msg"]: "Type a company name or a ticker."}
                return
            yield {C["cand_msg"]: "Looking the company up (SEC filings, price check) ...", C["cand"]: gr.Radio(choices=[], visible=False), C["go_btn"]: gr.Button(visible=False)}
            res = await asyncio.to_thread(_state["resolver"].resolve, text)
            rows = [(f"{c.ticker}: {c.name} | {c.exchange} | market cap " + (f"${c.market_cap / 1e9:,.1f}B" if c.market_cap else "n/a") +
                     (f" | {c.filer_note}" if c.filer_note else ""), c.ticker) for c in res.candidates]
            yield {C["cand_msg"]: res.message, C["cand"]: gr.Radio(choices=rows, value=rows[0][1] if len(rows) == 1 else None, visible=bool(rows)),
                   C["go_btn"]: gr.Button(visible=bool(rows))}

        async def go(ticker: str):
            if not ticker:
                yield {C["cand_msg"]: "Pick one candidate first."}
                return
            thread_id = str(uuid.uuid4())
            graph = await get_graph()
            yield {C["thread"]: thread_id, C["progress"]: "Starting ...", C["go_btn"]: gr.Button(visible=False)}
            async for u in drive(thread_id, {"user_input": ticker}, "start", ticker):
                yield u

        async def open_thread(thread_id: str):
            if not thread_id:
                yield await view("")
                return
            graph = await get_graph()
            snap = await graph.aget_state(cfg(thread_id))
            if not snap.values:
                yield await view("", note="Nothing stored for this thread id.")
                return
            yield await view(thread_id)

        async def refresh():
            await get_graph()
            return {C["pending"]: gr.Dropdown(choices=await waiting_choices(), value=None)}

        async def approve(thread_id, confirmed):
            async for u in decide(thread_id, {"action": "approve", "confirmed": bool(confirmed)}):
                yield u

        async def cancel(thread_id):
            async for u in decide(thread_id, {"action": "cancel"}):
                yield u

        async def change_sections(thread_id, picked, text):
            async for u in decide(thread_id, {"action": "request_changes", "sections": picked or [], "instruction": text or ""}):
                yield u

        async def change_assumptions(thread_id, a, b, c, d):
            vals = {"equity_risk_premium": None if a is None else a / 100, "terminal_growth": None if b is None else b / 100,
                    "scenario_spread_pts": c, "fade_target_growth": None if d is None else d / 100}
            async for u in decide(thread_id, {"action": "request_changes", "sections": ["assumptions"], "assumptions": {k: v for k, v in vals.items() if v is not None}}):
                yield u

        async def change_peers(thread_id, text):
            async for u in decide(thread_id, {"action": "edit_peers", "peers": [p for p in (text or "").split(",")]}):
                yield u

        async def new_data(thread_id, which, confirmed):
            async for u in decide(thread_id, {"action": "request_changes", "sections": ["new_data"], "area": which, "confirmed": bool(confirmed)}):
                yield u

        async def reapprove(thread_id):
            graph = await get_graph()
            await graph.aupdate_state(cfg(thread_id), {"feedback": {"action": "approve", "confirmed": True}}, as_node="review")
            async for u in drive(thread_id, None, "resume"):
                yield u

        async def overwrite_yes(thread_id):
            async for u in drive(thread_id, Command(resume=True), "resume"):
                yield u

        async def overwrite_no(thread_id):
            async for u in drive(thread_id, Command(resume=False), "resume"):
                yield u

        async def back(thread_id):
            await return_to_review(await get_graph(), thread_id)
            yield await view(thread_id, note="Back at the review: the memo is as it was.")

        demo.load(on_load, None, outs)
        find_btn.click(find, company, outs)
        company.submit(find, company, outs)
        go_btn.click(go, cand, outs)
        pending.input(open_thread, pending, outs)           # opens from the event's own value, not from a later click
        open_btn.click(open_thread, pending, outs)
        refresh_btn.click(refresh, None, outs)
        approve_btn.click(approve, [thread, confirm_failed], outs)
        cancel_btn.click(cancel, thread, outs)
        changes_btn.click(change_sections, [thread, sections, instruction], outs)
        assume_btn.click(change_assumptions, [thread, erp, tg, spread, fade], outs)
        peers_btn.click(change_peers, [thread, peers_in], outs)
        data_btn.click(new_data, [thread, area, confirm_cost], outs)
        reapprove_btn.click(reapprove, thread, outs)
        yes_btn.click(overwrite_yes, thread, outs)
        no_btn.click(overwrite_no, thread, outs)
        back_btn.click(back, thread, outs)
    return demo


def main(port: int = 7860) -> None:
    load_env_file()
    demo = build_ui()
    demo.queue()
    try:
        demo.launch(server_name="127.0.0.1", server_port=port)      # show_api is not accepted by Gradio 6.30
    finally:
        asyncio.run(close_app())
