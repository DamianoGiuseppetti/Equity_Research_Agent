"""Command line: run, resume, threads. Plain argparse and print; no framework."""
import argparse
import asyncio
import json
import os
import warnings
import sys
import uuid
from pathlib import Path

from langgraph.types import Command

from .config import data_dir, load_env_file, run_dir
from .graph import build_graph, open_checkpointer
from .ledger import LedgerFile
from .llm import RunStopped
from .memo_writer import SECTIONS
from .resolver import Resolver
from .runner import run_segment
from .trace import TraceLinks
from .sources import EdgarClient, SecError, YahooQuotes


def make_resolver() -> Resolver:
    user_agent = os.environ.get("EDGAR_USER_AGENT", "")
    return Resolver(EdgarClient(user_agent, data_dir()), YahooQuotes())  # EdgarClient rejects a blank User-Agent


def ask(prompt: str) -> str:
    return input(f"{prompt} ").strip()


# ---- interrupts ----------------------------------------------------------------------------
def choose_company(payload: dict, preset: str | None):
    print(payload["message"])
    for c in payload["candidates"]:
        cap = f"${c['market_cap'] / 1e9:,.1f}B" if c.get("market_cap") else "n/a"
        print(f"  {c['ticker']:<6} {c['name']:<40} {c['exchange']:<14} market cap {cap}")
    return preset or ask("Choose a ticker:")


def review_decision(payload: dict, auto_approve: bool) -> dict:
    print("\n" + Path(payload["memo_path"]).read_text())
    if payload["gaps"]:
        print("Data gaps:")
        for g in payload["gaps"]:
            print(f"  - [{g['area']}] {g['missing']}: {g['reason']}. Effect: {g['effect']}")
    for problem in payload["check_failures"]:
        print(f"  CHECK FAILED: {problem}")
    print(f"Review round {payload['round']}. Actions: {', '.join(payload['actions'])}")
    if auto_approve:
        print("--approve given: approving.")
        return {"action": "approve"}
    action = ask("Action [approve / request_changes / edit_peers / cancel]:")
    if action == "request_changes":
        for i, name in enumerate(SECTIONS, 1):
            print(f"  {i:>2}. {name}")
        picked = ask("Section numbers (comma separated), or 'assumptions', or 'new_data':")
        sections = [s.strip() for s in picked.split(",")]
        sections = [SECTIONS[int(s) - 1] if s.isdigit() else s for s in sections]
        area = ask("Which area to re-collect (market_data / filings / macro / news):") if "new_data" in sections else ""
        return {"action": action, "sections": sections, "area": area, "instruction": ask("Instruction:")}
    if action == "edit_peers":
        return {"action": action, "peers": [p.strip() for p in ask("Peer tickers, comma separated:").split(",")]}
    return {"action": action}


# ---- driving the graph ----------------------------------------------------------------------
async def drive(graph, graph_input, thread_id, args, what: str = "start") -> int:
    """Run segments, answer interrupts, repeat until the graph stops for good. Returns the exit code."""
    while True:
        seg = await run_segment(graph, graph_input, thread_id, what,
                                lambda node, verb: print(f"  {'restored' if verb == 'restored' else 'done    '} {node}", flush=True))
        links = TraceLinks(run_dir(thread_id))
        if seg.error:
            print(f"\nRun stopped: {seg.error}\n  {links.text()}\n  The thread {thread_id} is kept; 'threads' lists it.", file=sys.stderr)
            return 1
        if not seg.interrupts:
            break
        payload = seg.interrupts[0]
        if payload.get("type") == "choose_company":
            graph_input = Command(resume=choose_company(payload, getattr(args, "choose", None)))
        elif payload.get("type") == "confirm_overwrite":
            print(f"{payload['folder']} already exists (thread {payload.get('previous_thread')}).")
            graph_input = Command(resume=bool(getattr(args, "overwrite", False)) or ask("Overwrite it? The old memo is kept as memo_previous.md [y/N]:").lower() == "y")
        else:
            graph_input = Command(resume=review_decision(payload, getattr(args, "approve", False)))
        args.choose, what = None, "resume"  # a preset answer is used once
    await print_outcome(graph, {"configurable": {"thread_id": thread_id}})
    return 0


async def print_outcome(graph, cfg) -> None:
    v = (await graph.aget_state(cfg)).values
    print()
    if not v.get("company"):
        print("Not researched:", v.get("resolution", {}).get("message", "no result"))
    elif v.get("abort_reason"):
        print(v["abort_reason"])
    elif v.get("approved"):
        print(f"Approved. {v.get('export_message') or v.get('export_dir')}")
    else:
        print("Cancelled before export.")
    ledger = LedgerFile(v["ledger_path"]).load() if v.get("ledger_path") else None
    print(f"Thread {v.get('thread_id')}  profile {v.get('profile')}  facts in ledger {len(ledger) if ledger is not None else 0}")
    for area, c in sorted(v.get("coverage", {}).items()):
        print(f"  coverage {area:<12} {c['fetched']} of {c['expected']}")
    if v.get("ledger_path"):
        links = TraceLinks(Path(v["ledger_path"]).parent)
        if v.get("export_dir"):
            links.patch_run_json(v["export_dir"])
        print(f"  {links.text()}")
    print(f"  data gaps {len(v.get('gaps', []))}  (run folder: {Path(v['ledger_path']).parent if v.get('ledger_path') else '-'})")
    if v.get("ledger_path"):
        folder = Path(v["ledger_path"]).parent
        cost_file = folder / "cost.json"
        if cost_file.exists():
            t = json.loads(cost_file.read_text())["totals"]
            print(f"  LLM cost ${t['cost_usd']:.4f} over {t['calls']} calls ({t['input_tokens']:,} in, {t['output_tokens']:,} out); log: {cost_file}")
        if (folder / "memo.md").exists():
            print(f"  memo: {folder / 'memo.md'}   failed checks: {len(v.get('check_failures', []))}")


def status_of(snap) -> str:
    v = snap.values
    pending = [i.value for t in snap.tasks for i in t.interrupts]
    if pending:
        return "waiting for your company choice" if pending[0].get("type") == "choose_company" else "waiting at review"
    if snap.next:
        return f"interrupted, resumable (next: {', '.join(snap.next)})"
    if not v.get("company"):
        return f"rejected ({v.get('resolution', {}).get('status', 'unknown')})"
    if v.get("abort_reason"):
        return "stopped by data_gate"
    return "finished (approved)" if v.get("approved") else "cancelled"


# ---- commands --------------------------------------------------------------------------------
async def cmd_run(args) -> int:
    thread_id = str(uuid.uuid4())
    print(f"Thread {thread_id}", flush=True)
    async with open_checkpointer() as saver:
        graph = build_graph(saver, make_resolver())
        return await drive(graph, {"user_input": args.query}, thread_id, args)


async def cmd_resume(args) -> int:
    cfg = {"configurable": {"thread_id": args.thread_id}}
    async with open_checkpointer() as saver:
        graph = build_graph(saver, make_resolver())
        snap = await graph.aget_state(cfg)
        if not snap.values:
            print(f"No stored run for thread {args.thread_id}. Use 'threads' to list them.")
            return 1
        if not snap.next:
            print(f"Thread {args.thread_id} is already complete: {status_of(snap)}.")
            return 0
        print(f"Resuming {args.thread_id}: {status_of(snap)}", flush=True)
        return await drive(graph, None, args.thread_id, args, "resume")  # None continues from the last checkpoint


async def cmd_threads(args) -> int:
    async with open_checkpointer() as saver:
        graph = build_graph(saver, make_resolver())
        cur = await saver.conn.execute("select thread_id from checkpoints group by thread_id order by max(checkpoint_id) desc")
        ids = [r[0] for r in await cur.fetchall()]
        if not ids:
            print("No stored threads.")
        for tid in ids:
            snap = await graph.aget_state({"configurable": {"thread_id": tid}})
            company = (snap.values.get("company") or {}).get("ticker") or snap.values.get("user_input", "?")
            print(f"{tid}  {company:<10} {status_of(snap)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    warnings.filterwarnings("ignore", module=r"yfinance.*")      # yfinance 1.7 prints pandas deprecation warnings
    parser = argparse.ArgumentParser(prog="python -m equity_agent", description="Equity research agent: run, resume, threads, app")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="research a company by name or ticker")
    p.add_argument("query")
    p.add_argument("--choose", metavar="TICKER", help="answer the ambiguous-company question without asking")
    p.add_argument("--approve", action="store_true", help="approve at the review step without asking")
    p.add_argument("--overwrite", action="store_true", help="confirm overwriting an existing outputs folder without asking")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("resume", help="continue a stored run")
    p.add_argument("thread_id")
    p.add_argument("--choose", metavar="TICKER")
    p.add_argument("--approve", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_resume)
    p = sub.add_parser("app", help="open the Gradio app")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=None)
    p = sub.add_parser("threads", help="list stored runs and their status")
    p.set_defaults(func=cmd_threads)
    args = parser.parse_args(argv)
    if args.command == "app":
        from .app import main as run_app     # imported here: the graph and the CLI never need Gradio
        run_app(args.port)
        return 0
    try:
        return asyncio.run(args.func(args))
    except (SecError, ValueError, RunStopped) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
