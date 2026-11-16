"""export: reachable only from the approve path (route_feedback). Nothing is written to outputs/ before this node runs.

Writes outputs/<TICKER>_<date>/ with memo.md (numbered references), memo.pdf, ledger.json and run.json. If the folder exists, the
node asks for confirmation first (an interrupt before any write) and keeps the old text as memo_previous.md. If the PDF fails,
the other three files are still written and the message names the missing file.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from langgraph.types import interrupt

from ..config import output_dir, run_dir
from ..export import number_references
from ..memo_writer import SECTIONS
from ..schemas import GraphState
from ..services import Services
from .memo import cited_ids, load_context, render


def trace_links(folder: Path) -> list[dict]:
    """The LangSmith links recorded for this thread (trace.json, written by the driver), or an empty list."""
    path = folder / "trace.json"
    return json.loads(path.read_text()) if path.exists() else []


def run_summary(state: GraphState, folder: Path, now: datetime) -> dict:
    cost_path = folder / "cost.json"
    cost = json.loads(cost_path.read_text()) if cost_path.exists() else {}
    return {"thread_id": state["thread_id"], "company": state["company"], "generated_at": now.isoformat(),
            "profile": state.get("profile"), "model_ids": state.get("model_ids", {}),
            "cost": {"totals": cost.get("totals"), "by_step": cost.get("by_step")},
            "coverage": state.get("coverage", {}), "gap_register": state.get("gaps", []), "failed_checks": state.get("check_failures", []),
            "review_rounds": state.get("review_round", 0), "valuation": state.get("valuation", {}),
            "extract_summary": state.get("extract_summary"), "sentiment_summary": state.get("sentiment_summary"),
            "trace": trace_links(folder) or [{"url": None, "note": "no trace link recorded for this thread"}]}


def make_export(services: Services):
    async def export(state: GraphState) -> dict:
        if not state.get("approved"):
            raise RuntimeError("export reached without approval")
        folder = run_dir(state["thread_id"])
        ctx, ledger = load_context(state, folder)
        now = datetime.now(timezone.utc)
        target = output_dir() / f"{state['company']['ticker']}_{now.date().isoformat()}"
        previous = None
        if target.exists():
            # First line with an effect is below this question. LangGraph runs the node again from the top on resume, which is safe here.
            old = json.loads((target / "run.json").read_text()) if (target / "run.json").exists() else {}
            if not interrupt({"type": "confirm_overwrite", "folder": str(target), "previous_thread": old.get("thread_id")}):
                return {"export_dir": "", "export_message": f"Export cancelled: {target} was left as it was.", "node_log": ["export"]}
            previous = (target / "memo.md").read_text() if (target / "memo.md").exists() else None

        parts = json.loads((folder / "memo_parts.json").read_text())
        parts = {**parts, "Sources and Methodology": {}}
        cited, texts = cited_ids(parts)
        refs = number_references(texts, cited, {f.id for f in ledger.facts})
        memo = render(state, ctx, parts, folder, refs, generated_at=now.strftime("%Y-%m-%d %H:%M UTC"))

        target.mkdir(parents=True, exist_ok=True)
        if previous is not None:
            (target / "memo_previous.md").write_text(previous)
        (target / "memo.md").write_text(memo)
        (target / "ledger.json").write_text(ledger.to_json())
        (target / "run.json").write_text(json.dumps(run_summary(state, folder, now), indent=1, default=str))
        price = ledger.get("market.price.last_close").as_of if "market.price.last_close" in ledger else "n/a"
        line = f"{state['company']['name']} ({state['company']['ticker']}) | price date {price} | not investment advice"
        message = f"Exported to {target}: memo.md, ledger.json, run.json"
        try:
            services.pdf_writer(memo, line, target / "memo.pdf")
            message += ", memo.pdf"
        except Exception as exc:     # the Markdown stays; the user is told which file is missing
            message += f". memo.pdf was NOT written ({type(exc).__name__}: {str(exc)[:200]})"
        return {"export_dir": str(target), "export_message": message, "node_log": ["export"]}

    return export
