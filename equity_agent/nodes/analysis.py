"""analyze: one LLM call, checked in code."""
import json

from ..assessment import run_analysis, thin_area_gaps
from ..config import run_dir
from ..ledger import LedgerFile
from ..schemas import GraphState
from ..services import Services


def make_analyze(services: Services):
    async def analyze(state: GraphState) -> dict:
        """A malformed call (after the wrapper's one retry) raises LLMOutputError, which stops the run with a clear message."""
        folder = run_dir(state["thread_id"])
        ledger = LedgerFile(state["ledger_path"]).load()
        assessment, dropped = await run_analysis(services.llm_client(folder), state["company"], ledger, state.get("gaps", []))
        (folder / "assessment.json").write_text(json.dumps({"assessment": assessment, "dropped": dropped}, indent=1))
        counts = {a: len(assessment[a]["findings"]) for a in ("financial", "technical", "macro", "news")}
        return {"assessment": {"path": str(folder / "assessment.json"), "findings": counts, "dropped": len(dropped)},
                "gaps": [g.model_dump() for g in thin_area_gaps(assessment)], "node_log": ["analyze"]}

    return analyze
