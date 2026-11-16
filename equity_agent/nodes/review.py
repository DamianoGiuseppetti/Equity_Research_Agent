"""review (interrupt only) and route_feedback (plain code, all the routing).

review shows the reviewer what the run produced and takes one decision. route_feedback reads the decision and sends the graph on:
export (approve), draft_memo (sections), valuation (assumptions, peers), a collector (new data), the end (cancel), or back to
review with a notice when a value is rejected or a confirmation is needed.
"""
import json

from langgraph.graph import END
from langgraph.types import Command, interrupt

from ..config import load_config, run_dir
from ..feedback import ASSUMPTION_SECTIONS, VALUATION_SECTIONS, validate_assumptions, with_dependents
from ..memo_writer import SECTIONS, WRITTEN_BY_MODEL
from ..schemas import GraphState
from .collectors import COLLECTORS


async def review(state: GraphState) -> dict:
    """Interrupt and nothing else. LangGraph re-runs this node from its first line on resume, so it must have no side effects."""
    rounds_left = load_config().feedback.max_revision_rounds - state.get("review_round", 0)
    actions = ["approve", "request_changes", "edit_peers", "cancel"] if rounds_left > 0 else ["approve", "cancel"]
    val = state.get("valuation") or {}
    decision = interrupt({"memo_path": state["memo_path"], "check_failures": state.get("check_failures", []), "gaps": state.get("gaps", []),
                          "coverage": state.get("coverage", {}), "peers": val.get("peers", {}), "assumptions": val.get("assumptions", {}),
                          "rating": val.get("rating"), "round": state.get("review_round", 0), "rounds_left": max(rounds_left, 0),
                          "actions": actions, "notice": state.get("review_notice", ""), "thread_id": state.get("thread_id")})
    return {"feedback": decision, "node_log": ["review"]}


def first_run_cost(thread_id: str) -> float:
    path = run_dir(thread_id) / "cost.json"
    return json.loads(path.read_text())["totals"]["cost_usd"] if path.exists() else 0.0


async def route_feedback(state: GraphState) -> Command:
    fb = state["feedback"]
    action = fb.get("action")
    rounds = state.get("review_round", 0)
    log = ["route_feedback"]

    def again(notice: str) -> Command:     # back to the review page with a message; nothing else changes
        return Command(goto="review", update={"review_notice": notice, "node_log": log})

    def revise(goto: str, sections: list[str], chosen: list[str], feedback_extra: dict | None = None, **extra) -> Command:
        return Command(goto=goto, update={"review_round": rounds + 1, "review_notice": "", "node_log": log, **extra,
                                          "feedback": {**fb, **(feedback_extra or {}), "sections": sections, "chosen": chosen}})

    if action == "approve":
        failures = state.get("check_failures", [])
        if failures and not fb.get("confirmed"):
            return again(f"{len(failures)} check(s) failed in this memo (listed above). Approving exports it with those failures listed in "
                         "Sources and Methodology. Approve again with the confirmation to continue.")
        return Command(goto="export", update={"approved": True, "review_notice": "", "node_log": log})
    if action == "cancel":
        return Command(goto=END, update={"approved": False, "node_log": log})
    if rounds >= load_config().feedback.max_revision_rounds:
        return again("All revision rounds are used. Only Approve or Cancel remain.")

    if action == "edit_peers":
        peers = [p.strip().upper() for p in fb.get("peers") or [] if p.strip()]
        if not peers:
            return again("Edit peers needs at least one ticker.")
        return revise("valuation", VALUATION_SECTIONS, VALUATION_SECTIONS, {"peers": peers})

    if action == "request_changes":
        targets = list(fb.get("sections") or [])
        if "new_data" in targets:     # the most expensive path: say so, with the cost of the first run, and wait for a confirmation
            area = fb.get("area")
            if area not in COLLECTORS:
                return again(f"New data needs one area to re-collect: {', '.join(COLLECTORS)}.")
            if not fb.get("confirmed"):
                return again(f"New data is the most expensive path: it collects {area} again, then runs the data gate, the analysis, the valuation and a "
                             f"full redraft of the memo. This thread has cost ${first_run_cost(state['thread_id']):.4f} so far, and the first run alone was "
                             "most of that. Send again with the confirmation to continue.")
            return revise(area, list(WRITTEN_BY_MODEL), list(WRITTEN_BY_MODEL))
        if "assumptions" in targets:
            clean, errors = validate_assumptions(fb.get("assumptions") or {})
            if errors or not clean:
                return again("Assumptions not changed. " + (" ".join(errors) if errors else "Give at least one value."))
            return revise("valuation", ASSUMPTION_SECTIONS, ASSUMPTION_SECTIONS,
                          assumption_overrides={**(state.get("assumption_overrides") or {}), **clean})
        chosen = [s for s in targets if s in SECTIONS and s != "Sources and Methodology"]
        if not chosen:
            return again("Pick at least one section to change (Sources and Methodology is built by code and cannot be redrafted).")
        if not str(fb.get("instruction", "")).strip():
            return again("Write an instruction for the section(s) you picked.")
        return revise("draft_memo", with_dependents(chosen), chosen)
    return again(f"Unknown action {action!r}.")
