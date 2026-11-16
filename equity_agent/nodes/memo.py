"""draft_memo (writer role, one call per section) and check_memo (plain-code checks, one redraft per failing section).

The sections live in memo_parts.json (text, tables, ids, problems per section) and memo.md is rendered from it, so a
redraft of one section leaves the others byte-identical. The state keeps only paths and the list of check failures.
"""
import json
from pathlib import Path

from ..config import run_dir
from ..ledger import Ledger, LedgerFile
from ..llm import LLMClient
from ..memo_checks import check_section
from ..memo_tables import header, sources_section
from ..memo_writer import DRAFT_ORDER, SECTIONS, WRITTEN_BY_MODEL, Context, draft_section, lead_and_tables, tag_gaps
from ..export import numberize
from ..numbers import ID_BRACKET, split_ids
from ..schemas import GraphState
from ..services import Services


def load_context(state: GraphState, folder: Path) -> tuple[Context, Ledger]:
    ledger = LedgerFile(state["ledger_path"]).load()
    assessment = json.loads(Path(state["assessment"]["path"]).read_text())["assessment"]
    gaps = state.get("gaps", [])
    return Context(state["company"], ledger, assessment, gaps, tag_gaps(gaps), state.get("valuation") or {}), ledger


def cited_ids(parts: dict) -> tuple[set[str], list[str]]:
    """(every ledger id the memo rests on, the narrative texts in section order). Tables add their ids to table_ids."""
    cited: set[str] = set()
    texts = []
    for name in SECTIONS:
        if name == "Sources and Methodology":
            continue
        p = parts[name]
        texts += [p["lead"], p["text"]]
        cited.update(i for g in ID_BRACKET.findall(p["text"] + p["lead"]) for i in split_ids(g))
        cited.update(p["table_ids"])
    return cited, texts


def render(state: GraphState, ctx: Context, parts: dict, folder: Path, refs: dict[str, int] | None = None, generated_at: str = "") -> str:
    """memo.md from the parts. Sources and Methodology is built here, so it always shows the current check results.
    With refs (the exported file), ids and gap tags in the text become numbered links."""
    cited, _ = cited_ids({**parts, "Sources and Methodology": {}})
    fix = (lambda t: numberize(t, refs)) if refs is not None else (lambda t: t)
    failed = {n: p["problems"] for n, p in parts.items() if p.get("problems")}
    blocks = [header(state, ctx.ledger, generated_at)]
    for name in SECTIONS:
        if name == "Sources and Methodology":
            cost = json.loads((folder / "cost.json").read_text()) if (folder / "cost.json").exists() else None
            body = sources_section(state, ctx.ledger, cited, ctx.gap_index, failed, cost, folder, refs)
            blocks.append(f"## {name}\n\n{body}")
            continue
        p = parts[name]
        banner = f"> **FAILED CHECKS:** {'; '.join(p['problems'])}\n\n" if p.get("problems") else ""
        body = "\n\n".join(x for x in (banner.strip(), fix(p["lead"]), fix(p["text"]), p["table"]) if x)
        blocks.append(f"## {name}\n\n{body}")
    return "\n\n".join(blocks) + "\n"


def make_draft_memo(services: Services):
    async def draft_memo(state: GraphState) -> dict:
        folder = run_dir(state["thread_id"])
        ctx, _ = load_context(state, folder)
        client = services.llm_client(folder)
        feedback = state.get("feedback") or {}
        redraft = list(feedback.get("sections") or [])
        parts_path = folder / "memo_parts.json"
        old = json.loads(parts_path.read_text()) if parts_path.exists() and redraft else {}
        parts: dict = {}
        for name in DRAFT_ORDER:
            if name in old and name not in redraft:
                parts[name] = old[name]                       # kept exactly as it was
            else:
                d = await draft_section(client, name, ctx, instruction=feedback.get("instruction", "") if name in feedback.get("chosen", redraft) else "")
                lead, table, ids = lead_and_tables(name, ctx, state)
                parts[name] = {"text": d.text, "lead": lead, "table": table, "table_ids": sorted(ids), "allowed": sorted(d.allowed),
                               "gap_tags": d.gap_tags, "problems": [], "redrafted": False}
            ctx.drafted[name] = parts[name]["text"]
            ctx.allowed[name] = set(parts[name]["allowed"])
        parts_path.write_text(json.dumps(parts, indent=1))
        path = folder / "memo.md"
        path.write_text(render(state, ctx, {**parts, "Sources and Methodology": {}}, folder))
        return {"memo_path": str(path), "node_log": ["draft_memo"]}

    return draft_memo


def make_check_memo(services: Services):
    async def check_memo(state: GraphState) -> dict:
        folder = run_dir(state["thread_id"])
        ctx, ledger = load_context(state, folder)
        parts = json.loads((folder / "memo_parts.json").read_text())
        client = services.llm_client(folder)
        for name in WRITTEN_BY_MODEL:      # the Executive Summary is redrafted from the other sections and the ids they were given
            ctx.drafted[name], ctx.allowed[name] = parts[name]["text"], set(parts[name]["allowed"])
        for name in WRITTEN_BY_MODEL:
            p = parts[name]
            problems = check_section(p["text"], ledger, set(p["allowed"]), p["gap_tags"], set(ctx.gap_index))
            if problems:      # one redraft with the list of problems; if it still fails, the section is marked and the run goes on
                d = await draft_section(client, name, ctx, previous=p["text"], problems=problems)
                p["first_attempt"] = {"text": p["text"], "problems": problems}      # kept so the review step can show what was redrafted and why
                p.update(text=d.text, allowed=sorted(d.allowed), gap_tags=d.gap_tags, redrafted=True)
                problems = check_section(p["text"], ledger, set(p["allowed"]), p["gap_tags"], set(ctx.gap_index))
            p["problems"] = problems
        (folder / "memo_parts.json").write_text(json.dumps(parts, indent=1))
        path = folder / "memo.md"
        path.write_text(render(state, ctx, {**parts, "Sources and Methodology": {}}, folder))
        failures = [f"{name}: {problem}" for name in WRITTEN_BY_MODEL for problem in parts[name]["problems"]]
        return {"check_failures": failures, "memo_path": str(path), "node_log": ["check_memo"]}

    return check_memo
