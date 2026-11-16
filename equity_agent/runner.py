"""Run one segment of a thread (a start or a resume) and always record its LangSmith link, also when the segment stops on an error.

The CLI and the Gradio app both drive the graph through run_segment, so they behave the same. This module does not import
any UI code.
"""
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .config import ROLES, active_profile, model_id, run_dir
from .llm import RunStopped
from .trace import TraceLinks, graph_config


@dataclass
class Segment:
    thread_id: str
    interrupts: list[dict] = field(default_factory=list)     # payloads of the interrupts the graph is waiting at
    error: str = ""                                           # why the segment stopped, empty if it ran to an interrupt or the end
    trace: dict = field(default_factory=dict)                 # the link entry (url or note)
    next: tuple = ()


async def run_segment(graph, graph_input, thread_id: str, what: str, on_node: Callable[[str, str], None] | None = None) -> Segment:
    """Stream the graph until it ends or waits at an interrupt. A RunStopped (cost cap, malformed analysis) or any other error
    ends the segment with `error` set; the checkpoint stays, so the thread can be looked at or resumed."""
    run_id = uuid.uuid4()
    cfg = graph_config(thread_id, run_id, active_profile(), {r: model_id(r) for r in ROLES})
    seg = Segment(thread_id)
    try:
        async for chunk in graph.astream(graph_input, cfg, stream_mode="updates"):
            cached = chunk.get("__metadata__", {}).get("cached")       # LangGraph replays updates saved before a crash
            for node in chunk:
                if node not in ("__interrupt__", "__metadata__") and on_node:
                    on_node(node, "restored" if cached else "done")
    except RunStopped as exc:
        seg.error = str(exc)
    except Exception as exc:       # deliberate: whatever stops the run, the link and a message must still be produced
        seg.error = f"{type(exc).__name__}: {str(exc)[:400]}"
    links = TraceLinks(run_dir(thread_id))
    seg.trace = await asyncio.to_thread(links.record, run_id, what)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    seg.interrupts = [i.value for t in snap.tasks for i in t.interrupts]
    seg.next = tuple(snap.next)
    return seg


async def return_to_review(graph, thread_id: str) -> None:
    """After a revision stopped (cost cap, error): put the thread back at the review with the memo as it was. The failed round is not
    counted, and the feedback that caused it is cleared. Files are untouched, because a revision writes them only when it finishes."""
    cfg = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(cfg)
    await graph.aupdate_state(cfg, {"feedback": {}, "review_round": max(0, snap.values.get("review_round", 1) - 1)}, as_node="check_memo")
