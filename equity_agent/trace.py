"""LangSmith trace links. The run id is chosen here, before the run starts, so a link exists even for a run that fails or is
stopped; the URL is built after the run with get_run_url on the read run, retried because ingestion lags.

Each invoke of the graph is its own root trace, so a thread has one link per invoke (start, each resume). They are listed in
trace.json in the run folder and copied into run.json. Nothing here may block or break a run: every failure becomes a reason.
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(os.environ.get("LANGSMITH_API_KEY"))


def graph_config(thread_id: str, run_id: uuid.UUID | None = None, profile: str = "", model_ids: dict | None = None) -> dict:
    """Config for graph.ainvoke / astream: the thread, our own run id and metadata for LangSmith."""
    return {"configurable": {"thread_id": thread_id}, "run_id": run_id or uuid.uuid4(),
            "metadata": {"thread_id": thread_id, "profile": profile, "model_ids": model_ids or {}}}


def build_url(run_id: uuid.UUID, client=None, tries: int = 8, delay: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> tuple[str | None, str]:
    """(url, "") or (None, reason). Blocking: run it in a thread. Never raises."""
    if not tracing_enabled():
        return None, "tracing is off (LANGSMITH_TRACING is not true or LANGSMITH_API_KEY is not set)"
    try:
        if client is None:
            from langsmith import Client
            client = Client()
        project = os.environ.get("LANGSMITH_PROJECT") or "default"
        last = "the run was not readable yet"
        for attempt in range(tries):
            try:
                run = client.read_run(run_id)
                return client.get_run_url(run=run, project_name=project), ""
            except Exception as exc:      # ingestion lags: a first read often gets a 404
                last = f"{type(exc).__name__}: {str(exc)[:150]}"
                if attempt < tries - 1:
                    sleep(delay)
        return None, f"LangSmith did not return the run after {tries} tries ({last})"
    except Exception as exc:
        return None, f"LangSmith is not reachable ({type(exc).__name__}: {str(exc)[:150]})"


class TraceLinks:
    """trace.json of one thread."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / "trace.json"

    def links(self) -> list[dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def record(self, run_id: uuid.UUID, what: str, **kw) -> dict:
        """Look the link up (blocking) and append it. `what` says which invoke: start, resume, review, ..."""
        url, reason = build_url(run_id, **kw)
        entry = {"run_id": str(run_id), "what": what, "url": url, "note": "" if url else f"trace link unavailable: {reason}",
                 "at": datetime.now(timezone.utc).isoformat()}
        self.path.write_text(json.dumps(self.links() + [entry], indent=1))
        return entry

    def text(self) -> str:
        """One line for the screen: the latest link, or why there is none."""
        links = self.links()
        if not links:
            return "trace link: none recorded"
        last = links[-1]
        return f"trace: {last['url']}" if last["url"] else last["note"]

    def patch_run_json(self, export_dir: str) -> None:
        """Copy the links into the exported run.json (the export ran before its own trace could be read)."""
        path = Path(export_dir) / "run.json"
        if export_dir and path.exists():
            data = json.loads(path.read_text())
            data["trace"] = self.links()
            path.write_text(json.dumps(data, indent=1, default=str))
