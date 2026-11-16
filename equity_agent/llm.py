"""Role-based model factory and the one wrapper every LLM call goes through.

get_llm is the only place that turns a role into a chat model. LLMClient adds what every node needs: one retry on a
malformed result, a token and cost log per call in cost.json, and a hard stop when a run reaches its cost cap.
"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from .config import Role, active_profile, load_config, model_id, price_per_million, thinking_can_be_disabled

T = TypeVar("T", bound=BaseModel)


def get_llm(role: Role, max_tokens: int = 4096, thinking_off: bool = False) -> ChatAnthropic:
    # No temperature: Sonnet 5.5 rejects non-default sampling values. Free-text calls turn thinking off: on a long prompt Haiku
    # spent the whole token budget thinking and returned no text. Forced structured output does not need it.
    model, extra = model_id(role), {}
    if thinking_off:
        if thinking_can_be_disabled(model):
            extra = {"thinking": {"type": "disabled"}}
        else:       # Sonnet 5.5 always thinks: thinking tokens count against max_tokens, so the budget is larger
            max_tokens = max(max_tokens * 3, 12000)
    return ChatAnthropic(model=model, max_tokens=max_tokens, **extra)


class RunStopped(Exception):
    """A condition that must end the run with a clear message (the CLI prints it)."""


class CostCapExceeded(RunStopped):
    pass


class LLMOutputError(RunStopped):
    """The model returned a malformed result twice."""


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = price_per_million(model)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class CostLog:
    """cost.json of one run: every call, and totals. Written after each call, so a crash loses nothing.

    The file is read-merge-written under a lock and renamed into place, like the ledger. Because it lives on disk,
    the cap also holds across a resumed run.
    """

    _locks: dict[str, threading.Lock] = {}
    _guard = threading.Lock()

    def __init__(self, path: Path) -> None:
        self.path = path
        with CostLog._guard:
            self._lock = CostLog._locks.setdefault(str(path.resolve()), threading.Lock())

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text())["calls"] if self.path.exists() else []

    def total(self) -> float:
        with self._lock:
            return sum(c["cost_usd"] for c in self._load())

    def add(self, record: dict) -> None:
        with self._lock:
            calls = self._load() + [record]
            by_step: dict[str, dict] = {}
            for c in calls:
                s = by_step.setdefault(c["step"], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
                s["calls"] += 1
                s["input_tokens"] += c["input_tokens"]
                s["output_tokens"] += c["output_tokens"]
                s["cost_usd"] = round(s["cost_usd"] + c["cost_usd"], 8)
            totals = {"calls": len(calls), "input_tokens": sum(c["input_tokens"] for c in calls),
                      "output_tokens": sum(c["output_tokens"] for c in calls),
                      "cost_usd": round(sum(c["cost_usd"] for c in calls), 8)}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"totals": totals, "by_step": by_step, "calls": calls}, indent=1))
            os.replace(tmp, self.path)


class LLMClient:
    """Calls the model for a role. `step` names the caller in the cost log (for example "extract.item1a")."""

    def __init__(self, run_folder: Path, factory: Callable[..., object] = get_llm, cap: float | None = None) -> None:
        self.cost = CostLog(run_folder / "cost.json")
        self.factory = factory
        self.cap = cap if cap is not None else float(os.environ.get("EQUITY_AGENT_MAX_RUN_COST_USD") or load_config().cost.max_run_cost_usd[active_profile()])    # the variable is for tests and demos of the stop

    def _check_cap(self, step: str) -> None:
        spent = self.cost.total()
        if spent >= self.cap:
            raise CostCapExceeded(f"Run stopped before '{step}': the logged cost ${spent:.4f} has reached the cap of "
                                  f"${self.cap:.2f} (max_run_cost_usd in config.yaml). Nothing more was sent to the model.")

    def _log(self, role: Role, step: str, attempt: int, raw, ok: bool) -> None:
        usage = getattr(raw, "usage_metadata", None) or {}
        tin, tout = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        model = model_id(role)
        self.cost.add({"time": datetime.now(timezone.utc).isoformat(), "step": step, "role": role, "model": model,
                       "attempt": attempt, "ok": ok, "input_tokens": tin, "output_tokens": tout,
                       "cost_usd": round(cost_usd(model, tin, tout), 8)})

    async def structured(self, role: Role, schema: type[T], messages: list, step: str, max_tokens: int = 4096) -> T:
        """Forced structured output. One retry if the result is missing or fails the schema; then LLMOutputError."""
        reason = ""
        for attempt in (1, 2):
            self._check_cap(step)
            runnable = self.factory(role, max_tokens=max_tokens).with_structured_output(schema, include_raw=True)
            result = await runnable.ainvoke(messages)
            parsed, error = result.get("parsed"), result.get("parsing_error")
            ok = isinstance(parsed, schema) and error is None
            self._log(role, step, attempt, result.get("raw"), ok)
            if ok:
                return parsed
            reason = str(error) if error else "the model returned no structured result"
        raise LLMOutputError(f"The model returned a malformed result twice for '{step}': {reason[:300]}")

    async def text(self, role: Role, messages: list, step: str, max_tokens: int = 4000) -> str:
        """Plain text output (the writer). One retry if the reply is empty or was cut off at max_tokens.

        Thinking is switched off (see get_llm). Only the text blocks are returned.
        """
        for attempt in (1, 2):
            self._check_cap(step)
            reply = await self.factory(role, max_tokens=max_tokens, thinking_off=True).ainvoke(messages)
            content = reply.content if isinstance(reply.content, str) else "".join(
                b.get("text", "") for b in reply.content if isinstance(b, dict) and b.get("type", "text") == "text")
            cut_off = (getattr(reply, "response_metadata", None) or {}).get("stop_reason") == "max_tokens"
            ok = bool(content.strip()) and not cut_off
            self._log(role, step, attempt, reply, ok)
            if ok:
                return content.strip()
        raise LLMOutputError(f"The model returned an empty or cut-off reply twice for '{step}'.")
