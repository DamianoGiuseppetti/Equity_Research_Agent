"""The facts ledger: every number in the memo is looked up here by id."""
import json
import os
import threading
from pathlib import Path

from .schemas import Fact


class Ledger:
    def __init__(self, facts: list[Fact] | None = None) -> None:
        self._facts: dict[str, Fact] = {}
        for fact in facts or []:
            self.add(fact)

    def add(self, fact: Fact) -> None:
        if fact.id in self._facts:
            raise ValueError(f"Duplicate fact id {fact.id!r}: ids are unique inside a ledger")
        self._facts[fact.id] = fact

    def get(self, fact_id: str) -> Fact:
        try:
            return self._facts[fact_id]
        except KeyError:
            raise KeyError(f"No fact with id {fact_id!r} in the ledger") from None

    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self._facts

    def __len__(self) -> int:
        return len(self._facts)

    @property
    def facts(self) -> list[Fact]:
        return list(self._facts.values())

    def to_dicts(self) -> list[dict]:
        return [f.model_dump(mode="json") for f in self._facts.values()]

    def to_json(self) -> str:
        return json.dumps(self.to_dicts(), indent=2)

    @classmethod
    def from_dicts(cls, items: list[dict]) -> "Ledger":
        return cls([Fact.model_validate(i) for i in items])

    @classmethod
    def from_json(cls, text: str) -> "Ledger":
        return cls.from_dicts(json.loads(text))


class LedgerFile:
    """The ledger of one run, kept in ledger.json so the graph state holds only its path.

    Collectors run in parallel and each writes its own facts, so every write is a locked read-merge-write
    followed by an atomic rename (a kill in the middle cannot leave a half-written file). A fact id must
    start with its owner's name ("market.", "news."), and a write replaces everything that owner wrote
    before: re-running a node after feedback or a crash gives the same ledger, not duplicates.
    """

    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with LedgerFile._locks_guard:
            self._lock = LedgerFile._locks.setdefault(str(self.path.resolve()), threading.Lock())

    def load(self) -> Ledger:
        return Ledger.from_json(self.path.read_text()) if self.path.exists() else Ledger()

    def write_facts(self, owner: str, facts: list[Fact | dict]) -> int:
        validated = [f if isinstance(f, Fact) else Fact.model_validate(f) for f in facts]
        wrong = [f.id for f in validated if not f.id.startswith(f"{owner}.")]
        if wrong:
            raise ValueError(f"Fact ids {wrong} must start with '{owner}.': a node may only write its own facts")
        with self._lock:
            kept = [f for f in self.load().facts if not f.id.startswith(f"{owner}.")]
            merged = Ledger(kept + validated)  # raises on a duplicate id inside the new batch
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(merged.to_json())
            os.replace(tmp, self.path)
        return len(validated)
