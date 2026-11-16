"""Helpers shared by the nodes."""
from ..ledger import LedgerFile
from ..schemas import GraphState


def write_facts(state: GraphState, owner: str, facts: list[dict]) -> dict:
    """Write a node's facts to the run's ledger file. Returns the state update holding only the counter."""
    count = LedgerFile(state["ledger_path"]).write_facts(owner, facts)
    return {"fact_counts": {owner: count}}
