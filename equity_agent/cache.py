"""Disk cache for raw pulls. Every entry keeps the time it was fetched, and a TTL decides when to fetch again."""
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .config import cache_dir

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


@dataclass(frozen=True)
class Pull:
    payload: Any
    retrieved_at: datetime      # when the data was really fetched, also for a cache hit
    from_cache: bool


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def entry_path(namespace: str, key: str) -> Path:
    """Where a cached pull lives on disk (article records point at it)."""
    return cache_dir() / _safe(namespace) / f"{_safe(key)}.json"


def cached_pull(namespace: str, key: str, ttl_hours: float, fetch: Callable[[], Any],
                now: datetime | None = None) -> Pull:
    """Return the cached payload if it is younger than the TTL, otherwise call fetch() and store the result.

    One lock per entry, so two collectors asking for the same data at once fetch it once.
    An unreadable cache file is treated as a miss.
    """
    now = now or datetime.now(timezone.utc)
    path = entry_path(namespace, key)
    with _locks_guard:
        lock = _locks.setdefault(str(path.resolve()), threading.Lock())
    with lock:
        if path.exists():
            try:
                entry = json.loads(path.read_text())
                retrieved = datetime.fromisoformat(entry["retrieved_at"])
                if now - retrieved < timedelta(hours=ttl_hours):
                    return Pull(entry["payload"], retrieved, True)
            except (ValueError, KeyError):
                pass
        payload = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"retrieved_at": now.isoformat(), "payload": payload}))
        os.replace(tmp, path)
        return Pull(payload, now, False)
