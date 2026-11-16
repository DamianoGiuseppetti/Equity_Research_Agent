"""FRED observations through requests, with bounded retries. The same pattern as the SEC client: retry on 429 and 5xx."""
import time
from typing import Callable, Protocol

import requests

OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 1.0


class FredError(Exception):
    """FRED could not be reached or refused the request."""


class FredProvider(Protocol):
    def observations(self, series_id: str, start: str) -> list[list]:
        """[[observation date "2026-09-01", value], ...], oldest first, missing values left out."""
        ...


class FredClient:
    def __init__(self, api_key: str, sleep: Callable[[float], None] = time.sleep, get=requests.get) -> None:
        if not api_key:
            raise FredError("FRED_API_KEY is not set")
        self._key, self._sleep, self._get = api_key, sleep, get

    def observations(self, series_id: str, start: str) -> list[list]:
        params = {"series_id": series_id, "api_key": self._key, "file_type": "json", "observation_start": start}
        last = "no attempt made"
        for attempt in range(MAX_ATTEMPTS):
            try:
                r = self._get(OBSERVATIONS_URL, params=params, timeout=30)
            except requests.RequestException as exc:
                last = f"network error: {type(exc).__name__}"
            else:
                if r.status_code == 200:
                    # FRED marks a missing observation with "."
                    return [[o["date"], float(o["value"])] for o in r.json()["observations"] if o["value"] != "."]
                last = f"HTTP {r.status_code}"
                if r.status_code != 429 and r.status_code < 500:
                    break  # for example 400 for an unknown series: retrying will not help
            self._sleep(BACKOFF_SECONDS * 2 ** attempt)
        raise FredError(f"FRED request for {series_id} failed ({last})")  # never includes the API key
