"""Settings: validated defaults (config.yaml), model profiles (models.yaml), paths and the .env file."""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PACKAGE_DIR = Path(__file__).parent

Role = Literal["extract", "analyst", "writer"]
ROLES: tuple[str, ...] = get_args(Role)
DEFAULT_PROFILE = "demo"      # the published agent: Haiku for extraction, Sonnet 5.5 for analysis and writing. "dev" is Haiku everywhere, for testing.


class _Section(BaseModel):
    # extra="forbid": a typo in config.yaml is an error, not a silently ignored key
    model_config = ConfigDict(extra="forbid", frozen=True)


class Scope(_Section):
    financial_sic_codes: list[int]
    successor_ciks: dict[int, int] = {}


class Checks(_Section):
    mismatch_tolerance: float = Field(gt=0, lt=1)


class News(_Section):
    max_articles: int = Field(gt=0)
    window_days: int = Field(gt=0)
    text_cap_words: int = Field(gt=0)
    low_coverage_min_scored: int = Field(gt=0)


class Filings(_Section):
    section_token_cap: int = Field(gt=0)
    min_section_chars: int = Field(gt=0)


class Valuation(_Section):
    equity_risk_premium: float = Field(gt=0, lt=1)
    terminal_growth: float = Field(gt=0, lt=1)
    scenario_spread_pts: float = Field(gt=0)
    fade_target_growth: float = Field(gt=0, lt=1)
    margin_shift_pts: float = Field(gt=0)
    near_zero_fcf_margin: float = Field(gt=0, lt=1)
    min_peers: int = Field(gt=0)
    cost_of_debt_fallback_spread_pts: float = Field(gt=0)


class Cost(_Section):
    max_run_cost_usd: dict[str, float]      # profile -> cap in USD

    @model_validator(mode="after")
    def positive(self) -> "Cost":
        if not self.max_run_cost_usd or any(v <= 0 for v in self.max_run_cost_usd.values()):
            raise ValueError("max_run_cost_usd needs a positive value for each profile")
        return self


class Feedback(_Section):
    max_revision_rounds: int = Field(gt=0)


class Cache(_Section):
    market_data_ttl_hours: float = Field(gt=0)
    filings_ttl_hours: float = Field(gt=0)
    macro_ttl_hours: float = Field(gt=0)
    news_ttl_hours: float = Field(gt=0)


class Expected(_Section):
    market_data_items: int = Field(gt=0)
    filings_sections: int = Field(gt=0)


class SeriesDef(_Section):
    label: str
    unit: str
    change: Literal["diff", "pct"]


class Macro(_Section):
    common_series: list[str]
    series: dict[str, SeriesDef]
    sector_series: dict[str, list[str]]

    @model_validator(mode="after")
    def every_series_is_defined(self) -> "Macro":
        used = set(self.common_series) | {s for ids in self.sector_series.values() for s in ids}
        missing = used - set(self.series)
        if missing:
            raise ValueError(f"series used but not defined: {sorted(missing)}")
        return self


class Config(_Section):
    scope: Scope
    checks: Checks
    news: News
    filings: Filings
    valuation: Valuation
    cost: Cost
    feedback: Feedback
    cache: Cache
    expected: Expected
    macro: Macro


@lru_cache
def load_config() -> Config:
    return Config.model_validate(yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text()))


@lru_cache
def _profiles() -> dict[str, dict[str, str]]:
    return yaml.safe_load((PACKAGE_DIR / "models.yaml").read_text())["profiles"]


def thinking_can_be_disabled(model: str) -> bool:
    return model in yaml.safe_load((PACKAGE_DIR / "models.yaml").read_text())["thinking_can_be_disabled"]


@lru_cache
def _prices() -> dict[str, dict[str, float]]:
    return yaml.safe_load((PACKAGE_DIR / "models.yaml").read_text())["prices"]


def price_per_million(model: str) -> tuple[float, float]:
    """(input, output) USD per million tokens. A model without a price is an error: cost must never be guessed."""
    try:
        price = _prices()[model]
    except KeyError:
        raise ValueError(f"No price for model {model!r} in models.yaml; add it before using the model") from None
    return price["input"], price["output"]


def active_profile() -> str:
    """Read the profile on every call, so a test or a shell can switch it without a restart."""
    name = os.environ.get("EQUITY_AGENT_PROFILE", DEFAULT_PROFILE)
    if name not in _profiles():
        raise ValueError(f"Unknown EQUITY_AGENT_PROFILE {name!r}. Choose one of: {sorted(_profiles())}")
    return name


def model_id(role: Role) -> str:
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}. Choose one of: {list(ROLES)}")
    return _profiles()[active_profile()][role]


def data_dir() -> Path:
    """Local state (checkpoints, SEC file cache). Git-ignored. EQUITY_AGENT_DATA_DIR overrides it."""
    path = Path(os.environ.get("EQUITY_AGENT_DATA_DIR", ".data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """Raw pulls and article text. Git-ignored. EQUITY_AGENT_CACHE_DIR overrides it."""
    path = Path(os.environ.get("EQUITY_AGENT_CACHE_DIR", ".cache"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir() -> Path:
    """Where an approved memo is exported: outputs/<TICKER>_<date>/. Git-ignored. EQUITY_AGENT_OUTPUT_DIR overrides it."""
    return Path(os.environ.get("EQUITY_AGENT_OUTPUT_DIR", "outputs"))


def run_dir(thread_id: str) -> Path:
    """Work folder of one run: ledger.json, memo and (later) sections. Git-ignored."""
    path = (data_dir() / "runs" / thread_id).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_env_file(path: Path = Path(".env")) -> None:
    """Copy KEY=VALUE lines from .env into the environment without overriding variables already set.

    Written by hand because python-dotenv is not in the approved stack.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))
