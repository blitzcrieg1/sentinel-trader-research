"""
Sentinel Trader — Configuration.

All thresholds, limits, API keys, and tunables live here.
Loaded from environment variables with pydantic-settings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — loaded from .env / environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── AI provider ──────────────────────────────────────────────────────
    ai_provider: Literal["gemini", "gemini_dual", "nvidia", "dual", "cerebras"] | None = Field(
        default=None,
        description=(
            "AI backend. Auto-detect from keys. gemini_dual = two Gemini keys for 2× RPM."
        ),
    )
    gemini_api_key: str = Field("", description="Google Gemini API key (primary)")
    gemini_api_key_2: str = Field("", description="Second Gemini API key (RPM spillover)")
    gemini_model: str = Field(
        default="gemini-3.1-flash-lite", description="Gemini model ID to use",
    )
    nvidia_api_key: str = Field("", description="NVIDIA NIM API key")
    nvidia_model: str = Field(
        default="meta/llama-3.3-70b-instruct", description="NVIDIA model ID to use",
    )
    nvidia_max_rpm: int = Field(
        30, ge=1, le=1000,
        description="Max NVIDIA NIM requests per rolling 60s window (dual-mode spillover)",
    )
    cerebras_api_key: str = Field("", description="Cerebras API key")
    cerebras_model: str = Field(
        default="gpt-oss-120b", description="Cerebras model ID to use",
    )
    cerebras_max_rpm: int = Field(
        1000, ge=1, le=5000,
        description="Max Cerebras requests per rolling 60s window",
    )

    # ── MEXC ─────────────────────────────────────────────────────────────
    mexc_api_key: str = Field(..., description="MEXC API key (sub-account, withdraw-disabled)")
    mexc_secret_key: str = Field(..., description="MEXC API secret")

    # ── CryptoPanic (news sentiment) ─────────────────────────────────────
    cryptopanic_api_token: str = Field(
        "", description="CryptoPanic API auth token; empty = news layer disabled"
    )

    # ── Telegram (optional — empty/dummy token disables admin bot) ───────
    telegram_bot_token: str = Field(
        "", description="Telegram Bot API token; empty or placeholder = disabled",
    )
    telegram_admin_chat_id: int = Field(..., description="Whitelisted admin chat ID")

    # ── Scan Configuration ───────────────────────────────────────────────
    scan_symbols: list[str] = Field(
        default=[
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", 
            "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "PEPE/USDT", "WIF/USDT",
            "SUI/USDT", "APT/USDT", "INJ/USDT", "RENDER/USDT", "ARB/USDT",
            "OP/USDT", "TIA/USDT", "NEAR/USDT", "FET/USDT", "RUNE/USDT"
        ],
        description="Symbols to scan (CCXT swap format appended automatically)",
    )
    scan_interval_minutes: int = Field(
        15,
        ge=5,
        le=240,
        description="Minutes between scan cycles (aligned to candle close)",
    )

    @field_validator("scan_interval_minutes")
    @classmethod
    def scan_interval_must_divide_day(cls, v: int) -> int:
        if 1440 % v != 0:
            raise ValueError(
                f"scan_interval_minutes={v} must evenly divide 1440 (minutes per day)"
            )
        return v

    scan_jitter_min_sec: float = Field(
        2.0, ge=0.0, le=30.0,
        description="Min seconds to wait after candle close before scanning",
    )
    scan_jitter_max_sec: float = Field(
        5.0, ge=0.5, le=60.0,
        description="Max seconds to wait after candle close before scanning",
    )
    exec_timeout_sec: int = Field(
        30, ge=5, le=120,
        description="Timeout for a single broker execution call in seconds",
    )

    # ── Risk Parameters ──────────────────────────────────────────────────
    risk_per_trade_pct: float = Field(
        0.5, ge=0.1, le=2.0,
        description="Max risk per trade as % of equity",
    )
    max_leverage: int = Field(3, ge=1, le=10, description="Maximum leverage")
    max_concurrent_positions: int = Field(
        2, ge=1, le=5,
        description="Max open positions at any time",
    )
    daily_loss_limit_pct: float = Field(
        2.0, ge=0.5, le=10.0,
        description="Daily loss limit as % of equity → auto-halt",
    )
    weekly_loss_limit_pct: float = Field(
        5.0, ge=1.0, le=20.0,
        description="Weekly loss limit as % of equity → halt until /resume",
    )
    confidence_threshold: float = Field(
        0.50, ge=0.0, le=1.0,
        description="AI confidence below this → auto no_trade",
    )
    post_loss_cooldown_sec: int = Field(
        1800, ge=0, le=7200,
        description="Seconds to wait after a losing trade before next entry",
    )
    min_sl_pct: float = Field(
        0.3, ge=0.1, le=2.0,
        description="Minimum stop-loss distance as % of entry",
    )
    max_sl_pct: float = Field(
        5.0, ge=1.0, le=15.0,
        description="Maximum stop-loss distance as % of entry",
    )
    min_rr_ratio: float = Field(
        1.5, ge=0.5, le=5.0,
        description="Minimum reward:risk ratio (TP1 distance / SL distance). "
                    "Replay sweep (2026-06-16) showed 2.5 was a near-worst operating "
                    "point for both models; 1.5 is strictly less-bad and lets the bot trade.",
    )
    limit_entry_max_deviation_pct: float = Field(
        1.0, ge=0.1, le=5.0,
        description="Max % deviation for limit entry from current price",
    )
    max_consecutive_losses: int = Field(
        3, ge=1, le=10,
        description="Consecutive losses before requiring /resume",
    )
    min_notional_usdt: float = Field(
        3.0, ge=1.0, le=100.0,
        description="Minimum position notional in USDT — below this, skip the trade",
    )
    max_notional_per_symbol_usdt: float = Field(
        1000.0, ge=10.0, le=1_000_000.0,
        description="Maximum notional value per position per symbol (USDT)",
    )
    min_sl_atr_multiple: float = Field(
        0.1, ge=0.1, le=3.0,
        description="SL distance must be >= this × ATR(14) (flash-wick guard)",
    )
    warn_sl_atr_multiple: float = Field(
        1.0, ge=0.1, le=5.0,
        description="SL distance below this × ATR(14) emits a warning (still passes)",
    )
    rsi_overbought_threshold: float = Field(
        70.0, ge=60.0, le=90.0,
        description="1h RSI above this blocks new long entries (chasing overbought)",
    )
    rsi_oversold_threshold: float = Field(
        30.0, ge=10.0, le=40.0,
        description="1h RSI below this blocks new short entries (chasing oversold)",
    )
    max_consecutive_exec_errors: int = Field(
        3, ge=1, le=10,
        description="Consecutive execution errors before kill switch",
    )

    # ── Trailing Stop ────────────────────────────────────────────────────
    trailing_stop_enabled: bool = Field(
        default=True, description="Enable trailing stop loss"
    )
    trailing_stop_pct: float = Field(
        default=1.0, description="Trail distance as % of current price (gap is fixed in price-relative terms)"
    )
    trailing_stop_activation_pct: float = Field(
        default=0.5, description="Only activate trailing after price moves this % in profit"
    )

    # ── AI Budget ────────────────────────────────────────────────────────
    max_ai_calls_per_day: int = Field(
        5000, ge=10, le=150_000,
        description="Daily AI call budget (cost + runaway guard)",
    )
    max_consecutive_malformed: int = Field(
        3, ge=1, le=10,
        description="Consecutive malformed AI responses before kill switch",
    )
    ai_timeout_sec: int = Field(
        30, ge=5, le=120,
        description="Timeout for a single AI call in seconds",
    )
    ai_max_rpm: int = Field(
        60, ge=1, le=4000,
        description="Default max RPM for Gemini key 1 (paid tier; 3.1 Flash Lite ≈4K)",
    )
    gemini_key1_max_rpm: int = Field(
        0, ge=0, le=4000,
        description="Key 1 RPM cap (0 = use AI_MAX_RPM). Paid key — keep high.",
    )
    gemini_key2_max_rpm: int = Field(
        10, ge=1, le=1000,
        description="Key 2 RPM cap. Free-tier backup — keep low to avoid 429s.",
    )
    gemini_key1_max_daily_calls: int = Field(
        0, ge=0, le=150_000,
        description="Stop routing to GEMINI_API_KEY after this many calls/day (0=unlimited)",
    )
    gemini_key2_max_daily_calls: int = Field(
        300, ge=0, le=5000,
        description="Stop routing to GEMINI_API_KEY_2 after this many calls/day (0=unlimited)",
    )
    scan_stagger_sec: float = Field(
        15.0, ge=1.0, le=120.0,
        description="Seconds between starting each symbol pipeline in a scan cycle",
    )

    # ── Paper Broker ─────────────────────────────────────────────────────
    paper_starting_equity: float = Field(
        10_000.0, ge=10.0, le=1_000_000.0,
        description="Initial virtual USDT balance for the paper broker",
    )
    paper_slippage_pct: float = Field(
        0.05, ge=0.0, le=1.0,
        description="Simulated slippage % applied against the trade direction",
    )
    paper_fee_pct: float = Field(
        0.02, ge=0.0, le=1.0,
        description="Simulated taker fee % per fill (MEXC futures taker = 0.02%; maker = 0.00%)",
    )

    # ── Data Sanity ──────────────────────────────────────────────────────
    max_candle_staleness_factor: float = Field(
        2.0, ge=1.5, le=5.0,
        description="Candle must be < this × timeframe old",
    )
    max_price_deviation_pct: float = Field(
        1.0, ge=0.1, le=5.0,
        description="Max ticker vs candle close deviation %",
    )

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: Path = Field(
        Path("data"),
        description="Directory for SQLite DB and backups",
    )
    log_dir: Path = Field(
        Path("data/logs"),
        description="Directory for JSONL log files",
    )

    # ── Position Manager ─────────────────────────────────────────────────
    position_loop_interval_sec: int = Field(
        60, ge=10, le=300,
        description="Seconds between position manager loop ticks",
    )
    heartbeat_interval_sec: int = Field(
        3600, ge=600, le=7200,
        description="Seconds between Telegram heartbeat messages",
    )

    # ── CCXT Tuning ──────────────────────────────────────────────────────
    ccxt_timeout_ms: int = Field(30000, ge=5000, le=60000)
    ccxt_market_load_max_attempts: int = Field(5, ge=1, le=20)
    ccxt_market_load_retry_delay_sec: int = Field(5, ge=1, le=60)
    ccxt_fetch_max_attempts: int = Field(3, ge=1, le=10)
    ccxt_fetch_retry_delay_sec: int = Field(2, ge=1, le=30)

    # ── Validators ───────────────────────────────────────────────────────
    @field_validator("scan_symbols", mode="before")
    @classmethod
    def parse_scan_symbols(cls, v: str | list[str]) -> list[str]:
        """Accept comma-separated string or list."""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @model_validator(mode="after")
    def resolve_ai_provider(self) -> Self:
        """Pick provider from env keys."""
        has_gemini = bool(self.gemini_api_key.strip())
        has_gemini_2 = bool(self.gemini_api_key_2.strip())
        has_nvidia = bool(self.nvidia_api_key.strip())
        has_cerebras = bool(self.cerebras_api_key.strip())

        if self.ai_provider is None:
            if has_cerebras:
                object.__setattr__(self, "ai_provider", "cerebras")
            elif has_gemini and has_gemini_2:
                object.__setattr__(self, "ai_provider", "gemini_dual")
            elif has_gemini and has_nvidia:
                object.__setattr__(self, "ai_provider", "dual")
            elif has_gemini:
                object.__setattr__(self, "ai_provider", "gemini")
            elif has_nvidia:
                object.__setattr__(self, "ai_provider", "nvidia")
            else:
                raise ValueError(
                    "Missing AI credentials: set CEREBRAS_API_KEY, GEMINI_API_KEY, or NVIDIA_API_KEY in .env"
                )
        elif self.ai_provider == "cerebras" and not has_cerebras:
            raise ValueError("AI_PROVIDER=cerebras but CEREBRAS_API_KEY is empty")
        elif self.ai_provider == "gemini" and not has_gemini:
            raise ValueError("AI_PROVIDER=gemini but GEMINI_API_KEY is empty")
        elif self.ai_provider == "gemini_dual" and not (has_gemini and has_gemini_2):
            raise ValueError(
                "AI_PROVIDER=gemini_dual requires GEMINI_API_KEY and GEMINI_API_KEY_2"
            )
        elif self.ai_provider == "nvidia" and not has_nvidia:
            raise ValueError("AI_PROVIDER=nvidia but NVIDIA_API_KEY is empty")
        elif self.ai_provider == "dual" and not (has_gemini and has_nvidia):
            raise ValueError(
                "AI_PROVIDER=dual requires both GEMINI_API_KEY and NVIDIA_API_KEY"
            )
        return self

    @property
    def telegram_enabled(self) -> bool:
        """True when a real Telegram bot token is configured."""
        from sentinel.admin.telegram import telegram_token_valid

        return telegram_token_valid(self.telegram_bot_token)

    @property
    def gemini_dual_enabled(self) -> bool:
        """True when a second Gemini key absorbs primary RPM overflow."""
        return (
            self.ai_provider == "gemini_dual"
            and bool(self.gemini_api_key.strip())
            and bool(self.gemini_api_key_2.strip())
        )

    @property
    def nvidia_fallback_enabled(self) -> bool:
        """True when NVIDIA can absorb overflow / 429 from Gemini."""
        return (
            self.ai_provider == "dual"
            and bool(self.nvidia_api_key.strip())
            and bool(self.gemini_api_key.strip())
        )

    @property
    def ai_model(self) -> str:
        """Active model ID for the configured AI provider."""
        if self.ai_provider == "cerebras":
            return self.cerebras_model
        if self.ai_provider == "nvidia":
            return self.nvidia_model
        if self.ai_provider == "dual":
            return f"{self.gemini_model} (+ {self.nvidia_model} fallback)"
        if self.ai_provider == "gemini_dual":
            return f"{self.gemini_model} (×2 keys)"
        return self.gemini_model

    @property
    def db_path(self) -> Path:
        """Full path to the SQLite database file."""
        return self.data_dir / "sentinel.db"

    @property
    def swap_symbols(self) -> list[str]:
        """Symbols in CCXT swap format (e.g. BTC/USDT:USDT)."""
        result = []
        for sym in self.scan_symbols:
            clean = sym.replace(":USDT", "").strip()
            result.append(f"{clean}:USDT")
        return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — call from anywhere to get config."""
    return Settings()  # type: ignore[call-arg]
