"""
Mean Reversion config service — DB-backed with 5-minute TTL cache.

Extracted from mean_reversion_strategy.py (Apr 2026) to match the pattern
used by all other strategy config services under services/.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import psycopg2
import psycopg2.extras


@dataclass
class MeanReversionConfig:
    """All parameters settable via DB (mean_reversion_global_config) with
    per-pair overrides via mean_reversion_pair_overrides (direct columns or
    JSONB parameter_overrides)."""

    strategy_name: str = "MEAN_REVERSION"
    version: str = "1.0.0"

    # Hard ADX gates — always on
    hard_adx_gate_enabled: bool = True
    adx_hard_ceiling_primary: float = 22.0
    adx_hard_ceiling_htf: float = 25.0
    adx_period: int = 14

    # Bollinger
    bb_period: int = 20
    bb_mult: float = 2.0

    # RSI
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    # Support / Resistance (used to boost confidence, not to gate)
    sr_lookback_bars: int = 50
    sr_proximity_pips: float = 10.0

    # Risk
    fixed_stop_loss_pips: float = 12.0
    fixed_take_profit_pips: float = 18.0
    min_confidence: float = 0.50
    max_confidence: float = 0.85
    sl_buffer_pips: float = 2.0

    # Timeframes
    primary_timeframe: str = "15m"
    confirmation_timeframe: str = "1h"

    # Cooldown
    signal_cooldown_minutes: int = 45

    # Routing
    trust_regime_routing: bool = True

    # Entry mode: "rejection" (prev breach + close back inside) or "touch" (band touch + RSI extreme on same candle)
    entry_mode: str = "rejection"

    # Optional session window gate (None = no gate; hour values are UTC inclusive)
    session_start_hour: Optional[int] = None
    session_end_hour: Optional[int] = None

    # Low-volatility regime filter — replaces ADX gate when enabled per-pair
    # Uses ATR ≤ threshold AND flat EMA slope instead of ADX ceiling
    low_vol_regime_filter_enabled: bool = False
    regime_atr_max_pips: float = 7.0            # 15m scale (research used 3.0 on 5m)
    regime_ema_period: int = 50
    regime_ema_lookback_candles: int = 24        # candles for EMA slope measurement
    regime_ema_max_change_pips: float = 5.0      # 15m scale (research used 4.5 on 5m)

    # Pair-level state (populated from DB)
    _pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Per-pair getters — always prefer direct column → JSONB → global
    # ------------------------------------------------------------------

    def get_for_pair(self, epic: str, param_name: str, default: Any = None) -> Any:
        if epic in self._pair_overrides:
            row = self._pair_overrides[epic]
            if param_name in row and row[param_name] is not None:
                return row[param_name]
            jsonb = row.get("parameter_overrides") or {}
            if param_name in jsonb and jsonb[param_name] is not None:
                return jsonb[param_name]
        if hasattr(self, param_name):
            return getattr(self, param_name)
        return default

    def get_pair_adx_hard_ceiling_primary(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "adx_hard_ceiling_primary",
                                        self.adx_hard_ceiling_primary))

    def get_pair_adx_hard_ceiling_htf(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "adx_hard_ceiling_htf",
                                        self.adx_hard_ceiling_htf))

    def get_pair_bb_mult(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "bb_mult", self.bb_mult))

    def get_pair_rsi_oversold(self, epic: str) -> int:
        return int(self.get_for_pair(epic, "rsi_oversold", self.rsi_oversold))

    def get_pair_rsi_overbought(self, epic: str) -> int:
        return int(self.get_for_pair(epic, "rsi_overbought", self.rsi_overbought))

    def get_pair_fixed_stop_loss(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "fixed_stop_loss_pips",
                                        self.fixed_stop_loss_pips))

    def get_pair_fixed_take_profit(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "fixed_take_profit_pips",
                                        self.fixed_take_profit_pips))

    def get_pair_min_confidence(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "min_confidence", self.min_confidence))

    def get_pair_max_confidence(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "max_confidence", self.max_confidence))

    def get_pair_primary_timeframe(self, epic: str) -> str:
        return str(self.get_for_pair(epic, "primary_timeframe", self.primary_timeframe))

    def get_pair_entry_mode(self, epic: str) -> str:
        return str(self.get_for_pair(epic, "entry_mode", self.entry_mode))

    def get_pair_session_start_hour(self, epic: str) -> Optional[int]:
        v = self.get_for_pair(epic, "session_start_hour", self.session_start_hour)
        return int(v) if v is not None else None

    def get_pair_session_end_hour(self, epic: str) -> Optional[int]:
        v = self.get_for_pair(epic, "session_end_hour", self.session_end_hour)
        return int(v) if v is not None else None

    def get_pair_low_vol_regime_filter_enabled(self, epic: str) -> bool:
        return bool(self.get_for_pair(epic, "low_vol_regime_filter_enabled", self.low_vol_regime_filter_enabled))

    def get_pair_regime_atr_max_pips(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "regime_atr_max_pips", self.regime_atr_max_pips))

    def get_pair_regime_ema_period(self, epic: str) -> int:
        return int(self.get_for_pair(epic, "regime_ema_period", self.regime_ema_period))

    def get_pair_regime_ema_lookback_candles(self, epic: str) -> int:
        return int(self.get_for_pair(epic, "regime_ema_lookback_candles", self.regime_ema_lookback_candles))

    def get_pair_regime_ema_max_change_pips(self, epic: str) -> float:
        return float(self.get_for_pair(epic, "regime_ema_max_change_pips", self.regime_ema_max_change_pips))

    def is_pair_enabled(self, epic: str) -> bool:
        if epic in self._pair_overrides:
            return bool(self._pair_overrides[epic].get("is_enabled", True))
        return True

    def is_pair_monitor_only(self, epic: str) -> bool:
        return bool(self.get_for_pair(epic, "monitor_only", False))

    # ------------------------------------------------------------------
    # DB loader
    # ------------------------------------------------------------------

    @classmethod
    def from_database(cls, database_url: Optional[str] = None) -> "MeanReversionConfig":
        config = cls()

        if database_url is None:
            database_url = os.getenv(
                "STRATEGY_CONFIG_DATABASE_URL",
                "postgresql://postgres:postgres@postgres:5432/strategy_config",
            )

        try:
            conn = psycopg2.connect(database_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            cur.execute(
                """
                SELECT * FROM mean_reversion_global_config
                 WHERE is_active = TRUE
                 ORDER BY id DESC LIMIT 1
                """
            )
            row = cur.fetchone()
            if row:
                for key in row.keys():
                    if not hasattr(config, key):
                        continue
                    v = row[key]
                    if v is None:
                        continue
                    default_val = getattr(cls, key, None)
                    if isinstance(default_val, bool):
                        v = bool(v)
                    elif isinstance(default_val, int):
                        v = int(v)
                    elif isinstance(default_val, float):
                        v = float(v)
                    setattr(config, key, v)

            cur.execute("SELECT * FROM mean_reversion_pair_overrides")
            config._pair_overrides = {r["epic"]: dict(r) for r in cur.fetchall()}

            cur.close()
            conn.close()

            logging.info(
                f"[MEAN_REVERSION] Loaded config v{config.version}: "
                f"bb({config.bb_period},{config.bb_mult}) "
                f"rsi({config.rsi_oversold}/{config.rsi_overbought}) "
                f"adx≤{config.adx_hard_ceiling_primary}/{config.adx_hard_ceiling_htf} "
                f"SL/TP={config.fixed_stop_loss_pips}/{config.fixed_take_profit_pips} "
                f"{len(config._pair_overrides)} pair overrides"
            )
        except Exception as e:
            logging.warning(f"[MEAN_REVERSION] Using defaults; DB load failed: {e}")

        return config


class MeanReversionConfigService:
    _instance: Optional["MeanReversionConfigService"] = None

    def __init__(self) -> None:
        self._config: Optional[MeanReversionConfig] = None
        self._last_refresh: Optional[datetime] = None
        self._cache_ttl_seconds = 300

    @classmethod
    def get_instance(cls) -> "MeanReversionConfigService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_config(self) -> MeanReversionConfig:
        now = datetime.now()
        if (self._config is None
                or self._last_refresh is None
                or (now - self._last_refresh).total_seconds() > self._cache_ttl_seconds):
            self._config = MeanReversionConfig.from_database()
            self._last_refresh = now
        return self._config

    def refresh(self) -> MeanReversionConfig:
        self._config = None
        return self.get_config()


def get_mean_reversion_config() -> MeanReversionConfig:
    return MeanReversionConfigService.get_instance().get_config()
