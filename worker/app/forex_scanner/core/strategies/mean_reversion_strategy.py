#!/usr/bin/env python3
"""
Mean Reversion Strategy v1.0 — Bollinger + RSI extremes with HTF alignment

VERSION: 1.0.0
DATE: 2026-04-18
STATUS: Monitor-only on six pairs validated by 90-day standalone eval

Data provenance
---------------
Driven by the Apr 18 2026 canonical-data analysis after the RANGING_MARKET
revival failed (best PF 0.60 with hard ADX gates). A standalone 90-day
evaluation (scripts/eval_mean_reversion.py) showed the BB+RSI mean-reversion
thesis has a real edge on six of nine FX pairs when HTF-aligned:

    EURJPY PF 1.77  NZDUSD PF 1.50  USDCAD PF 1.50
    USDCHF PF 1.33  USDJPY PF 1.17  AUDJPY PF 1.15
    (EURUSD / AUDUSD / GBPUSD < 1.0 — disabled per-pair)

Design decisions (bake the lessons from RANGING_MARKET in from day 1)
---------------------------------------------------------------------
- Hard ADX gates on BOTH primary (15m) and HTF (1h), always enforced.
  No trust_regime_routing bypass loophole — that was RANGING_MARKET's
  catastrophic bug.
- Top-level signal fields: `adx`, `adx_htf`, `market_regime` (so
  alert_history stamps correctly without the nested-key bug that
  caused 24% historical regime mismatch).
- EMA-Wilder ADX fallback (matches DataFetcher) — no SMA divergence.
- Signal logic kept deliberately simple (BB band touch + RSI extreme);
  all oscillator-confluence complexity removed. The data says simple
  wins on this market.
- DB-backed config; per-pair overrides via JSONB `parameter_overrides`
  (monitor_only flag, disabled_reason, etc).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

from .strategy_registry import StrategyInterface, register_strategy


# ============================================================================
# CONFIG
# ============================================================================

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


# ============================================================================
# CONFIG SERVICE (singleton with TTL cache)
# ============================================================================

class MeanReversionConfigService:
    _instance = None

    def __init__(self):
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
                or (now - self._last_refresh).seconds > self._cache_ttl_seconds):
            self._config = MeanReversionConfig.from_database()
            self._last_refresh = now
        return self._config

    def refresh(self) -> MeanReversionConfig:
        self._config = None
        return self.get_config()


def get_mean_reversion_config() -> MeanReversionConfig:
    return MeanReversionConfigService.get_instance().get_config()


# ============================================================================
# STRATEGY
# ============================================================================

@register_strategy("MEAN_REVERSION")
class MeanReversionStrategy(StrategyInterface):
    """BB band touch + RSI extreme + hard ADX gates on primary (15m) and HTF (1h).

    Signal generation
    -----------------
    BUY when:
        close <= lower_BB       AND
        RSI <= rsi_oversold     AND
        15m_ADX <= adx_hard_ceiling_primary AND
        1h_ADX  <= adx_hard_ceiling_htf

    SELL when:
        close >= upper_BB       AND
        RSI >= rsi_overbought   AND
        15m_ADX <= adx_hard_ceiling_primary AND
        1h_ADX  <= adx_hard_ceiling_htf

    Confidence scoring
    ------------------
    min_confidence + ratio × (max_confidence - min_confidence)
    where ratio combines:
        - RSI extremity (how far past threshold, capped)
        - BB band penetration (|close - band| / ATR proxy)
        - S/R proximity bonus
    """

    def __init__(self, config=None, logger=None, db_manager=None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self.config = get_mean_reversion_config()

        self._cooldowns: Dict[str, datetime] = {}
        self._current_timestamp: Optional[datetime] = None

        self.logger.info(
            f"[MEAN_REVERSION] v{self.config.version} initialized | "
            f"hard ADX: 15m≤{self.config.adx_hard_ceiling_primary} "
            f"1h≤{self.config.adx_hard_ceiling_htf}"
        )

    @property
    def strategy_name(self) -> str:
        return "MEAN_REVERSION"

    def get_required_timeframes(self) -> List[str]:
        return [self.config.confirmation_timeframe, self.config.primary_timeframe]

    # ------------------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------------------

    def detect_signal(
        self,
        df_trigger: pd.DataFrame = None,
        df_4h: pd.DataFrame = None,
        epic: str = "",
        pair: str = "",
        df_entry: pd.DataFrame = None,
        current_timestamp: datetime = None,
        routing_context: Dict = None,
        **kwargs,
    ) -> Optional[Dict]:
        self._current_timestamp = current_timestamp

        df = df_trigger if df_trigger is not None else df_entry
        if df is None or len(df) < max(self.config.bb_period, self.config.rsi_period) + 5:
            return None

        if not self.config.is_pair_enabled(epic):
            self.logger.debug(f"[MEAN_REVERSION] {epic} disabled — skipping")
            return None

        if not self._check_cooldown(epic):
            return None

        adx_value = self._get_adx(df)
        adx_htf = self._get_adx(df_4h) if df_4h is not None and len(df_4h) >= 30 else None

        # Hard ADX gates — always enforced (no trust_regime_routing bypass)
        if self.config.hard_adx_gate_enabled:
            pri_ceiling = self.config.get_pair_adx_hard_ceiling_primary(epic)
            if adx_value is not None and adx_value > pri_ceiling:
                self.logger.debug(
                    f"[MEAN_REVERSION] ❌ {epic} 15m ADX {adx_value:.1f} > {pri_ceiling}"
                )
                return None
            htf_ceiling = self.config.get_pair_adx_hard_ceiling_htf(epic)
            if adx_htf is not None and adx_htf > htf_ceiling:
                self.logger.debug(
                    f"[MEAN_REVERSION] ❌ {epic} 1h ADX {adx_htf:.1f} > {htf_ceiling}"
                )
                return None

        # Compute BB + RSI
        close = df["close"].astype(float)
        bb_period = self.config.bb_period
        bb_mult = self.config.get_pair_bb_mult(epic)
        ma = close.rolling(bb_period).mean()
        sd = close.rolling(bb_period).std()
        upper = ma + bb_mult * sd
        lower = ma - bb_mult * sd
        rsi = self._rsi(close, self.config.rsi_period)

        latest_close = float(close.iloc[-1])
        latest_upper = float(upper.iloc[-1]) if pd.notna(upper.iloc[-1]) else None
        latest_lower = float(lower.iloc[-1]) if pd.notna(lower.iloc[-1]) else None
        latest_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
        if latest_upper is None or latest_lower is None or latest_rsi is None:
            return None

        rsi_os = self.config.get_pair_rsi_oversold(epic)
        rsi_ob = self.config.get_pair_rsi_overbought(epic)

        direction: Optional[str] = None
        extremity = 0.0
        if latest_close <= latest_lower and latest_rsi <= rsi_os:
            direction = "BUY"
            # extremity: how deep past thresholds
            rsi_depth = max(0.0, (rsi_os - latest_rsi)) / max(rsi_os, 1)
            bb_depth = max(0.0, (latest_lower - latest_close))
            extremity = min(1.0, rsi_depth + bb_depth / max(abs(latest_upper - latest_lower), 1e-6))
        elif latest_close >= latest_upper and latest_rsi >= rsi_ob:
            direction = "SELL"
            rsi_depth = max(0.0, (latest_rsi - rsi_ob)) / max(100 - rsi_ob, 1)
            bb_depth = max(0.0, (latest_close - latest_upper))
            extremity = min(1.0, rsi_depth + bb_depth / max(abs(latest_upper - latest_lower), 1e-6))

        if direction is None:
            return None

        # Confidence: min + extremity × range
        min_conf = self.config.get_pair_min_confidence(epic)
        max_conf = self.config.get_pair_max_confidence(epic)
        confidence = round(min_conf + extremity * (max_conf - min_conf), 3)

        # Build signal
        sl_pips = self.config.get_pair_fixed_stop_loss(epic)
        tp_pips = self.config.get_pair_fixed_take_profit(epic)
        monitor_only = self.config.is_pair_monitor_only(epic)

        now = datetime.now(timezone.utc)
        signal = {
            "signal": direction,
            "signal_type": direction.lower(),
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair,
            "entry_price": latest_close,
            "stop_loss_pips": sl_pips,
            "take_profit_pips": tp_pips,
            "confidence_score": confidence,
            "confidence": confidence,
            "risk_pips": sl_pips,
            "reward_pips": tp_pips,
            "signal_timestamp": now.isoformat(),
            "timestamp": now,
            "version": self.config.version,

            # Top-level fields that alert_history reads directly
            "adx": adx_value,
            "adx_htf": adx_htf,
            "rsi": latest_rsi,
            "market_regime": "ranging",
            "regime": "ranging",

            "monitor_only": monitor_only,

            "strategy_indicators": {
                "adx": adx_value,
                "adx_htf": adx_htf,
                "rsi": latest_rsi,
                "bb_upper": latest_upper,
                "bb_lower": latest_lower,
                "bb_mid": float(ma.iloc[-1]),
                "bb_mult": bb_mult,
                "extremity": round(extremity, 3),
            },
        }

        self._set_cooldown(epic)
        adx_str = f"{adx_value:.1f}" if adx_value is not None else "na"
        adx_htf_str = f"{adx_htf:.1f}" if adx_htf is not None else "na"
        self.logger.info(
            f"[MEAN_REVERSION] ✅ {direction} {epic} @ {latest_close:.5f} "
            f"RSI={latest_rsi:.1f} ADX(15m)={adx_str} ADX(1h)={adx_htf_str} "
            f"conf={confidence:.2f}"
        )
        return signal

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    def _get_adx(self, df: pd.DataFrame) -> Optional[float]:
        """Prefer pre-stamped df['adx'] (DataFetcher EMA-Wilder); fall back to
        recomputing with the same EMA-Wilder formula so standalone calls
        match live values exactly."""
        if "adx" in df.columns:
            try:
                v = df["adx"].iloc[-1]
                if v is not None and not pd.isna(v):
                    return float(v)
            except Exception:
                pass
        try:
            period = self.config.adx_period
            high, low, close = df["high"], df["low"], df["close"]
            tr = pd.concat(
                [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
                axis=1,
            ).max(axis=1)
            up = high - high.shift(1)
            dn = low.shift(1) - low
            plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
            minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
            a = 1.0 / period
            atr = tr.ewm(alpha=a, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
            minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            adx = dx.ewm(alpha=a, adjust=False).mean()
            return float(adx.iloc[-1])
        except Exception as e:
            self.logger.debug(f"[MEAN_REVERSION] ADX calc error: {e}")
            return None

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _check_cooldown(self, epic: str) -> bool:
        if epic not in self._cooldowns:
            return True
        now = self._current_timestamp or datetime.now(timezone.utc)
        if getattr(now, "tzinfo", None) is None:
            now = now.replace(tzinfo=timezone.utc)
        if now >= self._cooldowns[epic]:
            del self._cooldowns[epic]
            return True
        return False

    def _set_cooldown(self, epic: str) -> None:
        now = self._current_timestamp or datetime.now(timezone.utc)
        if getattr(now, "tzinfo", None) is None:
            now = now.replace(tzinfo=timezone.utc)
        self._cooldowns[epic] = now + timedelta(minutes=self.config.signal_cooldown_minutes)
