#!/usr/bin/env python3
"""
EURUSD range-fade profiles.

Purpose
-------
Provide narrow, backtest-first EURUSD fade profiles that do not depend heavily
on ADX. The setup fades local extremes only when higher timeframe context is
supportive and the market is not obviously expanding.

Design
------
- Pair scope: EURUSD only (`CS.D.EURUSD.CEEM.IP`)
- Primary timeframe: profile-dependent (`15m` or `5m`)
- Higher timeframe context: 1h
- Entry family: controlled mean reversion / range fade
- Deployment mode: monitor-only by default
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from ...services.range_fade_config_service import (
        EURUSD_EPIC,
        EURUSDRangeFadeConfig,
        build_range_fade_config,
        get_range_fade_config,
    )
except ImportError:
    from forex_scanner.services.range_fade_config_service import (
        EURUSD_EPIC,
        EURUSDRangeFadeConfig,
        build_range_fade_config,
        get_range_fade_config,
    )

from .strategy_registry import StrategyInterface, register_strategy


logger = logging.getLogger(__name__)


def apply_config_overrides(
    cfg: EURUSDRangeFadeConfig, overrides: Optional[Dict[str, Any]]
) -> EURUSDRangeFadeConfig:
    """Apply backtest override values to the strategy config in place."""
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            continue
        current = getattr(cfg, key)
        try:
            if isinstance(current, bool):
                new_val = str(value).strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                new_val = int(float(value))
            elif isinstance(current, float):
                new_val = float(value)
            elif isinstance(current, list):
                new_val = list(value) if not isinstance(value, str) else [v.strip() for v in value.split(",") if v.strip()]
            else:
                new_val = value
            setattr(cfg, key, new_val)
        except Exception as exc:
            logger.warning("RANGE_FADE override failed for %s=%r: %s", key, value, exc)
    return cfg


@register_strategy("RANGE_FADE")
@register_strategy("EURUSD_RANGE_FADE")
class EURUSDRangeFadeStrategy(StrategyInterface):
    def __init__(self, config=None, logger=None, db_manager=None, config_override: dict = None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self._config_override: Optional[Dict[str, Any]] = config_override
        profile_name = None
        if isinstance(self._config_override, dict):
            profile_name = self._config_override.get("erf_profile") or self._config_override.get("profile")
        base_config = config or get_range_fade_config(profile_name)
        self.config = base_config
        apply_config_overrides(self.config, self._config_override)
        self._cooldowns: Dict[str, datetime] = {}
        self._current_timestamp: Optional[datetime] = None

        self.logger.info(
            "[RANGE_FADE] profile=%s v%s initialized | TF=%s HTF=%s "
            "BB(%s, %.1f) RSI(%s, %s/%s) SL/TP=%.1f/%.1f monitor_only=%s",
            self.config.profile_name,
            self.config.version,
            self.config.primary_timeframe,
            self.config.confirmation_timeframe,
            self.config.bb_period,
            self.config.bb_mult,
            self.config.rsi_period,
            self.config.rsi_oversold,
            self.config.rsi_overbought,
            self.config.fixed_stop_loss_pips,
            self.config.fixed_take_profit_pips,
            self.config.monitor_only,
        )

    @property
    def strategy_name(self) -> str:
        return self.config.strategy_name

    def get_required_timeframes(self) -> List[str]:
        return [self.config.confirmation_timeframe, self.config.primary_timeframe]

    def reset_cooldowns(self) -> None:
        self._cooldowns.clear()

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
        cfg = self.config
        df = df_trigger if df_trigger is not None else df_entry
        if df is None or len(df) < max(cfg.bb_period, cfg.range_lookback_bars) + 5:
            return None

        if epic != EURUSD_EPIC or not cfg.is_pair_enabled(epic):
            return None

        if not self._check_cooldown(epic):
            return None

        now = self._resolve_now(df)
        if not cfg.is_session_allowed(now.hour):
            return None

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ma = close.rolling(cfg.bb_period).mean()
        sd = close.rolling(cfg.bb_period).std()
        upper = ma + cfg.bb_mult * sd
        lower = ma - cfg.bb_mult * sd
        rsi = self._rsi(close, cfg.rsi_period)
        atr = self._atr(df, 14)

        latest_close = float(close.iloc[-1])
        latest_upper = self._safe_float(upper.iloc[-1])
        latest_lower = self._safe_float(lower.iloc[-1])
        latest_mid = self._safe_float(ma.iloc[-1])
        latest_rsi = self._safe_float(rsi.iloc[-1])
        latest_atr = self._safe_float(atr.iloc[-1])
        latest_bar_range = self._safe_float((high.iloc[-1] - low.iloc[-1]))
        if None in (latest_upper, latest_lower, latest_mid, latest_rsi, latest_atr, latest_bar_range):
            return None

        pip = 0.0001
        band_width_pips = (latest_upper - latest_lower) / pip
        if band_width_pips < cfg.min_band_width_pips or band_width_pips > cfg.max_band_width_pips:
            return None
        if latest_bar_range / pip > cfg.max_current_range_pips:
            return None

        prior_high = high.rolling(cfg.range_lookback_bars).max().shift(1)
        prior_low = low.rolling(cfg.range_lookback_bars).min().shift(1)
        range_high = self._safe_float(prior_high.iloc[-1])
        range_low = self._safe_float(prior_low.iloc[-1])
        if range_high is None or range_low is None:
            return None

        distance_to_low_pips = (latest_close - range_low) / pip
        distance_to_high_pips = (range_high - latest_close) / pip

        htf_bias = self._get_htf_bias(df_4h)
        if htf_bias is None:
            return None

        direction: Optional[str] = None
        rsi_extremity = 0.0
        band_penetration = 0.0
        range_proximity = 0.0

        if (
            latest_close <= latest_lower
            and latest_rsi <= cfg.rsi_oversold
            and distance_to_low_pips <= cfg.range_proximity_pips
            and htf_bias in ("bullish", "neutral")
        ):
            direction = "BUY"
            rsi_extremity = max(0.0, cfg.rsi_oversold - latest_rsi) / max(cfg.rsi_oversold, 1)
            band_penetration = max(0.0, latest_lower - latest_close) / max(latest_atr, 1e-6)
            range_proximity = max(0.0, cfg.range_proximity_pips - distance_to_low_pips) / max(cfg.range_proximity_pips, 1e-6)
        elif (
            latest_close >= latest_upper
            and latest_rsi >= cfg.rsi_overbought
            and distance_to_high_pips <= cfg.range_proximity_pips
            and htf_bias in ("bearish", "neutral")
        ):
            direction = "SELL"
            rsi_extremity = max(0.0, latest_rsi - cfg.rsi_overbought) / max(100 - cfg.rsi_overbought, 1)
            band_penetration = max(0.0, latest_close - latest_upper) / max(latest_atr, 1e-6)
            range_proximity = max(0.0, cfg.range_proximity_pips - distance_to_high_pips) / max(cfg.range_proximity_pips, 1e-6)

        if direction is None:
            return None

        score = min(1.0, 0.45 * rsi_extremity + 0.35 * min(1.0, band_penetration) + 0.20 * range_proximity)
        confidence = round(cfg.min_confidence + score * (cfg.max_confidence - cfg.min_confidence), 3)

        signal = {
            "signal": direction,
            "signal_type": direction.lower(),
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair,
            "entry_price": latest_close,
            "stop_loss_pips": cfg.fixed_stop_loss_pips,
            "take_profit_pips": cfg.fixed_take_profit_pips,
            "risk_pips": cfg.fixed_stop_loss_pips,
            "reward_pips": cfg.fixed_take_profit_pips,
            "confidence": confidence,
            "confidence_score": confidence,
            "signal_timestamp": now.isoformat(),
            "timestamp": now,
            "version": cfg.version,
            "market_regime": "range_fade",
            "regime": "range_fade",
            "monitor_only": cfg.monitor_only,
            "adx": self._get_adx(df),
            "adx_htf": self._get_adx(df_4h) if df_4h is not None and len(df_4h) >= 30 else None,
            "rsi": latest_rsi,
            "strategy_indicators": {
                "bb_upper": latest_upper,
                "bb_lower": latest_lower,
                "bb_mid": latest_mid,
                "band_width_pips": round(band_width_pips, 2),
                "rsi": latest_rsi,
                "htf_bias": htf_bias,
                "range_high": range_high,
                "range_low": range_low,
                "distance_to_low_pips": round(distance_to_low_pips, 2),
                "distance_to_high_pips": round(distance_to_high_pips, 2),
            },
        }

        self._set_cooldown(epic)
        self.logger.info(
            "[RANGE_FADE] %s %s @ %.5f RSI=%.1f htf=%s conf=%.2f",
            direction,
            epic,
            latest_close,
            latest_rsi,
            htf_bias,
            confidence,
        )
        return signal

    def _get_htf_bias(self, df_1h: Optional[pd.DataFrame]) -> Optional[str]:
        if df_1h is None or len(df_1h) < self.config.htf_ema_period + self.config.htf_slope_bars + 5:
            return None
        close = df_1h["close"].astype(float)
        ema = close.ewm(span=self.config.htf_ema_period, adjust=False).mean()
        latest_close = float(close.iloc[-1])
        latest_ema = self._safe_float(ema.iloc[-1])
        slope_ref = self._safe_float(ema.iloc[-1 - self.config.htf_slope_bars])
        if latest_ema is None or slope_ref is None:
            return None
        if latest_close > latest_ema and latest_ema > slope_ref:
            return "bullish"
        if latest_close < latest_ema and latest_ema < slope_ref:
            return "bearish"
        return "neutral" if self.config.allow_neutral_htf else None

    def _resolve_now(self, df: pd.DataFrame) -> datetime:
        now = self._current_timestamp
        if now is None:
            if "start_time" in df.columns:
                now = df["start_time"].iloc[-1]
            else:
                now = df.index[-1]
        if isinstance(now, pd.Timestamp):
            now = now.to_pydatetime()
        if getattr(now, "tzinfo", None) is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    def _get_adx(self, df: Optional[pd.DataFrame]) -> Optional[float]:
        if df is None or len(df) < 20:
            return None
        if "adx" in df.columns:
            v = df["adx"].iloc[-1]
            if v is not None and not pd.isna(v):
                return float(v)
        try:
            period = 14
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
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
        except Exception:
            return None

    def _check_cooldown(self, epic: str) -> bool:
        if epic not in self._cooldowns:
            return True
        now = self._current_timestamp or datetime.now(timezone.utc)
        if isinstance(now, pd.Timestamp):
            now = now.to_pydatetime()
        if getattr(now, "tzinfo", None) is None:
            now = now.replace(tzinfo=timezone.utc)
        if now >= self._cooldowns[epic]:
            del self._cooldowns[epic]
            return True
        return False

    def _set_cooldown(self, epic: str) -> None:
        now = self._current_timestamp or datetime.now(timezone.utc)
        if isinstance(now, pd.Timestamp):
            now = now.to_pydatetime()
        if getattr(now, "tzinfo", None) is None:
            now = now.replace(tzinfo=timezone.utc)
        self._cooldowns[epic] = now + timedelta(minutes=self.config.signal_cooldown_minutes)


def create_range_fade_strategy(config=None, logger=None, db_manager=None, config_override=None):
    return EURUSDRangeFadeStrategy(
        config=config,
        logger=logger,
        db_manager=db_manager,
        config_override=config_override,
    )


def create_eurusd_range_fade_strategy(config=None, logger=None, db_manager=None, config_override=None):
    return create_range_fade_strategy(
        config=config,
        logger=logger,
        db_manager=db_manager,
        config_override=config_override,
    )
