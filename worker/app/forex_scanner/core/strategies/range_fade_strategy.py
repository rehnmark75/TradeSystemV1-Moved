#!/usr/bin/env python3
"""
Range-fade strategy (5m primary).

Purpose
-------
Provide a narrow, backtest-first fade strategy that does not depend heavily
on ADX. The setup fades local extremes only when higher timeframe context is
supportive and the market is not obviously expanding.

Design
------
- Primary timeframe: 5m (only supported profile)
- Higher timeframe context: 1h
- Entry family: controlled mean reversion / range fade
- Deployment mode: monitor-only by default
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from services.range_fade_config_service import (
        EURUSD_EPIC,
        RangeFadeConfig,
        build_range_fade_config,
        get_range_fade_config,
    )
except ImportError:
    from forex_scanner.services.range_fade_config_service import (
        EURUSD_EPIC,
        RangeFadeConfig,
        build_range_fade_config,
        get_range_fade_config,
    )

from .strategy_registry import StrategyInterface, register_strategy

try:
    from ...alerts.strategy_rejection_manager import StrategyRejectionManager
except ImportError:
    try:
        from forex_scanner.alerts.strategy_rejection_manager import StrategyRejectionManager
    except ImportError:
        StrategyRejectionManager = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def apply_config_overrides(
    cfg: RangeFadeConfig, overrides: Optional[Dict[str, Any]]
) -> RangeFadeConfig:
    """Apply backtest override values to the strategy config in place."""
    if not overrides:
        return cfg
    applied: Dict[str, Any] = {}
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
            applied[key] = new_val
        except Exception as exc:
            logger.warning("RANGE_FADE override failed for %s=%r: %s", key, value, exc)
    if applied and hasattr(cfg, "backtest_overrides"):
        cfg.backtest_overrides.update(applied)
    return cfg


@register_strategy("RANGE_FADE")
class RangeFadeStrategy(StrategyInterface):
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
        self._cooldowns: Dict[Tuple[str, str], datetime] = {}  # (epic, direction) → unblock_time
        self._post_loss_session_blocks: Dict[str, str] = {}  # epic → blocked session key
        self._current_timestamp: Optional[datetime] = None
        self._rej_counts: Dict[str, int] = {}
        self._rej_last_log: datetime = datetime.now(timezone.utc)
        self._rej_log_interval_minutes: int = 15

        self._rej_mgr = None
        if db_manager is not None and StrategyRejectionManager is not None:
            try:
                self._rej_mgr = StrategyRejectionManager("RANGE_FADE", db_manager)
            except Exception:
                pass

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
        self._post_loss_session_blocks.clear()

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
        self.flush_rejections()
        cfg = self.config
        df = df_trigger if df_trigger is not None else df_entry
        if df is None or len(df) < max(cfg.bb_period, cfg.range_lookback_bars) + 5:
            self._reject(epic, "insufficient_data")
            return None

        if not cfg.is_pair_enabled(epic):
            self._reject(epic, "pair_disabled")
            return None

        now = self._resolve_now(df)
        if not self._check_post_loss_session_block(epic, now):
            self._reject(epic, "post_loss_session_block")
            return None

        if not cfg.is_session_allowed(now.hour, epic):
            self._reject(epic, "session_blocked")
            return None

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        bb_mult = cfg.get_pair_bb_mult(epic)
        ma = close.rolling(cfg.bb_period).mean()
        sd = close.rolling(cfg.bb_period).std()
        upper = ma + bb_mult * sd
        lower = ma - bb_mult * sd
        rsi = self._rsi(close, cfg.rsi_period)
        atr = self._atr(df, 14)
        _, _, macd_histogram = self._macd(close)

        latest_close = float(close.iloc[-1])
        latest_upper = self._safe_float(upper.iloc[-1])
        latest_lower = self._safe_float(lower.iloc[-1])
        latest_mid = self._safe_float(ma.iloc[-1])
        latest_rsi = self._safe_float(rsi.iloc[-1])
        latest_atr = self._safe_float(atr.iloc[-1])
        latest_macd_histogram = self._safe_float(macd_histogram.iloc[-1])
        latest_bar_range = self._safe_float((high.iloc[-1] - low.iloc[-1]))
        if None in (latest_upper, latest_lower, latest_mid, latest_rsi, latest_atr, latest_macd_histogram, latest_bar_range):
            self._reject(epic, "indicator_nan")
            return None

        er_period = cfg.get_pair_er_period(epic)
        latest_er = self._compute_er(close, er_period)

        rsi_oversold = cfg.get_pair_rsi_oversold(epic, "BUY")
        rsi_overbought = cfg.get_pair_rsi_overbought(epic, "SELL")
        range_proximity_pips = cfg.get_pair_range_proximity_pips(epic)
        min_band_width_pips = cfg.get_pair_min_band_width_pips(epic)
        max_band_width_pips = cfg.get_pair_max_band_width_pips(epic)
        max_current_range_pips = cfg.get_pair_max_current_range_pips(epic)
        min_macd_histogram_pips = cfg.get_pair_min_macd_histogram_pips(epic)
        range_lookback_bars = cfg.get_pair_range_lookback_bars(epic)

        pip = 0.01 if "JPY" in epic.upper() else 0.0001
        macd_histogram_pips = abs(latest_macd_histogram) / pip
        if macd_histogram_pips < min_macd_histogram_pips:
            self._reject(
                epic,
                "macd_histogram_too_low",
                details={
                    "macd_histogram_pips": round(macd_histogram_pips, 2),
                    "minimum": min_macd_histogram_pips,
                },
            )
            return None

        band_width_pips = (latest_upper - latest_lower) / pip
        if band_width_pips < min_band_width_pips or band_width_pips > max_band_width_pips:
            self._reject(epic, "band_width_out_of_range")
            return None
        if latest_bar_range / pip > max_current_range_pips:
            self._reject(epic, "bar_range_too_wide")
            return None

        adx_val = self._get_adx(df)
        adx_htf_val = self._get_adx(df_4h) if df_4h is not None and len(df_4h) >= 30 else None

        prior_high = high.rolling(range_lookback_bars).max().shift(1)
        prior_low = low.rolling(range_lookback_bars).min().shift(1)
        range_high = self._safe_float(prior_high.iloc[-1])
        range_low = self._safe_float(prior_low.iloc[-1])
        if range_high is None or range_low is None:
            self._reject(epic, "no_prior_range")
            return None

        distance_to_low_pips = (latest_close - range_low) / pip
        distance_to_high_pips = (range_high - latest_close) / pip

        htf_bias = self._get_htf_bias(df_4h, epic)
        if htf_bias is None:
            self._reject(epic, "no_htf_bias")
            return None

        direction: Optional[str] = None
        rsi_extremity = 0.0
        band_penetration = 0.0
        range_proximity = 0.0

        if (
            latest_close <= latest_lower
            and latest_rsi <= rsi_oversold
            and distance_to_low_pips <= range_proximity_pips
            and cfg.is_htf_bias_allowed(epic, "BUY", htf_bias)
        ):
            direction = "BUY"
            rsi_extremity = max(0.0, rsi_oversold - latest_rsi) / max(rsi_oversold, 1)
            band_penetration = max(0.0, latest_lower - latest_close) / max(latest_atr, 1e-6)
            range_proximity = max(0.0, range_proximity_pips - distance_to_low_pips) / max(range_proximity_pips, 1e-6)
        elif (
            latest_close >= latest_upper
            and latest_rsi >= rsi_overbought
            and distance_to_high_pips <= range_proximity_pips
            and cfg.is_htf_bias_allowed(epic, "SELL", htf_bias)
        ):
            direction = "SELL"
            rsi_extremity = max(0.0, latest_rsi - rsi_overbought) / max(100 - rsi_overbought, 1)
            band_penetration = max(0.0, latest_close - latest_upper) / max(latest_atr, 1e-6)
            range_proximity = max(0.0, range_proximity_pips - distance_to_high_pips) / max(range_proximity_pips, 1e-6)

        if direction is None:
            self._reject(epic, "no_trigger")
            return None
        if not self._check_cooldown(epic, direction):
            self._reject(epic, "cooldown", direction=direction)
            return None
        if not cfg.is_direction_allowed(epic, direction):
            self._reject(epic, "direction_blocked", direction=direction)
            return None
        if not cfg.is_direction_session_allowed(now.hour, epic, direction):
            self._reject(epic, "direction_session_blocked", direction=direction)
            return None
        adx_ceil = cfg.get_pair_adx_ceiling(epic, direction)
        if adx_val is not None and adx_ceil > 0 and adx_val > adx_ceil:
            self._reject(
                epic,
                "adx_ceiling",
                direction=direction,
                details={"adx": round(adx_val, 1), "ceiling": adx_ceil},
            )
            return None

        htf_adx_ceil = cfg.get_pair_htf_adx_ceiling(epic)
        if adx_htf_val is not None and htf_adx_ceil > 0 and adx_htf_val > htf_adx_ceil:
            self._reject(
                epic,
                "htf_adx_ceiling",
                direction=direction,
                details={"adx_htf": round(adx_htf_val, 1), "htf_ceiling": htf_adx_ceil},
            )
            return None

        er_floor = cfg.get_pair_er_floor(epic)
        if latest_er is not None and er_floor > 0.0 and latest_er < er_floor:
            self._reject(
                epic,
                "er_floor",
                direction=direction,
                details={"er": round(latest_er, 3), "floor": er_floor},
            )
            return None

        er_ceil = cfg.get_pair_er_ceiling(epic, direction)
        if latest_er is not None and er_ceil < 999.0 and latest_er > er_ceil:
            self._reject(
                epic,
                "er_ceiling",
                direction=direction,
                details={"er": latest_er, "ceiling": er_ceil},
            )
            return None

        score = min(1.0, 0.45 * rsi_extremity + 0.35 * min(1.0, band_penetration) + 0.20 * range_proximity)
        confidence = round(cfg.min_confidence + score * (cfg.max_confidence - cfg.min_confidence), 3)

        pip_size = 0.01 if "JPY" in epic.upper() else 0.0001
        sl_pips, tp_pips = self._resolve_sl_tp_pips(cfg, epic, band_width_pips)
        sl_distance = sl_pips * pip_size
        tp_distance = tp_pips * pip_size
        if direction == "SELL":
            stop_loss_price = latest_close + sl_distance
            take_profit_price = latest_close - tp_distance
        else:
            stop_loss_price = latest_close - sl_distance
            take_profit_price = latest_close + tp_distance
        rr_ratio = (tp_pips / sl_pips) if sl_pips else 0.0

        signal = {
            "signal": direction,
            "signal_type": direction.lower(),
            "strategy": self.strategy_name,
            "epic": epic,
            "pair": pair,
            "entry_price": latest_close,
            "stop_loss": round(stop_loss_price, 5),
            "take_profit": round(take_profit_price, 5),
            "stop_loss_pips": sl_pips,
            "take_profit_pips": tp_pips,
            "risk_pips": sl_pips,
            "reward_pips": tp_pips,
            "risk_reward_ratio": round(rr_ratio, 4),
            "confidence": confidence,
            "confidence_score": confidence,
            "signal_timestamp": now.isoformat(),
            "timestamp": now,
            "version": cfg.version,
            "market_regime": "range_fade",
            "regime": "range_fade",
            "monitor_only": cfg.is_pair_monitor_only(epic),
            "scalp_mode": False,
            "adx": adx_val,
            "adx_htf": adx_htf_val,
            "rsi": latest_rsi,
            "efficiency_ratio": latest_er,
            "strategy_indicators": {
                "bb_upper": latest_upper,
                "bb_lower": latest_lower,
                "bb_mid": latest_mid,
                "band_width_pips": round(band_width_pips, 2),
                "bb_mult": bb_mult,
                "macd_histogram_pips": round(macd_histogram_pips, 2),
                "min_macd_histogram_pips": min_macd_histogram_pips,
                "adx_ceiling": adx_ceil,
                "adx_htf": adx_htf_val,
                "dynamic_sl_tp_enabled": cfg.dynamic_sl_tp_enabled,
                "allowed_directions": cfg._override(epic, "allowed_directions", cfg.allowed_directions),
                "rsi": latest_rsi,
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
                "htf_bias": htf_bias,
                "range_high": range_high,
                "range_low": range_low,
                "distance_to_low_pips": round(distance_to_low_pips, 2),
                "distance_to_high_pips": round(distance_to_high_pips, 2),
                # Timeframe hints for chart generator: 4h macro + 1h HTF bias + 5m entry
                "tier1_ema": {"timeframe": "1h"},
                "tier2_swing": {"timeframe": "5m"},  # same as entry → deduplicates to 3 panels
                "tier3_entry": {"timeframe": "5m"},
            },
        }

        # LPF gate — run before arming cooldown so a blocked signal doesn't
        # suppress the next valid setup for the full cooldown window.
        if getattr(self, 'LPF_ENABLED', True):
            try:
                try:
                    from .lpf_gate import apply_lpf_gate
                except ImportError:
                    from forex_scanner.core.strategies.lpf_gate import apply_lpf_gate
                signal = apply_lpf_gate(signal, self.logger, backtest_timestamp=now)
            except Exception as _lpf_exc:
                self.logger.warning("LPF gate error (letting signal through): %s", _lpf_exc)

        if signal is not None:
            self._set_cooldown(epic, direction)
            self.logger.info(
                "[RANGE_FADE] %s %s @ %.5f RSI=%.1f htf=%s conf=%.2f",
                direction,
                epic,
                latest_close,
                latest_rsi,
                htf_bias,
                confidence,
            )
            try:
                from forex_scanner.core.strategies.helpers.smc_performance_metrics import enrich_signal_with_performance_metrics
                signal = enrich_signal_with_performance_metrics(
                    signal, df_entry=df_entry, df_trigger=df_trigger, df_htf=df_4h, epic=epic, logger=self.logger
                )
            except Exception as _pm_exc:
                self.logger.warning("[RANGE_FADE] Performance metrics failed: %s", _pm_exc)
        return signal

    def _resolve_sl_tp_pips(
        self,
        cfg: RangeFadeConfig,
        epic: str,
        band_width_pips: float,
    ) -> tuple[float, float]:
        if not cfg.dynamic_sl_tp_enabled:
            return (
                cfg.get_pair_fixed_stop_loss_pips(epic),
                cfg.get_pair_fixed_take_profit_pips(epic),
            )

        sl_raw = band_width_pips * cfg.dynamic_sl_band_width_sl_mult
        tp_raw = band_width_pips * cfg.dynamic_sl_band_width_tp_mult
        sl_pips = min(max(sl_raw, cfg.dynamic_sl_min_pips), cfg.dynamic_sl_max_pips)
        tp_pips = min(max(tp_raw, cfg.dynamic_tp_min_pips), cfg.dynamic_tp_max_pips)
        return round(float(sl_pips), 2), round(float(tp_pips), 2)

    def _get_htf_bias(self, df_1h: Optional[pd.DataFrame], epic: str = "") -> Optional[str]:
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
        allow_neutral = bool(self.config._override(epic, "allow_neutral_htf", self.config.allow_neutral_htf)) if epic else self.config.allow_neutral_htf
        return "neutral" if allow_neutral else None

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

    @staticmethod
    def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line, macd_signal, macd_line - macd_signal

    @staticmethod
    def _compute_er(close: pd.Series, period: int = 14) -> Optional[float]:
        """Kaufman Efficiency Ratio: 1.0 = clean trend, 0.0 = choppy/ranging."""
        if len(close) < period + 1:
            return None
        vals = close.values
        direction = abs(float(vals[-1]) - float(vals[-period - 1]))
        volatility = sum(abs(float(vals[-i]) - float(vals[-i - 1])) for i in range(1, period + 1))
        if volatility == 0:
            return 0.0
        return round(direction / volatility, 4)

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

    def _check_cooldown(self, epic: str, direction: str) -> bool:
        key = (epic, direction.upper())
        if key not in self._cooldowns:
            return True
        now = self._current_timestamp or datetime.now(timezone.utc)
        if isinstance(now, pd.Timestamp):
            now = now.to_pydatetime()
        if getattr(now, "tzinfo", None) is None:
            now = now.replace(tzinfo=timezone.utc)
        if now >= self._cooldowns[key]:
            del self._cooldowns[key]
            return True
        return False

    def _set_cooldown(self, epic: str, direction: str) -> None:
        now = self._current_timestamp or datetime.now(timezone.utc)
        if isinstance(now, pd.Timestamp):
            now = now.to_pydatetime()
        if getattr(now, "tzinfo", None) is None:
            now = now.replace(tzinfo=timezone.utc)
        same_dir = direction.upper()
        opp_dir = "SELL" if same_dir == "BUY" else "BUY"
        same_mins = self.config.get_pair_same_direction_cooldown_minutes(epic)
        opp_mins = self.config.get_pair_signal_cooldown_minutes(epic)
        if same_mins > 0:
            self._cooldowns[(epic, same_dir)] = now + timedelta(minutes=same_mins)
        if opp_mins > 0:
            self._cooldowns[(epic, opp_dir)] = now + timedelta(minutes=opp_mins)

    @staticmethod
    def _session_key(hour: int) -> str:
        if hour >= 22 or hour < 6:
            return "Asian"
        if hour < 12:
            return "London"
        if hour < 16:
            return "Overlap"
        if hour < 21:
            return "NY"
        return "Late"

    def mark_signal_outcome(self, epic: str, was_loss: bool, signal_hour: int) -> None:
        """Call after a trade resolves. Sets a same-session block if the trade was a loss."""
        if was_loss:
            self._post_loss_session_blocks[epic] = self._session_key(signal_hour)
        elif epic in self._post_loss_session_blocks:
            del self._post_loss_session_blocks[epic]

    def _check_post_loss_session_block(self, epic: str, now: datetime) -> bool:
        """
        Returns False (block) when the most recent RANGE_FADE signal for this epic was
        a loss in the same session bucket as the current hour.

        In backtest mode (no db_manager), falls back to the in-memory state set by
        mark_signal_outcome(). In live mode also queries trade_log as a safety net.
        """
        if not self.config.is_post_loss_session_block_enabled(epic):
            return True

        current_session = self._session_key(now.hour)

        # In-memory check (works in both backtest and live)
        if self._post_loss_session_blocks.get(epic) == current_session:
            return False

        # Live-only DB check: most recent closed RANGE_FADE trade for this epic
        if self.db_manager is not None:
            try:
                query = """
                    SELECT tl.pips_gained,
                           EXTRACT(HOUR FROM ah.created_at)::int AS signal_hour
                    FROM trade_log tl
                    JOIN alert_history ah ON ah.id = tl.alert_id
                    WHERE ah.epic = :epic
                      AND ah.strategy = 'RANGE_FADE'
                      AND tl.status IN ('closed', 'CLOSED')
                      AND tl.closed_at IS NOT NULL
                    ORDER BY ah.created_at DESC
                    LIMIT 1
                """
                rows = self.db_manager.execute_query(query, {"epic": epic})
                if rows is not None and not rows.empty:
                    pips_gained = rows.iloc[0]["pips_gained"]
                    signal_hour = rows.iloc[0]["signal_hour"]
                    if pips_gained is not None and float(pips_gained) < 0:
                        prev_session = self._session_key(int(signal_hour))
                        if prev_session == current_session:
                            self._post_loss_session_blocks[epic] = current_session
                            return False
            except Exception as exc:
                self.logger.debug("[RANGE_FADE] post_loss_session DB check failed: %s", exc)

        return True

    def _reject(self, epic: str, reason: str, direction: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        stage = reason.upper()
        self._rej_counts[stage] = self._rej_counts.get(stage, 0) + 1
        self.logger.info("[RANGE_FADE] %s ❌ %s", epic, stage)
        if self._rej_mgr is not None:
            now = self._current_timestamp or datetime.now(timezone.utc)
            if isinstance(now, pd.Timestamp):
                now = now.to_pydatetime()
            if getattr(now, "tzinfo", None) is None:
                now = now.replace(tzinfo=timezone.utc)
            self._rej_mgr.reject(
                stage=stage,
                reason=reason,
                epic=epic,
                pair=epic,
                direction=direction,
                hour_utc=now.hour,
                scan_timestamp=now,
                details=details,
            )

    def flush_rejections(self) -> None:
        if self._rej_mgr is not None:
            self._rej_mgr.flush()
        if not self._rej_counts:
            return
        now = datetime.now(timezone.utc)
        elapsed = (now - self._rej_last_log).total_seconds() / 60.0
        if elapsed < self._rej_log_interval_minutes:
            return
        top = sorted(self._rej_counts.items(), key=lambda x: -x[1])[:6]
        total = sum(self._rej_counts.values())
        self.logger.info(
            "[RANGE_FADE] Rejection rollup (last %dmin, %d rejections): %s",
            int(elapsed), total, dict(top),
        )
        self._rej_counts.clear()
        self._rej_last_log = now


def create_range_fade_strategy(config=None, logger=None, db_manager=None, config_override=None):
    return RangeFadeStrategy(
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


# Back-compat alias (deprecated — to be removed in a future release)
EURUSDRangeFadeStrategy = RangeFadeStrategy
