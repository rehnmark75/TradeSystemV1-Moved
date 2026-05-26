"""
SMC_MOMENTUM Strategy — Liquidity Sweep + Rejection Wick

Identifies prior-day and recent-swing liquidity pools on the 15m chart,
detects sweep-and-rejection (stop-hunt then close back inside), and enters
WITH the 4H EMA50 trend direction.

Gate 1 validated (May 3 2026): PF 1.215, 5/5 pairs, n=646 (with HTF filter).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .strategy_registry import register_strategy, StrategyInterface
except ImportError:
    from forex_scanner.core.strategies.strategy_registry import register_strategy, StrategyInterface

try:
    from forex_scanner.alerts.strategy_rejection_manager import StrategyRejectionManager
except ImportError:
    try:
        from ...alerts.strategy_rejection_manager import StrategyRejectionManager
    except ImportError:
        StrategyRejectionManager = None  # type: ignore[assignment,misc]

try:
    from services.smc_momentum_config_service import (
        get_smc_momentum_config,
        apply_config_overrides,
        SMCMomentumConfig,
    )
except ImportError:
    from forex_scanner.services.smc_momentum_config_service import (
        get_smc_momentum_config,
        apply_config_overrides,
        SMCMomentumConfig,
    )

try:
    from forex_scanner.core.strategies.helpers.liquidity_pool_detector import get_liquidity_pools
    from forex_scanner.core.strategies.helpers.sweep_detector import detect_sweep, SweepResult
except ImportError:
    from .helpers.liquidity_pool_detector import get_liquidity_pools
    from .helpers.sweep_detector import detect_sweep, SweepResult


logger = logging.getLogger(__name__)

_PIP_FX = 0.0001
_PIP_JPY = 0.01
_JPY_PAIRS = frozenset({"USDJPY", "EURJPY", "AUDJPY", "GBPJPY", "NZDJPY", "CADJPY", "CHFJPY"})


def _is_jpy(pair: str) -> bool:
    return pair.upper() in _JPY_PAIRS or "JPY" in pair.upper()


def _pip(pair: str) -> float:
    return _PIP_JPY if _is_jpy(pair) else _PIP_FX


def _calculate_atr(df_1h: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Return the latest ATR value from 1H bars."""
    if df_1h is None or len(df_1h) < period + 1:
        return None
    h = df_1h["high"].values
    l = df_1h["low"].values
    c = df_1h["close"].values
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    if len(tr) < period:
        return None
    return float(np.mean(tr[-period:]))


def _htf_bias(df_4h: pd.DataFrame, ema_period: int = 50) -> Optional[str]:
    """
    Return 'bullish' if close > EMA50, 'bearish' if close < EMA50, None if insufficient data.
    """
    if df_4h is None or len(df_4h) < ema_period:
        return None
    closes = df_4h["close"].values
    ema = float(np.mean(closes[-ema_period:]))
    last_close = float(closes[-1])
    return "bullish" if last_close > ema else "bearish"


def _htf_distance_pips(df_4h: pd.DataFrame, ema_period: int, pair: str) -> Optional[float]:
    """Return absolute distance in pips between 4H close and EMA50."""
    if df_4h is None or len(df_4h) < ema_period:
        return None
    closes = df_4h["close"].values
    ema = float(np.mean(closes[-ema_period:]))
    last_close = float(closes[-1])
    pip = _PIP_JPY if _is_jpy(pair) else _PIP_FX
    return abs(last_close - ema) / pip


@register_strategy("SMC_MOMENTUM")
class SMCMomentumStrategy(StrategyInterface):
    """
    SMC Liquidity Sweep + Rejection Wick strategy.

    Entry mechanic: 15m bar sweeps a liquidity pool by 3-15 pips and closes
    back inside. Entry WITH the 4H EMA50 trend direction (sweeping opposite,
    reverting with trend). SL beyond swept level; TP = ATR(14, 1H) * multiplier.
    """

    def __init__(
        self,
        config=None,
        logger: Optional[logging.Logger] = None,
        db_manager=None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager

        self.config: SMCMomentumConfig = get_smc_momentum_config()
        if config_override:
            self.config = apply_config_overrides(self.config, config_override)

        self._cooldowns: Dict[str, datetime] = {}
        self._pending_rejections: List[Dict] = []

        self._rej_mgr = None
        if db_manager is not None and StrategyRejectionManager is not None:
            try:
                self._rej_mgr = StrategyRejectionManager("SMC_MOMENTUM", db_manager)
            except Exception:
                pass

        self.logger.info(
            f"SMC_MOMENTUM Strategy v{self.config.version} initialized "
            f"(htf_alignment={self.config.htf_alignment_required})"
        )

    @property
    def strategy_name(self) -> str:
        return "SMC_MOMENTUM"

    def get_required_timeframes(self) -> List[str]:
        return [self.config.htf_timeframe, self.config.entry_timeframe, self.config.atr_timeframe]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect_signal(
        self,
        df_trigger: Optional[pd.DataFrame] = None,   # 15m bars
        df_4h: Optional[pd.DataFrame] = None,         # 4H bars for HTF bias
        epic: str = "",
        pair: str = "",
        df_1h: Optional[pd.DataFrame] = None,         # 1H bars for ATR
        current_time: Optional[datetime] = None,      # simulation time (backtest-aware)
        **kwargs,
    ) -> Optional[Dict]:
        """
        Detect a liquidity sweep + rejection signal.

        df_trigger: 15m OHLC bars (entry timeframe)
        df_4h:      4H OHLC bars (HTF EMA50 bias)
        df_1h:      1H OHLC bars (ATR calculation)
        """
        cfg = self.config
        now = current_time if current_time is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        # --- 0. Master kill switch ---
        if not cfg.is_pair_enabled(epic):
            return None

        # --- 1. Data sufficiency ---
        min_15m = cfg.swing_max_age_bars + cfg.swing_pivot_bars + 10
        if df_trigger is None or len(df_trigger) < min_15m:
            self.logger.debug(f"[SMC_MOMENTUM] Insufficient 15m data for {epic}: {0 if df_trigger is None else len(df_trigger)} bars")
            self._log_rejection(epic, "INSUFFICIENT_DATA", f"need {min_15m} bars", hour_utc=now.hour)
            return None
        if df_4h is None or len(df_4h) < cfg.htf_ema_period + 5:
            self.logger.debug(f"[SMC_MOMENTUM] Insufficient 4H data for {epic}")
            self._log_rejection(epic, "INSUFFICIENT_HTF_DATA", "not enough 4H bars", hour_utc=now.hour)
            return None

        # --- 2. Session gates (rollover / Asian / kill-zone) ---
        if "start_time" in df_trigger.columns:
            last_ts = pd.to_datetime(df_trigger["start_time"].iloc[-1], utc=True)
        else:
            last_ts = now
        hour_utc = last_ts.hour
        if cfg.is_pair_blocked_session(epic, hour_utc):
            self.logger.debug(f"[SMC_MOMENTUM] Session block active (hour {hour_utc} UTC) for {epic}")
            self._log_rejection(epic, "SESSION", f"hour={hour_utc} blocked", hour_utc=hour_utc)
            return None

        # --- 3. Cooldown ---
        if not self._check_cooldown(epic, cfg.get_pair_cooldown_minutes(epic), now):
            self._log_rejection(epic, "COOLDOWN", "within cooldown window", hour_utc=hour_utc)
            return None

        # --- 4. HTF bias + distance gate ---
        bias = _htf_bias(df_4h, cfg.htf_ema_period)
        if cfg.get_pair_htf_alignment_required(epic) and bias is None:
            self.logger.debug(f"[SMC_MOMENTUM] No clear HTF bias for {epic}")
            self._log_rejection(epic, "NO_HTF_BIAS", "HTF EMA bias indeterminate", hour_utc=hour_utc)
            return None
        htf_min_dist = cfg.get_pair_htf_min_distance_pips(epic)
        if htf_min_dist > 0:
            dist = _htf_distance_pips(df_4h, cfg.htf_ema_period, pair)
            if dist is not None and dist < htf_min_dist:
                self.logger.debug(
                    f"[SMC_MOMENTUM] HTF distance {dist:.1f} pips < {htf_min_dist} threshold for {epic}"
                )
                self._log_rejection(epic, "HTF_DISTANCE", f"{dist:.1f} pips < {htf_min_dist}", hour_utc=hour_utc,
                                    details={"dist_pips": round(dist, 2), "min_pips": htf_min_dist})
                return None

        # --- 5. Liquidity pool detection ---
        pool_highs, pool_lows = get_liquidity_pools(
            df_trigger,
            pivot_bars=cfg.swing_pivot_bars,
            max_age_bars=cfg.swing_max_age_bars,
        )

        if not pool_highs and not pool_lows:
            self.logger.debug(f"[SMC_MOMENTUM] No liquidity pools found for {epic}")
            self._log_rejection(epic, "NO_LIQUIDITY_POOLS", "no swing highs/lows in lookback", hour_utc=hour_utc)
            return None

        is_jpy = _is_jpy(pair)

        # --- 6. Sweep detection ---
        sweep = detect_sweep(
            df=df_trigger,
            pool_highs=pool_highs,
            pool_lows=pool_lows,
            sweep_min_pips=cfg.get_pair_sweep_min(epic, is_jpy),
            sweep_max_pips=cfg.get_pair_sweep_max(epic, is_jpy),
            wick_min_pct=cfg.get_pair_wick_min_pct(epic),
            is_jpy=is_jpy,
            double_sweep_lookback=6,
        )

        if sweep is None:
            self._log_rejection(epic, "NO_SWEEP", "no sweep-and-rejection pattern", hour_utc=hour_utc)
            return None

        # --- 7. HTF alignment gate (load-bearing: PF 0.90 → 1.22) ---
        # BUY (swept low → reversal up): requires bullish HTF bias
        # SELL (swept high → reversal down): requires bearish HTF bias
        if cfg.get_pair_htf_alignment_required(epic) and bias is not None:
            if sweep.direction == "BUY" and bias != "bullish":
                self._log_rejection(epic, "HTF_MISALIGN", f"BUY but bias={bias}", direction="BUY", hour_utc=hour_utc)
                return None
            if sweep.direction == "SELL" and bias != "bearish":
                self._log_rejection(epic, "HTF_MISALIGN", f"SELL but bias={bias}", direction="SELL", hour_utc=hour_utc)
                return None

        # --- 8. Momentum quality filter (optional) ---
        filter_mode = cfg.get_pair_momentum_filter_mode(epic)
        if filter_mode == "atr_expansion":
            if not self._atr_expansion_ok(df_trigger, cfg.atr_expansion_threshold):
                self._log_rejection(epic, "ATR_EXPANSION", "sweep bar TR not expanded", hour_utc=hour_utc)
                return None

        # --- 9. ATR for TP ---
        atr = _calculate_atr(df_1h, cfg.atr_period)
        if atr is None:
            # Fallback: estimate from 15m ATR, scale to 1H equivalent (~2× 15m ATR)
            atr = self._atr_from_15m(df_trigger, cfg.atr_period)
        if atr is None or atr <= 0:
            self.logger.debug(f"[SMC_MOMENTUM] Cannot compute ATR for {epic}")
            self._log_rejection(epic, "ATR_UNAVAILABLE", "ATR is None or zero", hour_utc=hour_utc)
            return None

        # --- 10. Confidence scoring ---
        confidence = self._score_confidence(sweep, bias, df_trigger, cfg, epic, is_jpy)
        min_conf = cfg.get_pair_min_confidence(epic)
        if confidence < min_conf:
            self._log_rejection(epic, "LOW_CONFIDENCE", f"{confidence:.3f} < {min_conf}",
                                direction=sweep.direction, hour_utc=hour_utc,
                                details={"confidence": round(confidence, 3), "min_confidence": min_conf})
            return None

        # --- 11. SL / TP ---
        pip_size = _pip(pair)
        sl_buffer = cfg.get_pair_sl_buffer_pips(epic, is_jpy) * pip_size
        tp_atr_mult = cfg.get_pair_tp_atr_multiplier(epic)

        bar = df_trigger.iloc[-1]
        entry_price = float(bar["close"])

        if sweep.direction == "BUY":
            sl_price = sweep.pool_level - sl_buffer
            sl_pips = (entry_price - sl_price) / pip_size
            tp_price = entry_price + atr * tp_atr_mult
            tp_pips = (tp_price - entry_price) / pip_size
        else:
            sl_price = sweep.pool_level + sl_buffer
            sl_pips = (sl_price - entry_price) / pip_size
            tp_price = entry_price - atr * tp_atr_mult
            tp_pips = (entry_price - tp_price) / pip_size

        if sl_pips <= 0 or tp_pips <= 0:
            self.logger.debug(f"[SMC_MOMENTUM] Degenerate SL/TP for {epic}: sl={sl_pips:.1f} tp={tp_pips:.1f}")
            return None

        # --- 12. Build signal ---
        signal = self._build_signal(
            epic=epic,
            pair=pair,
            sweep=sweep,
            entry_price=entry_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            confidence=confidence,
            bias=bias,
            current_time=now,
        )

        self._set_cooldown(epic, cfg.get_pair_cooldown_minutes(epic), now)
        self.logger.info(
            f"[SMC_MOMENTUM] Signal {sweep.direction} {pair} @ {entry_price:.5f} "
            f"conf={confidence:.2f} pool={sweep.pool_level:.5f} "
            f"({sweep.pool_type}, +{sweep.excess_pips}pip excess)"
        )
        try:
            from forex_scanner.core.strategies.helpers.smc_performance_metrics import enrich_signal_with_performance_metrics
            signal = enrich_signal_with_performance_metrics(
                signal, df_entry=None, df_trigger=df_trigger, df_htf=df_4h, epic=epic, logger=self.logger
            )
        except Exception as _pm_exc:
            self.logger.warning("[SMC_MOMENTUM] Performance metrics failed: %s", _pm_exc)
        return signal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _atr_expansion_ok(self, df: pd.DataFrame, threshold: float) -> bool:
        """Return True if the latest bar's TR is >= threshold × rolling-20-bar ATR."""
        if len(df) < 22:
            return False
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        tr_full = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        if len(tr_full) < 21:
            return False
        current_tr = tr_full[-1]
        rolling_avg = float(np.mean(tr_full[-21:-1]))
        return rolling_avg > 0 and current_tr >= threshold * rolling_avg

    def _atr_from_15m(self, df: pd.DataFrame, period: int) -> Optional[float]:
        """Fallback ATR from 15m bars, scaled to 1H (~4 bars per hour, so ×2 for 1H equiv)."""
        if df is None or len(df) < period + 1:
            return None
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        if len(tr) < period:
            return None
        return float(np.mean(tr[-period:])) * 2.0

    def _score_confidence(
        self,
        sweep: SweepResult,
        bias: Optional[str],
        df: pd.DataFrame,
        cfg: SMCMomentumConfig,
        epic: str,
        is_jpy: bool,
    ) -> float:
        score = 0.55  # base

        # Sweep AGAINST HTF trend wick (reverting WITH trend): +0.07
        if bias is not None:
            aligned = (sweep.direction == "BUY" and bias == "bullish") or \
                      (sweep.direction == "SELL" and bias == "bearish")
            if aligned:
                score += 0.07

        # Double sweep (consecutive sweeps of same pool within 6 bars): +0.06
        if sweep.is_double_sweep:
            score += 0.06

        # Wick > 70% of candle range: +0.04
        if sweep.wick_pct >= 0.70:
            score += 0.04

        # Sweep distance 5–10 pips (sweet spot): +0.03
        if 5.0 <= sweep.excess_pips <= 10.0:
            score += 0.03

        # ATR expansion: +0.03
        if self._atr_expansion_ok(df, cfg.atr_expansion_threshold):
            score += 0.03

        return min(score, cfg.max_confidence)

    def _build_signal(
        self,
        epic: str,
        pair: str,
        sweep: SweepResult,
        entry_price: float,
        sl_pips: float,
        tp_pips: float,
        confidence: float,
        bias: Optional[str],
        current_time: Optional[datetime] = None,
    ) -> Dict:
        ts = current_time if current_time is not None else datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return {
            "signal": sweep.direction,
            "signal_type": sweep.direction.lower(),
            "strategy": "SMC_MOMENTUM",
            "epic": epic,
            "pair": pair,
            "monitor_only": self.config.is_monitor_only(epic),
            "is_traded": self.config.is_traded(epic),
            "entry_price": entry_price,
            "risk_pips": round(sl_pips, 1),
            "reward_pips": round(tp_pips, 1),
            "entry_type": "REJECTION_WICK",
            "trigger_type": "LIQUIDITY_SWEEP",
            "nearest_support": sweep.pool_level if sweep.direction == "BUY" else None,
            "nearest_resistance": sweep.pool_level if sweep.direction == "SELL" else None,
            "confidence_score": round(confidence, 4),
            "confidence": round(confidence, 4),
            "signal_timestamp": ts.isoformat(),
            "timestamp": ts,
            "version": self.config.version,
            "strategy_metadata": {
                "swept_pool_type": sweep.pool_type,
                "pool_level": sweep.pool_level,
                "excess_pips": sweep.excess_pips,
                "wick_pct": sweep.wick_pct,
                "is_double_sweep": sweep.is_double_sweep,
                "htf_bias": bias,
                "monitor_only": self.config.is_monitor_only(epic),
                "is_traded": self.config.is_traded(epic),
            },
        }

    # ------------------------------------------------------------------
    # Cooldown management
    # ------------------------------------------------------------------

    def _check_cooldown(self, epic: str, cooldown_minutes: int, now: Optional[datetime] = None) -> bool:
        if epic not in self._cooldowns:
            return True
        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        cooldown_end = self._cooldowns[epic]
        if cooldown_end.tzinfo is None:
            cooldown_end = cooldown_end.replace(tzinfo=timezone.utc)
        if now >= cooldown_end:
            del self._cooldowns[epic]
            return True
        remaining = int((cooldown_end - now).total_seconds() / 60)
        self.logger.debug(f"[SMC_MOMENTUM] {epic} in cooldown ({remaining}m remaining)")
        return False

    def _set_cooldown(self, epic: str, cooldown_minutes: int, now: Optional[datetime] = None) -> None:
        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        self._cooldowns[epic] = now + timedelta(minutes=cooldown_minutes)

    def reset_cooldowns(self) -> None:
        """Clear all cooldowns — called by backtest harness at start of new run."""
        self._cooldowns.clear()
        self.logger.debug("[SMC_MOMENTUM] Cooldowns reset")

    # ------------------------------------------------------------------
    # Rejection logging
    # ------------------------------------------------------------------

    def _log_rejection(
        self,
        epic: str,
        reason: str,
        detail: str = "",
        direction: Optional[str] = None,
        hour_utc: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._pending_rejections.append({
            "epic": epic,
            "reason": reason,
            "detail": detail,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        if self._rej_mgr is not None:
            now = self._current_timestamp if hasattr(self, "_current_timestamp") and self._current_timestamp else datetime.now(timezone.utc)
            if isinstance(now, pd.Timestamp):
                now = now.to_pydatetime()
            if getattr(now, "tzinfo", None) is None:
                now = now.replace(tzinfo=timezone.utc)
            self._rej_mgr.reject(
                stage=reason,
                reason=detail or reason,
                epic=epic,
                pair=epic,
                direction=direction,
                hour_utc=hour_utc if hour_utc is not None else now.hour,
                scan_timestamp=now,
                details=details,
            )

    def flush_rejections(self) -> None:
        """Called by signal_detector after each scan tick; clears pending log buffer."""
        if self._rej_mgr is not None:
            self._rej_mgr.flush()
        self._pending_rejections.clear()
