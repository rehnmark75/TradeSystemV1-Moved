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

import copy
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .strategy_registry import StrategyInterface, register_strategy

try:
    from forex_scanner.services.mean_reversion_config_service import (
        MeanReversionConfig, MeanReversionConfigService, get_mean_reversion_config,
    )
except ImportError:
    from services.mean_reversion_config_service import (  # type: ignore[no-redef]
        MeanReversionConfig, MeanReversionConfigService, get_mean_reversion_config,
    )

try:
    from forex_scanner.services.regime_classifier import get_adx_from_df
except ImportError:
    from services.regime_classifier import get_adx_from_df  # type: ignore[no-redef]

try:
    from forex_scanner.alerts.strategy_rejection_manager import StrategyRejectionManager
except ImportError:
    try:
        from alerts.strategy_rejection_manager import StrategyRejectionManager
    except ImportError:
        StrategyRejectionManager = None  # type: ignore[assignment,misc]



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

    def __init__(self, config=None, logger=None, db_manager=None, config_override: Optional[Dict[str, Any]] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self.config = get_mean_reversion_config()

        if config_override:
            self.config = copy.deepcopy(self.config)
            self._apply_config_override(config_override)

        self._cooldowns: Dict[str, datetime] = {}
        self._current_timestamp: Optional[datetime] = None

        self._rej_mgr = None
        if db_manager is not None and StrategyRejectionManager is not None:
            try:
                self._rej_mgr = StrategyRejectionManager("MEAN_REVERSION", db_manager)
            except Exception:
                pass

        self.logger.info(
            f"[MEAN_REVERSION] v{self.config.version} initialized | "
            f"hard ADX: 15m≤{self.config.adx_hard_ceiling_primary} "
            f"1h≤{self.config.adx_hard_ceiling_htf}"
            + (f" | overrides={len(config_override)}" if config_override else "")
        )

    # ------------------------------------------------------------------
    # Ablation / backtest-isolation support
    # ------------------------------------------------------------------

    _PAIR_LEVEL_KEYS = {"is_enabled", "monitor_only"}

    @staticmethod
    def _coerce_value(key: str, value: Any, reference: Any) -> Any:
        """Type-coerce override value to match the reference attribute's type."""
        if isinstance(reference, bool):
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "1", "yes", "on")
        if isinstance(reference, int) and not isinstance(reference, bool):
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return reference
        if isinstance(reference, float):
            try:
                return float(value)
            except (ValueError, TypeError):
                return reference
        return value

    def _apply_config_override(self, overrides: Dict[str, Any]) -> None:
        """Apply a flat key=value override dict to the per-instance config copy.

        Global attributes (rsi_oversold, bb_mult, adx_hard_ceiling_*, ...) are
        written onto the MeanReversionConfig dataclass and any shadowing per-pair
        entries are wiped so the override wins. Pair-level flags (is_enabled,
        monitor_only) are broadcast across every loaded pair override so a
        single-pair backtest isn't blocked by DB state.
        """
        pair_rows = getattr(self.config, "_pair_overrides", None) or {}

        for raw_key, raw_value in overrides.items():
            key = str(raw_key)
            if key in self._PAIR_LEVEL_KEYS:
                # Force the flag onto every loaded pair override row + wipe
                # JSONB shadow. Global dataclass has no attribute for these.
                for row in pair_rows.values():
                    coerced = self._coerce_value(key, raw_value, row.get(key, False))
                    row[key] = coerced
                    jsonb = row.get("parameter_overrides") or {}
                    if key in jsonb:
                        jsonb.pop(key, None)
                        row["parameter_overrides"] = jsonb
                continue

            # Global attr path: wipe per-pair shadows, then set global.
            for row in pair_rows.values():
                row.pop(key, None)
                jsonb = row.get("parameter_overrides") or {}
                if key in jsonb:
                    jsonb.pop(key, None)
                    row["parameter_overrides"] = jsonb

            if hasattr(self.config, key):
                current = getattr(self.config, key)
                coerced = self._coerce_value(key, raw_value, current)
                setattr(self.config, key, coerced)

    @property
    def strategy_name(self) -> str:
        return "MEAN_REVERSION"

    def get_config(self):
        return self.config

    def get_required_timeframes(self) -> List[str]:
        return [self.config.confirmation_timeframe, self.config.primary_timeframe]

    def flush_rejections(self) -> None:
        if self._rej_mgr is not None:
            self._rej_mgr.flush()

    def _reject(
        self,
        stage: str,
        reason: str,
        epic: str,
        pair: str,
        hour_utc: Optional[int] = None,
        direction: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._rej_mgr is not None:
            self._rej_mgr.reject(
                stage=stage,
                reason=reason,
                epic=epic,
                pair=pair,
                direction=direction,
                hour_utc=hour_utc,
                scan_timestamp=self._current_timestamp,
                details=details,
            )

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

        # Resolve UTC hour once for all rejection calls below
        _eval_ts = current_timestamp or datetime.now(timezone.utc)
        if getattr(_eval_ts, "tzinfo", None) is None:
            _eval_ts = _eval_ts.replace(tzinfo=timezone.utc)
        _utc_hour = _eval_ts.hour

        df = df_trigger if df_trigger is not None else df_entry
        if df is None or len(df) < max(self.config.bb_period, self.config.rsi_period) + 5:
            self._reject("INSUFFICIENT_DATA", "not enough rows", epic, pair, hour_utc=_utc_hour)
            return None

        if not self.config.is_pair_enabled(epic):
            self.logger.debug(f"[MEAN_REVERSION] {epic} disabled — skipping")
            return None

        if not self._check_cooldown(epic):
            self._reject("COOLDOWN", "cooldown active", epic, pair, hour_utc=_utc_hour)
            return None

        # Session window gate (optional — configured per-pair)
        session_start = self.config.get_pair_session_start_hour(epic)
        session_end = self.config.get_pair_session_end_hour(epic)
        if session_start is not None and session_end is not None:
            if not (session_start <= _utc_hour <= session_end):
                self.logger.debug(
                    f"[MEAN_REVERSION] {epic}: outside session window (hour={_utc_hour}, "
                    f"window={session_start}-{session_end})"
                )
                self._reject("SESSION", f"hour={_utc_hour} outside {session_start}-{session_end}",
                             epic, pair, hour_utc=_utc_hour,
                             details={"hour": _utc_hour, "session_start": session_start, "session_end": session_end})
                return None

        adx_value = self._get_adx(df)
        adx_htf = self._get_adx(df_4h) if df_4h is not None and len(df_4h) >= 30 else None

        use_low_vol_filter = self.config.get_pair_low_vol_regime_filter_enabled(epic)
        if use_low_vol_filter:
            # Low-vol regime filter: ATR ≤ threshold + flat EMA slope.
            # Used instead of ADX for touch-entry pairs (late-session fade).
            if not self._low_vol_regime_passes(df, epic):
                self._reject("LOW_VOL_REGIME", "low-vol regime filter failed", epic, pair,
                             hour_utc=_utc_hour,
                             details={"adx": round(adx_value, 2) if adx_value is not None else None})
                return None
        elif self.config.hard_adx_gate_enabled:
            # Hard ADX gates — always enforced (no trust_regime_routing bypass).
            # Fail-closed: missing/NaN ADX blocks the signal (a reversion strategy
            # with no regime context is unsafe).
            pri_ceiling = self.config.get_pair_adx_hard_ceiling_primary(epic)
            htf_ceiling = self.config.get_pair_adx_hard_ceiling_htf(epic)
            if adx_value is None or pd.isna(adx_value):
                self.logger.debug(f"[MEAN_REVERSION] ❌ {epic} 15m ADX unavailable — fail-closed")
                self._reject("ADX_PRIMARY", "15m ADX unavailable (fail-closed)", epic, pair,
                             hour_utc=_utc_hour)
                return None
            if adx_value > pri_ceiling:
                self.logger.debug(
                    f"[MEAN_REVERSION] ❌ {epic} 15m ADX {adx_value:.1f} > {pri_ceiling}"
                )
                self._reject("ADX_PRIMARY", f"15m ADX {adx_value:.1f} > ceiling {pri_ceiling}",
                             epic, pair, hour_utc=_utc_hour,
                             details={"adx_15m": round(adx_value, 2), "ceiling": pri_ceiling})
                return None
            if adx_htf is None or pd.isna(adx_htf):
                self.logger.debug(f"[MEAN_REVERSION] ❌ {epic} 1h ADX unavailable — fail-closed")
                self._reject("ADX_HTF", "1h ADX unavailable (fail-closed)", epic, pair,
                             hour_utc=_utc_hour,
                             details={"adx_15m": round(adx_value, 2)})
                return None
            if adx_htf > htf_ceiling:
                self.logger.debug(
                    f"[MEAN_REVERSION] ❌ {epic} 1h ADX {adx_htf:.1f} > {htf_ceiling}"
                )
                self._reject("ADX_HTF", f"1h ADX {adx_htf:.1f} > ceiling {htf_ceiling}",
                             epic, pair, hour_utc=_utc_hour,
                             details={"adx_15m": round(adx_value, 2), "adx_1h": round(adx_htf, 2),
                                      "ceiling": htf_ceiling})
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
        entry_mode = self.config.get_pair_entry_mode(epic)

        direction: Optional[str] = None
        extremity = 0.0
        band_width = max(abs(latest_upper - latest_lower), 1e-6)

        if entry_mode == "touch":
            # Touch entry: current candle close beyond band with RSI extreme.
            # No waiting for next candle — fires at the extreme itself.
            if latest_close <= latest_lower and latest_rsi <= rsi_os:
                direction = "BUY"
                rsi_depth = max(0.0, (rsi_os - latest_rsi)) / max(rsi_os, 1)
                bb_depth = max(0.0, latest_lower - latest_close)
                extremity = min(1.0, rsi_depth + bb_depth / band_width)
            elif latest_close >= latest_upper and latest_rsi >= rsi_ob:
                direction = "SELL"
                rsi_depth = max(0.0, (latest_rsi - rsi_ob)) / max(100 - rsi_ob, 1)
                bb_depth = max(0.0, latest_close - latest_upper)
                extremity = min(1.0, rsi_depth + bb_depth / band_width)
        else:
            # Rejection entry (default): prev candle breaches band with RSI extreme,
            # current candle closes back inside — confirming rejection.
            prev_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
            prev_upper = float(upper.iloc[-2]) if pd.notna(upper.iloc[-2]) else latest_upper
            prev_lower = float(lower.iloc[-2]) if pd.notna(lower.iloc[-2]) else latest_lower
            prev_rsi = float(rsi.iloc[-2]) if len(rsi) >= 2 and pd.notna(rsi.iloc[-2]) else latest_rsi

            if prev_close <= prev_lower and prev_rsi <= rsi_os and latest_close > latest_lower:
                direction = "BUY"
                rsi_depth = max(0.0, (rsi_os - prev_rsi)) / max(rsi_os, 1)
                bb_depth = max(0.0, prev_lower - prev_close)
                extremity = min(1.0, rsi_depth + bb_depth / band_width)
            elif prev_close >= prev_upper and prev_rsi >= rsi_ob and latest_close < latest_upper:
                direction = "SELL"
                rsi_depth = max(0.0, (prev_rsi - rsi_ob)) / max(100 - rsi_ob, 1)
                bb_depth = max(0.0, prev_close - prev_upper)
                extremity = min(1.0, rsi_depth + bb_depth / band_width)

        if direction is None:
            self._reject("NO_PATTERN", "no BB+RSI pattern matched", epic, pair,
                         hour_utc=_utc_hour,
                         details={"rsi": round(latest_rsi, 2) if latest_rsi is not None else None,
                                  "close": latest_close, "upper_bb": latest_upper,
                                  "lower_bb": latest_lower,
                                  "adx_15m": round(adx_value, 2) if adx_value is not None else None,
                                  "adx_1h": round(adx_htf, 2) if adx_htf is not None else None})
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

        # Regime label derived from ADX rather than hardcoded "ranging" — the
        # alert pipeline will otherwise relabel from ADX anyway.
        if adx_value is None or pd.isna(adx_value):
            regime_label = "unknown"
        elif adx_value < 18:
            regime_label = "ranging"
        elif adx_value < 25:
            regime_label = "weak_trend"
        else:
            regime_label = "trending"

        # UTC session label so analytics/LPF can scope by session.
        h = now.hour
        if 23 <= h or h < 7:
            session = "asian"
        elif 7 <= h < 12:
            session = "london"
        elif 12 <= h < 16:
            session = "overlap"
        elif 16 <= h < 21:
            session = "newyork"
        else:
            session = "off_hours"

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
            "market_regime": regime_label,
            "regime": regime_label,
            "market_session": session,

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
                "entry_mode": entry_mode,
            },
        }

        self._set_cooldown(epic)
        adx_str = f"{adx_value:.1f}" if adx_value is not None else "na"
        adx_htf_str = f"{adx_htf:.1f}" if adx_htf is not None else "na"
        self.logger.info(
            f"[MEAN_REVERSION] ✅ {direction} {epic} @ {latest_close:.5f} "
            f"RSI={latest_rsi:.1f} ADX(15m)={adx_str} ADX(1h)={adx_htf_str} "
            f"conf={confidence:.2f} [{entry_mode} entry]"
        )

        # LPF gate — strategy-side opt-in (LPF_ENABLED = True)
        if getattr(self, 'LPF_ENABLED', True):
            try:
                try:
                    from .lpf_gate import apply_lpf_gate
                except ImportError:
                    from forex_scanner.core.strategies.lpf_gate import apply_lpf_gate
                signal = apply_lpf_gate(signal, self.logger, backtest_timestamp=self._current_timestamp)
            except Exception as _lpf_exc:
                self.logger.warning("LPF gate error (letting signal through): %s", _lpf_exc)
        return signal

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    def _low_vol_regime_passes(self, df: pd.DataFrame, epic: str) -> bool:
        """ATR14 ≤ threshold AND EMA50 slope flat over lookback candles.

        Used in place of the ADX gate for touch-entry pairs (late-session fade).
        The ATR ceiling is calibrated for the strategy's primary timeframe (15m by default).
        """
        cfg = self.config
        pip = 0.01 if "JPY" in epic.upper() else 0.0001

        atr_max = cfg.get_pair_regime_atr_max_pips(epic)
        ema_period = cfg.get_pair_regime_ema_period(epic)
        lookback = cfg.get_pair_regime_ema_lookback_candles(epic)
        ema_max_change = cfg.get_pair_regime_ema_max_change_pips(epic)

        close = df["close"].astype(float)
        if len(close) < max(ema_period + lookback, 20):
            self.logger.debug(f"[MEAN_REVERSION] {epic}: insufficient rows for low-vol filter")
            return False

        # ATR14 check
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr_price = tr.rolling(14).mean().iloc[-1]
        if pd.isna(atr_price) or atr_price <= 0:
            self.logger.debug(f"[MEAN_REVERSION] {epic}: ATR unavailable for low-vol filter")
            return False
        atr_pips = atr_price / pip
        if atr_pips > atr_max:
            self.logger.debug(
                f"[MEAN_REVERSION] {epic}: low-vol ATR block ({atr_pips:.2f} > {atr_max})"
            )
            return False

        # EMA slope check
        ema = close.ewm(span=ema_period, adjust=False).mean()
        ema_change_pips = abs(float(ema.iloc[-1]) - float(ema.iloc[-1 - lookback])) / pip
        if ema_change_pips >= ema_max_change:
            self.logger.debug(
                f"[MEAN_REVERSION] {epic}: low-vol EMA block (change={ema_change_pips:.2f} >= {ema_max_change})"
            )
            return False

        return True

    def _get_adx(self, df: pd.DataFrame) -> Optional[float]:
        return get_adx_from_df(df, self.config.adx_period)

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
