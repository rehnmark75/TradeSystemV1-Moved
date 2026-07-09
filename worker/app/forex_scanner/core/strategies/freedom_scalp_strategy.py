#!/usr/bin/env python3
"""
FREEDOMSCALP Strategy - Gaussian trend-flip scalp on gold (TradingView port)

VERSION: 1.0.0
DATE: 2026-07-09
STATUS: Demo-only forward test (real demo execution, live disabled)

Faithful port of the user's TradingView Pine v6 strategy
"FreedomScalp V3 - Gaussian (Price Units)":
  - N-pole recursive Gaussian filter on 5m close
  - linreg(smoothing_length, flatten_offset) of the Gaussian output
  - SuperTrend(factor 0.15, ATR 21) computed ON the smoothed line
  - entry when the smoothed line crosses its SuperTrend (trend flip)
  - filters: EMA50 regime gate, spike bar filter, session window,
    optional ADX / ATR-expansion regime gates (off by default, TV parity)

⚠️ VALIDATION STATUS (freedomscalp_lab.py, Jul 9 2026): REFUTED on 6y of 5m
gold under pessimistic fills (PF < 1 every year 2020-2024; 2026 edge exists
only under TradingView's optimistic intrabar trailing fill model — PF 1.59 TV
fills vs 1.05 with 1m exit walk). Wired anyway as a DEMO forward test to
arbitrate the fill-model dispute with real broker fills. Do not promote to
live without demo PF > 1.2 over 100+ trades. See
memory/project_freedomscalp_gaussian_gold_jul9.md.

Gold pip convention: 1 pip = $0.10 (TP $4.20 = 42 pips, SL $6.00 = 60 pips).
"""

import math
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

from .strategy_registry import register_strategy, StrategyInterface


_SC_ENGINE = None


def _strategy_config_engine():
    """Engine for the strategy_config database (house config-service pattern:
    config tables live in strategy_config, NOT the forex DB the scanner's
    db_manager points at)."""
    global _SC_ENGINE
    if _SC_ENGINE is None:
        from sqlalchemy import create_engine
        url = os.getenv(
            'STRATEGY_CONFIG_DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/strategy_config')
        _SC_ENGINE = create_engine(url, pool_pre_ping=True)
    return _SC_ENGINE


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class FreedomScalpConfig:
    """Loaded from strategy_config.freedomscalp_global_config."""

    strategy_name: str = "FREEDOMSCALP"
    version: str = "1.0.0"

    # Timeframes (5m single-timeframe strategy)
    entry_timeframe: str = "5m"

    # Gaussian core (TV defaults)
    gaussian_length: int = 8
    gaussian_poles: int = 2
    smoothing_length: int = 10
    flatten_offset: int = 3
    supertrend_factor: float = 0.15
    supertrend_atr_period: int = 21

    # Filters (TV defaults)
    ema_filter_enabled: bool = True
    ema_period: int = 50
    spike_filter_enabled: bool = True
    spike_atr_mult: float = 3.0
    adx_gate_enabled: bool = False
    adx_threshold: float = 20.0
    atr_expansion_enabled: bool = False
    atr_expansion_mult: float = 1.0

    # Session window in UTC (Pine default 0600-2030 UTC+2)
    session_enabled: bool = True
    session_start_utc: int = 400     # HHMM
    session_end_utc: int = 1830      # HHMM

    # Risk (gold pips: 1 pip = $0.10)
    fixed_stop_loss_pips: float = 60.0
    fixed_take_profit_pips: float = 42.0

    # Confidence
    min_confidence: float = 0.60
    base_confidence: float = 0.62

    # Cooldown
    signal_cooldown_minutes: int = 30

    @classmethod
    def from_database(cls, db_manager=None) -> 'FreedomScalpConfig':
        config = cls()
        try:
            rows = pd.read_sql(
                """SELECT parameter_name, parameter_value, value_type
                   FROM freedomscalp_global_config WHERE is_active = TRUE""",
                _strategy_config_engine())
            casts = {'int': int, 'float': float,
                     'bool': lambda v: str(v).lower() in ('true', '1', 'yes')}
            for _, row in rows.iterrows():
                name = row['parameter_name']
                if not hasattr(config, name):
                    continue
                cast = casts.get(row['value_type'], str)
                try:
                    setattr(config, name, cast(row['parameter_value']))
                except (ValueError, TypeError):
                    logging.getLogger(__name__).warning(
                        "FREEDOMSCALP config cast failed for %s=%r",
                        name, row['parameter_value'])
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Could not load FREEDOMSCALP config from database: {e}")
        return config


class FreedomScalpConfigService:
    """Singleton config service with 5-minute cache (house pattern)."""

    _instance: Optional['FreedomScalpConfigService'] = None
    _config: Optional[FreedomScalpConfig] = None
    _pair_rows: Optional[pd.DataFrame] = None
    _last_refresh: Optional[datetime] = None
    _cache_ttl_seconds: int = 300

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._db_manager = None
        self.logger = logging.getLogger(__name__)

    @classmethod
    def get_instance(cls) -> 'FreedomScalpConfigService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_db_manager(self, db_manager) -> None:
        self._db_manager = db_manager
        self._config = None

    def _refresh_if_stale(self) -> None:
        now = datetime.now()
        if (self._config is None or self._last_refresh is None
                or (now - self._last_refresh).seconds > self._cache_ttl_seconds):
            self._config = FreedomScalpConfig.from_database(self._db_manager)
            self._pair_rows = None
            try:
                self._pair_rows = pd.read_sql(
                    """SELECT epic, pair_name, is_enabled, is_traded,
                              fixed_stop_loss_pips, fixed_take_profit_pips,
                              min_confidence, signal_cooldown_minutes,
                              COALESCE(parameter_overrides->>'monitor_only', 'false') = 'true'
                                  AS monitor_only
                       FROM freedomscalp_pair_overrides""",
                    _strategy_config_engine())
            except Exception as e:
                self.logger.warning(f"FREEDOMSCALP pair overrides load failed: {e}")
            self._last_refresh = now

    def get_config(self) -> FreedomScalpConfig:
        self._refresh_if_stale()
        return self._config

    def get_pair_override(self, epic: str) -> Optional[dict]:
        self._refresh_if_stale()
        if self._pair_rows is None or self._pair_rows.empty:
            return None
        m = self._pair_rows[self._pair_rows['epic'] == epic]
        return None if m.empty else m.iloc[0].to_dict()

    def is_pair_enabled(self, epic: str) -> bool:
        row = self.get_pair_override(epic)
        return bool(row and row.get('is_enabled'))


def get_freedomscalp_config() -> FreedomScalpConfig:
    return FreedomScalpConfigService.get_instance().get_config()


# ==============================================================================
# INDICATOR PORTS (validated against freedomscalp_lab.py, which reproduces
# the TradingView strategy tester output on identical data)
# ==============================================================================

def _gaussian_alpha(length: int, order: int) -> float:
    freq = 2.0 * math.pi / length
    b = (1.0 - math.cos(freq)) / (1.414 ** (2.0 / order) - 1.0)
    return -b + math.sqrt(b * b + 2.0 * b)


def _gaussian_smooth(src: np.ndarray, poles: int, a: float) -> np.ndarray:
    n = len(src)
    v = np.zeros(n)
    oma = 1.0 - a
    coeffs = {1: [1.0], 2: [2.0, -1.0], 3: [3.0, -3.0, 1.0],
              4: [4.0, -6.0, 4.0, -1.0]}[max(1, min(4, poles))]
    gain = a ** len(coeffs)
    for i in range(n):
        acc = gain * src[i]
        for k, cf in enumerate(coeffs, start=1):
            if i >= k:
                acc += cf * (oma ** k) * v[i - k]
        v[i] = acc
    return v


def _linreg(src: np.ndarray, length: int, offset: int) -> np.ndarray:
    """Pine ta.linreg: LSRL over window evaluated at x = length-1-offset."""
    n = len(src)
    x = np.arange(length, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    xe = (length - 1 - offset) - xm
    coef = (1.0 / length) + (x - xm) * xe / denom
    out = np.full(n, np.nan)
    if n >= length:
        out[length - 1:] = np.convolve(src, coef[::-1], mode='valid')
    return out


def _rma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    a = np.nan_to_num(arr, nan=0.0)
    if len(arr) < period:
        return out
    out[period - 1] = a[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(arr)):
        out[i] = out[i - 1] + alpha * (a[i] - out[i - 1])
    return out


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return _rma(tr, period)


def _pine_supertrend(src: np.ndarray, atr_arr: np.ndarray, factor: float) -> np.ndarray:
    """The Pine script's custom supertrend computed on the smoothed line."""
    n = len(src)
    st = np.full(n, np.nan)
    up_f = np.zeros(n); dn_f = np.zeros(n)
    for i in range(n):
        av = atr_arr[i] if not np.isnan(atr_arr[i]) else 0.0
        up = src[i] + factor * av
        dn = src[i] - factor * av
        pdn = dn_f[i - 1] if i >= 1 else 0.0
        pup = up_f[i - 1] if i >= 1 else 0.0
        psrc = src[i - 1] if i >= 1 else src[i]
        dn = dn if (dn > pdn or psrc < pdn) else pdn
        up = up if (up < pup or psrc > pup) else pup
        dn_f[i] = dn; up_f[i] = up
        pst = st[i - 1] if i >= 1 else np.nan
        if i == 0 or np.isnan(atr_arr[i - 1]):
            d = 1
        elif pst == pup:
            d = -1 if src[i] > up else 1
        else:
            d = 1 if src[i] < dn else -1
        st[i] = dn if d == -1 else up
    return st


# ==============================================================================
# STRATEGY
# ==============================================================================

GOLD_PIP = 0.1  # 1 gold pip = $0.10 (system convention, matches XAU_GOLD)

MIN_BARS = 250  # EMA50 + gaussian warmup + supertrend ATR21


@register_strategy('FREEDOMSCALP')
class FreedomScalpStrategy(StrategyInterface):
    """Gaussian trend-flip scalp (TradingView FreedomScalp V3 port)."""

    def __init__(self, config=None, logger=None, db_manager=None, config_override=None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager

        if db_manager:
            FreedomScalpConfigService.get_instance().set_db_manager(db_manager)
        self.config = get_freedomscalp_config()

        # bt.py --override contract (feedback_strategy_override_wiring)
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        self._cooldowns: Dict[str, datetime] = {}
        self._pending_rejections: List[Dict] = []
        self.logger.info(f"FREEDOMSCALP Strategy v{self.config.version} initialized")

    @property
    def strategy_name(self) -> str:
        return "FREEDOMSCALP"

    def get_required_timeframes(self) -> List[str]:
        return [self.config.entry_timeframe]

    # -------------------------------------------------------------- detection
    def detect_signal(
        self,
        df_trigger: pd.DataFrame = None,
        epic: str = "",
        pair: str = "",
        **kwargs
    ) -> Optional[Dict]:
        cfg = self.config

        if df_trigger is None or len(df_trigger) < MIN_BARS:
            self.logger.debug(f"[FREEDOMSCALP] Insufficient 5m data for {epic}: "
                              f"{0 if df_trigger is None else len(df_trigger)} bars")
            return None

        svc = FreedomScalpConfigService.get_instance()
        if self.db_manager is not None and not svc.is_pair_enabled(epic):
            return None

        df = df_trigger
        bar_ts = self._bar_timestamp(df)

        # Cooldown keyed to BAR time, not wall time — wall-clock cooldowns
        # never expire inside a backtest that replays days in seconds.
        if not self._check_cooldown(epic, bar_ts):
            return None
        c = df['close'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)

        # --- core: gaussian -> linreg -> supertrend on smoothed line
        alpha = _gaussian_alpha(cfg.gaussian_length, cfg.gaussian_poles)
        g = _gaussian_smooth(c, cfg.gaussian_poles, alpha)
        final = _linreg(g, cfg.smoothing_length, cfg.flatten_offset)
        fin = np.nan_to_num(final, nan=c[0])
        atr21 = _atr(h, l, c, cfg.supertrend_atr_period)
        st = _pine_supertrend(fin, atr21, cfg.supertrend_factor)
        trend = np.where(fin > st, 1, -1)

        # signal = trend flip on the latest closed bar
        if len(trend) < 2 or trend[-1] == trend[-2]:
            return None
        direction = 'BUY' if trend[-1] > 0 else 'SELL'

        price = float(c[-1])

        # --- filters (TV parity)
        if cfg.session_enabled and bar_ts is not None:
            hhmm = bar_ts.hour * 100 + bar_ts.minute
            if not (cfg.session_start_utc <= hhmm < cfg.session_end_utc):
                self._log_rejection(epic, 'out_of_session', hhmm)
                return None

        atr14 = _atr(h, l, c, 14)
        if cfg.spike_filter_enabled and len(atr14) >= 2 and not np.isnan(atr14[-2]):
            if (h[-1] - l[-1]) > atr14[-2] * cfg.spike_atr_mult:
                self._log_rejection(epic, 'spike_bar', float(h[-1] - l[-1]))
                return None

        ema = pd.Series(c).ewm(span=cfg.ema_period, adjust=False).mean().values
        if cfg.ema_filter_enabled:
            if direction == 'BUY' and price <= ema[-1]:
                self._log_rejection(epic, 'below_ema', float(ema[-1]))
                return None
            if direction == 'SELL' and price >= ema[-1]:
                self._log_rejection(epic, 'above_ema', float(ema[-1]))
                return None

        adx_val = self._adx(h, l, c)
        if cfg.adx_gate_enabled and adx_val is not None and adx_val <= cfg.adx_threshold:
            self._log_rejection(epic, 'adx_below_threshold', adx_val)
            return None

        atr_expanding = None
        if len(atr14) >= 64:
            atr_slow = pd.Series(atr14).rolling(50).mean().values
            if not np.isnan(atr_slow[-1]):
                atr_expanding = bool(atr14[-1] > atr_slow[-1] * cfg.atr_expansion_mult)
        if cfg.atr_expansion_enabled and atr_expanding is False:
            self._log_rejection(epic, 'atr_contracting', float(atr14[-1]))
            return None

        # --- SL/TP from per-pair override or global (gold pips, $0.10)
        row = svc.get_pair_override(epic) or {}
        sl_pips = float(row.get('fixed_stop_loss_pips') or cfg.fixed_stop_loss_pips)
        tp_pips = float(row.get('fixed_take_profit_pips') or cfg.fixed_take_profit_pips)

        # --- confidence: flat base + regime bumps, floor-gated
        confidence = cfg.base_confidence
        if adx_val is not None and adx_val > 20:
            confidence += 0.05
        if atr_expanding:
            confidence += 0.05
        confidence = min(confidence, 0.90)
        min_conf = float(row.get('min_confidence') or cfg.min_confidence)
        if confidence < min_conf:
            self._log_rejection(epic, 'low_confidence', confidence)
            return None

        signal = self._build_signal(
            epic=epic,
            pair=pair or 'XAUUSD',
            direction=direction,
            entry_price=price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            confidence=confidence,
            monitor_only=bool(row.get('monitor_only', False)),
            df_trigger=df,
            indicators={
                'gaussian_final': float(fin[-1]),
                'supertrend': float(st[-1]),
                'ema_filter': float(ema[-1]),
                'atr14_pips': float(atr14[-1] / GOLD_PIP) if not np.isnan(atr14[-1]) else None,
                'adx': adx_val,
                'atr_expanding': atr_expanding,
            },
        )
        self._set_cooldown(epic, bar_ts)
        return signal

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _bar_timestamp(df: pd.DataFrame) -> Optional[datetime]:
        """Latest bar timestamp; start_time is a column, not the index
        (feedback_simulation_time)."""
        if 'start_time' in df.columns:
            ts = pd.Timestamp(df['start_time'].iloc[-1])
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[-1]
        else:
            return None
        return ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts

    @staticmethod
    def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> Optional[float]:
        if len(c) < period * 3:
            return None
        up = h - np.roll(h, 1); up[0] = 0
        dn = np.roll(l, 1) - l; dn[0] = 0
        pdm = np.where((up > dn) & (up > 0), up, 0.0)
        mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
        pc = np.roll(c, 1); pc[0] = c[0]
        tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
        atr_di = _rma(tr, period)
        with np.errstate(invalid='ignore', divide='ignore'):
            pdi = 100 * _rma(pdm, period) / atr_di
            mdi = 100 * _rma(mdm, period) / atr_di
            dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
        adx = _rma(np.nan_to_num(dx), period)
        return float(adx[-1]) if not np.isnan(adx[-1]) else None

    def _build_signal(
        self,
        epic: str,
        pair: str,
        direction: str,
        entry_price: float,
        sl_pips: float,
        tp_pips: float,
        confidence: float,
        **kwargs
    ) -> Dict:
        now = datetime.now(timezone.utc)
        signal = {
            'signal':      direction,
            'signal_type': direction,
            'direction':   direction,
            'strategy': self.strategy_name,
            'epic': epic,
            'pair': pair,
            'entry_price': entry_price,
            'risk_pips':   sl_pips,
            'reward_pips': tp_pips,
            'entry_type': 'MOMENTUM',
            'confidence_score': confidence,
            'confidence': confidence,
            'signal_timestamp': now.isoformat(),
            'timestamp': now,
            'version': self.config.version,
            'monitor_only': bool(kwargs.get('monitor_only', False)),
            'strategy_indicators': kwargs.get('indicators', {}),
        }
        try:
            from .helpers.smc_performance_metrics import enrich_signal_with_performance_metrics
            signal = enrich_signal_with_performance_metrics(
                signal,
                df_entry=kwargs.get('df_trigger'),
                df_trigger=kwargs.get('df_trigger'),
                df_htf=None,
                epic=epic,
                logger=self.logger,
            )
        except Exception as _pm_exc:
            self.logger.warning("Performance metrics enrichment failed: %s", _pm_exc)
        return signal

    # -------------------------------------------------------------- cooldowns
    @staticmethod
    def _naive_utc(ts: Optional[datetime]) -> datetime:
        if ts is None:
            return datetime.utcnow()
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts

    def _check_cooldown(self, epic: str, bar_ts: Optional[datetime]) -> bool:
        if epic not in self._cooldowns:
            return True
        if self._naive_utc(bar_ts) >= self._cooldowns[epic]:
            del self._cooldowns[epic]
            return True
        return False

    def _set_cooldown(self, epic: str, bar_ts: Optional[datetime]) -> None:
        row = FreedomScalpConfigService.get_instance().get_pair_override(epic) or {}
        minutes = int(row.get('signal_cooldown_minutes') or self.config.signal_cooldown_minutes)
        self._cooldowns[epic] = self._naive_utc(bar_ts) + timedelta(minutes=minutes)

    def reset_cooldowns(self) -> None:
        self._cooldowns.clear()

    # -------------------------------------------------------------- rejections
    def _log_rejection(self, epic: str, reason: str, value: Any = None) -> None:
        self.logger.debug(f"[FREEDOMSCALP] reject {epic}: {reason} = {value}")
        self._pending_rejections.append({
            'epic': epic,
            'strategy': self.strategy_name,
            'reason': reason,
            'value': value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    def flush_rejections(self) -> None:
        self._pending_rejections.clear()


def create_freedomscalp_strategy(
    config=None,
    db_manager=None,
    logger=None,
    config_override=None,
) -> FreedomScalpStrategy:
    """Factory (bt.py --override contract: forward config_override)."""
    return FreedomScalpStrategy(
        config=config,
        logger=logger,
        db_manager=db_manager,
        config_override=config_override,
    )
