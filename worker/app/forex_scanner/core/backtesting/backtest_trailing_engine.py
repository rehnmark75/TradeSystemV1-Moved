"""
BacktestTrailingEngine — drop-in replacement for TrailingStopSimulator.

Uses the SAME config dicts as live trading (PAIR_TRAILING_CONFIGS /
SCALP_TRAILING_CONFIGS from dev-app/config.py, mounted into the worker
container as trailing_config_live.py).

Key improvements over the old TrailingStopSimulator:
  - Config source is identical to live trading — no drift
  - Early break-even stage is implemented (was missing — the #1 profit lever)
  - Scalp mode uses full progressive trailing, not fixed SL/TP
  - Spread + slippage applied at entry price
  - config_override flows into trailing params (--override works end-to-end)
  - Bar evaluation is worst-case-first: SL checked before stage transitions
"""

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Pip-multiplier helper (reuse from existing simulator)
# ---------------------------------------------------------------------------

try:
    from core.trading.trailing_stop_simulator import _resolve_pip_multiplier
except ImportError:
    try:
        from forex_scanner.core.trading.trailing_stop_simulator import _resolve_pip_multiplier
    except ImportError:
        def _resolve_pip_multiplier(epic: Optional[str], logger=None) -> float:  # type: ignore[misc]
            if epic and 'JPY' in epic.upper():
                return 100.0
            return 10000.0


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_TRAILING_OVERRIDE_KEYS = frozenset({
    'early_breakeven_trigger_points', 'early_breakeven_buffer_points',
    'stage1_trigger_points', 'stage1_lock_points',
    'stage2_trigger_points', 'stage2_lock_points',
    'stage3_trigger_points', 'stage3_atr_multiplier', 'stage3_min_distance',
    'min_trail_distance', 'break_even_trigger_points',
    'spread_pips', 'slippage_pips',
})

# Feature flag: set BACKTEST_USE_LEGACY_SIMULATOR=1 to fall back to old simulator
# for A/B comparison runs. Remove after validation is complete.
BACKTEST_USE_LEGACY_SIMULATOR = os.getenv('BACKTEST_USE_LEGACY_SIMULATOR', '0') == '1'


def _load_pair_config(epic: str, is_scalp: bool, config_override: Dict[str, Any]) -> dict:
    """Load per-pair trailing config from live source and apply --override values."""
    try:
        try:
            from config_trailing_live import PAIR_TRAILING_CONFIGS, SCALP_TRAILING_CONFIGS
        except ImportError:
            from forex_scanner.config_trailing_live import PAIR_TRAILING_CONFIGS, SCALP_TRAILING_CONFIGS

        source = SCALP_TRAILING_CONFIGS if is_scalp else PAIR_TRAILING_CONFIGS
        fallback_key = 'CS.D.EURUSD.CEEM.IP'
        cfg = dict(source.get(epic) or source.get(fallback_key) or {})

        if not cfg:
            logging.getLogger(__name__).warning(
                f"No trailing config found for {epic!r} (is_scalp={is_scalp}), "
                f"and fallback key {fallback_key!r} also missing. "
                f"Using hardcoded defaults — check docker volume mount."
            )
    except Exception as exc:
        logging.getLogger(__name__).warning(f"Could not load live trailing config: {exc}")
        cfg = {}

    unknown = {k for k in config_override if k in _TRAILING_OVERRIDE_KEYS}
    for key in unknown:
        existing = cfg.get(key)
        try:
            cfg[key] = type(existing)(config_override[key]) if existing is not None else config_override[key]
        except (TypeError, ValueError):
            cfg[key] = config_override[key]

    return cfg


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

def _calc_atr(df: pd.DataFrame, up_to_bar: int, n: int = 15) -> Optional[float]:
    """Average True Range of the *n* bars ending before *up_to_bar*."""
    start = max(0, up_to_bar - n)
    window = df.iloc[start:up_to_bar]
    if len(window) < 2:
        return None
    trs = []
    for i in range(1, len(window)):
        prev_close = window.iloc[i - 1]['close']
        row = window.iloc[i]
        tr = max(
            row['high'] - row['low'],
            abs(row['high'] - prev_close),
            abs(row['low'] - prev_close),
        )
        trs.append(tr)
    return sum(trs) / len(trs) if trs else None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestTrailingEngine:
    """
    Simulates trade exit logic using the same trailing config as live trading.

    Drop-in replacement for TrailingStopSimulator; identical call signature:
        result_dict = engine.simulate_trade(signal, df, signal_idx)

    The returned dict contains the same keys as the old simulator so all
    downstream backtest_scanner.py code works without modification.
    """

    def __init__(
        self,
        epic: Optional[str] = None,
        is_scalp_trade: bool = False,
        spread_pips: float = 1.5,
        slippage_pips: float = 0.5,
        config_override: Optional[Dict[str, Any]] = None,
        max_bars: int = 200,
        strategy: str = 'DEFAULT',
        logger: Optional[logging.Logger] = None,
        # Accept but ignore old-simulator kwargs for drop-in compatibility
        **_kwargs,
    ):
        self.epic = epic
        self.is_scalp_trade = is_scalp_trade
        self.max_bars = max_bars
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        self._config_override = config_override or {}

        # spread/slippage can also come from config_override
        self.spread_pips = float(self._config_override.get('spread_pips', spread_pips))
        self.slippage_pips = float(self._config_override.get('slippage_pips', slippage_pips))

        self._pair_cfg = _load_pair_config(epic or '', is_scalp_trade, self._config_override)
        self.pip_multiplier = _resolve_pip_multiplier(epic, self.logger)

        # Expose for compatibility with callers that read these directly
        self.initial_stop_pips = float(self._pair_cfg.get('min_trail_distance', 12))
        self.target_pips = float(self._pair_cfg.get('stage1_trigger_points', 14))

        source_label = 'SCALP' if is_scalp_trade else 'STANDARD'
        self.logger.debug(
            f"[BacktestTrailingEngine] {epic} ({source_label}) — "
            f"earlyBE={self._pair_cfg.get('early_breakeven_trigger_points')}pts "
            f"BE={self._pair_cfg.get('break_even_trigger_points')}pts "
            f"S1={self._pair_cfg.get('stage1_trigger_points')}→{self._pair_cfg.get('stage1_lock_points')} "
            f"S2={self._pair_cfg.get('stage2_trigger_points')}→{self._pair_cfg.get('stage2_lock_points')} "
            f"S3={self._pair_cfg.get('stage3_trigger_points')}pts "
            f"spread={self.spread_pips}pip slippage={self.slippage_pips}pip"
        )

    # ------------------------------------------------------------------
    # Public API  (same interface as TrailingStopSimulator.simulate_trade)
    # ------------------------------------------------------------------

    def simulate_trade(
        self,
        signal: Dict[str, Any],
        df: pd.DataFrame,
        signal_idx: int,
    ) -> Dict[str, Any]:
        """Simulate trade and return enriched signal dict matching old simulator shape."""
        enhanced = signal.copy()
        try:
            entry_price = (
                signal.get('entry_price')
                or signal.get('current_price')
                or signal.get('price')
                or df.iloc[signal_idx]['close']
            )
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()
            is_long = signal_type in ('BUY', 'BULL', 'LONG')
            signal_timestamp = signal.get('signal_timestamp') or signal.get('timestamp')

            max_lookback = min(self.max_bars, len(df) - signal_idx - 1)
            if max_lookback <= 0:
                enhanced.update(_no_data_result())
                return enhanced

            future_df = df.iloc[signal_idx + 1: signal_idx + 1 + max_lookback]

            sim = self._run_simulation(float(entry_price), is_long, future_df, signal)

            # Exit price and timestamps
            if sim['exit_bar'] is not None and sim['exit_bar'] < len(future_df):
                exit_price = future_df.iloc[sim['exit_bar']]['close']
                holding_minutes = (sim['exit_bar'] + 1) * 15
                exit_timestamp = None
                if signal_timestamp:
                    try:
                        from datetime import timedelta
                        ts = (pd.to_datetime(signal_timestamp)
                              if isinstance(signal_timestamp, str) else signal_timestamp)
                        exit_timestamp = ts + timedelta(minutes=holding_minutes)
                    except Exception:
                        pass
            else:
                exit_price = future_df.iloc[-1]['close'] if len(future_df) > 0 else entry_price
                holding_minutes = max_lookback * 15
                exit_timestamp = None

            final_profit = sim['final_profit']
            final_loss = sim['final_loss']
            rr = (round(final_profit / final_loss, 2) if final_loss > 0
                  else (float('inf') if final_profit > 0 else 0))

            enhanced.update({
                # DB-compatible result
                'trade_result': _map_db_result(sim['trade_outcome']),
                'trade_outcome': sim['trade_outcome'],
                'exit_reason': sim['exit_reason'],
                'exit_price': exit_price,
                'exit_timestamp': exit_timestamp,
                'pips_gained': sim['exit_pnl'],

                # P&L display
                'profit': round(final_profit, 1),
                'loss': round(final_loss, 1),
                'risk_reward': rr,

                # Performance
                'holding_time_minutes': holding_minutes,
                'max_favorable_excursion_pips': sim['best_profit_pips'],
                'max_adverse_excursion_pips': sim['worst_loss_pips'],

                # Classification
                'is_winner': sim['is_winner'],
                'is_loser': sim['is_loser'],

                # Legacy compatibility fields
                'max_profit_pips': round(final_profit, 1),
                'max_loss_pips': round(final_loss, 1),
                'profit_loss_ratio': rr,
                'entry_price': entry_price,
                'exit_pnl': sim['exit_pnl'],
                'exit_bar': sim['exit_bar'],

                # Stage tracking
                'trailing_stop_used': sim['stage_reached'] > 0,
                'stop_moved_to_breakeven': sim['be_triggered'],
                'stage1_triggered': sim['s1_triggered'],
                'stage2_triggered': sim['s2_triggered'],
                'stage3_triggered': sim['s3_triggered'],
                'stage_reached': sim['stage_reached'],
                'best_profit_achieved': round(sim['best_profit_pips'], 1),

                # Config info
                'target_pips': float(signal.get('reward_pips') or self.target_pips),
                'initial_stop_pips': float(signal.get('risk_pips') or self.initial_stop_pips),
                'lookback_bars': max_lookback,

                # Transparency fields (new — not in old simulator)
                'spread_applied_pips': self.spread_pips,
                'trailing_config_source': 'live_scalp' if self.is_scalp_trade else 'live_standard',
            })

        except Exception as exc:
            self.logger.error(f"BacktestTrailingEngine.simulate_trade: {exc}", exc_info=True)
            enhanced.update({
                'trade_result': 'SIMULATION_ERROR',
                'exit_reason': str(exc),
                'is_winner': False,
                'is_loser': False,
            })

        return enhanced

    # ------------------------------------------------------------------
    # Core simulation loop
    # ------------------------------------------------------------------

    def _run_simulation(
        self,
        entry_price: float,
        is_long: bool,
        future_df: pd.DataFrame,
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg = self._pair_cfg
        point_value = 1.0 / self.pip_multiplier

        # Apply spread + slippage at fill (widens effective entry against the trader)
        entry_cost = (self.spread_pips + self.slippage_pips) * point_value
        fill_price = entry_price + entry_cost if is_long else entry_price - entry_cost

        # Initial SL/TP from signal if provided, otherwise from config defaults
        risk_pips = float(signal.get('risk_pips') or 0)
        reward_pips = float(signal.get('reward_pips') or 0)
        initial_sl_pips = risk_pips if risk_pips > 0 else float(cfg.get('min_trail_distance', 12))
        target_pips = reward_pips if reward_pips > 0 else float(cfg.get('stage1_trigger_points', 14))

        if is_long:
            current_sl = fill_price - initial_sl_pips * point_value
            current_tp = fill_price + target_pips * point_value
        else:
            current_sl = fill_price + initial_sl_pips * point_value
            current_tp = fill_price - target_pips * point_value

        # Stage trigger/lock values from live config
        early_be_trig = float(cfg.get('early_breakeven_trigger_points', 10))
        early_be_buf  = float(cfg.get('early_breakeven_buffer_points', 3))
        be_trig       = float(cfg.get('break_even_trigger_points', 12))
        s1_trig       = float(cfg.get('stage1_trigger_points', 14))
        s1_lock       = float(cfg.get('stage1_lock_points', 8))
        s2_trig       = float(cfg.get('stage2_trigger_points', 25))
        s2_lock       = float(cfg.get('stage2_lock_points', 15))
        s3_trig       = float(cfg.get('stage3_trigger_points', 35))
        s3_atr_mult   = float(cfg.get('stage3_atr_multiplier', 2.0))

        # State
        early_be_done = be_done = s1_done = s2_done = s3_active = False
        stage_reached = 0
        best_profit = worst_loss = 0.0

        trade_closed = False
        exit_pnl: float = 0.0
        exit_bar: Optional[int] = None
        exit_reason = 'TIMEOUT'

        for bar_idx, (_, bar) in enumerate(future_df.iterrows()):
            high, low = float(bar['high']), float(bar['low'])

            if is_long:
                # Worst-case: check if low hits SL BEFORE checking high triggers stage transitions.
                # Rationale: intra-bar order unknown; conservative assumption prevents over-optimism.
                if low <= current_sl:
                    exit_pnl = (current_sl - fill_price) * self.pip_multiplier
                    exit_reason = 'TRAILING_STOP' if (be_done or s1_done) else 'STOP_LOSS'
                    exit_bar = bar_idx
                    trade_closed = True
                    break

                profit_pts = (high - fill_price) * self.pip_multiplier
                loss_pts   = (fill_price - low) * self.pip_multiplier
                best_price_this_bar = high

            else:  # SELL
                if high >= current_sl:
                    exit_pnl = (fill_price - current_sl) * self.pip_multiplier
                    exit_reason = 'TRAILING_STOP' if (be_done or s1_done) else 'STOP_LOSS'
                    exit_bar = bar_idx
                    trade_closed = True
                    break

                profit_pts = (fill_price - low) * self.pip_multiplier
                loss_pts   = (high - fill_price) * self.pip_multiplier
                best_price_this_bar = low

            best_profit = max(best_profit, profit_pts)
            worst_loss  = max(worst_loss,  loss_pts)

            # Stage transitions — only move SL forward (never widen)
            if not early_be_done and profit_pts >= early_be_trig:
                new_sl = (fill_price + early_be_buf * point_value if is_long
                          else fill_price - early_be_buf * point_value)
                current_sl = _advance_sl(current_sl, new_sl, is_long)
                early_be_done = True
                stage_reached = max(stage_reached, 1)

            if not be_done and profit_pts >= be_trig:
                current_sl = _advance_sl(current_sl, fill_price, is_long)
                be_done = True
                stage_reached = max(stage_reached, 1)

            if not s1_done and profit_pts >= s1_trig:
                new_sl = (fill_price + s1_lock * point_value if is_long
                          else fill_price - s1_lock * point_value)
                current_sl = _advance_sl(current_sl, new_sl, is_long)
                s1_done = True
                stage_reached = max(stage_reached, 2)

            if not s2_done and profit_pts >= s2_trig:
                new_sl = (fill_price + s2_lock * point_value if is_long
                          else fill_price - s2_lock * point_value)
                current_sl = _advance_sl(current_sl, new_sl, is_long)
                s2_done = True
                stage_reached = max(stage_reached, 3)

            if profit_pts >= s3_trig:
                s3_active = True
                stage_reached = max(stage_reached, 4)

            if s3_active:
                atr = _calc_atr(future_df, bar_idx)
                if atr is not None:
                    new_sl = (best_price_this_bar - s3_atr_mult * atr if is_long
                              else best_price_this_bar + s3_atr_mult * atr)
                    current_sl = _advance_sl(current_sl, new_sl, is_long)

            # TP hit check (after stage updates — gives the best realistic fill)
            if (is_long and high >= current_tp) or (not is_long and low <= current_tp):
                exit_pnl = (current_tp - fill_price if is_long else fill_price - current_tp) * self.pip_multiplier
                exit_reason = 'PROFIT_TARGET'
                exit_bar = bar_idx
                trade_closed = True
                break

        # Timeout: close at last bar close
        if not trade_closed:
            if len(future_df) > 0:
                final_price = float(future_df.iloc[-1]['close'])
                exit_pnl = (final_price - fill_price if is_long else fill_price - final_price) * self.pip_multiplier
            else:
                exit_pnl = 0.0

        final_profit = max(exit_pnl, 0.0)
        final_loss   = max(-exit_pnl, 0.0)

        return {
            'trade_outcome':  _classify_outcome(exit_reason, exit_pnl),
            'exit_reason':    exit_reason,
            'is_winner':      exit_pnl > 0,
            'is_loser':       exit_pnl < 0,
            'exit_pnl':       round(exit_pnl, 2),
            'exit_bar':       exit_bar,
            'final_profit':   round(final_profit, 2),
            'final_loss':     round(final_loss, 2),
            'best_profit_pips': round(best_profit, 1),
            'worst_loss_pips':  round(worst_loss, 1),
            'stage_reached':  stage_reached,
            'be_triggered':   be_done,
            'early_be_triggered': early_be_done,
            's1_triggered':   s1_done,
            's2_triggered':   s2_done,
            's3_triggered':   s3_active,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _advance_sl(current_sl: float, new_sl: float, is_long: bool) -> float:
    """Move SL only in the favourable direction — never widen it."""
    return max(current_sl, new_sl) if is_long else min(current_sl, new_sl)


def _classify_outcome(exit_reason: str, exit_pnl: float) -> str:
    if exit_reason == 'PROFIT_TARGET':
        return 'WIN'
    if exit_reason == 'TIMEOUT':
        if exit_pnl > 5.0:
            return 'WIN_TIMEOUT'
        if exit_pnl < -3.0:
            return 'LOSE_TIMEOUT'
        return 'TIMEOUT'
    return 'WIN' if exit_pnl > 0 else 'LOSE'


def _map_db_result(trade_outcome: str) -> str:
    if trade_outcome in ('WIN', 'WIN_TIMEOUT'):
        return 'win'
    if trade_outcome in ('LOSE', 'LOSE_TIMEOUT'):
        return 'loss'
    return 'breakeven'


def _no_data_result() -> dict:
    return {
        'trade_result': 'NO_DATA',
        'exit_reason': 'NO_FUTURE_DATA',
        'pips_gained': 0,
        'profit': 0.0,
        'loss': 0.0,
        'risk_reward': 0.0,
        'is_winner': False,
        'is_loser': False,
        'max_profit_pips': 0.0,
        'max_loss_pips': 0.0,
    }
