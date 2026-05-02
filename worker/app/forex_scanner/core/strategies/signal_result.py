"""
Typed contract for strategy signal dictionaries.

All strategies must return Optional[SignalResult] from detect_signal().
SignalResult IS a plain dict at runtime (TypedDict has no overhead), so
existing .get() callers need no changes.

Required fields are in _SignalResultRequired. Optional-but-load-bearing fields
(entry_type, market_regime, market_regime_detected) are in SignalResult with
total=False so type-checkers flag missing required fields without breaking
strategies that legitimately omit optional ones.
"""

from datetime import datetime
from typing import Any, Dict, Optional, TypedDict


class _SignalResultRequired(TypedDict):
    """Fields that every strategy MUST populate."""
    signal_type: str        # 'bull' | 'bear' | 'buy' | 'sell'
    strategy: str           # e.g. 'SMC_SIMPLE'
    epic: str               # IG epic identifier
    entry_price: float
    risk_pips: float        # stop-loss distance in pips — required by BacktestScanner
    reward_pips: float      # take-profit distance in pips — required by BacktestScanner
    confidence_score: float # 0.0–1.0


class SignalResult(_SignalResultRequired, total=False):
    """
    Full signal contract.

    Required fields (from _SignalResultRequired): signal_type, strategy, epic,
    entry_price, risk_pips, reward_pips, confidence_score.

    All fields below are optional — strategies should populate what they can;
    signal_detector and LPF will fall back gracefully on missing optional fields.
    """

    # Direction alias (some consumers check 'signal' key)
    signal: str             # 'BUY' | 'SELL' — same as signal_type but uppercased

    # Pair metadata
    pair: str               # human-readable pair name, e.g. 'EURUSD'

    # Regime — load-bearing for LPF and backtest regime filtering
    market_regime: str                  # regime reported by strategy
    market_regime_detected: str         # regime backfilled by signal_detector routing

    # Entry classification — load-bearing for LPF direction gates
    entry_type: str         # 'PULLBACK' | 'MOMENTUM' | 'REVERSAL' | 'PATTERN' | 'DIVERGENCE'

    # Raw SL/TP prices (absolute levels, not distances)
    stop_loss: float
    take_profit: float

    # Confidence alias
    confidence: float       # same value as confidence_score

    # Timing
    signal_timestamp: str   # ISO-format string
    timestamp: datetime

    # Technical indicators (used by LPF rules)
    adx: float
    adx_htf: float
    rsi: float
    stoch_k: float
    atr_percentile: float
    efficiency_ratio: float
    ema_distance_pips: float
    volume_trend: str
    all_timeframes_aligned: Optional[bool]
    market_session: str
    market_bias: str
    htf_bias: str

    # Strategy metadata
    version: str
    strategy_indicators: Dict[str, Any]

    # Claude AI enrichment (added post-signal by IntegrationManager)
    market_intelligence: Dict[str, Any]

    # LPF metadata (written by LPF, read by downstream)
    lpf_penalty: float
    lpf_would_block: bool
