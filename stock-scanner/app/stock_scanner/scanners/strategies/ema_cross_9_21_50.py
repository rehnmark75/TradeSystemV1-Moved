"""
EMA 9/21/50 Cross Scanner
=========================

Live scanner wrapper around EmaCross92150Strategy. Same shared-strategy pattern
as EMAPullbackScanner: it fetches candle data and runs the strategy's `scan()`
so live signals share ONE code path with the walk-forward backtest in
`analysis/ema_cross_replay.py`.

Entry: close > EMA50, close > EMA21, and price crosses UP through EMA9.

STATUS: ACTIVE / live-traded (Jul 1 2026, user directive). In
DEFAULT_ENABLED_SCANNERS, registered in stock_signal_scanners, and NOT in
route.ts MONITOR_ONLY_SCANNERS -> its signals enter the live day-trade pool
(subject to the auto-trader gates: score>=65, EMA50 fresh-cross, VWAP, spread).
Runs the validated config together with the ATR-trail exit + breadth gate
(both ON). LIVE-ACCOUNT real money; honest edge ~1.7 PF (not 2.0). Revert to
monitor-only = remove from DEFAULT_ENABLED_SCANNERS or add to
MONITOR_ONLY_SCANNERS.

OPTIMIZATION (Jun-30 2026, multi-agent + ema_cross_lab.py walk-forward):
The raw fixed-bracket edge is thin (PF ~1.07). A validated config reaches
PF ~1.7 ALL / ~1.76 OOS at 3-5 trades/day (top-500 liquid). The levers:
  1. ENTRY  — full EMA stack 9>=21>=50 (now a hard gate here via require_stack)
              + market-breadth>=50% regime gate  [breadth gate = execution-layer]
  2. RANK   — score by TREND PERSISTENCE (EMA50 slope)-led conviction so the
              auto-trader's top-50 / score>=65 pool trades only the best ~5/day.
              That ranking now drives composite_score (see strategy._confidence).
  3. EXIT   — ATR-trailing (fixed ~1.5xATR stop until +2%, then trail ~3xATR);
              biggest single lever (+0.25 PF, flat 2-3.5x plateau). This lives in
              the auto-trader (currently fixed SL2.5/TP7) -> trailing-stop-engineer.
PF 2.0 sustained was NOT reachable without overfitting (universe/month sensitive).
Full detail: analysis/ema_cross_lab.py + memory project_ema_cross_9_21_50_*.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional

from ..base_scanner import (
    BaseScanner, SignalSetup, ScannerConfig,
    SignalType, QualityTier,
)
from ...strategies.ema_cross_9_21_50 import EmaCross92150Strategy, EmaCrossSignal
from ...core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)


@dataclass
class EmaCross92150Config(ScannerConfig):
    """Configuration for the EMA 9/21/50 cross scanner."""
    # Risk bracket — mirror the live auto-trader (SL 2.5%, TP 7%)
    stop_loss_pct: float = 2.5
    take_profit_pct: float = 7.0
    min_price: float = 3.0
    min_relative_volume: float = 0.0   # 0 = off

    max_signals_per_run: int = 50
    min_score_threshold: int = 0       # confidence does not gate entry here


class EmaCross92150Scanner(BaseScanner):
    """Scanner that emits EMA 9/21/50 cross signals using the shared strategy."""

    def __init__(self, db_manager, config: EmaCross92150Config = None, scorer=None):
        super().__init__(db_manager, config or EmaCross92150Config(), scorer)
        self.strategy = EmaCross92150Strategy(
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
            min_price=self.config.min_price,
            min_relative_volume=self.config.min_relative_volume,
        )
        self.data_provider = BacktestDataProvider(db_manager)

    @property
    def scanner_name(self) -> str:
        return "ema_cross_9_21_50"

    @property
    def description(self) -> str:
        return "Above EMA50 & EMA21, crosses up through EMA9 (shared strategy with backtest)"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            calculation_date = datetime.now().date()
        elif isinstance(calculation_date, datetime):
            calculation_date = calculation_date.date()

        tickers = await self.get_all_active_tickers()
        logger.info(f"Scanning {len(tickers)} active stocks for EMA 9/21/50 cross")

        signals: List[SignalSetup] = []
        scanned = 0
        for ticker in tickers:
            try:
                df = await self.data_provider.get_historical_data(
                    ticker=ticker,
                    start_date=calculation_date - timedelta(days=200),
                    end_date=calculation_date,
                    timeframe="1d",
                )
                if df.empty or len(df) < 60:
                    continue
                scanned += 1

                sig = self.strategy.scan(df, ticker, sector=None)
                if sig:
                    setup = self._convert(sig)
                    if setup:
                        signals.append(setup)
                        logger.info(
                            f"{ticker}: SIGNAL {sig.quality_tier} "
                            f"conf {sig.confidence:.0%} @ {sig.entry_price}"
                        )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{ticker}: error scanning - {e}")
                continue

        signals.sort(key=lambda s: s.composite_score, reverse=True)
        signals = signals[: self.config.max_signals_per_run]
        high_q = sum(1 for s in signals if s.is_high_quality)
        self.log_scan_summary(len(tickers), len(signals), high_q)
        return signals

    def _convert(self, sig: EmaCrossSignal) -> Optional[SignalSetup]:
        tier_map = {
            "A+": QualityTier.A_PLUS, "A": QualityTier.A, "B": QualityTier.B,
            "C": QualityTier.C, "D": QualityTier.D,
        }
        entry = sig.entry_price
        stop = sig.stop_loss_price
        risk_pct = abs(entry - stop) / entry * 100 if entry > 0 else 0

        factors = [
            "Above EMA50 (intermediate uptrend)",
            "Above EMA21 (short-term uptrend)",
            "Crossed up through EMA9 (entry trigger)",
        ]
        if sig.ema_9 >= sig.ema_21 >= sig.ema_50:
            factors.append("EMA stack aligned (9>21>50)")
        if sig.rsi is not None:
            factors.append(f"RSI {sig.rsi:.0f}")
        if sig.relative_volume:
            factors.append(f"RVOL {sig.relative_volume:.1f}x")

        desc = (
            f"{sig.ticker} reclaimed EMA9 while holding above EMA21/EMA50 "
            f"(uptrend momentum-pullback entry). "
            f"Quality {sig.quality_tier} ({sig.confidence:.0%})."
        )

        return SignalSetup(
            ticker=sig.ticker,
            scanner_name=self.scanner_name,
            signal_type=SignalType.BUY,
            signal_timestamp=sig.signal_timestamp,
            entry_price=Decimal(str(round(entry, 4))),
            stop_loss=Decimal(str(round(stop, 4))),
            take_profit_1=Decimal(str(round(sig.take_profit_price, 4))),
            take_profit_2=None,
            risk_reward_ratio=Decimal(str(round(sig.risk_reward_ratio, 2))),
            risk_percent=Decimal(str(round(risk_pct, 2))),
            composite_score=int(sig.confidence * 100),
            quality_tier=tier_map.get(sig.quality_tier, QualityTier.B),
            confluence_score=Decimal(str(round(sig.confidence * 25, 2))),
            setup_description=desc,
            confluence_factors=factors[:8],
            timeframe="daily",
            market_regime="Uptrend - EMA9 reclaim",
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data={
                "ema_9": sig.ema_9,
                "ema_21": sig.ema_21,
                "ema_50": sig.ema_50,
                "rsi": sig.rsi,
                "atr": sig.atr,
                "relative_volume": sig.relative_volume,
            },
        )

    def _calculate_entry_levels(self, candidate: Dict[str, Any], signal_type: SignalType):
        """Required by BaseScanner; entry levels actually come from strategy.scan()."""
        price = Decimal(str(candidate.get("current_price", 0)))
        return (price, price * Decimal("0.975"), price * Decimal("1.07"), None)
