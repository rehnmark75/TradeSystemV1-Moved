"""
Smart Money Reclaim Scanner (LuxAlgo-inspired)
==============================================

Live scanner wrapper around SmartMoneyReclaimStrategy. Same shared-strategy
pattern as EmaCross92150Scanner: it fetches daily candle data + the point-in-time
screening/breadth context and runs the strategy's `scan()`, so live signals share
ONE code path with the walk-forward backtest in
`analysis/smart_money_reclaim_replay.py`.

Setup: a prior swing low is liquidity-swept (lower low) WHILE Chaikin money flow
rises (bullish divergence = accumulation), then price reclaims the swept level
from a discount zone, on confirming volume, in a non-bear regime.

STATUS: monitor-only forward-test (Jul 1 2026). Registered in
stock_signal_scanners and in DEFAULT_ENABLED_SCANNERS (so it RUNS and logs
signals) but ALSO listed in MONITOR_ONLY_SCANNERS in
trading-ui/app/api/signals/top/route.ts -> its signals are excluded from the
live day-trade pool. Risk bracket mirrors the live auto-trader (SL capped 2.5%,
TP 7%) for direct comparability with ema_cross_9_21_50. Promote by REMOVING it
from MONITOR_ONLY_SCANNERS (do not remove from DEFAULT_ENABLED_SCANNERS).
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
from ...strategies.smart_money_reclaim import (
    SmartMoneyReclaimStrategy, SmartMoneyReclaimSignal,
)
from ...core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)


@dataclass
class SmartMoneyReclaimConfig(ScannerConfig):
    """Configuration for the smart-money sweep-reclaim scanner."""
    # Risk bracket — mirror the live auto-trader (SL 2.5%, TP 7%)
    stop_loss_pct: float = 2.5
    take_profit_pct: float = 7.0

    # Universe liquidity / quality floors (applied in scan())
    min_price: float = 3.0
    min_dollar_volume: float = 5_000_000.0   # avg_dollar_volume floor
    max_atr_percent: float = 15.0            # exclude lottery tickers

    max_signals_per_run: int = 50
    min_score_threshold: int = 55            # self-contained go/no-go in strategy


class SmartMoneyReclaimScanner(BaseScanner):
    """Scanner emitting money-flow-divergence sweep-reclaim signals (monitor-only)."""

    def __init__(self, db_manager, config: SmartMoneyReclaimConfig = None, scorer=None):
        super().__init__(db_manager, config or SmartMoneyReclaimConfig(), scorer)
        self.strategy = SmartMoneyReclaimStrategy(
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
        )
        self.data_provider = BacktestDataProvider(db_manager)

    @property
    def scanner_name(self) -> str:
        return "smart_money_reclaim"

    @property
    def description(self) -> str:
        return ("LuxAlgo-inspired money-flow divergence sweep-reclaim "
                "(monitor-only; shared strategy with backtest)")

    async def _get_universe_metrics(self, calc_date) -> Dict[str, Dict[str, Any]]:
        """Latest per-ticker screening metrics AS-OF the scan date, pre-filtered
        by the liquidity/quality universe floors. Returns {ticker: row-dict}."""
        rows = await self.db.fetch(
            """
            SELECT DISTINCT ON (ticker)
                   ticker, current_price, avg_dollar_volume, atr_percent,
                   relative_volume, rs_trend, rs_vs_spy
            FROM stock_screening_metrics
            WHERE calculation_date <= $1
              AND current_price >= $2
              AND avg_dollar_volume >= $3
              AND (atr_percent IS NULL OR atr_percent <= $4)
            ORDER BY ticker, calculation_date DESC
            """,
            calc_date,
            self.config.min_price,
            self.config.min_dollar_volume,
            self.config.max_atr_percent,
        )
        return {r["ticker"]: dict(r) for r in rows}

    async def _get_market_context(self, calc_date) -> Optional[Dict[str, Any]]:
        """Latest market_context row AS-OF the scan date (breadth co-gate)."""
        row = await self.db.fetchrow(
            """
            SELECT market_regime, pct_above_sma50, calculation_date
            FROM market_context
            WHERE calculation_date <= $1
            ORDER BY calculation_date DESC
            LIMIT 1
            """,
            calc_date,
        )
        return dict(row) if row else None

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            calculation_date = datetime.now().date()
        elif isinstance(calculation_date, datetime):
            calculation_date = calculation_date.date()

        metrics_by_ticker = await self._get_universe_metrics(calculation_date)
        market = await self._get_market_context(calculation_date)
        if market is None:
            logger.warning(
                "%s: no market_context row as-of %s — breadth co-gate flagged missing",
                self.scanner_name, calculation_date,
            )

        tickers = list(metrics_by_ticker.keys())
        logger.info(
            f"Scanning {len(tickers)} liquid stocks for smart-money sweep-reclaim"
        )

        signals: List[SignalSetup] = []
        for ticker in tickers:
            try:
                df = await self.data_provider.get_historical_data(
                    ticker=ticker,
                    start_date=calculation_date - timedelta(days=260),
                    end_date=calculation_date,
                    timeframe="1d",
                )
                if df.empty or len(df) < 110:
                    continue

                sig = self.strategy.scan(
                    df, ticker,
                    metrics=metrics_by_ticker.get(ticker),
                    market=market,
                    sector=None,
                )
                if sig:
                    setup = self._convert(sig)
                    if setup:
                        signals.append(setup)
                        logger.info(
                            f"{ticker}: SIGNAL {sig.quality_tier} "
                            f"score {sig.composite_score} @ {sig.entry_price} "
                            f"(zone {sig.zone_pos:.2f}, div {sig.divergence_delta:+.3f})"
                        )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{ticker}: error scanning - {e}")
                continue

        signals.sort(key=lambda s: s.composite_score, reverse=True)
        signals = signals[: self.config.max_signals_per_run]
        high_q = sum(1 for s in signals if s.is_high_quality)
        self.log_scan_summary(len(tickers), len(signals), high_q)
        return signals

    def _convert(self, sig: SmartMoneyReclaimSignal) -> Optional[SignalSetup]:
        tier_map = {
            "A+": QualityTier.A_PLUS, "A": QualityTier.A, "B": QualityTier.B,
            "C": QualityTier.C, "D": QualityTier.D,
        }
        entry = sig.entry_price
        stop = sig.stop_loss_price
        risk_pct = abs(entry - stop) / entry * 100 if entry > 0 else 0

        factors = [
            "Swing-low liquidity sweep",
            "Money-flow bullish divergence (accumulation into sweep)",
            "Reclaim of swept level",
            f"Discount zone (pos {sig.zone_pos:.2f})",
        ]
        if sig.relative_volume:
            factors.append(f"RVOL {sig.relative_volume:.1f}x")
        if sig.cmf_20 is not None:
            factors.append(f"CMF20 {sig.cmf_20:+.2f}")

        regime = sig.raw_data.get("market_regime") or "unknown"
        desc = (
            f"{sig.ticker} swept swing low {sig.swing_low} to {sig.sweep_low} "
            f"while money flow rose (CMF5 divergence {sig.divergence_delta:+.3f}), "
            f"then reclaimed from a discount zone. Quality {sig.quality_tier} "
            f"(score {sig.composite_score})."
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
            composite_score=sig.composite_score,
            quality_tier=tier_map.get(sig.quality_tier, QualityTier.B),
            volume_score=Decimal(str(round(
                sig.raw_data.get("confluence_components", {}).get("volume_conf", 0), 2))),
            confluence_score=Decimal(str(round(sig.composite_score, 2))),
            setup_description=desc,
            confluence_factors=factors[:8],
            timeframe="daily",
            market_regime=f"Sweep-reclaim / {regime}",
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data=sig.raw_data,
        )

    def _calculate_entry_levels(self, candidate: Dict[str, Any], signal_type: SignalType):
        """Required by BaseScanner; entry levels actually come from strategy.scan()."""
        price = Decimal(str(candidate.get("current_price", 0)))
        return (price, price * Decimal("0.975"), price * Decimal("1.07"), None)
