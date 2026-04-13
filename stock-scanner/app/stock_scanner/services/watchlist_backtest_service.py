"""
Watchlist Backtest Service

Runs historical backtests for watchlist signals (EMA 50 crossover) and stores
summary stats back into stock_watchlist_results for the latest scan date.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from ..core.backtest.backtest_data_provider import BacktestDataProvider
from ..core.database.async_database_manager import AsyncDatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestGrade:
    score: Optional[float]       # 0-100 composite, sample-size-adjusted
    grade: str                   # A+, A, B, C, D, F, N/A
    confidence: str              # none, low, medium, high
    supports_signal: Optional[str]  # supports, neutral, contradicts, insufficient_data


@dataclass
class BacktestSummary:
    total_signals: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    profit_factor: Optional[float]
    avg_hold_days: Optional[float]
    grade: Optional[BacktestGrade] = None


class WatchlistBacktestService:
    """Backtest EMA 50 crossover signals for today's watchlist entries."""

    MIN_VOLUME = 1_000_000
    DEFAULT_DAYS = 90
    DEFAULT_MAX_HOLD_DAYS = 20
    FULLSETUP_DAYS = 180
    FULLSETUP_MIN_RS = 60     # Minimum RS percentile at crossover date
    FULLSETUP_MIN_DAQ = 50    # Minimum DAQ score at crossover date
    SIGNAL_EMA50_SLOPE_MIN = 1.005   # EMA50 must rise ≥0.5% over 5 bars
    SIGNAL_REL_VOL_MIN = 1.25        # Crossover day volume ≥1.25x 20d average
    SIGNAL_CLOSE_ABOVE_EMA50_MIN = 1.001  # Close ≥0.1% above EMA50

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db = db_manager
        self.data_provider = BacktestDataProvider(db_manager)

    async def run_ema50_today(self, days: int = DEFAULT_DAYS, max_hold_days: int = DEFAULT_MAX_HOLD_DAYS) -> int:
        scan_date = await self._get_latest_scan_date()
        if not scan_date:
            logger.warning("No scan_date found for ema_50_crossover watchlist")
            return 0

        rows = await self.db.fetch(
            """
            SELECT
                id,
                ticker,
                scan_date,
                crossover_date,
                price,
                suggested_stop_loss,
                suggested_target_1,
                suggested_target_2,
                risk_reward_ratio,
                risk_percent
            FROM stock_watchlist_results
            WHERE watchlist_name = 'ema_50_crossover'
              AND scan_date = $1
              AND status = 'active'
              AND tv_overall_signal = 'STRONG BUY'
              AND rs_percentile >= 60
              AND (daq_score < 70 OR daq_score IS NULL)
            ORDER BY ticker
            """,
            scan_date
        )

        if not rows:
            logger.info("No active ema_50_crossover rows for scan_date %s", scan_date)
            return 0

        processed = 0
        for row in rows:
            ticker = row['ticker']
            price = row['price']
            if price is None or price <= 0:
                logger.debug("Skipping %s (invalid price)", ticker)
                continue

            risk_pct, tp_pct = self._derive_risk_targets(row)
            if risk_pct <= 0 or tp_pct <= 0:
                logger.debug("Skipping %s (invalid risk/tp pct)", ticker)
                continue

            signal_date = row.get('crossover_date') or scan_date
            summary, validation = await self._backtest_ticker(
                ticker=ticker,
                end_date=scan_date,
                days=days,
                risk_pct=risk_pct,
                tp_pct=tp_pct,
                max_hold_days=max_hold_days,
                signal_date=signal_date
            )

            await self._store_summary(row['id'], scan_date, days, summary, validation)
            processed += 1

        logger.info("EMA50 watchlist backtest complete: %s tickers", processed)
        return processed

    async def run_fullsetup_today(
        self,
        days: int = FULLSETUP_DAYS,
        max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
        min_rs: int = FULLSETUP_MIN_RS,
        min_daq: int = FULLSETUP_MIN_DAQ,
    ) -> int:
        """
        Full-setup backtest: EMA50 crossover + RS ≥ min_rs + DAQ ≥ min_daq at crossover date.

        Looks back `days` days (default 180) for each active EMA50 watchlist ticker.
        RS is pulled from stock_screening_metrics (date-keyed, daily).
        DAQ is pulled from stock_watchlist_results (date-keyed, populated on scan days).
        Only simulates trades where BOTH filters pass at the crossover date.
        """
        scan_date = await self._get_latest_scan_date()
        if not scan_date:
            logger.warning("No scan_date found for ema_50_crossover watchlist (fullsetup)")
            return 0

        rows = await self.db.fetch(
            """
            SELECT
                id,
                ticker,
                scan_date,
                crossover_date,
                price,
                suggested_stop_loss,
                suggested_target_1,
                suggested_target_2,
                risk_reward_ratio,
                risk_percent,
                sector
            FROM stock_watchlist_results
            WHERE watchlist_name = 'ema_50_crossover'
              AND scan_date = $1
              AND status = 'active'
            ORDER BY ticker
            """,
            scan_date
        )

        if not rows:
            logger.info("No active ema_50_crossover rows for scan_date %s (fullsetup)", scan_date)
            return 0

        processed = 0
        sector_accum: dict = {}  # sector -> {signals, wins, losses, pnl_sum, hold_sum}

        for row in rows:
            ticker = row['ticker']
            price = row['price']
            if price is None or price <= 0:
                continue

            risk_pct, tp_pct = self._derive_risk_targets(row)
            if risk_pct <= 0 or tp_pct <= 0:
                continue

            signal_date = row.get('crossover_date') or scan_date
            summary, validation, filtered_count = await self._backtest_fullsetup(
                ticker=ticker,
                end_date=scan_date,
                days=days,
                risk_pct=risk_pct,
                tp_pct=tp_pct,
                max_hold_days=max_hold_days,
                min_rs=min_rs,
                min_daq=min_daq,
            )

            await self._store_summary_fullsetup(
                row_id=row['id'],
                scan_date=scan_date,
                days=days,
                min_rs=min_rs,
                min_daq=min_daq,
                summary=summary,
                filtered_count=filtered_count,
            )

            # Accumulate for sector stats
            sector = row.get('sector') or 'Unknown'
            if summary.total_signals > 0:
                acc = sector_accum.setdefault(sector, {'signals': 0, 'wins': 0, 'losses': 0, 'pnl_sum': 0.0, 'hold_sum': 0.0})
                acc['signals'] += summary.total_signals
                acc['wins'] += summary.wins
                acc['losses'] += summary.losses
                acc['pnl_sum'] += summary.total_pnl
                acc['hold_sum'] += (summary.avg_hold_days or 0) * summary.total_signals

            processed += 1

        # Store sector-wide aggregates
        await self._store_sector_stats(sector_accum, scan_date, days, min_rs, min_daq)

        logger.info("Full-setup watchlist backtest complete: %s tickers", processed)
        return processed

    async def _backtest_fullsetup(
        self,
        ticker: str,
        end_date: date,
        days: int,
        risk_pct: float,
        tp_pct: float,
        max_hold_days: int,
        min_rs: int,
        min_daq: int,
    ) -> tuple:
        """Run backtest filtering crossovers by RS + DAQ quality at the crossover date."""
        start_date = end_date - timedelta(days=days)
        df = await self.data_provider.get_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d',
            include_warmup=True
        )

        if df.empty:
            empty = BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0, None, None)
            empty.grade = self.compute_grade(empty)
            return empty, {}, 0

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        raw_signals = self._find_signals(df, start_date)

        if not raw_signals:
            empty = BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0, None, None)
            empty.grade = self.compute_grade(empty)
            return empty, {}, 0

        # For each signal, check RS + DAQ quality at the crossover date
        filtered_count = 0
        qualified_signals = []
        for idx in raw_signals:
            xover_date = df.iloc[idx]['timestamp'].date()
            rs_ok = await self._check_rs_at_date(ticker, xover_date, min_rs)
            daq_ok = await self._check_daq_at_date(ticker, xover_date, min_daq)
            filtered_count += 1
            if rs_ok and daq_ok:
                qualified_signals.append(idx)

        if not qualified_signals:
            empty = BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0, None, None)
            empty.grade = self.compute_grade(empty)
            return empty, {}, filtered_count

        wins = 0
        losses = 0
        total_pnl = 0.0
        win_sum = 0.0
        loss_sum = 0.0
        hold_days_sum = 0
        executed_signals = 0
        last_exit_idx = -1

        for idx in qualified_signals:
            if idx <= last_exit_idx:
                continue
            result = self._simulate_trade(df, idx, risk_pct, tp_pct, max_hold_days)
            pnl = result['pnl']
            total_pnl += pnl
            hold_days_sum += result['hold_days']
            last_exit_idx = result['exit_idx']
            executed_signals += 1

            if pnl > 0:
                wins += 1
                win_sum += pnl
            elif pnl < 0:
                losses += 1
                loss_sum += pnl

        total_signals = executed_signals
        win_rate = round((wins / total_signals) * 100, 2) if total_signals else 0.0
        avg_pnl = round(total_pnl / total_signals, 4) if total_signals else 0.0
        profit_factor = None
        if loss_sum < 0:
            profit_factor = round(win_sum / abs(loss_sum), 4) if abs(loss_sum) > 0 else None
        avg_hold_days = round(hold_days_sum / total_signals, 2) if total_signals else None

        summary = BacktestSummary(
            total_signals=total_signals,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            total_pnl=round(total_pnl, 4),
            profit_factor=profit_factor,
            avg_hold_days=avg_hold_days
        )
        summary.grade = self.compute_grade(summary)
        return summary, {}, filtered_count

    async def _check_rs_at_date(self, ticker: str, check_date: date, min_rs: int) -> bool:
        """Look up RS percentile from stock_screening_metrics at or near check_date."""
        row = await self.db.fetchrow(
            """
            SELECT rs_percentile
            FROM stock_screening_metrics
            WHERE ticker = $1
              AND calculation_date <= $2
            ORDER BY calculation_date DESC
            LIMIT 1
            """,
            ticker, check_date
        )
        if not row or row['rs_percentile'] is None:
            return True  # No data → don't penalise (benefit of the doubt)
        return int(row['rs_percentile']) >= min_rs

    async def _check_daq_at_date(self, ticker: str, check_date: date, min_daq: int) -> bool:
        """Look up DAQ score from stock_watchlist_results at or near check_date."""
        row = await self.db.fetchrow(
            """
            SELECT daq_score
            FROM stock_watchlist_results
            WHERE ticker = $1
              AND watchlist_name = 'ema_50_crossover'
              AND scan_date <= $2
              AND daq_score IS NOT NULL
            ORDER BY scan_date DESC
            LIMIT 1
            """,
            ticker, check_date
        )
        if not row or row['daq_score'] is None:
            return True  # No history → benefit of the doubt
        return int(row['daq_score']) >= min_daq

    async def _store_summary_fullsetup(
        self,
        row_id: int,
        scan_date: date,
        days: int,
        min_rs: int,
        min_daq: int,
        summary: 'BacktestSummary',
        filtered_count: int,
    ) -> None:
        end_date = scan_date
        start_date = scan_date - timedelta(days=days)
        grade = summary.grade

        await self.db.execute(
            """
            UPDATE stock_watchlist_results
            SET
                bt_fullsetup_180d_signals       = $1,
                bt_fullsetup_180d_wins          = $2,
                bt_fullsetup_180d_losses        = $3,
                bt_fullsetup_180d_win_rate      = $4,
                bt_fullsetup_180d_avg_pnl       = $5,
                bt_fullsetup_180d_total_pnl     = $6,
                bt_fullsetup_180d_profit_factor = $7,
                bt_fullsetup_180d_avg_hold_days = $8,
                bt_fullsetup_180d_start_date    = $9,
                bt_fullsetup_180d_end_date      = $10,
                bt_fullsetup_180d_last_run      = $11,
                bt_fullsetup_180d_score         = $12,
                bt_fullsetup_180d_grade         = $13,
                bt_fullsetup_180d_confidence    = $14,
                bt_fullsetup_180d_supports_signal = $15,
                bt_fullsetup_180d_filtered_count = $16,
                bt_fullsetup_min_rs             = $17,
                bt_fullsetup_min_daq            = $18
            WHERE id = $19
            """,
            summary.total_signals,
            summary.wins,
            summary.losses,
            summary.win_rate,
            summary.avg_pnl,
            summary.total_pnl,
            summary.profit_factor,
            summary.avg_hold_days,
            start_date,
            end_date,
            datetime.utcnow(),
            grade.score if grade else None,
            grade.grade if grade else None,
            grade.confidence if grade else None,
            grade.supports_signal if grade else None,
            filtered_count,
            min_rs,
            min_daq,
            row_id
        )

    async def _store_sector_stats(
        self,
        sector_accum: dict,
        stat_date: date,
        lookback_days: int,
        min_rs: int,
        min_daq: int,
    ) -> None:
        """Upsert sector-wide backtest aggregates."""
        for sector, acc in sector_accum.items():
            total_signals = acc['signals']
            if total_signals == 0:
                continue
            wins = acc['wins']
            losses = acc['losses']
            win_rate = round((wins / total_signals) * 100, 2)
            avg_pnl = round(acc['pnl_sum'] / total_signals, 4)
            avg_hold = round(acc['hold_sum'] / total_signals, 2)
            win_pnl = acc['pnl_sum'] if wins else 0.0
            loss_pnl = abs(acc['pnl_sum'] - win_pnl) if losses else 0.0
            profit_factor = round(win_pnl / loss_pnl, 4) if loss_pnl > 0 else None

            await self.db.execute(
                """
                INSERT INTO stock_backtest_sector_stats
                    (sector, watchlist_name, stat_date, lookback_days, min_rs, min_daq,
                     total_signals, total_wins, total_losses,
                     win_rate, avg_pnl, profit_factor, avg_hold_days)
                VALUES ($1, 'ema_50_crossover', $2, $3, $4, $5,
                        $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (sector, watchlist_name, stat_date, lookback_days, min_rs, min_daq)
                DO UPDATE SET
                    total_signals = EXCLUDED.total_signals,
                    total_wins    = EXCLUDED.total_wins,
                    total_losses  = EXCLUDED.total_losses,
                    win_rate      = EXCLUDED.win_rate,
                    avg_pnl       = EXCLUDED.avg_pnl,
                    profit_factor = EXCLUDED.profit_factor,
                    avg_hold_days = EXCLUDED.avg_hold_days
                """,
                sector, stat_date, lookback_days, min_rs, min_daq,
                total_signals, wins, losses, win_rate, avg_pnl, profit_factor, avg_hold
            )

    @staticmethod
    def compute_grade(summary: 'BacktestSummary') -> BacktestGrade:
        """Compute a composite backtest grade accounting for sample size."""
        if summary.total_signals == 0:
            return BacktestGrade(score=None, grade="N/A", confidence="none", supports_signal=None)

        signals = summary.total_signals
        win_rate = summary.win_rate / 100.0  # stored as 0-100, normalize
        # If PF is null (no losses), treat as max cap — 100% WR is excellent
        pf = summary.profit_factor if summary.profit_factor is not None else (3.0 if summary.wins > 0 else 0.0)
        avg_pnl = summary.avg_pnl

        # Sample size multiplier — gates everything
        sample_mult = {1: 0.40, 2: 0.60, 3: 0.75, 4: 0.85}.get(signals, 1.0)

        # Win rate component (0-40 pts)
        wr_score = min(40.0, max(0.0, (win_rate - 0.40) / 0.60 * 40.0))

        # Profit factor component (0-35 pts)
        pf_capped = min(pf, 3.0)
        pf_score = min(35.0, max(0.0, (pf_capped - 1.0) / 2.0 * 35.0))

        # Avg PnL component (0-25 pts) — 5% avg = full marks
        pnl_score = min(25.0, max(0.0, (avg_pnl / 5.0) * 25.0))

        raw_score = wr_score + pf_score + pnl_score
        composite = round(raw_score * sample_mult, 1)

        # Letter grade
        if composite >= 80:
            grade = "A+"
        elif composite >= 68:
            grade = "A"
        elif composite >= 55:
            grade = "B"
        elif composite >= 42:
            grade = "C"
        elif composite >= 28:
            grade = "D"
        else:
            grade = "F"

        # Cap grade by sample size
        if signals == 1 and grade in ("A+", "A"):
            grade = "B"
        if signals == 2 and grade == "A+":
            grade = "A"

        # Statistical confidence
        if signals >= 5:
            confidence = "high"
        elif signals >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        # Signal support verdict
        if confidence == "low":
            supports_signal = "insufficient_data"
        elif win_rate >= 0.60 and pf >= 1.15 and avg_pnl > 0:
            supports_signal = "supports"
        elif win_rate <= 0.45 or pf < 0.90:
            supports_signal = "contradicts"
        else:
            supports_signal = "neutral"

        return BacktestGrade(
            score=composite,
            grade=grade,
            confidence=confidence,
            supports_signal=supports_signal
        )

    async def _get_latest_scan_date(self) -> Optional[date]:
        row = await self.db.fetchrow(
            """
            SELECT MAX(scan_date) AS max_date
            FROM stock_watchlist_results
            WHERE watchlist_name = 'ema_50_crossover'
            """
        )
        return row['max_date'] if row else None

    def _derive_risk_targets(self, row) -> Tuple[float, float]:
        price = float(row['price'])
        stop_loss = row['suggested_stop_loss']
        target_1 = row['suggested_target_1']
        risk_pct = 0.0
        tp_pct = 0.0

        if stop_loss is not None:
            risk_pct = (price - float(stop_loss)) / price if price > 0 else 0.0
        elif row['risk_percent'] is not None:
            risk_pct = float(row['risk_percent']) / 100.0

        if target_1 is not None:
            tp_pct = (float(target_1) - price) / price if price > 0 else 0.0
        else:
            rr = float(row['risk_reward_ratio'] or 2.0)
            tp_pct = risk_pct * rr

        if tp_pct <= 0 and row['suggested_target_2'] is not None:
            tp_pct = (float(row['suggested_target_2']) - price) / price if price > 0 else 0.0

        return max(risk_pct, 0.0), max(tp_pct, 0.0)

    async def _backtest_ticker(
        self,
        ticker: str,
        end_date: date,
        days: int,
        risk_pct: float,
        tp_pct: float,
        max_hold_days: int,
        signal_date: Optional[date] = None
    ) -> Tuple[BacktestSummary, dict]:
        start_date = end_date - timedelta(days=days)
        df = await self.data_provider.get_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d',
            include_warmup=True
        )

        if df.empty:
            empty = BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0, None, None)
            empty.grade = self.compute_grade(empty)
            return empty, {'signal_validated': False, 'signal_validation_reasons': 'no_price_data', 'bt_stop_method': None}

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        validation = self._validate_signal(df, signal_date or end_date)
        signals = self._find_signals(df, start_date)
        if not signals:
            empty = BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0, None, None)
            empty.grade = self.compute_grade(empty)
            return empty, validation

        wins = 0
        losses = 0
        total_pnl = 0.0
        win_sum = 0.0
        loss_sum = 0.0
        hold_days_sum = 0
        executed_signals = 0
        last_exit_idx = -1

        for idx in signals:
            if idx <= last_exit_idx:
                continue
            result = self._simulate_trade(df, idx, risk_pct, tp_pct, max_hold_days)
            pnl = result['pnl']
            total_pnl += pnl
            hold_days_sum += result['hold_days']
            last_exit_idx = result['exit_idx']
            executed_signals += 1

            if pnl > 0:
                wins += 1
                win_sum += pnl
            elif pnl < 0:
                losses += 1
                loss_sum += pnl

        total_signals = executed_signals
        win_rate = round((wins / total_signals) * 100, 2) if total_signals else 0.0
        avg_pnl = round(total_pnl / total_signals, 4) if total_signals else 0.0
        profit_factor = None
        if loss_sum < 0:
            profit_factor = round(win_sum / abs(loss_sum), 4) if abs(loss_sum) > 0 else None
        avg_hold_days = round(hold_days_sum / total_signals, 2) if total_signals else None

        summary = BacktestSummary(
            total_signals=total_signals,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            total_pnl=round(total_pnl, 4),
            profit_factor=profit_factor,
            avg_hold_days=avg_hold_days
        )
        summary.grade = self.compute_grade(summary)
        return summary, validation

    def _find_signals(self, df: pd.DataFrame, start_date: date) -> List[int]:
        signals: List[int] = []
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            if row['timestamp'].date() < start_date:
                continue

            price = row['close']
            ema_50 = row.get('ema_50')
            ema_200 = row.get('ema_200')
            volume_sma = row.get('volume_sma_20')

            if pd.isna(price) or pd.isna(ema_50) or pd.isna(ema_200):
                continue

            if price <= ema_200 or price <= ema_50:
                continue

            prev_ema_50 = prev.get('ema_50')
            if pd.isna(prev_ema_50):
                continue

            if prev['close'] >= prev_ema_50:
                continue

            if volume_sma is None or pd.isna(volume_sma) or volume_sma < self.MIN_VOLUME:
                continue

            # --- Visual quality filters (clean crossovers only) ---
            # 1. EMA50 slope: must be rising ≥0.5% over last 5 bars
            if i < 5:
                continue
            ema_50_5bars = df.iloc[i - 5].get('ema_50')
            if pd.isna(ema_50_5bars) or float(ema_50) <= float(ema_50_5bars) * self.SIGNAL_EMA50_SLOPE_MIN:
                continue

            # 2. EMA stack: EMA20 must be above EMA50 at crossover (pullback in uptrend)
            ema_20 = row.get('ema_20')
            if pd.isna(ema_20) or float(ema_20) <= float(ema_50):
                continue

            # 3. Volume confirmation: crossover day must be elevated vs 20d average
            rel_vol = row.get('relative_volume')
            if rel_vol is None or pd.isna(rel_vol) or float(rel_vol) < self.SIGNAL_REL_VOL_MIN:
                continue

            # 4. Close distance: not a marginal tag of EMA50
            if float(price) < float(ema_50) * self.SIGNAL_CLOSE_ABOVE_EMA50_MIN:
                continue

            signals.append(i)

        return signals

    def _simulate_trade(self, df: pd.DataFrame, idx: int, risk_pct: float, tp_pct: float, max_hold_days: int) -> dict:
        entry_price = float(df.iloc[idx]['close'])
        atr = df.iloc[idx].get('atr')
        if atr is not None and not pd.isna(atr) and float(atr) > 0:
            stop_loss = entry_price - (1.5 * float(atr))
            take_profit = entry_price + (2.25 * float(atr))
            stop_method = 'atr'
        else:
            stop_loss = entry_price * (1 - risk_pct)
            take_profit = entry_price * (1 + tp_pct)
            stop_method = 'pct'

        exit_price = entry_price
        hold_days = 0
        exit_reason = 'TIMEOUT'
        exit_idx = idx

        max_idx = min(idx + max_hold_days, len(df) - 1)
        for j in range(idx + 1, max_idx + 1):
            row = df.iloc[j]
            low = float(row['low'])
            high = float(row['high'])
            hold_days = j - idx
            exit_idx = j

            if low <= stop_loss and high >= take_profit:
                exit_price = stop_loss
                exit_reason = 'SL_HIT'
                break
            if low <= stop_loss:
                exit_price = stop_loss
                exit_reason = 'SL_HIT'
                break
            if high >= take_profit:
                exit_price = take_profit
                exit_reason = 'TP_HIT'
                break

        if exit_reason == 'TIMEOUT':
            exit_price = float(df.iloc[max_idx]['close'])
            exit_idx = max_idx

        pnl = (exit_price - entry_price) / entry_price * 100

        return {
            'exit_reason': exit_reason,
            'exit_price': exit_price,
            'pnl': pnl,
            'hold_days': hold_days,
            'exit_idx': exit_idx,
            'bt_stop_method': stop_method
        }

    def _validate_signal(self, df: pd.DataFrame, signal_date: date) -> dict:
        """
        Validate the current watchlist signal using the same visual-quality filters
        applied in _find_signals(). Checks the signal_date bar specifically.
        """
        reasons: List[str] = []

        idxs = df.index[df['timestamp'].dt.date == signal_date].tolist()
        if not idxs:
            return {'signal_validated': False, 'signal_validation_reasons': f'missing_bar:{signal_date.isoformat()}', 'bt_stop_method': None}

        i = int(idxs[-1])
        if i <= 0:
            return {'signal_validated': False, 'signal_validation_reasons': 'insufficient_history', 'bt_stop_method': None}

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        price = row.get('close')
        ema_50 = row.get('ema_50')
        ema_200 = row.get('ema_200')
        volume_sma = row.get('volume_sma_20')
        prev_ema_50 = prev.get('ema_50')
        prev_close = prev.get('close')

        # Guard: missing required indicators
        required = {'close': price, 'ema_50': ema_50, 'ema_200': ema_200, 'prev_ema_50': prev_ema_50, 'prev_close': prev_close}
        if any(v is None or pd.isna(v) for v in required.values()):
            return {'signal_validated': False, 'signal_validation_reasons': 'missing_indicators', 'bt_stop_method': None}

        validated = True

        if float(price) <= float(ema_200):
            validated = False
            reasons.append('below_ema200')

        if float(price) <= float(ema_50):
            validated = False
            reasons.append('below_ema50')

        if float(prev_close) >= float(prev_ema_50):
            validated = False
            reasons.append('no_crossover')

        if volume_sma is None or pd.isna(volume_sma) or float(volume_sma) < self.MIN_VOLUME:
            validated = False
            reasons.append('low_avg_volume')

        if i < 5:
            validated = False
            reasons.append('insufficient_slope_history')
        else:
            ema_50_5bars = df.iloc[i - 5].get('ema_50')
            if ema_50_5bars is None or pd.isna(ema_50_5bars):
                validated = False
                reasons.append('missing_ema50_slope')
            elif float(ema_50) <= float(ema_50_5bars) * self.SIGNAL_EMA50_SLOPE_MIN:
                validated = False
                reasons.append('ema50_flat')

        ema_20 = row.get('ema_20')
        if ema_20 is None or pd.isna(ema_20) or float(ema_20) <= float(ema_50):
            validated = False
            reasons.append('ema_stack')

        rel_vol = row.get('relative_volume')
        if rel_vol is None or pd.isna(rel_vol) or float(rel_vol) < self.SIGNAL_REL_VOL_MIN:
            validated = False
            reasons.append('low_rel_volume')

        if float(price) < float(ema_50) * self.SIGNAL_CLOSE_ABOVE_EMA50_MIN:
            validated = False
            reasons.append('marginal_close')

        atr = row.get('atr')
        bt_stop_method = 'atr' if atr is not None and not pd.isna(atr) and float(atr) > 0 else 'pct'

        reason_text = 'validated' if validated else ('failed:' + ','.join(reasons) if reasons else 'failed')
        return {
            'signal_validated': validated,
            'signal_validation_reasons': reason_text,
            'bt_stop_method': bt_stop_method
        }

    async def _store_summary(self, row_id: int, scan_date: date, days: int, summary: BacktestSummary, validation: dict) -> None:
        end_date = scan_date
        start_date = scan_date - timedelta(days=days)
        grade = summary.grade

        await self.db.execute(
            """
            UPDATE stock_watchlist_results
            SET
                bt_ema50_90d_signals = $1,
                bt_ema50_90d_wins = $2,
                bt_ema50_90d_losses = $3,
                bt_ema50_90d_win_rate = $4,
                bt_ema50_90d_avg_pnl = $5,
                bt_ema50_90d_total_pnl = $6,
                bt_ema50_90d_profit_factor = $7,
                bt_ema50_90d_avg_hold_days = $8,
                bt_ema50_90d_start_date = $9,
                bt_ema50_90d_end_date = $10,
                bt_ema50_90d_last_run = $11,
                bt_ema50_90d_score = $12,
                bt_ema50_90d_grade = $13,
                bt_ema50_90d_confidence = $14,
                bt_ema50_90d_supports_signal = $15,
                signal_validated = $16,
                signal_validation_reasons = $17,
                bt_stop_method = $18
            WHERE id = $19
            """,
            summary.total_signals,
            summary.wins,
            summary.losses,
            summary.win_rate,
            summary.avg_pnl,
            summary.total_pnl,
            summary.profit_factor,
            summary.avg_hold_days,
            start_date,
            end_date,
            datetime.utcnow(),
            grade.score if grade else None,
            grade.grade if grade else None,
            grade.confidence if grade else None,
            grade.supports_signal if grade else None,
            validation.get('signal_validated'),
            validation.get('signal_validation_reasons'),
            validation.get('bt_stop_method'),
            row_id
        )
