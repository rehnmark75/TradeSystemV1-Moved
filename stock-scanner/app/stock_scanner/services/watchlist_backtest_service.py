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
class BacktestSummary:
    total_signals: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    profit_factor: Optional[float]
    avg_hold_days: Optional[float]


class WatchlistBacktestService:
    """Backtest EMA 50 crossover signals for today's watchlist entries."""

    MIN_VOLUME = 1_000_000
    DEFAULT_DAYS = 90
    DEFAULT_MAX_HOLD_DAYS = 20

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

            summary = await self._backtest_ticker(
                ticker=ticker,
                end_date=scan_date,
                days=days,
                risk_pct=risk_pct,
                tp_pct=tp_pct,
                max_hold_days=max_hold_days
            )

            await self._store_summary(row['id'], scan_date, days, summary)
            processed += 1

        logger.info("EMA50 watchlist backtest complete: %s tickers", processed)
        return processed

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
        max_hold_days: int
    ) -> BacktestSummary:
        start_date = end_date - timedelta(days=days)
        df = await self.data_provider.get_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d',
            include_warmup=True
        )

        if df.empty:
            return BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0, None, None)

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        signals = self._find_signals(df, start_date)
        if not signals:
            return BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0, None, None)

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

        return BacktestSummary(
            total_signals=total_signals,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            total_pnl=round(total_pnl, 4),
            profit_factor=profit_factor,
            avg_hold_days=avg_hold_days
        )

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

            signals.append(i)

        return signals

    def _simulate_trade(self, df: pd.DataFrame, idx: int, risk_pct: float, tp_pct: float, max_hold_days: int) -> dict:
        entry_price = float(df.iloc[idx]['close'])
        stop_loss = entry_price * (1 - risk_pct)
        take_profit = entry_price * (1 + tp_pct)

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
            'exit_idx': exit_idx
        }

    async def _store_summary(self, row_id: int, scan_date: date, days: int, summary: BacktestSummary) -> None:
        end_date = scan_date
        start_date = scan_date - timedelta(days=days)

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
                bt_ema50_90d_last_run = $11
            WHERE id = $12
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
            row_id
        )
