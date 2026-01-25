"""
Relative Strength Calculator

Calculates Relative Strength metrics for stocks:
- RS vs SPY (stock performance / SPY performance)
- RS Percentile (rank among all stocks, 1-100)
- RS Trend (improving/stable/deteriorating)
- Sector RS (stock vs sector ETF)

Runs as a post-processing step after main metrics calculation.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# SPY is the benchmark for relative strength
SPY_TICKER = 'SPY'

# Sector ETF mapping (also stored in DB)
SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Health Care': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Communication Services': 'XLC',
    'Industrials': 'XLI',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB',
}

SECTOR_NAME_ALIASES = {
    'Technology': ['Technology'],
    'Health Care': ['Health Care', 'Healthcare'],
    'Financials': ['Financials', 'Financial Services'],
    'Consumer Discretionary': ['Consumer Discretionary', 'Consumer Cyclical'],
    'Consumer Staples': ['Consumer Staples', 'Consumer Defensive'],
    'Communication Services': ['Communication Services'],
    'Industrials': ['Industrials'],
    'Energy': ['Energy'],
    'Utilities': ['Utilities'],
    'Real Estate': ['Real Estate'],
    'Materials': ['Materials', 'Basic Materials'],
}


class RSCalculator:
    """
    Calculate Relative Strength metrics for all stocks.

    RS is calculated as: (Stock 20-day return) / (SPY 20-day return)
    - RS > 1.0 means stock is outperforming SPY
    - RS < 1.0 means stock is underperforming SPY

    RS Percentile ranks stocks from 1-100 based on their RS values.
    """

    def __init__(self, db_manager):
        self.db = db_manager
        self._spy_returns: Dict[str, float] = {}  # Cache SPY returns by date

    def _get_sector_aliases(self, sector: str) -> List[str]:
        """Return sector name aliases for matching instrument data."""
        return SECTOR_NAME_ALIASES.get(sector, [sector])

    async def calculate_all_rs(
        self,
        calculation_date: date = None,
        concurrency: int = 50
    ) -> Dict[str, Any]:
        """
        Calculate RS metrics for all stocks on a given date.

        This should be run AFTER the main metrics calculation is complete.

        Args:
            calculation_date: Date to calculate RS for
            concurrency: Number of parallel updates

        Returns:
            Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("RELATIVE STRENGTH CALCULATION")
        logger.info("=" * 60)

        start_time = datetime.now()

        if calculation_date is None:
            calculation_date = datetime.now().date()  # Use today - represents when pipeline ran

        # Step 1: Get SPY 20-day return
        spy_return = await self._get_benchmark_return(SPY_TICKER, calculation_date, 20)
        if spy_return is None:
            logger.error("Cannot calculate RS: SPY data not available")
            return {'error': 'SPY data not available'}

        logger.info(f"SPY 20-day return: {spy_return:.2f}%")

        # Step 2: Get all stocks with perf_1m (20-day proxy)
        stocks = await self._get_stocks_with_metrics(calculation_date)
        logger.info(f"Processing {len(stocks)} stocks")

        if not stocks:
            logger.warning("No stocks found with metrics for RS calculation")
            return {'total': 0, 'processed': 0}

        # Step 3: Calculate RS for each stock
        rs_data = []
        for stock in stocks:
            ticker = stock['ticker']
            stock_return = stock.get('price_change_20d') or stock.get('perf_1m')

            if stock_return is not None and spy_return != 0:
                stock_return = float(stock_return)
                # RS ratio: stock return / SPY return (normalized)
                # If SPY is up 5% and stock is up 10%, RS = 2.0 (outperforming 2x)
                rs_vs_spy = stock_return / spy_return if spy_return != 0 else 1.0
                rs_data.append({
                    'ticker': ticker,
                    'rs_vs_spy': round(rs_vs_spy, 4),
                    'stock_return': stock_return,
                    'sector': stock.get('sector')
                })

        # Step 4: Calculate percentiles
        if rs_data:
            rs_values = [d['rs_vs_spy'] for d in rs_data]
            for data in rs_data:
                # Percentile: what % of stocks have lower RS
                count_below = sum(1 for r in rs_values if r < data['rs_vs_spy'])
                data['rs_percentile'] = int((count_below / len(rs_values)) * 100)

        # Step 5: Calculate RS trend (compare to 5 days ago)
        await self._calculate_rs_trends(rs_data, calculation_date)

        # Step 6: Update database
        updated = await self._update_rs_metrics(rs_data, calculation_date, concurrency)

        elapsed = (datetime.now() - start_time).total_seconds()

        stats = {
            'calculation_date': str(calculation_date),
            'spy_return': round(spy_return, 2),
            'total_stocks': len(stocks),
            'processed': len(rs_data),
            'updated': updated,
            'duration_seconds': round(elapsed, 2)
        }

        logger.info(f"RS calculation complete:")
        logger.info(f"  Processed: {len(rs_data)}")
        logger.info(f"  Updated: {updated}")
        logger.info(f"  Duration: {elapsed:.1f}s")

        return stats

    async def _get_benchmark_return(
        self,
        ticker: str,
        calc_date: date,
        days: int
    ) -> Optional[float]:
        """Get the N-day return for a benchmark (SPY, sector ETF)."""
        # Try synthesized candles first, then raw candles
        rows = []
        for table in ['stock_candles_synthesized', 'stock_candles']:
            if table == 'stock_candles_synthesized':
                query = """
                    SELECT close
                    FROM stock_candles_synthesized
                    WHERE ticker = $1 AND timeframe = '1d'
                      AND DATE(timestamp) <= $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                """
            else:
                query = """
                    SELECT close
                    FROM stock_candles
                    WHERE ticker = $1
                      AND timeframe = '1d'
                      AND DATE(timestamp) <= $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                """
            rows = await self.db.fetch(query, ticker, calc_date, days + 1)
            if len(rows) >= days + 1:
                break

        # If no database data for ETFs, try yfinance
        if len(rows) < days + 1:
            try:
                import yfinance as yf
                etf = yf.Ticker(ticker)
                # Use 3mo period to ensure we have enough data
                hist = etf.history(period="3mo")
                if len(hist) >= days + 1:
                    closes = hist['Close'].values.tolist()
                    rows = [{'close': c} for c in reversed(closes)]
            except Exception as e:
                logger.debug(f"Failed to fetch {ticker} from yfinance: {e}")

        if len(rows) < days + 1:
            return None

        current = float(rows[0]['close'])
        past = float(rows[-1]['close'])

        if past > 0:
            return ((current / past) - 1) * 100

        return None

    async def _get_stocks_with_metrics(self, calc_date: date) -> List[Dict]:
        """Get all stocks that have metrics for the given date."""
        query = """
            SELECT
                m.ticker,
                m.price_change_20d,
                m.perf_1m,
                i.sector
            FROM stock_screening_metrics m
            LEFT JOIN stock_instruments i ON m.ticker = i.ticker
            WHERE m.calculation_date = $1
              AND (m.price_change_20d IS NOT NULL OR m.perf_1m IS NOT NULL)
        """
        rows = await self.db.fetch(query, calc_date)
        return [dict(r) for r in rows]

    async def _calculate_rs_trends(
        self,
        rs_data: List[Dict],
        calc_date: date
    ) -> None:
        """
        Calculate RS trend by comparing current RS to 5 days ago.

        Trend categories:
        - improving: RS increased by >5%
        - stable: RS changed by <5%
        - deteriorating: RS decreased by >5%
        """
        past_date = calc_date - timedelta(days=7)  # Account for weekends

        # Get past RS values
        query = """
            SELECT ticker, rs_vs_spy
            FROM stock_screening_metrics
            WHERE calculation_date = (
                SELECT MAX(calculation_date)
                FROM stock_screening_metrics
                WHERE calculation_date <= $1
            )
              AND rs_vs_spy IS NOT NULL
        """
        rows = await self.db.fetch(query, past_date)
        past_rs = {r['ticker']: float(r['rs_vs_spy']) for r in rows}

        for data in rs_data:
            ticker = data['ticker']
            current_rs = data['rs_vs_spy']

            if ticker in past_rs and past_rs[ticker] != 0:
                change_pct = ((current_rs / past_rs[ticker]) - 1) * 100

                if change_pct > 5:
                    data['rs_trend'] = 'improving'
                elif change_pct < -5:
                    data['rs_trend'] = 'deteriorating'
                else:
                    data['rs_trend'] = 'stable'
            else:
                data['rs_trend'] = 'stable'  # No history, assume stable

    async def _update_rs_metrics(
        self,
        rs_data: List[Dict],
        calc_date: date,
        concurrency: int
    ) -> int:
        """Update RS columns in stock_screening_metrics."""
        semaphore = asyncio.Semaphore(concurrency)
        updated = 0

        async def update_one(data: Dict) -> bool:
            async with semaphore:
                try:
                    query = """
                        UPDATE stock_screening_metrics
                        SET rs_vs_spy = $1,
                            rs_percentile = $2,
                            rs_trend = $3
                        WHERE ticker = $4 AND calculation_date = $5
                    """
                    await self.db.execute(
                        query,
                        data['rs_vs_spy'],
                        data['rs_percentile'],
                        data.get('rs_trend', 'stable'),
                        data['ticker'],
                        calc_date
                    )
                    return True
                except Exception as e:
                    logger.warning(f"Failed to update RS for {data['ticker']}: {e}")
                    return False

        tasks = [update_one(d) for d in rs_data]
        results = await asyncio.gather(*tasks)

        return sum(1 for r in results if r)

    async def calculate_sector_rs(self, calculation_date: date = None) -> Dict[str, Any]:
        """
        Calculate sector-level RS and update sector_analysis table.

        This provides sector rotation analysis:
        - Sector RS vs SPY
        - Sector stage (leading/weakening/lagging/improving)
        - Top stocks in each sector
        """
        logger.info("Calculating Sector RS...")

        if calculation_date is None:
            calculation_date = datetime.now().date()  # Use today - represents when pipeline ran

        # Get SPY return
        spy_return = await self._get_benchmark_return(SPY_TICKER, calculation_date, 20)
        if spy_return is None:
            return {'error': 'SPY data not available'}

        sectors_processed = 0

        for sector, etf in SECTOR_ETF_MAP.items():
            try:
                # Get sector ETF return
                sector_return = await self._get_benchmark_return(etf, calculation_date, 20)

                if sector_return is None:
                    logger.debug(f"No data for sector ETF {etf}")
                    continue

                # Calculate sector RS
                sector_rs = sector_return / spy_return if spy_return != 0 else 1.0

                # Get previous RS for trend
                past_date = calculation_date - timedelta(days=7)
                prev_query = """
                    SELECT rs_vs_spy
                    FROM sector_analysis
                    WHERE sector = $1 AND calculation_date <= $2
                    ORDER BY calculation_date DESC LIMIT 1
                """
                prev_rows = await self.db.fetch(prev_query, sector, past_date)
                prev_rs = float(prev_rows[0]['rs_vs_spy']) if prev_rows else sector_rs

                # Determine trend
                change_pct = ((sector_rs / prev_rs) - 1) * 100 if prev_rs != 0 else 0
                if change_pct > 5:
                    rs_trend = 'improving'
                elif change_pct < -5:
                    rs_trend = 'deteriorating'
                else:
                    rs_trend = 'stable'

                # Determine stage
                if sector_rs > 1.0 and rs_trend in ('improving', 'stable'):
                    stage = 'leading'
                elif sector_rs > 1.0 and rs_trend == 'deteriorating':
                    stage = 'weakening'
                elif sector_rs < 1.0 and rs_trend in ('deteriorating', 'stable'):
                    stage = 'lagging'
                else:
                    stage = 'improving'

                # Get top stocks in sector
                top_stocks_query = """
                    SELECT m.ticker, m.rs_percentile, m.price_change_20d
                    FROM stock_screening_metrics m
                    JOIN stock_instruments i ON m.ticker = i.ticker
                    WHERE m.calculation_date = $1
                      AND i.sector = ANY($2::text[])
                      AND m.rs_percentile IS NOT NULL
                    ORDER BY m.rs_percentile DESC
                    LIMIT 5
                """
                sector_aliases = self._get_sector_aliases(sector)
                top_rows = await self.db.fetch(top_stocks_query, calculation_date, sector_aliases)
                top_stocks = [
                    {'ticker': r['ticker'], 'rs_percentile': r['rs_percentile']}
                    for r in top_rows
                ]

                # Count stocks in sector
                count_query = """
                    SELECT COUNT(*) as cnt
                    FROM stock_instruments
                    WHERE sector = ANY($1::text[]) AND is_active = TRUE
                """
                count_row = await self.db.fetch(count_query, sector_aliases)
                stock_count = count_row[0]['cnt'] if count_row else 0

                # Get 1d and 5d returns for sector
                sector_1d = await self._get_benchmark_return(etf, calculation_date, 1)
                sector_5d = await self._get_benchmark_return(etf, calculation_date, 5)

                # Insert/update sector_analysis
                import json
                upsert_query = """
                    INSERT INTO sector_analysis (
                        calculation_date, sector, sector_etf,
                        sector_return_1d, sector_return_5d, sector_return_20d,
                        rs_vs_spy, rs_trend, stocks_in_sector,
                        top_stocks, sector_stage
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                    )
                    ON CONFLICT (calculation_date, sector)
                    DO UPDATE SET
                        sector_return_1d = EXCLUDED.sector_return_1d,
                        sector_return_5d = EXCLUDED.sector_return_5d,
                        sector_return_20d = EXCLUDED.sector_return_20d,
                        rs_vs_spy = EXCLUDED.rs_vs_spy,
                        rs_trend = EXCLUDED.rs_trend,
                        stocks_in_sector = EXCLUDED.stocks_in_sector,
                        top_stocks = EXCLUDED.top_stocks,
                        sector_stage = EXCLUDED.sector_stage
                """
                await self.db.execute(
                    upsert_query,
                    calculation_date, sector, etf,
                    sector_1d, sector_5d, sector_return,
                    round(sector_rs, 4), rs_trend, stock_count,
                    json.dumps(top_stocks), stage
                )

                sectors_processed += 1
                logger.debug(f"Sector {sector}: RS={sector_rs:.2f}, Stage={stage}")

            except Exception as e:
                logger.warning(f"Failed to process sector {sector}: {e}")

        # Calculate sector percentiles
        await self._update_sector_percentiles(calculation_date)

        logger.info(f"Sector RS calculation complete: {sectors_processed} sectors")
        return {'sectors_processed': sectors_processed}

    async def _update_sector_percentiles(self, calc_date: date) -> None:
        """Update rs_percentile for sectors based on their RS values."""
        query = """
            WITH ranked AS (
                SELECT
                    id,
                    PERCENT_RANK() OVER (ORDER BY rs_vs_spy) * 100 as pct
                FROM sector_analysis
                WHERE calculation_date = $1
            )
            UPDATE sector_analysis sa
            SET rs_percentile = ROUND(r.pct)
            FROM ranked r
            WHERE sa.id = r.id
        """
        await self.db.execute(query, calc_date)


class MarketRegimeCalculator:
    """
    Calculate market regime based on SPY and breadth indicators.

    Regimes:
    - bull_confirmed: SPY > SMA200, SMA200 rising
    - bull_weakening: SPY > SMA200, SMA200 flat/falling
    - bear_weakening: SPY < SMA200, SMA200 rising
    - bear_confirmed: SPY < SMA200, SMA200 falling
    """

    def __init__(self, db_manager):
        self.db = db_manager

    async def calculate_market_regime(
        self,
        calculation_date: date = None
    ) -> Dict[str, Any]:
        """Calculate and store market regime for the given date."""
        logger.info("Calculating Market Regime...")

        if calculation_date is None:
            calculation_date = datetime.now().date()  # Use today - represents when pipeline ran

        # Get SPY data
        spy_data = await self._get_spy_metrics(calculation_date)
        if not spy_data:
            return {'error': 'SPY metrics not available'}

        # Calculate breadth indicators
        breadth = await self._calculate_breadth(calculation_date)

        # Determine regime
        regime = self._classify_regime(spy_data, breadth)

        # Determine volatility regime
        vol_regime = await self._calculate_volatility_regime(calculation_date)

        # Build recommended strategies based on regime
        import json
        if regime == 'bull_confirmed':
            strategies = {'trend_following': 0.8, 'breakout': 0.7, 'momentum': 0.7, 'mean_reversion': 0.2}
        elif regime == 'bull_weakening':
            strategies = {'trend_following': 0.5, 'breakout': 0.3, 'momentum': 0.4, 'mean_reversion': 0.4}
        elif regime == 'bear_weakening':
            strategies = {'trend_following': 0.3, 'breakout': 0.2, 'momentum': 0.3, 'mean_reversion': 0.6}
        else:  # bear_confirmed
            strategies = {'trend_following': 0.2, 'breakout': 0.1, 'momentum': 0.2, 'mean_reversion': 0.7}

        # Store in database
        upsert_query = """
            INSERT INTO market_context (
                calculation_date, market_regime,
                spy_price, spy_sma50, spy_sma200,
                spy_vs_sma50_pct, spy_vs_sma200_pct, spy_trend,
                pct_above_sma200, pct_above_sma50, pct_above_sma20,
                new_highs_count, new_lows_count, high_low_ratio,
                advancing_count, declining_count, ad_ratio,
                avg_atr_pct, volatility_regime,
                recommended_strategies
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
            )
            ON CONFLICT (calculation_date)
            DO UPDATE SET
                market_regime = EXCLUDED.market_regime,
                spy_price = EXCLUDED.spy_price,
                spy_sma50 = EXCLUDED.spy_sma50,
                spy_sma200 = EXCLUDED.spy_sma200,
                spy_vs_sma50_pct = EXCLUDED.spy_vs_sma50_pct,
                spy_vs_sma200_pct = EXCLUDED.spy_vs_sma200_pct,
                spy_trend = EXCLUDED.spy_trend,
                pct_above_sma200 = EXCLUDED.pct_above_sma200,
                pct_above_sma50 = EXCLUDED.pct_above_sma50,
                pct_above_sma20 = EXCLUDED.pct_above_sma20,
                new_highs_count = EXCLUDED.new_highs_count,
                new_lows_count = EXCLUDED.new_lows_count,
                high_low_ratio = EXCLUDED.high_low_ratio,
                advancing_count = EXCLUDED.advancing_count,
                declining_count = EXCLUDED.declining_count,
                ad_ratio = EXCLUDED.ad_ratio,
                avg_atr_pct = EXCLUDED.avg_atr_pct,
                volatility_regime = EXCLUDED.volatility_regime,
                recommended_strategies = EXCLUDED.recommended_strategies
        """

        await self.db.execute(
            upsert_query,
            calculation_date, regime,
            spy_data.get('price'), spy_data.get('sma50'), spy_data.get('sma200'),
            spy_data.get('vs_sma50_pct'), spy_data.get('vs_sma200_pct'), spy_data.get('trend'),
            breadth.get('pct_above_sma200'), breadth.get('pct_above_sma50'), breadth.get('pct_above_sma20'),
            breadth.get('new_highs'), breadth.get('new_lows'), breadth.get('hl_ratio'),
            breadth.get('advancing'), breadth.get('declining'), breadth.get('ad_ratio'),
            vol_regime.get('avg_atr'), vol_regime.get('regime'),
            json.dumps(strategies)
        )

        logger.info(f"Market Regime: {regime}")
        logger.info(f"  SPY vs SMA200: {spy_data.get('vs_sma200_pct', 0):.1f}%")
        logger.info(f"  % Above SMA200: {breadth.get('pct_above_sma200', 0):.1f}%")
        logger.info(f"  Volatility: {vol_regime.get('regime', 'unknown')}")

        return {
            'regime': regime,
            'spy_data': spy_data,
            'breadth': breadth,
            'volatility': vol_regime,
            'strategies': strategies
        }

    async def _get_spy_metrics(self, calc_date: date) -> Optional[Dict]:
        """Get SPY price and moving averages."""
        # Try synthesized candles first, then raw candles
        for table in ['stock_candles_synthesized', 'stock_candles']:
            if table == 'stock_candles_synthesized':
                query = """
                    SELECT close
                    FROM stock_candles_synthesized
                    WHERE ticker = 'SPY' AND timeframe = '1d'
                      AND DATE(timestamp) <= $1
                    ORDER BY timestamp DESC
                    LIMIT 220
                """
            else:
                query = """
                    SELECT close
                    FROM stock_candles
                    WHERE ticker = 'SPY'
                      AND timeframe = '1d'
                      AND DATE(timestamp) <= $1
                    ORDER BY timestamp DESC
                    LIMIT 220
                """
            rows = await self.db.fetch(query, calc_date)
            if len(rows) >= 200:
                break

        # If no database data, try yfinance
        if len(rows) < 200:
            logger.info("SPY data not in database, fetching from yfinance...")
            try:
                import yfinance as yf
                spy = yf.Ticker("SPY")
                hist = spy.history(period="1y")
                if len(hist) >= 200:
                    closes = hist['Close'].values.tolist()
                    # Convert to list of dicts format
                    rows = [{'close': c} for c in reversed(closes)]
            except Exception as e:
                logger.warning(f"Failed to fetch SPY from yfinance: {e}")
                return None

        if len(rows) < 200:
            return None

        closes = [float(r['close']) for r in rows]
        price = closes[0]
        sma50 = np.mean(closes[:50])
        sma200 = np.mean(closes[:200])

        # Check trend (is SMA200 rising or falling over last 20 days?)
        sma200_20_ago = np.mean(closes[20:220])
        if sma200 > sma200_20_ago * 1.005:
            trend = 'rising'
        elif sma200 < sma200_20_ago * 0.995:
            trend = 'falling'
        else:
            trend = 'flat'

        return {
            'price': round(price, 2),
            'sma50': round(sma50, 2),
            'sma200': round(sma200, 2),
            'vs_sma50_pct': round(((price / sma50) - 1) * 100, 2),
            'vs_sma200_pct': round(((price / sma200) - 1) * 100, 2),
            'trend': trend
        }

    async def _calculate_breadth(self, calc_date: date) -> Dict:
        """Calculate market breadth indicators."""
        query = """
            SELECT
                COUNT(*) FILTER (WHERE price_vs_sma200 > 0) as above_sma200,
                COUNT(*) FILTER (WHERE price_vs_sma50 > 0) as above_sma50,
                COUNT(*) FILTER (WHERE price_vs_sma20 > 0) as above_sma20,
                COUNT(*) FILTER (WHERE high_low_signal = 'new_high') as new_highs,
                COUNT(*) FILTER (WHERE high_low_signal = 'new_low') as new_lows,
                COUNT(*) FILTER (WHERE price_change_1d > 0) as advancing,
                COUNT(*) FILTER (WHERE price_change_1d < 0) as declining,
                COUNT(*) as total
            FROM stock_screening_metrics
            WHERE calculation_date = $1
        """
        rows = await self.db.fetch(query, calc_date)

        if not rows:
            return {}

        r = rows[0]
        total = r['total'] or 1

        return {
            'pct_above_sma200': round((r['above_sma200'] / total) * 100, 1),
            'pct_above_sma50': round((r['above_sma50'] / total) * 100, 1),
            'pct_above_sma20': round((r['above_sma20'] / total) * 100, 1),
            'new_highs': r['new_highs'],
            'new_lows': r['new_lows'],
            'hl_ratio': round(r['new_highs'] / max(r['new_lows'], 1), 2),
            'advancing': r['advancing'],
            'declining': r['declining'],
            'ad_ratio': round(r['advancing'] / max(r['declining'], 1), 2),
            'total': total
        }

    async def _calculate_volatility_regime(self, calc_date: date) -> Dict:
        """Calculate average ATR% and classify volatility regime."""
        query = """
            SELECT AVG(atr_percent) as avg_atr
            FROM stock_screening_metrics
            WHERE calculation_date = $1
              AND atr_percent IS NOT NULL
        """
        rows = await self.db.fetch(query, calc_date)

        avg_atr = float(rows[0]['avg_atr']) if rows and rows[0]['avg_atr'] else 3.0
        if np.isnan(avg_atr):
            avg_atr = 3.0

        # Classify based on historical norms
        if avg_atr < 2.0:
            regime = 'low'
        elif avg_atr < 4.0:
            regime = 'normal'
        elif avg_atr < 6.0:
            regime = 'high'
        else:
            regime = 'extreme'

        return {
            'avg_atr': round(avg_atr, 2),
            'regime': regime
        }

    def _classify_regime(self, spy_data: Dict, breadth: Dict) -> str:
        """
        Classify market regime based on SPY and breadth.

        Rules:
        - Bull Confirmed: SPY > SMA200, trend rising, breadth healthy
        - Bull Weakening: SPY > SMA200 but trend flat/falling or breadth weak
        - Bear Weakening: SPY < SMA200 but trend rising or breadth improving
        - Bear Confirmed: SPY < SMA200, trend falling, breadth weak
        """
        above_sma200 = spy_data['vs_sma200_pct'] > 0
        trend = spy_data['trend']
        pct_healthy = breadth.get('pct_above_sma200', 0) > 50

        if above_sma200:
            if trend == 'rising' and pct_healthy:
                return 'bull_confirmed'
            else:
                return 'bull_weakening'
        else:
            if trend in ('rising', 'flat') or pct_healthy:
                return 'bear_weakening'
            else:
                return 'bear_confirmed'
