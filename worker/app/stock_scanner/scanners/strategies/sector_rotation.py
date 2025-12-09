"""
Sector Rotation Scanner

Identifies stocks in strong/weak sectors for rotation plays.

Strategy Logic:
1. Analyze sector performance across multiple timeframes
2. Identify leading sectors (money flowing in)
3. Identify lagging sectors (money flowing out)
4. Find best stocks within strong sectors (leaders)
5. Find short candidates in weak sectors (laggards)

Sector Rotation Theory:
- Money flows between sectors based on economic cycle
- Leading indicators: Technology, Consumer Discretionary
- Defensive sectors: Utilities, Healthcare, Staples
- Late cycle: Energy, Materials
- Track relative strength to identify rotation
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ..base_scanner import (
    BaseScanner, SignalSetup, SignalType, QualityTier, ScannerConfig
)

logger = logging.getLogger(__name__)


class SectorRotationConfig(ScannerConfig):
    """Configuration specific to Sector Rotation Scanner"""

    # Sector analysis thresholds
    top_sectors_count: int = 3         # Number of top sectors to scan
    bottom_sectors_count: int = 2      # Number of bottom sectors for shorts

    # Leader/laggard thresholds
    min_sector_outperformance: float = 2.0  # % above market average
    min_stock_outperformance: float = 3.0   # % above sector average

    # Technical requirements
    require_trend_alignment: bool = True
    min_relative_strength: float = 1.1  # RS vs sector

    # Volume requirements
    min_relative_volume: float = 0.8

    # Risk parameters
    atr_stop_multiplier: float = 1.5
    tp1_rr_ratio: float = 2.0
    tp2_rr_ratio: float = 3.0

    # Allow short signals for sector laggards
    allow_sell_signals: bool = True


class SectorRotationScanner(BaseScanner):
    """
    Sector Rotation Scanner - Captures sector momentum.

    Identifies:
    1. Strong sectors with money inflow
    2. Leaders within strong sectors
    3. Weak sectors with money outflow
    4. Laggards for potential shorts

    Scoring:
    - Sector strength relative to market
    - Stock strength relative to sector
    - Technical trend alignment
    - Volume confirmation
    """

    # Sector groupings for analysis
    SECTORS = [
        'Technology', 'Healthcare', 'Financials', 'Consumer Cyclical',
        'Consumer Defensive', 'Industrials', 'Energy', 'Utilities',
        'Real Estate', 'Basic Materials', 'Communication Services'
    ]

    def __init__(self, db_manager, config: SectorRotationConfig = None, scorer=None):
        super().__init__(db_manager, config or SectorRotationConfig(), scorer)
        self._sector_performance: Dict[str, Dict] = {}

    @property
    def scanner_name(self) -> str:
        return "sector_rotation"

    @property
    def description(self) -> str:
        return "Sector leaders and laggards for rotation plays"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Scan for sector rotation opportunities.

        Args:
            calculation_date: Date to scan

        Returns:
            List of SignalSetup objects
        """
        logger.info("=" * 60)
        logger.info("SECTOR ROTATION SCANNER")
        logger.info("=" * 60)

        if calculation_date is None:
            calculation_date = datetime.now().date()

        # Step 1: Calculate sector performance
        self._sector_performance = await self._calculate_sector_performance(calculation_date)
        logger.info(f"Analyzed {len(self._sector_performance)} sectors")

        # Step 2: Identify top and bottom sectors
        ranked_sectors = self._rank_sectors()
        top_sectors = ranked_sectors[:self.config.top_sectors_count]
        bottom_sectors = ranked_sectors[-self.config.bottom_sectors_count:]

        logger.info(f"Top sectors: {[s[0] for s in top_sectors]}")
        logger.info(f"Bottom sectors: {[s[0] for s in bottom_sectors]}")

        signals = []

        # Step 3: Find leaders in top sectors (BUY)
        for sector, _ in top_sectors:
            sector_signals = await self._find_sector_leaders(
                sector, calculation_date, SignalType.BUY
            )
            signals.extend(sector_signals)

        # Step 4: Find laggards in bottom sectors (SELL) if enabled
        if self.config.allow_sell_signals:
            for sector, _ in bottom_sectors:
                sector_signals = await self._find_sector_laggards(
                    sector, calculation_date, SignalType.SELL
                )
                signals.extend(sector_signals)

        # Sort by score and limit
        signals.sort(key=lambda x: x.composite_score, reverse=True)
        signals = signals[:self.config.max_signals_per_run]

        # Log summary
        buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        sell_count = len(signals) - buy_count
        high_quality = sum(1 for s in signals if s.is_high_quality)
        logger.info(f"Generated {len(signals)} signals (BUY: {buy_count}, SELL: {sell_count})")
        self.log_scan_summary(len(signals), len(signals), high_quality)

        return signals

    async def _calculate_sector_performance(
        self,
        calculation_date: datetime
    ) -> Dict[str, Dict]:
        """
        Calculate performance metrics for each sector.

        Returns dict with sector performance data:
        - avg_1d_change, avg_5d_change, avg_20d_change
        - stock_count
        - relative_strength vs market
        """
        query = """
            SELECT
                i.sector,
                COUNT(*) as stock_count,
                AVG(m.price_change_1d) as avg_1d_change,
                AVG(m.price_change_5d) as avg_5d_change,
                AVG(m.price_change_20d) as avg_20d_change,
                AVG(w.relative_volume) as avg_volume
            FROM stock_instruments i
            JOIN stock_screening_metrics m ON i.ticker = m.ticker
            JOIN stock_watchlist w ON i.ticker = w.ticker
                AND w.calculation_date = m.calculation_date
            WHERE m.calculation_date = $1
              AND i.sector IS NOT NULL
              AND i.sector != ''
              AND m.data_quality = 'good'
            GROUP BY i.sector
            HAVING COUNT(*) >= 5
            ORDER BY AVG(m.price_change_5d) DESC
        """

        rows = await self.db.fetch(query, calculation_date)

        # Calculate market average
        all_changes = [float(r['avg_5d_change'] or 0) for r in rows]
        market_avg = sum(all_changes) / len(all_changes) if all_changes else 0

        sector_data = {}
        for row in rows:
            sector = row['sector']
            avg_5d = float(row['avg_5d_change'] or 0)

            sector_data[sector] = {
                'stock_count': row['stock_count'],
                'avg_1d_change': float(row['avg_1d_change'] or 0),
                'avg_5d_change': avg_5d,
                'avg_20d_change': float(row['avg_20d_change'] or 0),
                'avg_volume': float(row['avg_volume'] or 1),
                'relative_strength': avg_5d - market_avg,
                'market_avg': market_avg
            }

        return sector_data

    def _rank_sectors(self) -> List[Tuple[str, float]]:
        """
        Rank sectors by relative strength.

        Returns list of (sector_name, score) tuples, sorted by score desc.
        """
        rankings = []

        for sector, data in self._sector_performance.items():
            # Weighted score: 50% 5d change, 30% 20d change, 20% 1d change
            score = (
                data['avg_5d_change'] * 0.5 +
                data['avg_20d_change'] * 0.3 +
                data['avg_1d_change'] * 0.2
            )
            rankings.append((sector, score))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    async def _find_sector_leaders(
        self,
        sector: str,
        calculation_date: datetime,
        signal_type: SignalType
    ) -> List[SignalSetup]:
        """
        Find leading stocks within a strong sector.

        Leaders have:
        - Outperforming sector average
        - Strong technical trend
        - Good volume
        """
        config = self.config
        sector_data = self._sector_performance.get(sector, {})
        sector_avg = sector_data.get('avg_5d_change', 0)

        query = """
            SELECT
                w.ticker,
                w.tier,
                w.score as watchlist_score,
                w.current_price,
                w.atr_percent,
                w.relative_volume,
                w.trend_strength,
                w.rsi_signal,
                w.sma_cross_signal,
                w.macd_cross_signal,
                w.candlestick_pattern,
                w.pct_from_52w_high,
                m.price_change_1d,
                m.price_change_5d,
                m.price_change_20d,
                m.rsi_14,
                m.macd_histogram,
                m.atr_14,
                m.sma_20,
                m.sma_50,
                m.sma_200,
                m.avg_volume_20,
                i.name,
                i.sector,
                i.market_cap,
                i.analyst_rating
            FROM stock_watchlist w
            JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            JOIN stock_instruments i ON w.ticker = i.ticker
            WHERE w.calculation_date = $1
              AND i.sector = $2
              AND w.tier <= $3
              AND m.price_change_5d > $4
              AND m.data_quality = 'good'
              AND w.relative_volume >= $5
            ORDER BY m.price_change_5d DESC
            LIMIT 20
        """

        # Leaders must outperform sector
        min_change = sector_avg + config.min_stock_outperformance

        rows = await self.db.fetch(
            query,
            calculation_date,
            sector,
            config.max_tier,
            min_change,
            config.min_relative_volume
        )

        signals = []
        for row in rows:
            candidate = dict(row)
            signal = self._create_signal(candidate, signal_type, sector_data)
            if signal and signal.composite_score >= config.min_score_threshold:
                signals.append(signal)

        return signals

    async def _find_sector_laggards(
        self,
        sector: str,
        calculation_date: datetime,
        signal_type: SignalType
    ) -> List[SignalSetup]:
        """
        Find lagging stocks within a weak sector for shorts.

        Laggards have:
        - Underperforming sector average
        - Weak technical trend
        - Breaking down
        """
        config = self.config
        sector_data = self._sector_performance.get(sector, {})
        sector_avg = sector_data.get('avg_5d_change', 0)

        query = """
            SELECT
                w.ticker,
                w.tier,
                w.score as watchlist_score,
                w.current_price,
                w.atr_percent,
                w.relative_volume,
                w.trend_strength,
                w.rsi_signal,
                w.sma_cross_signal,
                w.macd_cross_signal,
                w.candlestick_pattern,
                w.pct_from_52w_high,
                m.price_change_1d,
                m.price_change_5d,
                m.price_change_20d,
                m.rsi_14,
                m.macd_histogram,
                m.atr_14,
                m.sma_20,
                m.sma_50,
                m.sma_200,
                m.avg_volume_20,
                i.name,
                i.sector,
                i.market_cap,
                i.analyst_rating
            FROM stock_watchlist w
            JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            JOIN stock_instruments i ON w.ticker = i.ticker
            WHERE w.calculation_date = $1
              AND i.sector = $2
              AND w.tier <= $3
              AND m.price_change_5d < $4
              AND m.data_quality = 'good'
            ORDER BY m.price_change_5d ASC
            LIMIT 10
        """

        # Laggards must underperform sector
        max_change = sector_avg - config.min_stock_outperformance

        rows = await self.db.fetch(
            query,
            calculation_date,
            sector,
            config.max_tier,
            max_change
        )

        signals = []
        for row in rows:
            candidate = dict(row)
            signal = self._create_signal(candidate, signal_type, sector_data)
            if signal and signal.composite_score >= config.min_score_threshold:
                signals.append(signal)

        return signals

    def _create_signal(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType,
        sector_data: Dict[str, Any]
    ) -> Optional[SignalSetup]:
        """Create signal from candidate."""
        ticker = candidate['ticker']
        config = self.config

        # Check trend alignment if required
        if config.require_trend_alignment:
            sma_signal = candidate.get('sma_cross_signal', '')
            if signal_type == SignalType.BUY and sma_signal in ['death_cross', 'bearish']:
                return None
            if signal_type == SignalType.SELL and sma_signal in ['golden_cross', 'bullish']:
                return None

        # Calculate scores
        sector_score = self._calculate_sector_score(candidate, sector_data, signal_type)
        momentum_score = self._calculate_momentum_score(candidate, signal_type)
        technical_score = self._calculate_technical_score(candidate, signal_type)
        volume_score = self._calculate_volume_score(candidate)
        confluence_score = self._calculate_confluence_score(candidate, signal_type)

        composite_score = int(
            sector_score + momentum_score + technical_score +
            volume_score + confluence_score
        )

        # Build signal
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, signal_type)
        risk_pct = abs(entry - stop) / entry * 100

        # Build confluence factors
        sector = candidate.get('sector', 'Unknown')
        confluence_factors = self._build_rotation_confluence(candidate, sector_data, signal_type)

        quality_tier = QualityTier.from_score(composite_score)
        position_size = self.calculate_position_size(
            Decimal(str(config.max_risk_per_trade_pct)),
            entry, stop, quality_tier
        )

        direction = "Leader" if signal_type == SignalType.BUY else "Laggard"
        return SignalSetup(
            ticker=ticker,
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            signal_timestamp=datetime.now(),
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=Decimal(str(config.tp1_rr_ratio)),
            risk_percent=Decimal(str(risk_pct)),
            composite_score=composite_score,
            quality_tier=quality_tier,
            trend_score=Decimal(str(technical_score)),
            momentum_score=Decimal(str(momentum_score)),
            volume_score=Decimal(str(volume_score)),
            pattern_score=Decimal(str(sector_score)),
            confluence_score=Decimal(str(confluence_score)),
            setup_description=f"Sector {direction}: {sector}",
            confluence_factors=confluence_factors,
            timeframe="daily",
            market_regime=candidate.get('trend_strength', ''),
            suggested_position_size_pct=position_size,
            max_risk_per_trade_pct=Decimal(str(config.max_risk_per_trade_pct)),
            raw_data=candidate
        )

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """Calculate entry, stop, and take-profit levels."""
        price = Decimal(str(candidate['current_price']))
        atr = Decimal(str(candidate.get('atr_14', 0) or candidate['current_price'] * 0.02))

        entry = price
        stop = self.calculate_atr_based_stop(entry, atr, signal_type)
        tp1, tp2 = self.calculate_take_profits(entry, stop, signal_type)

        return entry, stop, tp1, tp2

    def _calculate_sector_score(
        self,
        candidate: Dict[str, Any],
        sector_data: Dict[str, Any],
        signal_type: SignalType
    ) -> float:
        """Score sector positioning (0-25 points)."""
        score = 0

        sector_rs = sector_data.get('relative_strength', 0)
        stock_change_5d = float(candidate.get('price_change_5d', 0) or 0)
        sector_avg = sector_data.get('avg_5d_change', 0)

        # Sector strength (0-15)
        if signal_type == SignalType.BUY:
            if sector_rs > 5:
                score += 15
            elif sector_rs > 2:
                score += 10
            elif sector_rs > 0:
                score += 5
        else:  # SELL
            if sector_rs < -5:
                score += 15
            elif sector_rs < -2:
                score += 10
            elif sector_rs < 0:
                score += 5

        # Stock vs sector outperformance (0-10)
        stock_outperformance = stock_change_5d - sector_avg
        if signal_type == SignalType.BUY:
            if stock_outperformance > 5:
                score += 10
            elif stock_outperformance > 3:
                score += 7
            elif stock_outperformance > 0:
                score += 4
        else:
            if stock_outperformance < -5:
                score += 10
            elif stock_outperformance < -3:
                score += 7
            elif stock_outperformance < 0:
                score += 4

        return min(25, score)

    def _calculate_momentum_score(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> float:
        """Score momentum (0-25 points)."""
        score = 0

        change_1d = float(candidate.get('price_change_1d', 0) or 0)
        change_5d = float(candidate.get('price_change_5d', 0) or 0)
        change_20d = float(candidate.get('price_change_20d', 0) or 0)

        if signal_type == SignalType.BUY:
            # Recent momentum (0-10)
            if change_5d > 5:
                score += 10
            elif change_5d > 2:
                score += 7
            elif change_5d > 0:
                score += 3

            # Longer term trend (0-10)
            if change_20d > 10:
                score += 10
            elif change_20d > 5:
                score += 7
            elif change_20d > 0:
                score += 3

            # Daily continuation (0-5)
            if change_1d > 0:
                score += 5
        else:  # SELL
            if change_5d < -5:
                score += 10
            elif change_5d < -2:
                score += 7
            elif change_5d < 0:
                score += 3

            if change_20d < -10:
                score += 10
            elif change_20d < -5:
                score += 7
            elif change_20d < 0:
                score += 3

            if change_1d < 0:
                score += 5

        return min(25, score)

    def _calculate_technical_score(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> float:
        """Score technical alignment (0-15 points)."""
        score = 0

        sma_signal = candidate.get('sma_cross_signal', '')
        macd_signal = candidate.get('macd_cross_signal', '')
        rsi_signal = candidate.get('rsi_signal', '')

        if signal_type == SignalType.BUY:
            # SMA (0-5)
            if sma_signal in ['golden_cross', 'bullish']:
                score += 5
            # MACD (0-5)
            if macd_signal in ['bullish_cross', 'bullish']:
                score += 5
            # RSI not overbought (0-5)
            if rsi_signal not in ['overbought', 'overbought_extreme']:
                score += 5
        else:  # SELL
            if sma_signal in ['death_cross', 'bearish']:
                score += 5
            if macd_signal in ['bearish_cross', 'bearish']:
                score += 5
            if rsi_signal not in ['oversold', 'oversold_extreme']:
                score += 5

        return min(15, score)

    def _calculate_volume_score(self, candidate: Dict[str, Any]) -> float:
        """Score volume confirmation (0-20 points)."""
        score = 0

        rel_vol = float(candidate.get('relative_volume', 1) or 1)

        if rel_vol >= 2.0:
            score += 15
        elif rel_vol >= 1.5:
            score += 12
        elif rel_vol >= 1.0:
            score += 8
        elif rel_vol >= 0.8:
            score += 4

        # Tier bonus
        tier = candidate.get('tier', 4)
        if tier == 1:
            score += 5
        elif tier == 2:
            score += 3

        return min(20, score)

    def _calculate_confluence_score(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> float:
        """Score additional confluence (0-15 points)."""
        score = 0

        # Watchlist score (0-5)
        watchlist_score = float(candidate.get('watchlist_score', 50) or 50)
        if watchlist_score >= 80:
            score += 5
        elif watchlist_score >= 70:
            score += 3
        elif watchlist_score >= 60:
            score += 1

        # Market cap / quality (0-5)
        market_cap = candidate.get('market_cap', 0) or 0
        if market_cap > 10_000_000_000:  # > $10B
            score += 5
        elif market_cap > 1_000_000_000:  # > $1B
            score += 3

        # Analyst rating (0-5)
        rating = candidate.get('analyst_rating', '')
        if signal_type == SignalType.BUY and rating and 'buy' in str(rating).lower():
            score += 5
        elif signal_type == SignalType.SELL and rating and 'sell' in str(rating).lower():
            score += 5

        return min(15, score)

    def _build_rotation_confluence(
        self,
        candidate: Dict[str, Any],
        sector_data: Dict[str, Any],
        signal_type: SignalType
    ) -> List[str]:
        """Build confluence factors for rotation play."""
        factors = []

        sector = candidate.get('sector', 'Unknown')
        sector_rs = sector_data.get('relative_strength', 0)

        # Core rotation factor
        if signal_type == SignalType.BUY:
            factors.append(f"Strong Sector: {sector} (+{sector_rs:.1f}% vs market)")
        else:
            factors.append(f"Weak Sector: {sector} ({sector_rs:.1f}% vs market)")

        # Stock vs sector
        stock_change = float(candidate.get('price_change_5d', 0) or 0)
        sector_avg = sector_data.get('avg_5d_change', 0)
        outperformance = stock_change - sector_avg

        if signal_type == SignalType.BUY and outperformance > 0:
            factors.append(f"Sector Leader: +{outperformance:.1f}% vs sector")
        elif signal_type == SignalType.SELL and outperformance < 0:
            factors.append(f"Sector Laggard: {outperformance:.1f}% vs sector")

        # Add standard technical factors
        factors.extend(self.build_confluence_factors(candidate))

        return factors
