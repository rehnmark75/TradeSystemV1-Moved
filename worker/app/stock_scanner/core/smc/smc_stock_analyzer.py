"""
SMC Stock Analyzer

Smart Money Concepts analysis for stocks on daily timeframe.
Provides market structure analysis including:
- Swing point detection (HH, HL, LH, LL)
- Break of Structure (BOS) detection
- Change of Character (CHoCH) detection
- Premium/Discount zone calculation
- Order Block identification

Simplified from forex SMC for daily stock analysis.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SwingType(Enum):
    """Type of swing point."""
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    EQUAL_HIGH = "EQH"
    EQUAL_LOW = "EQL"


class StructureType(Enum):
    """Market structure type."""
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


class ZoneType(Enum):
    """Premium/Discount zone classification."""
    EXTREME_PREMIUM = "Extreme Premium"
    PREMIUM = "Premium"
    EQUILIBRIUM_HIGH = "Equilibrium High"
    EQUILIBRIUM = "Equilibrium"
    EQUILIBRIUM_LOW = "Equilibrium Low"
    DISCOUNT = "Discount"
    EXTREME_DISCOUNT = "Extreme Discount"


@dataclass
class SwingPoint:
    """A detected swing point."""
    index: int
    price: float
    timestamp: datetime
    swing_type: SwingType
    is_high: bool
    strength: float = 0.0


@dataclass
class StructureBreak:
    """A structure break (BOS or CHoCH)."""
    index: int
    price: float
    timestamp: datetime
    break_type: str  # 'BOS' or 'CHoCH'
    direction: str   # 'Bullish' or 'Bearish'
    significance: float = 0.0


@dataclass
class OrderBlock:
    """An institutional order block."""
    start_index: int
    end_index: int
    top: float
    bottom: float
    ob_type: str  # 'Bullish' or 'Bearish'
    timestamp: datetime
    strength: float = 0.0
    is_valid: bool = True


@dataclass
class SMCAnalysis:
    """Complete SMC analysis result for a stock."""
    ticker: str
    calculation_date: datetime

    # Market Structure
    smc_trend: str = "Neutral"
    smc_bias: str = "Neutral"

    # Last BOS
    last_bos_type: Optional[str] = None
    last_bos_date: Optional[datetime] = None
    last_bos_price: Optional[float] = None

    # Last CHoCH
    last_choch_type: Optional[str] = None
    last_choch_date: Optional[datetime] = None

    # Current Swings
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    swing_high_date: Optional[datetime] = None
    swing_low_date: Optional[datetime] = None

    # Premium/Discount
    premium_discount_zone: str = "Equilibrium"
    zone_position: float = 50.0
    weekly_range_high: Optional[float] = None
    weekly_range_low: Optional[float] = None

    # Order Block
    nearest_ob_type: Optional[str] = None
    nearest_ob_price: Optional[float] = None
    nearest_ob_distance: Optional[float] = None

    # Confluence Score
    smc_confluence_score: float = 50.0


class SMCStockConfig:
    """Configuration for SMC stock analysis."""

    def __init__(self):
        # Swing Detection
        self.swing_length = 5  # Bars for swing confirmation
        self.min_swing_bars = 2  # Minimum bars for progressive confirmation
        self.equal_level_tolerance = 0.002  # 0.2% tolerance for equal levels

        # Structure Detection
        self.structure_confirmation = 2  # Bars to confirm structure
        self.bos_threshold = 0.005  # 0.5% minimum move for BOS
        self.min_structure_significance = 0.3

        # Order Blocks
        self.order_block_length = 3  # Consolidation bars
        self.order_block_volume_factor = 1.5  # Volume threshold
        self.max_order_blocks = 3  # Max OBs to track
        self.order_block_max_age = 20  # Days

        # Premium/Discount
        self.use_weekly_context = True
        self.weekly_lookback = 5  # Weeks for context

        # Analysis Depth
        self.lookback_days = 60  # Days of data to analyze


class SMCStockAnalyzer:
    """
    Smart Money Concepts analyzer for stocks.

    Analyzes daily stock data to identify:
    - Market structure (bullish/bearish trends)
    - Key swing points and structure breaks
    - Premium/discount zones for entry timing
    - Order blocks for institutional interest
    """

    def __init__(self, db_manager, config: SMCStockConfig = None):
        self.db = db_manager
        self.config = config or SMCStockConfig()

    async def analyze_all_stocks(
        self,
        calculation_date: datetime = None,
        tickers: List[str] = None
    ) -> Dict[str, SMCAnalysis]:
        """
        Analyze all stocks (or specified tickers) for SMC patterns.

        Args:
            calculation_date: Date to analyze (defaults to yesterday)
            tickers: Optional list of tickers to analyze

        Returns:
            Dictionary mapping ticker to SMCAnalysis
        """
        logger.info("=" * 60)
        logger.info("SMC STOCK ANALYSIS")
        logger.info("=" * 60)

        start_time = datetime.now()

        if calculation_date is None:
            calculation_date = datetime.now().date() - timedelta(days=1)

        # Get tickers to analyze
        if tickers is None:
            tickers = await self._get_active_tickers()

        logger.info(f"Analyzing {len(tickers)} stocks for SMC patterns")

        results = {}
        successful = 0
        failed = 0

        for ticker in tickers:
            try:
                analysis = await self.analyze_ticker(ticker, calculation_date)
                if analysis:
                    results[ticker] = analysis
                    successful += 1
            except Exception as e:
                logger.warning(f"Failed to analyze {ticker}: {e}")
                failed += 1

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(f"\nSMC Analysis complete:")
        logger.info(f"  Successful: {successful}/{len(tickers)}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Duration: {elapsed:.2f}s")

        # Summary statistics
        bullish = sum(1 for a in results.values() if a.smc_trend == "Bullish")
        bearish = sum(1 for a in results.values() if a.smc_trend == "Bearish")
        neutral = sum(1 for a in results.values() if a.smc_trend == "Neutral")
        logger.info(f"  Trends: Bullish={bullish}, Bearish={bearish}, Neutral={neutral}")

        return results

    async def analyze_ticker(
        self,
        ticker: str,
        calculation_date: datetime = None
    ) -> Optional[SMCAnalysis]:
        """
        Analyze a single ticker for SMC patterns.

        Args:
            ticker: Stock ticker symbol
            calculation_date: Date to analyze

        Returns:
            SMCAnalysis object or None if insufficient data
        """
        if calculation_date is None:
            calculation_date = datetime.now().date() - timedelta(days=1)

        # Fetch daily candles
        candles = await self._get_daily_candles(ticker, self.config.lookback_days)

        if len(candles) < 30:  # Need at least 30 days
            return None

        # Convert to numpy arrays
        opens = np.array([float(c['open']) for c in candles])
        highs = np.array([float(c['high']) for c in candles])
        lows = np.array([float(c['low']) for c in candles])
        closes = np.array([float(c['close']) for c in candles])
        volumes = np.array([float(c['volume']) for c in candles])
        timestamps = [c['timestamp'] for c in candles]

        current_price = closes[-1]

        # Detect swing points
        swing_highs, swing_lows = self._detect_swings(highs, lows, timestamps)

        # Classify swings
        classified_swings = self._classify_swings(swing_highs, swing_lows)

        # Detect structure breaks
        structure_breaks = self._detect_structure_breaks(classified_swings, highs, lows, timestamps)

        # Determine market structure/trend
        smc_trend, smc_bias = self._determine_trend(classified_swings, structure_breaks)

        # Find last BOS and CHoCH
        last_bos = self._get_last_break(structure_breaks, 'BOS')
        last_choch = self._get_last_break(structure_breaks, 'CHoCH')

        # Get current swing levels
        recent_high, recent_low = self._get_recent_swing_levels(swing_highs, swing_lows)

        # Calculate premium/discount zone
        zone, zone_position, weekly_high, weekly_low = self._calculate_zones(
            current_price, highs, lows, timestamps
        )

        # Detect order blocks
        order_blocks = self._detect_order_blocks(
            opens, highs, lows, closes, volumes, timestamps
        )

        # Find nearest order block
        nearest_ob = self._get_nearest_order_block(order_blocks, current_price)

        # Calculate confluence score
        confluence_score = self._calculate_confluence_score(
            smc_trend, smc_bias, zone, nearest_ob, last_bos, last_choch
        )

        return SMCAnalysis(
            ticker=ticker,
            calculation_date=calculation_date,
            smc_trend=smc_trend,
            smc_bias=smc_bias,
            last_bos_type=last_bos.direction if last_bos else None,
            last_bos_date=last_bos.timestamp if last_bos else None,
            last_bos_price=last_bos.price if last_bos else None,
            last_choch_type=last_choch.direction if last_choch else None,
            last_choch_date=last_choch.timestamp if last_choch else None,
            swing_high=recent_high.price if recent_high else None,
            swing_low=recent_low.price if recent_low else None,
            swing_high_date=recent_high.timestamp if recent_high else None,
            swing_low_date=recent_low.timestamp if recent_low else None,
            premium_discount_zone=zone,
            zone_position=zone_position,
            weekly_range_high=weekly_high,
            weekly_range_low=weekly_low,
            nearest_ob_type=nearest_ob.ob_type if nearest_ob else None,
            nearest_ob_price=(nearest_ob.top + nearest_ob.bottom) / 2 if nearest_ob else None,
            nearest_ob_distance=self._calculate_ob_distance(nearest_ob, current_price) if nearest_ob else None,
            smc_confluence_score=confluence_score
        )

    def _detect_swings(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        timestamps: List[datetime]
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Detect swing highs and lows using pivot analysis."""
        swing_length = self.config.swing_length
        swing_highs = []
        swing_lows = []

        for i in range(swing_length, len(highs) - swing_length):
            # Check for swing high
            left_highs = highs[i - swing_length:i]
            right_highs = highs[i + 1:i + swing_length + 1]

            if highs[i] > left_highs.max() and highs[i] > right_highs.max():
                strength = self._calculate_swing_strength(highs, i, swing_length)
                swing_highs.append(SwingPoint(
                    index=i,
                    price=highs[i],
                    timestamp=timestamps[i],
                    swing_type=SwingType.HIGHER_HIGH,  # Will be reclassified
                    is_high=True,
                    strength=strength
                ))

            # Check for swing low
            left_lows = lows[i - swing_length:i]
            right_lows = lows[i + 1:i + swing_length + 1]

            if lows[i] < left_lows.min() and lows[i] < right_lows.min():
                strength = self._calculate_swing_strength(lows, i, swing_length)
                swing_lows.append(SwingPoint(
                    index=i,
                    price=lows[i],
                    timestamp=timestamps[i],
                    swing_type=SwingType.LOWER_LOW,  # Will be reclassified
                    is_high=False,
                    strength=strength
                ))

        return swing_highs, swing_lows

    def _calculate_swing_strength(
        self,
        prices: np.ndarray,
        index: int,
        swing_length: int
    ) -> float:
        """Calculate strength of a swing point based on price extension."""
        if index < swing_length or index >= len(prices) - swing_length:
            return 0.5

        left_prices = prices[index - swing_length:index]
        right_prices = prices[index + 1:index + swing_length + 1]

        # Calculate extension from surrounding prices
        avg_surrounding = (left_prices.mean() + right_prices.mean()) / 2
        extension = abs(prices[index] - avg_surrounding) / avg_surrounding

        # Normalize to 0-1 range
        return min(1.0, extension * 20)  # 5% extension = 1.0 strength

    def _classify_swings(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> List[SwingPoint]:
        """Classify swings as HH, HL, LH, LL, EQH, EQL."""
        # Combine and sort by index
        all_swings = sorted(swing_highs + swing_lows, key=lambda x: x.index)

        if len(all_swings) < 2:
            return all_swings

        tolerance = self.config.equal_level_tolerance

        # Classify highs
        prev_high = None
        for swing in all_swings:
            if swing.is_high:
                if prev_high is None:
                    swing.swing_type = SwingType.HIGHER_HIGH
                else:
                    diff = (swing.price - prev_high.price) / prev_high.price
                    if abs(diff) <= tolerance:
                        swing.swing_type = SwingType.EQUAL_HIGH
                    elif diff > 0:
                        swing.swing_type = SwingType.HIGHER_HIGH
                    else:
                        swing.swing_type = SwingType.LOWER_HIGH
                prev_high = swing

        # Classify lows
        prev_low = None
        for swing in all_swings:
            if not swing.is_high:
                if prev_low is None:
                    swing.swing_type = SwingType.HIGHER_LOW
                else:
                    diff = (swing.price - prev_low.price) / prev_low.price
                    if abs(diff) <= tolerance:
                        swing.swing_type = SwingType.EQUAL_LOW
                    elif diff > 0:
                        swing.swing_type = SwingType.HIGHER_LOW
                    else:
                        swing.swing_type = SwingType.LOWER_LOW
                prev_low = swing

        return all_swings

    def _detect_structure_breaks(
        self,
        swings: List[SwingPoint],
        highs: np.ndarray,
        lows: np.ndarray,
        timestamps: List[datetime]
    ) -> List[StructureBreak]:
        """Detect BOS and CHoCH from classified swings."""
        breaks = []

        if len(swings) < 3:
            return breaks

        # Track previous structure direction
        prev_direction = None

        for i in range(2, len(swings)):
            current = swings[i]
            prev = swings[i - 1]
            prev_prev = swings[i - 2]

            # Determine current implied direction
            if current.is_high:
                if current.swing_type == SwingType.HIGHER_HIGH:
                    current_direction = "Bullish"
                elif current.swing_type == SwingType.LOWER_HIGH:
                    current_direction = "Bearish"
                else:
                    current_direction = prev_direction
            else:
                if current.swing_type == SwingType.LOWER_LOW:
                    current_direction = "Bearish"
                elif current.swing_type == SwingType.HIGHER_LOW:
                    current_direction = "Bullish"
                else:
                    current_direction = prev_direction

            if current_direction is None:
                continue

            # Calculate price move significance
            if prev_prev.price != 0:
                price_move = abs(current.price - prev_prev.price) / prev_prev.price
            else:
                price_move = 0

            if price_move < self.config.bos_threshold:
                prev_direction = current_direction
                continue

            # Detect BOS (continuation) or CHoCH (reversal)
            if prev_direction is None:
                # First structure break
                breaks.append(StructureBreak(
                    index=current.index,
                    price=current.price,
                    timestamp=current.timestamp,
                    break_type='BOS',
                    direction=current_direction,
                    significance=current.strength
                ))
            elif current_direction == prev_direction:
                # BOS - continuation
                breaks.append(StructureBreak(
                    index=current.index,
                    price=current.price,
                    timestamp=current.timestamp,
                    break_type='BOS',
                    direction=current_direction,
                    significance=current.strength
                ))
            else:
                # CHoCH - reversal
                breaks.append(StructureBreak(
                    index=current.index,
                    price=current.price,
                    timestamp=current.timestamp,
                    break_type='CHoCH',
                    direction=current_direction,
                    significance=current.strength * 1.5  # CHoCH is more significant
                ))

            prev_direction = current_direction

        return breaks

    def _determine_trend(
        self,
        swings: List[SwingPoint],
        breaks: List[StructureBreak]
    ) -> Tuple[str, str]:
        """Determine overall trend and current bias."""
        if not swings:
            return "Neutral", "Neutral"

        # Count recent swing types
        recent_swings = swings[-6:]  # Last 6 swings

        bullish_count = sum(1 for s in recent_swings
                          if s.swing_type in [SwingType.HIGHER_HIGH, SwingType.HIGHER_LOW])
        bearish_count = sum(1 for s in recent_swings
                          if s.swing_type in [SwingType.LOWER_HIGH, SwingType.LOWER_LOW])

        # Determine trend from swing pattern
        if bullish_count >= 4:
            trend = "Bullish"
        elif bearish_count >= 4:
            trend = "Bearish"
        else:
            trend = "Neutral"

        # Determine bias from recent breaks
        if breaks:
            last_break = breaks[-1]
            bias = last_break.direction

            # If CHoCH, it's a stronger signal
            if last_break.break_type == 'CHoCH':
                trend = bias  # CHoCH overrides swing-based trend
        else:
            bias = trend

        return trend, bias

    def _get_last_break(
        self,
        breaks: List[StructureBreak],
        break_type: str
    ) -> Optional[StructureBreak]:
        """Get the most recent break of specified type."""
        for b in reversed(breaks):
            if b.break_type == break_type:
                return b
        return None

    def _get_recent_swing_levels(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> Tuple[Optional[SwingPoint], Optional[SwingPoint]]:
        """Get most recent swing high and low."""
        recent_high = swing_highs[-1] if swing_highs else None
        recent_low = swing_lows[-1] if swing_lows else None
        return recent_high, recent_low

    def _calculate_zones(
        self,
        current_price: float,
        highs: np.ndarray,
        lows: np.ndarray,
        timestamps: List[datetime]
    ) -> Tuple[str, float, float, float]:
        """Calculate premium/discount zone and weekly context."""
        # Use recent range for zone calculation
        lookback = min(20, len(highs))
        range_high = highs[-lookback:].max()
        range_low = lows[-lookback:].min()

        # Weekly context (last 5 weeks = ~25 trading days)
        weekly_lookback = min(25, len(highs))
        weekly_high = highs[-weekly_lookback:].max()
        weekly_low = lows[-weekly_lookback:].min()

        # Calculate position in range
        if range_high == range_low:
            zone_position = 50.0
        else:
            zone_position = ((current_price - range_low) / (range_high - range_low)) * 100

        # Classify zone
        if zone_position >= 90:
            zone = "Extreme Premium"
        elif zone_position >= 70:
            zone = "Premium"
        elif zone_position >= 60:
            zone = "Equilibrium High"
        elif zone_position >= 40:
            zone = "Equilibrium"
        elif zone_position >= 30:
            zone = "Equilibrium Low"
        elif zone_position >= 10:
            zone = "Discount"
        else:
            zone = "Extreme Discount"

        return zone, round(zone_position, 2), round(weekly_high, 4), round(weekly_low, 4)

    def _detect_order_blocks(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamps: List[datetime]
    ) -> List[OrderBlock]:
        """Detect institutional order blocks."""
        order_blocks = []
        ob_length = self.config.order_block_length
        vol_factor = self.config.order_block_volume_factor

        if len(closes) < ob_length + 5:
            return order_blocks

        # Calculate average volume
        avg_volume = volumes[-20:].mean() if len(volumes) >= 20 else volumes.mean()

        for i in range(ob_length, len(closes) - 1):
            # Check for consolidation followed by strong move
            consolidation = closes[i - ob_length:i]
            cons_range = consolidation.max() - consolidation.min()
            cons_avg = consolidation.mean()

            if cons_avg == 0:
                continue

            # Check consolidation is tight (< 2% range)
            if cons_range / cons_avg > 0.02:
                continue

            # Check for strong move after consolidation
            move = closes[i] - cons_avg
            move_pct = abs(move) / cons_avg

            # Check volume spike
            if volumes[i] < avg_volume * vol_factor:
                continue

            # Minimum move threshold (0.5%)
            if move_pct < 0.005:
                continue

            # Create order block
            ob_type = "Bullish" if move > 0 else "Bearish"
            ob_top = highs[i - ob_length:i].max()
            ob_bottom = lows[i - ob_length:i].min()

            # Calculate strength
            strength = min(1.0, (move_pct * 20) * (volumes[i] / avg_volume / 2))

            order_blocks.append(OrderBlock(
                start_index=i - ob_length,
                end_index=i,
                top=ob_top,
                bottom=ob_bottom,
                ob_type=ob_type,
                timestamp=timestamps[i],
                strength=strength,
                is_valid=True
            ))

        # Return most recent order blocks
        return order_blocks[-self.config.max_order_blocks:]

    def _get_nearest_order_block(
        self,
        order_blocks: List[OrderBlock],
        current_price: float
    ) -> Optional[OrderBlock]:
        """Find the nearest valid order block to current price."""
        if not order_blocks:
            return None

        nearest = None
        min_distance = float('inf')

        for ob in order_blocks:
            if not ob.is_valid:
                continue

            ob_mid = (ob.top + ob.bottom) / 2
            distance = abs(current_price - ob_mid)

            if distance < min_distance:
                min_distance = distance
                nearest = ob

        return nearest

    def _calculate_ob_distance(
        self,
        order_block: OrderBlock,
        current_price: float
    ) -> float:
        """Calculate percentage distance to order block."""
        ob_mid = (order_block.top + order_block.bottom) / 2
        distance_pct = ((current_price - ob_mid) / current_price) * 100
        return round(distance_pct, 2)

    def _calculate_confluence_score(
        self,
        trend: str,
        bias: str,
        zone: str,
        nearest_ob: Optional[OrderBlock],
        last_bos: Optional[StructureBreak],
        last_choch: Optional[StructureBreak]
    ) -> float:
        """Calculate overall SMC confluence score (0-100)."""
        score = 50.0  # Base score

        # Trend clarity (+/- 15)
        if trend == bias:
            score += 15 if trend != "Neutral" else 0
        else:
            score -= 10

        # Zone alignment (+/- 15)
        if trend == "Bullish":
            if "Discount" in zone:
                score += 15
            elif "Premium" in zone:
                score -= 10
        elif trend == "Bearish":
            if "Premium" in zone:
                score += 15
            elif "Discount" in zone:
                score -= 10

        # Recent structure break (+/- 10)
        if last_bos:
            score += 5 * last_bos.significance
        if last_choch:
            score += 10 * last_choch.significance

        # Order block presence (+10)
        if nearest_ob:
            score += 10 * nearest_ob.strength

        return round(max(0, min(100, score)), 2)

    async def _get_daily_candles(
        self,
        ticker: str,
        limit: int = 60
    ) -> List[Dict[str, Any]]:
        """Fetch daily candles for a ticker."""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM stock_candles_synthesized
            WHERE ticker = $1 AND timeframe = '1d'
            ORDER BY timestamp DESC
            LIMIT $2
        """
        rows = await self.db.fetch(query, ticker, limit)
        # Reverse to get chronological order
        return list(reversed([dict(r) for r in rows]))

    async def _get_active_tickers(self) -> List[str]:
        """Get list of active tickers with sufficient data."""
        query = """
            SELECT DISTINCT i.ticker
            FROM stock_instruments i
            JOIN stock_candles_synthesized c ON i.ticker = c.ticker
            WHERE i.is_active = TRUE
              AND i.is_tradeable = TRUE
              AND c.timeframe = '1d'
            GROUP BY i.ticker
            HAVING COUNT(*) >= 30
            ORDER BY i.ticker
        """
        rows = await self.db.fetch(query)
        return [r['ticker'] for r in rows]

    async def save_analysis(self, analysis: SMCAnalysis) -> bool:
        """Save SMC analysis to stock_screening_metrics."""
        query = """
            UPDATE stock_screening_metrics
            SET
                smc_trend = $2,
                smc_bias = $3,
                last_bos_type = $4,
                last_bos_date = $5,
                last_bos_price = $6,
                last_choch_type = $7,
                last_choch_date = $8,
                swing_high = $9,
                swing_low = $10,
                swing_high_date = $11,
                swing_low_date = $12,
                premium_discount_zone = $13,
                zone_position = $14,
                weekly_range_high = $15,
                weekly_range_low = $16,
                nearest_ob_type = $17,
                nearest_ob_price = $18,
                nearest_ob_distance = $19,
                smc_confluence_score = $20
            WHERE ticker = $1
              AND calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        """

        try:
            await self.db.execute(
                query,
                analysis.ticker,
                analysis.smc_trend,
                analysis.smc_bias,
                analysis.last_bos_type,
                analysis.last_bos_date,
                analysis.last_bos_price,
                analysis.last_choch_type,
                analysis.last_choch_date,
                analysis.swing_high,
                analysis.swing_low,
                analysis.swing_high_date,
                analysis.swing_low_date,
                analysis.premium_discount_zone,
                analysis.zone_position,
                analysis.weekly_range_high,
                analysis.weekly_range_low,
                analysis.nearest_ob_type,
                analysis.nearest_ob_price,
                analysis.nearest_ob_distance,
                analysis.smc_confluence_score
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save SMC analysis for {analysis.ticker}: {e}")
            return False

    async def run_analysis_pipeline(
        self,
        calculation_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Run full SMC analysis pipeline for all stocks.

        Returns statistics about the analysis run.
        """
        logger.info("Starting SMC analysis pipeline...")

        results = await self.analyze_all_stocks(calculation_date)

        # Save results
        saved = 0
        failed = 0

        for ticker, analysis in results.items():
            success = await self.save_analysis(analysis)
            if success:
                saved += 1
            else:
                failed += 1

        stats = {
            'analyzed': len(results),
            'saved': saved,
            'failed': failed,
            'bullish': sum(1 for a in results.values() if a.smc_trend == "Bullish"),
            'bearish': sum(1 for a in results.values() if a.smc_trend == "Bearish"),
            'neutral': sum(1 for a in results.values() if a.smc_trend == "Neutral"),
        }

        logger.info(f"SMC pipeline complete: {stats}")
        return stats
