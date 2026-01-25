"""
Fundamental Deep Analyzer

Performs deep fundamental analysis including:
- Financial quality screen (ROE, margins, debt)
- Catalyst timing (earnings, events)
- Institutional activity (ownership, short interest)
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional

from .models import (
    FundamentalDeepResult,
    QualityScreenResult,
    CatalystAnalysisResult,
    InstitutionalAnalysisResult,
    DeepAnalysisConfig,
)

logger = logging.getLogger(__name__)


class FundamentalDeepAnalyzer:
    """
    Performs deep fundamental analysis on stock signals.

    Components:
    1. Financial Quality Screen (15% of DAQ)
    2. Catalyst Timing Analysis (10% of DAQ)
    3. Institutional Activity (0% for MVP - data tracked but not weighted)
    """

    def __init__(self, db_manager, config: Optional[DeepAnalysisConfig] = None):
        """
        Initialize fundamental analyzer.

        Args:
            db_manager: Database manager for fetching fundamental data
            config: Deep analysis configuration
        """
        self.db = db_manager
        self.config = config or DeepAnalysisConfig()

    async def analyze(
        self,
        ticker: str,
        signal: Dict[str, Any]
    ) -> FundamentalDeepResult:
        """
        Perform complete fundamental deep analysis.

        Args:
            ticker: Stock ticker
            signal: Signal data

        Returns:
            FundamentalDeepResult with all component scores
        """
        # Fetch fundamental data
        fundamental_data = await self._fetch_fundamentals(ticker)

        # Run analysis components
        quality_result = self._analyze_quality(fundamental_data)
        catalyst_result = self._analyze_catalyst(fundamental_data)
        institutional_result = self._analyze_institutional(fundamental_data, signal)

        return FundamentalDeepResult(
            quality=quality_result,
            catalyst=catalyst_result,
            institutional=institutional_result
        )

    async def _fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Fetch fundamental data from stock_instruments table"""
        query = """
            SELECT
                ticker, name, sector, industry,
                -- Valuation
                trailing_pe, forward_pe, price_to_book, price_to_sales,
                peg_ratio, enterprise_to_ebitda, enterprise_value,
                -- Growth
                revenue_growth, earnings_growth, earnings_quarterly_growth,
                -- Profitability
                profit_margin, operating_margin, gross_margin,
                return_on_equity, return_on_assets,
                -- Financial Health
                debt_to_equity, current_ratio, quick_ratio,
                -- Risk Metrics
                beta, short_ratio, short_percent_float,
                -- Ownership
                institutional_percent, insider_percent,
                -- Dividend
                dividend_yield, dividend_rate, payout_ratio,
                -- 52-Week Data
                fifty_two_week_high, fifty_two_week_low, fifty_two_week_change,
                fifty_day_average, two_hundred_day_average,
                -- Analyst Data
                analyst_rating, target_price, target_high, target_low, number_of_analysts,
                -- Calendar
                earnings_date, ex_dividend_date,
                -- Meta
                fundamentals_updated_at
            FROM stock_instruments
            WHERE ticker = $1
        """
        row = await self.db.fetchrow(query, ticker)

        if not row:
            return {}

        return {k: v for k, v in dict(row).items() if v is not None}

    # =========================================================================
    # FINANCIAL QUALITY SCREEN (15% of DAQ)
    # =========================================================================

    def _analyze_quality(self, data: Dict[str, Any]) -> QualityScreenResult:
        """
        Analyze financial quality metrics.

        Scoring based on:
        - ROE > 15%: High quality
        - Profit margin > 10%: Profitable
        - Debt/Equity < 1.0: Low debt
        - Current ratio > 1.5: Good liquidity

        Score calculation:
        - Excellent fundamentals: 90-100
        - Good fundamentals: 70-89
        - Average fundamentals: 50-69
        - Weak fundamentals: 30-49
        - Poor fundamentals: 0-29
        """
        if not data:
            return QualityScreenResult(
                score=50,
                details={'error': 'No fundamental data available'}
            )

        quality_flags = []
        risk_flags = []
        score = 50  # Base score

        # Return on Equity
        roe = data.get('return_on_equity')
        if roe is not None:
            roe_pct = roe * 100 if roe < 1 else roe  # Handle decimal vs percentage
            if roe_pct > 20:
                score += 15
                quality_flags.append(f'Excellent ROE ({roe_pct:.1f}%)')
            elif roe_pct > 15:
                score += 10
                quality_flags.append(f'Good ROE ({roe_pct:.1f}%)')
            elif roe_pct > 10:
                score += 5
            elif roe_pct < 5:
                score -= 10
                risk_flags.append(f'Low ROE ({roe_pct:.1f}%)')
            elif roe_pct < 0:
                score -= 20
                risk_flags.append(f'Negative ROE ({roe_pct:.1f}%)')

        # Profit Margin
        profit_margin = data.get('profit_margin')
        if profit_margin is not None:
            pm_pct = profit_margin * 100 if profit_margin < 1 else profit_margin
            if pm_pct > 20:
                score += 15
                quality_flags.append(f'High profit margin ({pm_pct:.1f}%)')
            elif pm_pct > 10:
                score += 10
                quality_flags.append(f'Good profit margin ({pm_pct:.1f}%)')
            elif pm_pct > 5:
                score += 5
            elif pm_pct < 0:
                score -= 15
                risk_flags.append(f'Negative profit margin ({pm_pct:.1f}%)')

        # Operating Margin
        operating_margin = data.get('operating_margin')
        if operating_margin is not None:
            om_pct = operating_margin * 100 if operating_margin < 1 else operating_margin
            if om_pct > 15:
                score += 5
            elif om_pct < 0:
                score -= 5
                risk_flags.append('Negative operating margin')

        # Debt to Equity
        debt_equity = data.get('debt_to_equity')
        if debt_equity is not None:
            if debt_equity < 0.5:
                score += 10
                quality_flags.append(f'Low debt (D/E: {debt_equity:.2f})')
            elif debt_equity < 1.0:
                score += 5
                quality_flags.append(f'Moderate debt (D/E: {debt_equity:.2f})')
            elif debt_equity > 2.0:
                score -= 10
                risk_flags.append(f'High debt (D/E: {debt_equity:.2f})')
            elif debt_equity > 3.0:
                score -= 15
                risk_flags.append(f'Very high debt (D/E: {debt_equity:.2f})')

        # Current Ratio (Liquidity)
        current_ratio = data.get('current_ratio')
        if current_ratio is not None:
            if current_ratio > 2.0:
                score += 5
                quality_flags.append('Strong liquidity')
            elif current_ratio > 1.5:
                score += 3
            elif current_ratio < 1.0:
                score -= 10
                risk_flags.append(f'Low liquidity (Current ratio: {current_ratio:.2f})')

        # Clamp score to 0-100 range
        score = max(0, min(100, score))

        return QualityScreenResult(
            score=score,
            roe=roe,
            profit_margin=profit_margin,
            operating_margin=operating_margin,
            debt_to_equity=debt_equity,
            current_ratio=current_ratio,
            quality_flags=quality_flags,
            risk_flags=risk_flags,
            details={
                'gross_margin': data.get('gross_margin'),
                'return_on_assets': data.get('return_on_assets'),
                'quick_ratio': data.get('quick_ratio'),
            }
        )

    # =========================================================================
    # CATALYST TIMING ANALYSIS (10% of DAQ)
    # =========================================================================

    def _analyze_catalyst(self, data: Dict[str, Any]) -> CatalystAnalysisResult:
        """
        Analyze catalyst timing risk (earnings dates, events).

        Score is INVERTED risk - high score = low catalyst risk.

        Scoring:
        - No earnings within 14 days: 100
        - Earnings 8-14 days away: 80
        - Earnings 4-7 days away: 50
        - Earnings within 3 days: 20
        - Ex-dividend within 3 days: -10 penalty
        """
        earnings_date = data.get('earnings_date')
        ex_div_date = data.get('ex_dividend_date')

        now = datetime.now()

        # Initialize
        days_to_earnings = None
        earnings_within_7d = False
        earnings_within_14d = False
        ex_dividend_soon = False
        risk_level = 'low'
        score = 100  # Start at max (no risk)

        # Check earnings date
        if earnings_date:
            # Convert to datetime if needed
            if isinstance(earnings_date, str):
                try:
                    earnings_date = datetime.fromisoformat(earnings_date.replace('Z', '+00:00'))
                except ValueError:
                    earnings_date = None
            elif isinstance(earnings_date, date) and not isinstance(earnings_date, datetime):
                # Convert date to datetime for comparison
                earnings_date = datetime.combine(earnings_date, datetime.min.time())

            if earnings_date and earnings_date > now:
                days_to_earnings = (earnings_date - now).days

                if days_to_earnings <= 3:
                    score = 20
                    risk_level = 'high'
                    earnings_within_7d = True
                    earnings_within_14d = True
                elif days_to_earnings <= 7:
                    score = 50
                    risk_level = 'medium'
                    earnings_within_7d = True
                    earnings_within_14d = True
                elif days_to_earnings <= 14:
                    score = 80
                    risk_level = 'low'
                    earnings_within_14d = True
                # else: score stays 100

        # Check ex-dividend date
        if ex_div_date:
            # Convert to datetime if needed
            if isinstance(ex_div_date, str):
                try:
                    ex_div_date = datetime.fromisoformat(ex_div_date.replace('Z', '+00:00'))
                except ValueError:
                    ex_div_date = None
            elif isinstance(ex_div_date, date) and not isinstance(ex_div_date, datetime):
                # Convert date to datetime for comparison
                ex_div_date = datetime.combine(ex_div_date, datetime.min.time())

            if ex_div_date and ex_div_date > now:
                days_to_ex_div = (ex_div_date - now).days
                if days_to_ex_div <= 3:
                    ex_dividend_soon = True
                    score -= 10  # Small penalty

        # Clamp score
        score = max(0, min(100, score))

        return CatalystAnalysisResult(
            score=score,
            earnings_within_7d=earnings_within_7d,
            earnings_within_14d=earnings_within_14d,
            earnings_date=earnings_date,
            days_to_earnings=days_to_earnings,
            ex_dividend_soon=ex_dividend_soon,
            ex_dividend_date=ex_div_date,
            risk_level=risk_level,
            details={
                'analyst_rating': data.get('analyst_rating'),
                'target_price': data.get('target_price'),
                'number_of_analysts': data.get('number_of_analysts'),
            }
        )

    # =========================================================================
    # INSTITUTIONAL ACTIVITY (0% for MVP - tracked but not weighted)
    # =========================================================================

    def _analyze_institutional(
        self,
        data: Dict[str, Any],
        signal: Dict[str, Any]
    ) -> InstitutionalAnalysisResult:
        """
        Analyze institutional ownership and short interest.

        Not weighted in DAQ for MVP, but data is tracked for reference.

        Flags:
        - High short interest (>20%): squeeze potential if bullish signal
        - Low institutional ownership (<20%): less institutional support
        - High insider ownership: alignment of interests
        """
        institutional_pct = data.get('institutional_percent')
        insider_pct = data.get('insider_percent')
        short_pct = data.get('short_percent_float')
        short_ratio = data.get('short_ratio')

        score = 50  # Base score (not weighted in DAQ for MVP)
        high_short_interest = False
        squeeze_potential = False

        # Check short interest
        if short_pct is not None:
            short_pct_value = short_pct * 100 if short_pct < 1 else short_pct
            if short_pct_value > 20:
                high_short_interest = True
                # If signal is bullish and high short interest, could be squeeze
                signal_type = signal.get('signal_type', '').upper()
                if signal_type in ('BUY', 'LONG'):
                    squeeze_potential = True
                    score += 10  # Potential upside from squeeze

        # Check institutional ownership
        if institutional_pct is not None:
            inst_pct_value = institutional_pct * 100 if institutional_pct < 1 else institutional_pct
            if inst_pct_value > 70:
                score += 10  # Strong institutional support
            elif inst_pct_value < 20:
                score -= 10  # Low institutional interest

        # Clamp score
        score = max(0, min(100, score))

        return InstitutionalAnalysisResult(
            score=score,
            institutional_percent=institutional_pct,
            insider_percent=insider_pct,
            short_percent_float=short_pct,
            short_ratio=short_ratio,
            high_short_interest=high_short_interest,
            squeeze_potential=squeeze_potential,
            details={
                'beta': data.get('beta'),
            }
        )
