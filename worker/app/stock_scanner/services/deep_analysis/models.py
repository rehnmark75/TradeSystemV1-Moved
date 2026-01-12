"""
Deep Analysis Models

Dataclasses for deep analysis results and component scores.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List, Optional
from enum import Enum
import numpy as np


def _convert_for_json(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types to Python native types"""
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_for_json(item) for item in obj]
    return obj


class DAQGrade(Enum):
    """Deep Analysis Quality Grade"""
    A_PLUS = 'A+'
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'

    @classmethod
    def from_score(cls, score: int) -> 'DAQGrade':
        """Convert DAQ score to grade"""
        if score >= 85:
            return cls.A_PLUS
        elif score >= 70:
            return cls.A
        elif score >= 60:
            return cls.B
        elif score >= 50:
            return cls.C
        else:
            return cls.D


class TrendDirection(Enum):
    """Trend direction for multi-timeframe analysis"""
    BULLISH = 'bullish'
    BEARISH = 'bearish'
    NEUTRAL = 'neutral'


class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_BULLISH = 'strong_bullish'
    BULLISH = 'bullish'
    NEUTRAL = 'neutral'
    BEARISH = 'bearish'
    STRONG_BEARISH = 'strong_bearish'


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe"""
    timeframe: str  # '1h', '4h', '1d'
    trend: TrendDirection
    ema_aligned: bool  # Price above key EMAs
    macd_bullish: bool
    rsi_level: Optional[float] = None
    volume_confirm: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MTFAnalysisResult:
    """Multi-Timeframe Analysis Result"""
    score: int  # 0-100
    timeframes: Dict[str, TimeframeAnalysis] = field(default_factory=dict)
    confluence_count: int = 0  # How many TFs align with signal direction
    total_timeframes: int = 3
    signal_direction: TrendDirection = TrendDirection.NEUTRAL
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_aligned(self) -> bool:
        """Check if all timeframes align"""
        return self.confluence_count == self.total_timeframes


@dataclass
class VolumeAnalysisResult:
    """Volume Profile Analysis Result"""
    score: int  # 0-100
    relative_volume: float  # vs 20-day average
    is_accumulation: bool  # Volume on up days > down days
    is_distribution: bool
    unusual_volume: bool  # > 2x average
    volume_trend: str  # 'increasing', 'decreasing', 'stable'
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SMCAnalysisResult:
    """Smart Money Concepts Analysis Result"""
    score: int  # 0-100
    smc_trend: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    smc_bias: Optional[str] = None
    last_bos_type: Optional[str] = None  # Break of Structure type
    last_bos_date: Optional[datetime] = None
    last_choch_type: Optional[str] = None  # Change of Character type
    nearest_ob_type: Optional[str] = None  # Order Block type
    nearest_ob_distance: Optional[float] = None  # % distance to nearest OB
    premium_discount_zone: Optional[str] = None
    zone_position: Optional[float] = None
    confluence_score: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TechnicalDeepResult:
    """Combined Technical Deep Analysis Result"""
    mtf: MTFAnalysisResult
    volume: VolumeAnalysisResult
    smc: SMCAnalysisResult

    @property
    def weighted_score(self) -> int:
        """Calculate weighted technical score (weights: MTF 20%, Volume 10%, SMC 15%)"""
        # Normalize to 0-45 range (45% of total DAQ)
        mtf_weighted = (self.mtf.score / 100) * 20
        volume_weighted = (self.volume.score / 100) * 10
        smc_weighted = (self.smc.score / 100) * 15
        return int(mtf_weighted + volume_weighted + smc_weighted)


@dataclass
class QualityScreenResult:
    """Financial Quality Screen Result"""
    score: int  # 0-100
    roe: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quality_flags: List[str] = field(default_factory=list)  # 'high_roe', 'profitable', 'low_debt', etc.
    risk_flags: List[str] = field(default_factory=list)  # 'high_debt', 'negative_margins', etc.
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CatalystAnalysisResult:
    """Catalyst Timing Analysis Result (earnings, events)"""
    score: int  # 0-100 (inverted risk - high score = no imminent catalyst risk)
    earnings_within_7d: bool = False
    earnings_within_14d: bool = False
    earnings_date: Optional[datetime] = None
    days_to_earnings: Optional[int] = None
    ex_dividend_soon: bool = False
    ex_dividend_date: Optional[datetime] = None
    risk_level: str = 'low'  # 'low', 'medium', 'high'
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstitutionalAnalysisResult:
    """Institutional Activity Analysis Result"""
    score: int  # 0-100
    institutional_percent: Optional[float] = None
    insider_percent: Optional[float] = None
    short_percent_float: Optional[float] = None
    short_ratio: Optional[float] = None
    high_short_interest: bool = False  # > 20%
    squeeze_potential: bool = False  # High short + bullish signal
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FundamentalDeepResult:
    """Combined Fundamental Deep Analysis Result"""
    quality: QualityScreenResult
    catalyst: CatalystAnalysisResult
    institutional: InstitutionalAnalysisResult

    @property
    def weighted_score(self) -> int:
        """Calculate weighted fundamental score (weights: Quality 15%, Catalyst 10%, Institutional 0% for MVP)"""
        # Normalize to 0-25 range (25% of total DAQ)
        quality_weighted = (self.quality.score / 100) * 15
        catalyst_weighted = (self.catalyst.score / 100) * 10
        # institutional_weighted = (self.institutional.score / 100) * 0  # Skip for MVP
        return int(quality_weighted + catalyst_weighted)


@dataclass
class NewsSentimentResult:
    """News Sentiment Analysis Result"""
    score: int  # 0-100
    sentiment_value: float = 0.0  # -1 to 1
    sentiment_level: str = 'neutral'  # 'very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish'
    articles_count: int = 0
    confidence: float = 0.0  # 0-1
    top_headlines: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketRegimeResult:
    """Market Regime Analysis Result"""
    score: int  # 0-100 (how well signal aligns with regime)
    regime: MarketRegime = MarketRegime.NEUTRAL
    spy_trend: TrendDirection = TrendDirection.NEUTRAL
    spy_change_1w: Optional[float] = None
    spy_change_1m: Optional[float] = None
    signal_regime_aligned: bool = False  # Does signal direction match regime?
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectorRotationResult:
    """Sector Rotation Analysis Result"""
    score: int  # 0-100
    sector: Optional[str] = None
    sector_etf: Optional[str] = None
    sector_rs: Optional[float] = None  # Relative strength vs SPY
    sector_outperforming: bool = False
    sector_change_1w: Optional[float] = None
    sector_change_1m: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextualDeepResult:
    """Combined Contextual Deep Analysis Result"""
    news: NewsSentimentResult
    regime: MarketRegimeResult
    sector: SectorRotationResult

    @property
    def weighted_score(self) -> int:
        """Calculate weighted contextual score (weights: News 10%, Regime 10%, Sector 10%)"""
        # Normalize to 0-30 range (30% of total DAQ)
        news_weighted = (self.news.score / 100) * 10
        regime_weighted = (self.regime.score / 100) * 10
        sector_weighted = (self.sector.score / 100) * 10
        return int(news_weighted + regime_weighted + sector_weighted)


@dataclass
class DeepAnalysisResult:
    """Complete Deep Analysis Result"""
    signal_id: int
    ticker: str
    analysis_timestamp: datetime

    # Component Results
    technical: TechnicalDeepResult
    fundamental: FundamentalDeepResult
    contextual: ContextualDeepResult

    # Composite Score
    daq_score: int = 0
    daq_grade: DAQGrade = DAQGrade.D

    # Risk Flags
    earnings_within_7d: bool = False
    high_short_interest: bool = False
    low_liquidity: bool = False
    extreme_volatility: bool = False
    sector_underperforming: bool = False

    # Processing Metadata
    analysis_duration_ms: int = 0
    components_analyzed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def calculate_daq_score(self) -> None:
        """Calculate composite DAQ score from components"""
        # Technical: 45% (MTF 20%, Volume 10%, SMC 15%)
        technical_score = self.technical.weighted_score

        # Fundamental: 25% (Quality 15%, Catalyst 10%)
        fundamental_score = self.fundamental.weighted_score

        # Contextual: 30% (News 10%, Regime 10%, Sector 10%)
        contextual_score = self.contextual.weighted_score

        self.daq_score = technical_score + fundamental_score + contextual_score
        self.daq_grade = DAQGrade.from_score(self.daq_score)

        # Set risk flags
        self.earnings_within_7d = self.fundamental.catalyst.earnings_within_7d
        self.high_short_interest = self.fundamental.institutional.high_short_interest
        self.sector_underperforming = not self.contextual.sector.sector_outperforming

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        # Build dictionary and convert numpy types for JSON serialization
        data = {
            'signal_id': self.signal_id,
            'ticker': self.ticker,
            'analysis_timestamp': self.analysis_timestamp,
            'daq_score': self.daq_score,
            'daq_grade': self.daq_grade.value,

            # Technical scores
            'mtf_score': self.technical.mtf.score,
            'volume_score': self.technical.volume.score,
            'smc_score': self.technical.smc.score,

            # Fundamental scores
            'quality_score': self.fundamental.quality.score,
            'catalyst_score': self.fundamental.catalyst.score,
            'institutional_score': self.fundamental.institutional.score,

            # Contextual scores
            'news_score': self.contextual.news.score,
            'regime_score': self.contextual.regime.score,
            'sector_score': self.contextual.sector.score,

            # Risk flags
            'earnings_within_7d': bool(self.earnings_within_7d),
            'high_short_interest': bool(self.high_short_interest),
            'low_liquidity': bool(self.low_liquidity),
            'extreme_volatility': bool(self.extreme_volatility),
            'sector_underperforming': bool(self.sector_underperforming),

            # Details (JSONB) - convert numpy types for JSON serialization
            'mtf_details': _convert_for_json(self.technical.mtf.details),
            'volume_details': _convert_for_json(self.technical.volume.details),
            'smc_details': _convert_for_json(self.technical.smc.details),
            'fundamental_details': _convert_for_json(self.fundamental.quality.details),
            'context_details': _convert_for_json(self.contextual.regime.details),

            # News
            'news_summary': self.contextual.news.summary,
            'news_articles_count': self.contextual.news.articles_count,
            'top_headlines': _convert_for_json(self.contextual.news.top_headlines),

            # Metadata
            'analysis_duration_ms': self.analysis_duration_ms,
            'components_analyzed': self.components_analyzed,
            'errors': self.errors if self.errors else None,
        }
        return data


@dataclass
class DeepAnalysisConfig:
    """Configuration for deep analysis"""
    # Enable/disable components
    enabled: bool = True
    min_tier_for_auto: str = 'A'  # Minimum signal tier for auto-analysis

    # Component weights (must sum to 100)
    mtf_weight: int = 20
    volume_weight: int = 10
    smc_weight: int = 15
    quality_weight: int = 15
    catalyst_weight: int = 10
    news_weight: int = 10
    regime_weight: int = 10
    sector_weight: int = 10

    # Thresholds
    high_short_interest_threshold: float = 20.0  # %
    low_liquidity_threshold: int = 100000  # avg volume
    extreme_volatility_threshold: float = 5.0  # ATR%
    earnings_risk_days: int = 7

    # Timeframes for MTF analysis (aligned with daily trading timeframe)
    # 4h = entry confirmation, 1d = trading timeframe, 1w = major trend
    mtf_timeframes: List[str] = field(default_factory=lambda: ['4h', '1d', '1w'])

    # Rate limiting
    max_concurrent_analyses: int = 5
    cooldown_hours: int = 4  # Don't re-analyze same ticker within this period
