from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, Enum, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()


class ImpactLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HOLIDAY = "holiday"


class EventStatus(enum.Enum):
    UPCOMING = "upcoming"
    RELEASED = "released"
    REVISED = "revised"


class ScrapeStatus(enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    IN_PROGRESS = "in_progress"


class EconomicEvent(Base):
    """Economic calendar events from Forex Factory and other sources"""
    __tablename__ = "economic_events"

    id = Column(Integer, primary_key=True, index=True)

    # Event identification
    event_name = Column(String(255), nullable=False, index=True)
    currency = Column(String(3), nullable=False, index=True)  # USD, EUR, GBP, etc.
    country = Column(String(50), nullable=True, index=True)

    # Timing
    event_date = Column(DateTime, nullable=False, index=True)
    event_time = Column(String(10), nullable=True)  # "09:30", "14:00", etc.
    timezone = Column(String(50), nullable=True, default="UTC")

    # Impact and importance
    impact_level = Column(Enum(ImpactLevel), nullable=False, index=True)
    importance_score = Column(Float, nullable=True)  # 0.0-1.0 calculated importance

    # Economic values
    previous_value = Column(String(50), nullable=True)
    forecast_value = Column(String(50), nullable=True)
    actual_value = Column(String(50), nullable=True)
    revised_value = Column(String(50), nullable=True)

    # Numeric conversions (when possible)
    previous_numeric = Column(Float, nullable=True)
    forecast_numeric = Column(Float, nullable=True)
    actual_numeric = Column(Float, nullable=True)

    # Event details
    category = Column(String(100), nullable=True)  # "Employment", "Inflation", etc.
    description = Column(Text, nullable=True)
    frequency = Column(String(50), nullable=True)  # "Monthly", "Quarterly", etc.

    # Status and metadata
    status = Column(Enum(EventStatus), nullable=False, default=EventStatus.UPCOMING)
    source = Column(String(50), nullable=False, default="forex_factory")
    source_url = Column(String(500), nullable=True)
    external_id = Column(String(100), nullable=True, index=True)  # Source-specific ID

    # Market impact tracking
    market_moving = Column(Boolean, nullable=True)  # Historical market impact flag
    volatility_expected = Column(Boolean, nullable=True, default=False)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    scraped_at = Column(DateTime, nullable=False, default=func.now())

    # Indexes for common queries
    __table_args__ = (
        Index('idx_event_date_currency', 'event_date', 'currency'),
        Index('idx_currency_impact', 'currency', 'impact_level'),
        Index('idx_date_impact', 'event_date', 'impact_level'),
        Index('idx_upcoming_events', 'event_date', 'status'),
    )

    def __repr__(self):
        return f"<EconomicEvent(event_name='{self.event_name}', currency='{self.currency}', date='{self.event_date}')>"


class ScrapeLog(Base):
    """Track scraping operations and their results"""
    __tablename__ = "economic_scrape_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Scrape details
    scrape_date = Column(DateTime, nullable=False, default=func.now(), index=True)
    data_source = Column(String(50), nullable=False, default="forex_factory")
    scrape_type = Column(String(50), nullable=False, default="weekly")  # "weekly", "daily", "manual"

    # Results
    status = Column(Enum(ScrapeStatus), nullable=False, index=True)
    events_found = Column(Integer, nullable=True, default=0)
    events_new = Column(Integer, nullable=True, default=0)
    events_updated = Column(Integer, nullable=True, default=0)
    events_failed = Column(Integer, nullable=True, default=0)

    # Performance metrics
    duration_seconds = Column(Float, nullable=True)
    start_time = Column(DateTime, nullable=False, default=func.now())
    end_time = Column(DateTime, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_count = Column(Integer, nullable=True, default=0)
    retry_count = Column(Integer, nullable=True, default=0)

    # Metadata
    user_agent = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    response_status = Column(Integer, nullable=True)

    # Date range scraped
    date_from = Column(DateTime, nullable=True)
    date_to = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=func.now())

    __table_args__ = (
        Index('idx_scrape_date_status', 'scrape_date', 'status'),
        Index('idx_source_type', 'data_source', 'scrape_type'),
    )

    def __repr__(self):
        return f"<ScrapeLog(date='{self.scrape_date}', status='{self.status}', events='{self.events_found}')>"


class NewsImpactAnalysis(Base):
    """Track correlation between economic events and trading signals/outcomes"""
    __tablename__ = "news_impact_analysis"

    id = Column(Integer, primary_key=True, index=True)

    # Link to economic event
    economic_event_id = Column(Integer, nullable=False, index=True)

    # Link to trading activity (if any)
    alert_id = Column(Integer, nullable=True, index=True)  # Links to alert_history.id
    trade_id = Column(Integer, nullable=True, index=True)  # Links to trade_log.id

    # Analysis details
    currency_pair = Column(String(10), nullable=False, index=True)  # EURUSD, GBPUSD, etc.
    impact_detected = Column(Boolean, nullable=False, default=False)
    volatility_increase = Column(Float, nullable=True)  # Percentage increase in volatility
    price_movement_pips = Column(Float, nullable=True)  # Price movement in pips

    # Timing analysis
    time_before_event_minutes = Column(Integer, nullable=True)  # Minutes before event
    time_after_event_minutes = Column(Integer, nullable=True)   # Minutes after event

    # Impact scoring
    impact_score = Column(Float, nullable=True)  # 0.0-1.0 calculated impact
    surprise_factor = Column(Float, nullable=True)  # Deviation from forecast

    # Market context
    session_active = Column(String(50), nullable=True)  # "London", "New York", etc.
    market_conditions = Column(String(100), nullable=True)  # "trending", "ranging", etc.

    # Analysis metadata
    analysis_type = Column(String(50), nullable=False, default="automated")
    confidence_level = Column(Float, nullable=True)  # 0.0-1.0

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_economic_currency_impact', 'currency_pair', 'impact_detected'),
        Index('idx_economic_event_alert', 'economic_event_id', 'alert_id'),
    )

    def __repr__(self):
        return f"<NewsImpactAnalysis(event_id='{self.economic_event_id}', pair='{self.currency_pair}', impact='{self.impact_detected}')>"