from sqlalchemy import Column, DateTime, String, Integer, Float, UniqueConstraint
from .db import Base

class Candle(Base):
    __tablename__ = "candles"  # plural table name for convention

    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime, nullable=False)
    epic = Column(String, nullable=False, index=True)
    timeframe = Column(Integer, nullable=False)  # in minutes, e.g., 5 or 15
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)

# IG streaming-specific candle model with unique constraint and audit fields
class IGCandle(Base):
    __tablename__ = "ig_candles"

    # Composite primary key
    start_time = Column(DateTime, primary_key=True, nullable=False)
    epic = Column(String, primary_key=True, nullable=False, index=True)
    timeframe = Column(Integer, primary_key=True, nullable=False)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    ltv = Column(Integer, nullable=True)
    cons_tick_count = Column(Integer, nullable=True)
    
    # Data quality and audit fields
    data_source = Column(String(50), nullable=False, default='unknown')
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    quality_score = Column(Float, nullable=False, default=1.0)
    validation_flags = Column(String, nullable=True)  # PostgreSQL array stored as string


class FailedGap(Base):
    """Track gaps that couldn't be backfilled to avoid repeatedly reporting them"""
    __tablename__ = "failed_gaps"

    # Composite primary key to uniquely identify the gap
    epic = Column(String, primary_key=True, nullable=False)
    timeframe = Column(Integer, primary_key=True, nullable=False)
    gap_start = Column(DateTime, primary_key=True, nullable=False)
    gap_end = Column(DateTime, nullable=False)

    # Tracking information
    failure_reason = Column(String, nullable=False)  # 'no_data_available', 'market_closed', 'api_error', etc.
    first_failed_at = Column(DateTime, nullable=False)
    last_attempted_at = Column(DateTime, nullable=False)
    attempt_count = Column(Integer, nullable=False, default=1)

    # Optional metadata
    missing_candles = Column(Integer, nullable=True)
    gap_duration_minutes = Column(Integer, nullable=True)


