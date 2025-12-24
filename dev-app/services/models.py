from sqlalchemy import Column, Float, String, Integer, DateTime, Boolean, Numeric, Date, Text, ARRAY
from sqlalchemy.dialects.postgresql import JSON
from .db import Base
from datetime import datetime


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

class TradeLog(Base):
    """
    Trade execution and monitoring log.

    IMPORTANT - Column Naming Clarification:
    ----------------------------------------
    - entry_price: The actual ORDER ENTRY level (for limit/stop orders, this is the
                   stop-entry price, NOT the market price at signal time)
    - limit_price: The TAKE PROFIT level (broker terminology: "limit" = profit target)
                   NOT the limit order entry price! This stores the TP absolute price.
    - sl_price:    The STOP LOSS level (absolute price)
    - tp_price:    Alternative TP storage (used by some flows, same as limit_price concept)

    For stop-entry orders (momentum confirmation style):
    - BUY stop: entry_price is ABOVE market (enter when price breaks up)
    - SELL stop: entry_price is BELOW market (enter when price breaks down)

    The offset between signal generation price and entry_price is typically 2-3 pips
    per strategy config (LIMIT_OFFSET_MAX_PIPS / MOMENTUM_OFFSET_PIPS).
    """
    __tablename__ = "trade_log"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    # Order entry level (for stop-entry orders: the momentum confirmation price)
    entry_price = Column(Float, nullable=False)
    # Take profit level (NOT the limit order entry - this is the TP target price!)
    limit_price = Column(Float, nullable=True)
    # Stop loss level (absolute price)
    sl_price = Column(Float, nullable=True)
    # Alternative TP storage (same concept as limit_price)
    tp_price = Column(Float, nullable=True)
    direction = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    moved_to_breakeven = Column(Boolean, nullable=False, default=False)

    # IG-specific fields
    deal_id = Column(String, nullable=True)
    deal_reference = Column(String, nullable=True)
    endpoint = Column(String, nullable=True)

    # ✅ Existing tracking-related fields
    status = Column(String, nullable=False, default="pending")  # 'pending', 'tracking', 'closed'
    trigger_distance = Column(Float, nullable=True)  # in price units (e.g., 0.0005 for EURUSD)
    min_stop_distance_points = Column(Float, nullable=True) 
    trigger_time = Column(DateTime, nullable=True)   # time when last trigger occurred
    last_trigger_price = Column(Float, nullable=True)  # track last trigger price
    monitor_until = Column(DateTime, nullable=True)  # optional: stop tracking after this time
    closed_at = Column(DateTime, nullable=True)
    alert_id = Column(Integer, nullable=True, index=True)  # Links to forex scanner alert_history.id

    # Partial close tracking columns
    current_size = Column(Float, nullable=True, default=1.0)  # Tracks remaining position size (1.0 → 0.5 after partial close)
    partial_close_executed = Column(Boolean, nullable=False, default=False)  # True if partial close was executed
    partial_close_time = Column(DateTime, nullable=True)  # Timestamp when partial close occurred

    # Early break-even tracking columns (v2.8.0: risk elimination before partial close)
    early_be_executed = Column(Boolean, nullable=False, default=False)  # True if early BE was executed
    early_be_time = Column(DateTime, nullable=True)  # Timestamp when early BE occurred

    # Time-based protection tracking (v2.9.0: protect trades after X mins in profit)
    time_protection_executed = Column(Boolean, nullable=False, default=False)  # True if time protection was executed

    # 🔥 NEW: P&L tracking columns (added for deal correlation)
    profit_loss = Column(Numeric(12, 2), nullable=True)
    pnl_currency = Column(String(10), nullable=True, default='SEK')  # Currency (SEK, USD, etc.)
    updated_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow)
    pnl_updated_at = Column(DateTime, nullable=True)  # When P&L was last updated via 
    
    # Activity correlation fields
    position_reference = Column(String(20))
    activity_correlated = Column(Boolean, default=False)
    lifecycle_duration_minutes = Column(Integer)
    stop_limit_changes_count = Column(Integer, default=0)
    activity_open_deal_id = Column(String(50))
    activity_close_deal_id = Column(String(50))
    
    # Price-based P/L fields
    calculated_pnl = Column(Numeric(12, 4))
    gross_pnl = Column(Numeric(12, 4))
    spread_cost = Column(Numeric(12, 4))
    pips_gained = Column(Numeric(10, 2))
    entry_price_calculated = Column(Numeric(12, 6))
    exit_price_calculated = Column(Numeric(12, 6))
    trade_direction = Column(String(10))
    trade_size = Column(Numeric(10, 2))
    pip_value = Column(Numeric(10, 4))
    pnl_calculation_method = Column(String(20))
    pnl_calculated_at = Column(DateTime)

    def __repr__(self):
        return f"<TradeLog(id={self.id}, symbol={self.symbol}, deal_id={self.deal_id}, profit_loss={self.profit_loss})>"


class AlertHistory(Base):
    __tablename__ = "alert_history"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    epic = Column(String(50), nullable=False, index=True)
    pair = Column(String(10), nullable=False)
    signal_type = Column(String(10), nullable=False, index=True)
    strategy = Column(String(100), nullable=False, index=True)
    confidence_score = Column(Numeric(5, 4), nullable=False, index=True)
    price = Column(Numeric(10, 5), nullable=False)
    bid_price = Column(Numeric(10, 5), nullable=True)
    ask_price = Column(Numeric(10, 5), nullable=True)
    spread_pips = Column(Numeric(5, 2), nullable=True)
    timeframe = Column(String(10), nullable=False)
    strategy_config = Column(JSON, nullable=True)
    strategy_indicators = Column(JSON, nullable=True)
    strategy_metadata = Column(JSON, nullable=True)
    
    # Technical indicators
    ema_short = Column(Numeric(10, 5), nullable=True)
    ema_long = Column(Numeric(10, 5), nullable=True)
    ema_trend = Column(Numeric(10, 5), nullable=True)
    macd_line = Column(Numeric(10, 6), nullable=True)
    macd_signal = Column(Numeric(10, 6), nullable=True)
    macd_histogram = Column(Numeric(10, 6), nullable=True)
    
    # Volume analysis
    volume = Column(Numeric(15, 2), nullable=True)
    volume_ratio = Column(Numeric(8, 4), nullable=True)
    volume_confirmation = Column(Boolean, nullable=True, default=False)
    
    # Support/Resistance
    nearest_support = Column(Numeric(10, 5), nullable=True)
    nearest_resistance = Column(Numeric(10, 5), nullable=True)
    distance_to_support_pips = Column(Numeric(8, 2), nullable=True)
    distance_to_resistance_pips = Column(Numeric(8, 2), nullable=True)
    risk_reward_ratio = Column(Numeric(8, 4), nullable=True)
    
    # Market context
    market_session = Column(String(50), nullable=True)
    is_market_hours = Column(Boolean, nullable=True, default=True)
    market_regime = Column(String(50), nullable=True)
    signal_trigger = Column(String(100), nullable=True, index=True)
    signal_conditions = Column(JSON, nullable=True)
    crossover_type = Column(String(100), nullable=True)
    
    # Claude AI Analysis
    claude_analysis = Column(Text, nullable=True)
    alert_message = Column(Text, nullable=True)
    alert_level = Column(String(50), nullable=True, default='INFO')
    status = Column(String(50), nullable=True, default='ACTIVE', index=True)
    processed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=True, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow)
    market_timestamp = Column(DateTime, nullable=True, index=True)
    
    # Deduplication and tracking
    signal_hash = Column(String(32), nullable=True, index=True)
    data_source = Column(String(50), nullable=True, default='live_scanner', index=True)
    cooldown_key = Column(String(100), nullable=True, index=True)
    dedup_metadata = Column(JSON, nullable=True)
    
    # Claude scoring
    claude_score = Column(Integer, nullable=True, index=True)
    claude_decision = Column(String(50), nullable=True, index=True)
    claude_approved = Column(Boolean, nullable=True, index=True)
    claude_reason = Column(Text, nullable=True)
    claude_mode = Column(String(50), nullable=True)
    claude_raw_response = Column(Text, nullable=True)
    strategy_config_hash = Column(String(64), nullable=True)
    
    # Smart Money Concepts
    smart_money_validated = Column(Boolean, nullable=True, default=False, index=True)
    smart_money_type = Column(String(50), nullable=True, index=True)
    smart_money_score = Column(Numeric(5, 4), nullable=True, index=True)
    market_structure_analysis = Column(JSON, nullable=True)
    order_flow_analysis = Column(JSON, nullable=True)
    enhanced_confidence_score = Column(Numeric(5, 4), nullable=True, index=True)
    confluence_details = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<AlertHistory(id={self.id}, epic={self.epic}, signal_type={self.signal_type}, strategy={self.strategy})>"


class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    epic = Column(Text, nullable=False)
    start_time = Column(DateTime, nullable=False)
    direction = Column(Text, nullable=False)  # CHECK constraint: 'BUY' or 'SELL'
    price = Column(Numeric, nullable=False)
    alert_type = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=True, default=datetime.utcnow)
    zone_confirmed = Column(Boolean, nullable=True, default=False)
    
    def __repr__(self):
        return f"<Alert(id={self.id}, epic={self.epic}, direction={self.direction}, price={self.price})>"


class BrokerTransaction(Base):
    __tablename__ = "broker_transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_date = Column(Date, nullable=False, index=True)
    instrument_name = Column(String(255), nullable=False)
    instrument_epic = Column(String(100), nullable=True, index=True)
    period = Column(String(50), nullable=True)
    profit_loss_amount = Column(Numeric(12, 2), nullable=False)
    profit_loss_currency = Column(String(10), nullable=False)
    transaction_type = Column(String(50), nullable=False)
    reference = Column(String(50), nullable=False, unique=True, index=True)
    open_level = Column(Numeric(12, 6), nullable=True)
    close_level = Column(Numeric(12, 6), nullable=True)
    position_size = Column(Numeric(12, 4), nullable=True)
    trade_direction = Column(String(10), nullable=True)
    pips_gained = Column(Numeric(8, 2), nullable=True)
    trade_result = Column(String(20), nullable=True)
    cash_transaction = Column(Boolean, nullable=True, default=False)
    deal_id = Column(String(50), nullable=True, index=True)
    deal_reference = Column(String(50), nullable=True)
    created_at = Column(DateTime, nullable=True, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<BrokerTransaction(id={self.id}, reference={self.reference}, profit_loss={self.profit_loss_amount})>"



# IG streaming-specific candle model with unique constraint
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
    data_source = Column(String(50), nullable=True, default='unknown')
    created_at = Column(DateTime, nullable=True, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow)
    quality_score = Column(Numeric(3, 2), nullable=True, default=1.0)
    validation_flags = Column(ARRAY(Text), nullable=True)
