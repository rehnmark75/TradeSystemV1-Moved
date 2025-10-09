# ================================
# 1. CORE DATA STRUCTURES
# ================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import pytz
from scipy.signal import argrelextrema
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

class SignalType(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    
class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED_WIN = "CLOSED_WIN"
    CLOSED_LOSS = "CLOSED_LOSS"
    CLOSED_BE = "CLOSED_BE"

@dataclass
class Signal:
    signal_type: SignalType
    epic: str
    timestamp: datetime
    price: float
    confidence_score: float
    ema_9: float
    ema_21: float
    ema_200: float
    spread_pips: float = 2.0
    pip_multiplier: int = 10000
    volume_ratio: Optional[float] = None
    distance_to_support_pips: Optional[float] = None
    distance_to_resistance_pips: Optional[float] = None
    trend_alignment: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class Trade:
    trade_id: str
    signal: Signal
    entry_price: float
    entry_time: datetime
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl_pips: float = 0.0
    pnl_currency: float = 0.0
    max_profit_pips: float = 0.0
    max_loss_pips: float = 0.0
    duration_minutes: int = 0

@dataclass
class Portfolio:
    initial_balance: float
    current_balance: float
    equity: float
    margin_used: float
    free_margin: float
    open_trades: List[Trade] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0