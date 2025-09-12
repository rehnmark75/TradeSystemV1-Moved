"""
21 EMA Trend Reversal Exit System
Standalone risk management module for early trade exits
"""

from typing import List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from services.models import TradeLog, IGCandle


class EMATrendExit:
    """21 EMA trend reversal detection for early trade exits"""
    
    def __init__(self, logger, order_sender=None):
        self.logger = logger
        self.order_sender = order_sender
        self.ema_period = 21
        self.confirmation_candles = 2  # Number of candles that must close wrong side
        self.timeframe = 60  # 1-hour candles for trend analysis
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return []
        
        ema_values = []
        multiplier = 2 / (period + 1)
        
        # Start with SMA for first value
        sma = sum(prices[:period]) / period
        ema_values.append(sma)
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    def get_recent_candles_with_ema(self, db: Session, symbol: str, limit: int = 50) -> List[dict]:
        """Get recent candles with 21 EMA calculated"""
        try:
            # Get recent 1-hour candles
            candles = (db.query(IGCandle)
                      .filter(IGCandle.epic == symbol, IGCandle.timeframe == self.timeframe)
                      .order_by(IGCandle.start_time.desc())
                      .limit(limit)
                      .all())
            
            if len(candles) < self.ema_period + self.confirmation_candles:
                self.logger.warning(f"[EMA EXIT] Not enough candle data for {symbol}: {len(candles)} candles")
                return []
            
            # Reverse to chronological order for EMA calculation
            candles = list(reversed(candles))
            
            # Extract close prices
            close_prices = [candle.close for candle in candles]
            
            # Calculate 21 EMA
            ema_values = self.calculate_ema(close_prices, self.ema_period)
            
            if not ema_values:
                return []
            
            # Combine candle data with EMA (only for candles that have EMA)
            candles_with_ema = []
            ema_start_index = self.ema_period - 1  # EMA starts from period-1 index
            
            for i, ema in enumerate(ema_values):
                candle_index = ema_start_index + i
                candle = candles[candle_index]
                
                candles_with_ema.append({
                    'candle': candle,
                    'ema_21': ema,
                    'close': candle.close,
                    'time': candle.start_time
                })
            
            self.logger.debug(f"[EMA EXIT] {symbol}: Calculated EMA for {len(candles_with_ema)} candles")
            return candles_with_ema
            
        except Exception as e:
            self.logger.error(f"[EMA EXIT ERROR] Failed to get candle data for {symbol}: {e}")
            return []
    
    def check_trend_reversal(self, candles_with_ema: List[dict], trade_direction: str) -> Tuple[bool, str]:
        """
        Check if the last N candles closed on wrong side of 21 EMA
        Returns: (should_exit, reason)
        """
        if len(candles_with_ema) < self.confirmation_candles:
            return False, "insufficient_data"
        
        # Get the last N candles for analysis
        recent_candles = candles_with_ema[-self.confirmation_candles:]
        
        direction = trade_direction.upper()
        wrong_side_count = 0
        candle_details = []
        
        for candle_data in recent_candles:
            close = candle_data['close']
            ema = candle_data['ema_21']
            time = candle_data['time']
            
            if direction == "BUY":
                # For BUY trades: wrong side = close below EMA
                is_wrong_side = close < ema
                side = "below" if is_wrong_side else "above"
            else:  # SELL
                # For SELL trades: wrong side = close above EMA
                is_wrong_side = close > ema
                side = "above" if is_wrong_side else "below"
            
            if is_wrong_side:
                wrong_side_count += 1
            
            candle_details.append(f"{time.strftime('%H:%M')} close={close:.5f} {side} EMA={ema:.5f}")
        
        # Trigger exit if ALL recent candles closed on wrong side
        should_exit = wrong_side_count == self.confirmation_candles
        
        if should_exit:
            reason = f"trend_reversal_{self.confirmation_candles}_candles"
            self.logger.warning(f"[EMA REVERSAL] {direction} trade: {wrong_side_count}/{self.confirmation_candles} candles wrong side")
            for detail in candle_details:
                self.logger.warning(f"[EMA DETAIL] {detail}")
        else:
            reason = f"trend_ok_{wrong_side_count}/{self.confirmation_candles}_wrong"
            self.logger.debug(f"[EMA CHECK] {direction} trade: {reason}")
        
        return should_exit, reason
    
    def should_exit_trade(self, trade: TradeLog, db: Session) -> Tuple[bool, str]:
        """
        Main method to check if trade should exit due to EMA trend reversal
        Returns: (should_exit, reason)
        """
        try:
            # Get candles with EMA data
            candles_with_ema = self.get_recent_candles_with_ema(db, trade.symbol)
            
            if not candles_with_ema:
                return False, "no_ema_data"
            
            # Check for trend reversal
            should_exit, reason = self.check_trend_reversal(candles_with_ema, trade.direction)
            
            if should_exit:
                self.logger.warning(f"[EMA EXIT SIGNAL] Trade {trade.id} {trade.symbol} {trade.direction}: {reason}")
                
                # Log current EMA context
                latest = candles_with_ema[-1]
                self.logger.info(f"[EMA CONTEXT] Latest: close={latest['close']:.5f}, EMA21={latest['ema_21']:.5f}")
            
            return should_exit, reason
            
        except Exception as e:
            self.logger.error(f"[EMA EXIT ERROR] Trade {trade.id} {trade.symbol}: {e}")
            return False, f"error_{str(e)}"
    
    def execute_ema_exit(self, trade: TradeLog, reason: str, db: Session) -> bool:
        """Execute the EMA-based exit by marking trade for closure"""
        try:
            self.logger.warning(f"[EMA EXIT EXECUTING] Trade {trade.id} {trade.symbol}: {reason}")
            
            # Mark trade for closure (main system will handle the actual close)
            trade.status = "ema_exit_pending"
            trade.trigger_time = datetime.utcnow()
            trade.exit_reason = reason
            db.commit()
            
            self.logger.info(f"[EMA EXIT MARKED] Trade {trade.id} {trade.symbol} marked for closure: {reason}")
            return True
                
        except Exception as e:
            self.logger.error(f"[EMA EXIT ERROR] Failed to execute exit for trade {trade.id}: {e}")
            return False