# services/data_fetcher_adapter.py
"""
Data Fetcher Adapter for MarketIntelligenceEngine
Adapts Streamlit's database connection to work with the intelligence system
"""

import pandas as pd
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import numpy as np

try:
    from services.indicators import apply_indicators
except ImportError:
    apply_indicators = None

from sqlalchemy import text


class StreamlitDataFetcher:
    """Data fetcher adapter for MarketIntelligenceEngine in Streamlit context"""
    
    def __init__(self, engine=None):
        self.logger = logging.getLogger(__name__)
        self.engine = engine
        self.logger.info("📊 StreamlitDataFetcher initialized")
    
    def get_enhanced_data(self, epic: str, pair_name: str, timeframe: str = '15m', 
                         lookback_hours: int = 48) -> Optional[pd.DataFrame]:
        """
        Get enhanced market data with indicators for intelligence analysis
        
        Args:
            epic: Trading epic (e.g., 'CS.D.EURUSD.MINI.IP')
            pair_name: Currency pair name (e.g., 'EURUSD')  
            timeframe: Timeframe ('5m', '15m', '1h')
            lookback_hours: Hours of historical data to fetch
            
        Returns:
            DataFrame with OHLC data + indicators or None if no data
        """
        try:
            # Check if engine is available
            if self.engine is None:
                self.logger.debug(f"Database engine not available for {epic}")
                return None
            
            # Map timeframes to database timeframe values (integers)
            timeframe_map = {
                '5m': 5,
                '15m': 5,  # Use 5m data for 15m requests since 15m not available
                '1h': 60,
                '60m': 60
            }
            
            db_timeframe = timeframe_map.get(timeframe, 5)  # Default to 5m if not found
            
            # Calculate lookback time
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=lookback_hours)
            
            # Use SQL query similar to data.py approach
            sql = """
                SELECT start_time, open, high, low, close, volume, ltv
                FROM ig_candles 
                WHERE epic = :epic AND timeframe = :timeframe
                    AND start_time >= :start_time AND start_time <= :end_time
                    AND NOT (open = high AND high = low AND low = close)
                ORDER BY start_time ASC
            """
            
            query = text(sql)
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    query, 
                    conn, 
                    params={
                        "epic": epic, 
                        "timeframe": db_timeframe,
                        "start_time": start_time,
                        "end_time": end_time
                    }
                )
            
            if df.empty:
                self.logger.debug(f"No candle data found for {epic} {timeframe}")
                return None
            
            if len(df) < 10:
                self.logger.debug(f"Insufficient data for {epic}: {len(df)} candles")
                return None
            
            # Ensure proper datetime index
            df['start_time'] = pd.to_datetime(df['start_time'])
            df = df.set_index('start_time').sort_index()
            
            # Fill missing volume data
            df['volume'] = df['volume'].fillna(0)
            df['ltv'] = df['ltv'].fillna(0)
            
            # Add technical indicators required by intelligence engine
            df = self._add_technical_indicators(df)
            
            # Add additional columns expected by intelligence system
            df = self._add_intelligence_columns(df)
            
            self.logger.debug(f"✅ Fetched {len(df)} candles for {epic} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch data for {epic} {timeframe}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators required by intelligence engine"""
        try:
            # Use existing indicators service to add EMAs and other indicators
            if apply_indicators is not None:
                df_enhanced = apply_indicators(df.copy())
                return df_enhanced
            else:
                # Add basic EMAs manually if indicators service not available
                if len(df) >= 21:
                    df['ema_9'] = df['close'].ewm(span=9).mean()
                    df['ema_21'] = df['close'].ewm(span=21).mean()
                    df['ema_50'] = df['close'].ewm(span=50).mean()
                    if len(df) >= 200:
                        df['ema_200'] = df['close'].ewm(span=200).mean()
                return df
            
        except Exception as e:
            self.logger.warning(f"Failed to add technical indicators: {e}")
            return df
    
    def _add_intelligence_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional columns expected by MarketIntelligenceEngine"""
        try:
            # Ensure we have the columns that intelligence engine expects
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'ltv']
            
            for col in required_columns:
                if col not in df.columns:
                    if col == 'ltv' and 'volume' in df.columns:
                        df['ltv'] = df['volume']
                    elif col == 'volume' and 'ltv' in df.columns:
                        df['volume'] = df['ltv']
                    else:
                        df[col] = 0
            
            # Add price change columns
            if 'close' in df.columns:
                df['price_change'] = df['close'].pct_change()
                df['price_change_abs'] = df['close'].diff()
            
            # Add basic volatility measure (required for intelligence analysis)
            if all(col in df.columns for col in ['high', 'low', 'close']):
                df['true_range'] = np.maximum(
                    df['high'] - df['low'],
                    np.maximum(
                        abs(df['high'] - df['close'].shift(1)),
                        abs(df['low'] - df['close'].shift(1))
                    )
                )
                
                # Simple ATR
                df['atr'] = df['true_range'].rolling(window=14, min_periods=1).mean()
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Failed to add intelligence columns: {e}")
            return df
    
    def get_market_data_summary(self, epic_list: List[str]) -> Dict:
        """Get summary of available market data for intelligence analysis"""
        try:
            if self.engine is None:
                return {
                    'epics_available': 0,
                    'total_candles': 0,
                    'timeframes_available': [],
                    'date_range': {},
                    'data_quality': 'no_engine'
                }
                
            summary = {
                'epics_available': 0,
                'total_candles': 0,
                'timeframes_available': [],
                'date_range': {},
                'data_quality': 'unknown'
            }
            
            available_epics = []
            total_candles = 0
            
            for epic in epic_list:
                try:
                    # Check if we have data for this epic using SQL
                    sql = """
                        SELECT COUNT(*) as count
                        FROM ig_candles 
                        WHERE epic = :epic 
                            AND start_time >= :min_time
                    """
                    
                    query = text(sql)
                    with self.engine.connect() as conn:
                        result = conn.execute(query, {
                            "epic": epic,
                            "min_time": datetime.utcnow() - timedelta(hours=48)
                        }).fetchone()
                        
                        count = result[0] if result else 0
                        if count > 0:
                            available_epics.append(epic)
                            total_candles += count
                        
                except Exception as e:
                    self.logger.debug(f"Error checking data for {epic}: {e}")
                    continue
            
            # Get timeframes available
            try:
                sql = "SELECT DISTINCT timeframe FROM ig_candles WHERE timeframe IS NOT NULL"
                with self.engine.connect() as conn:
                    result = conn.execute(text(sql)).fetchall()
                    summary['timeframes_available'] = [row[0] for row in result]
            except:
                summary['timeframes_available'] = ['15m']
            
            # Get date range
            try:
                sql = "SELECT MIN(start_time) as oldest, MAX(start_time) as newest FROM ig_candles"
                with self.engine.connect() as conn:
                    result = conn.execute(text(sql)).fetchone()
                    if result and result[0] and result[1]:
                        summary['date_range'] = {
                            'oldest': result[0],
                            'newest': result[1]
                        }
            except:
                pass
            
            summary['epics_available'] = len(available_epics)
            summary['total_candles'] = total_candles
            summary['available_epics'] = available_epics
            
            # Determine data quality
            if len(available_epics) >= len(epic_list) * 0.8:  # 80% of epics available
                summary['data_quality'] = 'good'
            elif len(available_epics) >= len(epic_list) * 0.5:  # 50% available
                summary['data_quality'] = 'fair'
            else:
                summary['data_quality'] = 'poor'
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get market data summary: {e}")
            return {
                'epics_available': 0,
                'total_candles': 0,
                'timeframes_available': [],
                'date_range': {},
                'data_quality': 'error'
            }
    
    def __del__(self):
        """Clean up resources"""
        pass  # No cleanup needed for SQLAlchemy engine