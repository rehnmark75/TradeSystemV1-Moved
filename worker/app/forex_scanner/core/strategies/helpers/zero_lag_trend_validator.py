# core/strategies/helpers/zero_lag_trend_validator.py
"""
Zero Lag Trend Validator Module
Validates trends and provides multi-timeframe confirmation for Zero Lag strategy
"""

import pandas as pd
import logging
from typing import Optional, Dict
try:
    import config
except ImportError:
    from forex_scanner import config


class ZeroLagTrendValidator:
    """Handles all trend validation and multi-timeframe analysis for Zero Lag signals"""
    
    def __init__(self, logger: logging.Logger = None, enhanced_validation: bool = True):
        self.logger = logger or logging.getLogger(__name__)
        self.enhanced_validation = enhanced_validation
        self.eps = 1e-8  # Epsilon for stability
    
    def validate_ema200_macro_trend(self, row: pd.Series, signal_type: str) -> bool:
        """
        EMA 200 Macro Trend Validation
        
        Rules:
        - BULL signals: Price must be ABOVE EMA 200 with upward slope
        - BEAR signals: Price must be BELOW EMA 200 with downward slope
        
        Args:
            row: DataFrame row with price and EMA data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if macro trend supports signal
        """
        try:
            if not getattr(config, 'EMA_200_TREND_FILTER_ENABLED', True):
                return True
            
            ema_200 = row.get('ema_200', 0)
            close_price = row.get('close', 0)
            
            if ema_200 == 0 or close_price == 0:
                self.logger.debug("EMA 200 or close price not available")
                return True  # Allow signal if data not available
            
            # Calculate EMA 200 slope if available
            ema_200_slope = self._calculate_ema200_slope(row)
            
            # More generous buffer for noise reduction
            buffer_pips = getattr(config, 'EMA_200_BUFFER_PIPS', 5.0)  # Increased from 2.0
            pip_multiplier = 100 if close_price > 50 else 10000
            buffer_distance = buffer_pips / pip_multiplier
            
            if signal_type == 'BULL':
                # RELAXED: Bull signals allowed if close to or above EMA 200
                price_above = close_price > ema_200 - buffer_distance  # Allow some distance below
                slope_ok = ema_200_slope >= -self.eps * 5 if ema_200_slope is not None else True  # More lenient slope
                
                # Allow BULL if price is reasonable close to EMA200 OR slope is positive
                bull_ema_valid = price_above or (slope_ok and ema_200_slope is not None and ema_200_slope > 0)
                
                if bull_ema_valid:
                    distance = (close_price - ema_200) * pip_multiplier
                    self.logger.debug(f"✅ EMA 200 macro trend OK for BULL: {distance:+.1f} pips from EMA200")
                    return True
                else:
                    distance_below = (ema_200 - close_price) * pip_multiplier
                    self.logger.debug(f"❌ EMA 200 BULL: too far below ({distance_below:.1f} pips) + negative slope")
                    return False
            
            elif signal_type == 'BEAR':
                # RELAXED: Bear signals allowed if close to or below EMA 200
                price_below = close_price < ema_200 + buffer_distance  # Allow some distance above
                slope_ok = ema_200_slope <= self.eps * 5 if ema_200_slope is not None else True  # More lenient slope
                
                # Allow BEAR if price is reasonably close to EMA200 OR slope is negative
                bear_ema_valid = price_below or (slope_ok and ema_200_slope is not None and ema_200_slope < 0)
                
                if bear_ema_valid:
                    distance = (close_price - ema_200) * pip_multiplier
                    self.logger.debug(f"✅ EMA 200 macro trend OK for BEAR: {distance:+.1f} pips from EMA200")
                    return True
                else:
                    distance_above = (close_price - ema_200) * pip_multiplier
                    self.logger.debug(f"❌ EMA 200 BEAR: too far above ({distance_above:.1f} pips) + positive slope")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating EMA 200 macro trend: {e}")
            return True  # Allow signal on error
    
    def _calculate_ema200_slope(self, row: pd.Series) -> Optional[float]:
        """Calculate EMA 200 slope if previous data available"""
        try:
            # Look for previous EMA 200 value in the row (if available)
            ema_200_current = row.get('ema_200', 0)
            ema_200_prev = row.get('ema_200_prev', None)
            
            if ema_200_prev is not None and ema_200_current != 0:
                slope = ema_200_current - ema_200_prev
                return slope
            
            return None  # Slope calculation not available
            
        except Exception as e:
            self.logger.debug(f"EMA 200 slope calculation failed: {e}")
            return None
    
    def validate_zero_lag_trend(self, latest_row: pd.Series, signal_type: str) -> bool:
        """
        Validate Zero Lag trend alignment
        
        Args:
            latest_row: DataFrame row with Zero Lag data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if Zero Lag trend supports signal
        """
        try:
            close = latest_row.get('close', 0)
            zlema = latest_row.get('zlema', 0)
            trend_state = latest_row.get('trend', 0)
            zlema_slope = latest_row.get('zlema_slope', 0)
            
            if close == 0 or zlema == 0:
                self.logger.debug("Zero Lag trend validation: missing data")
                return True
            
            if signal_type == 'BULL':
                # Bull signals: RELAXED - price above ZLEMA OR positive momentum indicators
                price_above = close > zlema
                uptrend_or_neutral = trend_state >= -0.5  # More lenient (allow slight downtrend)
                positive_momentum = zlema_slope > -self.eps * 10  # Allow small negative slope
                
                # BULL is valid if price is above ZLEMA OR if momentum is positive
                bull_valid = price_above or (uptrend_or_neutral and positive_momentum)
                
                if bull_valid:
                    conditions = []
                    if price_above:
                        conditions.append("price above ZLEMA")
                    if uptrend_or_neutral:
                        conditions.append("neutral/up trend")
                    if positive_momentum:
                        conditions.append("positive momentum")
                    
                    self.logger.debug(f"✅ Zero Lag trend OK for BULL: {', '.join(conditions)}")
                    return True
                else:
                    self.logger.debug(f"❌ Zero Lag BULL trend: price below ZLEMA + negative momentum")
                    return False
            
            elif signal_type == 'BEAR':
                # Bear signals: RELAXED - price below ZLEMA OR negative momentum indicators
                price_below = close < zlema
                downtrend_or_neutral = trend_state <= 0.5  # More lenient (allow slight uptrend)
                negative_momentum = zlema_slope < self.eps * 10  # Allow small positive slope
                
                # BEAR is valid if price is below ZLEMA OR if momentum is negative
                bear_valid = price_below or (downtrend_or_neutral and negative_momentum)
                
                if bear_valid:
                    conditions = []
                    if price_below:
                        conditions.append("price below ZLEMA")
                    if downtrend_or_neutral:
                        conditions.append("neutral/down trend")
                    if negative_momentum:
                        conditions.append("negative momentum")
                    
                    self.logger.debug(f"✅ Zero Lag trend OK for BEAR: {', '.join(conditions)}")
                    return True
                else:
                    self.logger.debug(f"❌ Zero Lag BEAR trend: price above ZLEMA + positive momentum")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating Zero Lag trend: {e}")
            return True
    
    def validate_higher_timeframe_alignment(self, epic: str, current_time: pd.Timestamp, 
                                           signal_type: str, data_fetcher=None) -> bool:
        """
        Validate signal against higher timeframe trend (15m -> 1H)
        
        Args:
            epic: Trading instrument identifier
            current_time: Current timestamp
            signal_type: 'BULL' or 'BEAR'
            data_fetcher: Data fetcher instance
            
        Returns:
            True if higher timeframe confirms signal
        """
        try:
            if not getattr(config, 'HIGHER_TIMEFRAME_VALIDATION', True):
                return True
            
            if not data_fetcher:
                self.logger.debug("No data fetcher available for higher timeframe validation")
                return True
            
            # Extract pair from epic
            parts = epic.split('.')
            if len(parts) >= 3:
                pair = parts[2].replace('MINI', '').replace('CFD', '')
            else:
                self.logger.debug(f"Cannot extract pair from epic: {epic}")
                return True
            
            # Get 1H data for higher timeframe analysis
            df_1h = data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe='1h',
                lookback_hours=200
            )
            
            if df_1h is None or df_1h.empty:
                self.logger.debug(f"No 1H data available for {epic}")
                return True
            
            # Find the most recent 1H candle with proper timezone handling
            if 'start_time' in df_1h.columns:
                try:
                    # Ensure both timestamps are timezone-aware or timezone-naive for comparison
                    df_start_times = df_1h['start_time']
                    
                    # Convert current_time to timezone-naive if needed for comparison
                    if hasattr(current_time, 'tz_localize') and current_time.tz is not None:
                        # current_time is timezone-aware, convert to UTC then remove timezone
                        current_time_utc = current_time.tz_convert('UTC').tz_localize(None)
                    else:
                        # current_time is already timezone-naive
                        current_time_utc = current_time
                    
                    # Convert DataFrame start_times to timezone-naive if needed
                    if hasattr(df_start_times.iloc[0], 'tz') and df_start_times.iloc[0].tz is not None:
                        # DataFrame has timezone-aware timestamps, convert to UTC then remove timezone
                        df_start_times = df_start_times.dt.tz_convert('UTC').dt.tz_localize(None)
                    
                    # Now perform the comparison with both as timezone-naive
                    valid_candles = df_1h[df_start_times <= current_time_utc]
                    if valid_candles.empty:
                        latest_1h = df_1h.iloc[-1]
                    else:
                        latest_1h = valid_candles.iloc[-1]
                except Exception as tz_error:
                    self.logger.debug(f"Timezone comparison error in 1H, using latest candle: {tz_error}")
                    # Fallback to latest candle if timezone comparison fails
                    latest_1h = df_1h.iloc[-1]
            else:
                latest_1h = df_1h.iloc[-1]
            
            # Check 1H Zero Lag trend
            h1_close = latest_1h.get('close', 0)
            h1_zlema = latest_1h.get('zlema', 0) 
            h1_ema200 = latest_1h.get('ema_200', 0)
            h1_trend = latest_1h.get('trend', 0)
            
            if h1_close == 0 or h1_zlema == 0 or h1_ema200 == 0:
                self.logger.debug("1H data incomplete, allowing signal")
                return True
            
            if signal_type == 'BULL':
                # More balanced 1H bull signal validation - require majority of conditions
                price_above_ema200 = h1_close > h1_ema200
                price_above_zlema = h1_close > h1_zlema  
                trend_neutral_or_up = h1_trend >= -0.5  # Allow slight downtrend
                
                # Count positive conditions
                bull_conditions = [price_above_ema200, price_above_zlema, trend_neutral_or_up]
                bull_score = sum(bull_conditions)
                
                # BULL valid if 2 out of 3 conditions are met (majority rule)
                h1_bullish = bull_score >= 2
                
                if h1_bullish:
                    condition_details = []
                    if price_above_ema200: condition_details.append("above EMA200")
                    if price_above_zlema: condition_details.append("above ZLEMA")
                    if trend_neutral_or_up: condition_details.append("neutral/up trend")
                    
                    self.logger.debug(f"✅ 1H timeframe confirms BULL signal ({bull_score}/3): {', '.join(condition_details)}")
                    return True
                else:
                    failed_conditions = []
                    if not price_above_ema200: failed_conditions.append("below EMA200")
                    if not price_above_zlema: failed_conditions.append("below ZLEMA")
                    if not trend_neutral_or_up: failed_conditions.append("strong downtrend")
                    
                    self.logger.debug(f"❌ 1H timeframe does not support BULL signal ({bull_score}/3): {', '.join(failed_conditions)}")
                    return False
            
            elif signal_type == 'BEAR':
                # More balanced 1H bear signal validation - require majority of conditions
                price_below_ema200 = h1_close < h1_ema200
                price_below_zlema = h1_close < h1_zlema
                trend_neutral_or_down = h1_trend <= 0.5  # Allow slight uptrend
                
                # Count positive conditions
                bear_conditions = [price_below_ema200, price_below_zlema, trend_neutral_or_down]
                bear_score = sum(bear_conditions)
                
                # BEAR valid if 2 out of 3 conditions are met (majority rule)
                h1_bearish = bear_score >= 2
                
                if h1_bearish:
                    condition_details = []
                    if price_below_ema200: condition_details.append("below EMA200")
                    if price_below_zlema: condition_details.append("below ZLEMA")
                    if trend_neutral_or_down: condition_details.append("neutral/down trend")
                    
                    self.logger.debug(f"✅ 1H timeframe confirms BEAR signal ({bear_score}/3): {', '.join(condition_details)}")
                    return True
                else:
                    failed_conditions = []
                    if not price_below_ema200: failed_conditions.append("above EMA200")
                    if not price_below_zlema: failed_conditions.append("above ZLEMA")  
                    if not trend_neutral_or_down: failed_conditions.append("strong uptrend")
                    
                    self.logger.debug(f"❌ 1H timeframe does not support BEAR signal ({bear_score}/3): {', '.join(failed_conditions)}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating higher timeframe: {e}")
            return True  # Allow signal on error
    
    def validate_multi_timeframe_alignment(self, epic: str, current_time: pd.Timestamp, 
                                          signal_type: str, data_fetcher=None) -> Dict:
        """
        Validate signal against multiple higher timeframes (15m -> 1H -> 4H)
        
        Args:
            epic: Trading instrument identifier
            current_time: Current timestamp
            signal_type: 'BULL' or 'BEAR'
            data_fetcher: Data fetcher instance
            
        Returns:
            Dictionary with validation results for each timeframe
        """
        try:
            results = {
                'mtf_validation_enabled': getattr(config, 'HIGHER_TIMEFRAME_VALIDATION', True) and getattr(config, 'ZERO_LAG_MTF_ENABLED', True),
                'h1_validation': True,
                'h4_validation': True,
                'overall_valid': True,
                'validation_details': {},
                'strict_mode': getattr(config, 'ZERO_LAG_MTF_STRICT_MODE', False)
            }
            
            if not results['mtf_validation_enabled']:
                results['validation_details']['disabled'] = "MTF validation disabled in config"
                return results
            
            if not data_fetcher:
                self.logger.debug("No data fetcher available for multi-timeframe validation")
                results['validation_details']['no_data_fetcher'] = "Data fetcher not available"
                return results
            
            # Extract pair from epic
            parts = epic.split('.')
            if len(parts) >= 3:
                pair = parts[2].replace('MINI', '').replace('CFD', '')
            else:
                self.logger.debug(f"Cannot extract pair from epic: {epic}")
                results['validation_details']['pair_extraction_failed'] = f"Cannot extract pair from {epic}"
                return results
            
            # Validate 1H timeframe (if required)
            if getattr(config, 'ZERO_LAG_MTF_REQUIRE_1H', True):
                results['h1_validation'] = self._validate_single_timeframe(
                    epic, pair, current_time, signal_type, '1h', data_fetcher
                )
            else:
                results['h1_validation'] = True
                results['validation_details']['h1_skipped'] = "1H validation disabled in config"
            
            # Validate 4H timeframe (if required)
            if getattr(config, 'ZERO_LAG_MTF_REQUIRE_4H', True):
                results['h4_validation'] = self._validate_single_timeframe(
                    epic, pair, current_time, signal_type, '4h', data_fetcher
                )
            else:
                results['h4_validation'] = True
                results['validation_details']['h4_skipped'] = "4H validation disabled in config"
            
            # Overall validation logic
            if results['strict_mode']:
                # Strict mode: both 1H and 4H must pass
                results['overall_valid'] = results['h1_validation'] and results['h4_validation']
            else:
                # Lenient mode: either 1H or 4H can pass (or both disabled)
                # Give extra weight to 1H since it's more reactive and relevant for entries
                if results['h1_validation']:
                    # If 1H passes, that's usually sufficient for signal confirmation
                    results['overall_valid'] = True
                    results['validation_details']['primary_confirmation'] = "1H validation sufficient"
                elif results['h4_validation']:
                    # If only 4H passes, that's also acceptable (longer-term trend alignment)
                    results['overall_valid'] = True
                    results['validation_details']['primary_confirmation'] = "4H validation sufficient"
                else:
                    # Both timeframes failed
                    results['overall_valid'] = False
                    results['validation_details']['primary_confirmation'] = "Both timeframes failed"
            
            # Log results - only show failures for live trading
            h1_status = "✓" if results['h1_validation'] else "✗"
            h4_status = "✓" if results['h4_validation'] else "✗"
            mode_str = "STRICT" if results['strict_mode'] else "LENIENT"
            
            if results['overall_valid']:
                confirmation_reason = results['validation_details'].get('primary_confirmation', 'Both timeframes passed')
                self.logger.info(f"✅ MTF validation PASSED for {signal_type}: 1H={h1_status} 4H={h4_status} ({confirmation_reason})")
            else:
                # Only show failures at info level for live trading visibility
                self.logger.info(f"❌ MTF validation FAILED for {signal_type}: 1H={h1_status} 4H={h4_status} (Both timeframes failed)")
            
            # Add detailed results for debugging
            results['validation_details']['mode'] = mode_str
            results['validation_details']['h1_status'] = h1_status
            results['validation_details']['h4_status'] = h4_status
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe validation: {e}")
            # Return permissive result on error
            return {
                'mtf_validation_enabled': True,
                'h1_validation': True,
                'h4_validation': True,
                'overall_valid': True,
                'validation_details': {'error': str(e)}
            }
    
    def _validate_single_timeframe(self, epic: str, pair: str, current_time: pd.Timestamp,
                                  signal_type: str, timeframe: str, data_fetcher) -> bool:
        """
        Validate signal against a single higher timeframe
        
        Args:
            epic: Trading instrument identifier
            pair: Currency pair (e.g., 'EURUSD')
            current_time: Current timestamp
            signal_type: 'BULL' or 'BEAR'
            timeframe: Timeframe to validate ('1h', '4h')
            data_fetcher: Data fetcher instance
            
        Returns:
            True if timeframe confirms signal
        """
        try:
            # Get data for the specified timeframe using config settings
            if timeframe == '4h':
                lookback_hours = getattr(config, 'MTF_4H_LOOKBACK_HOURS', 800)
            elif timeframe == '1h':
                lookback_hours = getattr(config, 'MTF_1H_LOOKBACK_HOURS', 200)
            else:
                lookback_hours = 200  # Default fallback
            
            df_htf = data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=lookback_hours
            )
            
            if df_htf is None or df_htf.empty:
                self.logger.debug(f"No {timeframe.upper()} data available for {epic}")
                return True  # Allow signal if no data
            
            # Find the most recent candle with proper timezone handling
            if 'start_time' in df_htf.columns:
                try:
                    # Ensure both timestamps are timezone-aware or timezone-naive for comparison
                    df_start_times = df_htf['start_time']
                    
                    # Convert current_time to timezone-naive if needed for comparison
                    if hasattr(current_time, 'tz_localize') and current_time.tz is not None:
                        # current_time is timezone-aware, convert to UTC then remove timezone
                        current_time_utc = current_time.tz_convert('UTC').tz_localize(None)
                    else:
                        # current_time is already timezone-naive
                        current_time_utc = current_time
                    
                    # Convert DataFrame start_times to timezone-naive if needed
                    if hasattr(df_start_times.iloc[0], 'tz') and df_start_times.iloc[0].tz is not None:
                        # DataFrame has timezone-aware timestamps, convert to UTC then remove timezone
                        df_start_times = df_start_times.dt.tz_convert('UTC').dt.tz_localize(None)
                    
                    # Now perform the comparison with both as timezone-naive
                    valid_candles = df_htf[df_start_times <= current_time_utc]
                    if valid_candles.empty:
                        latest_htf = df_htf.iloc[-1]
                    else:
                        latest_htf = valid_candles.iloc[-1]
                except Exception as tz_error:
                    self.logger.debug(f"Timezone comparison error in {timeframe}, using latest candle: {tz_error}")
                    # Fallback to latest candle if timezone comparison fails
                    latest_htf = df_htf.iloc[-1]
            else:
                latest_htf = df_htf.iloc[-1]
            
            # Check Zero Lag trend alignment
            htf_close = latest_htf.get('close', 0)
            htf_zlema = latest_htf.get('zlema', 0) 
            htf_ema200 = latest_htf.get('ema_200', 0)
            htf_trend = latest_htf.get('trend', 0)
            
            if htf_close == 0 or htf_zlema == 0 or htf_ema200 == 0:
                self.logger.debug(f"{timeframe.upper()} data incomplete, allowing signal")
                return True
            
            if signal_type == 'BULL':
                # More balanced bull signal validation - require majority of conditions
                price_above_ema200 = htf_close > htf_ema200
                price_above_zlema = htf_close > htf_zlema  
                # For 4H timeframe, be even more lenient with trend since it moves slowly
                trend_threshold = -1.0 if timeframe == '4h' else -0.5
                trend_neutral_or_up = htf_trend >= trend_threshold  # More lenient for 4H
                
                # Count positive conditions
                bull_conditions = [price_above_ema200, price_above_zlema, trend_neutral_or_up]
                bull_score = sum(bull_conditions)
                
                # BULL valid if 2 out of 3 conditions are met (majority rule)
                htf_bullish = bull_score >= 2
                
                if htf_bullish:
                    condition_details = []
                    if price_above_ema200: condition_details.append("above EMA200")
                    if price_above_zlema: condition_details.append("above ZLEMA")
                    if trend_neutral_or_up: condition_details.append("neutral/up trend")
                    
                    self.logger.debug(f"✅ {timeframe.upper()} timeframe confirms BULL signal ({bull_score}/3): {', '.join(condition_details)}")
                    self.logger.debug(f"    {timeframe.upper()}: close={htf_close:.5f}, EMA200={htf_ema200:.5f}, ZLEMA={htf_zlema:.5f}, trend={htf_trend}")
                    return True
                else:
                    failed_conditions = []
                    if not price_above_ema200: failed_conditions.append("below EMA200")
                    if not price_above_zlema: failed_conditions.append("below ZLEMA")
                    if not trend_neutral_or_up: failed_conditions.append("strong downtrend")
                    
                    self.logger.debug(f"❌ {timeframe.upper()} timeframe does not support BULL signal ({bull_score}/3): {', '.join(failed_conditions)}")
                    self.logger.debug(f"    {timeframe.upper()}: close={htf_close:.5f}, EMA200={htf_ema200:.5f}, ZLEMA={htf_zlema:.5f}, trend={htf_trend}")
                    return False
            
            elif signal_type == 'BEAR':
                # More balanced bear signal validation - require majority of conditions
                price_below_ema200 = htf_close < htf_ema200
                price_below_zlema = htf_close < htf_zlema
                # For 4H timeframe, be even more lenient with trend since it moves slowly
                trend_threshold = 1.0 if timeframe == '4h' else 0.5
                trend_neutral_or_down = htf_trend <= trend_threshold  # More lenient for 4H
                
                # Count positive conditions
                bear_conditions = [price_below_ema200, price_below_zlema, trend_neutral_or_down]
                bear_score = sum(bear_conditions)
                
                # BEAR valid if 2 out of 3 conditions are met (majority rule)
                htf_bearish = bear_score >= 2
                
                if htf_bearish:
                    condition_details = []
                    if price_below_ema200: condition_details.append("below EMA200")
                    if price_below_zlema: condition_details.append("below ZLEMA")
                    if trend_neutral_or_down: condition_details.append("neutral/down trend")
                    
                    self.logger.debug(f"✅ {timeframe.upper()} timeframe confirms BEAR signal ({bear_score}/3): {', '.join(condition_details)}")
                    self.logger.debug(f"    {timeframe.upper()}: close={htf_close:.5f}, EMA200={htf_ema200:.5f}, ZLEMA={htf_zlema:.5f}, trend={htf_trend}")
                    return True
                else:
                    failed_conditions = []
                    if not price_below_ema200: failed_conditions.append("above EMA200")
                    if not price_below_zlema: failed_conditions.append("above ZLEMA")  
                    if not trend_neutral_or_down: failed_conditions.append("strong uptrend")
                    
                    self.logger.debug(f"❌ {timeframe.upper()} timeframe does not support BEAR signal ({bear_score}/3): {', '.join(failed_conditions)}")
                    self.logger.debug(f"    {timeframe.upper()}: close={htf_close:.5f}, EMA200={htf_ema200:.5f}, ZLEMA={htf_zlema:.5f}, trend={htf_trend}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating {timeframe} timeframe: {e}")
            return True  # Allow signal on error

    def validate_volatility_conditions(self, latest_row: pd.Series) -> bool:
        """
        Validate that volatility conditions are suitable for trading
        
        Args:
            latest_row: DataFrame row with volatility data
            
        Returns:
            True if volatility is within acceptable range
        """
        try:
            close = latest_row.get('close', 0)
            volatility = latest_row.get('volatility', 0)
            
            if close == 0 or volatility == 0:
                return True  # Allow if data not available
            
            volatility_ratio = volatility / close
            
            # Acceptable volatility range for forex
            min_volatility = getattr(config, 'MIN_VOLATILITY_RATIO', 0.0005)  # 0.05%
            max_volatility = getattr(config, 'MAX_VOLATILITY_RATIO', 0.05)    # 5%
            
            if min_volatility <= volatility_ratio <= max_volatility:
                self.logger.debug(f"✅ Volatility OK: {volatility_ratio:.4f} ({volatility_ratio*100:.2f}%)")
                return True
            else:
                if volatility_ratio < min_volatility:
                    self.logger.debug(f"❌ Volatility too low: {volatility_ratio:.4f} < {min_volatility:.4f}")
                else:
                    self.logger.debug(f"❌ Volatility too high: {volatility_ratio:.4f} > {max_volatility:.4f}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating volatility conditions: {e}")
            return True
    
    def check_market_session_filter(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if current time is within acceptable trading sessions
        
        Args:
            timestamp: Current market timestamp
            
        Returns:
            True if within trading session
        """
        try:
            if not getattr(config, 'SESSION_FILTER_ENABLED', False):
                return True
            
            # Convert to UTC hour
            utc_hour = timestamp.hour
            
            # Define major forex sessions (UTC)
            london_session = (7, 16)  # 8 AM - 5 PM London time
            ny_session = (12, 21)     # 1 PM - 10 PM London time (8 AM - 5 PM NY)
            
            # Allow trading during major sessions
            in_london = london_session[0] <= utc_hour <= london_session[1]
            in_ny = ny_session[0] <= utc_hour <= ny_session[1]
            
            if in_london or in_ny:
                session = "London" if in_london else "New York"
                self.logger.debug(f"✅ Trading session OK: {session} session active")
                return True
            else:
                self.logger.debug(f"❌ Outside major trading sessions (hour: {utc_hour} UTC)")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking market session: {e}")
            return True
    
    def get_trend_summary(self, latest_row: pd.Series) -> Dict:
        """
        Get comprehensive trend analysis summary
        
        Args:
            latest_row: Current market data row
            
        Returns:
            Dictionary with trend summary
        """
        try:
            close = latest_row.get('close', 0)
            zlema = latest_row.get('zlema', 0)
            ema_200 = latest_row.get('ema_200', 0)
            trend_state = latest_row.get('trend', 0)
            zlema_slope = latest_row.get('zlema_slope', 0)
            
            return {
                'close': close,
                'zlema': zlema,
                'ema_200': ema_200,
                'price_above_zlema': close > zlema if zlema != 0 else None,
                'price_above_ema200': close > ema_200 if ema_200 != 0 else None,
                'trend_state': trend_state,
                'trend_description': self._describe_trend_state(trend_state),
                'zlema_slope': zlema_slope,
                'zlema_slope_direction': 'rising' if zlema_slope > self.eps else 'falling' if zlema_slope < -self.eps else 'flat',
                'macro_trend': 'bullish' if close > ema_200 else 'bearish' if close < ema_200 else 'neutral',
                'zero_lag_bias': 'bullish' if close > zlema else 'bearish' if close < zlema else 'neutral'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating trend summary: {e}")
            return {}
    
    def _describe_trend_state(self, trend_state: int) -> str:
        """Convert numerical trend state to description"""
        if trend_state == 1:
            return "uptrend"
        elif trend_state == -1:
            return "downtrend"
        else:
            return "sideways"