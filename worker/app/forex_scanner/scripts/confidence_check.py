# confidence_diagnostic.py
"""
BB+Supertrend Confidence Scoring Diagnostic
Diagnoses why signals are being rejected due to low confidence

Run from: forex_scanner/scripts/
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Add the parent directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def diagnose_confidence_scoring(epic: str = 'CS.D.EURUSD.MINI.IP'):
    """Diagnose why confidence scoring is rejecting signals"""
    logger.info("🔍 BB+Supertrend Confidence Scoring Diagnostic...")
    
    try:
        import config
        from core.database import DatabaseManager
        from core.signal_detector import SignalDetector
        
        # Initialize components
        db_manager = DatabaseManager(config.DATABASE_URL)
        detector = SignalDetector(db_manager, getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm'))
        
        # Get data
        pair = 'EURUSD'
        if hasattr(config, 'PAIR_INFO'):
            pair_info = config.PAIR_INFO.get(epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
        
        df = detector.data_fetcher.get_enhanced_data(epic, pair, timeframe='15m', lookback_hours=72)
        
        if df is None or len(df) < 50:
            logger.error("❌ Insufficient data for analysis")
            return False
        
        logger.info(f"📊 Analyzing {len(df)} bars for confidence scoring issues...")
        
        # Find bars where conditions are met
        signal_conditions_met = []
        confidence_scores = []
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Check buy conditions
            buy_bb = current['close'] <= current.get('bb_lower', float('inf'))
            buy_st = current.get('supertrend_direction') == 1
            buy_cross = previous['close'] > previous.get('bb_lower', 0)
            
            # Check sell conditions
            sell_bb = current['close'] >= current.get('bb_upper', 0)
            sell_st = current.get('supertrend_direction') == -1
            sell_cross = previous['close'] < previous.get('bb_upper', float('inf'))
            
            conditions_met = (buy_bb and buy_st and buy_cross) or (sell_bb and sell_st and sell_cross)
            
            if conditions_met:
                signal_type = 'BUY' if (buy_bb and buy_st and buy_cross) else 'SELL'
                timestamp = current.get('start_time', f'Bar {i}')
                
                logger.info(f"\n📍 {timestamp}: {signal_type} conditions met")
                logger.info(f"   Price: {current['close']:.5f}")
                logger.info(f"   BB Upper: {current.get('bb_upper', 'N/A'):.5f}")
                logger.info(f"   BB Lower: {current.get('bb_lower', 'N/A'):.5f}")
                logger.info(f"   Supertrend: {current.get('supertrend', 'N/A'):.5f}")
                logger.info(f"   ST Direction: {current.get('supertrend_direction', 'N/A')}")
                
                # Calculate confidence factors manually
                confidence_factors = {}
                
                try:
                    # Base confidence
                    base_confidence = getattr(config, 'BB_SUPERTREND_BASE_CONFIDENCE', 0.65)
                    confidence_factors['base_score'] = base_confidence
                    logger.info(f"   Base Confidence: {base_confidence:.3f}")
                    
                    # BB position factor
                    bb_range = current['bb_upper'] - current['bb_lower']
                    if bb_range > 0:
                        if signal_type == 'BUY':
                            bb_position = (current['close'] - current['bb_lower']) / bb_range
                            bb_factor = max(0, 1 - bb_position)
                        else:  # SELL
                            bb_position = (current['bb_upper'] - current['close']) / bb_range
                            bb_factor = max(0, 1 - bb_position)
                        
                        confidence_factors['bb_position'] = bb_factor
                        logger.info(f"   BB Position Factor: {bb_factor:.3f}")
                    else:
                        confidence_factors['bb_position'] = 0.5
                        logger.info(f"   BB Position Factor: 0.5 (no range)")
                    
                    # Supertrend strength
                    if current['supertrend'] > 0 and current['close'] > 0:
                        st_distance = abs(current['close'] - current['supertrend']) / current['close']
                        st_factor = min(1.0, st_distance * 10)
                        confidence_factors['supertrend_strength'] = st_factor
                        logger.info(f"   Supertrend Strength: {st_factor:.3f}")
                    else:
                        confidence_factors['supertrend_strength'] = 0.5
                        logger.info(f"   Supertrend Strength: 0.5 (default)")
                    
                    # Volume confirmation
                    if 'volume' in df.columns and len(df) >= 20:
                        avg_volume = df['volume'].rolling(20).mean().iloc[i]
                        if avg_volume > 0:
                            volume_ratio = current['volume'] / avg_volume
                            volume_factor = min(1.0, volume_ratio / 2)
                            confidence_factors['volume_confirmation'] = volume_factor
                            logger.info(f"   Volume Factor: {volume_factor:.3f} (ratio: {volume_ratio:.2f})")
                        else:
                            confidence_factors['volume_confirmation'] = 0.5
                            logger.info(f"   Volume Factor: 0.5 (no avg volume)")
                    else:
                        confidence_factors['volume_confirmation'] = 0.5
                        logger.info(f"   Volume Factor: 0.5 (no volume data)")
                    
                    # BB squeeze factor
                    if i >= 20:
                        recent_df = df.iloc[i-19:i+1]
                        avg_bb_width = (recent_df['bb_upper'] - recent_df['bb_lower']).mean()
                        current_bb_width = current['bb_upper'] - current['bb_lower']
                        if avg_bb_width > 0:
                            squeeze_factor = 1 - (current_bb_width / avg_bb_width)
                            squeeze_factor = max(0, min(1, squeeze_factor + 0.5))
                            confidence_factors['bb_squeeze'] = squeeze_factor
                            logger.info(f"   BB Squeeze Factor: {squeeze_factor:.3f}")
                        else:
                            confidence_factors['bb_squeeze'] = 0.5
                            logger.info(f"   BB Squeeze Factor: 0.5 (no width data)")
                    else:
                        confidence_factors['bb_squeeze'] = 0.5
                        logger.info(f"   BB Squeeze Factor: 0.5 (insufficient history)")
                    
                    # Calculate final confidence
                    weights = {
                        'base_score': 0.3,
                        'bb_position': 0.25,
                        'supertrend_strength': 0.2,
                        'volume_confirmation': 0.15,
                        'bb_squeeze': 0.1
                    }
                    
                    total_score = 0
                    total_weight = 0
                    
                    logger.info(f"   📊 Confidence Calculation:")
                    for factor, value in confidence_factors.items():
                        if factor in weights:
                            weight = weights[factor]
                            contribution = value * weight
                            total_score += contribution
                            total_weight += weight
                            logger.info(f"     {factor}: {value:.3f} × {weight:.2f} = {contribution:.3f}")
                    
                    if total_weight > 0:
                        final_confidence = total_score / total_weight
                    else:
                        final_confidence = base_confidence
                    
                    # Apply max confidence cap
                    max_confidence = getattr(config, 'BB_SUPERTREND_MAX_CONFIDENCE', 0.95)
                    final_confidence = min(final_confidence, max_confidence)
                    
                    logger.info(f"   📈 Final Confidence: {final_confidence:.3f}")
                    
                    # Check against threshold
                    min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.7)
                    logger.info(f"   🎯 Min Required: {min_confidence:.3f}")
                    
                    if final_confidence >= min_confidence:
                        logger.info(f"   ✅ SIGNAL WOULD BE ACCEPTED!")
                    else:
                        logger.info(f"   ❌ SIGNAL REJECTED (confidence too low)")
                        logger.info(f"   💡 Confidence gap: {min_confidence - final_confidence:.3f}")
                    
                    confidence_scores.append(final_confidence)
                    signal_conditions_met.append({
                        'timestamp': timestamp,
                        'signal_type': signal_type,
                        'confidence': final_confidence,
                        'factors': confidence_factors.copy()
                    })
                    
                except Exception as e:
                    logger.error(f"   ❌ Error calculating confidence: {e}")
        
        # Summary
        logger.info(f"\n📊 CONFIDENCE ANALYSIS SUMMARY:")
        logger.info(f"   Total conditions met: {len(signal_conditions_met)}")
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            max_confidence_found = max(confidence_scores)
            min_confidence_found = min(confidence_scores)
            
            logger.info(f"   Average confidence: {avg_confidence:.3f}")
            logger.info(f"   Max confidence: {max_confidence_found:.3f}")
            logger.info(f"   Min confidence: {min_confidence_found:.3f}")
            
            min_required = getattr(config, 'MIN_CONFIDENCE', 0.7)
            signals_that_would_pass = len([c for c in confidence_scores if c >= min_required])
            
            logger.info(f"   Required threshold: {min_required:.3f}")
            logger.info(f"   Signals that would pass: {signals_that_would_pass}/{len(confidence_scores)}")
            
            if signals_that_would_pass == 0:
                logger.error("❌ NO SIGNALS PASS CONFIDENCE THRESHOLD!")
                logger.info("💡 SOLUTIONS:")
                logger.info(f"   1. Lower MIN_CONFIDENCE to {max_confidence_found:.3f}")
                logger.info(f"   2. Lower BB_SUPERTREND_BASE_CONFIDENCE to 0.5")
                logger.info(f"   3. Adjust BB parameters for better positioning")
                
                # Find the best confidence and suggest threshold
                if max_confidence_found > 0:
                    suggested_threshold = max_confidence_found - 0.05
                    logger.info(f"   💡 Suggested MIN_CONFIDENCE: {suggested_threshold:.3f}")
            else:
                logger.info(f"✅ {signals_that_would_pass} signals would pass with current settings")
        else:
            logger.warning("⚠️ No signal conditions met in analysis period")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Confidence diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_immediate_fixes():
    """Suggest immediate configuration fixes"""
    logger.info("\n🔧 IMMEDIATE FIXES TO TRY:")
    logger.info("1. Lower confidence thresholds in config.py:")
    logger.info("   MIN_CONFIDENCE = 0.5")
    logger.info("   BB_SUPERTREND_BASE_CONFIDENCE = 0.5")
    logger.info("   BB_SUPERTREND_MAX_CONFIDENCE = 0.8")
    
    logger.info("\n2. More sensitive BB parameters:")
    logger.info("   BB_PERIOD = 14")
    logger.info("   BB_STD_DEV = 1.8")
    
    logger.info("\n3. More responsive Supertrend:")
    logger.info("   SUPERTREND_PERIOD = 8")
    logger.info("   SUPERTREND_MULTIPLIER = 2.5")
    
    logger.info("\n4. Test command after changes:")
    logger.info("   python main.py debug-bb-supertrend --epic CS.D.EURUSD.MINI.IP")

if __name__ == '__main__':
    try:
        epic = sys.argv[1] if len(sys.argv) > 1 else 'CS.D.EURUSD.MINI.IP'
        
        logger.info(f"🔍 Starting confidence diagnostic for {epic}")
        
        success = diagnose_confidence_scoring(epic)
        
        if not success:
            logger.error("❌ Confidence diagnostic failed!")
        else:
            logger.info("✅ Confidence diagnostic completed!")
        
        suggest_immediate_fixes()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Diagnostic interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()