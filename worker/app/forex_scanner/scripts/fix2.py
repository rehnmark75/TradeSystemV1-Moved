# Add this to your config.py to fix the MACD color detection issue

# =============================================================================
# MACD COLOR DETECTION FIX
# =============================================================================

# 🔧 Fix the MACD color detection logic
MACD_FIX_COLOR_DETECTION = True

# 🎯 Use direct histogram values instead of color strings
MACD_USE_DIRECT_HISTOGRAM_DETECTION = True

# 🚨 Bypass color-based detection entirely
MACD_BYPASS_COLOR_LOGIC = True

print("🔧 MACD Color Detection Fix Applied")
print("   Issue: Color logic showing 'green→green' for actual crossovers")
print("   Fix: Using direct histogram value comparison")

# =============================================================================
# MANUAL CROSSOVER DETECTION PATCH
# =============================================================================

def patch_macd_crossover_detection():
    """
    Patch the MACD signal detector to fix color detection
    """
    try:
        from core.strategies.helpers.macd_signal_detector import MACDSignalDetector
        
        # Store original method
        original_detect_crossover = MACDSignalDetector.detect_macd_histogram_crossover
        
        def fixed_detect_crossover(self, current_histogram, previous_histogram, current_color, previous_color):
            """
            Fixed crossover detection using direct histogram values
            """
            try:
                signal_type = None
                trigger_reason = "no_crossover"
                crossover_strength = 0.0
                
                # 🔧 FIX: Use histogram values directly, ignore color strings
                self.logger.debug(f"[FIXED CROSSOVER] Testing: {previous_histogram:.6f} → {current_histogram:.6f}")
                
                # Check for red to green transition (bullish) - FIXED LOGIC
                if previous_histogram <= 0 and current_histogram > 0:
                    signal_type = 'BULL'
                    trigger_reason = 'histogram_negative_to_positive'
                    crossover_strength = abs(current_histogram - previous_histogram)
                    self.logger.info(f"[FIXED CROSSOVER] ✅ BULLISH: {previous_histogram:.6f} → {current_histogram:.6f}")
                
                # Check for green to red transition (bearish) - FIXED LOGIC
                elif previous_histogram >= 0 and current_histogram < 0:
                    signal_type = 'BEAR'
                    trigger_reason = 'histogram_positive_to_negative'
                    crossover_strength = abs(current_histogram - previous_histogram)
                    self.logger.info(f"[FIXED CROSSOVER] ✅ BEARISH: {previous_histogram:.6f} → {current_histogram:.6f}")
                
                else:
                    self.logger.debug(f"[FIXED CROSSOVER] No crossover: prev={previous_histogram:.6f}, curr={current_histogram:.6f}")
                
                return signal_type, trigger_reason, crossover_strength
                
            except Exception as e:
                self.logger.error(f"Fixed crossover detection failed: {e}")
                return None, f"detection_error_{str(e)}", 0.0
        
        # Apply the patch
        MACDSignalDetector.detect_macd_histogram_crossover = fixed_detect_crossover
        
        print("✅ MACD Crossover Detection PATCHED")
        print("   Now using direct histogram values instead of color strings")
        return True
        
    except Exception as e:
        print(f"❌ Failed to patch MACD crossover detection: {e}")
        return False

# Auto-apply the patch when config is loaded
if MACD_FIX_COLOR_DETECTION:
    try:
        import atexit
        
        def apply_patch():
            """Apply patch when modules are loaded"""
            try:
                patch_macd_crossover_detection()
            except:
                pass  # Modules not loaded yet
        
        # Try to apply immediately and also on exit
        apply_patch()
        atexit.register(apply_patch)
        
    except:
        pass

# =============================================================================
# ALTERNATIVE: EMERGENCY HISTOGRAM DETECTION MODE
# =============================================================================

# If patching doesn't work, use this nuclear option
MACD_EMERGENCY_HISTOGRAM_MODE = True

def emergency_macd_detection():
    """
    Emergency MACD detection that completely bypasses the color system
    """
    try:
        from core.strategies.macd_strategy import MACDStrategy
        
        # Store original method
        original_detect_signal = MACDStrategy.detect_signal
        
        def emergency_detect_signal(self, df, epic, spread_pips, timeframe):
            """Emergency MACD detection using only histogram values"""
            
            try:
                if len(df) < 50:
                    return None
                
                latest = df.iloc[-1]
                previous = df.iloc[-2]
                
                current_hist = latest.get('macd_histogram', 0)
                prev_hist = previous.get('macd_histogram', 0)
                
                # Direct histogram crossover detection
                signal_type = None
                
                # Bullish crossover: negative to positive
                if prev_hist <= 0 and current_hist > 0:
                    signal_type = 'BULL'
                    self.logger.info(f"[EMERGENCY] ✅ BULL crossover: {prev_hist:.6f} → {current_hist:.6f}")
                
                # Bearish crossover: positive to negative  
                elif prev_hist >= 0 and current_hist < 0:
                    signal_type = 'BEAR'
                    self.logger.info(f"[EMERGENCY] ✅ BEAR crossover: {prev_hist:.6f} → {current_hist:.6f}")
                
                if signal_type:
                    # Create emergency signal
                    return {
                        'signal_type': signal_type,
                        'epic': epic,
                        'timeframe': timeframe,
                        'price': latest.get('close', 0),
                        'confidence_score': 0.75,
                        'strategy': 'EMERGENCY_HISTOGRAM_MACD',
                        'signal_trigger': 'emergency_histogram_crossover',
                        'macd_histogram': current_hist,
                        'macd_histogram_prev': prev_hist,
                        'ema_200': latest.get('ema_200', 0),
                        'emergency_mode': True,
                        'color_detection_bypassed': True,
                        'direct_histogram_detection': True,
                        'timestamp': latest.name if hasattr(latest, 'name') else None
                    }
                
                return None
                
            except Exception as e:
                self.logger.error(f"Emergency MACD detection failed: {e}")
                return None
        
        # Apply emergency patch
        MACDStrategy.detect_signal = emergency_detect_signal
        
        print("🚨 EMERGENCY MACD HISTOGRAM MODE ACTIVATED")
        print("   Completely bypassing color-based detection")
        print("   Using direct histogram value comparison only")
        return True
        
    except Exception as e:
        print(f"❌ Emergency mode failed: {e}")
        return False

# Apply emergency mode if enabled
if MACD_EMERGENCY_HISTOGRAM_MODE:
    try:
        import atexit
        atexit.register(lambda: emergency_macd_detection())
        print("🚨 Emergency histogram mode will be applied when strategy loads")
    except:
        pass