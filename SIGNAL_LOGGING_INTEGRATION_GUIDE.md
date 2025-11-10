# Signal Decision Logging Integration Guide

This guide shows where to add logging calls in `smc_structure_strategy.py` to capture all rejection/approval decisions.

## Helper Method Added

```python
def _log_decision(self, timestamp, epic, pair, direction, decision, rejection_reason=None, rejection_step=None):
    """Log signal decision with current context."""
    if not self.decision_logger:
        return

    self.decision_logger.log_signal_decision(
        timestamp=timestamp,
        epic=epic,
        pair=pair,
        direction=direction,
        decision=decision,
        rejection_reason=rejection_reason,
        rejection_step=rejection_step,
        **self._current_decision_context
    )
    self._current_decision_context = {}
```

## Logging Locations in `detect_smc_structure_signals()`

### 1. COOLDOWN REJECTION (~line 387)

**After:**
```python
if not can_trade:
    self.logger.info(f"   ❌ Signal rejected: {reason}")
    return None
```

**Add:**
```python
if not can_trade:
    self.logger.info(f"   ❌ Signal rejected: {reason}")
    self._current_decision_context.update({'cooldown_active': True})
    self._log_decision(current_time, epic, pair, 'unknown', 'REJECTED', 'COOLDOWN_ACTIVE', 'COOLDOWN_CHECK')
    return None
```

### 2. SESSION FILTER REJECTION (~line 396)

**After:**
```python
if self.session_filter_enabled and not is_valid_session:
    self.logger.info(f"   ❌ Signal blocked: Asian session")
    return None
```

**Add:**
```python
if self.session_filter_enabled and not is_valid_session:
    self.logger.info(f"   ❌ Signal blocked: Asian session")
    self._current_decision_context.update({'session_valid': False})
    self._log_decision(current_time, epic, pair, 'unknown', 'REJECTED', 'SESSION_FILTERED', 'SESSION_CHECK')
    return None
```

### 3. HTF TREND DATA (Store for later logging, ~line 454)

**After:**
```python
trend_analysis = self.trend_analyzer.analyze_structure(...)
```

**Add:**
```python
# Store HTF data in context for logging
self._current_decision_context.update({
    'htf_trend': final_trend,
    'htf_strength': final_strength,
    'htf_structure': trend_analysis['structure_type'],
    'htf_in_pullback': trend_analysis['in_pullback'],
    'htf_pullback_depth': trend_analysis['pullback_depth']
})
```

### 4. HTF MISALIGNMENT REJECTION (~line 459)

**After:**
```python
if not bos_choch_direction:
    self.logger.info(f"   ❌ Signal rejected: No BOS/CHoCH detected on 15m")
    return None
```

**Add:**
```python
if not bos_choch_direction:
    self.logger.info(f"   ❌ Signal rejected: No BOS/CHoCH detected on 15m")
    self._current_decision_context.update({'bos_detected': False})
    self._log_decision(current_time, epic, pair, direction_str, 'REJECTED', 'NO_BOS_CHOCH', 'BOS_DETECTION')
    return None
```

### 5. BOS QUALITY DATA (~line 577)

**After:**
```python
if bos_choch_info:
    # Store BOS data
```

**Add:**
```python
self._current_decision_context.update({
    'bos_detected': True,
    'bos_direction': bos_choch_info['direction'],
    'bos_quality': bos_choch_info.get('quality', 0)
})
```

### 6. LOW BOS QUALITY REJECTION (~line 598)

**After:**
```python
if bos_choch_info['quality'] < MIN_BOS_QUALITY:
    self.logger.info(f"   ❌ Weak BOS/CHoCH detected - quality too low")
    return None
```

**Add:**
```python
if quality_score < MIN_BOS_QUALITY:
    self.logger.info(f"   ❌ Weak BOS/CHoCH detected - quality too low")
    self._log_decision(current_time, epic, pair, direction_str, 'REJECTED', 'LOW_BOS_QUALITY', 'BOS_DETECTION')
    return None
```

### 7. ORDER BLOCK DATA (~line 690)

**After:**
```python
# Store OB data if found
```

**Add:**
```python
self._current_decision_context.update({
    'ob_found': ob_level is not None,
    'ob_distance_pips': ob_distance_pips if ob_level else None
})
```

### 8. PATTERN DATA (~line 796)

**After:**
```python
rejection_pattern = self.pattern_detector.detect_rejection_patterns(...)
```

**Add:**
```python
self._current_decision_context.update({
    'pattern_found': rejection_pattern is not None,
    'pattern_type': rejection_pattern['pattern_type'] if rejection_pattern else None,
    'pattern_strength': rejection_pattern['strength'] if rejection_pattern else None
})
```

### 9. SR DATA (~line 780)

**After:**
```python
nearest_level = self.sr_detector.get_nearest_level(...)
```

**Add:**
```python
self._current_decision_context.update({
    'sr_level': nearest_level['price'],
    'sr_type': nearest_level['type'],
    'sr_strength': nearest_level['strength'],
    'sr_distance_pips': nearest_level['distance_pips']
})
```

### 10. PREMIUM/DISCOUNT DATA (~line 810)

**After:**
```python
zone_info = self.market_structure.get_premium_discount_zone(...)
```

**Add:**
```python
if zone_info:
    self._current_decision_context.update({
        'premium_discount_zone': zone_info['zone'],
        'entry_quality': zone_info['entry_quality_buy'] if direction_str == 'bullish' else zone_info['entry_quality_sell'],
        'zone_position_pct': zone_info['price_position'] * 100
    })
```

### 11. PREMIUM/DISCOUNT REJECTION (~line 844, 862)

**After bullish premium rejection:**
```python
if zone == 'premium' and not (is_strong_trend and final_trend == 'BULL'):
    self.logger.info(f"   ❌ BULLISH entry in PREMIUM zone - poor timing")
    return None
```

**Add:**
```python
    self._log_decision(current_time, epic, pair, 'bullish', 'REJECTED', 'PREMIUM_DISCOUNT_REJECT', 'PREMIUM_DISCOUNT_CHECK')
    return None
```

### 12. R:R AND CONFIDENCE DATA (~line 1009)

**After:**
```python
confidence = htf_score + pattern_score + sr_score + rr_score
```

**Add:**
```python
self._current_decision_context.update({
    'risk_pips': risk_pips,
    'reward_pips': reward_pips,
    'rr_ratio': rr_ratio,
    'confidence': confidence,
    'htf_score': htf_score,
    'pattern_score': pattern_score,
    'sr_score': sr_score,
    'rr_score': rr_score,
    'entry_price': entry_price,
    'stop_loss': stop_loss,
    'take_profit': take_profit
})
```

### 13. LOW CONFIDENCE REJECTION (~line 1022)

**After:**
```python
if confidence < MIN_CONFIDENCE:
    self.logger.info(f"   ❌ Signal confidence too low")
    return None
```

**Add:**
```python
    self._log_decision(current_time, epic, pair, direction_str, 'REJECTED', 'LOW_CONFIDENCE', 'CONFIDENCE_CHECK')
    return None
```

### 14. SIGNAL APPROVED (~line 1120)

**Before return signal:**
```python
# Log approved signal
self._log_decision(current_time, epic, pair, direction_str, 'APPROVED')

return signal
```

## Summary

Total logging points: **14 locations**
- 12 rejection points
- 1 approval point
- Multiple context update points

Each rejection captures:
- What filter rejected it
- All available filter values at that point
- Rejection reason and step

This provides complete audit trail of every signal evaluation.
