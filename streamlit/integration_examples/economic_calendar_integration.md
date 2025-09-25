# Economic Calendar Integration for TradingView Chart

This document shows how to integrate economic calendar data into the existing `tvchart.py` page.

## 🎯 Integration Points

### 1. Import Section (Line ~18)
Add these imports after the existing imports:

```python
# Add to existing imports section
from components.economic_calendar_widget import (
    render_economic_calendar_sidebar,
    render_upcoming_alerts,
    generate_chart_markers,
    render_market_intelligence_enhancement
)
```

### 2. Sidebar Economic Calendar Section (After Line ~365)
Add this section after the "Market Intelligence" expander and before "Trading Tools":

```python
# Economic Calendar Section - ADD THIS BLOCK
st.sidebar.divider()
economic_events = None
try:
    # Render upcoming alerts first (high priority events)
    render_upcoming_alerts(selected_epic)

    # Render main economic calendar sidebar
    economic_events = render_economic_calendar_sidebar(selected_epic)

except Exception as e:
    st.sidebar.error(f"Economic Calendar unavailable: {str(e)[:40]}...")
    logger.error(f"Economic calendar error: {e}")
```

### 3. Chart Markers Integration (After Line ~1188)
Add economic calendar markers to the existing marker system:

```python
# Add to existing marker processing section (around line 1188)
# After: visible_markers = visible_swing_markers + aligned_trade_markers + visible_sr_break_markers + aligned_risk_markers + aligned_regime_markers

# Add economic calendar markers
economic_markers = []
if economic_events and candles:
    try:
        from components.economic_calendar_widget import generate_chart_markers
        candle_times_list = [c["time"] for c in candles]
        economic_markers = generate_chart_markers(
            economic_events,
            candle_times_list,
            timeframe
        )
        logger.info(f"Added {len(economic_markers)} economic calendar markers")
    except Exception as e:
        logger.warning(f"Failed to add economic markers: {e}")

# Update visible_markers to include economic markers
visible_markers = (visible_swing_markers + aligned_trade_markers +
                  visible_sr_break_markers + aligned_risk_markers +
                  aligned_regime_markers + economic_markers)
```

### 4. Market Intelligence Enhancement (After Line ~1542)
Enhance the existing market intelligence display:

```python
# Enhance existing market intelligence section (around line 1542)
# After: st.warning(f"{regime_icon} **{regime_name.upper()} REGIME** ({confidence:.1%} confidence) | Volatility: {volatility_pct:.0f}th percentile")

# Add economic calendar enhancement
try:
    render_market_intelligence_enhancement(
        selected_epic,
        {
            'regime_name': regime_name,
            'confidence': confidence,
            'volatility_percentile': volatility_pct
        }
    )
except Exception as e:
    logger.warning(f"Failed to enhance market intelligence: {e}")
```

### 5. Chart Legend Update (Line ~462)
Add economic calendar legend to the existing legend section:

```python
# Add to existing legend (around line 462, add new column)
with col5:  # Or create col6 if needed
    st.markdown("""
    **Economic Calendar:**
    - 🔴 Red Arrow = High Impact Event
    - 🟡 Yellow Arrow = Medium Impact
    - 🟢 Green Arrow = Low Impact
    - *Arrows show event timing*
    """)
```

## 🚀 Quick Test Integration

For a minimal test, add just this single line after the epic selection (around line 141):

```python
# MINIMAL TEST - Add after: selected_epic = selected_epic_display
try:
    from services.economic_calendar_service import get_economic_calendar_service
    calendar_service = get_economic_calendar_service()
    base_currency, quote_currency = calendar_service.extract_currencies_from_epic(selected_epic)
    st.sidebar.success(f"📅 Economic Calendar Ready: {base_currency}/{quote_currency}")

    # Show next high-impact event as test
    high_impact = calendar_service.get_upcoming_high_impact_events(selected_epic, 48)
    if high_impact:
        next_event = high_impact[0]
        event_name = next_event.get('event_name', 'Unknown')
        currency = next_event.get('currency', '')
        st.sidebar.info(f"⚡ Next High Impact: {currency} {event_name}")
except Exception as e:
    st.sidebar.warning(f"Economic Calendar: {str(e)[:40]}...")
```

## 🎨 UI Display Examples

### Sidebar Widget Display
```
┌─ 📅 Economic Calendar - EUR/USD ─┐
│ ☑️ Today  ☑️ Tomorrow  ☐ Week   │
│ Impact: [High] [Medium] [Low]    │
│                                  │
│ 🔴 **EUR**              In 2h    │
│ ECB Interest Rate Decision       │
│ Prev: 4.00%  Fcst: 4.25%       │
│ ────────────────────────────────  │
│                                  │
│ 🟡 **USD**              In 6h    │
│ Core CPI m/m                     │
│ Prev: 0.3%   Fcst: 0.2%         │
│                                  │
│ ☑️ Show on Chart                 │
└──────────────────────────────────┘
```

### Alert Display
```
┌────── Alert ──────┐
│ ⚠️ EUR High Impact │
│ ECB Interest Rate  │
│ Expected: 4.25%    │
│ Previous: 4.00%    │
│ **In 2h 15m**      │
└────────────────────┘
```

### Enhanced Market Intelligence
```
🎯 TRENDING REGIME (87.3%) | 📅 ECB Decision In 2h
⚡ High volatility expected - Consider adjusting position sizes
```

## 🔧 Configuration Options

The integration supports these configuration options:

- **Impact Filter**: High/Medium/Low events
- **Time Range**: Today, Tomorrow, This Week
- **Chart Markers**: Optional timeline markers
- **Alert Threshold**: Hours ahead for alerts
- **Currency Auto-Detection**: From epic format

## 📊 Data Flow

```
Selected Epic → Currency Extraction → API Call → Event Filtering → UI Display
     ↓               ↓                  ↓            ↓             ↓
"EURUSD"    →    [EUR, USD]    →    Events[]   →   Relevant[]  → Sidebar
```

## ⚡ Performance Notes

- API calls are made only when sidebar is expanded
- Events are cached in session state
- Maximum 10 events displayed in sidebar
- Chart markers limited to visible timeframe
- Automatic error handling and fallbacks

## 🧪 Testing

1. **Currency Extraction Test**: Verify currencies are extracted correctly from various epic formats
2. **API Connection Test**: Ensure economic-calendar service is accessible
3. **Event Filtering Test**: Confirm only relevant currency events are shown
4. **Chart Marker Test**: Verify markers appear at correct times on chart
5. **Error Handling Test**: Graceful degradation when service unavailable