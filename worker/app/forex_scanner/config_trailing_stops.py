# config_trailing_stops.py
"""
Progressive 3-Stage Trailing Stop Configuration with MFE Protection
Used by both live trading and backtest systems

v2.7.0 OPTIMIZATION (2025-12-23) - MFE/MAE Analysis Based:
- BE triggers optimized per-pair based on actual MFE/MAE data analysis
- Used weighted average of BUY/SELL optimal values (SELL often needs higher)
- Key insight: SELL positions typically need higher BE triggers than BUY
- Future: Consider direction-specific BE triggers in trailing stop manager

BUY vs SELL Analysis Summary (optimal_be_trigger):
  EURJPY: BUY=27, SELL=55 → Use 40 (SELL-weighted, high discrepancy)
  GBPUSD: BUY=25, SELL=46 → Use 35 (average, SELL needs more room)
  USDJPY: BUY=33, SELL=47 → Use 40 (average)
  EURUSD: BUY=24, SELL=12 → Use 20 (BUY-weighted, SELL is lower)
  NZDUSD: BUY=12, SELL=17 → Use 15 (both low, use average)
  AUDUSD: BUY=13, SELL=28 → Use 20 (SELL-weighted)
  USDCAD: BUY=20, SELL=18 → Use 20 (keep similar)
  USDCHF: BUY=26, SELL=25 → Use 26 (already optimized v2.6.0)
  AUDJPY: BUY=33, SELL=25 → Use 28 (BUY-weighted)

v2.5.0 OPTIMIZATION (2025-12-22):
- Increased all trigger points by ~40% to let winners develop further
- Increased lock points to capture more profit when triggered
- Problem solved: Winners were being cut short at 30-50% of TP target
- Analysis showed avg win 80 SEK vs avg loss 91 SEK (should be 2:1 ratio)

Stage 2.5 (MFE Protection):
- Triggers when profit reaches mfe_protection_threshold_pct of target (default: 70%)
- AND profit declines by mfe_protection_decline_pct from peak (default: 10%)
- Locks mfe_protection_lock_pct of MFE (default: 60%)
- Example: Target=30 pips, MFE=26 pips (86% of target)
  - If profit drops to 23.4 pips (10% decline from 26), trigger protection
  - Lock 60% of 26 = 15.6 pips minimum exit
"""

# 🆕 Stage 2.5: Global MFE Protection defaults (can be overridden per pair)
MFE_PROTECTION_DEFAULTS = {
    'mfe_protection_threshold_pct': 0.70,  # Trigger when profit >= 70% of target
    'mfe_protection_decline_pct': 0.10,    # Trigger on 10% decline from peak MFE
    'mfe_protection_lock_pct': 0.60,       # Lock 60% of MFE when triggered
}

PAIR_TRAILING_CONFIGS = {
    # ========== MAJOR PAIRS - Standard Volatility ==========

    # CEEM epic uses scaled pricing (11646 instead of 1.1646), 1 pip = 1 point
    'CS.D.EURUSD.CEEM.IP': {
        # v2.7.0: MFE/MAE Analysis - BUY opt=24, SELL opt=12 → Use 20 (BUY-weighted)
        # SELL has lower optimal BE but BUY needs more room
        'stage1_trigger_points': 25,      # v2.7.0: 25 (was 28) - adjusted to optimal
        'stage1_lock_points': 10,         # v2.7.0: 10 (was 12) - proportional
        'stage2_trigger_points': 35,      # v2.7.0: 35 (was 38) - adjusted
        'stage2_lock_points': 22,         # v2.7.0: 22 (was 25) - proportional
        'stage3_trigger_points': 42,      # v2.7.0: 42 (was 45) - adjusted
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 20,  # v2.7.0: 20 (was 25) - from MFE/MAE analysis
    },

    'CS.D.AUDUSD.MINI.IP': {
        # v2.7.0: MFE/MAE Analysis - BUY opt=13, SELL opt=28 → Use 20 (SELL-weighted)
        # BUY shows lower MFE, SELL needs more room before BE
        'stage1_trigger_points': 25,      # v2.7.0: 25 (was 28) - adjusted
        'stage1_lock_points': 10,         # v2.7.0: 10 (was 12) - proportional
        'stage2_trigger_points': 35,      # v2.7.0: 35 (was 38) - adjusted
        'stage2_lock_points': 22,         # v2.7.0: 22 (was 25) - proportional
        'stage3_trigger_points': 42,      # v2.7.0: 42 (was 45) - adjusted
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 20,  # v2.7.0: 20 (was 25) - from MFE/MAE analysis
    },

    'CS.D.NZDUSD.MINI.IP': {
        # v2.7.0: MFE/MAE Analysis - BUY opt=12, SELL opt=17 → Use 15 (average)
        # Both directions show lower optimal BE - NZDUSD less volatile
        'stage1_trigger_points': 20,      # v2.7.0: 20 (was 28) - significant reduction
        'stage1_lock_points': 8,          # v2.7.0: 8 (was 12) - proportional
        'stage2_trigger_points': 28,      # v2.7.0: 28 (was 38) - adjusted
        'stage2_lock_points': 18,         # v2.7.0: 18 (was 25) - proportional
        'stage3_trigger_points': 35,      # v2.7.0: 35 (was 45) - adjusted
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 12,         # v2.7.0: 12 (was 15) - NZDUSD less volatile
        'break_even_trigger_points': 15,  # v2.7.0: 15 (was 25) - from MFE/MAE analysis
    },

    'CS.D.USDCAD.MINI.IP': {
        # v2.7.0: MFE/MAE Analysis - BUY opt=20, SELL opt=18 → Use 20 (keep similar)
        # Both directions agree, current config is close to optimal
        'stage1_trigger_points': 25,      # v2.7.0: 25 (was 28) - slight adjustment
        'stage1_lock_points': 10,         # v2.7.0: 10 (was 12) - proportional
        'stage2_trigger_points': 35,      # v2.7.0: 35 (was 38) - adjusted
        'stage2_lock_points': 22,         # v2.7.0: 22 (was 25) - proportional
        'stage3_trigger_points': 42,      # v2.7.0: 42 (was 45) - adjusted
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 20,  # v2.7.0: 20 (was 25) - from MFE/MAE analysis
    },

    'CS.D.USDCHF.MINI.IP': {
        # v2.7.0: MFE/MAE Analysis - BUY opt=26, SELL opt=25 → Use 26 (confirmed v2.6.0 was close)
        # v2.6.0 USDCHF OPTIMIZATION (2025-12-23):
        # Analysis: 40.7% win rate, -4,917 SEK loss, stops too tight (9-13 pips getting hit by noise)
        # Solution: Increase all triggers by 25% to let trades develop, wider stops for CHF volatility
        'stage1_trigger_points': 32,      # v2.7.0: 32 (was 35) - reduced slightly based on MFE data
        'stage1_lock_points': 14,         # v2.7.0: 14 (was 15) - proportional
        'stage2_trigger_points': 42,      # v2.7.0: 42 (was 48) - adjusted to optimal
        'stage2_lock_points': 28,         # v2.7.0: 28 (was 32) - proportional
        'stage3_trigger_points': 50,      # v2.7.0: 50 (was 56) - adjusted
        'stage3_atr_multiplier': 0.9,     # v2.6.0: 0.9 (was 0.8) - wider ATR trailing for CHF
        'stage3_min_distance': 4,         # v2.6.0: 4 (was 2) - more breathing room
        'min_trail_distance': 18,         # v2.6.0: 18 (was 15) - prevent noise stopouts
        'break_even_trigger_points': 26,  # v2.7.0: 26 (was 32) - from MFE/MAE analysis
    },

    # ========== GBP PAIRS - High Volatility ==========

    'CS.D.GBPUSD.MINI.IP': {
        # v2.7.0: MFE/MAE Analysis - BUY opt=25, SELL opt=46 → Use 35 (SELL-weighted)
        # SELL positions need significantly more room before BE trigger
        'stage1_trigger_points': 40,      # v2.7.0: 40 (was 35) - increased for SELL
        'stage1_lock_points': 16,         # v2.7.0: 16 (was 15) - proportional
        'stage2_trigger_points': 52,      # v2.7.0: 52 (was 45) - adjusted
        'stage2_lock_points': 34,         # v2.7.0: 34 (was 30) - protect more
        'stage3_trigger_points': 60,      # v2.7.0: 60 (was 55) - let SELL run
        'stage3_atr_multiplier': 1.0,     # Wider trailing
        'stage3_min_distance': 3,         # More distance
        'min_trail_distance': 18,         # Higher minimum
        'break_even_trigger_points': 35,  # v2.7.0: 35 (was 30) - compromise BUY=25/SELL=46
    },

    'CS.D.GBPJPY.MINI.IP': {
        'stage1_trigger_points': 32,      # v2.5.0: 32 (was 22) - GBP/JPY very volatile
        'stage1_lock_points': 12,         # v2.5.0: 12 (was 6) - lock more profit
        'stage2_trigger_points': 42,      # v2.5.0: 42 (was 28) - closer to TP target
        'stage2_lock_points': 28,         # v2.5.0: 28 (was 18) - protect 60%+ of target
        'stage3_trigger_points': 50,      # v2.5.0: 50 (was 32) - let trade run
        'stage3_atr_multiplier': 1.0,
        'stage3_min_distance': 3,
        'min_trail_distance': 18,
        'break_even_trigger_points': 28,  # v2.5.0: 28 (was 18) - GBP/JPY needs room
    },

    'CS.D.GBPAUD.MINI.IP': {
        'stage1_trigger_points': 15,
        'stage1_lock_points': 3,
        'stage2_trigger_points': 20,
        'stage2_lock_points': 12,
        'stage3_trigger_points': 22,
        'stage3_atr_multiplier': 1.0,
        'stage3_min_distance': 3,
        'min_trail_distance': 18,
        'break_even_trigger_points': 8,
    },

    'CS.D.GBPNZD.MINI.IP': {
        'stage1_trigger_points': 15,
        'stage1_lock_points': 3,
        'stage2_trigger_points': 20,
        'stage2_lock_points': 12,
        'stage3_trigger_points': 22,
        'stage3_atr_multiplier': 1.0,
        'stage3_min_distance': 3,
        'min_trail_distance': 18,
        'break_even_trigger_points': 8,
    },

    # ========== JPY PAIRS - Different Pip Scale ==========

    'CS.D.USDJPY.MINI.IP': {
        # v2.7.0: MFE/MAE Analysis - BUY opt=33, SELL opt=47 → Use 40 (average)
        # Both directions need higher BE than current, SELL especially
        'stage1_trigger_points': 45,      # v2.7.0: 45 (was 28) - significant increase
        'stage1_lock_points': 18,         # v2.7.0: 18 (was 10) - proportional
        'stage2_trigger_points': 55,      # v2.7.0: 55 (was 38) - adjusted
        'stage2_lock_points': 35,         # v2.7.0: 35 (was 22) - protect more
        'stage3_trigger_points': 65,      # v2.7.0: 65 (was 45) - let trade run
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 15,         # v2.7.0: 15 (was 12) - increased
        'break_even_trigger_points': 40,  # v2.7.0: 40 (was 25) - from MFE/MAE analysis
    },

    'CS.D.EURJPY.MINI.IP': {
        # v2.7.0: MFE/MAE Analysis - BUY opt=27, SELL opt=55 → Use 40 (SELL-weighted)
        # Huge discrepancy! SELL needs 2x higher BE than BUY
        'stage1_trigger_points': 45,      # v2.7.0: 45 (was 35) - increased for SELL
        'stage1_lock_points': 18,         # v2.7.0: 18 (was 12) - proportional
        'stage2_trigger_points': 55,      # v2.7.0: 55 (was 48) - adjusted
        'stage2_lock_points': 36,         # v2.7.0: 36 (was 30) - protect more
        'stage3_trigger_points': 65,      # v2.7.0: 65 (was 55) - let trade run
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 3,         # v2.7.0: 3 (was 2) - more room
        'min_trail_distance': 18,         # v2.7.0: 18 (was 15) - increased
        'break_even_trigger_points': 40,  # v2.7.0: 40 (was 28) - compromise BUY=27/SELL=55
    },

    'CS.D.AUDJPY.MINI.IP': {
        # v2.7.0: MFE/MAE Analysis - BUY opt=33, SELL opt=25 → Use 28 (BUY-weighted)
        # BUY needs more room, SELL is fine with current settings
        'stage1_trigger_points': 32,      # v2.7.0: 32 (was 25) - increased for BUY
        'stage1_lock_points': 13,         # v2.7.0: 13 (was 10) - proportional
        'stage2_trigger_points': 42,      # v2.7.0: 42 (was 32) - adjusted
        'stage2_lock_points': 26,         # v2.7.0: 26 (was 20) - protect more
        'stage3_trigger_points': 50,      # v2.7.0: 50 (was 38) - let trade run
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 4,
        'min_trail_distance': 15,
        'break_even_trigger_points': 28,  # v2.7.0: 28 (was 20) - from MFE/MAE analysis
        # Profit Protection Rule
        'enable_profit_protection': False,
    },

    'CS.D.CADJPY.MINI.IP': {
        'stage1_trigger_points': 17,      # v2.5.0: 17 (was 12) - let trade develop more
        'stage1_lock_points': 6,          # v2.5.0: 6 (was 2) - lock more profit
        'stage2_trigger_points': 22,      # v2.5.0: 22 (was 16) - closer to TP target
        'stage2_lock_points': 14,         # v2.5.0: 14 (was 10) - protect 60%+ of target
        'stage3_trigger_points': 25,      # v2.5.0: 25 (was 18) - let trade run
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 14,  # v2.5.0: 14 (was 7) - more breathing room
    },

    'CS.D.CHFJPY.MINI.IP': {
        'stage1_trigger_points': 17,      # v2.5.0: 17 (was 12) - let trade develop more
        'stage1_lock_points': 6,          # v2.5.0: 6 (was 2) - lock more profit
        'stage2_trigger_points': 22,      # v2.5.0: 22 (was 16) - closer to TP target
        'stage2_lock_points': 14,         # v2.5.0: 14 (was 10) - protect 60%+ of target
        'stage3_trigger_points': 25,      # v2.5.0: 25 (was 18) - let trade run
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 14,  # v2.5.0: 14 (was 7) - more breathing room
    },

    'CS.D.NZDJPY.MINI.IP': {
        'stage1_trigger_points': 17,      # v2.5.0: 17 (was 12) - let trade develop more
        'stage1_lock_points': 6,          # v2.5.0: 6 (was 2) - lock more profit
        'stage2_trigger_points': 22,      # v2.5.0: 22 (was 16) - closer to TP target
        'stage2_lock_points': 14,         # v2.5.0: 14 (was 10) - protect 60%+ of target
        'stage3_trigger_points': 25,      # v2.5.0: 25 (was 18) - let trade run
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 14,  # v2.5.0: 14 (was 7) - more breathing room
    },

    # ========== CROSS PAIRS - Medium-High Volatility ==========

    'CS.D.EURAUD.MINI.IP': {
        'stage1_trigger_points': 20,      # v2.5.0: 20 (was 14) - let trade develop more
        'stage1_lock_points': 8,          # v2.5.0: 8 (was 3) - lock more profit
        'stage2_trigger_points': 25,      # v2.5.0: 25 (was 18) - closer to TP target
        'stage2_lock_points': 16,         # v2.5.0: 16 (was 11) - protect 60%+ of target
        'stage3_trigger_points': 28,      # v2.5.0: 28 (was 20) - let trade run
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 16,
        'break_even_trigger_points': 15,  # v2.5.0: 15 (was 7) - more breathing room
    },

    'CS.D.EURNZD.MINI.IP': {
        'stage1_trigger_points': 20,      # v2.5.0: 20 (was 14) - let trade develop more
        'stage1_lock_points': 8,          # v2.5.0: 8 (was 3) - lock more profit
        'stage2_trigger_points': 25,      # v2.5.0: 25 (was 18) - closer to TP target
        'stage2_lock_points': 16,         # v2.5.0: 16 (was 11) - protect 60%+ of target
        'stage3_trigger_points': 28,      # v2.5.0: 28 (was 20) - let trade run
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 16,
        'break_even_trigger_points': 15,  # v2.5.0: 15 (was 7) - more breathing room
    },

    'CS.D.AUDNZD.MINI.IP': {
        'stage1_trigger_points': 18,      # v2.5.0: 18 (was 13) - let trade develop more
        'stage1_lock_points': 6,          # v2.5.0: 6 (was 2) - lock more profit
        'stage2_trigger_points': 24,      # v2.5.0: 24 (was 17) - closer to TP target
        'stage2_lock_points': 14,         # v2.5.0: 14 (was 10) - protect 60%+ of target
        'stage3_trigger_points': 27,      # v2.5.0: 27 (was 19) - let trade run
        'stage3_atr_multiplier': 0.85,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 14,  # v2.5.0: 14 (was 7) - more breathing room
    },

    'CS.D.EURGBP.MINI.IP': {
        'stage1_trigger_points': 15,      # v2.5.0: 15 (was 11) - let trade develop more
        'stage1_lock_points': 6,          # v2.5.0: 6 (was 2) - lock more profit
        'stage2_trigger_points': 20,      # v2.5.0: 20 (was 14) - closer to TP target
        'stage2_lock_points': 13,         # v2.5.0: 13 (was 9) - protect 60%+ of target
        'stage3_trigger_points': 22,      # v2.5.0: 22 (was 16) - let trade run
        'stage3_atr_multiplier': 0.75,
        'stage3_min_distance': 2,
        'min_trail_distance': 14,
        'break_even_trigger_points': 15,  # v2.5.0: 15 (was 12) - more breathing room
    },
}

# =============================================================================
# SCALP MODE TRAILING STOP CONFIGURATION
# =============================================================================
# Scalp-specific trailing configs with tighter triggers and faster profit locks
# Used when --scalp mode is enabled in backtests
#
# v3.1.0 UPDATES (Jan 2026):
# - Stage 2: Moved to 15 pips → lock +10 pips (solid profit before trailing)
# - Stage 3: Moved to 17 pips → dynamic trailing starts (was 15)
# - Benefit: Locks better profit before tight trailing begins
# =============================================================================

SCALP_TRAILING_CONFIGS = {
    # ========== USDCAD - HIGHEST PRIORITY (100% success rate) ==========
    'CS.D.USDCAD.MINI.IP': {
        'early_breakeven_trigger_points': 6,    # Quick BE protection
        'early_breakeven_buffer_points': 1,     # Lock +1 pip
        'stage1_trigger_points': 10,            # Stage1 at +10 pips
        'stage1_lock_points': 5,                # Lock +5 pips
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,           # Tighter ATR for scalping
        'stage3_min_distance': 5,
        'min_trail_distance': 6,                # 12 pip initial stop (optimal from data)
        'break_even_trigger_points': 6,
        'enable_partial_close': False,
        'partial_close_trigger_points': 10,
        'partial_close_size': 0.5,
    },

    # ========== MAJOR PAIRS (15 pip optimal stop) ==========
    'CS.D.EURUSD.CEEM.IP': {
        'early_breakeven_trigger_points': 8,    # From data: avoid premature BE
        'early_breakeven_buffer_points': 1,     # Lock +1 pip
        'stage1_trigger_points': 10,            # Stage1 at +10 pips
        'stage1_lock_points': 5,                # Lock +5 pips
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 5,
        'min_trail_distance': 8,                # 15 pip initial stop
        'break_even_trigger_points': 8,
        'enable_partial_close': False,
        'partial_close_trigger_points': 12,
        'partial_close_size': 0.5,
    },

    'CS.D.GBPUSD.MINI.IP': {
        'early_breakeven_trigger_points': 8,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 5,
        'min_trail_distance': 8,                # 15 pip initial stop
        'break_even_trigger_points': 8,
        'enable_partial_close': False,
        'partial_close_trigger_points': 12,
        'partial_close_size': 0.5,
    },

    'CS.D.AUDUSD.MINI.IP': {
        'early_breakeven_trigger_points': 6,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 4,
        'min_trail_distance': 8,                # 15 pip initial stop
        'break_even_trigger_points': 6,
        'enable_partial_close': False,
        'partial_close_trigger_points': 10,
        'partial_close_size': 0.5,
    },

    'CS.D.NZDUSD.MINI.IP': {
        'early_breakeven_trigger_points': 8,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 5,
        'min_trail_distance': 8,                # 15 pip initial stop
        'break_even_trigger_points': 8,
        'enable_partial_close': False,
        'partial_close_trigger_points': 12,
        'partial_close_size': 0.5,
    },

    'CS.D.USDCHF.MINI.IP': {
        'early_breakeven_trigger_points': 8,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 5,
        'min_trail_distance': 8,                # 15 pip initial stop
        'break_even_trigger_points': 8,
        'enable_partial_close': False,
        'partial_close_trigger_points': 12,
        'partial_close_size': 0.5,
    },

    # ========== JPY PAIRS (20 pip optimal stop - higher volatility) ==========
    'CS.D.USDJPY.MINI.IP': {
        'early_breakeven_trigger_points': 10,   # JPY pairs need more room
        'early_breakeven_buffer_points': 1.5,   # Lock +1.5 pips (higher volatility)
        'stage1_trigger_points': 15,            # Stage1 at +15 pips
        'stage1_lock_points': 8,                # Lock +8 pips
        'stage2_trigger_points': 20,            # Stage2 at +20 pips - lock solid profit
        'stage2_lock_points': 15,               # Lock +15 pips
        'stage3_trigger_points': 22,            # Stage3 dynamic trailing starts at +22
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 6,
        'min_trail_distance': 10,               # 20 pip initial stop
        'break_even_trigger_points': 10,
        'enable_partial_close': False,
        'partial_close_trigger_points': 15,
        'partial_close_size': 0.5,
    },

    'CS.D.EURJPY.MINI.IP': {
        'early_breakeven_trigger_points': 10,
        'early_breakeven_buffer_points': 1.5,
        'stage1_trigger_points': 15,
        'stage1_lock_points': 8,
        'stage2_trigger_points': 20,            # Stage2 at +20 pips - lock solid profit
        'stage2_lock_points': 15,               # Lock +15 pips
        'stage3_trigger_points': 22,            # Stage3 dynamic trailing starts at +22
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 6,
        'min_trail_distance': 10,               # 20 pip initial stop
        'break_even_trigger_points': 10,
        'enable_partial_close': False,
        'partial_close_trigger_points': 15,
        'partial_close_size': 0.5,
    },

    'CS.D.AUDJPY.MINI.IP': {
        'early_breakeven_trigger_points': 10,
        'early_breakeven_buffer_points': 1.5,
        'stage1_trigger_points': 15,
        'stage1_lock_points': 8,
        'stage2_trigger_points': 20,            # Stage2 at +20 pips - lock solid profit
        'stage2_lock_points': 15,               # Lock +15 pips
        'stage3_trigger_points': 22,            # Stage3 dynamic trailing starts at +22
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 6,
        'min_trail_distance': 10,               # 20 pip initial stop
        'break_even_trigger_points': 10,
        'enable_partial_close': False,
        'partial_close_trigger_points': 15,
        'partial_close_size': 0.5,
    },
}
