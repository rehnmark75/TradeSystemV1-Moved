# config_trailing_stops.py
"""
Progressive 3-Stage Trailing Stop Configuration with MFE Protection
Used by both live trading and backtest systems

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
        'stage1_trigger_points': 28,      # v2.5.0: 28 (was 20) - let trade develop more
        'stage1_lock_points': 12,         # v2.5.0: 12 (was 6) - lock more profit
        'stage2_trigger_points': 38,      # v2.5.0: 38 (was 28) - closer to TP target
        'stage2_lock_points': 25,         # v2.5.0: 25 (was 18) - protect 60%+ of target
        'stage3_trigger_points': 45,      # v2.5.0: 45 (was 32) - let trade run
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 25,  # v2.5.0: 25 (was 18) - more breathing room
    },

    'CS.D.AUDUSD.MINI.IP': {
        'stage1_trigger_points': 28,      # v2.5.0: 28 (was 20) - let trade develop more
        'stage1_lock_points': 12,         # v2.5.0: 12 (was 6) - lock more profit
        'stage2_trigger_points': 38,      # v2.5.0: 38 (was 28) - closer to TP target
        'stage2_lock_points': 25,         # v2.5.0: 25 (was 18) - protect 60%+ of target
        'stage3_trigger_points': 45,      # v2.5.0: 45 (was 32) - let trade run
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 25,  # v2.5.0: 25 (was 18) - more breathing room
    },

    'CS.D.NZDUSD.MINI.IP': {
        'stage1_trigger_points': 28,      # v2.5.0: 28 (was 20) - let trade develop more
        'stage1_lock_points': 12,         # v2.5.0: 12 (was 6) - lock more profit
        'stage2_trigger_points': 38,      # v2.5.0: 38 (was 28) - closer to TP target
        'stage2_lock_points': 25,         # v2.5.0: 25 (was 18) - protect 60%+ of target
        'stage3_trigger_points': 45,      # v2.5.0: 45 (was 32) - let trade run
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 25,  # v2.5.0: 25 (was 18) - more breathing room
    },

    'CS.D.USDCAD.MINI.IP': {
        'stage1_trigger_points': 28,      # v2.5.0: 28 (was 20) - let trade develop more
        'stage1_lock_points': 12,         # v2.5.0: 12 (was 6) - lock more profit
        'stage2_trigger_points': 38,      # v2.5.0: 38 (was 28) - closer to TP target
        'stage2_lock_points': 25,         # v2.5.0: 25 (was 18) - protect 60%+ of target
        'stage3_trigger_points': 45,      # v2.5.0: 45 (was 32) - let trade run
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 25,  # v2.5.0: 25 (was 18) - more breathing room
    },

    'CS.D.USDCHF.MINI.IP': {
        'stage1_trigger_points': 28,      # v2.5.0: 28 (was 20) - let trade develop more
        'stage1_lock_points': 12,         # v2.5.0: 12 (was 6) - lock more profit
        'stage2_trigger_points': 38,      # v2.5.0: 38 (was 28) - closer to TP target
        'stage2_lock_points': 25,         # v2.5.0: 25 (was 18) - protect 60%+ of target
        'stage3_trigger_points': 45,      # v2.5.0: 45 (was 32) - let trade run
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 2,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 25,  # v2.5.0: 25 (was 18) - more breathing room
    },

    # ========== GBP PAIRS - High Volatility ==========

    'CS.D.GBPUSD.MINI.IP': {
        'stage1_trigger_points': 35,      # v2.5.0: 35 (was 25) - GBP needs more room
        'stage1_lock_points': 15,         # v2.5.0: 15 (was 8) - lock more profit
        'stage2_trigger_points': 45,      # v2.5.0: 45 (was 32) - closer to TP target
        'stage2_lock_points': 30,         # v2.5.0: 30 (was 20) - protect 60%+ of target
        'stage3_trigger_points': 55,      # v2.5.0: 55 (was 38) - let trade run
        'stage3_atr_multiplier': 1.0,     # Wider trailing
        'stage3_min_distance': 3,         # More distance
        'min_trail_distance': 18,         # Higher minimum
        'break_even_trigger_points': 30,  # v2.5.0: 30 (was 20) - GBP is volatile
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
        'stage1_trigger_points': 28,      # v2.5.0: 28 (was 20) - let trade develop more
        'stage1_lock_points': 10,         # v2.5.0: 10 (was 5) - lock more profit
        'stage2_trigger_points': 38,      # v2.5.0: 38 (was 28) - closer to TP target
        'stage2_lock_points': 22,         # v2.5.0: 22 (was 15) - protect 60%+ of target
        'stage3_trigger_points': 45,      # v2.5.0: 45 (was 32) - let trade run
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 12,
        'break_even_trigger_points': 25,  # v2.5.0: 25 (was 18) - more breathing room
    },

    'CS.D.EURJPY.MINI.IP': {
        'stage1_trigger_points': 35,      # v2.5.0: 35 (was 25) - let trade develop more
        'stage1_lock_points': 12,         # v2.5.0: 12 (was 6) - lock more profit
        'stage2_trigger_points': 48,      # v2.5.0: 48 (was 35) - closer to TP target
        'stage2_lock_points': 30,         # v2.5.0: 30 (was 18) - protect 60%+ of target
        'stage3_trigger_points': 55,      # v2.5.0: 55 (was 40) - let trade run
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 28,  # v2.5.0: 28 (was 18) - EUR/JPY is volatile
    },

    'CS.D.AUDJPY.MINI.IP': {
        'stage1_trigger_points': 25,      # v2.5.0: 25 (was 18) - let trade develop more
        'stage1_lock_points': 10,         # v2.5.0: 10 (was 4) - lock more profit
        'stage2_trigger_points': 32,      # v2.5.0: 32 (was 22) - closer to TP target
        'stage2_lock_points': 20,         # v2.5.0: 20 (was 10) - protect 60%+ of target
        'stage3_trigger_points': 38,      # v2.5.0: 38 (was 26) - let trade run
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 4,
        'min_trail_distance': 15,
        'break_even_trigger_points': 20,  # v2.5.0: 20 (was 12) - more breathing room
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
