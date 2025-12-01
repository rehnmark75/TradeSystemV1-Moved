# config_trailing_stops.py
"""
Progressive 3-Stage Trailing Stop Configuration with MFE Protection
Used by both live trading and backtest systems

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

    'CS.D.EURUSD.MINI.IP': {
        'stage1_trigger_points': 20,      # Break-even trigger (was 16) - v1.5.0
        'stage1_lock_points': 6,          # Minimum profit lock (was 4) - lock MORE
        'stage2_trigger_points': 28,      # Profit lock trigger (was 20)
        'stage2_lock_points': 18,         # Profit guarantee (was 12)
        'stage3_trigger_points': 32,      # Start percentage trailing (was 23)
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 18,   # Move to BE after 18 pts (was 12) - CRITICAL
    },

    'CS.D.AUDUSD.MINI.IP': {
        'stage1_trigger_points': 20,      # Break-even trigger (was 16) - v1.5.0
        'stage1_lock_points': 6,          # Minimum profit lock (was 4)
        'stage2_trigger_points': 28,      # Profit lock trigger (was 22)
        'stage2_lock_points': 18,         # Profit guarantee (was 12)
        'stage3_trigger_points': 32,      # Start percentage trailing (was 23)
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 18,  # v1.5.0: 18 (was 12) - give trades room to breathe
    },

    'CS.D.NZDUSD.MINI.IP': {
        'stage1_trigger_points': 20,      # Break-even trigger (was 16) - v1.5.0
        'stage1_lock_points': 6,          # Minimum profit lock (was 4)
        'stage2_trigger_points': 28,      # Profit lock trigger (was 22)
        'stage2_lock_points': 18,         # Profit guarantee (was 12)
        'stage3_trigger_points': 32,      # Start percentage trailing (was 23)
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 18,  # v1.5.0: 18 (was 12)
    },

    'CS.D.USDCAD.MINI.IP': {
        'stage1_trigger_points': 20,      # Break-even trigger (was 16) - v1.5.0
        'stage1_lock_points': 6,          # Minimum profit lock (was 4)
        'stage2_trigger_points': 28,      # Profit lock trigger (was 22)
        'stage2_lock_points': 18,         # Profit guarantee (was 12)
        'stage3_trigger_points': 32,      # Start percentage trailing (was 23)
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 4,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 18,  # v1.5.0: 18 (was 12)
    },

    'CS.D.USDCHF.MINI.IP': {
        'stage1_trigger_points': 20,      # Break-even trigger (was 16) - v1.5.0
        'stage1_lock_points': 6,          # Minimum profit lock (was 4)
        'stage2_trigger_points': 28,      # Profit lock trigger (was 22)
        'stage2_lock_points': 18,         # Profit guarantee (was 12)
        'stage3_trigger_points': 32,      # Start percentage trailing (was 23)
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 2,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 18,  # v1.5.0: 18 (was 12)
    },

    # ========== GBP PAIRS - High Volatility ==========

    'CS.D.GBPUSD.MINI.IP': {
        'stage1_trigger_points': 25,      # Wider activation (was 20) - v1.5.0
        'stage1_lock_points': 8,          # More profit lock (was 4)
        'stage2_trigger_points': 32,      # Higher trigger (was 25)
        'stage2_lock_points': 20,         # More protection (was 15)
        'stage3_trigger_points': 38,      # Later trailing start (was 30)
        'stage3_atr_multiplier': 1.0,     # Wider trailing
        'stage3_min_distance': 3,         # More distance
        'min_trail_distance': 18,         # Higher minimum
        'break_even_trigger_points': 20,   # v1.5.0: 20 (was 12) - GBP needs more room
    },

    'CS.D.GBPJPY.MINI.IP': {
        'stage1_trigger_points': 22,      # Wide for high volatility (was 15) - v1.5.0
        'stage1_lock_points': 6,          # (was 3)
        'stage2_trigger_points': 28,      # (was 20)
        'stage2_lock_points': 18,         # (was 12)
        'stage3_trigger_points': 32,      # (was 22)
        'stage3_atr_multiplier': 1.0,
        'stage3_min_distance': 3,
        'min_trail_distance': 18,
        'break_even_trigger_points': 18,  # v1.5.0: 18 (was 8) - critical fix
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
        'stage1_trigger_points': 20,      # Tighter for JPY scale (was 15) - v1.5.0
        'stage1_lock_points': 5,          # (was 2)
        'stage2_trigger_points': 28,      # Adjusted for JPY (was 20)
        'stage2_lock_points': 15,         # (was 10)
        'stage3_trigger_points': 32,      # (was 25)
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 12,
        'break_even_trigger_points': 18,  # v1.5.0: 18 (was 12)
    },

    'CS.D.EURJPY.MINI.IP': {
        'stage1_trigger_points': 25,      # Slightly wider (more volatile) (was 20) - v1.5.0
        'stage1_lock_points': 6,          # (was 4)
        'stage2_trigger_points': 35,      # (was 30)
        'stage2_lock_points': 18,         # (was 10)
        'stage3_trigger_points': 40,      # (was 35)
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 18,  # v1.5.0: 18 (was 12)
    },

    'CS.D.AUDJPY.MINI.IP': {
        'stage1_trigger_points': 18,
        'stage1_lock_points': 4,
        'stage2_trigger_points': 22,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 26,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 4,
        'min_trail_distance': 15,
        'break_even_trigger_points': 12,
        # Profit Protection Rule
        'enable_profit_protection': False,
    },

    'CS.D.CADJPY.MINI.IP': {
        'stage1_trigger_points': 12,
        'stage1_lock_points': 2,
        'stage2_trigger_points': 16,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 18,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 7,
    },

    'CS.D.CHFJPY.MINI.IP': {
        'stage1_trigger_points': 12,
        'stage1_lock_points': 2,
        'stage2_trigger_points': 16,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 18,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 7,
    },

    'CS.D.NZDJPY.MINI.IP': {
        'stage1_trigger_points': 12,
        'stage1_lock_points': 2,
        'stage2_trigger_points': 16,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 18,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 7,
    },

    # ========== CROSS PAIRS - Medium-High Volatility ==========

    'CS.D.EURAUD.MINI.IP': {
        'stage1_trigger_points': 14,
        'stage1_lock_points': 3,
        'stage2_trigger_points': 18,
        'stage2_lock_points': 11,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 16,
        'break_even_trigger_points': 7,
    },

    'CS.D.EURNZD.MINI.IP': {
        'stage1_trigger_points': 14,
        'stage1_lock_points': 3,
        'stage2_trigger_points': 18,
        'stage2_lock_points': 11,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 16,
        'break_even_trigger_points': 7,
    },

    'CS.D.AUDNZD.MINI.IP': {
        'stage1_trigger_points': 13,
        'stage1_lock_points': 2,
        'stage2_trigger_points': 17,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 19,
        'stage3_atr_multiplier': 0.85,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 7,
    },

    'CS.D.EURGBP.MINI.IP': {
        'stage1_trigger_points': 11,      # Tighter (less volatile)
        'stage1_lock_points': 2,
        'stage2_trigger_points': 14,
        'stage2_lock_points': 9,
        'stage3_trigger_points': 16,
        'stage3_atr_multiplier': 0.75,
        'stage3_min_distance': 2,
        'min_trail_distance': 14,
        'break_even_trigger_points': 12,
    },
}
