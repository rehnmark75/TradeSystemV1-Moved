#!/bin/bash
# Follow-up ablation rows for XAU_GOLD (Apr 22 2026).
#
# The initial inverse-ablation in scripts/ablate_xau_gold.py tested each gate
# individually on top of a permissive baseline, but:
#   * block_ranging + adx_trending_threshold are a coupled pair — neither has
#     an effect when the other is at baseline values. They must be toggled
#     together.
#   * cooldown=0 in the permissive baseline inflates signals/month ~10x vs
#     any realistic deployment, distorting PF interpretation.
#   * min_confidence's binding behaviour needed a positive control at 0.75.
#
# This script runs those three follow-up rows to close the verification loop.
#
# Results (2026-04-22, 90d window, CS.D.CFEGOLD.CEE.IP, SL/TP 40/80 XAU pips):
#   Row A (permissive + OB/FVG + cd_30):               n=111  PF 2.35  WR 54.1%
#   Row B (A + block_ranging + ADX>=25):               n= 70  PF 3.83  WR 65.7%
#   Row C (permissive + min_confidence=0.75):          n=  0  (binding confirmed)
#
# Usage:
#   docker exec task-worker /app/forex_scanner/scripts/ablate_xau_followups.sh

set -e
BASE="docker exec task-worker python /app/forex_scanner/backtest_cli.py --epic CS.D.CFEGOLD.CEE.IP --days 90 --strategy XAU_GOLD"
PERM="--override fixed_sl_tp_override_enabled=true --override fixed_stop_loss_pips=40 --override fixed_take_profit_pips=80 \
--override block_ranging=false --override block_expansion=false --override adx_trending_threshold=0.0 --override atr_expansion_pct=100.0 \
--override session_filter_enabled=false --override asian_allowed=true --override macd_filter_enabled=false --override dxy_confluence_enabled=false \
--override bos_displacement_atr_mult=0.0 --override require_ob_or_fvg=false --override fib_pullback_min=0.0 --override fib_pullback_max=1.5 \
--override rsi_neutral_min=0.0 --override rsi_neutral_max=100.0 --override min_confidence=0.0 --override signal_cooldown_minutes=0"

run_row(){
  local label="$1"; shift
  echo "=== $label ==="
  $BASE $PERM "$@" 2>&1 | grep -E '📊 Total Signals|📊 Profit Factor|🎯 Win Rate|💵 Expectancy|✅ Winners|❌ Losers' | tail -6
  echo ""
}

run_row "A: permissive + OB/FVG + cooldown 30min (realistic deployment)" \
  --override require_ob_or_fvg=true --override signal_cooldown_minutes=30

run_row "B: permissive + OB/FVG + cooldown 30 + block_ranging + ADX>=25 (coupled regime gate)" \
  --override require_ob_or_fvg=true --override signal_cooldown_minutes=30 \
  --override block_ranging=true --override adx_trending_threshold=25.0

run_row "C: permissive + min_confidence=0.75 (hard-gate binding check)" \
  --override min_confidence=0.75
