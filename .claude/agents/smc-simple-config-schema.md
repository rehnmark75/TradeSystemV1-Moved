# SMC_SIMPLE Strategy Configuration Complete Picture

**Database:** `strategy_config`
**Last Updated:** May 9, 2026
**Scope:** All SMC_SIMPLE configuration tables, schema, current values (live & demo)

---

## 1. Global Configuration Table: `smc_simple_global_config`

**Schema:** 266 columns (comprehensive parameter set)
**Scope:** Single active row per `config_set` ('live' | 'demo')
**Query:** `SELECT * FROM smc_simple_global_config WHERE is_active=TRUE AND config_set='live'`

### A) Core Strategy Parameters
- `htf_timeframe`: '4h' — Higher timeframe for bias/structure
- `trigger_timeframe`: '15m' — Swing break detection
- `entry_timeframe`: '5m' — Entry confirmation
- `ema_period`: 50 — EMA calculation period
- `swing_lookback_bars`: 20 — Swing detection lookback
- `fib_pullback_min/max`: 0.236-0.700 — Fibonacci pullback zones

### B) Stop Loss / Take Profit Defaults (FIXED)
- `fixed_sl_tp_override_enabled`: true
- `fixed_stop_loss_pips`: 7.0 — GLOBAL default SL
- `fixed_take_profit_pips`: 10.0 — GLOBAL default TP
- `min_rr_ratio`: 1.5 — Minimum Risk:Reward ratio
- `min_tp_pips`: 8 — Minimum TP distance
- `sl_buffer_pips`: 6 — SL offset from structure

### C) Confidence & Filtering
- `min_confidence_threshold`: 0.480
- `max_confidence_threshold`: 0.640 — **CRITICAL: trades above 0.64 blocked!**
- `high_confidence_threshold`: 0.750
- `confidence_weights`: {"rr_ratio": 0.2, "ema_alignment": 0.2, ...}

### D) Technical Indicators
- `atr_period`: 14
- `macd_alignment_filter_enabled`: true
- `macd_alignment_mode`: 'momentum'
- `volume_filter_enabled`: true
- `volume_sma_period`: 20
- `rsi_divergence_enabled`: false

### E) Scalp Mode (Sub-strategy)
- `scalp_mode_enabled`: false — NOT enabled globally (per-pair only)
- `scalp_sl_pips`: 5.0 — Global default
- `scalp_tp_pips`: 5.0
- `scalp_htf_timeframe`: '1h'
- `scalp_trigger_timeframe`: '5m'
- `scalp_entry_timeframe`: '1m'
- `scalp_min_confidence`: 0.30
- `scalp_cooldown_minutes`: 15
- `scalp_rsi_filter_enabled`: true
- `scalp_two_pole_filter_enabled`: true
- `scalp_macd_filter_enabled`: true

### F) Session Filtering
- `session_filter_enabled`: true
- `london_session_start`: '07:00:00' UTC
- `london_session_end`: '16:00:00' UTC
- `ny_session_start`: '12:00:00' UTC
- `ny_session_end`: '21:00:00' UTC
- `allowed_sessions`: {london, new_york, overlap}
- `block_asian_session`: true

### G) Cooldown & Concurrency
- `signal_cooldown_hours`: 3
- `max_concurrent_signals`: 3
- `adaptive_cooldown_config`: Complex JSONB configuration

### H) Enabled Pairs (Global list)
- `enabled_pairs`: {CS.D.EURJPY.MINI.IP} — **ONLY 1 pair** in live!
- Note: Actual trading controlled via `smc_simple_pair_overrides.is_enabled`

### I) Risk Management
- `risk_per_trade_pct`: 1.0
- `sweep_protection_enabled`: true
- `sweep_protection_mode`: 'block'
- `sweep_rsi_threshold_buy`: 78.0
- `sweep_rsi_threshold_sell`: 22.0

### J) Debugging & Logging
- `enable_debug_logging`: true
- `log_rejected_signals`: true
- `log_swing_detection`: false
- `rejection_tracking_enabled`: true

### K) Backtest Specific
- `backtest_spread_pips`: 1.5
- `backtest_slippage_pips`: 0.5

### L) Other Advanced Features
- `htf_bias_enabled`: true
- `continuation_entry_enabled`: false
- `pattern_confirmation_enabled`: false
- `rsi_divergence_enabled`: false
- `regime_sl_tp_enabled`: false
- `rolling_perf_enabled`: false
- `block_htf_start_position`: false

---

## 2. Per-Pair Override Table: `smc_simple_pair_overrides`

**Schema:** 77 columns
**Scope:** One row per pair per `config_set` (config_id=2 for live, 3 for demo)
**Constraint:** UNIQUE(config_id, epic)

### DIRECT COLUMNS (overridable per-pair, NOT in JSONB)

```
id                                   integer PK
config_id                            integer FK → smc_simple_global_config
epic                                 varchar(50) NOT NULL
is_enabled                           boolean     [TRADING ON/OFF for this pair]

SL/TP OVERRIDES (NULL = use global):
├─ fixed_stop_loss_pips              numeric(5,1)
├─ fixed_take_profit_pips            numeric(5,1)
└─ sl_buffer_pips                    integer

CONFIDENCE THRESHOLDS (NULL = use global):
├─ min_confidence                    numeric(4,3)
├─ max_confidence                    numeric(4,3)
└─ macd_filter_enabled               boolean     [Pair-level MACD on/off]

VOLUME FILTERING:
├─ min_volume_ratio                  numeric(4,2)
├─ high_volume_confidence            numeric(4,3)
├─ low_atr_confidence                numeric(4,3)
├─ high_atr_confidence               numeric(4,3)
├─ near_ema_confidence               numeric(4,3)
└─ far_ema_confidence                numeric(4,3)

SCALP MODE (NULL = use global):
├─ scalp_enabled                     boolean     [Enable scalp mode for pair]
├─ scalp_sl_pips                     numeric(4,1)
├─ scalp_tp_pips                     numeric(4,1)
├─ scalp_max_spread_pips             numeric(4,2)
├─ scalp_ema_period                  integer
├─ scalp_swing_lookback_bars         integer
├─ scalp_htf_timeframe               varchar(10)
├─ scalp_trigger_timeframe           varchar(10)
├─ scalp_entry_timeframe             varchar(10)
├─ scalp_min_confidence              numeric(4,3)
├─ scalp_cooldown_minutes            integer
├─ scalp_qualification_mode          varchar(20)
├─ scalp_fib_pullback_min            numeric(4,2)
├─ scalp_reversal_enabled            boolean
├─ scalp_reversal_min_runway_pips    numeric(5,2)
├─ scalp_require_ema_stack_alignment boolean
├─ scalp_block_ranging_market        boolean
├─ scalp_block_low_volatility_trending boolean
├─ scalp_min_adx                     numeric(5,2)
├─ scalp_rsi_block_buy_min           numeric(5,2)
├─ scalp_rsi_block_buy_max           numeric(5,2)
├─ scalp_rsi_block_sell_min          numeric(5,2)
├─ scalp_rsi_block_sell_max          numeric(5,2)
├─ scalp_blocked_hours_utc           varchar(100)
└─ scalp_block_global_high_volatility boolean

SWING & EMA SETTINGS:
├─ swing_lookback_bars               integer
├─ swing_proximity_enabled           boolean
├─ swing_proximity_min_distance_pips integer
├─ swing_proximity_strict_mode       boolean
├─ min_swing_atr_multiplier          numeric(5,3)
├─ ema_period                        integer
├─ ema_slope_validation_enabled      boolean
├─ stop_offset_pips                  numeric(4,1)
├─ max_extension_atr                 numeric(4,2)
└─ max_momentum_staleness_bars       integer

OTHER:
├─ allow_asian_session               boolean
├─ direction_overrides_enabled       boolean
├─ smc_conflict_tolerance            integer
├─ htf_bias_mode                     varchar(20)
├─ htf_bias_min_threshold            numeric(4,3)
├─ fib_pullback_min_bull/bear        numeric(5,4)
├─ fib_pullback_max_bull/bear        numeric(5,4)
├─ momentum_min_depth_bull/bear      numeric(5,4)
├─ min_volume_ratio_bull/bear        numeric(5,4)
├─ min_confidence_bull/bear          numeric(5,4)
└─ blocking_conditions               jsonb

METADATA:
├─ description                       text
├─ created_at                        timestamp
├─ updated_at                        timestamp
├─ updated_by                        varchar(100)
└─ change_reason                     text
```

### JSONB-ONLY OVERRIDES (parameter_overrides column)

These keys live **ONLY** in the JSONB and are NOT direct columns:

**Critical Flags:**
- `monitor_only` — true|false — Trading disabled, signals logged only
- `max_confidence` — Override max_confidence threshold
- `MIN_CONFIDENCE_THRESHOLD` — Override min confidence
- `MAX_CONFIDENCE_THRESHOLD` — Override max confidence

**Signal Quality Filters:**
- `min_swing_significance` — Signal quality threshold
- `swing_significance_filter_mode` — ACTIVE|MONITORING|DISABLED
- `impulse_quality_filter_mode` — ACTIVE|MONITORING|DISABLED
- `impulse_score` — Signal quality threshold

**Fibonacci & Momentum:**
- `FIB_PULLBACK_MIN`, `FIB_PULLBACK_MAX` — Fib override
- `MOMENTUM_MIN_DEPTH` — Momentum threshold override
- `session_fib_max` — {"london": 1.0, "new_york": 1.0, ...}

**Technical Filters:**
- `MIN_BODY_PERCENTAGE` — Candle body filter
- `MIN_BREAKOUT_ATR_RATIO` — ATR confirmation requirement
- `mfi_min_slope` — Money Flow Index filter
- `mfi_filter_mode` — ACTIVE|MONITORING|DISABLED
- `require_body_close_break` — Candle body requirement
- `swing_strength_bars` — Swing validation bars

**Scalp-Specific (JSONB):**
- `scalp_min_adx` — Scalp-mode ADX minimum
- `scalp_entry_rsi_buy_max`, `scalp_entry_rsi_sell_min` — RSI filters
- `scalp_max_confidence`, `scalp_min_confidence` — Confidence caps
- `scalp_ema_buffer_pips` — Entry offset from EMA
- `scalp_session_start_hour`, `scalp_session_end_hour` — Session time
- `scalp_require_macd_alignment` — MACD confirmation
- `scalp_require_trending_regime` — Regime filter
- `scalp_require_rejection_candle` — Entry pattern requirement
- `scalp_rsi_filter_enabled`, `scalp_two_pole_filter_enabled` — Filter overrides
- `scalp_require_ema_stack_alignment` — EMA order requirement
- `scalp_block_ranging_market`, `scalp_block_low_volatility_trending` — Regime blocks
- `scalp_min_mtf_confluence` — Multi-timeframe alignment
- `scalp_disable_swing_proximity`, `scalp_macd_filter_enabled`, `scalp_sr_tolerance_pips`

**Regime-Specific SL/TP:**
- `regime_sl_tp_enabled` — Enable regime-based multipliers
- `low_vol_sl_mult`, `low_vol_tp_mult` — Low volatility regime
- `ranging_sl_mult`, `ranging_tp_mult` — Ranging regime
- `breakout_sl_mult`, `breakout_tp_mult` — Breakout regime
- `trending_sl_mult`, `trending_tp_mult` — Trending regime
- `high_vol_sl_mult`, `high_vol_tp_mult` — High volatility regime

**Advanced Features:**
- `block_volatility_states` — ["high", "extreme"] — Block when state matches
- `continuation_entry_enabled` — Enable continuation mode
- `atr_normalized_sl_tp_enabled` — Use ATR for SL/TP instead of fixed
- `atr_reference_pips` — ATR reference for normalization
- `smc_conflict_filter_enabled` — Block conflicting SMC signals
- `smc_reject_ranging_structure` — Block when ranging detected
- `smc_reject_order_flow_conflict` — Block order flow conflicts

---

## 3. Live Environment (config_id=2) - Current Pair Overrides

| Pair | Enabled | SL | TP | min_conf | max_conf | MACD | scalp_SL/TP | monitor_only |
|------|---------|----|----|----------|----------|------|-------------|--------------|
| AUDJPY | false | 12.0 | 14.0 | 0.480 | 0.640 | f | - | NO |
| AUDUSD | false | 10.0 | 10.0 | 0.550 | 0.640 | f | - | YES |
| EURGBP | false | 7.0 | 10.0 | 0.600 | 0.640 | - | - | NO |
| EURJPY | false | 15.0 | 20.0 | 0.440 | 0.640 | f | - | YES |
| EURUSD | false | 8.0 | 10.0 | 0.400 | 0.590 | f | 12.0/12.0 | YES |
| GBPJPY | false | - | 5.0 | 0.480 | - | f | - | NO |
| GBPUSD | false | 9.0 | 9.0 | 0.440 | 0.550 | t | 9.0/10.0 | YES |
| NZDUSD | false | 5.0 | 10.0 | 0.550 | - | f | 6.4/16.0 | YES |
| USDCAD | false | 9.0 | 10.0 | 0.550 | 0.700 | f | 10.0/11.0 | NO |
| USDCHF | false | 12.0 | 18.0 | 0.650 | - | f | 12.0/18.0 | NO |
| USDJPY | false | 8.0 | 10.0 | 0.300 | 0.640 | f | 12.0/13.0 | NO |

**Summary:**
- All pairs disabled at `is_enabled=false` (LIVE TRADING OFF)
- Many pairs have `monitor_only=true` in JSONB (signals logged but not traded)
- Only EURJPY in `enabled_pairs` (global list) but still `is_enabled=false`
- Per-pair SL/TP overrides typical (e.g., AUDJPY 12/14, EURJPY 15/20)
- Confidence ranges vary: min 0.300-0.650, max 0.550-0.700
- JPY pairs use wider SL (15-20 pips), majors use 8-10 pips

---

## 4. Demo Environment (config_id=3) - Current Pair Overrides

| Pair | Enabled | SL | TP | min_conf | max_conf | MACD | scalp_SL/TP | monitor_only |
|------|---------|----|----|----------|----------|------|-------------|--------------|
| AUDJPY | true | 12.0 | 14.0 | 0.480 | 0.640 | f | - | YES |
| AUDUSD | true | 10.0 | 10.0 | 0.550 | 0.640 | f | - | YES |
| CFEGOLD | true | - | - | - | - | - | - | YES |
| EURGBP | false | 7.0 | 10.0 | 0.600 | 0.640 | - | - | YES |
| EURJPY | true | 15.0 | 20.0 | 0.440 | 0.640 | f | 15.0/20.0 | NO |
| EURUSD | true | 8.0 | 10.0 | 0.400 | 0.590 | f | 12.0/12.0 | NO |
| GBPJPY | false | - | 5.0 | 0.480 | - | f | - | YES |
| GBPUSD | true | 9.0 | 9.0 | 0.440 | 0.550 | t | 9.0/10.0 | NO |
| NZDUSD | true | 5.0 | 10.0 | 0.550 | - | f | 6.4/16.0 | YES |
| USDCAD | true | 9.0 | 10.0 | 0.550 | 0.700 | f | 10.0/11.0 | YES |
| USDCHF | false | 12.0 | 18.0 | 0.650 | - | f | 12.0/18.0 | YES |
| USDJPY | true | 8.0 | 10.0 | 0.300 | 0.640 | f | 12.0/13.0 | YES |

**Summary:**
- 9 pairs enabled: AUDJPY, AUDUSD, CFEGOLD, EURJPY, EURUSD, GBPUSD, NZDUSD, USDCAD, USDJPY
- 3 pairs disabled: EURGBP, GBPJPY, USDCHF
- Many still have `monitor_only=true` (active but not traded)
- CFEGOLD is for XAU_GOLD strategy (mapped separately)
- Strong per-pair customization: EURJPY has 30+ JSONB keys, EURUSD has 20+

---

## 5. Column Categorization: DIRECT vs JSONB

### DIRECT COLUMNS (Nullable, NULL = use global default)
- `fixed_stop_loss_pips` — SL override
- `fixed_take_profit_pips` — TP override
- `sl_buffer_pips` — SL offset
- `min_confidence`, `max_confidence` — Confidence thresholds
- `macd_filter_enabled` — MACD on/off
- `min_volume_ratio` — Volume threshold
- All `scalp_*` columns (20 direct columns)
- All `swing_*` columns (swing proximity, lookback, etc.)
- All `fib_pullback_*_bull/bear` columns
- All `momentum_*_bull/bear` columns
- `ema_period`, `ema_slope_validation_enabled`
- `stop_offset_pips`, `max_extension_atr`
- `max_momentum_staleness_bars`
- `allow_asian_session`, `direction_overrides_enabled`
- `smc_conflict_tolerance`, `htf_bias_mode`, `htf_bias_min_threshold`

### JSONB-ONLY (parameter_overrides)

Everything else that isn't a direct column lives in the JSONB. Key examples:
- `monitor_only` — THE CRITICAL FLAG (kills trading, logs signals)
- Regime-specific multipliers (low_vol_sl_mult, ranging_tp_mult, etc.)
- Complex nested objects
- Pair-specific filter modes
- Advanced flags
- All custom thresholds not represented by direct columns

**RULE: When updating pair config:**
- ✓ Use direct columns for: SL/TP, confidence, MACD, volume, scalp basics
- ✓ Use JSONB for: everything else (monitor_only, regime multipliers, custom flags)
- ✗ NEVER modify direct columns if they should be JSONB-only
- ✗ NEVER mix the same setting across both (choose one location)

---

## 6. Other Relevant Tables

### A) trailing_pair_config
- **Scope:** Per-strategy, per-config_set (live|demo), per-epic, per-mode (scalp|regular)
- **Unique constraint:** (strategy, config_set, epic, is_scalp)
- **Key columns:**
  - `break_even_trigger_points`, `early_breakeven_trigger_points`
  - `stage1_trigger_points`, `stage1_lock_points`
  - `stage2_trigger_points`, `stage2_lock_points`
  - `stage3_trigger_points`, `stage3_atr_multiplier`, `stage3_min_distance`
  - `min_trail_distance`
  - `partial_close_trigger_points`, `partial_close_size`
- **Strategies:** DEFAULT, SMC_SIMPLE, XAU_GOLD, RANGE_FADE, MEAN_REVERSION, RANGE_STRUCTURE

### B) smc_simple_config_audit
- Tracks all changes to global_config and pair_overrides
- Useful for: auditing config changes, debugging which setting changed when

### C) smc_simple_parameter_metadata
- UI metadata for displaying parameters in Streamlit/web UI
- Used by: frontend UI rendering

### D) loss_prevention_rules & loss_prevention_decisions
- LPF filter rules (32 total rules as of May 2026)
- Applied BEFORE trading (blocks signals that fail filter)
- Configured per strategy
- Can be per-pair via loss_prevention_pair_config

### E) scanner_global_config
- Scanner-level settings (scan interval, data batch size, etc.)
- Used by: trade_scan.py, bt.py (backtest)
- Critical fields: scan_interval, data_batch_size, use_1m_base_synthesis

---

## 7. Critical Notes & Gotchas

### 1. PRECEDENCE: Per-pair direct column > Global default (NULL = use global)
- If `fixed_stop_loss_pips` is NULL in pair_overrides → uses global 7.0 pips
- If `fixed_stop_loss_pips` is 12.0 → uses 12.0 pips
- SAME for confidence, scalp settings, everything with per-pair direct columns

### 2. JSONB OVERRIDES PRECEDENCE: Parameter_overrides > Direct columns
- If "max_confidence" is in parameter_overrides JSON → uses JSONB value
- If "max_confidence" direct column is set → direct column ignored
- **Always check both places when troubleshooting!**

### 3. LIVE CONFIG IS PRODUCTION - BE CAREFUL
- config_id=2 is live trading (real money)
- All live pairs are currently disabled (is_enabled=false)
- Many have monitor_only=true (safe state)
- Changes here affect REAL trading when enabled

### 4. DEMO IS SAFE FOR TESTING
- config_id=3 is demo/paper trading
- 9 pairs enabled for testing
- Use demo for testing parameter changes before going live

### 5. MONITOR_ONLY FLAG
- `monitor_only=true` in parameter_overrides (JSONB) → signals generated but NOT traded
- Signals still logged to alert_history, visible in UI
- Easiest way to pause trading without disabling: set monitor_only=true

**SQL to pause trading:**
```sql
UPDATE smc_simple_pair_overrides 
SET parameter_overrides = jsonb_set(parameter_overrides, '{monitor_only}', '"true"')
WHERE config_id=2 AND epic='CS.D.EURUSD.CEEM.IP'
```

### 6. PER-PAIR SL/TP IS TYPICAL
- Most pairs have custom SL/TP (not using global 7.0/10.0)
- JPY pairs: 12-20 pips SL (wider, higher volatility)
- Majors: 8-10 pips SL (tighter, lower volatility)
- **ALWAYS set per-pair SL/TP during pair enablement**

### 7. MAX_CONFIDENCE CAP IS CRITICAL
- `max_confidence` (direct column or JSONB override) = HARD CEILING
- Trades with confidence > max_confidence are BLOCKED by strategy code
- Global max: 0.640 (64%)
- Live EURJPY override: 0.590 (59%)
- Live EURUSD override: 0.590 (59%)
- **Sweet spot empirically: 0.55-0.59 (76.7% WR as of May 2026)**

### 8. SCALP MODE SYSTEM
- `scalp_enabled` (direct column) = turn scalp on/off per pair
- Scalp has own timeframes: HTF 1h, Trigger 5m, Entry 1m
- Scalp has own confidence floor (default 0.300)
- Scalp has own SL/TP
- Scalp has own cooldown (default 15 min)
- Scalp has own filters: RSI block ranges, two-pole filter, MACD filter

### 9. COMPLEXITY WARNING: EXCESSIVE CUSTOMIZATION
- Some pairs (EURUSD, EURJPY) have 20-30+ JSONB keys
- Hard to audit, harder to revert
- Consider periodically consolidating JSONB back to defaults
- Always document WHY a custom setting exists (change_reason column)

### 10. CONFIG CHANGES DO NOT AFFECT RUNNING BACKTESTS
- Backtests read smc_simple_global_config at START time
- Changing config mid-backtest doesn't affect running job
- Always restart backtest to pick up new config

### 11. FIELD NAMING INCONSISTENCY IN JSONB
- Some keys are UPPERCASE: FIB_PULLBACK_MIN, MIN_BODY_PERCENTAGE, etc.
- Some keys are snake_case: min_swing_significance, mfi_filter_mode
- Reason: Historical migration, different sources
- **Always match exact casing when setting JSONB keys!**

---

## 8. Sample Queries for Common Tasks

### Check current live configuration
```sql
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT epic, is_enabled, fixed_stop_loss_pips, fixed_take_profit_pips,
       min_confidence, max_confidence, parameter_overrides->>'monitor_only'
FROM smc_simple_pair_overrides WHERE config_id=2 ORDER BY epic;"
```

### Enable trading on a pair (live)
```sql
UPDATE smc_simple_pair_overrides
SET parameter_overrides = parameter_overrides - 'monitor_only'
WHERE config_id=2 AND epic='CS.D.EURUSD.CEEM.IP'
```

### Pause trading without disabling (via monitor_only)
```sql
UPDATE smc_simple_pair_overrides
SET parameter_overrides = jsonb_set(parameter_overrides, '{monitor_only}', '"true"')
WHERE config_id=2 AND epic='CS.D.EURUSD.CEEM.IP'
```

### Set per-pair SL/TP (live)
```sql
UPDATE smc_simple_pair_overrides
SET fixed_stop_loss_pips=9, fixed_take_profit_pips=15
WHERE config_id=2 AND epic='CS.D.EURUSD.CEEM.IP'
```

### Check global defaults
```sql
SELECT fixed_stop_loss_pips, fixed_take_profit_pips, min_confidence_threshold,
       max_confidence_threshold, enabled_pairs
FROM smc_simple_global_config WHERE config_set='live' AND is_active=TRUE
```

### View all pair overrides with their status
```sql
SELECT config_id, epic, is_enabled,
       fixed_stop_loss_pips, fixed_take_profit_pips,
       min_confidence, max_confidence,
       parameter_overrides->>'monitor_only' as monitor_only,
       jsonb_object_keys(parameter_overrides) as override_keys
FROM smc_simple_pair_overrides
WHERE config_id IN (2, 3)
ORDER BY config_id, epic
```

### Check what changed recently
```sql
SELECT id, pair_override_id, change_description, changed_by, changed_at
FROM smc_simple_config_audit
ORDER BY changed_at DESC LIMIT 20
```

---

## Quick Reference: Key Thresholds

| Setting | Global | Live EURJPY | Live EURUSD | Demo (typical) |
|---------|--------|-------------|-------------|---|
| max_confidence | 0.640 | 0.640 | 0.590 | 0.640 |
| min_confidence | 0.480 | 0.440 | 0.400 | 0.480 |
| fixed_stop_loss_pips | 7.0 | 15.0 | 8.0 | varies |
| fixed_take_profit_pips | 10.0 | 20.0 | 10.0 | varies |
| scalp_min_confidence | 0.300 | 0.300 | 0.450 | 0.300 |
| scalp_tp_pips | 5.0 | 20.0 | 12.0 | varies |
| scalp_sl_pips | 5.0 | 18.0 | 12.0 | varies |

---

**File:** `.claude/agents/smc-simple-config-schema.md`
**For Use By:** db-expert, strategy-lifecycle-manager, trading-strategy-analyst
