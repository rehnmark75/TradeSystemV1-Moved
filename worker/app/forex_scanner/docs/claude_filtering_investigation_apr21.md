# Claude Filtering Collapse — Investigation (Apr 21 2026)

## Symptom
Only 7 approvals in the last 7 days across all enabled pairs (vs 30–40 expected).

## Daily approval rate since Apr 1 (from `alert_history`, excludes monitor-only skips)

| Day | Scored | Approved | Approval % | Avg Score |
|---|---|---|---|---|
| Apr 01 | 3 | 2 | 66.7% | 6.33 |
| Apr 03 | 10 | 5 | 50.0% | 5.00 |
| Apr 06 | 4 | 3 | 75.0% | 6.00 |
| Apr 07 | 19 | 8 | 42.1% | 4.79 |
| Apr 09 | 23 | 12 | 52.2% | 5.61 |
| Apr 13 | 31 | 11 | 35.5% | 4.81 |
| Apr 14 | 15 | 6 | 40.0% | 5.33 |
| **Apr 15** | **13** | **3** | **23.1%** | **4.31** |
| **Apr 16** | **14** | **1** | **7.1%** | **3.64** |
| Apr 17 | 12 | 1 | 8.3% | 3.75 |
| Apr 20 | 13 | 1 | 7.7% | 3.77 |
| Apr 21 | 9 | 0 | 0.0% | 3.44 |

**Inflection: Apr 15 → Apr 16.** Average Claude score fell from ~5.3 to ~3.7. Approval rate collapsed from ~40% to ~8%.

## Threshold is unchanged
- `require_claude_approval = true`
- `min_claude_quality_score = 6` (hard-coded `score >= 6` in `claude_api.py:1208`, `response_parser.py:378/481`)
- `claude_fail_secure = true` (Claude failure → REJECT)
- Threshold has NOT moved — the score distribution has.

## Score distribution (14d)

| Score | Count | Decision |
|---|---|---|
| 7 | 32 (but only 4 recent) | APPROVE ✅ |
| 6 | 10 | APPROVE ✅ |
| 5 | 25 | REJECT ❌ (one-off-threshold) |
| 4 | 36 | REJECT |
| 3 | 55 | REJECT (dominant bucket) |
| 2 | 4 | REJECT |
| 0 | 2 | REJECT |
| 8+ | **0** | (none — ceiling has vanished) |

61% of scored signals land in 2–4; 0 signals score ≥8. The prompt is capping the high end.

## Root cause: prompt hardening cluster Apr 14–16

12 Claude-related commits in 72h. The damaging ones:

| Commit | Date | What it did |
|---|---|---|
| `686f6de` | Apr 14 15:10 | Align Claude chart TFs with scalp strategy (5m/15m) |
| `980d1b5` | Apr 14 15:52 | **Role-based chart TFs + explicit `scalp_mode` flag** — added 5M EMA micro-structure block |
| `7e1ef6e` | Apr 15 06:12 | **Inject authoritative 4H structure into prompt + chart** — forces Claude to treat computed 4H trend as ground truth |
| `6faa594` | Apr 15 11:41 | Soften counter-trend cap (intended to INCREASE approval — marginal) |
| `01d092f` | Apr 15 12:49 | Scalp-aware HTF + proportional S/R buffer |
| `9d35cba` | **Apr 16 21:21** | **Bump Claude API models to 4.x defaults** — new model + stricter prompt = compounding |
| `d92d47b` | Apr 18 | Strategy-aware branch for MEAN_REVERSION |

## Evidence from actual rejection reasons (verbatim, last 5 days)

1. *"The 5M EMA micro-structure is explicitly bullish (EMA 9 > EMA 21 > EMA 50) while this is a BEAR entry — the EMAs on the trigger timeframe are crossed against the [direction]"* → score 3
2. *"Entry at 158.91950 is essentially AT the identified resistance level of 158.92000 (0.0 pips clearance), triggering the hard rejection cap for entries at or within 5 pips of major resistance on a BUY"* → score 3
3. *"The nearest resistance at 114.079 is only 2.9 pips above entry, well within the 7-pip hard cap threshold — this is a direct automatic rejection"* → score 3
4. *"The 4H macro structure is confirmed RANGING ... 5M EMA micro-structure is BULLISH, directly opposing this BEAR signal"* → score 3
5. *"RSI at 31.9 — deeply oversold territory for a SELL"* → score 5

These are **deterministic hard caps** now baked into the prompt. SMC reversal entries fire AGAINST short-term micro-structure by design (that's the edge), so the 5M EMA micro-structure check alone auto-rejects a huge fraction of SMC_SIMPLE signals.

## What the approved signals show

All 6 approvals (last 7d) are **trend-continuation** entries aligned with 4H HTF and 5m EMA stack (USDCAD BEAR, USDJPY BEAR, EURUSD BULL momentum, AUDJPY BULL). Not a single counter-trend or reversal approved — confirming the prompt is now a trend-following gate, not an SMC validator.

## Four hard caps to revisit (in `prompt_builder.py`)

1. **5M EMA micro-structure conflict** → currently phrased as automatic rejection. SMC reversals INTENTIONALLY fire against 5m micro — this is guaranteed to blanket-reject the strategy's edge cases. Change to scoring penalty only (−1 or −2), not auto-reject.
2. **S/R proximity "5–10 pip hard cap"** (lines ~800–823) → even after the Apr 15 proportional-buffer fix, reasons show entries 0–3 pips from S/R still auto-rejecting. Make it "score ≤4" (already in the text — but Claude is treating it as "auto-reject"), or scale to ATR.
3. **4H RANGING as blocker** → the new authoritative 4H structure block is being read as "if 4H is RANGING, reject all entries." Scalp entries on 5m/15m should be allowed during 4H ranging — that's the scalp premise. Add explicit carve-out for `scalp_mode=true`.
4. **RSI overbought/oversold ≤5 rule** → RSI 31–34 flagged as "oversold for SELL" is over-aggressive; many winning SMC shorts fire at RSI 30–40 during a down-leg.

## Model bump contribution

`claude_model = sonnet` (bumped Apr 16 to 4.x default). Sonnet 4.x is more literal about following "HARD CAP / AUTOMATIC REJECTION" instructions than 3.5 was. Combined with a stricter prompt → worst of both. Consider A/B-testing the previous Sonnet SKU against the new default prompt.

## Recommended fixes (ordered)

1. **Soften the 5M EMA micro-structure cap**: change "automatic rejection" / "mandatory rejection" language to "penalty of −2". Keep it as a red flag, not a kill switch.
2. **Scope hard caps by strategy**: SMC_SIMPLE reversal entries should skip the micro-structure cap; trend/continuation entries keep it.
3. **Temporary relief lever**: lower `min_claude_quality_score` from 6 → 5 in `scanner_global_config` to recover the ~11 borderline signals/week the new prompt suppresses. Reversible SQL-only change; zero code risk.
4. **Prompt A/B**: freeze current prompt as `strict_v2`, revert to Apr 12 prompt (before the 5M EMA block) as `v1_baseline`, run both over the next 72h on alternate signals.
5. **Monitor `avg_score` and `approval_pct` daily** — if `avg_score` stays below 4.5 after fixes, the prompt is still over-indexed on hard caps.

## SQL to watch this

```sql
SELECT DATE(alert_timestamp) day,
       COUNT(*) FILTER (WHERE claude_score IS NOT NULL) scored,
       COUNT(*) FILTER (WHERE claude_approved) approved,
       ROUND(100.0*COUNT(*) FILTER (WHERE claude_approved)
             / NULLIF(COUNT(*) FILTER (WHERE claude_score IS NOT NULL),0), 1) pct,
       ROUND(AVG(claude_score) FILTER (WHERE claude_score IS NOT NULL)::numeric, 2) avg_score
FROM alert_history
WHERE alert_timestamp >= NOW() - INTERVAL '14 days'
GROUP BY day ORDER BY day DESC;
```

## References
- Prompt source: `worker/app/forex_scanner/alerts/analysis/prompt_builder.py` (5M EMA block at L974, S/R cap at L800–823)
- Threshold source: `worker/app/forex_scanner/alerts/claude_api.py:1208`
- DB knobs: `scanner_global_config.require_claude_approval`, `min_claude_quality_score`, `claude_model`
- Inflection commits: `686f6de`, `980d1b5`, `7e1ef6e`, `9d35cba` (model bump)
