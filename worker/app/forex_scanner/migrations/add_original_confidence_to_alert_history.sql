-- Migration: add_original_confidence_to_alert_history.sql
-- Date: 2026-06-08
-- Purpose: Persist the PRE-news-filter (signal-quality) confidence separately
--          from the stored confidence_score.
--
-- Background:
--   TradeValidator's news filter (core/trading/trade_validator.py) overwrites
--   signal['confidence_score'] with original × {0.7 high-impact | 0.56 high+critical}
--   AFTER the strategy's own confidence floor (e.g. RANGE_FADE min_reject_confidence)
--   has already been checked on the original value. alert_history then stores the
--   reduced number. As a result, alert_history.confidence_score is the post-news
--   value, NOT the signal-quality confidence the strategy gated on, which silently
--   contaminates any confidence-band analysis for trades fired near high-impact news.
--
--   This column stores the pre-news value. The save path now writes:
--     original_confidence = signal['original_confidence']  (set only when the news
--     filter reduced confidence) else confidence_score (no reduction -> equal).
--
-- Analysis guidance:
--   Bucket on COALESCE(original_confidence, confidence_score) for signal-quality.
--   confidence_score remains "what actually executed" (post-news).

ALTER TABLE alert_history
ADD COLUMN IF NOT EXISTS original_confidence DECIMAL(5,4);

COMMENT ON COLUMN alert_history.original_confidence IS
'Pre-news-filter signal-quality confidence (what the strategy floor gated on). confidence_score is the post-news-reduction value that executed. Rows before 2026-06-08 are backfilled = confidence_score (historical news reductions not recoverable).';

-- Backfill existing rows: confidence_score is the best available estimate of the
-- pre-news value (only the small fraction fired near high-impact news differ, and
-- their true original was never persisted, so it is not recoverable).
UPDATE alert_history
SET original_confidence = confidence_score
WHERE original_confidence IS NULL;
