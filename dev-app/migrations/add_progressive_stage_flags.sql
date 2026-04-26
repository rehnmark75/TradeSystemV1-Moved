-- Add progressive trailing stage flags (Apr 2026)
-- Fixes: moved_to_stage1/stage2 were assigned in Python but never persisted (no columns).
-- Effect: re-execution guards in trailing_class.py now actually prevent stage re-firing
-- across monitor cycles after a session refresh.

ALTER TABLE trade_log
    ADD COLUMN IF NOT EXISTS moved_to_stage1 BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE trade_log
    ADD COLUMN IF NOT EXISTS moved_to_stage2 BOOLEAN NOT NULL DEFAULT FALSE;
