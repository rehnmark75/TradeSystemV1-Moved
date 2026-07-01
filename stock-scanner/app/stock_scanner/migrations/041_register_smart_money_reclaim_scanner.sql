-- 041_register_smart_money_reclaim_scanner.sql
-- Register smart_money_reclaim in the scanner registry so its signals satisfy the
-- stock_scanner_signals -> stock_signal_scanners FK and can be saved.
-- Monitor-only forward-test (Jul 1 2026): RUNS + logs signals but is excluded
-- from the live tradable pool via MONITOR_ONLY_SCANNERS in
-- trading-ui/app/api/signals/top/route.ts. Idempotent.

INSERT INTO stock_signal_scanners
    (scanner_name, description, min_score_threshold, is_active)
VALUES
    ('smart_money_reclaim',
     'LuxAlgo-inspired money-flow divergence sweep-reclaim (monitor-only)',
     55, TRUE)
ON CONFLICT (scanner_name) DO NOTHING;
