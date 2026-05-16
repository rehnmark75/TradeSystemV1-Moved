-- Daily PnL Gate: config + audit tables
-- Run against: strategy_config database

CREATE TABLE IF NOT EXISTS daily_pnl_gate_config (
    environment        VARCHAR(10) PRIMARY KEY,
    is_enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    profit_limit_sek   NUMERIC(10, 2) NOT NULL DEFAULT 200.00,
    loss_limit_sek     NUMERIC(10, 2) NOT NULL DEFAULT -300.00,
    updated_at         TIMESTAMP NOT NULL DEFAULT NOW()
);

INSERT INTO daily_pnl_gate_config (environment) VALUES ('demo'), ('live')
ON CONFLICT (environment) DO NOTHING;

CREATE TABLE IF NOT EXISTS daily_pnl_gate_blocks (
    id               SERIAL PRIMARY KEY,
    blocked_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    environment      VARCHAR(10) NOT NULL,
    limit_hit        VARCHAR(10) NOT NULL,   -- 'profit' | 'loss'
    daily_pnl_sek    NUMERIC(10, 2) NOT NULL,
    profit_limit_sek NUMERIC(10, 2) NOT NULL,
    loss_limit_sek   NUMERIC(10, 2) NOT NULL,
    epic             VARCHAR(40),
    direction        VARCHAR(4),
    alert_id         INTEGER,
    trigger_source   VARCHAR(40)
);

CREATE INDEX IF NOT EXISTS idx_daily_pnl_gate_blocks_env_date
    ON daily_pnl_gate_blocks (environment, DATE(blocked_at));
