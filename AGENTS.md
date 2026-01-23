# AGENTS.md

Project quick-reference for coding agents. Derived from `CLAUDE.md`.

## Critical: Docker-only execution
- Run all Python/SQL inside containers; do not execute on host.
- Use `docker compose` (v2, space-separated), never `docker-compose` (v1).
- Safer restarts: `docker restart <container>` or `docker compose up -d --no-deps <container>`.

## Containers and ownership
- `fastapi-dev`: live trade execution + trailing/breakeven; source of truth is `dev-app/config.py`.
- `task-worker`: strategy scanning/backtesting; configs in `worker/app/forex_scanner/`.
- `streamlit`: dashboard; reads trailing config via mount.

## Key entry points
- Live scanner: `worker/app/trade_scan.py` (container: `task-worker`)
- Backtest: `worker/app/forex_scanner/bt.py` (container: `task-worker`)
- Trailing monitor: `dev-app/trailing_class.py` (container: `fastapi-dev`)

## Trailing stops (live)
- Live trailing config: `dev-app/config.py` (`PAIR_TRAILING_CONFIGS`, `SCALP_TRAILING_CONFIGS`).
- Restart to apply: `docker restart fastapi-dev`.
- Backtest trailing config is separate: `worker/app/forex_scanner/config_trailing_stops.py`.

## Scalp trailing system
- Scalp trades use `SCALP_TRAILING_CONFIGS` via `is_scalp_trade` flag.
- VSL is deprecated; IG native stops + progressive trailing are used.

## Backtesting notes
- `--timeframe` controls scan interval, not strategy timeframes.
- Recommended: `--timeframe 5m` to match live scan frequency.

## Strategy configuration (SMC Simple)
- Source of truth is the `strategy_config` database, not config files.
- Global config: `smc_simple_global_config`; overrides: `smc_simple_pair_overrides`.
- When adding scanner config fields, update `direct_fields` in
  `worker/app/forex_scanner/services/scanner_config_service.py`.

## Path mapping
- `worker/app/` on host maps to `/app/` inside `task-worker`.
