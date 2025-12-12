# Docker Environment - All Commands Run in Containers

This skill ensures you understand that **everything in TradeSystemV1 runs inside Docker containers**. Never run scripts, database commands, or Python code directly on the host machine.

---

## Critical Rule

**ALL commands must be executed inside the appropriate Docker container.**

- Python scripts → `task-worker` container
- Database queries → `postgres` container (or via `task-worker`)
- Streamlit app → `streamlit` container
- Redis commands → `redis` container

---

## Container Names

| Container | Purpose | Image |
|-----------|---------|-------|
| `task-worker` | Forex scanner, strategies, backtests, all Python scripts | Custom Python image |
| `postgres` | PostgreSQL database | postgres:15 |
| `streamlit` | Web UI dashboard | Custom Streamlit image |
| `redis` | Caching, task queues | redis:alpine |

---

## Running Commands

### Python Scripts
```bash
# WRONG - Do not run on host
python worker/app/forex_scanner/some_script.py

# CORRECT - Run inside task-worker container
docker exec -it task-worker python /app/forex_scanner/some_script.py
```

### Database Queries
```bash
# WRONG - Do not run on host
psql -U postgres -d trading

# CORRECT - Run inside postgres container
docker exec -it postgres psql -U postgres -d trading

# OR via task-worker (if using Python/SQLAlchemy)
docker exec -it task-worker python -c "from core.database import get_engine; ..."
```

### Interactive Python Shell
```bash
# CORRECT
docker exec -it task-worker python
docker exec -it task-worker ipython
```

### Bash Shell Access
```bash
# Get shell inside container
docker exec -it task-worker bash
docker exec -it postgres bash
docker exec -it streamlit bash
```

---

## Path Mapping

The host filesystem is mounted into containers:

| Host Path | Container Path |
|-----------|----------------|
| `worker/app/` | `/app/` |
| `worker/app/forex_scanner/` | `/app/forex_scanner/` |
| `streamlit/` | `/app/` (in streamlit container) |

### Example Path Translation
```bash
# Host path
/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/strategies/my_strategy.py

# Container path (task-worker)
/app/forex_scanner/strategies/my_strategy.py
```

---

## Common Operations

### Run a Backtest
```bash
docker exec -it task-worker python /app/forex_scanner/backtests/run_backtest.py
```

### Run Forex Scanner
```bash
docker exec -it task-worker python /app/forex_scanner/main.py --pairs EURUSD,GBPUSD
```

### Query Database
```bash
# Direct SQL
docker exec -it postgres psql -U postgres -d trading -c "SELECT * FROM ig_candles LIMIT 5;"

# From file
docker exec -i postgres psql -U postgres -d trading < query.sql
```

### Check Logs
```bash
docker logs task-worker --tail 100 -f
docker logs postgres --tail 50
docker logs streamlit --tail 50
```

### Restart Services
```bash
docker restart task-worker
docker restart streamlit
docker-compose restart  # All services
```

---

## Docker Compose Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose build task-worker
docker-compose up -d task-worker

# View running containers
docker-compose ps
docker ps

# View logs
docker-compose logs -f task-worker
```

---

## Database Connection from Python

Inside the `task-worker` container, database connections use internal Docker networking:

```python
# Connection string uses container name as host
DATABASE_URL = "postgresql://postgres:password@postgres:5432/trading"

# NOT localhost - containers communicate via Docker network
# WRONG: postgresql://postgres:password@localhost:5432/trading
```

---

## Environment Variables

Environment variables are set in:
- `docker-compose.yml` - Service-level env vars
- `.env` file - Shared secrets (API keys, passwords)

Access inside container:
```bash
docker exec -it task-worker env | grep DATABASE
docker exec -it task-worker printenv
```

---

## File Operations

### Copy Files Into Container
```bash
docker cp local_file.py task-worker:/app/forex_scanner/
```

### Copy Files Out of Container
```bash
docker cp task-worker:/app/forex_scanner/logs/backtest.log ./
```

### View Files Inside Container
```bash
docker exec -it task-worker ls -la /app/forex_scanner/
docker exec -it task-worker cat /app/forex_scanner/config.py
```

---

## Debugging

### Check Container Status
```bash
docker ps -a                    # All containers
docker inspect task-worker      # Detailed info
docker stats                    # Resource usage
```

### Container Won't Start
```bash
docker logs task-worker         # Check startup errors
docker-compose up task-worker   # Run in foreground to see errors
```

### Permission Issues
```bash
# Files created in container may have different ownership
docker exec -it task-worker chown -R 1000:1000 /app/forex_scanner/logs/
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Run Python script | `docker exec -it task-worker python /app/script.py` |
| Interactive Python | `docker exec -it task-worker python` |
| SQL query | `docker exec -it postgres psql -U postgres -d trading -c "..."` |
| Container shell | `docker exec -it task-worker bash` |
| View logs | `docker logs task-worker -f` |
| Restart service | `docker restart task-worker` |
| Rebuild | `docker-compose build task-worker && docker-compose up -d` |

---

## Remember

1. **Never run `python` directly on the host** for forex_scanner code
2. **Never run `psql` directly on the host** - use the postgres container
3. **Paths inside containers start with `/app/`** not the full host path
4. **Database host is `postgres`** (container name), not `localhost`
5. **Changes to mounted files are immediately visible** in containers (no rebuild needed for code changes)
