# Container Monitoring System - Implementation Plan

## Overview

Build a centralized monitoring container (`system-monitor`) that:
1. Monitors all 12 containers in the TradeSystemV1 system
2. Sends admin notifications (Telegram/Email) on issues
3. Provides a Streamlit dashboard for real-time status visualization

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TradeSystemV1 Monitoring System                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐      ┌──────────────────────────────────────────┐    │
│  │  system-monitor  │      │              Streamlit UI                 │    │
│  │    Container     │◄────►│  📊 Infrastructure Status Page           │    │
│  │                  │      │  (New page: infrastructure_status.py)     │    │
│  │  - Docker API    │      └──────────────────────────────────────────┘    │
│  │  - Health Checks │                                                       │
│  │  - Metrics DB    │      ┌──────────────────────────────────────────┐    │
│  │  - Alert Engine  │─────►│         Notification Channels            │    │
│  │                  │      │  📱 Telegram  │  📧 Email  │  🔔 Webhook │    │
│  └────────┬─────────┘      └──────────────────────────────────────────┘    │
│           │                                                                  │
│           ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    Monitored Containers (12)                        │    │
│  ├──────────────┬──────────────┬──────────────┬──────────────────────┤    │
│  │ fastapi-dev  │ fastapi-prod │ fastapi-stream│ task-worker         │    │
│  │ postgres     │ pgadmin      │ nginx         │ certbot             │    │
│  │ streamlit    │ tradingview  │ vector-db     │ economic-calendar   │    │
│  │ db-backup    │              │               │                      │    │
│  └──────────────┴──────────────┴──────────────┴──────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: System Monitor Container

### 1.1 Container Structure

```
system-monitor/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── health_status.py       # Health status models
│   │   └── container_metrics.py   # Metrics models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── docker_monitor.py      # Docker API integration
│   │   ├── health_checker.py      # Service health checks
│   │   ├── metrics_collector.py   # Metrics collection
│   │   └── alert_manager.py       # Alert logic & thresholds
│   ├── notifications/
│   │   ├── __init__.py
│   │   ├── telegram_notifier.py   # Telegram bot integration
│   │   ├── email_notifier.py      # SMTP email sender
│   │   └── webhook_notifier.py    # Generic webhook support
│   └── api/
│       ├── __init__.py
│       └── routes.py              # REST API endpoints
└── tests/
    └── test_monitor.py
```

### 1.2 Core Functionality

**Docker Monitor Service** (`docker_monitor.py`):
```python
# Key capabilities:
- Connect to Docker daemon via /var/run/docker.sock
- List all containers with status (running, stopped, restarting)
- Get container stats (CPU, memory, network I/O)
- Monitor container health check status
- Detect restart loops (>3 restarts in 5 minutes)
- Track uptime and last restart time
```

**Health Checker Service** (`health_checker.py`):
```python
# Per-service health checks:
HEALTH_ENDPOINTS = {
    'fastapi-dev': 'http://fastapi-dev:8000/health',
    'fastapi-prod': 'http://fastapi-prod:8000/health',
    'fastapi-stream': 'http://fastapi-stream:8000/health',
    'tradingview': 'http://tradingview:8080/health',
    'vector-db': 'http://vector-db:8090/health',
    'economic-calendar': 'http://economic-calendar:8091/health',
    'postgres': 'postgresql://postgres:postgres@postgres:5432/forex',
    'streamlit': 'http://streamlit:8501/_stcore/health',
}

# Health check types:
- HTTP GET (FastAPI services)
- TCP connection (PostgreSQL)
- Docker health check status
- Custom logic per service
```

**Alert Manager** (`alert_manager.py`):
```python
# Alert conditions:
ALERT_CONDITIONS = {
    'container_down': {
        'severity': 'critical',
        'threshold': '1 minute',
        'channels': ['telegram', 'email']
    },
    'container_restarting': {
        'severity': 'warning',
        'threshold': '3 restarts in 5 minutes',
        'channels': ['telegram']
    },
    'health_check_failed': {
        'severity': 'warning',
        'threshold': '3 consecutive failures',
        'channels': ['telegram']
    },
    'high_memory': {
        'severity': 'warning',
        'threshold': '90%',
        'channels': ['telegram']
    },
    'high_cpu': {
        'severity': 'warning',
        'threshold': '95% for 2 minutes',
        'channels': ['telegram']
    },
    'disk_space_low': {
        'severity': 'critical',
        'threshold': '< 10%',
        'channels': ['telegram', 'email']
    }
}

# Alert deduplication:
- Cooldown period per alert type (default: 15 minutes)
- Escalation after repeated alerts
- Recovery notifications when issues resolve
```

### 1.3 Docker Compose Addition

```yaml
# Add to docker-compose.yml:
system-monitor:
  build:
    context: ./system-monitor
  container_name: system-monitor
  restart: unless-stopped
  ports:
    - "8095:8095"
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock:ro
    - ./logs/system-monitor:/app/logs
  environment:
    - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/forex
    - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
    - SMTP_HOST=${SMTP_HOST:-}
    - SMTP_PORT=${SMTP_PORT:-587}
    - SMTP_USER=${SMTP_USER:-}
    - SMTP_PASSWORD=${SMTP_PASSWORD:-}
    - ADMIN_EMAIL=${ADMIN_EMAIL:-}
    - MONITOR_INTERVAL=30  # seconds
    - ALERT_COOLDOWN=900   # 15 minutes
  networks:
    - lab-net
  depends_on:
    - postgres
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8095/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 20s
```

### 1.4 API Endpoints

```
GET  /health                    # Monitor service health
GET  /api/v1/status             # Overall system status summary
GET  /api/v1/containers         # All containers with status
GET  /api/v1/containers/{name}  # Single container details
GET  /api/v1/metrics            # Current metrics snapshot
GET  /api/v1/metrics/history    # Historical metrics (last 24h)
GET  /api/v1/alerts             # Recent alerts
GET  /api/v1/alerts/active      # Currently active (unresolved) alerts
POST /api/v1/alerts/{id}/acknowledge  # Acknowledge an alert
POST /api/v1/test-notification  # Test notification channels
```

---

## Phase 2: Notification System

### 2.1 Telegram Integration

**Setup Requirements:**
1. Create Telegram bot via @BotFather
2. Get bot token and chat ID
3. Add to `.env` file

**Implementation** (`telegram_notifier.py`):
```python
class TelegramNotifier:
    """
    Sends formatted alerts to Telegram.

    Message format:
    🚨 CRITICAL: Container Down
    ━━━━━━━━━━━━━━━━━━━━━
    📦 Container: fastapi-prod
    ⏰ Time: 2025-12-06 14:30:22
    📊 Status: Stopped (Exit Code: 1)
    🔄 Last Restart: 5 min ago

    🔧 Suggested Action:
    Check logs: docker logs fastapi-prod
    """

    # Features:
    - Rich formatting with emojis
    - Severity-based icons (🚨 critical, ⚠️ warning, ✅ resolved)
    - Inline buttons for quick actions
    - Rate limiting to prevent spam
```

### 2.2 Email Integration

**Implementation** (`email_notifier.py`):
```python
class EmailNotifier:
    """
    Sends HTML-formatted alert emails.

    Features:
    - HTML template with system status table
    - Batch alerts (collect for 5 min, send summary)
    - Priority headers for critical alerts
    - Attachment support for logs
    """
```

### 2.3 Alert Message Templates

```python
TEMPLATES = {
    'container_down': {
        'title': '🚨 Container Down: {container_name}',
        'body': '''
Container {container_name} has stopped.
Status: {status}
Exit Code: {exit_code}
Last Log: {last_log_line}

Suggested Actions:
1. Check logs: docker logs {container_name}
2. Restart: docker restart {container_name}
3. Check dependencies
'''
    },
    'container_recovered': {
        'title': '✅ Container Recovered: {container_name}',
        'body': '''
Container {container_name} is now running.
Downtime: {downtime}
Status: Healthy
'''
    },
    # ... more templates
}
```

---

## Phase 3: Streamlit Dashboard

### 3.1 New Page: Infrastructure Status

**File:** `streamlit/pages/infrastructure_status.py`

**Features:**
1. **System Overview Cards**
   - Total containers: running/stopped/unhealthy
   - Overall system health score
   - Last check timestamp
   - Active alerts count

2. **Container Grid**
   - Visual cards for each container
   - Color-coded status (green/yellow/red)
   - Quick stats (uptime, CPU, memory)
   - Click to drill down

3. **Container Detail View**
   - Full container info
   - Real-time logs (last 100 lines)
   - Metrics charts (CPU/memory over time)
   - Health check history
   - Recent restarts
   - Quick actions (restart, view logs)

4. **Alert Timeline**
   - Chronological list of alerts
   - Filter by severity/container
   - Acknowledge button
   - Link to affected container

5. **Historical Metrics**
   - 24h/7d/30d view
   - Container uptime chart
   - Resource usage trends
   - Alert frequency analysis

### 3.2 Page Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🖥️ Infrastructure Status                              Last updated: 14:30  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ 🟢 12/13 │ │ 🟡 1     │ │ 🔴 0     │ │ ⚠️ 2     │ │ Health: 92%     │  │
│  │ Running  │ │ Warning  │ │ Critical │ │ Alerts   │ │ ████████░░       │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Container Status Grid                         │    │
│  ├──────────────┬──────────────┬──────────────┬──────────────────────┬─┤    │
│  │ 🟢 postgres  │ 🟢 fastapi-  │ 🟢 fastapi-  │ 🟡 fastapi-stream    │ │    │
│  │ Up 7d 4h    │ dev          │ prod         │ High CPU (87%)       │ │    │
│  │ CPU: 2%     │ Up 7d 4h     │ Up 7d 4h     │ Up 7d 4h             │ │    │
│  │ Mem: 340MB  │ CPU: 5%      │ CPU: 12%     │ Mem: 890MB           │ │    │
│  ├──────────────┼──────────────┼──────────────┼──────────────────────┤ │    │
│  │ 🟢 streamlit│ 🟢 tradingview│ 🟢 vector-db │ 🟢 economic-calendar│ │    │
│  │ Up 7d 4h    │ Up 7d 4h     │ Up 7d 4h     │ Up 7d 4h             │ │    │
│  │ CPU: 3%     │ CPU: 1%      │ CPU: 4%      │ CPU: 2%              │ │    │
│  │ Mem: 450MB  │ Mem: 280MB   │ Mem: 520MB   │ Mem: 180MB           │ │    │
│  └──────────────┴──────────────┴──────────────┴──────────────────────┴─┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  📋 Recent Alerts                                        [View All] │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ ⚠️ 14:25 | fastapi-stream | High CPU usage (87%)        [Ack]     │    │
│  │ ⚠️ 13:10 | task-worker    | Health check failed (1x)    [Ack]     │    │
│  │ ✅ 12:45 | vector-db      | Recovered after restart               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

[Click container card to expand detail view]

┌─────────────────────────────────────────────────────────────────────────────┐
│  📦 Container Details: fastapi-stream                              [Close]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Status: 🟡 Running (Warning)     Health: ⚠️ Degraded                      │
│  Uptime: 7 days, 4 hours          Restarts: 2 (last 7 days)                │
│  Image: tradesystemv1-fastapi     Port: 8003:8000                          │
│                                                                              │
│  ┌────────────────────────────────┐  ┌────────────────────────────────┐    │
│  │  CPU Usage (24h)               │  │  Memory Usage (24h)            │    │
│  │  ┌──────────────────────────┐  │  │  ┌──────────────────────────┐  │    │
│  │  │    ╭─╮    ╭───╮          │  │  │  │────────────────────────  │  │    │
│  │  │───╯  ╰───╯    ╰──────   │  │  │  │                          │  │    │
│  │  └──────────────────────────┘  │  │  └──────────────────────────┘  │    │
│  │  Avg: 45%  Max: 87%  Now: 87% │  │  Avg: 780MB Max: 920MB         │    │
│  └────────────────────────────────┘  └────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  📜 Recent Logs                                         [Full Logs] │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ 14:30:01 INFO  Processing stream data for EURUSD                    │    │
│  │ 14:30:02 INFO  Received 1,240 ticks                                 │    │
│  │ 14:29:58 WARN  High processing latency: 450ms                       │    │
│  │ 14:29:55 INFO  Connected to IG streaming API                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  [🔄 Restart Container]  [📋 View Full Logs]  [⚙️ Container Config]        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Implementation Components

```python
# streamlit/pages/infrastructure_status.py

# Key functions:
def fetch_system_status() -> Dict:
    """Fetch overall system status from system-monitor API"""

def render_status_cards(status: Dict):
    """Render top-level status metric cards"""

def render_container_grid(containers: List[Dict]):
    """Render clickable container status grid"""

def render_container_detail(container_name: str):
    """Render expanded container detail view"""

def render_alert_timeline(alerts: List[Dict]):
    """Render recent alerts with acknowledge buttons"""

def render_metrics_charts(container_name: str, timerange: str):
    """Render CPU/memory charts for container"""

# New service file: streamlit/services/infrastructure_service.py
class InfrastructureService:
    """Client for system-monitor API"""

    def __init__(self, base_url: str = "http://system-monitor:8095"):
        self.base_url = base_url

    def get_status(self) -> Dict
    def get_containers(self) -> List[Dict]
    def get_container_detail(name: str) -> Dict
    def get_metrics_history(name: str, hours: int) -> List[Dict]
    def get_alerts(active_only: bool = False) -> List[Dict]
    def acknowledge_alert(alert_id: str) -> bool
    def get_container_logs(name: str, lines: int = 100) -> str
```

---

## Phase 4: Database Schema

### 4.1 Metrics Storage

```sql
-- Add to forex database or create monitoring schema

CREATE TABLE IF NOT EXISTS container_metrics (
    id SERIAL PRIMARY KEY,
    container_name VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(20) NOT NULL,  -- running, stopped, restarting, paused
    health_status VARCHAR(20),    -- healthy, unhealthy, starting, none
    cpu_percent DECIMAL(5,2),
    memory_bytes BIGINT,
    memory_percent DECIMAL(5,2),
    network_rx_bytes BIGINT,
    network_tx_bytes BIGINT,
    restart_count INTEGER,
    uptime_seconds INTEGER
);

CREATE INDEX idx_container_metrics_name_time
ON container_metrics(container_name, timestamp DESC);

-- Retention: Keep 30 days of metrics
-- Auto-cleanup via scheduled task

CREATE TABLE IF NOT EXISTS system_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,  -- critical, warning, info
    container_name VARCHAR(100),
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    resolved_at TIMESTAMPTZ,
    notification_sent BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_alerts_active
ON system_alerts(created_at DESC)
WHERE resolved_at IS NULL;
```

---

## Phase 5: Implementation Steps

### Step 1: Create System Monitor Container (Day 1-2)
1. Create `system-monitor/` directory structure
2. Implement Docker monitoring service
3. Implement health checker for all services
4. Create FastAPI REST API
5. Add to docker-compose.yml
6. Test basic functionality

### Step 2: Implement Notifications (Day 2-3)
1. Create Telegram bot and get credentials
2. Implement Telegram notifier
3. Implement email notifier (optional)
4. Create alert manager with thresholds
5. Test notification delivery
6. Implement alert deduplication

### Step 3: Create Streamlit Dashboard (Day 3-4)
1. Create `infrastructure_status.py` page
2. Implement infrastructure service client
3. Build status overview cards
4. Build container grid view
5. Build container detail view
6. Build alert timeline
7. Add metrics charts

### Step 4: Database & Metrics (Day 4-5)
1. Create database tables
2. Implement metrics collection loop
3. Add historical metrics API
4. Implement metrics cleanup job
5. Add uptime tracking

### Step 5: Testing & Polish (Day 5)
1. Test all notification channels
2. Test alert conditions
3. Add error handling
4. Documentation
5. Deploy to production

---

## Configuration

### Environment Variables (add to .env)

```bash
# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Email Configuration (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ADMIN_EMAIL=admin@yourdomain.com

# Monitor Configuration
MONITOR_INTERVAL=30
ALERT_COOLDOWN=900
```

---

## Container Monitoring Matrix

| Container | Health Check | Expected Resources | Critical? |
|-----------|-------------|-------------------|-----------|
| postgres | TCP:5432 + query | CPU: 5%, Mem: 500MB | ✅ Yes |
| fastapi-dev | HTTP /health | CPU: 10%, Mem: 300MB | No |
| fastapi-prod | HTTP /health | CPU: 15%, Mem: 400MB | ✅ Yes |
| fastapi-stream | HTTP /health | CPU: 20%, Mem: 600MB | ✅ Yes |
| streamlit | HTTP /_stcore/health | CPU: 5%, Mem: 500MB | No |
| tradingview | HTTP /health | CPU: 5%, Mem: 300MB | No |
| vector-db | HTTP /health | CPU: 5%, Mem: 600MB | No |
| economic-calendar | HTTP /health | CPU: 3%, Mem: 200MB | No |
| task-worker | Docker health | CPU: 30%, Mem: 800MB | ✅ Yes |
| db-backup | Custom script | CPU: 2%, Mem: 100MB | No |
| nginx | TCP:80 | CPU: 2%, Mem: 50MB | ✅ Yes |
| pgadmin | HTTP:4445 | CPU: 2%, Mem: 200MB | No |

---

## Summary

**Total Effort:** ~5 development days

**Components:**
1. **system-monitor container** - Central monitoring service
2. **Telegram bot** - Real-time admin notifications
3. **Streamlit page** - Visual dashboard with drill-down
4. **Database tables** - Metrics history storage

**Key Benefits:**
- Proactive issue detection (before users notice)
- Historical metrics for capacity planning
- Quick drill-down for debugging
- Reduced MTTR (Mean Time To Recovery)
- Audit trail of system health

**Risk Mitigation:**
- Monitor container itself has health check
- Graceful degradation if DB unavailable
- Rate limiting on notifications
- Alert deduplication prevents spam
