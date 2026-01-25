"""
Configuration settings for the System Monitor service.
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql://postgres:postgres@postgres:5432/forex"

    # Telegram notifications
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_enabled: bool = True

    # Email notifications
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    admin_email: Optional[str] = None
    email_enabled: bool = False

    # Monitoring settings
    monitor_interval: int = 30  # seconds between checks
    alert_cooldown: int = 900  # 15 minutes between duplicate alerts
    metrics_retention_days: int = 30

    # Health check timeouts
    health_check_timeout: int = 45  # seconds (increased from 10 - fastapi-dev blocks during IG API calls)

    # Metrics collection
    collect_container_metrics: bool = False  # Disable by default - stats calls are slow
    metrics_timeout: int = 3  # seconds per container

    # Alert thresholds
    cpu_warning_threshold: float = 85.0
    cpu_critical_threshold: float = 95.0
    memory_warning_threshold: float = 85.0
    memory_critical_threshold: float = 95.0
    restart_loop_threshold: int = 3  # restarts in restart_loop_window
    restart_loop_window: int = 300  # 5 minutes
    health_check_failures_threshold: int = 3

    # Container configuration
    critical_containers: List[str] = [
        "postgres",
        "fastapi-prod",
        "fastapi-stream",
        "task-worker",
        "nginx",
        "stock-scheduler"
    ]

    # Containers to exclude from monitoring (run-once, temporary, or irrelevant)
    excluded_containers: List[str] = [
        "certbot",           # Only runs when generating/renewing SSL certs
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Container health check endpoints
HEALTH_ENDPOINTS = {
    "fastapi-prod": "http://fastapi-prod:8000/position-closer/health",
    "fastapi-stream": "http://fastapi-stream:8000/stream/health",
    "tradingview": "http://tradingview:8080/health",
    "vector-db": "http://vector-db:8090/health",
    "economic-calendar": "http://economic-calendar:8091/health",
}

# Custom headers for health checks (e.g., authentication bypass)
HEALTH_HEADERS = {
    "fastapi-prod": {"x-apim-gateway": "verified"},
    "fastapi-stream": {"x-apim-gateway": "verified"},
}

# Containers that use TCP check instead of HTTP
TCP_CHECK_CONTAINERS = {
    "postgres": ("postgres", 5432),
    "nginx": ("nginx", 80),
    "fastapi-dev": ("fastapi-dev", 8000),  # Health endpoint times out, use TCP check instead
}

# Container descriptions for UI
CONTAINER_DESCRIPTIONS = {
    "postgres": "PostgreSQL Database - Core data storage",
    "fastapi-dev": "Development API - Trading endpoints (demo)",
    "fastapi-prod": "Production API - Trading endpoints (live)",
    "fastapi-stream": "Stream API - Real-time market data",
    "nginx": "Reverse Proxy - SSL termination & routing",
    "streamlit": "Web UI - Analytics dashboard",
    "tradingview": "TradingView Service - Script indexing & RAG",
    "vector-db": "Vector Database - Semantic search (ChromaDB)",
    "economic-calendar": "Economic Calendar - Event scraping",
    "task-worker": "Task Worker - Background jobs & forex scanner",
    "stock-scheduler": "Stock Scheduler - Daily stock data updates & weekly instrument sync",
    "db-backup": "Database Backup - Automated backups",
    "pgadmin": "PgAdmin - Database management UI",
    "certbot": "Certbot - SSL certificate management",
}

settings = Settings()
