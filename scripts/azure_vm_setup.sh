#!/bin/bash
# =============================================================================
# Azure Backtest VM Setup Script
# One-time initialization for the Azure VM
# Run this after creating the VM: ssh azureuser@<VM_IP> < azure_vm_setup.sh
# =============================================================================

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== Azure Backtest VM Setup ===${NC}"
echo "This script will set up Docker, PostgreSQL, and the backtest environment."
echo ""

# =============================================================================
# 1. System Updates
# =============================================================================
echo -e "${BLUE}[1/7] Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# =============================================================================
# 2. Install Docker
# =============================================================================
echo -e "${BLUE}[2/7] Installing Docker...${NC}"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}Docker installed successfully${NC}"
else
    echo -e "${YELLOW}Docker already installed${NC}"
fi

# =============================================================================
# 3. Install Azure CLI
# =============================================================================
echo -e "${BLUE}[3/7] Installing Azure CLI...${NC}"
if ! command -v az &> /dev/null; then
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    echo -e "${GREEN}Azure CLI installed successfully${NC}"
else
    echo -e "${YELLOW}Azure CLI already installed${NC}"
fi

# =============================================================================
# 4. Create Directory Structure
# =============================================================================
echo -e "${BLUE}[4/7] Creating directory structure...${NC}"
sudo mkdir -p /data/postgres
sudo mkdir -p /data/sync
sudo mkdir -p /data/backtest
sudo mkdir -p /data/logs
sudo mkdir -p /data/app
sudo chown -R $USER:$USER /data
echo -e "${GREEN}Directories created:${NC}"
echo "  /data/postgres  - PostgreSQL data"
echo "  /data/sync      - Data import/export"
echo "  /data/backtest  - Backtest outputs"
echo "  /data/logs      - Application logs"
echo "  /data/app       - Application code"

# =============================================================================
# 5. Copy docker-compose file
# =============================================================================
echo -e "${BLUE}[5/7] Setting up docker-compose...${NC}"
cat > /data/docker-compose.yml << 'COMPOSE_EOF'
# Azure Backtest Infrastructure
# Optimized for Standard_B16s_v2 (16 vCPUs, 64GB RAM)

version: "3.8"

services:
  postgres:
    image: postgres:15
    container_name: postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: forex
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - /data/postgres:/var/lib/postgresql/data
      - /data/sync:/sync
    shm_size: 16gb
    command:
      - "postgres"
      - "-c"
      - "shared_buffers=16GB"
      - "-c"
      - "effective_cache_size=48GB"
      - "-c"
      - "work_mem=512MB"
      - "-c"
      - "maintenance_work_mem=2GB"
      - "-c"
      - "max_parallel_workers=16"
      - "-c"
      - "max_parallel_workers_per_gather=8"
      - "-c"
      - "max_worker_processes=20"
      - "-c"
      - "parallel_tuple_cost=0.001"
      - "-c"
      - "parallel_setup_cost=100"
      - "-c"
      - "wal_buffers=256MB"
      - "-c"
      - "checkpoint_completion_target=0.9"
      - "-c"
      - "max_wal_size=4GB"
      - "-c"
      - "max_connections=100"
      - "-c"
      - "listen_addresses=*"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

  task-worker:
    build:
      context: /data/app
      dockerfile: Dockerfile
    image: task-worker:local
    container_name: task-worker
    restart: "no"
    volumes:
      - /data/backtest:/app/backtest_output
      - /data/logs:/app/logs
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/forex
      - STRATEGY_CONFIG_DATABASE_URL=postgresql://postgres:postgres@postgres:5432/strategy_config
      - PARALLEL_WORKERS=14
      - CHUNK_DAYS=7
      - ORDER_API_URL=
      - IG_API_KEY=
      - IG_PWD=
      - CLAUDE_API_KEY=
      - MINIO_ENDPOINT=
    depends_on:
      postgres:
        condition: service_healthy
    command: ["tail", "-f", "/dev/null"]

networks:
  default:
    name: backtest-network
COMPOSE_EOF

echo -e "${GREEN}docker-compose.yml created at /data/docker-compose.yml${NC}"

# =============================================================================
# 6. Start PostgreSQL
# =============================================================================
echo -e "${BLUE}[6/7] Starting PostgreSQL...${NC}"
cd /data

# Need to run docker as new group member
sudo docker compose up -d postgres
echo "Waiting for PostgreSQL to be ready..."
sleep 15

# Create additional databases
echo "Creating databases..."
sudo docker exec postgres psql -U postgres -c "CREATE DATABASE IF NOT EXISTS strategy_config;" 2>/dev/null || \
    sudo docker exec postgres psql -U postgres -c "CREATE DATABASE strategy_config;" 2>/dev/null || \
    echo "strategy_config database may already exist"

echo -e "${GREEN}PostgreSQL is running${NC}"

# =============================================================================
# 7. Run Database Migrations
# =============================================================================
echo -e "${BLUE}[7/7] Database setup notes...${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT: You need to run migrations manually after pushing code.${NC}"
echo ""
echo "After pushing the task-worker image and syncing data, run:"
echo "  docker exec task-worker python -c \"from forex_scanner.core.database import DatabaseManager; print('DB OK')\""
echo ""

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Log out and back in (for docker group): exit && ssh azureuser@\$(hostname -I | awk '{print \$1}')"
echo "  2. Login to ACR: az acr login --name tradesystembacktest"
echo "  3. Pull task-worker: docker pull tradesystembacktest.azurecr.io/task-worker:latest"
echo "  4. Push data from local: ./scripts/azure_backtest.sh push-data"
echo "  5. Start task-worker: cd /data && docker compose up -d task-worker"
echo ""
echo "VM Information:"
echo "  - Data directory: /data"
echo "  - PostgreSQL: localhost:5432 (internal only)"
echo "  - Docker compose: /data/docker-compose.yml"
echo ""
echo -e "${GREEN}Ready for backtesting!${NC}"
