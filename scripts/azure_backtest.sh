#!/bin/bash
# =============================================================================
# Azure Backtest CLI
# Main command-line interface for Azure backtest operations
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration - Override these with environment variables
AZURE_RG="${AZURE_RG:-Backtesting}"
AZURE_VM_NAME="${AZURE_VM_NAME:-backtest-vm}"
AZURE_VM_IP="${AZURE_VM_IP:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/azure_backtest_rsa}"
PROJECT_DIR="${SCRIPT_DIR}/.."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

show_help() {
    cat << EOF
${BLUE}Azure Backtest CLI${NC}
==================

Manage and run backtests on Azure VM (48 cores, 384GB RAM).

${YELLOW}Usage:${NC} $0 <command> [options]

${YELLOW}VM Management:${NC}
  start-vm           Start the Azure VM (if stopped)
  stop-vm            Stop and deallocate VM (no charges when stopped)
  vm-status          Check VM power state
  vm-ip              Get VM public IP address

${YELLOW}Data Sync:${NC}
  push-data [days]   Push ig_candles + strategy_config to Azure (default: 30 days)
  pull-results [id]  Pull backtest results from Azure (use 'all' for all results)

${YELLOW}Backtest Execution:${NC}
  run <args>         Run backtest on Azure (same args as bt.py)
  status             Show running backtests on Azure
  logs               Tail backtest logs from Azure
  list               List recent backtest executions on Azure

${YELLOW}Setup & Maintenance:${NC}
  setup              Run initial VM setup script
  push-code          Sync worker code to Azure VM and rebuild container
  ssh                Open SSH session to Azure VM
  build              Build task-worker container on Azure VM

${YELLOW}Examples:${NC}
  $0 start-vm                                    # Start the VM
  $0 push-data 60                                # Push 60 days of candle data
  $0 run EURUSD 30 --parallel --workers 44       # Run 30-day EURUSD backtest
  $0 run --all 14 --vary fixed_stop_loss_pips=8:14:2  # Parameter variation
  $0 list                                        # List executions
  $0 pull-results 42                             # Pull execution ID 42
  $0 stop-vm                                     # Stop VM to save costs

${YELLOW}Environment Variables:${NC}
  AZURE_VM_IP        Azure VM public IP (auto-detected if not set)
  SSH_KEY            Path to SSH private key (default: ~/.ssh/azure_backtest_rsa)
  AZURE_RG           Resource group (default: Bactesting)
  AZURE_VM_NAME      VM name (default: backtest-vm)
  ACR_NAME           Container registry name (default: tradesystembacktest)

EOF
}

check_az_login() {
    if ! az account show &>/dev/null; then
        echo -e "${RED}Error: Not logged into Azure CLI${NC}"
        echo "Run: az login"
        exit 1
    fi
}

get_vm_ip() {
    if [[ -n "${AZURE_VM_IP}" ]]; then
        echo "${AZURE_VM_IP}"
        return
    fi

    local ip
    ip=$(az vm show -d -g "${AZURE_RG}" -n "${AZURE_VM_NAME}" --query publicIps -o tsv 2>/dev/null || echo "")

    if [[ -z "${ip}" ]]; then
        echo -e "${RED}Error: Could not get VM IP. Is the VM running?${NC}" >&2
        exit 1
    fi

    echo "${ip}"
}

check_vm_running() {
    local state
    state=$(az vm show -d -g "${AZURE_RG}" -n "${AZURE_VM_NAME}" --query powerState -o tsv 2>/dev/null || echo "unknown")

    if [[ "${state}" != "VM running" ]]; then
        echo -e "${RED}Error: VM is not running (state: ${state})${NC}"
        echo "Run: $0 start-vm"
        exit 1
    fi
}

check_ssh_key() {
    if [[ ! -f "${SSH_KEY}" ]]; then
        echo -e "${RED}Error: SSH key not found: ${SSH_KEY}${NC}"
        echo ""
        echo "Generate a new key pair:"
        echo "  ssh-keygen -t rsa -b 4096 -f ${SSH_KEY}"
        echo ""
        echo "Then add the public key to the VM:"
        echo "  az vm user update -g ${AZURE_RG} -n ${AZURE_VM_NAME} -u azureuser --ssh-key-value \"\$(cat ${SSH_KEY}.pub)\""
        exit 1
    fi
}

# =============================================================================
# VM Management Commands
# =============================================================================

cmd_start_vm() {
    check_az_login
    echo -e "${BLUE}Starting Azure VM: ${AZURE_VM_NAME}...${NC}"

    az vm start --resource-group "${AZURE_RG}" --name "${AZURE_VM_NAME}"

    echo -e "${GREEN}VM started${NC}"
    echo "Waiting for services to be ready..."
    sleep 30

    local ip
    ip=$(get_vm_ip)
    echo ""
    echo -e "${GREEN}VM is ready!${NC}"
    echo "IP: ${ip}"
    echo ""
    echo "Start Docker services:"
    echo "  ssh -i ${SSH_KEY} azureuser@${ip} 'cd /data && docker compose up -d'"
}

cmd_stop_vm() {
    check_az_login
    echo -e "${BLUE}Stopping Azure VM: ${AZURE_VM_NAME}...${NC}"
    echo -e "${YELLOW}(VM will be deallocated - no compute charges while stopped)${NC}"

    az vm deallocate --resource-group "${AZURE_RG}" --name "${AZURE_VM_NAME}"

    echo -e "${GREEN}VM stopped and deallocated${NC}"
    echo "Data on managed disk is preserved."
    echo "Run '$0 start-vm' to restart."
}

cmd_vm_status() {
    check_az_login
    echo -e "${BLUE}VM Status:${NC}"

    az vm show -d -g "${AZURE_RG}" -n "${AZURE_VM_NAME}" \
        --query "{Name:name, State:powerState, IP:publicIps, Size:hardwareProfile.vmSize}" \
        -o table
}

cmd_vm_ip() {
    check_az_login
    get_vm_ip
}

# =============================================================================
# Data Sync Commands
# =============================================================================

cmd_push_data() {
    local days="${1:-30}"
    check_ssh_key
    check_vm_running

    export AZURE_VM_IP="$(get_vm_ip)"
    export SSH_KEY

    echo -e "${BLUE}Pushing data to Azure (${days} days)...${NC}"
    "${SCRIPT_DIR}/azure_backtest_push.sh" "${days}"
}

cmd_pull_results() {
    local execution_id="${1:-}"
    check_ssh_key
    check_vm_running

    export AZURE_VM_IP="$(get_vm_ip)"
    export SSH_KEY

    "${SCRIPT_DIR}/azure_backtest_pull.sh" "${execution_id}"
}

# =============================================================================
# Backtest Execution Commands
# =============================================================================

cmd_run() {
    check_ssh_key
    check_vm_running

    local ip
    ip=$(get_vm_ip)

    echo -e "${BLUE}Running backtest on Azure...${NC}"
    echo "Command: python /app/forex_scanner/bt.py $*"
    echo ""

    # Ensure task-worker is running, then execute backtest
    ssh -i "${SSH_KEY}" "azureuser@${ip}" << REMOTE_SCRIPT
# Start container if not running
if ! docker ps --format '{{.Names}}' | grep -q '^task-worker$'; then
    echo "Starting task-worker container..."
    cd /data && docker compose up -d task-worker
    sleep 3
fi

# Run backtest
docker exec task-worker python /app/forex_scanner/bt.py $*
REMOTE_SCRIPT
}

cmd_status() {
    check_ssh_key
    check_vm_running

    local ip
    ip=$(get_vm_ip)

    echo -e "${BLUE}Checking running backtests...${NC}"

    ssh -i "${SSH_KEY}" "azureuser@${ip}" \
        "docker exec task-worker ps aux | grep -E 'bt.py|backtest' | grep -v grep || echo 'No backtests running'"
}

cmd_logs() {
    check_ssh_key
    check_vm_running

    local ip
    ip=$(get_vm_ip)

    echo -e "${BLUE}Tailing backtest logs...${NC}"
    echo "(Press Ctrl+C to stop)"
    echo ""

    ssh -i "${SSH_KEY}" "azureuser@${ip}" \
        "docker logs -f task-worker --tail 100"
}

cmd_list() {
    check_ssh_key
    check_vm_running

    local ip
    ip=$(get_vm_ip)

    echo -e "${BLUE}Recent backtest executions on Azure:${NC}"
    echo ""

    ssh -i "${SSH_KEY}" "azureuser@${ip}" << 'REMOTE_SCRIPT'
docker exec postgres psql -U postgres -d forex -c "
    SELECT
        id,
        LEFT(execution_name, 30) as name,
        strategy_name as strategy,
        status,
        to_char(start_time, 'MM-DD HH24:MI') as started,
        COALESCE(completed_combinations, 0) as signals
    FROM backtest_executions
    ORDER BY start_time DESC
    LIMIT 15;
"
REMOTE_SCRIPT
}

# =============================================================================
# Setup & Maintenance Commands
# =============================================================================

cmd_setup() {
    check_ssh_key
    check_vm_running

    local ip
    ip=$(get_vm_ip)

    echo -e "${BLUE}Running VM setup script...${NC}"
    ssh -i "${SSH_KEY}" "azureuser@${ip}" < "${SCRIPT_DIR}/azure_vm_setup.sh"
}

cmd_push_code() {
    check_ssh_key
    check_vm_running

    local ip
    ip=$(get_vm_ip)

    echo -e "${BLUE}Syncing worker code to Azure VM...${NC}"

    # Create app directory on Azure
    ssh -i "${SSH_KEY}" "azureuser@${ip}" "mkdir -p /data/app"

    # Sync worker/app directory directly to /data/app (matches local volume mount structure)
    # This puts forex_scanner at /data/app/forex_scanner (builds to /app/forex_scanner in container)
    rsync -avz --progress \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'logs' \
        --exclude '*.log' \
        --exclude 'archive' \
        --exclude 'venv' \
        --exclude '.pytest_cache' \
        -e "ssh -i ${SSH_KEY}" \
        "${PROJECT_DIR}/worker/app/" \
        "azureuser@${ip}:/data/app/"

    # Copy Dockerfile and requirements.txt from worker/
    scp -i "${SSH_KEY}" \
        "${PROJECT_DIR}/worker/Dockerfile" \
        "${PROJECT_DIR}/worker/requirements.txt" \
        "azureuser@${ip}:/data/app/"

    # Update docker-compose.yml on Azure to use correct build context
    ssh -i "${SSH_KEY}" "azureuser@${ip}" \
        "sed -i 's|context: /data/app/worker|context: /data/app|' /data/docker-compose.yml"

    echo -e "${GREEN}Code synced to /data/app/${NC}"
    echo "  - forex_scanner -> /data/app/forex_scanner/"
    echo "  - Dockerfile -> /data/app/Dockerfile"
    echo "  - requirements.txt -> /data/app/requirements.txt"
    echo ""
    echo "Now build the container:"
    echo "  $0 build"
}

cmd_build() {
    check_ssh_key
    check_vm_running

    local ip
    ip=$(get_vm_ip)

    echo -e "${BLUE}Building task-worker container on Azure VM...${NC}"

    ssh -i "${SSH_KEY}" "azureuser@${ip}" << 'REMOTE_SCRIPT'
cd /data
docker compose build task-worker
echo ""
echo "Starting task-worker container..."
docker compose up -d task-worker
docker ps --format "table {{.Names}}\t{{.Status}}"
REMOTE_SCRIPT

    echo -e "${GREEN}Build complete and container started${NC}"
}

cmd_ssh() {
    check_ssh_key
    check_vm_running

    local ip
    ip=$(get_vm_ip)

    echo -e "${BLUE}Connecting to Azure VM...${NC}"
    ssh -i "${SSH_KEY}" "azureuser@${ip}"
}

# =============================================================================
# Main Command Router
# =============================================================================

case "${1:-help}" in
    # VM Management
    start-vm)       cmd_start_vm ;;
    stop-vm)        cmd_stop_vm ;;
    vm-status)      cmd_vm_status ;;
    vm-ip)          cmd_vm_ip ;;

    # Data Sync
    push-data)      shift; cmd_push_data "$@" ;;
    pull-results)   shift; cmd_pull_results "$@" ;;

    # Backtest Execution
    run)            shift; cmd_run "$@" ;;
    status)         cmd_status ;;
    logs)           cmd_logs ;;
    list)           cmd_list ;;

    # Setup & Maintenance
    setup)          cmd_setup ;;
    push-code)      cmd_push_code ;;
    build)          cmd_build ;;
    ssh)            cmd_ssh ;;

    # Help
    help|--help|-h) show_help ;;

    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
