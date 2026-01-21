#!/bin/bash
# Host-level script to clean up stale pending orders in the trading system
#
# This script runs the cleanup inside the task-worker Docker container
# Install with: crontab -e
# Add line: */15 * * * * /home/hr/Projects/TradeSystemV1/scripts/cleanup_stale_orders.sh >> /var/log/cleanup_stale_orders.log 2>&1

# Change to project directory
cd /home/hr/Projects/TradeSystemV1

# Run cleanup inside docker container
docker exec task-worker python /app/forex_scanner/maintenance/cleanup_stale_orders.py

# Exit with the same exit code as the cleanup script
exit $?
