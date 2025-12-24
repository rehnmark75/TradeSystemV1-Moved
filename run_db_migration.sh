#!/bin/bash
# Run database migration for early breakeven tracking

echo "Running database migration for early breakeven columns..."
docker exec fastapi-dev python3 /app/run_migration.py
echo "Migration script completed."
