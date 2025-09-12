#!/bin/bash

set -e

PGADMIN_DIR="./pgadmin"
KEY_FILE="$PGADMIN_DIR/pgadmin4.key"
SERVERS_FILE="$PGADMIN_DIR/servers.json"

echo "🔍 Checking pgAdmin bootstrap files..."

# 1. Create pgadmin folder if missing
mkdir -p "$PGADMIN_DIR"

# 2. Generate pgadmin4.key if missing
if [[ ! -f "$KEY_FILE" ]]; then
    echo "🔐 Generating crypt key: $KEY_FILE"
    openssl rand -base64 24 > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
else
    echo "✅ Found existing crypt key: $KEY_FILE"
fi

# 3. Warn if servers.json is missing
if [[ ! -f "$SERVERS_FILE" ]]; then
    echo "⚠️  Warning: $SERVERS_FILE not found."
    echo "   You can create it with:"
    echo "   cat > $SERVERS_FILE <<EOF"
    echo '   {
      "Servers": {
        "1": {
          "Name": "Postgres",
          "Group": "Servers",
          "Host": "postgres",
          "Port": 5432,
          "MaintenanceDB": "forex",
          "Username": "postgres",
          "SSLMode": "prefer"
        }
      }
    }'
    echo "   EOF"
else
    echo "✅ Found: $SERVERS_FILE"
fi

echo "✅ Bootstrap check complete."
