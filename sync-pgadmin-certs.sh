#!/bin/bash

# Source certs from Let's Encrypt live folder
SOURCE="./certs/live/trader.nordicbynature.se"
TARGET="./pgadmin/certs"

echo "🛡️  Syncing latest certs to pgAdmin..."

mkdir -p "$TARGET"

cp "$SOURCE/fullchain.pem" "$TARGET/server.cert"
cp "$SOURCE/privkey.pem" "$TARGET/server.key"

chmod 644 "$TARGET/server."*

echo "✅ Certificates updated. Restarting pgAdmin..."
docker-compose restart pgadmin