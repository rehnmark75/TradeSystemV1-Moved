# Database Backup & Restore Guide

This guide provides comprehensive instructions for backing up and restoring PostgreSQL databases in the TradeSystemV1 environment.

## 📋 Overview

The TradeSystemV1 backup system provides:
- **Automated daily backups** of both `forex` and `forex_config` databases
- **Vector database backup** (ChromaDB embeddings and indexes)
- **Application logs backup** (all service logs for debugging)
- **PgAdmin configuration backup** (saved queries and settings)
- **Intelligent retention policies** (7 daily, 4 weekly, 12 monthly backups)
- **Compression and integrity verification**
- **Comprehensive monitoring and health checks**
- **Easy restore procedures** for disaster recovery

## 🏗️ Backup System Architecture

### Components
- **`db-backup` service**: Automated backup container running daily at 2 AM
- **Backup scripts**: Located in `scripts/` directory
- **Storage**: Backups stored on external drive at `/media/hr/Data/TradeSystemV1-Backups/postgresbackup/`
- **Monitoring**: Health checks and status monitoring included

### File Structure
```
scripts/
├── backup_database.sh      # Original backup script
├── enhanced_backup.sh      # Enhanced backup script (currently used)
├── simple_backup.sh        # Simple backup script
├── backup_monitor.py       # Monitoring and health checks
├── backup_health.sh        # Simple health check for Docker
└── restore_database.sh     # Restore script

External Drive: /media/hr/Data/TradeSystemV1-Backups/postgresbackup/
├── forex_backup_YYYYMMDD_HHMMSS.sql.gz           # PostgreSQL forex database
├── forex_config_backup_YYYYMMDD_HHMMSS.sql.gz    # PostgreSQL config database
└── additional_backup_YYYYMMDD_HHMMSS.tar.gz      # Vector DB + Logs + PgAdmin config
```

## 🚀 Quick Start

### Start Backup Service
```bash
# Start the backup service
docker-compose up -d db-backup

# Check backup service status
docker-compose ps db-backup

# View backup logs
docker-compose logs -f db-backup
```

### Manual Backup
```bash
# Run immediate backup
docker exec db-backup /scripts/backup_database.sh

# Backup with verification
docker exec db-backup /scripts/backup_database.sh --verify

# Dry run to see what would be backed up
docker exec db-backup /scripts/backup_database.sh --dry-run
```

### Check Backup Status
```bash
# Quick health check
docker exec db-backup /scripts/backup_health.sh

# Detailed status report
docker exec db-backup /scripts/backup_monitor.py

# JSON format for automation
docker exec db-backup /scripts/backup_monitor.py --format json
```

## 📋 Backup Operations

### Automated Backups

The system automatically creates backups daily at 2 AM UTC. The backup service:

1. **Backs up both databases** (`forex` and `forex_config`)
2. **Compresses backups** using gzip
3. **Verifies integrity** of created backups
4. **Manages retention** according to policy
5. **Logs all activities** to `/app/logs/backup.log`

### Manual Backup Commands

```bash
# Enhanced backup (includes PostgreSQL + Vector DB + Logs + PgAdmin)
docker exec db-backup /scripts/enhanced_backup.sh

# Simple backup (PostgreSQL databases only)
docker exec db-backup /scripts/simple_backup.sh

# Original backup with options
docker exec db-backup /scripts/backup_database.sh [OPTIONS]

# Available options for backup_database.sh:
#   -h, --help     Show help message
#   -d, --dry-run  Show what would be done
#   -v, --verify   Verify existing backups only
#   --cleanup      Run cleanup only
#   --report       Generate backup report
```

### Backup File Naming

Backup files follow the pattern:
```
{database}_backup_{YYYYMMDD}_{HHMMSS}.sql.gz
additional_backup_{YYYYMMDD}_{HHMMSS}.tar.gz
```

Examples:
- `forex_backup_20250918_120000.sql.gz` (PostgreSQL forex database)
- `forex_config_backup_20250918_120003.sql.gz` (PostgreSQL config database)
- `additional_backup_20250918_120000.tar.gz` (Vector DB + Logs + PgAdmin config)

### Enhanced Backup Contents

The **additional_backup_*.tar.gz** file contains:
```
./vectordb/                    # ChromaDB vector database
./data/                        # Vector database data files
./logs/worker/                 # Task worker logs
./logs/vector-db/              # Vector database logs
./logs/tradingview/            # TradingView service logs
./logs/economic-calendar/      # Economic calendar logs
./logs/dev/                    # Development FastAPI logs
./logs/prod/                   # Production FastAPI logs
./logs/stream/                 # Streaming service logs
./logs/backup/                 # Backup operation logs
./pgadmin_config_*.tar.gz      # PgAdmin configuration backup
```

### Retention Policy

- **Daily**: Keep 7 most recent daily backups
- **Weekly**: Keep 4 Sunday backups (monthly retention)
- **Monthly**: Keep 12 backups from the 1st of each month (yearly retention)

## 🔄 Restore Operations

### Restore Scripts

The restore system provides flexible options for database restoration:

```bash
# Basic restore (interactive)
docker exec -it db-backup /scripts/restore_database.sh <backup-file>

# Force restore without prompts
docker exec db-backup /scripts/restore_database.sh --force <backup-file>

# Clean restore (drop and recreate database)
docker exec db-backup /scripts/restore_database.sh --clean <backup-file>

# Restore with verification
docker exec db-backup /scripts/restore_database.sh --verify <backup-file>

# Create backup before restore
docker exec db-backup /scripts/restore_database.sh --backup-before <backup-file>
```

### Restore Examples

```bash
# Restore latest forex backup
docker exec -it db-backup /scripts/restore_database.sh --database forex --latest

# List available backups
docker exec db-backup /scripts/restore_database.sh --list

# Restore specific backup with full safety
docker exec -it db-backup /scripts/restore_database.sh \
  --backup-before --verify --clean \
  forex_backup_20250918_120000.sql.gz

# Emergency restore (automated)
docker exec db-backup /scripts/restore_database.sh \
  --force --clean --verify \
  forex_backup_20250918_120000.sql.gz
```

### Restore Options

| Option | Description |
|--------|-------------|
| `--database NAME` | Target database (forex/forex_config) |
| `--force` | Skip confirmation prompts |
| `--clean` | Drop and recreate database before restore |
| `--verify` | Verify restore after completion |
| `--backup-before` | Create backup before restore |
| `--latest` | Use latest backup for specified database |
| `--list` | List available backup files |

## 📊 Monitoring & Health Checks

### Health Check Commands

```bash
# Simple health check (used by Docker)
docker exec db-backup /scripts/backup_health.sh
# Exit codes: 0=healthy, 1=warning, 2=critical

# Comprehensive monitoring
docker exec db-backup /scripts/backup_monitor.py

# Health check with detailed output
docker exec db-backup /scripts/backup_monitor.py --health-check

# Include statistics in report
docker exec db-backup /scripts/backup_monitor.py --include-stats

# JSON output for automation
docker exec db-backup /scripts/backup_monitor.py --format json --health-check
```

### Docker Health Status

```bash
# Check backup service health via Docker
docker inspect db-backup --format='{{.State.Health.Status}}'

# View health check history
docker inspect db-backup --format='{{range .State.Health.Log}}{{.Output}}{{end}}'
```

### Log Monitoring

```bash
# View backup logs
docker exec db-backup tail -f /app/logs/backup.log

# Search for errors
docker exec db-backup grep ERROR /app/logs/backup.log

# View recent backup completion
docker exec db-backup grep "Backup process completed" /app/logs/backup.log | tail -5
```

## 🚨 Emergency Procedures

### Complete System Recovery

If the entire database needs to be restored:

```bash
# 1. Stop all services that use the database
docker-compose stop fastapi-dev fastapi-prod task-worker streamlit

# 2. List available backups
docker exec db-backup /scripts/restore_database.sh --list

# 3. Restore forex database (latest)
docker exec db-backup /scripts/restore_database.sh \
  --force --clean --verify --database forex --latest

# 4. Restore forex_config database (latest)
docker exec db-backup /scripts/restore_database.sh \
  --force --clean --verify --database forex_config --latest

# 5. Restart all services
docker-compose up -d
```

### Point-in-Time Recovery

For recovery to a specific point in time:

```bash
# 1. Find backup closest to desired time
docker exec db-backup /scripts/restore_database.sh --list

# 2. Create safety backup
docker exec db-backup /scripts/backup_database.sh

# 3. Restore specific backup
docker exec db-backup /scripts/restore_database.sh \
  --backup-before --clean --verify \
  forex_backup_YYYYMMDD_HHMMSS.sql.gz

# 4. Verify data integrity
docker exec postgres psql -U postgres -d forex -c "SELECT COUNT(*) FROM ig_candles;"
```

### Backup Corruption Recovery

If backups are corrupted:

```bash
# 1. Check all backup integrity
docker exec db-backup /scripts/backup_monitor.py --health-check

# 2. Verify specific backup
docker exec db-backup /scripts/backup_database.sh --verify

# 3. Create new backup immediately
docker exec db-backup /scripts/backup_database.sh

# 4. Test restore with verification
docker exec db-backup /scripts/restore_database.sh --verify <backup-file>
```

## 🔧 Troubleshooting

### Common Issues

#### Backup Service Not Running
```bash
# Check service status
docker-compose ps db-backup

# Check logs for errors
docker-compose logs db-backup

# Restart service
docker-compose restart db-backup
```

#### Insufficient Disk Space
```bash
# Check disk usage
docker exec db-backup df -h

# Manual cleanup (removes old backups)
docker exec db-backup /scripts/backup_database.sh --cleanup

# Check backup directory size
docker exec db-backup du -sh /app/postgresbackup
```

#### Backup Integrity Issues
```bash
# Verify all backups
docker exec db-backup /scripts/backup_monitor.py --health-check

# Test specific backup
docker exec db-backup gzip -t /app/postgresbackup/forex_backup_*.sql.gz

# Re-create backup if corrupted
docker exec db-backup /scripts/backup_database.sh
```

### Log Analysis

Common log patterns to look for:

```bash
# Successful backups
grep "Backup process completed" /app/logs/backup.log

# Errors
grep "ERROR" /app/logs/backup.log

# Warnings
grep "WARN" /app/logs/backup.log

# Backup statistics
grep "Backup stats" /app/logs/backup.log
```

## ⚙️ Configuration

### Environment Variables

The backup service can be configured via environment variables in `docker-compose.yml`:

```yaml
environment:
  - BACKUP_SCHEDULE=0 2 * * *     # Cron schedule (daily at 2 AM)
  - BACKUP_RETENTION_DAYS=7       # Daily retention
  - WEEKLY_RETENTION=4             # Weekly retention
  - MONTHLY_RETENTION=12           # Monthly retention
```

### Custom Scheduling

To change backup timing, modify the `docker-compose.yml`:

```yaml
# For 6-hour backups:
command: >
  sh -c "
    apk add --no-cache docker-cli bc coreutils findutils &&
    while true; do
      /scripts/backup_database.sh &&
      sleep 21600  # 6 hours
    done
  "
```

### Storage Location

Backups are stored on external drive at `/media/hr/Data/TradeSystemV1-Backups/postgresbackup/` which:
- Is mounted directly from external drive in the backup container
- Provides additional storage capacity separate from system drive
- Should be safely unmounted when disconnecting external drive

## 📈 Best Practices

### Regular Testing
- **Monthly**: Test restore procedures with latest backups
- **Quarterly**: Perform full disaster recovery simulation
- **Before major updates**: Create verified backups

### Monitoring
- Monitor backup service health daily
- Set up alerts for backup failures
- Verify backup integrity weekly

### Security
- Backup files contain sensitive data
- Ensure proper file permissions
- Consider encryption for off-site storage

### Performance
- Backups run during low-activity hours (2 AM)
- Monitor disk space usage
- Clean up old backups regularly

## 🔗 Integration with Existing Systems

### Commands Integration

Add to your `claude-commands.md`:

```bash
# Database backup operations
docker exec db-backup /scripts/backup_database.sh              # Manual backup
docker exec db-backup /scripts/backup_monitor.py               # Status check
docker exec db-backup /scripts/restore_database.sh --list      # List backups

# Emergency restore
docker exec db-backup /scripts/restore_database.sh --force --clean --verify <backup-file>
```

### Monitoring Integration

The backup system integrates with your existing monitoring:
- Logs to `/app/logs/backup.log`
- Docker health checks every 6 hours
- JSON status output for automation

## 📞 Support

For backup system issues:

1. **Check health status**: `docker exec db-backup /scripts/backup_monitor.py --health-check`
2. **Review logs**: `docker-compose logs db-backup`
3. **Test restore**: Use `--dry-run` options first
4. **Manual intervention**: All scripts support `--help` for detailed usage

Remember: The backup system is designed to be robust and self-healing, but regular monitoring ensures optimal performance and data protection.