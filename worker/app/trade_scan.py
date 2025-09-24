#!/usr/bin/env python3
"""
Enhanced Trade Scanner - Docker Entry Point
Refactored to use modular architecture while maintaining Docker compatibility

This file serves as the main entry point for Docker containers and delegates
all complex functionality to the modular trading orchestrator and existing
components. 

FIXED: 
- Now properly honors config.py settings for Claude and Trading
- Uses new intelligence_preset system instead of old intelligence_mode
- Proper database manager initialization
- Fixed duplicate orchestrator creation

Usage:
    python trade_scan.py [scan|live|test-claude|docker|status] [scan_interval]
    
Docker CMD compatibility maintained:
    CMD ["python", "trade_scan.py"]
"""

import sys
import os
import time
import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pytz

class StockholmFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=pytz.timezone('Europe/Stockholm'))
        return dt.timetuple()

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=pytz.timezone('Europe/Stockholm'))
        return dt.strftime('%Y-%m-%d %H:%M:%S %Z')


def cleanup_old_log_files(log_dir: str, log_prefix: str, days_to_keep: int = 7):
    """
    Clean up old log files that exceed the retention period.

    Args:
        log_dir: Directory containing log files
        log_prefix: Log file prefix to match (e.g., 'forex_scanner')
        days_to_keep: Number of days to keep (default: 7)
    """
    try:
        import glob
        from pathlib import Path

        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Find all log files matching the prefix
        pattern = os.path.join(log_dir, f"{log_prefix}*.log*")
        log_files = glob.glob(pattern)

        removed_count = 0
        total_size_removed = 0

        for log_file in log_files:
            try:
                # Get file modification time
                file_path = Path(log_file)
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                # Remove file if older than cutoff
                if file_mtime < cutoff_date:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    removed_count += 1
                    total_size_removed += file_size
                    print(f"🗑️ Removed old log file: {log_file} ({file_size/1024/1024:.1f}MB)")

            except Exception as e:
                print(f"⚠️ Failed to remove log file {log_file}: {e}")

        if removed_count > 0:
            print(f"✅ Cleaned up {removed_count} old log files, freed {total_size_removed/1024/1024:.1f}MB")
        else:
            print(f"📁 No old log files found (keeping logs newer than {days_to_keep} days)")

    except Exception as e:
        print(f"❌ Error during log cleanup: {e}")

# Add the project root and forex_scanner to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if 'forex_scanner' in current_dir else current_dir

# Add both the current directory and potential parent directories to path
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'forex_scanner'))

# Also try common Docker paths
docker_paths = [
    '/app',
    '/app/forex_scanner',
    '/app/worker/app/forex_scanner'
]

for path in docker_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)

print(f"Current directory: {current_dir}")
print(f"Python path: {sys.path[:5]}...")  # Show first 5 paths

# Import from modular structure
try:
    from core.trading.trading_orchestrator import create_trading_orchestrator
    from core.database import DatabaseManager
    import config
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    # Try alternative import paths
    try:
        from forex_scanner.core.trading.trading_orchestrator import create_trading_orchestrator
        from forex_scanner.core.database import DatabaseManager
        from forex_scanner import config
        print("✅ Successfully imported modules (alternative path)")
    except ImportError as e2:
        print(f"❌ All import attempts failed: {e2}")
        print("Available paths:")
        for path in sys.path[:10]:
            print(f"  {path}")
        raise


class TradingSystem:
    """
    Main trading system entry point for Docker
    
    This class serves as a thin wrapper around the TradingOrchestrator,
    maintaining interface compatibility with the original TradingSystem
    while delegating all complex functionality to modular components.
    
    FIXED: Now properly respects config.py settings and uses new intelligence system.
    """
    
    def __init__(
        self,
        intelligence_mode: str = None,  # CHANGED: Default to None to use new system
        intelligence_preset: str = None,  # NEW: Support for new intelligence preset system
        enable_trading: bool = None,  # CHANGED: Default to None to use config
        enable_claude_analysis: bool = None,  # CHANGED: Default to None to use config
        scan_interval: int = None
    ):
        """
        Initialize the trading system
        
        Args:
            intelligence_mode: Legacy intelligence mode (deprecated)
            intelligence_preset: New intelligence preset ('disabled', 'minimal', 'balanced', etc.)
            enable_trading: Enable actual order execution (None = use config.py)
            enable_claude_analysis: Enable Claude AI analysis (None = use config.py)
            scan_interval: Scan interval in seconds (defaults to config)
        """
        # Setup logging first
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # FIXED: Determine intelligence configuration using new preset system
        if intelligence_preset is not None:
            self.intelligence_preset = intelligence_preset
            self.intelligence_mode = self._map_preset_to_mode(intelligence_preset)
            self.logger.info(f"🧠 Using explicit intelligence preset: {intelligence_preset}")
        elif intelligence_mode is not None:
            self.intelligence_mode = intelligence_mode
            self.intelligence_preset = self._map_mode_to_preset(intelligence_mode)
            self.logger.info(f"🧠 Using legacy intelligence mode: {intelligence_mode} → preset: {self.intelligence_preset}")
        else:
            # Use config.py settings
            self.intelligence_preset = getattr(config, 'INTELLIGENCE_PRESET', 'minimal')
            self.intelligence_mode = getattr(config, 'INTELLIGENCE_MODE', 'live_only')
            self.logger.info(f"🧠 Using config intelligence preset: {self.intelligence_preset}")
        
        # FIXED: Check multiple config variable names and only override if explicitly passed
        if enable_trading is not None:
            self.enable_trading = enable_trading
            self.logger.info(f"🔧 Trading explicitly set to: {enable_trading}")
        else:
            # Check multiple possible config variable names
            self.enable_trading = (
                getattr(config, 'AUTO_TRADING_ENABLED', False) or
                getattr(config, 'ENABLE_ORDER_EXECUTION', False) or
                getattr(config, 'TRADING_ENABLED', False) or
                getattr(config, 'LIVE_TRADING', False)
            )
            self.logger.info(f"🔧 Trading from config: {self.enable_trading}")
        
        if enable_claude_analysis is not None:
            self.enable_claude_analysis = enable_claude_analysis
            self.logger.info(f"🔧 Claude explicitly set to: {enable_claude_analysis}")
        else:
            # Check multiple possible config variable names
            self.enable_claude_analysis = (
                getattr(config, 'ENABLE_CLAUDE_ANALYSIS', False) or
                getattr(config, 'CLAUDE_ANALYSIS_ENABLED', False) or
                getattr(config, 'USE_CLAUDE_ANALYSIS', False) or
                getattr(config, 'CLAUDE_ANALYSIS_MODE', 'disabled') != 'disabled'
            )
            self.logger.info(f"🔧 Claude from config: {self.enable_claude_analysis}")
        
        self.scan_interval = scan_interval or getattr(config, 'SCAN_INTERVAL', 60)
        
        # ADDED: Debug configuration detection
        self.logger.info("🔧 TradingSystem Configuration Detection:")
        self.logger.info(f"   Intelligence preset: {self.intelligence_preset}")
        self.logger.info(f"   Intelligence mode: {self.intelligence_mode}")
        self.logger.info(f"   enable_trading parameter: {enable_trading}")
        self.logger.info(f"   enable_claude_analysis parameter: {enable_claude_analysis}")
        self.logger.info(f"   AUTO_TRADING_ENABLED config: {getattr(config, 'AUTO_TRADING_ENABLED', 'NOT_SET')}")
        self.logger.info(f"   INTELLIGENCE_PRESET config: {getattr(config, 'INTELLIGENCE_PRESET', 'NOT_SET')}")
        self.logger.info(f"   INTELLIGENCE_MODE config: {getattr(config, 'INTELLIGENCE_MODE', 'NOT_SET')}")
        self.logger.info(f"   CLAUDE_API_KEY config: {'SET' if getattr(config, 'CLAUDE_API_KEY', None) else 'NOT_SET'}")
        self.logger.info(f"   Final enable_trading: {self.enable_trading}")
        self.logger.info(f"   Final enable_claude_analysis: {self.enable_claude_analysis}")

        # Initialize database manager
        try:
            self.db_manager = DatabaseManager(config.DATABASE_URL)
            self.logger.info("✅ Database manager initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database manager: {e}")
            self.logger.warning("⚠️ Continuing without database - functionality will be limited")
            self.db_manager = None

        # FIXED: Create orchestrator with correct parameters (removed duplicate creation)
        try:
            self.orchestrator = create_trading_orchestrator(
                db_manager=self.db_manager,
                intelligence_mode=self.intelligence_mode,  # Legacy compatibility
                intelligence_preset=self.intelligence_preset,  # NEW: Preset system
                enable_trading=self.enable_trading,
                enable_claude_analysis=self.enable_claude_analysis,
                scan_interval=self.scan_interval
            )
            self.logger.info("✅ TradingSystem initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize TradingSystem: {e}")
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
        
        # Log initialization summary
        self._log_system_initialization()
    
    def _map_preset_to_mode(self, preset: str) -> str:
        """Map new intelligence preset to legacy mode for compatibility"""
        preset_to_mode_mapping = {
            'disabled': 'disabled',
            'minimal': 'live_only',
            'balanced': 'balanced',
            'conservative': 'enhanced',
            'testing': 'backtest_consistent'
        }
        return preset_to_mode_mapping.get(preset, 'live_only')
    
    def _map_mode_to_preset(self, mode: str) -> str:
        """Map legacy intelligence mode to new preset for compatibility"""
        mode_to_preset_mapping = {
            'disabled': 'disabled',
            'backtest_consistent': 'testing',
            'live_only': 'minimal',
            'balanced': 'balanced',
            'enhanced': 'conservative'
        }
        return mode_to_preset_mapping.get(mode, 'minimal')
    
    def _setup_logging(self):
        """Fixed logging configuration - no more duplicates"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(current_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Get root logger
        root_logger = logging.getLogger()
        
        # 🔧 CRITICAL FIX: Check if already configured
        if root_logger.handlers:
            # Already configured - don't add more handlers
            return
        
        # Configure logging parameters
        log_level = getattr(config, 'LOG_LEVEL', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler with rotation
        log_file = os.path.join(log_dir, 'trade_scan.log')
        file_handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        console_handler.setLevel(logging.INFO)
        
        # Set root logger level
        root_logger.setLevel(logging.INFO)
        
        # Add handlers ONCE
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Prevent propagation issues
        root_logger.propagate = False
    
    def _log_system_initialization(self):
        """Log system initialization details"""
        self.logger.info("🚀 Trading System Initialization Complete")
        self.logger.info(f"   Current directory: {current_dir}")
        self.logger.info(f"   Intelligence preset: {self.intelligence_preset}")
        self.logger.info(f"   Intelligence mode (legacy): {self.intelligence_mode}")
        self.logger.info(f"   Trading enabled: {self.enable_trading}")
        self.logger.info(f"   Claude analysis: {self.enable_claude_analysis}")
        self.logger.info(f"   Scan interval: {self.scan_interval}s")
        self.logger.info(f"   Database available: {'Yes' if self.db_manager else 'No'}")
        self.logger.info(f"   Configuration pairs: {len(getattr(config, 'EPIC_LIST', []))}")
    
    def scan_once(self) -> List[Dict]:
        """
        Perform a single scan across all configured pairs
        
        Returns:
            List of detected signals
        """
        try:
            self.logger.info("🔍 Starting single scan...")
            signals = self.orchestrator.scan_once()
            self.logger.info(f"✅ Single scan complete - found {len(signals)} signals")
            return signals
        except Exception as e:
            self.logger.error(f"❌ Error in scan_once: {e}")
            return []
    
    def run_live_trading(self, scan_interval: Optional[int] = None):
        """
        Start continuous live trading
        
        Args:
            scan_interval: Override scan interval in seconds
        """
        try:
            effective_interval = scan_interval or self.scan_interval
            self.logger.info(f"🔄 Starting live trading mode (interval: {effective_interval}s)")
            # Update scan interval if provided
            if scan_interval:
                self.orchestrator.update_scan_interval(scan_interval)
            return self.orchestrator.start_continuous_scan()
        except Exception as e:
            self.logger.error(f"❌ Error in live trading: {e}")
            raise
    
    def test_claude_minimal(self) -> bool:
        """
        Test Claude AI integration
        
        Returns:
            True if Claude integration is working, False otherwise
        """
        try:
            self.logger.info("🤖 Testing Claude integration...")
            result = self.orchestrator.test_claude_integration()
            self.logger.info(f"🤖 Claude test result: {'✅ PASSED' if result else '❌ FAILED'}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error testing Claude: {e}")
            return False
    
    def start_docker_mode(self):
        """
        Start Docker deployment mode with continuous operation
        
        This is the main mode used when the container starts with no arguments.
        Includes heartbeat logging and graceful shutdown handling.
        """
        try:
            self.logger.info("🐳 Starting Docker deployment mode...")
            self._log_docker_environment()
            return self.orchestrator.start_docker_mode()
        except Exception as e:
            self.logger.error(f"❌ Error in Docker mode: {e}")
            raise
    
    def run_scheduled_trading(self):
        """
        Run trading on a predefined schedule
        
        This mode allows for specific trading windows and schedules.
        """
        try:
            self.logger.info("📅 Starting scheduled trading mode...")
            
            import schedule
            
            # Schedule scans during market hours
            schedule.every().hour.at(":00").do(self._scheduled_scan)
            schedule.every().hour.at(":30").do(self._scheduled_scan)
            
            self.logger.info("📅 Schedule configured - scanning every 30 minutes")
            
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Scheduled trading stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Error in scheduled trading: {e}")
            raise
    
    def _scheduled_scan(self):
        """Perform a scheduled scan"""
        try:
            self.logger.info("⏰ Performing scheduled scan...")
            signals = self.scan_once()
            self.logger.info(f"⏰ Scheduled scan complete - found {len(signals)} signals")
        except Exception as e:
            self.logger.error(f"❌ Error in scheduled scan: {e}")
    
    def get_system_status(self) -> Dict:
        """
        Get comprehensive system status
        
        Returns:
            Dictionary containing system status information
        """
        try:
            # Get orchestrator status
            orchestrator_status = self.orchestrator.get_comprehensive_status()
            
            # Add trading system specific information
            system_status = {
                'trading_system': {
                    'version': '2.0.0-refactored',
                    'entry_point': 'trade_scan.py',
                    'mode': 'modular_orchestrator',
                    'initialization_time': datetime.now().isoformat()
                },
                'configuration': {
                    'intelligence_preset': self.intelligence_preset,
                    'intelligence_mode': self.intelligence_mode,
                    'trading_enabled': self.enable_trading,
                    'claude_enabled': self.enable_claude_analysis,
                    'scan_interval': self.scan_interval
                },
                'orchestrator': orchestrator_status
            }
            
            return system_status
            
        except Exception as e:
            self.logger.error(f"❌ Error getting system status: {e}")
            return {'error': str(e)}
    
    def _log_docker_environment(self):
        """Log Docker environment information"""
        self.logger.info("🐳 Docker Environment Information:")
        
        # Log environment variables
        docker_vars = ['HOSTNAME', 'PATH', 'PWD', 'HOME']
        for var in docker_vars:
            value = os.environ.get(var, 'Not set')
            self.logger.info(f"   {var}: {value}")
        
        # Log Python version and path
        self.logger.info(f"   Python version: {sys.version}")
        self.logger.info(f"   Python executable: {sys.executable}")
        
        # Log disk space
        try:
            import shutil
            disk_usage = shutil.disk_usage('/')
            total_gb = disk_usage.total / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            self.logger.info(f"   Disk space: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        except:
            self.logger.info("   Disk space: Unable to determine")
    
    def stop(self):
        """
        Gracefully stop the trading system
        """
        self.logger.info("🛑 Stopping trading system...")
        try:
            self.orchestrator.stop()
            self.logger.info("✅ Trading system stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Error stopping trading system: {e}")


def print_usage():
    """Print usage information"""
    print("🔧 Forex Trading Scanner - Docker Entry Point")
    print("")
    print("Usage:")
    print("  python trade_scan.py [command] [options]")
    print("")
    print("Commands:")
    print("  scan                    - Run single scan")
    print("  live [interval]         - Run continuous live trading")
    print("  test-claude             - Test Claude integration")
    print("  scheduled               - Run on predefined schedule")
    print("  docker                  - Run in Docker mode (default)")
    print("  status                  - Show system status")
    print("  help                    - Show this help message")
    print("")
    print("Examples:")
    print("  python trade_scan.py scan")
    print("  python trade_scan.py live 120")
    print("  python trade_scan.py test-claude")
    print("  python trade_scan.py docker")
    print("")
    print("Docker Usage:")
    print("  CMD [\"python\", \"trade_scan.py\"]              # Default Docker mode")
    print("  CMD [\"python\", \"trade_scan.py\", \"live\"]       # Live trading mode")
    print("  CMD [\"python\", \"trade_scan.py\", \"scheduled\"]  # Scheduled mode")
    print("")
    print("Environment Variables:")
    print("  INTELLIGENCE_PRESET     - Set intelligence preset (disabled, minimal, balanced)")
    print("  ENABLE_TRADING          - Enable/disable trading (true/false)")
    print("  ENABLE_CLAUDE           - Enable/disable Claude analysis (true/false)")


def main():
    """
    Main entry point for Docker and command line usage
    
    Maintains exact compatibility with original trade_scan.py interface
    while using the new modular architecture internally.
    
    FIXED: Now respects config.py settings by default and supports new intelligence system
    """
    # Setup basic logging for startup

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # MODIFY: Update console to use Stockholm timezone
    console_handler = logging.getLogger().handlers[0]  # Get the console handler from basicConfig
    console_handler.setFormatter(StockholmFormatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger(__name__)
    os.makedirs("logs", exist_ok=True)

    # NEW: Cleanup old log files before setting up new handler
    cleanup_old_log_files("logs", "forex_scanner", days_to_keep=7)

    # FIXED: Use TimedRotatingFileHandler with proper cleanup
    # This will create files like: forex_scanner.log, forex_scanner.log.2025-09-24, etc.
    file_handler = TimedRotatingFileHandler(
        "logs/forex_scanner.log",
        when='midnight',
        interval=1,
        backupCount=7,  # Keep 7 days of logs
        encoding='utf-8'
    )

    # Ensure the handler deletes old files properly
    file_handler.namer = lambda name: name.replace('.log', '') + '.log'

    # MODIFY: Set formatter with Stockholm timezone
    file_handler.setFormatter(StockholmFormatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add the handler
    logging.getLogger().addHandler(file_handler)

    logger.info(f"🗂️ Log rotation configured: keeping {file_handler.backupCount} days of logs")
    
    # Initialize trading system
    try:
        # FIXED: Use new intelligence preset system
        intelligence_preset = os.environ.get('INTELLIGENCE_PRESET', None)  # Don't override config by default
        intelligence_mode = os.environ.get('INTELLIGENCE_MODE', None)  # Legacy support
        
        # FIXED: Use None to let TradingSystem use config.py, only override if env vars are set
        enable_trading = None
        enable_claude = None
        
        if 'ENABLE_TRADING' in os.environ:
            enable_trading = os.environ.get('ENABLE_TRADING', '').lower() in ('true', '1', 'yes')
            logger.info(f"🔧 ENABLE_TRADING env var found: {enable_trading}")
        
        if 'ENABLE_CLAUDE' in os.environ:
            enable_claude = os.environ.get('ENABLE_CLAUDE', '').lower() in ('true', '1', 'yes')
            logger.info(f"🔧 ENABLE_CLAUDE env var found: {enable_claude}")
        
        # FIXED: Pass intelligence preset and better parameter handling
        trading_system = TradingSystem(
            intelligence_mode=intelligence_mode,  # Legacy compatibility
            intelligence_preset=intelligence_preset,  # NEW: Preset system
            enable_trading=enable_trading,  # None = use config.py
            enable_claude_analysis=enable_claude  # None = use config.py
        )
        print("✅ Trading system initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize trading system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        try:
            if command == 'scan':
                # Single scan mode
                print("🔍 Running single scan...")
                signals = trading_system.scan_once()
                print(f"✅ Scan complete - found {len(signals)} signals")
                
                # Print signal summary
                if signals:
                    print("\n📊 Signal Summary:")
                    for i, signal in enumerate(signals, 1):
                        epic = signal.get('epic', 'Unknown')
                        signal_type = signal.get('signal_type', 'Unknown')
                        confidence = signal.get('confidence_score', signal.get('confidence', 0))
                        if isinstance(confidence, (int, float)) and confidence > 1:
                            confidence = confidence / 100  # Convert percentage to decimal
                        print(f"  {i}. {epic} {signal_type} ({confidence:.1%})")
                else:
                    print("ℹ️ No signals detected. Check your configuration:")
                    print(f"   Intelligence preset: {trading_system.intelligence_preset}")
                    print(f"   Intelligence mode: {trading_system.intelligence_mode}")
                    print("   Try setting INTELLIGENCE_PRESET='disabled' in config.py for maximum signals")
                
            elif command == 'test-claude':
                # Test Claude functionality
                print("🤖 Testing Claude integration...")
                success = trading_system.test_claude_minimal()
                print(f"Claude test: {'✅ PASSED' if success else '❌ FAILED'}")
                
            elif command == 'live':
                # Live trading mode
                scan_interval = int(sys.argv[2]) if len(sys.argv) > 2 else None
                print(f"🔄 Starting live trading mode...")
                trading_system.run_live_trading(scan_interval)
                
            elif command == 'scheduled':
                # Scheduled trading mode
                print("📅 Starting scheduled trading mode...")
                trading_system.run_scheduled_trading()
                
            elif command == 'docker':
                # Docker deployment mode
                print("🐳 Starting Docker deployment mode...")
                trading_system.start_docker_mode()
                
            elif command == 'status':
                # Show system status
                print("📊 Retrieving system status...")
                status = trading_system.get_system_status()
                
                print("\n📊 System Status:")
                print(json.dumps(status, indent=2, default=str))
            
            elif command == 'scan-backtest':
                # MARKET CLOSED TESTING: Use recent historical data for scanning
                print("📊 Running market-closed scan with recent historical data...")
                days = int(sys.argv[2]) if len(sys.argv) > 2 else 2
                
                # Override settings for market-closed testing
                trading_system.orchestrator.force_historical_scan = True
                trading_system.orchestrator.historical_days = days
                
                signals = trading_system.scan_once()
                print(f"✅ Historical scan complete - found {len(signals)} signals over {days} days")
                
                if signals:
                    print("\n📊 Historical Signal Summary:")
                    for i, signal in enumerate(signals, 1):
                        epic = signal.get('epic', 'Unknown')
                        signal_type = signal.get('signal_type', 'Unknown')
                        confidence = signal.get('confidence_score', 0)
                        timestamp = signal.get('timestamp', 'Unknown')
                        print(f"  {i}. {epic} {signal_type} ({confidence:.1%}) at {timestamp}")
                
            elif command == 'intelligence-test':
                # MARKET INTELLIGENCE TESTING: Test intelligence with different presets
                print("🧠 Testing Market Intelligence System...")
                
                presets_to_test = ['disabled', 'minimal', 'balanced', 'conservative']
                results = {}
                
                for preset in presets_to_test:
                    print(f"\n🧪 Testing preset: {preset}")
                    
                    # Create new trading system with this preset
                    test_system = TradingSystem(
                        intelligence_preset=preset,
                        enable_trading=trading_system.enable_trading,
                        enable_claude_analysis=trading_system.enable_claude_analysis
                    )
                    
                    # Force use recent historical data
                    test_system.orchestrator.force_historical_scan = True
                    test_system.orchestrator.historical_days = 2
                    
                    signals = test_system.scan_once()
                    results[preset] = len(signals)
                    print(f"   {preset}: {len(signals)} signals")
                
                print(f"\n🧠 Intelligence Test Results:")
                for preset, count in results.items():
                    print(f"   {preset:12}: {count:3d} signals")
                
            elif command == 'market-conditions':
                # MARKET CONDITIONS TESTING: Show current market analysis
                print("🌍 Analyzing Current Market Conditions...")
                
                try:
                    # Get market conditions from orchestrator
                    conditions = trading_system.orchestrator.get_market_conditions()
                    
                    print("\n🌍 Current Market Analysis:")
                    print(f"   Volatility Regime: {conditions.get('volatility_regime', 'unknown')}")
                    print(f"   Trend Strength: {conditions.get('trend_strength', 'unknown')}")
                    print(f"   Market Regime: {conditions.get('market_regime', 'unknown')}")
                    print(f"   Trading Session: {conditions.get('current_session', 'unknown')}")
                    print(f"   Market Hours: {conditions.get('market_hours', 'unknown')}")
                    
                    # Test intelligence decisions
                    print(f"\n🧠 Intelligence Analysis:")
                    intelligence_score = trading_system.orchestrator.get_intelligence_score()
                    print(f"   Intelligence Score: {intelligence_score:.1%}")
                    print(f"   Recommended Action: {trading_system.orchestrator.get_intelligence_recommendation()}")
                    
                except Exception as e:
                    print(f"❌ Error getting market conditions: {e}")

            elif command == 'debug-data':
                # DEBUG: Compare live scanner data vs backtest data
                print("🔍 Debugging Live Scanner Data Access...")
                
                epic = sys.argv[2] if len(sys.argv) > 2 else 'CS.D.EURUSD.MINI.IP'
                
                print(f"\n📊 BACKTEST DATA (what works):")
                print("=" * 50)
                
                # Run backtest to see what data it uses
                try:
                    from commands.backtest_commands import BacktestCommands
                    backtest_cmd = BacktestCommands()
                    print("Running 2-day backtest...")
                    backtest_cmd.run_backtest(epic=epic, days=2, timeframe=getattr(config, 'DEFAULT_TIMEFRAME', '15m'))
                except Exception as e:
                    print(f"Backtest error: {e}")
                
                print(f"\n📊 LIVE SCANNER DATA (what fails):")
                print("=" * 50)
                
                # Debug live scanner data access
                try:
                    # Get the orchestrator's scanner
                    scanner = trading_system.orchestrator.scanner
                    
                    print(f"Scanner type: {type(scanner).__name__}")
                    print(f"Epic list: {scanner.epic_list}")
                    print(f"Min confidence: {scanner.min_confidence}")
                    print(f"Intelligence mode: {getattr(scanner, 'intelligence_mode', 'unknown')}")
                    
                    # Try to get data directly
                    if hasattr(scanner, 'signal_detector'):
                        signal_detector = scanner.signal_detector
                        print(f"Signal detector: {type(signal_detector).__name__}")
                        
                        # Get data the same way the scanner does
                        if hasattr(signal_detector, 'data_fetcher'):
                            data_fetcher = signal_detector.data_fetcher
                            print(f"Data fetcher: {type(data_fetcher).__name__}")
                            
                            # Fetch data for the epic
                            print(f"\n🔍 Fetching data for {epic}...")
                            df = data_fetcher.get_enhanced_data(epic, epic.replace('CS.D.', '').replace('.MINI.IP', ''))
                            
                            if df is not None and len(df) > 0:
                                print(f"✅ Data fetched: {len(df)} rows")
                                print(f"   Date range: {df.index[0]} to {df.index[-1]}")
                                print(f"   Latest price: {df['close'].iloc[-1]:.5f}")
                                
                                # Check for required indicators
                                required_cols = ['ema_9', 'ema_21', 'ema_200', 'macd', 'macd_signal']
                                missing_cols = [col for col in required_cols if col not in df.columns]
                                
                                if missing_cols:
                                    print(f"❌ Missing indicators: {missing_cols}")
                                else:
                                    print(f"✅ All indicators present")
                                    
                                    # Check latest indicator values
                                    latest = df.iloc[-1]
                                    print(f"   EMA 9: {latest.get('ema_9', 'N/A')}")
                                    print(f"   EMA 21: {latest.get('ema_21', 'N/A')}")
                                    print(f"   EMA 200: {latest.get('ema_200', 'N/A')}")
                                    
                                    # Check for NaN values
                                    nan_count = df[['ema_9', 'ema_21', 'ema_200']].isna().sum().sum()
                                    if nan_count > 0:
                                        print(f"❌ Found {nan_count} NaN values in EMAs")
                                    else:
                                        print(f"✅ No NaN values in EMAs")
                                    
                                    # Test strategy detection manually
                                    print(f"\n🧪 Testing strategy detection on this data...")
                                    try:
                                        # Test combined strategy
                                        signal = signal_detector.detect_combined_signals(epic, epic.replace('CS.D.', '').replace('.MINI.IP', ''))
                                        if signal:
                                            print(f"✅ Strategy detected signal: {signal['signal_type']} at {signal['confidence_score']:.1%}")
                                        else:
                                            print(f"❌ Strategy detected no signal")
                                            
                                            # Check if it's a crossover timing issue
                                            if len(df) >= 2:
                                                current_ema9 = df['ema_9'].iloc[-1]
                                                current_ema21 = df['ema_21'].iloc[-1]
                                                prev_ema9 = df['ema_9'].iloc[-2]
                                                prev_ema21 = df['ema_21'].iloc[-2]
                                                
                                                print(f"   Current EMA crossover state:")
                                                print(f"     EMA9 > EMA21: {current_ema9 > current_ema21}")
                                                print(f"     Previous EMA9 > EMA21: {prev_ema9 > prev_ema21}")
                                                
                                                if (current_ema9 > current_ema21) != (prev_ema9 > prev_ema21):
                                                    print(f"   🎯 CROSSOVER DETECTED but not triggering signal!")
                                                else:
                                                    print(f"   📊 No crossover in latest data")
                                    
                                    except Exception as strategy_error:
                                        print(f"❌ Strategy test error: {strategy_error}")
                            else:
                                print(f"❌ No data returned for {epic}")
                        else:
                            print(f"❌ No data_fetcher found")
                    else:
                        print(f"❌ No signal_detector found")
                        
                except Exception as e:
                    print(f"❌ Live scanner debug error: {e}")
                    import traceback
                    traceback.print_exc()

            elif command == 'force-scan':
                # FORCED SCANNING: Override all market conditions and intelligence
                print("🔧 Running FORCED scan (ignoring market conditions)...")
                
                # Override intelligence and market checks
                trading_system.orchestrator.force_ignore_market_hours = True
                trading_system.orchestrator.force_ignore_intelligence = True
                trading_system.orchestrator.force_historical_scan = True
                trading_system.orchestrator.historical_days = 1
                
                signals = trading_system.scan_once()
                print(f"✅ Forced scan complete - found {len(signals)} signals")
                
                if signals:
                    print("\n📊 Forced Scan Results:")
                    for i, signal in enumerate(signals, 1):
                        epic = signal.get('epic', 'Unknown')
                        signal_type = signal.get('signal_type', 'Unknown')
                        confidence = signal.get('confidence_score', 0)
                        strategy = signal.get('strategy', 'Unknown')
                        print(f"  {i}. {epic} {signal_type} ({confidence:.1%}) - {strategy}")
                else:
                    print("⚠️ No signals found even with forced conditions")
                    print("   This indicates a deeper issue with strategy logic or data")

                
            elif command in ('help', '--help', '-h'):
                # Show help
                print_usage()
                
            else:
                print(f"❌ Unknown command: {command}")
                print("\nRun 'python trade_scan.py help' for usage information")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Operation interrupted by user")
            trading_system.stop()
            
        except Exception as e:
            logger.error(f"❌ Command '{command}' failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    else:
        # Default: Docker mode for container deployment
        print("🐳 No command specified - starting default Docker mode...")
        try:
            trading_system.start_docker_mode()
        except KeyboardInterrupt:
            print("\n🛑 Docker mode interrupted by user")
            trading_system.stop()
        except Exception as e:
            logger.error(f"❌ Docker mode failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()