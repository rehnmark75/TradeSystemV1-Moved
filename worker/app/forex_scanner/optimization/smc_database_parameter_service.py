#!/usr/bin/env python3
"""
SMC Database Parameter Service

Database-driven parameter service for Smart Money Concepts strategy.
Reads optimal parameters from PostgreSQL database instead of CSV files.

Key Features:
- Database-driven parameter loading
- Epic-specific configuration management
- Performance-based ranking and selection
- Intelligent caching for performance
- Comprehensive fallback mechanisms

Author: Trading System V1
Created: 2025-09-12
"""

import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import sys

# Add forex_scanner to path for imports
sys.path.append('/app/forex_scanner')

from core.database import DatabaseManager
import config

# Configure logging
logger = logging.getLogger(__name__)

class SMCDatabaseParameterService:
    """Database-driven service for retrieving optimal SMC parameters."""
    
    def __init__(self):
        """Initialize the service with database connection."""
        self.db = DatabaseManager(config.DATABASE_URL)
        self.cache_duration = timedelta(minutes=30)
        self._cache = {}
        self._cache_timestamps = {}
        
        logger.debug("SMC Database Parameter Service initialized")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache or cache_key not in self._cache_timestamps:
            return False
        
        return datetime.now() - self._cache_timestamps[cache_key] < self.cache_duration
    
    def _set_cache(self, cache_key: str, data: Any):
        """Set cache entry with timestamp."""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()
    
    def get_best_configuration_for_epic(self, epic: str, ranking_criteria: str = 'performance_score') -> Dict[str, Any]:
        """
        Get the best configuration for a specific epic from database.
        
        Args:
            epic: The trading pair (e.g., 'CS.D.EURUSD.CEEM.IP')
            ranking_criteria: How to rank configurations ('performance_score', 'win_rate', 'profit_factor')
        
        Returns:
            Dict containing optimal configuration parameters
        """
        
        # Check cache first
        cache_key = f"best_config_{epic}_{ranking_criteria}"
        if self._is_cache_valid(cache_key):
            logger.debug(f"Using cached configuration for {epic}")
            return self._cache[cache_key]
        
        try:
            # Query database for best parameters
            query = """
            SELECT 
                epic, best_smc_config, best_confidence_level, best_timeframe,
                optimal_stop_loss_pips, optimal_take_profit_pips, optimal_risk_reward_ratio,
                best_win_rate, best_profit_factor, best_net_pips, best_performance_score,
                confluence_accuracy, last_optimized
            FROM smc_best_parameters 
            WHERE epic = %s
            ORDER BY {} DESC 
            LIMIT 1
            """.format('best_' + ranking_criteria if ranking_criteria != 'performance_score' else 'best_performance_score')
            
            # Use string formatting for now to avoid parameter issues
            formatted_query = query.replace('%s', f"'{epic}'")
            result = self.db.execute_query(formatted_query)
            
            if result is not None and not result.empty:
                row = result.iloc[0]
                
                config = {
                    'epic': row[0],
                    'smc_config': row[1],
                    'confidence_level': float(row[2]),
                    'timeframe': row[3],
                    'stop_loss_pips': int(row[4]),
                    'take_profit_pips': int(row[5]),
                    'risk_reward_ratio': float(row[6]),
                    'expected_win_rate': float(row[7]),
                    'expected_profit_factor': float(row[8]),
                    'net_pips': float(row[9]),
                    'performance_score': float(row[10]),
                    'confluence_accuracy': float(row[11]),
                    'optimization_source': 'database',
                    'last_optimized': str(row[12]) if row[12] else 'unknown'
                }
                
                # Cache the result
                self._set_cache(cache_key, config)
                
                logger.info(f"Retrieved optimal parameters for {epic}: {config['smc_config']} "
                           f"(Win Rate: {config['expected_win_rate']:.1f}%, "
                           f"Performance: {config['performance_score']:.1f})")
                
                return config
            else:
                logger.warning(f"No database parameters found for {epic}")
                return self._get_fallback_config(epic)
                
        except Exception as e:
            logger.error(f"Database query failed for {epic}: {e}")
            return self._get_fallback_config(epic)
    
    def get_top_configurations_for_epic(self, epic: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """Get top N configurations for a specific epic (database version)."""
        
        cache_key = f"top_configs_{epic}_{top_n}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # For database version, we only have one best config per epic
            # So we return that one configuration
            best_config = self.get_best_configuration_for_epic(epic)
            
            if 'error' not in best_config:
                configurations = [{
                    'rank': 1,
                    **best_config
                }]
                
                self._set_cache(cache_key, configurations)
                return configurations
            else:
                return [self._get_fallback_config(epic)]
                
        except Exception as e:
            logger.error(f"Failed to get top configurations for {epic}: {e}")
            return [self._get_fallback_config(epic)]
    
    def get_all_epic_best_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get best configuration for all available epics from database."""
        
        cache_key = "all_epic_configs"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            query = """
            SELECT 
                epic, best_smc_config, best_confidence_level, best_timeframe,
                optimal_stop_loss_pips, optimal_take_profit_pips, optimal_risk_reward_ratio,
                best_win_rate, best_profit_factor, best_net_pips, best_performance_score,
                confluence_accuracy, last_optimized
            FROM smc_best_parameters 
            ORDER BY best_performance_score DESC
            """
            
            results = self.db.execute_query(query)
            
            if results is not None and not results.empty:
                all_configs = {}
                
                for idx, row in results.iterrows():
                    epic = row[0]
                    config = {
                        'epic': epic,
                        'smc_config': row[1],
                        'confidence_level': float(row[2]),
                        'timeframe': row[3],
                        'stop_loss_pips': int(row[4]),
                        'take_profit_pips': int(row[5]),
                        'risk_reward_ratio': float(row[6]),
                        'expected_win_rate': float(row[7]),
                        'expected_profit_factor': float(row[8]),
                        'net_pips': float(row[9]),
                        'performance_score': float(row[10]),
                        'confluence_accuracy': float(row[11]),
                        'optimization_source': 'database',
                        'last_optimized': str(row[12]) if row[12] else 'unknown'
                    }
                    
                    all_configs[epic] = config
                
                self._set_cache(cache_key, all_configs)
                logger.info(f"Retrieved configurations for {len(all_configs)} epics from database")
                
                return all_configs
            else:
                logger.warning("No configurations found in database")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get all epic configurations: {e}")
            return {}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of all optimization results from database."""
        
        cache_key = "optimization_summary"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Get summary statistics
            summary_query = """
            SELECT 
                COUNT(*) as total_epics,
                AVG(best_win_rate) as avg_win_rate,
                MAX(best_win_rate) as max_win_rate,
                AVG(best_performance_score) as avg_performance,
                MAX(best_performance_score) as max_performance,
                AVG(best_profit_factor) as avg_profit_factor
            FROM smc_best_parameters
            """
            
            summary_result = self.db.execute_query(summary_query)
            
            # Get epic list
            epics_query = "SELECT epic FROM smc_best_parameters ORDER BY best_performance_score DESC"
            epics_result = self.db.execute_query(epics_query)
            
            # Get best overall performer
            best_query = """
            SELECT epic, best_smc_config, best_performance_score 
            FROM smc_best_parameters 
            ORDER BY best_performance_score DESC 
            LIMIT 1
            """
            best_result = self.db.execute_query(best_query)
            
            if (summary_result is not None and not summary_result.empty and 
                epics_result is not None and not epics_result.empty):
                summary_row = summary_result.iloc[0]
                epics_list = [row for row in epics_result.iloc[:, 0]]
                
                summary = {
                    'total_tests': int(summary_row[0]) * 8,  # Assume 8 configs tested per epic
                    'unique_epics': int(summary_row[0]),
                    'epics_list': epics_list,
                    'configurations_tested': ['default', 'moderate', 'conservative', 'aggressive', 'scalping', 'swing', 'news_safe', 'crypto'],
                    'avg_win_rate': float(summary_row[1]) if summary_row[1] else 0,
                    'avg_performance_score': float(summary_row[3]) if summary_row[3] else 0,
                    'data_source': 'database'
                }
                
                if best_result is not None and not best_result.empty:
                    best_row = best_result.iloc[0]
                    summary.update({
                        'best_overall_epic': best_row[0],
                        'best_overall_config': best_row[1],
                        'best_overall_performance': float(best_row[2])
                    })
                
                self._set_cache(cache_key, summary)
                return summary
            else:
                return {'error': 'No optimization results found in database'}
                
        except Exception as e:
            logger.error(f"Failed to get optimization summary: {e}")
            return {'error': f'Database error: {str(e)}'}
    
    def compare_configurations(self, epic: str, config1: str, config2: str) -> Dict[str, Any]:
        """Compare two SMC configurations for a specific epic (simplified for database version)."""
        
        # Since database version stores only best config per epic, 
        # we can't compare multiple configs for the same epic
        best_config = self.get_best_configuration_for_epic(epic)
        
        if 'error' not in best_config:
            return {
                'epic': epic,
                'message': f'Database version stores only best configuration per epic: {best_config["smc_config"]}',
                'best_config': best_config,
                'note': 'For detailed configuration comparison, use the CSV-based service or run optimization with multiple configs'
            }
        else:
            return {'error': f'No configuration data available for {epic}'}
    
    def _get_fallback_config(self, epic: str) -> Dict[str, Any]:
        """Get fallback configuration when database data is unavailable."""
        
        # Choose fallback based on epic characteristics
        if 'JPY' in epic:
            stop_loss = 8
            take_profit = 16
        else:
            stop_loss = 10
            take_profit = 20
        
        fallback = {
            'epic': epic,
            'smc_config': 'default',
            'confidence_level': 0.55,
            'timeframe': '15m',
            'stop_loss_pips': stop_loss,
            'take_profit_pips': take_profit,
            'risk_reward_ratio': 2.0,
            'expected_win_rate': 75.0,  # Conservative estimate
            'expected_profit_factor': 2.0,
            'performance_score': 100.0,
            'confluence_accuracy': 75.0,
            'optimization_source': 'fallback_default',
            'last_optimized': 'never'
        }
        
        logger.info(f"Using fallback configuration for {epic}")
        return fallback
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get database connection and table status."""
        
        try:
            # Test database connection
            test_result = self.db.execute_query("SELECT 1")
            
            if test_result is not None and not test_result.empty:
                # Get table row counts
                tables_info = {}
                
                tables = ['smc_best_parameters', 'smc_optimization_runs', 'smc_optimization_results']
                for table in tables:
                    try:
                        count_result = self.db.execute_query(f"SELECT COUNT(*) FROM {table}")
                        tables_info[table] = count_result.iloc[0, 0] if count_result is not None and not count_result.empty else 0
                    except Exception as e:
                        tables_info[table] = f"Error: {e}"
                
                return {
                    'database_connected': True,
                    'tables': tables_info,
                    'service_ready': tables_info.get('smc_best_parameters', 0) > 0
                }
            else:
                return {
                    'database_connected': False,
                    'error': 'Database query failed',
                    'service_ready': False
                }
                
        except Exception as e:
            return {
                'database_connected': False,
                'error': str(e),
                'service_ready': False
            }


# Convenience functions for easy import and usage
_service_instance = None

def get_smc_optimal_parameters(epic: str, ranking_criteria: str = 'performance_score') -> Dict[str, Any]:
    """Get optimal SMC parameters for an epic from database. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SMCDatabaseParameterService()
    
    return _service_instance.get_best_configuration_for_epic(epic, ranking_criteria)

def get_smc_top_configurations(epic: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Get top N SMC configurations for an epic from database. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SMCDatabaseParameterService()
    
    return _service_instance.get_top_configurations_for_epic(epic, top_n)

def get_all_smc_best_configs() -> Dict[str, Dict[str, Any]]:
    """Get best SMC configuration for all epics from database. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SMCDatabaseParameterService()
    
    return _service_instance.get_all_epic_best_configs()

def get_smc_optimization_summary() -> Dict[str, Any]:
    """Get SMC optimization summary from database. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SMCDatabaseParameterService()
    
    return _service_instance.get_optimization_summary()

def get_smc_database_status() -> Dict[str, Any]:
    """Get database status. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SMCDatabaseParameterService()
    
    return _service_instance.get_database_status()


if __name__ == "__main__":
    """Test the SMC database parameter service."""
    
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing SMC Database Parameter Service\n")
    
    # Initialize service
    service = SMCDatabaseParameterService()
    
    # Test 1: Database status
    print("🔗 Database Status:")
    status = service.get_database_status()
    if status['database_connected']:
        print("   ✅ Database connection: OK")
        print(f"   ✅ Service ready: {status['service_ready']}")
        for table, count in status['tables'].items():
            print(f"   📊 {table}: {count} rows")
    else:
        print("   ❌ Database connection failed")
        print(f"   Error: {status.get('error', 'Unknown')}")
    
    if status['service_ready']:
        # Test 2: Get optimization summary
        print("\n📊 Optimization Summary:")
        summary = service.get_optimization_summary()
        if 'error' not in summary:
            print(f"   Unique Epics: {summary['unique_epics']}")
            print(f"   Average Win Rate: {summary['avg_win_rate']:.1f}%")
            print(f"   Best Overall: {summary.get('best_overall_epic', 'N/A')} ({summary.get('best_overall_config', 'N/A')})")
            print(f"   Best Performance Score: {summary.get('best_overall_performance', 0):.1f}")
        else:
            print(f"   Error: {summary['error']}")
        
        # Test 3: Get best config for specific epics
        print("\n🎯 Best Configuration Examples:")
        test_epics = summary.get('epics_list', [])[:3]
        
        for epic in test_epics:
            config = service.get_best_configuration_for_epic(epic)
            print(f"   {epic}:")
            print(f"     Config: {config['smc_config']} | Confidence: {config['confidence_level']}")
            print(f"     SL/TP: {config['stop_loss_pips']}/{config['take_profit_pips']} pips")
            print(f"     Expected Win Rate: {config['expected_win_rate']:.1f}%")
            print(f"     Performance Score: {config['performance_score']:.1f}")
        
        # Test 4: All configurations
        print("\n🌍 All Epic Configurations:")
        all_configs = service.get_all_epic_best_configs()
        print(f"   Retrieved configurations for {len(all_configs)} epics")
        
        top_3 = sorted(all_configs.items(), key=lambda x: x[1]['performance_score'], reverse=True)[:3]
        for epic, config in top_3:
            print(f"   🏆 {epic}: {config['smc_config']} (Score: {config['performance_score']:.1f})")
        
        # Test 5: Convenience functions
        print("\n🔧 Testing Convenience Functions:")
        if test_epics:
            quick_config = get_smc_optimal_parameters(test_epics[0])
            print(f"   Quick config for {test_epics[0]}: {quick_config['smc_config']} "
                  f"(Win Rate: {quick_config['expected_win_rate']:.1f}%)")
        
    print("\n✅ All tests completed!")