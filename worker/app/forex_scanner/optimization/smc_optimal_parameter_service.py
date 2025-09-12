#!/usr/bin/env python3
"""
SMC Optimal Parameter Service

Provides intelligent parameter loading for Smart Money Concepts strategy based on optimization results.
This service reads CSV optimization data and provides the best configuration for each epic.

Key Features:
- Epic-specific optimal parameter loading
- Performance-based configuration ranking
- Fallback to default configurations when optimization data unavailable
- Market condition-aware parameter selection
- Caching for performance optimization

Author: Trading System V1
Created: 2025-09-12
"""

import pandas as pd
import os
import logging
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timedelta
import json

# Configure logging
logger = logging.getLogger(__name__)

class SMCOptimalParameterService:
    """Service for retrieving optimal SMC parameters based on optimization results."""
    
    def __init__(self, results_csv_path: str = None):
        """Initialize the service with path to optimization results CSV."""
        if results_csv_path is None:
            # Default path relative to the service location
            base_path = os.path.dirname(os.path.abspath(__file__))
            results_csv_path = os.path.join(base_path, 'results', 'smc_optimization_results.csv')
        
        self.results_csv_path = results_csv_path
        self._results_cache = None
        self._cache_timestamp = None
        self.cache_duration = timedelta(minutes=30)  # Cache for 30 minutes
        
        # Default fallback configurations
        self.default_configs = {
            'conservative': {
                'smc_config': 'conservative',
                'confidence_level': 0.60,
                'timeframe': '15m',
                'stop_loss_pips': 10,
                'take_profit_pips': 20,
                'risk_reward_ratio': 2.0,
                'description': 'Conservative SMC approach with higher confidence requirements'
            },
            'default': {
                'smc_config': 'default',
                'confidence_level': 0.55,
                'timeframe': '15m',
                'stop_loss_pips': 10,
                'take_profit_pips': 20,
                'risk_reward_ratio': 2.0,
                'description': 'Balanced SMC configuration'
            },
            'aggressive': {
                'smc_config': 'aggressive',
                'confidence_level': 0.50,
                'timeframe': '15m',
                'stop_loss_pips': 8,
                'take_profit_pips': 16,
                'risk_reward_ratio': 2.0,
                'description': 'More aggressive SMC approach with lower confidence requirements'
            }
        }
    
    def _load_optimization_results(self) -> pd.DataFrame:
        """Load optimization results from CSV with caching."""
        current_time = datetime.now()
        
        # Check if cache is valid
        if (self._results_cache is not None and 
            self._cache_timestamp is not None and 
            current_time - self._cache_timestamp < self.cache_duration):
            logger.debug("Using cached optimization results")
            return self._results_cache
        
        try:
            if not os.path.exists(self.results_csv_path):
                logger.warning(f"Optimization results CSV not found at {self.results_csv_path}")
                return pd.DataFrame()
            
            # Load CSV data
            df = pd.read_csv(self.results_csv_path)
            logger.info(f"Loaded {len(df)} optimization results from {self.results_csv_path}")
            
            # Cache the results
            self._results_cache = df
            self._cache_timestamp = current_time
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading optimization results: {e}")
            return pd.DataFrame()
    
    def get_best_configuration_for_epic(self, epic: str, ranking_criteria: str = 'performance_score') -> Dict[str, Any]:
        """
        Get the best configuration for a specific epic based on optimization results.
        
        Args:
            epic: The trading pair (e.g., 'CS.D.EURUSD.CEEM.IP')
            ranking_criteria: How to rank configurations ('performance_score', 'win_rate', 'profit_factor', 'net_pips')
        
        Returns:
            Dict containing optimal configuration parameters
        """
        df = self._load_optimization_results()
        
        if df.empty:
            logger.warning(f"No optimization results available, using default config for {epic}")
            return self._get_fallback_config(epic)
        
        # Filter results for the specific epic
        epic_results = df[df['epic'] == epic]
        
        if epic_results.empty:
            logger.warning(f"No optimization results found for epic {epic}, using default config")
            return self._get_fallback_config(epic)
        
        # Rank by the specified criteria (descending order for better performance)
        best_result = epic_results.nlargest(1, ranking_criteria).iloc[0]
        
        # Convert to configuration dictionary
        config = {
            'epic': epic,
            'smc_config': best_result['smc_config'],
            'confidence_level': float(best_result['confidence_level']),
            'timeframe': best_result['timeframe'],
            'stop_loss_pips': int(best_result['stop_loss_pips']),
            'take_profit_pips': int(best_result['take_profit_pips']),
            'risk_reward_ratio': float(best_result['risk_reward_ratio']),
            'expected_win_rate': float(best_result['win_rate']),
            'expected_profit_factor': float(best_result['profit_factor']),
            'performance_score': float(best_result['performance_score']),
            'optimization_source': 'csv_results',
            'last_optimized': best_result.get('timestamp', 'unknown')
        }
        
        logger.info(f"Best config for {epic}: {best_result['smc_config']} "
                   f"(Win Rate: {best_result['win_rate']:.1f}%, "
                   f"Performance: {best_result['performance_score']:.1f})")
        
        return config
    
    def get_top_configurations_for_epic(self, epic: str, top_n: int = 3, 
                                       ranking_criteria: str = 'performance_score') -> List[Dict[str, Any]]:
        """Get the top N configurations for a specific epic."""
        df = self._load_optimization_results()
        
        if df.empty:
            logger.warning(f"No optimization results available for {epic}")
            return [self._get_fallback_config(epic)]
        
        # Filter and rank results
        epic_results = df[df['epic'] == epic]
        if epic_results.empty:
            return [self._get_fallback_config(epic)]
        
        top_results = epic_results.nlargest(top_n, ranking_criteria)
        
        configurations = []
        for idx, result in top_results.iterrows():
            config = {
                'rank': len(configurations) + 1,
                'epic': epic,
                'smc_config': result['smc_config'],
                'confidence_level': float(result['confidence_level']),
                'timeframe': result['timeframe'],
                'stop_loss_pips': int(result['stop_loss_pips']),
                'take_profit_pips': int(result['take_profit_pips']),
                'risk_reward_ratio': float(result['risk_reward_ratio']),
                'expected_win_rate': float(result['win_rate']),
                'expected_profit_factor': float(result['profit_factor']),
                'performance_score': float(result['performance_score']),
                'total_signals': int(result['total_signals']),
                'net_pips': float(result['net_pips'])
            }
            configurations.append(config)
        
        return configurations
    
    def get_all_epic_best_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get best configuration for all available epics."""
        df = self._load_optimization_results()
        
        if df.empty:
            logger.warning("No optimization results available")
            return {}
        
        # Get unique epics
        epics = df['epic'].unique()
        
        best_configs = {}
        for epic in epics:
            best_configs[epic] = self.get_best_configuration_for_epic(epic)
        
        return best_configs
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of all optimization results."""
        df = self._load_optimization_results()
        
        if df.empty:
            return {'error': 'No optimization results available'}
        
        summary = {
            'total_tests': len(df),
            'unique_epics': df['epic'].nunique(),
            'epics_list': df['epic'].unique().tolist(),
            'configurations_tested': df['smc_config'].unique().tolist(),
            'avg_win_rate': df['win_rate'].mean(),
            'avg_performance_score': df['performance_score'].mean(),
            'best_overall_epic': df.loc[df['performance_score'].idxmax(), 'epic'],
            'best_overall_config': df.loc[df['performance_score'].idxmax(), 'smc_config'],
            'best_overall_performance': df['performance_score'].max(),
            'data_timestamp': df['timestamp'].max() if 'timestamp' in df.columns else 'unknown'
        }
        
        return summary
    
    def _get_fallback_config(self, epic: str) -> Dict[str, Any]:
        """Get fallback configuration when optimization results are unavailable."""
        # Choose fallback based on epic characteristics
        if 'JPY' in epic:
            # For JPY pairs, use slightly different pip values due to different pip structure
            fallback = self.default_configs['default'].copy()
            fallback['stop_loss_pips'] = 8
            fallback['take_profit_pips'] = 16
        else:
            fallback = self.default_configs['default'].copy()
        
        fallback.update({
            'epic': epic,
            'optimization_source': 'fallback_default',
            'expected_win_rate': 75.0,  # Conservative estimate
            'expected_profit_factor': 2.0,
            'performance_score': 100.0,
            'last_optimized': 'never'
        })
        
        logger.info(f"Using fallback configuration for {epic}")
        return fallback
    
    def compare_configurations(self, epic: str, config1: str, config2: str) -> Dict[str, Any]:
        """Compare two SMC configurations for a specific epic."""
        df = self._load_optimization_results()
        
        if df.empty:
            return {'error': 'No optimization results available for comparison'}
        
        epic_results = df[df['epic'] == epic]
        if epic_results.empty:
            return {'error': f'No results available for epic {epic}'}
        
        result1 = epic_results[epic_results['smc_config'] == config1]
        result2 = epic_results[epic_results['smc_config'] == config2]
        
        if result1.empty or result2.empty:
            return {'error': f'Configuration(s) not found for {epic}'}
        
        r1 = result1.iloc[0]
        r2 = result2.iloc[0]
        
        comparison = {
            'epic': epic,
            'config1': {
                'name': config1,
                'win_rate': float(r1['win_rate']),
                'performance_score': float(r1['performance_score']),
                'profit_factor': float(r1['profit_factor']),
                'net_pips': float(r1['net_pips']),
                'total_signals': int(r1['total_signals'])
            },
            'config2': {
                'name': config2,
                'win_rate': float(r2['win_rate']),
                'performance_score': float(r2['performance_score']),
                'profit_factor': float(r2['profit_factor']),
                'net_pips': float(r2['net_pips']),
                'total_signals': int(r2['total_signals'])
            },
            'winner': {
                'by_win_rate': config1 if r1['win_rate'] > r2['win_rate'] else config2,
                'by_performance': config1 if r1['performance_score'] > r2['performance_score'] else config2,
                'by_profit_factor': config1 if r1['profit_factor'] > r2['profit_factor'] else config2,
                'by_net_pips': config1 if r1['net_pips'] > r2['net_pips'] else config2
            }
        }
        
        return comparison


# Convenience functions for easy import and usage
_service_instance = None

def get_smc_optimal_parameters(epic: str, ranking_criteria: str = 'performance_score') -> Dict[str, Any]:
    """Get optimal SMC parameters for an epic. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SMCOptimalParameterService()
    
    return _service_instance.get_best_configuration_for_epic(epic, ranking_criteria)

def get_smc_top_configurations(epic: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Get top N SMC configurations for an epic. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SMCOptimalParameterService()
    
    return _service_instance.get_top_configurations_for_epic(epic, top_n)

def get_all_smc_best_configs() -> Dict[str, Dict[str, Any]]:
    """Get best SMC configuration for all epics. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SMCOptimalParameterService()
    
    return _service_instance.get_all_epic_best_configs()

def get_smc_optimization_summary() -> Dict[str, Any]:
    """Get SMC optimization summary. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SMCOptimalParameterService()
    
    return _service_instance.get_optimization_summary()


if __name__ == "__main__":
    """Test the SMC optimal parameter service."""
    
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing SMC Optimal Parameter Service\n")
    
    # Initialize service
    service = SMCOptimalParameterService()
    
    # Test 1: Get optimization summary
    print("📊 Optimization Summary:")
    summary = service.get_optimization_summary()
    if 'error' not in summary:
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Unique Epics: {summary['unique_epics']}")
        print(f"   Average Win Rate: {summary['avg_win_rate']:.1f}%")
        print(f"   Best Overall: {summary['best_overall_epic']} ({summary['best_overall_config']})")
        print(f"   Best Performance Score: {summary['best_overall_performance']:.1f}")
    else:
        print(f"   Error: {summary['error']}")
    
    # Test 2: Get best config for specific epic
    print("\n🎯 Best Configuration Examples:")
    test_epics = ['CS.D.EURUSD.CEEM.IP', 'CS.D.AUDUSD.MINI.IP', 'CS.D.GBPUSD.MINI.IP']
    
    for epic in test_epics:
        config = service.get_best_configuration_for_epic(epic)
        print(f"   {epic}:")
        print(f"     Config: {config['smc_config']} | Confidence: {config['confidence_level']}")
        print(f"     SL/TP: {config['stop_loss_pips']}/{config['take_profit_pips']} pips")
        print(f"     Expected Win Rate: {config['expected_win_rate']:.1f}%")
        print(f"     Performance Score: {config['performance_score']:.1f}")
    
    # Test 3: Get top 3 configurations for one epic
    print(f"\n🏆 Top 3 Configurations for {test_epics[0]}:")
    top_configs = service.get_top_configurations_for_epic(test_epics[0], top_n=3)
    for config in top_configs:
        print(f"   #{config['rank']}: {config['smc_config']} "
              f"(Win Rate: {config['expected_win_rate']:.1f}%, "
              f"Performance: {config['performance_score']:.1f})")
    
    # Test 4: Compare two configurations
    print(f"\n⚖️ Comparing 'default' vs 'conservative' for {test_epics[0]}:")
    comparison = service.compare_configurations(test_epics[0], 'default', 'conservative')
    if 'error' not in comparison:
        print(f"   Default: Win Rate {comparison['config1']['win_rate']:.1f}%, "
              f"Performance {comparison['config1']['performance_score']:.1f}")
        print(f"   Conservative: Win Rate {comparison['config2']['win_rate']:.1f}%, "
              f"Performance {comparison['config2']['performance_score']:.1f}")
        print(f"   Winner by Performance: {comparison['winner']['by_performance']}")
    
    # Test 5: Convenience functions
    print("\n🔧 Testing Convenience Functions:")
    quick_config = get_smc_optimal_parameters('CS.D.USDJPY.MINI.IP')
    print(f"   Quick config for USDJPY: {quick_config['smc_config']} "
          f"(Win Rate: {quick_config['expected_win_rate']:.1f}%)")
    
    print("\n✅ All tests completed!")