#!/usr/bin/env python3
"""
Best Parameters Retrieval Service

Comprehensive service for retrieving optimal parameters across all strategies.
Provides unified access to optimization results from EMA, MACD, ZeroLag, and SMC strategies.

Key Features:
- Multi-strategy parameter optimization
- Epic-specific configuration management
- Performance-based ranking and selection
- Market condition-aware parameter adjustment
- Comprehensive reporting and analysis

Author: Trading System V1
Created: 2025-09-12
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import os

# Configure logging
logger = logging.getLogger(__name__)

class BestParametersService:
    """Unified service for retrieving optimal parameters across all strategies."""
    
    def __init__(self):
        """Initialize the service with access to all strategy optimizations."""
        self.cache_duration = timedelta(minutes=30)
        self._cache = {}
        self._cache_timestamps = {}
        
        # Strategy services
        self._strategy_services = {}
        
        # Initialize available optimization services
        self._initialize_optimization_services()
    
    def _initialize_optimization_services(self):
        """Initialize all available optimization services."""
        
        # SMC Optimization Service
        try:
            from optimization.smc_optimal_parameter_service import SMCOptimalParameterService
            self._strategy_services['smc'] = SMCOptimalParameterService()
            logger.info("✅ SMC optimization service initialized")
        except ImportError as e:
            logger.warning(f"⚠️ SMC optimization service not available: {e}")
        
        # EMA Optimization Service
        try:
            from optimization.optimal_parameter_service import OptimalParameterService
            self._strategy_services['ema'] = OptimalParameterService()
            logger.info("✅ EMA optimization service initialized")
        except ImportError as e:
            logger.warning(f"⚠️ EMA optimization service not available: {e}")
        
        # MACD Optimization Service (if available)
        try:
            from optimization.macd_optimal_parameter_service import MACDOptimalParameterService
            self._strategy_services['macd'] = MACDOptimalParameterService()
            logger.info("✅ MACD optimization service initialized")
        except ImportError as e:
            logger.debug(f"MACD optimization service not available: {e}")
        
        # Zero-Lag Optimization Service (if available)
        try:
            from optimization.zerolag_optimal_parameter_service import ZeroLagOptimalParameterService
            self._strategy_services['zerolag'] = ZeroLagOptimalParameterService()
            logger.info("✅ Zero-Lag optimization service initialized")
        except ImportError as e:
            logger.debug(f"Zero-Lag optimization service not available: {e}")
        
        logger.info(f"🎯 Initialized {len(self._strategy_services)} optimization services: "
                   f"{list(self._strategy_services.keys())}")
    
    def get_best_strategy_for_epic(self, epic: str, ranking_criteria: str = 'performance_score') -> Dict[str, Any]:
        """
        Get the best performing strategy for a specific epic across all available strategies.
        
        Args:
            epic: Trading pair (e.g., 'CS.D.EURUSD.CEEM.IP')
            ranking_criteria: How to rank strategies ('performance_score', 'win_rate', 'profit_factor')
        
        Returns:
            Dict with best strategy and its parameters
        """
        
        strategy_performances = {}
        
        # Evaluate each available strategy for the epic
        for strategy_name, service in self._strategy_services.items():
            try:
                if strategy_name == 'smc':
                    config = service.get_best_configuration_for_epic(epic, ranking_criteria)
                    if config.get('optimization_source') != 'fallback_default':
                        strategy_performances[strategy_name] = {
                            'strategy': strategy_name,
                            'config': config,
                            'performance_score': config.get('performance_score', 0),
                            'win_rate': config.get('expected_win_rate', 0),
                            'profit_factor': config.get('expected_profit_factor', 1),
                            'optimization_source': config.get('optimization_source', 'unknown')
                        }
                
                elif strategy_name == 'ema':
                    # EMA service method
                    config = service.get_epic_optimal_parameters(epic)
                    if config:
                        strategy_performances[strategy_name] = {
                            'strategy': strategy_name,
                            'config': config,
                            'performance_score': config.get('performance_score', 0),
                            'win_rate': config.get('win_rate', 0),
                            'profit_factor': config.get('profit_factor', 1),
                            'optimization_source': config.get('optimization_source', 'unknown')
                        }
                        
                # Add more strategies as they become available
                elif hasattr(service, 'get_best_configuration_for_epic'):
                    config = service.get_best_configuration_for_epic(epic, ranking_criteria)
                    if config:
                        strategy_performances[strategy_name] = {
                            'strategy': strategy_name,
                            'config': config,
                            'performance_score': config.get('performance_score', 0),
                            'win_rate': config.get('win_rate', 0),
                            'profit_factor': config.get('profit_factor', 1),
                            'optimization_source': config.get('optimization_source', 'unknown')
                        }
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to get {strategy_name} parameters for {epic}: {e}")
        
        if not strategy_performances:
            logger.warning(f"No optimization data found for {epic} across any strategy")
            return {
                'error': f'No optimization data available for {epic}',
                'epic': epic,
                'available_strategies': list(self._strategy_services.keys())
            }
        
        # Rank strategies by the specified criteria
        best_strategy_name = max(strategy_performances.keys(), 
                               key=lambda s: strategy_performances[s].get(ranking_criteria, 0))
        
        best_performance = strategy_performances[best_strategy_name]
        
        result = {
            'epic': epic,
            'best_strategy': best_strategy_name,
            'ranking_criteria': ranking_criteria,
            'performance': best_performance,
            'all_strategies': strategy_performances,
            'recommendation': self._generate_strategy_recommendation(epic, best_performance),
            'alternatives': [
                {'strategy': name, 'score': perf.get(ranking_criteria, 0)} 
                for name, perf in strategy_performances.items() 
                if name != best_strategy_name
            ]
        }
        
        logger.info(f"🏆 Best strategy for {epic}: {best_strategy_name} "
                   f"(Score: {best_performance.get(ranking_criteria, 0):.1f})")
        
        return result
    
    def get_multi_epic_analysis(self, epics: List[str] = None, top_n: int = 5) -> Dict[str, Any]:
        """
        Analyze optimal strategies across multiple epics.
        
        Args:
            epics: List of epics to analyze. If None, analyzes all available epics.
            top_n: Number of top performing epics to include in detailed analysis
        
        Returns:
            Comprehensive multi-epic analysis
        """
        
        if epics is None:
            epics = self._get_all_available_epics()
        
        epic_analyses = {}
        strategy_summary = {}
        
        for epic in epics:
            try:
                analysis = self.get_best_strategy_for_epic(epic)
                if 'error' not in analysis:
                    epic_analyses[epic] = analysis
                    
                    # Track strategy performance summary
                    best_strategy = analysis['best_strategy']
                    if best_strategy not in strategy_summary:
                        strategy_summary[best_strategy] = {
                            'count': 0,
                            'total_score': 0,
                            'epics': []
                        }
                    
                    strategy_summary[best_strategy]['count'] += 1
                    strategy_summary[best_strategy]['total_score'] += analysis['performance'].get('performance_score', 0)
                    strategy_summary[best_strategy]['epics'].append(epic)
                    
            except Exception as e:
                logger.warning(f"Failed to analyze {epic}: {e}")
        
        # Calculate strategy averages
        for strategy, data in strategy_summary.items():
            data['average_score'] = data['total_score'] / data['count'] if data['count'] > 0 else 0
        
        # Get top performing epics
        top_epics = sorted(epic_analyses.items(), 
                          key=lambda x: x[1]['performance'].get('performance_score', 0), 
                          reverse=True)[:top_n]
        
        result = {
            'analysis_summary': {
                'total_epics_analyzed': len(epic_analyses),
                'strategies_compared': list(self._strategy_services.keys()),
                'top_performer': max(strategy_summary.items(), key=lambda x: x[1]['average_score'])[0] if strategy_summary else None
            },
            'strategy_summary': strategy_summary,
            'top_performing_epics': dict(top_epics),
            'all_epic_analyses': epic_analyses,
            'recommendations': self._generate_multi_epic_recommendations(epic_analyses, strategy_summary)
        }
        
        return result
    
    def get_strategy_comparison_report(self, epic: str) -> Dict[str, Any]:
        """Get detailed comparison of all strategies for a specific epic."""
        
        strategy_configs = {}
        
        for strategy_name, service in self._strategy_services.items():
            try:
                if strategy_name == 'smc':
                    config = service.get_best_configuration_for_epic(epic)
                    if config.get('optimization_source') != 'fallback_default':
                        strategy_configs[strategy_name] = config
                
                elif strategy_name == 'ema':
                    config = service.get_epic_optimal_parameters(epic)
                    if config:
                        strategy_configs[strategy_name] = config
                        
                elif hasattr(service, 'get_best_configuration_for_epic'):
                    config = service.get_best_configuration_for_epic(epic)
                    if config:
                        strategy_configs[strategy_name] = config
                        
            except Exception as e:
                logger.debug(f"Could not get {strategy_name} config for {epic}: {e}")
        
        if not strategy_configs:
            return {'error': f'No strategy configurations available for {epic}'}
        
        # Create comparison matrix
        comparison_matrix = []
        metrics = ['performance_score', 'win_rate', 'profit_factor']
        
        for strategy, config in strategy_configs.items():
            row = {'strategy': strategy}
            for metric in metrics:
                value = config.get(metric) or config.get(f'expected_{metric}', 0)
                row[metric] = value
            
            # Add strategy-specific details
            if strategy == 'smc':
                row['smc_config'] = config.get('smc_config', 'unknown')
                row['confidence_level'] = config.get('confidence_level', 0)
            elif strategy == 'ema':
                row['ema_config'] = config.get('ema_config', 'unknown')
                row['confidence'] = config.get('confidence', 0)
            
            comparison_matrix.append(row)
        
        # Find best performer for each metric
        best_performers = {}
        for metric in metrics:
            if comparison_matrix:
                best_performers[metric] = max(comparison_matrix, key=lambda x: x.get(metric, 0))['strategy']
        
        return {
            'epic': epic,
            'strategies_compared': list(strategy_configs.keys()),
            'comparison_matrix': comparison_matrix,
            'best_performers': best_performers,
            'recommendation': self._get_strategy_recommendation_from_comparison(comparison_matrix),
            'detailed_configs': strategy_configs
        }
    
    def _get_all_available_epics(self) -> List[str]:
        """Get all epics that have optimization data across any strategy."""
        all_epics = set()
        
        for strategy_name, service in self._strategy_services.items():
            try:
                if strategy_name == 'smc':
                    summary = service.get_optimization_summary()
                    if 'epics_list' in summary:
                        all_epics.update(summary['epics_list'])
                elif strategy_name == 'ema':
                    # Get EMA epics if the method exists
                    if hasattr(service, 'get_all_optimized_epics'):
                        ema_epics = service.get_all_optimized_epics()
                        all_epics.update(ema_epics)
            except Exception as e:
                logger.debug(f"Could not get epics from {strategy_name}: {e}")
        
        return list(all_epics)
    
    def _generate_strategy_recommendation(self, epic: str, best_performance: Dict) -> str:
        """Generate a recommendation for the best strategy."""
        strategy = best_performance['strategy']
        score = best_performance.get('performance_score', 0)
        win_rate = best_performance.get('win_rate', 0)
        
        if score > 500:
            performance_desc = "excellent"
        elif score > 300:
            performance_desc = "good"
        elif score > 100:
            performance_desc = "moderate"
        else:
            performance_desc = "basic"
        
        return (f"For {epic}, use {strategy.upper()} strategy with {performance_desc} "
               f"expected performance (Score: {score:.1f}, Win Rate: {win_rate:.1f}%)")
    
    def _generate_multi_epic_recommendations(self, epic_analyses: Dict, strategy_summary: Dict) -> List[str]:
        """Generate recommendations from multi-epic analysis."""
        recommendations = []
        
        if strategy_summary:
            # Most successful strategy overall
            best_strategy = max(strategy_summary.items(), key=lambda x: x[1]['average_score'])
            recommendations.append(
                f"Overall best performing strategy: {best_strategy[0].upper()} "
                f"(Average score: {best_strategy[1]['average_score']:.1f}, "
                f"Used for {best_strategy[1]['count']} epics)"
            )
            
            # Strategy diversity recommendation
            if len(strategy_summary) > 1:
                recommendations.append(
                    f"Strategy diversity: {len(strategy_summary)} different strategies "
                    f"are optimal across {len(epic_analyses)} epics - "
                    "consider using epic-specific optimization"
                )
        
        # High performance epics
        high_performers = [epic for epic, analysis in epic_analyses.items() 
                          if analysis['performance'].get('performance_score', 0) > 400]
        
        if high_performers:
            recommendations.append(
                f"High-performance epics ({len(high_performers)} found): "
                f"{', '.join(high_performers[:3])}"
                f"{'...' if len(high_performers) > 3 else ''}"
            )
        
        return recommendations
    
    def _get_strategy_recommendation_from_comparison(self, comparison_matrix: List[Dict]) -> str:
        """Get recommendation from strategy comparison matrix."""
        if not comparison_matrix:
            return "No strategies available for comparison"
        
        # Find strategy with highest overall score
        best_overall = max(comparison_matrix, key=lambda x: x.get('performance_score', 0))
        
        return (f"Recommended: {best_overall['strategy'].upper()} strategy "
               f"(Performance: {best_overall.get('performance_score', 0):.1f}, "
               f"Win Rate: {best_overall.get('win_rate', 0):.1f}%)")


# Convenience functions for easy access
_service_instance = None

def get_best_strategy_for_epic(epic: str, ranking_criteria: str = 'performance_score') -> Dict[str, Any]:
    """Get the best strategy for an epic. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = BestParametersService()
    
    return _service_instance.get_best_strategy_for_epic(epic, ranking_criteria)

def get_multi_epic_analysis(epics: List[str] = None, top_n: int = 5) -> Dict[str, Any]:
    """Get multi-epic analysis. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = BestParametersService()
    
    return _service_instance.get_multi_epic_analysis(epics, top_n)

def get_strategy_comparison_report(epic: str) -> Dict[str, Any]:
    """Get strategy comparison report. Convenience function."""
    global _service_instance
    if _service_instance is None:
        _service_instance = BestParametersService()
    
    return _service_instance.get_strategy_comparison_report(epic)

def get_optimization_overview() -> Dict[str, Any]:
    """Get overview of all available optimizations."""
    global _service_instance
    if _service_instance is None:
        _service_instance = BestParametersService()
    
    return {
        'available_strategies': list(_service_instance._strategy_services.keys()),
        'total_services': len(_service_instance._strategy_services),
        'all_epics': _service_instance._get_all_available_epics()
    }


if __name__ == "__main__":
    """Test the best parameters service."""
    
    import sys
    sys.path.append('/app/forex_scanner')
    
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing Best Parameters Service\n")
    
    # Initialize service
    service = BestParametersService()
    
    # Test 1: Get overview
    print("📊 Optimization Overview:")
    overview = get_optimization_overview()
    print(f"   Available Strategies: {overview['available_strategies']}")
    print(f"   Total Services: {overview['total_services']}")
    print(f"   Available Epics: {len(overview['all_epics'])}")
    
    if overview['all_epics']:
        # Test 2: Best strategy for specific epic
        test_epic = overview['all_epics'][0]
        print(f"\n🎯 Best Strategy for {test_epic}:")
        
        best_strategy = get_best_strategy_for_epic(test_epic)
        if 'error' not in best_strategy:
            print(f"   Winner: {best_strategy['best_strategy'].upper()}")
            print(f"   Performance Score: {best_strategy['performance']['performance_score']:.1f}")
            print(f"   Win Rate: {best_strategy['performance']['win_rate']:.1f}%")
            print(f"   Recommendation: {best_strategy['recommendation']}")
            
            # Show alternatives
            if best_strategy['alternatives']:
                print("   Alternatives:")
                for alt in best_strategy['alternatives']:
                    print(f"     - {alt['strategy'].upper()}: {alt['score']:.1f}")
        else:
            print(f"   Error: {best_strategy['error']}")
        
        # Test 3: Strategy comparison report
        print(f"\n⚖️ Strategy Comparison for {test_epic}:")
        comparison = get_strategy_comparison_report(test_epic)
        if 'error' not in comparison:
            print(f"   Strategies Compared: {comparison['strategies_compared']}")
            print(f"   Recommendation: {comparison['recommendation']}")
            
            print("   Performance Matrix:")
            for strategy_data in comparison['comparison_matrix']:
                print(f"     {strategy_data['strategy'].upper()}: "
                      f"Score {strategy_data.get('performance_score', 0):.1f}, "
                      f"Win Rate {strategy_data.get('win_rate', 0):.1f}%")
        else:
            print(f"   Error: {comparison['error']}")
        
        # Test 4: Multi-epic analysis (first 3 epics)
        test_epics = overview['all_epics'][:3]
        print(f"\n🌍 Multi-Epic Analysis ({len(test_epics)} epics):")
        
        multi_analysis = get_multi_epic_analysis(test_epics, top_n=2)
        if 'analysis_summary' in multi_analysis:
            summary = multi_analysis['analysis_summary']
            print(f"   Epics Analyzed: {summary['total_epics_analyzed']}")
            print(f"   Top Performer: {summary.get('top_performer', 'None')}")
            
            if multi_analysis['recommendations']:
                print("   Recommendations:")
                for rec in multi_analysis['recommendations']:
                    print(f"     - {rec}")
        
    print("\n✅ Best Parameters Service testing completed!")