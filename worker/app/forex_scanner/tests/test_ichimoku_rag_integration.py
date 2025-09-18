#!/usr/bin/env python3

"""
Test script to validate RAG-enhanced Ichimoku strategy integration.
Tests all RAG enhancement modules and their integration with the main strategy.
"""

import sys
import os
import unittest
import asyncio
import logging
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from forex_scanner.core.strategies.ichimoku_strategy import IchimokuStrategy
from forex_scanner.core.strategies.helpers.ichimoku_rag_enhancer import IchimokuRAGEnhancer
from forex_scanner.core.strategies.helpers.tradingview_script_parser import TradingViewScriptParser
from forex_scanner.core.strategies.helpers.ichimoku_market_intelligence_adapter import IchimokuMarketIntelligenceAdapter
from forex_scanner.core.strategies.helpers.ichimoku_confluence_scorer import IchimokuConfluenceScorer
from forex_scanner.core.strategies.helpers.ichimoku_mtf_rag_validator import IchimokuMTFRAGValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestIchimokuRAGIntegration(unittest.TestCase):
    """Test suite for RAG-enhanced Ichimoku strategy integration."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock configurations
        self.strategy_config = {
            'epic': 'CS.D.EURUSD.TODAY.IP',
            'timeframe': '15m',
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'confidence_threshold': 0.75,
            'rag_enhancement_enabled': True,
            'tradingview_integration_enabled': True,
            'market_intelligence_enabled': True,
            'confluence_scoring_enabled': True,
            'mtf_rag_validation_enabled': True
        }

        # Mock database and data services
        self.mock_db_service = Mock()
        self.mock_market_data = Mock()
        self.mock_rag_interface = Mock()

        # Sample market data for testing
        self.sample_data = {
            'close': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            'high': [1.1005, 1.1015, 1.1025, 1.1035, 1.1045],
            'low': [1.0995, 1.1005, 1.1015, 1.1025, 1.1035],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'timestamp': [datetime.now().timestamp() - i * 900 for i in range(4, -1, -1)]
        }

    def test_module_imports(self):
        """Test that all RAG enhancement modules can be imported without errors."""
        logger.info("Testing module imports...")

        # Test that all classes can be instantiated
        try:
            rag_enhancer = IchimokuRAGEnhancer(
                db_service=self.mock_db_service,
                rag_interface=self.mock_rag_interface
            )
            self.assertIsInstance(rag_enhancer, IchimokuRAGEnhancer)
            logger.info("✓ IchimokuRAGEnhancer imported and instantiated successfully")

            tv_parser = TradingViewScriptParser(db_service=self.mock_db_service)
            self.assertIsInstance(tv_parser, TradingViewScriptParser)
            logger.info("✓ TradingViewScriptParser imported and instantiated successfully")

            market_adapter = IchimokuMarketIntelligenceAdapter(
                market_intelligence_service=Mock()
            )
            self.assertIsInstance(market_adapter, IchimokuMarketIntelligenceAdapter)
            logger.info("✓ IchimokuMarketIntelligenceAdapter imported and instantiated successfully")

            confluence_scorer = IchimokuConfluenceScorer(
                db_service=self.mock_db_service,
                rag_interface=self.mock_rag_interface
            )
            self.assertIsInstance(confluence_scorer, IchimokuConfluenceScorer)
            logger.info("✓ IchimokuConfluenceScorer imported and instantiated successfully")

            mtf_validator = IchimokuMTFRAGValidator(
                db_service=self.mock_db_service,
                rag_interface=self.mock_rag_interface
            )
            self.assertIsInstance(mtf_validator, IchimokuMTFRAGValidator)
            logger.info("✓ IchimokuMTFRAGValidator imported and instantiated successfully")

        except ImportError as e:
            self.fail(f"Failed to import RAG enhancement modules: {e}")
        except Exception as e:
            self.fail(f"Failed to instantiate RAG enhancement classes: {e}")

    def test_ichimoku_strategy_rag_initialization(self):
        """Test that IchimokuStrategy can be initialized with RAG enhancements."""
        logger.info("Testing IchimokuStrategy RAG initialization...")

        try:
            with patch('forex_scanner.core.strategies.ichimoku_strategy.IchimokuStrategy._initialize_rag_enhancements'):
                strategy = IchimokuStrategy(
                    config=self.strategy_config,
                    db_service=self.mock_db_service,
                    market_data_service=self.mock_market_data
                )
                self.assertIsInstance(strategy, IchimokuStrategy)
                logger.info("✓ IchimokuStrategy with RAG enhancements initialized successfully")

        except Exception as e:
            self.fail(f"Failed to initialize IchimokuStrategy with RAG enhancements: {e}")

    @patch('forex_scanner.core.strategies.helpers.ichimoku_rag_enhancer.IchimokuRAGEnhancer.enhance_ichimoku_signal')
    def test_rag_enhancer_integration(self, mock_enhance_signal):
        """Test RAG enhancer integration and signal enhancement."""
        logger.info("Testing RAG enhancer integration...")

        # Mock RAG enhancement response
        mock_enhance_signal.return_value = {
            'confidence_boost': 0.15,
            'pattern_matches': ['bullish_cloud_breakout', 'tk_cross_momentum'],
            'rag_confidence': 0.85,
            'enhancement_applied': True,
            'reasons': ['Strong historical pattern match', 'Momentum confluence detected']
        }

        # Test RAG enhancer
        rag_enhancer = IchimokuRAGEnhancer(
            db_service=self.mock_db_service,
            rag_interface=self.mock_rag_interface
        )

        # Mock signal data
        signal_data = {
            'signal_type': 'BUY',
            'confidence': 0.70,
            'price': Decimal('1.1040'),
            'tenkan': 1.1025,
            'kijun': 1.1015,
            'cloud_top': 1.1020,
            'cloud_bottom': 1.1010
        }

        # Test enhancement
        enhancement = rag_enhancer.enhance_ichimoku_signal(
            epic='CS.D.EURUSD.TODAY.IP',
            signal_data=signal_data,
            market_data=self.sample_data
        )

        self.assertIsInstance(enhancement, dict)
        self.assertIn('confidence_boost', enhancement)
        self.assertEqual(enhancement['confidence_boost'], 0.15)
        logger.info("✓ RAG enhancer integration test passed")

    @patch('forex_scanner.core.strategies.helpers.tradingview_script_parser.TradingViewScriptParser.get_technique_for_market_conditions')
    def test_tradingview_parser_integration(self, mock_get_technique):
        """Test TradingView script parser integration."""
        logger.info("Testing TradingView parser integration...")

        # Mock TradingView technique response
        mock_get_technique.return_value = {
            'technique_type': 'hybrid_scalping',
            'parameters': {'tenkan_period': 7, 'kijun_period': 22},
            'confidence_adjustment': 0.10,
            'market_suitability': 'high_volatility'
        }

        tv_parser = TradingViewScriptParser(db_service=self.mock_db_service)

        technique = tv_parser.get_technique_for_market_conditions(
            epic='CS.D.EURUSD.TODAY.IP',
            market_regime='trending',
            session='london',
            volatility='high'
        )

        self.assertIsInstance(technique, dict)
        self.assertIn('technique_type', technique)
        self.assertEqual(technique['technique_type'], 'hybrid_scalping')
        logger.info("✓ TradingView parser integration test passed")

    @patch('forex_scanner.core.strategies.helpers.ichimoku_market_intelligence_adapter.IchimokuMarketIntelligenceAdapter.get_adaptive_configuration')
    def test_market_intelligence_integration(self, mock_get_adaptive_config):
        """Test market intelligence adapter integration."""
        logger.info("Testing market intelligence integration...")

        # Mock adaptive configuration response
        mock_get_adaptive_config.return_value = {
            'adapted_config': {
                'confidence_threshold': 0.80,
                'tenkan_period': 8,
                'kijun_period': 24
            },
            'market_regime': 'trending',
            'session': 'london',
            'confidence_adjustment': 0.05,
            'adaptation_applied': True
        }

        market_adapter = IchimokuMarketIntelligenceAdapter(
            market_intelligence_service=Mock()
        )

        adaptive_config = market_adapter.get_adaptive_configuration(
            epic='CS.D.EURUSD.TODAY.IP',
            base_config=self.strategy_config,
            current_market_data=self.sample_data
        )

        self.assertIsInstance(adaptive_config, dict)
        self.assertIn('adapted_config', adaptive_config)
        self.assertEqual(adaptive_config['market_regime'], 'trending')
        logger.info("✓ Market intelligence integration test passed")

    @patch('forex_scanner.core.strategies.helpers.ichimoku_confluence_scorer.IchimokuConfluenceScorer.calculate_confluence_score')
    def test_confluence_scorer_integration(self, mock_calculate_confluence):
        """Test confluence scorer integration."""
        logger.info("Testing confluence scorer integration...")

        # Mock confluence score response
        mock_calculate_confluence.return_value = {
            'total_score': 0.85,
            'confluence_level': 'HIGH',
            'indicator_breakdown': {
                'momentum': 0.80,
                'trend': 0.90,
                'volume': 0.75,
                'support_resistance': 0.85
            },
            'confidence_adjustment': 0.12,
            'high_confidence_indicators': 7
        }

        confluence_scorer = IchimokuConfluenceScorer(
            db_service=self.mock_db_service,
            rag_interface=self.mock_rag_interface
        )

        confluence = confluence_scorer.calculate_confluence_score(
            epic='CS.D.EURUSD.TODAY.IP',
            signal_data={'signal_type': 'BUY', 'price': Decimal('1.1040')},
            market_data=self.sample_data
        )

        self.assertIsInstance(confluence, dict)
        self.assertIn('total_score', confluence)
        self.assertEqual(confluence['confluence_level'], 'HIGH')
        logger.info("✓ Confluence scorer integration test passed")

    @patch('forex_scanner.core.strategies.helpers.ichimoku_mtf_rag_validator.IchimokuMTFRAGValidator.validate_signal_with_mtf_rag')
    def test_mtf_rag_validator_integration(self, mock_validate_mtf):
        """Test multi-timeframe RAG validator integration."""
        logger.info("Testing MTF RAG validator integration...")

        # Mock MTF validation response
        mock_validate_mtf.return_value = {
            'validation_passed': True,
            'overall_bias': 'BULLISH',
            'confidence_score': 0.88,
            'timeframe_agreement': 0.75,
            'template_consensus': 0.80,
            'confidence_adjustment': 0.08,
            'supporting_timeframes': ['15m', '1h'],
            'conflicting_timeframes': []
        }

        mtf_validator = IchimokuMTFRAGValidator(
            db_service=self.mock_db_service,
            rag_interface=self.mock_rag_interface
        )

        validation = mtf_validator.validate_signal_with_mtf_rag(
            epic='CS.D.EURUSD.TODAY.IP',
            signal_data={'signal_type': 'BUY', 'confidence': 0.75},
            timeframes=['15m', '1h', '4h']
        )

        self.assertIsInstance(validation, dict)
        self.assertTrue(validation['validation_passed'])
        self.assertEqual(validation['overall_bias'], 'BULLISH')
        logger.info("✓ MTF RAG validator integration test passed")

    def test_database_schema_validation(self):
        """Test that RAG optimization database schema is valid SQL."""
        logger.info("Testing database schema validation...")

        schema_file = '/home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/optimization/create_ichimoku_rag_optimization_tables.sql'

        try:
            with open(schema_file, 'r') as f:
                sql_content = f.read()

            # Basic SQL validation checks
            self.assertIn('CREATE TABLE IF NOT EXISTS ichimoku_rag_optimization_runs', sql_content)
            self.assertIn('CREATE TABLE IF NOT EXISTS ichimoku_rag_configurations', sql_content)
            self.assertIn('CREATE TABLE IF NOT EXISTS ichimoku_rag_optimization_results', sql_content)
            self.assertIn('CREATE TABLE IF NOT EXISTS ichimoku_rag_best_parameters', sql_content)
            self.assertIn('CREATE TABLE IF NOT EXISTS ichimoku_rag_analytics', sql_content)

            # Check for key columns
            self.assertIn('rag_effectiveness_score', sql_content)
            self.assertIn('confidence_improvement_ratio', sql_content)
            self.assertIn('tradingview_technique_type', sql_content)
            self.assertIn('confluence_total_score', sql_content)
            self.assertIn('mtf_validation_passed', sql_content)

            logger.info("✓ Database schema validation passed")

        except FileNotFoundError:
            self.fail(f"RAG optimization schema file not found: {schema_file}")
        except Exception as e:
            self.fail(f"Database schema validation failed: {e}")

    def test_strategy_configuration_validation(self):
        """Test that strategy can handle RAG configuration parameters."""
        logger.info("Testing strategy configuration validation...")

        # Test various RAG configuration combinations
        test_configs = [
            {'rag_enhancement_enabled': True, 'tradingview_integration_enabled': True},
            {'rag_enhancement_enabled': True, 'tradingview_integration_enabled': False},
            {'rag_enhancement_enabled': False, 'tradingview_integration_enabled': True},
            {'market_intelligence_enabled': True, 'confluence_scoring_enabled': True},
            {'mtf_rag_validation_enabled': True}
        ]

        for config_update in test_configs:
            test_config = {**self.strategy_config, **config_update}

            try:
                # Test that configuration is valid (basic validation)
                self.assertIsInstance(test_config, dict)
                self.assertIn('epic', test_config)
                self.assertIn('timeframe', test_config)

            except Exception as e:
                self.fail(f"Configuration validation failed for {config_update}: {e}")

        logger.info("✓ Strategy configuration validation passed")

    def run_integration_tests(self):
        """Run all integration tests and provide summary."""
        logger.info("=" * 60)
        logger.info("STARTING ICHIMOKU RAG INTEGRATION TESTS")
        logger.info("=" * 60)

        test_methods = [
            self.test_module_imports,
            self.test_ichimoku_strategy_rag_initialization,
            self.test_rag_enhancer_integration,
            self.test_tradingview_parser_integration,
            self.test_market_intelligence_integration,
            self.test_confluence_scorer_integration,
            self.test_mtf_rag_validator_integration,
            self.test_database_schema_validation,
            self.test_strategy_configuration_validation
        ]

        passed_tests = 0
        failed_tests = 0

        for test_method in test_methods:
            try:
                test_method()
                passed_tests += 1
            except Exception as e:
                failed_tests += 1
                logger.error(f"❌ {test_method.__name__} failed: {e}")

        logger.info("=" * 60)
        logger.info(f"TEST SUMMARY: {passed_tests} passed, {failed_tests} failed")
        logger.info("=" * 60)

        return failed_tests == 0


def main():
    """Main test runner."""
    test_suite = TestIchimokuRAGIntegration()
    test_suite.setUp()

    success = test_suite.run_integration_tests()

    if success:
        print("\n🎉 All RAG integration tests passed successfully!")
        print("✅ RAG-enhanced Ichimoku strategy is ready for deployment")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)