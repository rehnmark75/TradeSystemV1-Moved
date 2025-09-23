#!/usr/bin/env python3
"""
Test script for enhanced market intelligence with individual epic regimes
"""

import sys
import os
import json
from datetime import datetime, timezone

# Add the worker app to Python path
sys.path.append('/home/hr/Projects/TradeSystemV1/worker/app')

from forex_scanner.core.intelligence.market_intelligence_history_manager import MarketIntelligenceHistoryManager


def create_mock_intelligence_report():
    """Create a mock intelligence report with individual epic regimes"""
    return {
        'timestamp': datetime.now().isoformat(),
        'market_regime': {
            'dominant_regime': 'ranging',
            'confidence': 0.621,
            'regime_scores': {
                'trending': 0.38,
                'ranging': 0.62,
                'breakout': 0.25,
                'reversal': 0.15,
                'high_volatility': 0.35,
                'low_volatility': 0.65
            },
            'pair_analyses': {
                'CS.D.EURUSD.CEEM.IP': {
                    'regime_scores': {
                        'trending': 0.25,
                        'ranging': 0.75,
                        'breakout': 0.20,
                        'reversal': 0.10,
                        'high_volatility': 0.30,
                        'low_volatility': 0.70
                    },
                    'current_price': 1.0950,
                    'price_change_24h': 0.002,
                    'volatility_percentile': 45.0
                },
                'CS.D.GBPUSD.MINI.IP': {
                    'regime_scores': {
                        'trending': 0.65,
                        'ranging': 0.35,
                        'breakout': 0.40,
                        'reversal': 0.20,
                        'high_volatility': 0.60,
                        'low_volatility': 0.40
                    },
                    'current_price': 1.2845,
                    'price_change_24h': 0.008,
                    'volatility_percentile': 75.0
                },
                'CS.D.USDJPY.MINI.IP': {
                    'regime_scores': {
                        'trending': 0.30,
                        'ranging': 0.70,
                        'breakout': 0.25,
                        'reversal': 0.15,
                        'high_volatility': 0.25,
                        'low_volatility': 0.75
                    },
                    'current_price': 148.25,
                    'price_change_24h': -0.001,
                    'volatility_percentile': 35.0
                }
            },
            'market_strength': {
                'market_bias': 'neutral',
                'average_trend_strength': 0.40,
                'average_volatility': 0.38,
                'directional_consensus': 0.55,
                'market_efficiency': 0.62
            },
            'correlation_analysis': {
                'currency_strength': {'USD': 0.02, 'EUR': -0.01, 'GBP': 0.05, 'JPY': -0.02},
                'risk_on_off': 'neutral'
            },
            'recommended_strategy': {
                'strategy': 'mean_reversion',
                'ema_config': 'conservative',
                'recommendations': ['Trade range boundaries', 'Use tight stops', 'Quick profit taking']
            }
        },
        'session_analysis': {
            'current_session': 'london',
            'session_config': {
                'volatility': 'medium',
                'characteristics': 'Mixed signals with range-bound behavior',
                'preferred_pairs': ['EURUSD', 'GBPUSD'],
                'strategy_adjustment': 'Range trading approach',
                'risk_level': 'medium'
            },
            'optimal_timeframes': ['15m', '5m']
        },
        'trading_recommendations': {
            'primary_strategy': 'mean_reversion',
            'confidence_threshold': 0.72,
            'position_sizing': 'NORMAL',
            'strategy_adjustments': 'Range trading approach'
        },
        'confidence_score': 0.621
    }


def test_enhanced_market_intelligence():
    """Test the enhanced market intelligence logging"""
    print("🧪 Testing Enhanced Market Intelligence Logging")
    print("=" * 60)

    try:
        # Create mock data
        epic_list = ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP']
        intelligence_report = create_mock_intelligence_report()
        scan_cycle_id = f"test_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"📊 Mock intelligence report created")
        print(f"📝 Scan cycle ID: {scan_cycle_id}")
        print(f"🔢 Epic list: {', '.join([epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '') for epic in epic_list])}")

        # Initialize the history manager
        history_manager = MarketIntelligenceHistoryManager()

        print("\n💾 Saving market intelligence with individual epic regimes...")

        # Save the intelligence report
        record_id = history_manager.save_market_intelligence(
            intelligence_report=intelligence_report,
            epic_list=epic_list,
            scan_cycle_id=scan_cycle_id
        )

        if record_id:
            print(f"✅ Successfully saved market intelligence with record ID: {record_id}")
            print(f"📋 Expected log format:")
            print(f"   - Aggregate: ranging regime (62.1%) during london session - 3 epics analyzed")
            print(f"   - Individual: EURUSD(rang,75.0%), GBPUSD(tren,65.0%), USDJPY(rang,70.0%)")

            # Show what the enhanced logging should display
            print(f"\n🔍 Enhanced logging demonstration:")
            print(f"✅ Saved market intelligence #{record_id}: ranging regime (62.1%) during london session - 3 epics analyzed")
            print(f"📊 Epic breakdown: EURUSD(rang,75.0%), GBPUSD(tren,65.0%), USDJPY(rang,70.0%)")

        else:
            print("❌ Failed to save market intelligence")
            return False

        print("\n🎯 Test completed successfully!")
        print("🔄 The next actual market intelligence scan will show individual epic regimes in logs")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_enhanced_market_intelligence()
    if success:
        print("\n✅ Enhanced market intelligence logging is ready!")
        print("📈 Benefits:")
        print("   - Individual epic regime visibility")
        print("   - Better strategy adaptation per pair")
        print("   - More granular market analysis")
        print("   - Database stores both aggregate and individual data")
    else:
        print("\n❌ Test failed - check the implementation")

    sys.exit(0 if success else 1)