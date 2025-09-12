# examples/alert_analytics_examples.py
"""
Examples of how to use the enhanced Alert History Manager
with configurable EMA analytics
"""

import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from core.alerts.alert_history import AlertHistoryManager

class AlertAnalytics:
    """Enhanced analytics for alert history with EMA configuration insights"""
    
    def __init__(self, alert_manager: AlertHistoryManager):
        self.alert_manager = alert_manager
    
    def compare_ema_configurations(self, days: int = 30):
        """Compare performance across different EMA configurations"""
        performance = self.alert_manager.get_ema_config_performance(days)
        
        print(f"\n📊 EMA Configuration Performance (Last {days} days)")
        print("=" * 70)
        
        for config in performance['configurations']:
            print(f"\n🔧 {config['ema_config'].upper()} Configuration:")
            print(f"   EMAs: {int(config['short_ema'])}/{int(config['long_ema'])}/{int(config['trend_ema'])}")
            print(f"   Total Alerts: {config['total_alerts']}")
            print(f"   Avg Confidence: {config['avg_confidence']:.3f}")
            print(f"   Bull/Bear Ratio: {config['bull_signals']}/{config['bear_signals']}")
            print(f"   Unique Pairs: {config['unique_pairs']}")
            print(f"   High Priority: {config['high_alerts']}")
            
            if config['total_alerts'] > 0:
                bull_ratio = config['bull_signals'] / config['total_alerts']
                print(f"   Bull %: {bull_ratio*100:.1f}%")
    
    def analyze_ema_config_by_pair(self, days: int = 30):
        """Analyze which EMA configs work best for specific pairs"""
        stats = self.alert_manager.get_alert_statistics(days)
        
        print(f"\n📈 EMA Configuration by Currency Pair (Last {days} days)")
        print("=" * 70)
        
        # Group by pair and EMA config
        pair_analysis = {}
        for epic_stat in stats['by_epic']:
            pair = epic_stat['pair']
            ema_config = epic_stat['ema_config']
            
            if pair not in pair_analysis:
                pair_analysis[pair] = {}
            
            pair_analysis[pair][ema_config] = {
                'alerts': epic_stat['alert_count'],
                'confidence': epic_stat['avg_confidence'],
                'bull_ratio': epic_stat['bull_count'] / epic_stat['alert_count'] if epic_stat['alert_count'] > 0 else 0
            }
        
        for pair, configs in pair_analysis.items():
            print(f"\n💱 {pair}:")
            for config_name, metrics in configs.items():
                print(f"   {config_name}: {metrics['alerts']} alerts, "
                      f"confidence {metrics['confidence']:.3f}, "
                      f"bull ratio {metrics['bull_ratio']:.2f}")
    
    def find_optimal_ema_settings(self, min_alerts: int = 10):
        """Find the most effective EMA configurations"""
        performance = self.alert_manager.get_ema_config_performance(30)
        
        print(f"\n🎯 Optimal EMA Configuration Analysis")
        print("=" * 50)
        
        configs = performance['configurations']
        
        # Filter configs with minimum alert count
        valid_configs = [c for c in configs if c['total_alerts'] >= min_alerts]
        
        if not valid_configs:
            print("❌ Not enough data for analysis")
            return
        
        # Sort by different metrics
        by_confidence = sorted(valid_configs, key=lambda x: x['avg_confidence'], reverse=True)
        by_volume = sorted(valid_configs, key=lambda x: x['total_alerts'], reverse=True)
        
        print(f"\n🏆 Highest Confidence (min {min_alerts} alerts):")
        for i, config in enumerate(by_confidence[:3]):
            print(f"   {i+1}. {config['ema_config']}: {config['avg_confidence']:.3f} "
                  f"({config['total_alerts']} alerts)")
        
        print(f"\n📊 Most Active:")
        for i, config in enumerate(by_volume[:3]):
            print(f"   {i+1}. {config['ema_config']}: {config['total_alerts']} alerts "
                  f"(confidence: {config['avg_confidence']:.3f})")
        
        # Calculate efficiency score (confidence * log(alerts))
        import math
        for config in valid_configs:
            config['efficiency'] = config['avg_confidence'] * math.log(config['total_alerts'])
        
        by_efficiency = sorted(valid_configs, key=lambda x: x['efficiency'], reverse=True)
        
        print(f"\n⚡ Most Efficient (confidence × log(volume)):")
        for i, config in enumerate(by_efficiency[:3]):
            print(f"   {i+1}. {config['ema_config']}: {config['efficiency']:.3f} "
                  f"({config['avg_confidence']:.3f} conf, {config['total_alerts']} alerts)")
    
    def generate_ema_config_report(self, days: int = 30):
        """Generate comprehensive EMA configuration report"""
        stats = self.alert_manager.get_alert_statistics(days)
        performance = self.alert_manager.get_ema_config_performance(days)
        
        print(f"\n📋 EMA Configuration Report - {days} Day Analysis")
        print("=" * 60)
        print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall = stats['overall']
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Alerts: {overall['total_alerts']}")
        print(f"   Unique EMA Configs: {overall['unique_ema_configs']}")
        print(f"   Average Confidence: {overall['avg_confidence']:.3f}")
        print(f"   Bull/Bear Split: {overall['bull_signals']}/{overall['bear_signals']}")
        
        # Configuration breakdown
        print(f"\n🔧 Configuration Breakdown:")
        for config in performance['configurations']:
            alerts = config['total_alerts']
            confidence = config['avg_confidence']
            bull_pct = (config['bull_signals'] / alerts * 100) if alerts > 0 else 0
            
            print(f"\n   {config['ema_config'].upper()}:")
            print(f"      Settings: {int(config['short_ema'])}/{int(config['long_ema'])}/{int(config['trend_ema'])}")
            print(f"      Volume: {alerts} alerts ({alerts/overall['total_alerts']*100:.1f}% of total)")
            print(f"      Quality: {confidence:.3f} avg confidence")
            print(f"      Bias: {bull_pct:.1f}% bullish")
            print(f"      Coverage: {config['unique_pairs']} currency pairs")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if overall['unique_ema_configs'] > 1:
            configs = performance['configurations']
            best_confidence = max(configs, key=lambda x: x['avg_confidence'])
            most_active = max(configs, key=lambda x: x['total_alerts'])
            
            print(f"   • Highest Quality: '{best_confidence['ema_config']}' config "
                  f"({best_confidence['avg_confidence']:.3f} confidence)")
            print(f"   • Most Active: '{most_active['ema_config']}' config "
                  f"({most_active['total_alerts']} alerts)")
            
            # Check for underperforming configs
            avg_confidence = sum(c['avg_confidence'] for c in configs) / len(configs)
            underperformers = [c for c in configs if c['avg_confidence'] < avg_confidence * 0.9]
            
            if underperformers:
                print(f"   • Consider reviewing: {', '.join(c['ema_config'] for c in underperformers)}")
        else:
            print(f"   • Consider testing additional EMA configurations for comparison")
        
        return stats, performance
    
    def export_ema_analysis_csv(self, days: int = 30):
        """Export detailed EMA analysis to CSV"""
        performance = self.alert_manager.get_ema_config_performance(days)
        
        # Convert to DataFrame for easy export
        df = pd.DataFrame(performance['configurations'])
        
        # Add calculated metrics
        df['bull_percentage'] = (df['bull_signals'] / df['total_alerts'] * 100).round(1)
        df['bear_percentage'] = (df['bear_signals'] / df['total_alerts'] * 100).round(1)
        df['alerts_per_day'] = (df['total_alerts'] / days).round(1)
        
        filename = f"ema_config_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"✅ EMA analysis exported to {filename}")
        return filename
    
    def plot_ema_config_comparison(self, days: int = 30):
        """Create visualization comparing EMA configurations"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("❌ Matplotlib/Seaborn required for plotting")
            return
        
        performance = self.alert_manager.get_ema_config_performance(days)
        configs = performance['configurations']
        
        if len(configs) < 2:
            print("❌ Need at least 2 EMA configurations to compare")
            return
        
        # Prepare data
        config_names = [c['ema_config'] for c in configs]
        alerts = [c['total_alerts'] for c in configs]
        confidence = [c['avg_confidence'] for c in configs]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Alert volume comparison
        ax1.bar(config_names, alerts, color='skyblue', alpha=0.7)
        ax1.set_title('Alert Volume by EMA Configuration')
        ax1.set_ylabel('Number of Alerts')
        ax1.tick_params(axis='x', rotation=45)
        
        # Confidence comparison
        ax2.bar(config_names, confidence, color='lightgreen', alpha=0.7)
        ax2.set_title('Average Confidence by EMA Configuration')
        ax2.set_ylabel('Average Confidence Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        filename = f"ema_config_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison chart saved as {filename}")
        plt.show()


# Usage Examples
def main():
    """Example usage of enhanced alert analytics"""
    
    # Initialize (assuming you have db_manager set up)
    # alert_manager = AlertHistoryManager(db_manager)
    # analytics = AlertAnalytics(alert_manager)
    
    print("🚀 Enhanced Alert Analytics Examples")
    print("=" * 50)
    
    # Example usage (commented out since we need actual data)
    """
    # Compare EMA configurations
    analytics.compare_ema_configurations(days=30)
    
    # Analyze by currency pair
    analytics.analyze_ema_config_by_pair(days=30)
    
    # Find optimal settings
    analytics.find_optimal_ema_settings(min_alerts=5)
    
    # Generate comprehensive report
    analytics.generate_ema_config_report(days=30)
    
    # Export analysis
    analytics.export_ema_analysis_csv(days=30)
    
    # Create visualizations
    analytics.plot_ema_config_comparison(days=30)
    
    # Filter alerts by specific EMA config
    aggressive_alerts = alert_manager.get_recent_alerts(
        hours=48,
        ema_config='aggressive'
    )
    
    print(f"Found {len(aggressive_alerts)} aggressive EMA alerts in last 48 hours")
    
    # Export specific configuration data
    alert_manager.export_alerts_csv(
        days=30,
        ema_config='scalping',
        filename='scalping_performance.csv'
    )
    """
    
    print("\n✅ All analytics functions ready for use!")
    print("\nNext steps:")
    print("1. Integrate the updated AlertHistoryManager")
    print("2. Run migration for existing data")
    print("3. Start collecting data with different EMA configs")
    print("4. Use analytics to optimize your strategy")


if __name__ == "__main__":
    main()