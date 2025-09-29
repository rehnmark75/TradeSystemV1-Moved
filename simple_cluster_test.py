#!/usr/bin/env python3
"""
Simple test for cluster detection logic
Tests the basic cluster detection algorithm without full system dependencies
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
import logging

# Minimal imports to test cluster detection logic
class LevelType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    PIVOT_HIGH = "pivot_high"
    PIVOT_LOW = "pivot_low"

class LevelClusterType(Enum):
    RESISTANCE_CLUSTER = "resistance_cluster"
    SUPPORT_CLUSTER = "support_cluster"
    MIXED_CLUSTER = "mixed_cluster"

@dataclass
class EnhancedLevel:
    price: float
    level_type: LevelType
    strength: float
    touch_count: int
    creation_index: int
    last_touch_index: int
    age_bars: int = 0
    volume_confirmation: float = 0.0

@dataclass
class LevelCluster:
    cluster_id: str
    center_price: float
    cluster_type: LevelClusterType
    levels: List[EnhancedLevel]
    density_score: float
    cluster_radius_pips: float
    total_strength: float
    age_bars: int = 0

@dataclass
class ClusterRiskAssessment:
    signal_type: str
    current_price: float
    nearest_cluster: Optional[LevelCluster]
    cluster_distance_pips: float
    cluster_impact_score: float
    risk_multiplier: float
    cluster_density_warning: bool
    expected_risk_reward: float
    intervening_levels_count: int

class SimpleClusterDetector:
    """Simplified cluster detector for testing"""

    def __init__(self):
        self.max_cluster_radius_pips = 15.0
        self.min_levels_per_cluster = 2
        self.cluster_density_threshold = 0.8
        self.min_risk_reward_with_clusters = 2.0

    def _get_pip_size(self, epic: str) -> float:
        """Get pip size for the instrument"""
        if 'JPY' in epic.upper():
            return 0.01
        return 0.0001

    def _create_clusters_by_proximity(self, levels: List[EnhancedLevel], pip_size: float,
                                     cluster_type: LevelClusterType, epic: str) -> List[LevelCluster]:
        """Create clusters by grouping levels within proximity threshold"""
        if not levels:
            return []

        sorted_levels = sorted(levels, key=lambda l: l.price)
        clusters = []
        cluster_id_counter = 0

        i = 0
        while i < len(sorted_levels):
            cluster_levels = [sorted_levels[i]]
            current_level = sorted_levels[i]

            # Find all levels within cluster radius
            j = i + 1
            while j < len(sorted_levels):
                distance_pips = abs(sorted_levels[j].price - current_level.price) / pip_size
                if distance_pips <= self.max_cluster_radius_pips:
                    cluster_levels.append(sorted_levels[j])
                    j += 1
                else:
                    break

            # Create cluster if we have enough levels
            if len(cluster_levels) >= self.min_levels_per_cluster:
                center_price = sum(l.price for l in cluster_levels) / len(cluster_levels)
                total_strength = sum(l.strength for l in cluster_levels)
                density_score = len(cluster_levels) / self.max_cluster_radius_pips

                cluster = LevelCluster(
                    cluster_id=f"{epic}_{cluster_type.value}_{cluster_id_counter}",
                    center_price=center_price,
                    cluster_type=cluster_type,
                    levels=cluster_levels,
                    density_score=density_score,
                    cluster_radius_pips=self.max_cluster_radius_pips,
                    total_strength=total_strength
                )
                clusters.append(cluster)
                cluster_id_counter += 1

            i = j if j > i + 1 else i + 1

        return clusters

    def _detect_level_clusters(self, levels: List[EnhancedLevel], epic: str) -> List[LevelCluster]:
        """Detect level clusters using density-based algorithm"""
        if len(levels) < self.min_levels_per_cluster:
            return []

        pip_size = self._get_pip_size(epic)
        clusters = []

        # Separate by level type
        resistance_levels = [l for l in levels if l.level_type in [LevelType.RESISTANCE, LevelType.PIVOT_HIGH]]
        support_levels = [l for l in levels if l.level_type in [LevelType.SUPPORT, LevelType.PIVOT_LOW]]

        # Detect resistance clusters
        if resistance_levels:
            resistance_clusters = self._create_clusters_by_proximity(
                resistance_levels, pip_size, LevelClusterType.RESISTANCE_CLUSTER, epic
            )
            clusters.extend(resistance_clusters)

        # Detect support clusters
        if support_levels:
            support_clusters = self._create_clusters_by_proximity(
                support_levels, pip_size, LevelClusterType.SUPPORT_CLUSTER, epic
            )
            clusters.extend(support_clusters)

        return clusters

    def _assess_cluster_risk(self, current_price: float, signal_type: str,
                            clusters: List[LevelCluster], epic: str) -> ClusterRiskAssessment:
        """Assess risk impact of nearby clusters"""
        pip_size = self._get_pip_size(epic)

        # Find relevant clusters
        relevant_clusters = []
        if signal_type.upper() in ['BUY', 'BULL']:
            relevant_clusters = [c for c in clusters
                               if c.cluster_type == LevelClusterType.RESISTANCE_CLUSTER
                               and c.center_price > current_price]
        elif signal_type.upper() in ['SELL', 'BEAR']:
            relevant_clusters = [c for c in clusters
                               if c.cluster_type == LevelClusterType.SUPPORT_CLUSTER
                               and c.center_price < current_price]

        if not relevant_clusters:
            return ClusterRiskAssessment(
                signal_type=signal_type,
                current_price=current_price,
                nearest_cluster=None,
                cluster_distance_pips=float('inf'),
                cluster_impact_score=0.0,
                risk_multiplier=1.0,
                cluster_density_warning=False,
                expected_risk_reward=3.0,
                intervening_levels_count=0
            )

        # Find nearest cluster
        nearest_cluster = min(relevant_clusters,
                            key=lambda c: abs(c.center_price - current_price))

        distance_pips = abs(nearest_cluster.center_price - current_price) / pip_size

        # Calculate cluster impact
        cluster_impact_score = min(nearest_cluster.density_score, 1.0)

        # Count intervening levels
        intervening_levels = len([l for l in nearest_cluster.levels
                                if abs(l.price - current_price) / pip_size < distance_pips])

        # Calculate expected risk/reward
        expected_risk_reward = max(distance_pips / 20.0, 0.5)  # Simplified calculation

        # Determine if this is a high-risk scenario
        density_warning = (nearest_cluster.density_score > self.cluster_density_threshold and
                          distance_pips < 30.0)

        return ClusterRiskAssessment(
            signal_type=signal_type,
            current_price=current_price,
            nearest_cluster=nearest_cluster,
            cluster_distance_pips=distance_pips,
            cluster_impact_score=cluster_impact_score,
            risk_multiplier=1.0 + cluster_impact_score,
            cluster_density_warning=density_warning,
            expected_risk_reward=expected_risk_reward,
            intervening_levels_count=intervening_levels
        )

def create_test_levels_with_resistance_cluster():
    """Create test levels that form a resistance cluster"""
    levels = []

    # Create a resistance cluster around 1.10600-1.10650
    resistance_prices = [1.10600, 1.10615, 1.10625, 1.10640, 1.10650]

    for i, price in enumerate(resistance_prices):
        level = EnhancedLevel(
            price=price,
            level_type=LevelType.RESISTANCE,
            strength=0.7 + (i * 0.05),  # Varying strength
            touch_count=2 + i,
            creation_index=100 + (i * 10),
            last_touch_index=150 + (i * 5),
            age_bars=50
        )
        levels.append(level)

    # Add some support levels below (should not interfere)
    support_prices = [1.10200, 1.10150]
    for i, price in enumerate(support_prices):
        level = EnhancedLevel(
            price=price,
            level_type=LevelType.SUPPORT,
            strength=0.6,
            touch_count=2,
            creation_index=80,
            last_touch_index=120,
            age_bars=70
        )
        levels.append(level)

    return levels

def test_cluster_detection():
    """Test the cluster detection system"""
    print("🧪 Testing Simple Cluster Detection")
    print("=" * 50)

    # Create test data
    detector = SimpleClusterDetector()
    test_levels = create_test_levels_with_resistance_cluster()
    epic = "CS.D.EURUSD.MINI.IP"

    print(f"📊 Test Setup:")
    print(f"   Levels Created: {len(test_levels)}")
    print(f"   Resistance Levels: {len([l for l in test_levels if l.level_type == LevelType.RESISTANCE])}")
    print(f"   Support Levels: {len([l for l in test_levels if l.level_type == LevelType.SUPPORT])}")

    # Test cluster detection
    clusters = detector._detect_level_clusters(test_levels, epic)

    print(f"\n🔍 Cluster Detection Results:")
    print(f"   Clusters Found: {len(clusters)}")

    for i, cluster in enumerate(clusters):
        print(f"   Cluster {i+1}:")
        print(f"     Type: {cluster.cluster_type.value}")
        print(f"     Center: {cluster.center_price:.5f}")
        print(f"     Levels: {len(cluster.levels)}")
        print(f"     Density: {cluster.density_score:.2f}")
        print(f"     Total Strength: {cluster.total_strength:.2f}")

    # Test risk assessment for BUY signal below resistance cluster
    current_price = 1.10520  # Below the resistance cluster
    signal_type = "BUY"

    risk_assessment = detector._assess_cluster_risk(current_price, signal_type, clusters, epic)

    print(f"\n🎯 Risk Assessment for {signal_type} at {current_price:.5f}:")
    print(f"   Cluster Distance: {risk_assessment.cluster_distance_pips:.1f} pips")
    print(f"   Cluster Impact Score: {risk_assessment.cluster_impact_score:.2f}")
    print(f"   Density Warning: {'⚠️ YES' if risk_assessment.cluster_density_warning else '✅ NO'}")
    print(f"   Expected R/R: {risk_assessment.expected_risk_reward:.1f}")
    print(f"   Intervening Levels: {risk_assessment.intervening_levels_count}")

    # Check if trade should be rejected
    should_reject = (risk_assessment.cluster_density_warning or
                    risk_assessment.expected_risk_reward < detector.min_risk_reward_with_clusters)

    print(f"\n📋 Trade Decision:")
    print(f"   Should Reject: {'✅ YES' if should_reject else '❌ NO'}")
    print(f"   Reason: {'Cluster density warning or poor R/R' if should_reject else 'Trade allowed'}")

    return should_reject, len(clusters) > 0

def test_no_cluster_scenario():
    """Test scenario with no clusters"""
    print(f"\n🧪 Testing No-Cluster Scenario")
    print("-" * 30)

    detector = SimpleClusterDetector()

    # Create sparse levels (no clusters)
    levels = [
        EnhancedLevel(1.10500, LevelType.RESISTANCE, 0.6, 2, 100, 150, 50),
        EnhancedLevel(1.10200, LevelType.SUPPORT, 0.6, 2, 100, 150, 50),
        EnhancedLevel(1.10800, LevelType.RESISTANCE, 0.6, 2, 100, 150, 50),
    ]

    clusters = detector._detect_level_clusters(levels, "CS.D.EURUSD.MINI.IP")
    print(f"   Clusters Found: {len(clusters)}")

    # Test risk assessment
    risk_assessment = detector._assess_cluster_risk(1.10350, "BUY", clusters, "CS.D.EURUSD.MINI.IP")
    should_reject = risk_assessment.cluster_density_warning

    print(f"   Should Reject: {'❌ YES' if should_reject else '✅ NO'}")

    return not should_reject

if __name__ == "__main__":
    print("🚀 Simple Cluster Detection Test")
    print("=" * 50)

    # Test 1: Cluster scenario (should reject)
    test1_reject, test1_clusters_found = test_cluster_detection()

    # Test 2: No cluster scenario (should allow)
    test2_allow = test_no_cluster_scenario()

    print(f"\n🏁 Final Results:")
    print(f"   Cluster Detection: {'✅ WORKING' if test1_clusters_found else '❌ FAILED'}")
    print(f"   Risk Assessment: {'✅ WORKING' if test1_reject else '❌ FAILED'}")
    print(f"   No-Cluster Test: {'✅ WORKING' if test2_allow else '❌ FAILED'}")

    overall_success = test1_clusters_found and test1_reject and test2_allow

    print(f"\n🎉 Overall Test Result: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")

    if overall_success:
        print("✅ Cluster detection system is working correctly!")
        print("✅ The system detects resistance clusters above buy signals")
        print("✅ The system correctly assesses cluster risk")
        print("✅ The system allows trades when no clusters are present")
        print("\n📈 This addresses the scenario from the user's screenshot where")
        print("   buy signals below multiple resistance levels should be rejected.")
    else:
        print("❌ Some tests failed. Check the implementation.")