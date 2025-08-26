"""
Adaptive Differential Privacy for Financial Time Series
Integrates market conditions, data reliability, and temporal factors for optimal privacy-utility tradeoffs
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy.stats import norm

# Import our base mechanisms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privacy.dp_mechanisms import GaussianMechanism, NoiseCalibration, PrivacyBudget


@dataclass
class MarketConditions:
    """Market condition parameters for adaptive privacy"""
    volatility: float = 1.0      # Market volatility factor (1.0 = normal)
    volume: float = 1.0          # Trading volume factor (1.0 = normal)
    liquidity: float = 1.0       # Market liquidity factor (1.0 = normal) 
    news_intensity: float = 1.0  # News/event intensity (1.0 = normal)
    market_hours: bool = True    # Whether market is open
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            'volatility': self.volatility,
            'volume': self.volume,
            'liquidity': self.liquidity,
            'news_intensity': self.news_intensity,
            'market_hours': int(self.market_hours),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class DataReliability:
    """Data source reliability metrics"""
    source_name: str = "unknown"
    credibility_score: float = 0.8  # 0-1, higher = more credible
    freshness: float = 1.0          # Temporal freshness (1.0 = fresh, 0 = stale)
    completeness: float = 1.0       # Data completeness (0-1)
    consistency: float = 1.0        # Internal consistency (0-1)
    verification_score: float = 0.8 # External verification score (0-1)
    
    @property
    def overall_reliability(self) -> float:
        """Compute weighted overall reliability score"""
        weights = {
            'credibility': 0.3,
            'freshness': 0.25,
            'completeness': 0.2,
            'consistency': 0.15,
            'verification': 0.1
        }
        
        return (
            weights['credibility'] * self.credibility_score +
            weights['freshness'] * self.freshness +
            weights['completeness'] * self.completeness +
            weights['consistency'] * self.consistency +
            weights['verification'] * self.verification_score
        )


class AdaptivePrivacyCalibrator:
    """
    Calibrates differential privacy parameters based on:
    - Market conditions (volatility, volume, liquidity)
    - Data source reliability
    - Temporal factors (market hours, data freshness)
    - Privacy budget constraints
    """
    
    def __init__(self, 
                 base_privacy_budget: PrivacyBudget,
                 calibration_config: NoiseCalibration = None):
        self.base_budget = base_privacy_budget
        self.calibration = calibration_config or NoiseCalibration()
        self.logger = logging.getLogger(__name__)
        
        # Historical calibration data for learning
        self.calibration_history = []
        
        # Privacy regime settings
        self.privacy_regimes = self._initialize_privacy_regimes()
        
        self.logger.info("Adaptive Privacy Calibrator initialized")
    
    def _initialize_privacy_regimes(self) -> Dict[str, Dict[str, float]]:
        """Initialize different privacy regimes for different market conditions"""
        return {
            'high_volatility': {
                'epsilon_multiplier': 1.2,  # More privacy when market is volatile
                'delta_multiplier': 0.8,
                'description': 'High market volatility - increased privacy'
            },
            'low_liquidity': {
                'epsilon_multiplier': 1.5,  # More privacy when liquidity is low
                'delta_multiplier': 0.6,
                'description': 'Low market liquidity - enhanced privacy'
            },
            'market_closed': {
                'epsilon_multiplier': 0.7,  # Less privacy needed when market closed
                'delta_multiplier': 1.2,
                'description': 'Market closed - reduced privacy requirements'
            },
            'high_news_intensity': {
                'epsilon_multiplier': 1.3,  # More privacy during news events
                'delta_multiplier': 0.7,
                'description': 'High news intensity - increased privacy'
            },
            'normal': {
                'epsilon_multiplier': 1.0,
                'delta_multiplier': 1.0,
                'description': 'Normal market conditions'
            }
        }
    
    def determine_privacy_regime(self, 
                               market_conditions: MarketConditions) -> str:
        """
        Determine appropriate privacy regime based on market conditions
        
        Args:
            market_conditions: Current market conditions
            
        Returns:
            Privacy regime name
        """
        # Check conditions in order of priority
        if market_conditions.volatility > 1.5:
            return 'high_volatility'
        elif market_conditions.liquidity < 0.7:
            return 'low_liquidity'
        elif not market_conditions.market_hours:
            return 'market_closed'
        elif market_conditions.news_intensity > 1.3:
            return 'high_news_intensity'
        else:
            return 'normal'
    
    def calibrate_privacy_parameters(self,
                                   data_reliability: DataReliability,
                                   market_conditions: MarketConditions,
                                   base_epsilon: float,
                                   base_delta: float = None) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calibrate privacy parameters based on adaptive factors
        
        Args:
            data_reliability: Source reliability metrics
            market_conditions: Current market conditions
            base_epsilon: Base epsilon value
            base_delta: Base delta value
            
        Returns:
            Tuple of (calibrated_epsilon, calibrated_delta, metadata)
        """
        if base_delta is None:
            base_delta = self.base_budget.delta_total * 0.1
        
        # Determine privacy regime
        privacy_regime = self.determine_privacy_regime(market_conditions)
        regime_config = self.privacy_regimes[privacy_regime]
        
        # Start with regime-based adjustments
        epsilon_adj = base_epsilon * regime_config['epsilon_multiplier']
        delta_adj = base_delta * regime_config['delta_multiplier']
        
        # Reliability-based adjustments
        reliability = data_reliability.overall_reliability
        
        # Higher reliability = can use less privacy budget (more accurate utility)
        reliability_factor = 0.7 + (reliability * 0.6)  # Range: 0.7 - 1.3
        epsilon_adj *= reliability_factor
        
        # Freshness adjustment
        freshness_factor = 0.8 + (data_reliability.freshness * 0.4)  # Range: 0.8 - 1.2
        epsilon_adj *= freshness_factor
        
        # Market volatility fine-tuning
        if market_conditions.volatility > 1.0:
            # In high volatility, we need more privacy (larger epsilon = less privacy, so we increase it)
            volatility_factor = 1.0 + (market_conditions.volatility - 1.0) * 0.2
            epsilon_adj *= volatility_factor
        
        # Volume-based adjustment
        if market_conditions.volume > 1.5:
            # High volume = more trading activity to hide in
            volume_factor = 0.9  # Slightly less privacy needed
            epsilon_adj *= volume_factor
        elif market_conditions.volume < 0.5:
            # Low volume = less activity to hide in
            volume_factor = 1.1  # Slightly more privacy needed
            epsilon_adj *= volume_factor
        
        # Ensure we don't exceed budget constraints
        remaining_epsilon = self.base_budget.epsilon_remaining
        remaining_delta = self.base_budget.delta_remaining
        
        epsilon_final = min(epsilon_adj, remaining_epsilon * 0.9)  # Keep 10% buffer
        delta_final = min(delta_adj, remaining_delta * 0.9)
        
        # Metadata for logging
        calibration_metadata = {
            'privacy_regime': privacy_regime,
            'regime_description': regime_config['description'],
            'data_reliability': data_reliability.overall_reliability,
            'market_conditions': market_conditions.to_dict(),
            'adjustments': {
                'regime_epsilon_mult': regime_config['epsilon_multiplier'],
                'regime_delta_mult': regime_config['delta_multiplier'],
                'reliability_factor': reliability_factor,
                'freshness_factor': freshness_factor,
            },
            'base_epsilon': base_epsilon,
            'base_delta': base_delta,
            'final_epsilon': epsilon_final,
            'final_delta': delta_final,
            'budget_utilization': {
                'epsilon_used_pct': (self.base_budget.epsilon_consumed / self.base_budget.epsilon_total) * 100,
                'delta_used_pct': (self.base_budget.delta_consumed / self.base_budget.delta_total) * 100
            }
        }
        
        # Store for learning
        self.calibration_history.append({
            'timestamp': datetime.now(),
            'calibration': calibration_metadata.copy()
        })
        
        # Keep history manageable
        if len(self.calibration_history) > 1000:
            self.calibration_history = self.calibration_history[-500:]
        
        self.logger.debug(f"Privacy calibration: ε {base_epsilon:.4f} -> {epsilon_final:.4f}, "
                         f"δ {base_delta:.2e} -> {delta_final:.2e} ({privacy_regime})")
        
        return epsilon_final, delta_final, calibration_metadata
    
    def calibrate_noise_scale(self,
                            data_reliability: DataReliability,
                            market_conditions: MarketConditions,
                            base_epsilon: float,
                            base_delta: float = None,
                            sensitivity: float = 1.0) -> Tuple[float, Dict[str, Any]]:
        """
        Calibrate noise scale using adaptive privacy parameters
        
        Args:
            data_reliability: Source reliability metrics
            market_conditions: Market conditions
            base_epsilon: Base epsilon
            base_delta: Base delta
            sensitivity: Query sensitivity
            
        Returns:
            Tuple of (noise_scale, metadata)
        """
        # Get calibrated privacy parameters
        epsilon, delta, metadata = self.calibrate_privacy_parameters(
            data_reliability, market_conditions, base_epsilon, base_delta
        )
        
        # Compute Gaussian noise scale
        if epsilon <= 0 or delta <= 0:
            raise ValueError("Invalid privacy parameters after calibration")
        
        # Standard Gaussian mechanism: σ ≥ sqrt(2 * ln(1.25/δ)) * Δ / ε
        noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        
        # Apply additional market-based calibration
        market_noise_factor = self._compute_market_noise_factor(market_conditions)
        adaptive_noise_scale = noise_scale * market_noise_factor
        
        metadata['noise_calculation'] = {
            'base_noise_scale': noise_scale,
            'market_noise_factor': market_noise_factor,
            'adaptive_noise_scale': adaptive_noise_scale,
            'sensitivity': sensitivity
        }
        
        return adaptive_noise_scale, metadata
    
    def _compute_market_noise_factor(self, 
                                   market_conditions: MarketConditions) -> float:
        """
        Compute additional noise factor based on market microstructure
        
        Args:
            market_conditions: Current market conditions
            
        Returns:
            Market-based noise adjustment factor
        """
        base_factor = 1.0
        
        # High volatility market already has natural noise
        if market_conditions.volatility > 1.2:
            base_factor *= 0.95  # Slightly less artificial noise needed
        elif market_conditions.volatility < 0.8:
            base_factor *= 1.05  # More artificial noise in calm markets
        
        # Liquidity considerations
        if market_conditions.liquidity < 0.7:
            base_factor *= 1.1  # More noise in illiquid markets
        elif market_conditions.liquidity > 1.3:
            base_factor *= 0.95  # Less noise in highly liquid markets
        
        # News intensity
        if market_conditions.news_intensity > 1.2:
            base_factor *= 1.05  # More noise during news events
        
        return base_factor
    
    def get_calibration_statistics(self) -> Dict[str, Any]:
        """Get statistics about privacy calibration usage"""
        if not self.calibration_history:
            return {'error': 'No calibration history available'}
        
        recent_calibrations = self.calibration_history[-100:]  # Last 100
        
        # Regime usage
        regimes = [c['calibration']['privacy_regime'] for c in recent_calibrations]
        regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
        
        # Average adjustments
        epsilon_adjustments = [
            c['calibration']['final_epsilon'] / c['calibration']['base_epsilon'] 
            for c in recent_calibrations
        ]
        
        reliability_scores = [
            c['calibration']['data_reliability'] 
            for c in recent_calibrations
        ]
        
        return {
            'total_calibrations': len(self.calibration_history),
            'recent_calibrations': len(recent_calibrations),
            'regime_usage': regime_counts,
            'epsilon_adjustment_stats': {
                'mean': np.mean(epsilon_adjustments),
                'std': np.std(epsilon_adjustments),
                'min': np.min(epsilon_adjustments),
                'max': np.max(epsilon_adjustments)
            },
            'reliability_stats': {
                'mean': np.mean(reliability_scores),
                'std': np.std(reliability_scores),
                'min': np.min(reliability_scores),
                'max': np.max(reliability_scores)
            },
            'budget_status': {
                'epsilon_remaining': self.base_budget.epsilon_remaining,
                'delta_remaining': self.base_budget.delta_remaining,
                'epsilon_utilization_pct': (self.base_budget.epsilon_consumed / self.base_budget.epsilon_total) * 100,
                'delta_utilization_pct': (self.base_budget.delta_consumed / self.base_budget.delta_total) * 100
            }
        }


def test_adaptive_privacy():
    """Test adaptive privacy calibration"""
    
    print("🎯 Testing Adaptive Privacy Calibration...")
    print("=" * 60)
    
    # Initialize components
    privacy_budget = PrivacyBudget(epsilon_total=2.0, delta_total=1e-4)
    calibrator = AdaptivePrivacyCalibrator(privacy_budget)
    
    print("Components initialized")
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Normal Market Conditions',
            'data_reliability': DataReliability(
                source_name='matriks_api',
                credibility_score=0.9,
                freshness=1.0,
                completeness=1.0
            ),
            'market_conditions': MarketConditions(
                volatility=1.0,
                volume=1.0,
                liquidity=1.0,
                news_intensity=1.0,
                market_hours=True
            ),
            'base_epsilon': 0.5
        },
        
        {
            'name': 'High Volatility Market',
            'data_reliability': DataReliability(
                source_name='bloomberg_tr',
                credibility_score=0.95,
                freshness=0.9,
                completeness=0.95
            ),
            'market_conditions': MarketConditions(
                volatility=2.1,  # High volatility
                volume=1.8,
                liquidity=0.8,
                news_intensity=1.5,
                market_hours=True
            ),
            'base_epsilon': 0.5
        },
        
        {
            'name': 'Market Closed, Low Reliability',
            'data_reliability': DataReliability(
                source_name='news_crawler',
                credibility_score=0.6,  # Lower credibility
                freshness=0.7,
                completeness=0.8
            ),
            'market_conditions': MarketConditions(
                volatility=0.5,
                volume=0.1,
                liquidity=0.3,
                news_intensity=0.8,
                market_hours=False  # Market closed
            ),
            'base_epsilon': 0.5
        },
        
        {
            'name': 'News Event, High Intensity',
            'data_reliability': DataReliability(
                source_name='anadolu_ajansi',
                credibility_score=0.85,
                freshness=1.0,  # Very fresh news
                completeness=0.9
            ),
            'market_conditions': MarketConditions(
                volatility=1.6,
                volume=2.2,
                liquidity=1.1,
                news_intensity=2.5,  # High news intensity
                market_hours=True
            ),
            'base_epsilon': 0.5
        }
    ]
    
    print(f"\nTesting {len(test_scenarios)} scenarios...")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        
        # Calibrate privacy parameters
        epsilon, delta, metadata = calibrator.calibrate_privacy_parameters(
            scenario['data_reliability'],
            scenario['market_conditions'],
            scenario['base_epsilon']
        )
        
        # Calibrate noise scale
        noise_scale, noise_metadata = calibrator.calibrate_noise_scale(
            scenario['data_reliability'],
            scenario['market_conditions'],
            scenario['base_epsilon']
        )
        
        print(f"   Privacy regime: {metadata['privacy_regime']}")
        print(f"   Data reliability: {metadata['data_reliability']:.3f}")
        print(f"   Epsilon: {scenario['base_epsilon']:.3f} -> {epsilon:.3f}")
        print(f"   Delta: {metadata['base_delta']:.2e} -> {delta:.2e}")
        print(f"   Noise scale: {noise_scale:.4f}")
        
        # Simulate using the privacy budget
        privacy_budget.log_operation(
            operation_type=f"adaptive_test_{i}",
            epsilon_used=epsilon * 0.8,  # Use 80% of calibrated value
            delta_used=delta * 0.8,
            metadata={'scenario': scenario['name']}
        )
    
    # Display statistics
    print(f"\n📊 Calibration Statistics:")
    stats = calibrator.get_calibration_statistics()
    
    print(f"   Total calibrations: {stats['total_calibrations']}")
    print(f"   Privacy regimes used: {list(stats['regime_usage'].keys())}")
    print(f"   Average epsilon adjustment: {stats['epsilon_adjustment_stats']['mean']:.3f}")
    print(f"   Reliability score range: {stats['reliability_stats']['min']:.3f} - {stats['reliability_stats']['max']:.3f}")
    
    # Budget status
    budget_status = stats['budget_status']
    print(f"\n💰 Privacy Budget Status:")
    print(f"   Epsilon remaining: {budget_status['epsilon_remaining']:.3f}")
    print(f"   Delta remaining: {budget_status['delta_remaining']:.2e}")
    print(f"   Epsilon utilization: {budget_status['epsilon_utilization_pct']:.1f}%")
    print(f"   Delta utilization: {budget_status['delta_utilization_pct']:.1f}%")
    
    print(f"\n✅ Adaptive privacy calibration test completed!")


if __name__ == "__main__":
    test_adaptive_privacy()
