"""
Differential Privacy Mechanisms for BIST Trading System
Implements Gaussian noise, privacy budget tracking, and adaptive noise calibration
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import math
from scipy.stats import norm


@dataclass
class PrivacyBudget:
    """Tracks privacy budget consumption over time"""
    epsilon_total: float = 1.0  # Total epsilon budget
    delta_total: float = 1e-5   # Total delta budget
    epsilon_consumed: float = 0.0
    delta_consumed: float = 0.0
    operations: List[Dict] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def epsilon_remaining(self) -> float:
        """Remaining epsilon budget"""
        return max(0, self.epsilon_total - self.epsilon_consumed)
    
    @property
    def delta_remaining(self) -> float:
        """Remaining delta budget"""
        return max(0, self.delta_total - self.delta_consumed)
    
    @property
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted"""
        return self.epsilon_remaining <= 0 or self.delta_remaining <= 0
    
    def log_operation(self, operation_type: str, epsilon_used: float, 
                     delta_used: float, metadata: Dict = None):
        """Log privacy budget consumption"""
        self.epsilon_consumed += epsilon_used
        self.delta_consumed += delta_used
        
        self.operations.append({
            'timestamp': datetime.now(),
            'operation_type': operation_type,
            'epsilon_used': epsilon_used,
            'delta_used': delta_used,
            'epsilon_remaining': self.epsilon_remaining,
            'delta_remaining': self.delta_remaining,
            'metadata': metadata or {}
        })


@dataclass
class NoiseCalibration:
    """Calibration parameters for adaptive noise"""
    base_noise_multiplier: float = 1.0
    reliability_weight: float = 0.3  # Weight for source reliability
    temporal_weight: float = 0.2     # Weight for temporal decay
    volume_weight: float = 0.1       # Weight for trading volume
    volatility_weight: float = 0.4   # Weight for market volatility
    
    def calculate_adaptive_multiplier(self, 
                                    source_reliability: float = 0.8,
                                    time_decay: float = 1.0,
                                    volume_factor: float = 1.0,
                                    volatility_factor: float = 1.0) -> float:
        """
        Calculate adaptive noise multiplier based on market conditions
        
        Args:
            source_reliability: Reliability of data source (0-1)
            time_decay: Temporal decay factor (>0, 1 = no decay)
            volume_factor: Trading volume factor (>0, 1 = normal volume)
            volatility_factor: Market volatility factor (>0, 1 = normal volatility)
        
        Returns:
            Adaptive noise multiplier
        """
        # Higher reliability = less noise needed
        reliability_factor = 1.0 - (source_reliability * self.reliability_weight)
        
        # Recent data needs less noise
        temporal_factor = 1.0 + ((1.0 - time_decay) * self.temporal_weight)
        
        # Higher volume = more noise (more trading activity to hide)
        volume_factor = 1.0 + ((volume_factor - 1.0) * self.volume_weight)
        
        # Higher volatility = less additional noise needed (market already noisy)
        volatility_factor = 1.0 + ((1.0 - volatility_factor) * self.volatility_weight)
        
        adaptive_multiplier = (
            self.base_noise_multiplier * 
            reliability_factor * 
            temporal_factor * 
            volume_factor * 
            volatility_factor
        )
        
        return max(0.1, adaptive_multiplier)  # Ensure minimum noise


class GaussianMechanism:
    """
    Gaussian mechanism for differential privacy
    Implements adaptive noise calibration for financial time series
    """
    
    def __init__(self, 
                 privacy_budget: PrivacyBudget,
                 noise_calibration: NoiseCalibration = None,
                 clip_norm: float = 1.0):
        self.privacy_budget = privacy_budget
        self.noise_calibration = noise_calibration or NoiseCalibration()
        self.clip_norm = clip_norm
        self.logger = logging.getLogger(__name__)
    
    def calibrate_noise_scale(self, 
                            epsilon: float,
                            delta: float,
                            sensitivity: float = 1.0,
                            **adaptive_params) -> float:
        """
        Calibrate noise scale for Gaussian mechanism
        
        Args:
            epsilon: Privacy parameter
            delta: Privacy parameter  
            sensitivity: L2 sensitivity of the query
            **adaptive_params: Parameters for adaptive noise calibration
            
        Returns:
            Calibrated noise scale (sigma)
        """
        if epsilon <= 0 or delta <= 0:
            raise ValueError("Epsilon and delta must be positive")
        
        # Standard Gaussian mechanism noise scale
        # sigma >= sqrt(2 * log(1.25/delta)) * sensitivity / epsilon
        base_sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
        
        # Apply adaptive calibration
        if adaptive_params:
            adaptive_multiplier = self.noise_calibration.calculate_adaptive_multiplier(
                **adaptive_params
            )
            base_sigma *= adaptive_multiplier
            
            self.logger.debug(f"Adaptive noise multiplier: {adaptive_multiplier:.3f}")
        
        self.logger.debug(f"Calibrated noise scale: {base_sigma:.6f}")
        return base_sigma
    
    def add_noise(self, 
                  data: Union[torch.Tensor, np.ndarray, float],
                  epsilon: float,
                  delta: float = None,
                  sensitivity: float = 1.0,
                  **adaptive_params) -> Union[torch.Tensor, np.ndarray, float]:
        """
        Add calibrated Gaussian noise to data
        
        Args:
            data: Input data (tensor, array, or scalar)
            epsilon: Privacy parameter
            delta: Privacy parameter (uses budget default if None)
            sensitivity: L2 sensitivity
            **adaptive_params: Adaptive calibration parameters
            
        Returns:
            Noisy data of same type as input
        """
        if delta is None:
            delta = self.privacy_budget.delta_total * 0.1  # Use 10% of budget
        
        # Check privacy budget
        if self.privacy_budget.epsilon_remaining < epsilon:
            raise RuntimeError("Insufficient epsilon budget")
        if self.privacy_budget.delta_remaining < delta:
            raise RuntimeError("Insufficient delta budget")
        
        # Calibrate noise scale
        sigma = self.calibrate_noise_scale(epsilon, delta, sensitivity, **adaptive_params)
        
        # Generate noise based on data type
        if isinstance(data, torch.Tensor):
            noise = torch.normal(0, sigma, size=data.shape)
            noisy_data = data + noise
        elif isinstance(data, np.ndarray):
            noise = np.random.normal(0, sigma, size=data.shape)
            noisy_data = data + noise
        else:  # scalar
            noise = np.random.normal(0, sigma)
            noisy_data = data + noise
        
        # Log budget consumption
        self.privacy_budget.log_operation(
            operation_type="gaussian_mechanism",
            epsilon_used=epsilon,
            delta_used=delta,
            metadata={
                'sigma': sigma,
                'sensitivity': sensitivity,
                'data_shape': getattr(data, 'shape', 'scalar'),
                'adaptive_params': adaptive_params
            }
        )
        
        return noisy_data
    
    def private_mean(self, 
                    data: Union[torch.Tensor, np.ndarray],
                    epsilon: float,
                    delta: float = None,
                    **adaptive_params) -> float:
        """
        Compute differentially private mean
        
        Args:
            data: Input data
            epsilon: Privacy parameter
            delta: Privacy parameter
            **adaptive_params: Adaptive calibration parameters
            
        Returns:
            Private mean estimate
        """
        if isinstance(data, torch.Tensor):
            true_mean = torch.mean(data).item()
            n = data.numel()
        else:
            true_mean = np.mean(data)
            n = data.size
        
        # Sensitivity of mean is 1/n for bounded data
        sensitivity = 1.0 / n
        
        return self.add_noise(
            true_mean, 
            epsilon, 
            delta, 
            sensitivity,
            **adaptive_params
        )
    
    def private_sum(self, 
                   data: Union[torch.Tensor, np.ndarray],
                   epsilon: float,
                   delta: float = None,
                   **adaptive_params) -> float:
        """
        Compute differentially private sum
        
        Args:
            data: Input data
            epsilon: Privacy parameter  
            delta: Privacy parameter
            **adaptive_params: Adaptive calibration parameters
            
        Returns:
            Private sum estimate
        """
        if isinstance(data, torch.Tensor):
            true_sum = torch.sum(data).item()
        else:
            true_sum = np.sum(data)
        
        # Sensitivity of sum is 1 for bounded data
        sensitivity = 1.0
        
        return self.add_noise(
            true_sum,
            epsilon,
            delta, 
            sensitivity,
            **adaptive_params
        )


class PrivacyAccountant:
    """
    Advanced privacy accounting for complex compositions
    Implements RDP (Rényi Differential Privacy) accounting
    """
    
    def __init__(self, 
                 target_epsilon: float = 1.0,
                 target_delta: float = 1e-5):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.rdp_orders = np.linspace(1.1, 10.0, 20)  # RDP orders for accounting
        self.rdp_budget = np.zeros_like(self.rdp_orders)
        self.compositions = []
        self.logger = logging.getLogger(__name__)
    
    def compute_rdp_gaussian(self, 
                           noise_multiplier: float, 
                           orders: np.ndarray = None) -> np.ndarray:
        """
        Compute RDP for Gaussian mechanism
        
        Args:
            noise_multiplier: Noise multiplier (sigma)
            orders: RDP orders to compute
            
        Returns:
            RDP values for each order
        """
        if orders is None:
            orders = self.rdp_orders
        
        # RDP for Gaussian mechanism: alpha / (2 * sigma^2)
        rdp_values = orders / (2 * noise_multiplier**2)
        return rdp_values
    
    def add_composition(self, 
                       mechanism_type: str,
                       noise_multiplier: float,
                       steps: int = 1,
                       metadata: Dict = None):
        """
        Add mechanism composition to privacy accounting
        
        Args:
            mechanism_type: Type of mechanism ('gaussian', 'laplace', etc.)
            noise_multiplier: Noise multiplier used
            steps: Number of composition steps
            metadata: Additional metadata
        """
        if mechanism_type == 'gaussian':
            rdp_single = self.compute_rdp_gaussian(noise_multiplier)
            rdp_composition = rdp_single * steps
        else:
            raise NotImplementedError(f"Mechanism type {mechanism_type} not implemented")
        
        # Add to cumulative RDP budget
        self.rdp_budget += rdp_composition
        
        # Log composition
        self.compositions.append({
            'timestamp': datetime.now(),
            'mechanism_type': mechanism_type,
            'noise_multiplier': noise_multiplier,
            'steps': steps,
            'rdp_added': rdp_composition.copy(),
            'metadata': metadata or {}
        })
        
        self.logger.info(f"Added {mechanism_type} composition: {steps} steps, "
                        f"noise_multiplier={noise_multiplier:.4f}")
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Convert current RDP to (epsilon, delta) via privacy amplification
        
        Returns:
            Tuple of (epsilon, delta) spent
        """
        if np.all(self.rdp_budget == 0):
            return 0.0, 0.0
        
        # Find optimal order that minimizes epsilon
        eps_values = []
        for i, order in enumerate(self.rdp_orders):
            if order == 1:
                continue
            
            # Convert RDP to (ε, δ)-DP
            eps = self.rdp_budget[i] + math.log(1 / self.target_delta) / (order - 1)
            eps_values.append(eps)
        
        epsilon_spent = min(eps_values) if eps_values else float('inf')
        return epsilon_spent, self.target_delta
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        epsilon_spent, delta_spent = self.get_privacy_spent()
        
        epsilon_remaining = max(0, self.target_epsilon - epsilon_spent)
        delta_remaining = max(0, self.target_delta - delta_spent)
        
        return epsilon_remaining, delta_remaining
    
    def can_afford(self, 
                   mechanism_type: str,
                   noise_multiplier: float,
                   steps: int = 1) -> bool:
        """
        Check if we can afford a given privacy cost
        
        Args:
            mechanism_type: Type of mechanism
            noise_multiplier: Proposed noise multiplier
            steps: Number of steps
            
        Returns:
            True if affordable, False otherwise
        """
        # Simulate the composition
        temp_rdp = self.rdp_budget.copy()
        
        if mechanism_type == 'gaussian':
            rdp_cost = self.compute_rdp_gaussian(noise_multiplier) * steps
        else:
            raise NotImplementedError(f"Mechanism type {mechanism_type} not implemented")
        
        temp_rdp += rdp_cost
        
        # Check if this exceeds our budget
        eps_values = []
        for i, order in enumerate(self.rdp_orders):
            if order == 1:
                continue
            
            eps = temp_rdp[i] + math.log(1 / self.target_delta) / (order - 1)
            eps_values.append(eps)
        
        projected_epsilon = min(eps_values) if eps_values else float('inf')
        
        return projected_epsilon <= self.target_epsilon


def test_differential_privacy():
    """Test function for differential privacy mechanisms"""
    
    print("🔒 Testing Differential Privacy Mechanisms...")
    print("=" * 60)
    
    # Test 1: Privacy Budget Tracking
    print("\n1. Testing Privacy Budget Tracking...")
    budget = PrivacyBudget(epsilon_total=2.0, delta_total=1e-4)
    
    budget.log_operation("test_op", 0.5, 1e-5)
    print(f"   After operation: ε remaining = {budget.epsilon_remaining:.2f}, "
          f"δ remaining = {budget.delta_remaining:.2e}")
    
    # Test 2: Gaussian Mechanism  
    print("\n2. Testing Gaussian Mechanism...")
    gaussian = GaussianMechanism(budget)
    
    # Test with synthetic financial data
    price_data = torch.tensor([100.0, 101.2, 99.8, 102.1, 100.5])
    
    noisy_data = gaussian.add_noise(
        price_data, 
        epsilon=0.5,
        delta=1e-5,
        source_reliability=0.9,
        volatility_factor=1.2
    )
    
    print(f"   Original data: {price_data.tolist()}")
    print(f"   Noisy data: {[f'{x:.2f}' for x in noisy_data.tolist()]}")
    print(f"   Noise added: {[f'{(n-o):.2f}' for n, o in zip(noisy_data, price_data)]}")
    
    # Test 3: Private Statistics
    print("\n3. Testing Private Statistics...")
    
    large_data = torch.randn(1000) * 10 + 100  # Stock prices around 100
    
    true_mean = torch.mean(large_data).item()
    private_mean = gaussian.private_mean(large_data, epsilon=0.3, source_reliability=0.8)
    
    print(f"   True mean: {true_mean:.3f}")
    print(f"   Private mean: {private_mean:.3f}")
    print(f"   Error: {abs(private_mean - true_mean):.3f}")
    
    # Test 4: Privacy Accountant
    print("\n4. Testing Privacy Accountant...")
    accountant = PrivacyAccountant(target_epsilon=1.0, target_delta=1e-5)
    
    # Simulate multiple training steps
    accountant.add_composition('gaussian', noise_multiplier=1.2, steps=100)
    accountant.add_composition('gaussian', noise_multiplier=1.5, steps=50)
    
    epsilon_spent, delta_spent = accountant.get_privacy_spent()
    epsilon_remaining, delta_remaining = accountant.get_remaining_budget()
    
    print(f"   Privacy spent: ε = {epsilon_spent:.4f}, δ = {delta_spent:.2e}")
    print(f"   Privacy remaining: ε = {epsilon_remaining:.4f}, δ = {delta_remaining:.2e}")
    
    # Test affordability
    can_afford_more = accountant.can_afford('gaussian', noise_multiplier=2.0, steps=10)
    print(f"   Can afford 10 more steps with σ=2.0: {can_afford_more}")
    
    print(f"\n✅ Differential Privacy mechanisms test completed!")
    

if __name__ == "__main__":
    test_differential_privacy()
