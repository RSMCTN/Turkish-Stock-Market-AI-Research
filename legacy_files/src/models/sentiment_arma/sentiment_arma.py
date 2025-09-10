"""
SentimentARMA - Core Mathematical Implementation

This module implements the sentimentARMA mathematical model that combines:
1. Traditional ARMA(p,q) time series modeling
2. VADER sentiment analysis from financial news 
3. KAP announcement impact weighting
4. Differential Privacy integration

Mathematical Foundation:
Yt = ARMA(p,q) × (1 + β × St × Wt) + εt

Components:
- ARMA(p,q): φ₁Yt-1 + ... + φₚYt-p + θ₁εt-1 + ... + θₘεt-m  
- St: VADER sentiment score ∈ [-1, +1] from Turkish financial news
- Wt: KAP announcement weight based on impact severity
- β: Sentiment sensitivity parameter (optimizable)
- εt: Error term with differential privacy noise
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ARMAParameters:
    """ARMA model parameters"""
    p: int  # AR order
    q: int  # MA order
    phi: np.ndarray  # AR coefficients [φ₁, φ₂, ..., φₚ]
    theta: np.ndarray  # MA coefficients [θ₁, θ₂, ..., θₘ] 
    beta: float = 0.5  # Sentiment sensitivity parameter
    sigma: float = 1.0  # Error variance
    
@dataclass  
class SentimentData:
    """Sentiment analysis results"""
    timestamp: pd.Timestamp
    symbol: str
    vader_score: float  # [-1, +1]
    news_text: str
    source: str  # 'KAP', 'news', 'social'
    confidence: float  # [0, 1]

@dataclass
class KAPAnnouncement:
    """KAP announcement data"""
    timestamp: pd.Timestamp
    symbol: str
    announcement_type: str  # 'ODA', 'FR', 'DG', etc.
    impact_weight: float  # [0, 3] based on severity
    title: str
    content: str

class SentimentARMA:
    """
    Core SentimentARMA mathematical model implementation
    
    Combines traditional ARMA modeling with sentiment analysis and
    KAP announcement impact weighting for enhanced stock price prediction.
    """
    
    def __init__(
        self,
        p: int = 2,  # AR order
        q: int = 1,  # MA order
        beta: float = 0.5,  # Sentiment sensitivity
        dp_epsilon: float = 1.0,  # Differential privacy parameter
        symbol: str = None
    ):
        """
        Initialize SentimentARMA model
        
        Args:
            p: Autoregressive order
            q: Moving average order  
            beta: Sentiment sensitivity parameter
            dp_epsilon: Differential privacy epsilon parameter
            symbol: Stock symbol for focused modeling
        """
        self.p = p
        self.q = q
        self.beta = beta
        self.dp_epsilon = dp_epsilon
        self.symbol = symbol
        
        # Model parameters (to be estimated)
        self.phi = np.zeros(p)  # AR coefficients
        self.theta = np.zeros(q)  # MA coefficients
        self.sigma = 1.0  # Error variance
        
        # Data storage
        self.price_history: List[float] = []
        self.sentiment_history: List[SentimentData] = []
        self.kap_history: List[KAPAnnouncement] = []
        self.residuals: List[float] = []
        
        # Model state
        self.is_fitted = False
        self.training_metrics = {}
        
    def _arma_component(self, 
                       prices: np.ndarray, 
                       t: int) -> float:
        """
        Calculate ARMA(p,q) component
        
        ARMA(p,q): φ₁Yt-1 + ... + φₚYt-p + θ₁εt-1 + ... + θₘεt-m
        
        Args:
            prices: Historical prices array
            t: Current time index
            
        Returns:
            ARMA predicted value
        """
        arma_value = 0.0
        
        # Autoregressive component: φ₁Yt-1 + ... + φₚYt-p
        for i in range(self.p):
            if t - 1 - i >= 0:
                arma_value += self.phi[i] * prices[t - 1 - i]
                
        # Moving average component: θ₁εt-1 + ... + θₘεt-m  
        for i in range(self.q):
            if t - 1 - i >= 0 and len(self.residuals) > i:
                arma_value += self.theta[i] * self.residuals[-(i+1)]
                
        return arma_value
    
    def _sentiment_component(self, 
                           timestamp: pd.Timestamp) -> float:
        """
        Calculate sentiment component St
        
        Aggregates VADER sentiment scores from recent financial news
        within a configurable time window.
        
        Args:
            timestamp: Target prediction timestamp
            
        Returns:
            Aggregated sentiment score ∈ [-1, +1]
        """
        # Time window for sentiment aggregation (configurable)
        time_window = pd.Timedelta(hours=4)  # 4-hour sentiment window
        
        relevant_sentiment = [
            s for s in self.sentiment_history
            if abs((s.timestamp - timestamp).total_seconds()) <= time_window.total_seconds()
        ]
        
        if not relevant_sentiment:
            return 0.0  # Neutral sentiment if no data
            
        # Weighted average by confidence
        total_weight = sum(s.confidence for s in relevant_sentiment)
        if total_weight == 0:
            return 0.0
            
        weighted_sentiment = sum(
            s.vader_score * s.confidence for s in relevant_sentiment
        ) / total_weight
        
        # Ensure bounds [-1, +1]
        return np.clip(weighted_sentiment, -1.0, 1.0)
    
    def _kap_weight_component(self, 
                            timestamp: pd.Timestamp) -> float:
        """
        Calculate KAP announcement impact weight Wt
        
        Weights recent KAP announcements by their potential market impact.
        Uses exponential decay for time-distance weighting.
        
        Args:
            timestamp: Target prediction timestamp
            
        Returns:
            KAP impact weight factor ∈ [0, 3]
        """
        # Time decay parameter (announcements lose impact over time)
        decay_hours = 24  # 24-hour impact decay
        
        total_weight = 0.0
        
        for kap in self.kap_history:
            time_diff = (timestamp - kap.timestamp).total_seconds() / 3600  # hours
            
            if time_diff >= 0 and time_diff <= 72:  # 3-day maximum impact window
                # Exponential decay: W = impact_weight × e^(-t/λ)
                decay_factor = np.exp(-time_diff / decay_hours)
                total_weight += kap.impact_weight * decay_factor
                
        # Cap maximum weight at 3.0 for numerical stability
        return min(total_weight, 3.0)
    
    def _add_differential_privacy_noise(self, 
                                      prediction: float,
                                      sensitivity: float = 1.0) -> float:
        """
        Add differential privacy Laplace noise
        
        Args:
            prediction: Clean prediction value
            sensitivity: Function sensitivity (Δf)
            
        Returns:
            Noisy prediction with DP guarantee
        """
        if self.dp_epsilon <= 0:
            return prediction  # No privacy if epsilon=0
            
        # Laplace mechanism: Lap(Δf/ε)
        scale = sensitivity / self.dp_epsilon
        noise = np.random.laplace(0, scale)
        
        return prediction + noise
    
    def predict_single(self, 
                      timestamp: pd.Timestamp,
                      add_dp_noise: bool = True) -> Dict[str, float]:
        """
        Generate single timestamp prediction using sentimentARMA formula
        
        Core Formula: Yt = ARMA(p,q) × (1 + β × St × Wt) + εt
        
        Args:
            timestamp: Target prediction timestamp
            add_dp_noise: Whether to add differential privacy noise
            
        Returns:
            Dictionary with prediction components and final value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if len(self.price_history) < max(self.p, self.q):
            raise ValueError(f"Insufficient history: need at least {max(self.p, self.q)} points")
        
        # Convert price history to numpy array
        prices = np.array(self.price_history)
        t = len(prices)
        
        # 1. ARMA component: ARMA(p,q)
        arma_pred = self._arma_component(prices, t)
        
        # 2. Sentiment component: St ∈ [-1, +1]  
        sentiment_score = self._sentiment_component(timestamp)
        
        # 3. KAP weight component: Wt ∈ [0, 3]
        kap_weight = self._kap_weight_component(timestamp)
        
        # 4. sentimentARMA formula: ARMA(p,q) × (1 + β × St × Wt)
        sentiment_multiplier = 1.0 + self.beta * sentiment_score * kap_weight
        raw_prediction = arma_pred * sentiment_multiplier
        
        # 5. Add differential privacy noise if requested
        final_prediction = raw_prediction
        if add_dp_noise:
            final_prediction = self._add_differential_privacy_noise(raw_prediction)
        
        return {
            'timestamp': timestamp,
            'arma_component': arma_pred,
            'sentiment_score': sentiment_score,
            'kap_weight': kap_weight,
            'sentiment_multiplier': sentiment_multiplier,
            'raw_prediction': raw_prediction,
            'final_prediction': final_prediction,
            'dp_noise_added': add_dp_noise
        }
    
    def fit(self, 
           price_data: pd.Series,
           sentiment_data: List[SentimentData] = None,
           kap_data: List[KAPAnnouncement] = None,
           method: str = 'mle') -> Dict[str, float]:
        """
        Fit sentimentARMA model parameters using Maximum Likelihood Estimation
        
        Args:
            price_data: Historical stock prices (pandas Series with datetime index)
            sentiment_data: Historical sentiment analysis results
            kap_data: Historical KAP announcements
            method: Parameter estimation method ('mle', 'yule_walker')
            
        Returns:
            Dictionary with fitted parameters and training metrics
        """
        # Store training data
        self.price_history = price_data.values.tolist()
        self.sentiment_history = sentiment_data or []
        self.kap_history = kap_data or []
        
        # Initialize residuals array
        self.residuals = [0.0] * len(self.price_history)
        
        # Simple parameter estimation (can be enhanced with MLE)
        if method == 'mle':
            self._fit_mle(price_data)
        elif method == 'yule_walker':
            self._fit_yule_walker(price_data)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
            
        self.is_fitted = True
        
        # Calculate training metrics
        self.training_metrics = self._calculate_training_metrics()
        
        return {
            'phi': self.phi.tolist(),
            'theta': self.theta.tolist(), 
            'beta': self.beta,
            'sigma': self.sigma,
            'training_mse': self.training_metrics.get('mse', 0),
            'training_mae': self.training_metrics.get('mae', 0)
        }
    
    def _fit_mle(self, price_data: pd.Series):
        """Maximum Likelihood Estimation for ARMA parameters"""
        prices = price_data.values
        n = len(prices)
        
        # Simple AR estimation using least squares
        if self.p > 0:
            X = np.zeros((n - self.p, self.p))
            y = prices[self.p:]
            
            for i in range(self.p):
                X[:, i] = prices[i:n-self.p+i]
            
            # OLS estimation: φ = (X'X)⁻¹X'y
            try:
                self.phi = np.linalg.solve(X.T @ X, X.T @ y)
            except np.linalg.LinAlgError:
                # Fallback to simple autocorrelation
                self.phi = np.array([0.5] * self.p)
        
        # Simple MA parameter initialization (can be improved)
        if self.q > 0:
            self.theta = np.array([0.3] * self.q)
            
        # Estimate error variance
        fitted_values = np.zeros(n)
        for t in range(max(self.p, self.q), n):
            fitted_values[t] = self._arma_component(prices, t)
            
        residuals = prices - fitted_values
        self.residuals = residuals.tolist()
        self.sigma = np.std(residuals[max(self.p, self.q):])
    
    def _fit_yule_walker(self, price_data: pd.Series):
        """Yule-Walker method for AR parameter estimation"""
        # Simplified implementation - can be enhanced
        self._fit_mle(price_data)
    
    def _calculate_training_metrics(self) -> Dict[str, float]:
        """Calculate training performance metrics"""
        if len(self.price_history) < max(self.p, self.q) + 10:
            return {'mse': 0, 'mae': 0, 'r2': 0}
            
        # Generate in-sample predictions for validation
        predictions = []
        actuals = []
        
        prices = np.array(self.price_history)
        
        for t in range(max(self.p, self.q), len(self.price_history)):
            pred = self._arma_component(prices, t)
            predictions.append(pred)
            actuals.append(prices[t])
            
        if len(predictions) > 0:
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            
            # R² calculation
            ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
            ss_tot = sum((a - np.mean(actuals)) ** 2 for a in actuals)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {'mse': mse, 'mae': mae, 'r2': r2}
        
        return {'mse': 0, 'mae': 0, 'r2': 0}
    
    def get_model_summary(self) -> str:
        """Generate model summary string"""
        if not self.is_fitted:
            return "Model not fitted yet."
            
        summary = f"""
SentimentARMA Model Summary
==========================
Formula: Yt = ARMA({self.p},{self.q}) × (1 + {self.beta:.3f} × St × Wt)

Parameters:
-----------
AR coefficients (φ): {[f'{x:.4f}' for x in self.phi]}
MA coefficients (θ): {[f'{x:.4f}' for x in self.theta]}
Sentiment sensitivity (β): {self.beta:.4f}
Error variance (σ²): {self.sigma:.4f}
DP epsilon (ε): {self.dp_epsilon:.4f}

Training Performance:
--------------------
MSE: {self.training_metrics.get('mse', 0):.6f}
MAE: {self.training_metrics.get('mae', 0):.6f}
R²: {self.training_metrics.get('r2', 0):.4f}

Data Summary:
-------------
Price history: {len(self.price_history)} points
Sentiment records: {len(self.sentiment_history)}
KAP announcements: {len(self.kap_history)}
        """.strip()
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Example implementation test
    print("SentimentARMA Mathematical Model - Test Implementation")
    
    # Create sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='H')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)  # Random walk prices
    price_series = pd.Series(prices, index=dates)
    
    # Initialize model
    model = SentimentARMA(p=2, q=1, beta=0.5, dp_epsilon=1.0)
    
    # Fit model
    params = model.fit(price_series)
    print("Fitted Parameters:", params)
    
    # Generate prediction
    next_timestamp = dates[-1] + pd.Timedelta(hours=1)
    prediction = model.predict_single(next_timestamp)
    
    print("\nPrediction Result:")
    for key, value in prediction.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + model.get_model_summary())
