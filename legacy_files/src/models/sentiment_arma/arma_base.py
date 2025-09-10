"""
ARMA Base Mathematical Implementation

Traditional ARMA(p,q) modeling foundation for the sentimentARMA framework.
Provides core autoregressive and moving average components.

Mathematical Foundation:
ARMA(p,q): Yt = c + φ₁Yt-1 + ... + φₚYt-p + εt + θ₁εt-1 + ... + θₘεt-m

Where:
- Yt: Observed value at time t
- c: Constant term
- φᵢ: Autoregressive coefficients
- θⱼ: Moving average coefficients
- εt: White noise error term
- p: AR order, q: MA order
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from scipy import optimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ARMABase:
    """
    Traditional ARMA(p,q) model implementation
    
    Provides the mathematical foundation for autoregressive and moving average
    modeling that will be enhanced with sentiment analysis in sentimentARMA.
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize ARMA base model
        
        Args:
            p: Autoregressive order (number of lag observations)
            q: Moving average order (number of lag forecast errors)
        """
        self.p = p  # AR order
        self.q = q  # MA order
        
        # Model parameters
        self.const = 0.0  # Constant term
        self.phi = np.zeros(p)  # AR coefficients
        self.theta = np.zeros(q)  # MA coefficients
        self.sigma2 = 1.0  # Error variance
        
        # Model state
        self.is_fitted = False
        self.residuals = []
        self.fitted_values = []
        self.training_data = None
        self.loglikelihood = None
        self.aic = None
        self.bic = None
        
    def _validate_order(self):
        """Validate ARMA orders"""
        if self.p < 0 or self.q < 0:
            raise ValueError("ARMA orders must be non-negative")
        if self.p + self.q == 0:
            raise ValueError("At least one of p or q must be positive")
            
    def _prepare_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data matrices for ARMA estimation
        
        Args:
            data: Time series data
            
        Returns:
            (X, y) matrices for regression
        """
        n = len(data)
        max_lag = max(self.p, self.q)
        
        if n <= max_lag:
            raise ValueError(f"Data length ({n}) must be greater than max(p,q) = {max_lag}")
        
        # Create lagged matrices
        y = data[max_lag:]
        X = np.ones((len(y), 1))  # Constant term
        
        # Add AR terms: Yt-1, Yt-2, ..., Yt-p
        for i in range(1, self.p + 1):
            lag_data = data[max_lag-i:-i]
            X = np.column_stack([X, lag_data])
            
        return X, y
    
    def _likelihood_function(self, params: np.ndarray, data: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for ARMA model
        
        Args:
            params: Model parameters [const, φ₁,...,φₚ, θ₁,...,θₘ, σ²]
            data: Time series data
            
        Returns:
            Negative log-likelihood value
        """
        try:
            # Extract parameters
            const = params[0]
            phi = params[1:1+self.p] if self.p > 0 else np.array([])
            theta = params[1+self.p:1+self.p+self.q] if self.q > 0 else np.array([])
            sigma2 = abs(params[-1])  # Ensure positive variance
            
            n = len(data)
            max_lag = max(self.p, self.q, 1)
            
            # Initialize
            residuals = np.zeros(n)
            fitted = np.zeros(n)
            
            # Calculate residuals and fitted values
            for t in range(max_lag, n):
                # AR component
                ar_term = const
                for i in range(self.p):
                    if t - 1 - i >= 0:
                        ar_term += phi[i] * data[t - 1 - i]
                
                # MA component  
                ma_term = 0.0
                for i in range(self.q):
                    if t - 1 - i >= 0:
                        ma_term += theta[i] * residuals[t - 1 - i]
                
                fitted[t] = ar_term + ma_term
                residuals[t] = data[t] - fitted[t]
            
            # Calculate log-likelihood
            valid_residuals = residuals[max_lag:]
            n_valid = len(valid_residuals)
            
            if n_valid <= 0 or sigma2 <= 0:
                return np.inf
                
            ll = -0.5 * n_valid * np.log(2 * np.pi * sigma2)
            ll -= 0.5 * np.sum(valid_residuals**2) / sigma2
            
            return -ll  # Return negative for minimization
            
        except Exception:
            return np.inf
    
    def fit(self, data: Union[pd.Series, np.ndarray], method: str = 'mle') -> Dict[str, float]:
        """
        Fit ARMA model using Maximum Likelihood Estimation
        
        Args:
            data: Time series data
            method: Estimation method ('mle', 'ols', 'yule_walker')
            
        Returns:
            Dictionary with fitted parameters and statistics
        """
        self._validate_order()
        
        # Convert to numpy array
        if isinstance(data, pd.Series):
            self.training_data = data
            data_array = data.values
        else:
            data_array = np.array(data)
            self.training_data = pd.Series(data_array)
        
        if method == 'mle':
            return self._fit_mle(data_array)
        elif method == 'ols':
            return self._fit_ols(data_array)
        elif method == 'yule_walker':
            return self._fit_yule_walker(data_array)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _fit_mle(self, data: np.ndarray) -> Dict[str, float]:
        """Maximum Likelihood Estimation"""
        n_params = 1 + self.p + self.q + 1  # const + AR + MA + sigma2
        
        # Initial parameter guess
        initial_params = np.zeros(n_params)
        initial_params[0] = np.mean(data)  # Constant
        
        # AR initial values (small positive values)
        if self.p > 0:
            initial_params[1:1+self.p] = np.random.normal(0, 0.1, self.p)
            
        # MA initial values  
        if self.q > 0:
            initial_params[1+self.p:1+self.p+self.q] = np.random.normal(0, 0.1, self.q)
            
        initial_params[-1] = np.var(data)  # Initial variance
        
        # Bounds for stability
        bounds = []
        bounds.append((-np.inf, np.inf))  # Constant
        
        # AR bounds (for stationarity)
        for _ in range(self.p):
            bounds.append((-0.99, 0.99))
            
        # MA bounds (for invertibility)
        for _ in range(self.q):
            bounds.append((-0.99, 0.99))
            
        bounds.append((1e-6, np.inf))  # Variance must be positive
        
        try:
            # Optimize
            result = optimize.minimize(
                self._likelihood_function,
                initial_params,
                args=(data,),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                # Extract fitted parameters
                self.const = result.x[0]
                self.phi = result.x[1:1+self.p] if self.p > 0 else np.array([])
                self.theta = result.x[1+self.p:1+self.p+self.q] if self.q > 0 else np.array([])
                self.sigma2 = abs(result.x[-1])
                
                self.loglikelihood = -result.fun
            else:
                # Fallback to OLS if MLE fails
                return self._fit_ols(data)
                
        except Exception:
            # Fallback to OLS if optimization fails
            return self._fit_ols(data)
        
        self._calculate_fitted_values_and_residuals(data)
        self._calculate_information_criteria()
        self.is_fitted = True
        
        return self._get_fit_summary()
    
    def _fit_ols(self, data: np.ndarray) -> Dict[str, float]:
        """Ordinary Least Squares estimation (for AR components)"""
        if self.p == 0:
            # Pure MA model - use simple approach
            self.const = np.mean(data)
            self.theta = np.array([0.3] * self.q)  # Simple initialization
            self.sigma2 = np.var(data)
        else:
            # Fit AR part with OLS
            X, y = self._prepare_data(data)
            
            try:
                # OLS: β = (X'X)⁻¹X'y
                beta = np.linalg.solve(X.T @ X, X.T @ y)
                
                self.const = beta[0]
                self.phi = beta[1:1+self.p] if self.p > 0 else np.array([])
                
                # Simple MA initialization
                self.theta = np.array([0.3] * self.q) if self.q > 0 else np.array([])
                
                # Calculate residual variance
                fitted = X @ beta
                residuals = y - fitted
                self.sigma2 = np.var(residuals)
                
            except np.linalg.LinAlgError:
                # Fallback values
                self.const = np.mean(data)
                self.phi = np.array([0.5] * self.p)
                self.theta = np.array([0.3] * self.q)
                self.sigma2 = np.var(data)
        
        self._calculate_fitted_values_and_residuals(data)
        self._calculate_information_criteria()
        self.is_fitted = True
        
        return self._get_fit_summary()
    
    def _fit_yule_walker(self, data: np.ndarray) -> Dict[str, float]:
        """Yule-Walker estimation (simplified)"""
        # For now, fallback to OLS
        return self._fit_ols(data)
    
    def _calculate_fitted_values_and_residuals(self, data: np.ndarray):
        """Calculate fitted values and residuals"""
        n = len(data)
        max_lag = max(self.p, self.q, 1)
        
        self.fitted_values = np.zeros(n)
        self.residuals = np.zeros(n)
        
        for t in range(max_lag, n):
            # AR component
            fitted_val = self.const
            for i in range(self.p):
                if t - 1 - i >= 0:
                    fitted_val += self.phi[i] * data[t - 1 - i]
            
            # MA component
            for i in range(self.q):
                if t - 1 - i >= 0:
                    fitted_val += self.theta[i] * self.residuals[t - 1 - i]
            
            self.fitted_values[t] = fitted_val
            self.residuals[t] = data[t] - fitted_val
    
    def _calculate_information_criteria(self):
        """Calculate AIC and BIC"""
        if self.loglikelihood is None:
            # Approximate log-likelihood
            valid_residuals = self.residuals[max(self.p, self.q, 1):]
            n = len(valid_residuals)
            if n > 0 and self.sigma2 > 0:
                self.loglikelihood = -0.5 * n * np.log(2 * np.pi * self.sigma2)
                self.loglikelihood -= 0.5 * np.sum(valid_residuals**2) / self.sigma2
        
        if self.loglikelihood is not None:
            k = 1 + self.p + self.q + 1  # Number of parameters
            n = len(self.training_data) if self.training_data is not None else 0
            
            self.aic = 2 * k - 2 * self.loglikelihood
            self.bic = k * np.log(n) - 2 * self.loglikelihood if n > 0 else None
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Generate forecasts
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        data = self.training_data.values.copy()
        residuals = self.residuals.copy()
        
        for step in range(steps):
            # AR component
            pred = self.const
            for i in range(self.p):
                if len(data) - 1 - i >= 0:
                    pred += self.phi[i] * data[-(1+i)]
            
            # MA component (residuals become 0 for multi-step forecasts)
            for i in range(self.q):
                if step == 0 and len(residuals) - 1 - i >= 0:
                    pred += self.theta[i] * residuals[-(1+i)]
            
            predictions.append(pred)
            
            # Update data for next prediction
            data = np.append(data, pred)
            residuals = np.append(residuals, 0.0)  # Unknown future residuals
        
        return np.array(predictions)
    
    def _get_fit_summary(self) -> Dict[str, float]:
        """Get fitting summary statistics"""
        valid_residuals = self.residuals[max(self.p, self.q, 1):]
        
        if len(valid_residuals) > 0:
            mse = np.mean(valid_residuals**2)
            mae = np.mean(np.abs(valid_residuals))
            rmse = np.sqrt(mse)
        else:
            mse = mae = rmse = 0.0
        
        return {
            'const': self.const,
            'phi': self.phi.tolist() if len(self.phi) > 0 else [],
            'theta': self.theta.tolist() if len(self.theta) > 0 else [],
            'sigma2': self.sigma2,
            'loglikelihood': self.loglikelihood or 0.0,
            'aic': self.aic or 0.0,
            'bic': self.bic or 0.0,
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
    
    def summary(self) -> str:
        """Generate model summary"""
        if not self.is_fitted:
            return "ARMA model not fitted."
        
        summary = f"""
ARMA({self.p},{self.q}) Model Summary
================================

Parameters:
-----------
Constant: {self.const:.6f}
"""
        
        if self.p > 0:
            summary += "AR coefficients:\n"
            for i, phi in enumerate(self.phi):
                summary += f"  φ_{i+1}: {phi:.6f}\n"
        
        if self.q > 0:
            summary += "MA coefficients:\n"
            for i, theta in enumerate(self.theta):
                summary += f"  θ_{i+1}: {theta:.6f}\n"
        
        summary += f"""
Error variance (σ²): {self.sigma2:.6f}

Model Statistics:
-----------------
Log-likelihood: {self.loglikelihood:.2f}
AIC: {self.aic:.2f}
BIC: {self.bic:.2f}
"""
        
        return summary.strip()

# Example usage
if __name__ == "__main__":
    print("ARMA Base Model - Test Implementation")
    
    # Generate sample ARMA data
    np.random.seed(42)
    n = 200
    
    # True ARMA(2,1) process
    true_phi = [0.6, -0.3]  
    true_theta = [0.4]
    true_const = 1.0
    
    # Simulate ARMA process
    data = [true_const]  # Initialize
    errors = np.random.normal(0, 1, n + 10)
    
    for t in range(1, n + 10):
        ar_term = true_const
        if t >= 1:
            ar_term += true_phi[0] * data[t-1]
        if t >= 2:  
            ar_term += true_phi[1] * data[t-2]
            
        ma_term = errors[t]
        if t >= 1:
            ma_term += true_theta[0] * errors[t-1]
            
        value = ar_term + ma_term
        data.append(value)
    
    data = np.array(data[10:])  # Remove initialization period
    
    # Fit ARMA model
    model = ARMABase(p=2, q=1)
    results = model.fit(data, method='mle')
    
    print("Fitted Parameters:")
    print(f"Constant: {results['const']:.4f} (true: {true_const})")
    print(f"AR coeff: {results['phi']} (true: {true_phi})")
    print(f"MA coeff: {results['theta']} (true: {true_theta})")
    print(f"MSE: {results['mse']:.6f}")
    
    # Generate forecasts
    forecasts = model.predict(steps=5)
    print(f"\n5-step forecasts: {forecasts}")
    
    print("\n" + model.summary())
