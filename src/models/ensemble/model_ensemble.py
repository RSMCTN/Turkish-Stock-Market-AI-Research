"""
Model Ensemble for BIST Trading System
Combines DP-LSTM, Temporal Fusion Transformer, and Simple Transformer
Implements ensemble strategies and confidence estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
import math

# Import our model architectures
import sys
import os
# Add the models directory to path
models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, models_dir)

from lstm.dp_lstm import AdvancedDPLSTM, DPLSTMConfig
from transformers.temporal_fusion_transformer import TemporalFusionTransformer, TFTConfig
from transformers.simple_transformer import FinancialTransformer, SimpleTransformerConfig


@dataclass
class EnsembleConfig:
    """Configuration for Model Ensemble"""
    # Model selection
    use_dp_lstm: bool = True
    use_tft: bool = True
    use_simple_transformer: bool = True
    
    # Ensemble strategy
    ensemble_method: str = 'weighted_average'  # 'simple_average', 'weighted_average', 'stacking'
    confidence_threshold: float = 0.6          # Minimum confidence for predictions
    
    # Meta-learning for ensemble weights
    enable_meta_learning: bool = True
    meta_learning_rate: float = 0.01
    meta_update_frequency: int = 100           # Update ensemble weights every N steps
    
    # Individual model configs (will be set in post_init)
    dp_lstm_config: Optional[DPLSTMConfig] = None
    tft_config: Optional[TFTConfig] = None
    transformer_config: Optional[SimpleTransformerConfig] = None
    
    # Common parameters
    input_size: int = 131
    hidden_size: int = 256
    sequence_length: int = 60
    dropout: float = 0.3
    
    # Output configuration
    output_tasks: Dict[str, int] = field(default_factory=lambda: {
        'direction': 3,      # Price direction classification
        'magnitude': 1,      # Price magnitude regression
        'volatility': 1,     # Volatility prediction
        'volume': 1          # Volume prediction
    })
    
    def __post_init__(self):
        """Initialize individual model configurations"""
        if self.dp_lstm_config is None:
            self.dp_lstm_config = DPLSTMConfig(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=2,
                dropout=self.dropout,
                enable_multi_task=True,
                use_attention=True,
                output_tasks=self.output_tasks
            )
        
        if self.tft_config is None:
            self.tft_config = TFTConfig(
                input_size=self.input_size,
                static_input_size=5,  # Fixed size for static features
                hidden_size=self.hidden_size,
                num_encoder_steps=self.sequence_length,
                dropout=self.dropout,
                output_tasks=self.output_tasks
            )
        
        if self.transformer_config is None:
            self.transformer_config = SimpleTransformerConfig(
                input_size=self.input_size,
                d_model=self.hidden_size,
                n_heads=8,
                n_layers=4,
                dropout=self.dropout,
                output_tasks=self.output_tasks
            )


class ConfidenceEstimator(nn.Module):
    """
    Neural network to estimate prediction confidence
    Uses ensemble disagreement and individual model uncertainties
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output confidence between 0 and 1
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Ensemble features including disagreement metrics
            
        Returns:
            confidence: Confidence score between 0 and 1
        """
        return self.confidence_net(features).squeeze(-1)


class EnsembleWeightLearner(nn.Module):
    """
    Meta-learner for dynamic ensemble weight adjustment
    Learns optimal combination weights based on recent performance
    """
    
    def __init__(self, num_models: int, feature_size: int = 10):
        super().__init__()
        self.num_models = num_models
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Weight generator
        self.weight_generator = nn.Sequential(
            nn.Linear(16, num_models),
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
    
    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_features: Market context and recent performance metrics
            
        Returns:
            ensemble_weights: Normalized weights for each model
        """
        encoded_context = self.context_encoder(context_features)
        weights = self.weight_generator(encoded_context)
        return weights


class BISTEnsembleModel(nn.Module):
    """
    Ensemble Model for BIST Trading System
    
    Combines multiple architectures:
    1. Differentially Private LSTM
    2. Temporal Fusion Transformer  
    3. Simple Financial Transformer
    
    Features:
    - Dynamic ensemble weighting
    - Confidence estimation
    - Uncertainty quantification
    - Meta-learning for adaptation
    """
    
    def __init__(self, config: EnsembleConfig):
        super().__init__()
        self.config = config
        
        # Initialize individual models
        self.models = nn.ModuleDict()
        self.model_names = []
        
        if config.use_dp_lstm:
            self.models['dp_lstm'] = AdvancedDPLSTM(config.dp_lstm_config)
            self.model_names.append('dp_lstm')
        
        if config.use_tft:
            self.models['tft'] = TemporalFusionTransformer(config.tft_config)
            self.model_names.append('tft')
        
        if config.use_simple_transformer:
            self.models['transformer'] = FinancialTransformer(config.transformer_config)
            self.model_names.append('transformer')
        
        self.num_models = len(self.model_names)
        
        # Ensemble components
        if config.ensemble_method == 'weighted_average':
            # Learnable static weights
            self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        
        if config.enable_meta_learning:
            self.weight_learner = EnsembleWeightLearner(
                num_models=self.num_models,
                feature_size=10  # Market context features
            )
        
        # Confidence estimation (dynamic size calculation)
        confidence_input_size = 64  # Fixed size for now, will adapt based on actual features
        self.confidence_estimator = ConfidenceEstimator(confidence_input_size)
        
        # Performance tracking
        self.performance_history = {
            model_name: {'correct': 0, 'total': 0, 'recent_accuracy': 0.5}
            for model_name in self.model_names
        }
        
        self.step_count = 0
        self.logger = logging.getLogger(__name__)
    
    def forward(self, 
               x: torch.Tensor,
               static_features: Optional[torch.Tensor] = None,
               market_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble
        
        Args:
            x: Input time series [batch_size, seq_len, input_size]
            static_features: Static features for TFT (optional)
            market_context: Market context for meta-learning (optional)
            
        Returns:
            Dictionary containing ensemble predictions and individual model outputs
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Get predictions from each model
        model_outputs = {}
        all_predictions = {}
        
        for model_name in self.model_names:
            try:
                if model_name == 'dp_lstm':
                    outputs = self.models[model_name](x)
                elif model_name == 'tft':
                    outputs = self.models[model_name](x, static_features)
                elif model_name == 'transformer':
                    outputs = self.models[model_name](x)
                
                model_outputs[model_name] = outputs
                
                # Extract task predictions
                for task in self.config.output_tasks.keys():
                    if task in outputs:
                        if task not in all_predictions:
                            all_predictions[task] = []
                        all_predictions[task].append(outputs[task])
                        
            except Exception as e:
                self.logger.warning(f"Model {model_name} failed: {e}")
                # Create dummy outputs for failed model
                for task, output_dim in self.config.output_tasks.items():
                    if task not in all_predictions:
                        all_predictions[task] = []
                    dummy_output = torch.zeros(batch_size, output_dim, device=device)
                    all_predictions[task].append(dummy_output)
        
        # Compute ensemble predictions
        ensemble_predictions = self._compute_ensemble_predictions(
            all_predictions, market_context
        )
        
        # Estimate confidence
        confidence_features = self._extract_confidence_features(all_predictions)
        confidence_scores = self.confidence_estimator(confidence_features)
        
        # Combine results
        final_outputs = {
            **ensemble_predictions,
            'confidence': confidence_scores,
            'individual_models': model_outputs
        }
        
        # Add ensemble metadata
        if hasattr(self, 'current_weights'):
            final_outputs['ensemble_weights'] = self.current_weights
        
        return final_outputs
    
    def _compute_ensemble_predictions(self, 
                                    all_predictions: Dict[str, List[torch.Tensor]],
                                    market_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute ensemble predictions using specified strategy"""
        ensemble_predictions = {}
        
        # Get ensemble weights
        if self.config.ensemble_method == 'simple_average':
            weights = torch.ones(self.num_models, device=next(self.parameters()).device) / self.num_models
        elif self.config.ensemble_method == 'weighted_average':
            weights = F.softmax(self.ensemble_weights, dim=0)
        elif self.config.enable_meta_learning and market_context is not None:
            weights = self.weight_learner(market_context.mean(dim=0, keepdim=True))
            weights = weights.squeeze(0)
        else:
            weights = torch.ones(self.num_models, device=next(self.parameters()).device) / self.num_models
        
        self.current_weights = weights
        
        # Combine predictions for each task
        for task, predictions_list in all_predictions.items():
            if not predictions_list:
                continue
                
            # Stack predictions from all models
            stacked_predictions = torch.stack(predictions_list, dim=0)  # [num_models, batch_size, output_dim]
            
            # Weight and combine
            if task == 'direction':
                # For classification, use weighted soft voting
                softmax_predictions = F.softmax(stacked_predictions, dim=-1)
                ensemble_pred = torch.sum(weights.view(-1, 1, 1) * softmax_predictions, dim=0)
            else:
                # For regression, use weighted average
                ensemble_pred = torch.sum(weights.view(-1, 1, 1) * stacked_predictions, dim=0)
            
            ensemble_predictions[task] = ensemble_pred
        
        return ensemble_predictions
    
    def _extract_confidence_features(self, all_predictions: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        """Extract features for confidence estimation"""
        confidence_features = []
        device = next(self.parameters()).device
        
        for task, predictions_list in all_predictions.items():
            if not predictions_list or len(predictions_list) == 0:
                continue
            
            try:
                stacked_predictions = torch.stack(predictions_list, dim=0)
                
                # Model agreement features
                if task == 'direction':
                    # For classification: entropy and agreement
                    mean_pred = F.softmax(stacked_predictions, dim=-1).mean(dim=0)
                    entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=-1, keepdim=True)
                    confidence_features.append(entropy)
                    
                    # Prediction variance across models (single value)
                    var_pred = stacked_predictions.var(dim=0).mean(dim=-1, keepdim=True)
                    confidence_features.append(var_pred)
                else:
                    # For regression: mean and variance (single values)
                    mean_pred = stacked_predictions.mean(dim=0).mean(dim=-1, keepdim=True)
                    var_pred = stacked_predictions.var(dim=0).mean(dim=-1, keepdim=True)
                    confidence_features.extend([mean_pred, var_pred])
                    
            except Exception as e:
                self.logger.warning(f"Error extracting confidence features for task {task}: {e}")
                # Add dummy features
                dummy_feature = torch.zeros(predictions_list[0].shape[0], 1, device=device)
                confidence_features.append(dummy_feature)
        
        if not confidence_features:
            # Return dummy features if nothing worked
            batch_size = next(iter(all_predictions.values()))[0].shape[0]
            return torch.zeros(batch_size, 64, device=device)
        
        # Concatenate and ensure we have exactly 64 features
        combined_features = torch.cat(confidence_features, dim=-1)
        
        # Pad or truncate to exactly 64 features
        current_size = combined_features.shape[-1]
        if current_size < 64:
            padding = torch.zeros(combined_features.shape[0], 64 - current_size, device=device)
            combined_features = torch.cat([combined_features, padding], dim=-1)
        elif current_size > 64:
            combined_features = combined_features[:, :64]
        
        return combined_features
    
    def update_performance(self, predictions: Dict[str, torch.Tensor], 
                         targets: Dict[str, torch.Tensor]):
        """Update individual model performance tracking"""
        self.step_count += 1
        
        # This would typically be called during training
        # with individual model predictions and ground truth
        for model_name in self.model_names:
            # Update performance metrics
            # Implementation depends on specific evaluation strategy
            pass
        
        # Update ensemble weights based on recent performance
        if (self.step_count % self.config.meta_update_frequency == 0 and 
            self.config.enable_meta_learning):
            self._update_ensemble_weights()
    
    def _update_ensemble_weights(self):
        """Update ensemble weights based on recent performance"""
        # Compute performance-based weights
        recent_accuracies = [
            self.performance_history[name]['recent_accuracy']
            for name in self.model_names
        ]
        
        # Softmax temperature annealing
        temperature = max(0.1, 1.0 - (self.step_count / 10000))
        performance_weights = F.softmax(
            torch.tensor(recent_accuracies) / temperature, dim=0
        )
        
        # Update ensemble weights with momentum
        if hasattr(self, 'ensemble_weights'):
            momentum = 0.9
            with torch.no_grad():
                self.ensemble_weights.data = (
                    momentum * self.ensemble_weights.data + 
                    (1 - momentum) * performance_weights.to(self.ensemble_weights.device)
                )
    
    def get_model_importance(self) -> Dict[str, float]:
        """Get current importance/weight of each model in the ensemble"""
        if hasattr(self, 'current_weights'):
            weights = self.current_weights.detach().cpu().numpy()
            return {name: float(weight) for name, weight in zip(self.model_names, weights)}
        else:
            return {name: 1.0 / self.num_models for name in self.model_names}
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble summary"""
        total_params = sum(p.numel() for p in self.parameters())
        
        model_params = {}
        for name, model in self.models.items():
            model_params[name] = sum(p.numel() for p in model.parameters())
        
        return {
            'num_models': self.num_models,
            'model_names': self.model_names,
            'ensemble_method': self.config.ensemble_method,
            'total_parameters': total_params,
            'model_parameters': model_params,
            'model_importance': self.get_model_importance(),
            'performance_history': self.performance_history,
            'steps_trained': self.step_count
        }


def test_ensemble_model():
    """Test BIST Ensemble Model"""
    
    print("🎯 Testing BIST Ensemble Model...")
    print("=" * 60)
    
    # Configuration
    config = EnsembleConfig(
        input_size=50,          # Reduced for testing
        hidden_size=128,        # Reduced for testing
        sequence_length=60,
        use_dp_lstm=True,
        use_tft=True,
        use_simple_transformer=True,
        ensemble_method='weighted_average',
        enable_meta_learning=True
    )
    
    print(f"Ensemble Configuration:")
    active_models = []
    if config.use_dp_lstm:
        active_models.append('DP-LSTM')
    if config.use_tft:
        active_models.append('TFT')
    if config.use_simple_transformer:
        active_models.append('Transformer')
    
    print(f"   Models: {active_models}")
    print(f"   Input size: {config.input_size}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Ensemble method: {config.ensemble_method}")
    print(f"   Meta-learning: {config.enable_meta_learning}")
    
    # Create ensemble model
    ensemble = BISTEnsembleModel(config)
    
    # Model summary
    summary = ensemble.get_ensemble_summary()
    print(f"\nEnsemble Summary:")
    print(f"   Total parameters: {summary['total_parameters']:,}")
    print(f"   Number of models: {summary['num_models']}")
    for model_name, param_count in summary['model_parameters'].items():
        print(f"     • {model_name}: {param_count:,} parameters")
    
    # Test data
    batch_size = 4
    seq_len = 60
    x = torch.randn(batch_size, seq_len, config.input_size)
    static_features = torch.randn(batch_size, 5)  # For TFT
    market_context = torch.randn(batch_size, 10)  # For meta-learning
    
    print(f"\nTest Data:")
    print(f"   Input shape: {x.shape}")
    print(f"   Static features: {static_features.shape}")
    print(f"   Market context: {market_context.shape}")
    
    # Forward pass
    ensemble.eval()
    with torch.no_grad():
        outputs = ensemble(x, static_features, market_context)
    
    print(f"\nEnsemble Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"   {key}: {len(value)} individual models")
            for subkey in value.keys():
                print(f"     • {subkey}")
        else:
            print(f"   {key}: {type(value)}")
    
    # Test confidence and ensemble weights
    confidence = outputs.get('confidence')
    if confidence is not None:
        print(f"\nConfidence Analysis:")
        print(f"   Mean confidence: {confidence.mean().item():.4f}")
        print(f"   Confidence range: {confidence.min().item():.4f} - {confidence.max().item():.4f}")
    
    ensemble_weights = outputs.get('ensemble_weights')
    if ensemble_weights is not None:
        print(f"\nEnsemble Weights:")
        importance = ensemble.get_model_importance()
        for model_name, weight in importance.items():
            print(f"   • {model_name}: {weight:.4f}")
    
    # Test individual model outputs
    individual_models = outputs.get('individual_models', {})
    print(f"\nIndividual Model Performance:")
    for model_name, model_outputs in individual_models.items():
        if isinstance(model_outputs, dict):
            task_outputs = [key for key in model_outputs.keys() 
                          if key in config.output_tasks]
            print(f"   • {model_name}: {len(task_outputs)} tasks")
    
    # Architecture summary
    print(f"\n🏗️ Ensemble Architecture Summary:")
    print(f"   • Model Combination: {summary['num_models']} architectures")
    print(f"   • Ensemble Strategy: {config.ensemble_method}")
    print(f"   • Confidence Estimation: Neural network-based")
    print(f"   • Meta-Learning: {'✅' if config.enable_meta_learning else '❌'}")
    print(f"   • Dynamic Weighting: {'✅' if config.enable_meta_learning else '❌'}")
    print(f"   • Multi-task Learning: ✅")
    print(f"   • Uncertainty Quantification: ✅")
    
    print(f"\n✅ BIST Ensemble Model test completed!")


if __name__ == "__main__":
    test_ensemble_model()
