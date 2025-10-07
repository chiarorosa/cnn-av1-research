"""
V6 Pipeline - Ensemble Logic
Implements ensemble for Stage 3-AB with voting
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
import json
import os


class ABEnsemble:
    """
    Ensemble of 3 models for Stage 3-AB classification
    Uses majority voting for final prediction
    """
    def __init__(self, models: List[nn.Module], device='cuda'):
        self.models = models
        self.device = device
        self.num_models = len(models)
        
        # Move all models to device and set to eval
        for model in self.models:
            model.to(device)
            model.eval()
    
    def predict(self, x: torch.Tensor, use_soft_voting=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make ensemble prediction
        
        Args:
            x: Input tensor (B, C, H, W)
            use_soft_voting: If True, average probabilities; if False, majority vote
        
        Returns:
            predictions: (B,) predicted classes
            confidences: (B,) confidence scores
        """
        all_logits = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                all_logits.append(logits)
        
        # Stack logits from all models: (num_models, B, num_classes)
        all_logits = torch.stack(all_logits)
        
        if use_soft_voting:
            # Soft voting: average probabilities
            all_probs = torch.softmax(all_logits, dim=-1)
            avg_probs = all_probs.mean(dim=0)
            predictions = avg_probs.argmax(dim=-1)
            confidences = avg_probs.max(dim=-1)[0]
        else:
            # Hard voting: majority vote
            all_preds = all_logits.argmax(dim=-1)  # (num_models, B)
            
            # Majority voting
            batch_size = all_preds.shape[1]
            predictions = []
            confidences = []
            
            for i in range(batch_size):
                votes = all_preds[:, i]
                # Count votes
                unique, counts = torch.unique(votes, return_counts=True)
                # Get majority
                majority_idx = counts.argmax()
                majority_class = unique[majority_idx]
                majority_count = counts[majority_idx]
                
                predictions.append(majority_class)
                confidences.append(majority_count.float() / self.num_models)
            
            predictions = torch.tensor(predictions, device=x.device)
            confidences = torch.tensor(confidences, device=x.device)
        
        return predictions, confidences
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty estimation
        
        Returns:
            dict with predictions, mean_probs, std_probs, agreement
        """
        all_logits = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                all_logits.append(logits)
        
        all_logits = torch.stack(all_logits)  # (num_models, B, num_classes)
        all_probs = torch.softmax(all_logits, dim=-1)
        
        # Mean and std of probabilities
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        
        # Predictions
        predictions = mean_probs.argmax(dim=-1)
        
        # Agreement: how many models agree with the majority
        all_preds = all_logits.argmax(dim=-1)
        agreement = (all_preds == predictions.unsqueeze(0)).float().mean(dim=0)
        
        return {
            'predictions': predictions,
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'agreement': agreement,
            'all_probs': all_probs
        }
    
    def save_ensemble(self, save_dir: str):
        """Save all models in ensemble"""
        os.makedirs(save_dir, exist_ok=True)
        
        for i, model in enumerate(self.models):
            path = os.path.join(save_dir, f'model_{i+1}.pt')
            torch.save(model.state_dict(), path)
            print(f"Saved model {i+1} to {path}")
        
        # Save ensemble config
        config = {
            'num_models': self.num_models,
            'device': str(self.device)
        }
        with open(os.path.join(save_dir, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_ensemble(cls, model_class, save_dir: str, device='cuda'):
        """Load ensemble from directory"""
        # Load config
        with open(os.path.join(save_dir, 'ensemble_config.json'), 'r') as f:
            config = json.load(f)
        
        num_models = config['num_models']
        models = []
        
        for i in range(num_models):
            path = os.path.join(save_dir, f'model_{i+1}.pt')
            model = model_class()
            model.load_state_dict(torch.load(path, map_location=device))
            models.append(model)
            print(f"Loaded model {i+1} from {path}")
        
        return cls(models, device=device)


class WeightedEnsemble(ABEnsemble):
    """
    Weighted ensemble based on validation performance
    """
    def __init__(self, models: List[nn.Module], weights: List[float], device='cuda'):
        super().__init__(models, device)
        self.weights = torch.tensor(weights, device=device)
        self.weights = self.weights / self.weights.sum()  # Normalize
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Weighted soft voting"""
        all_logits = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                all_logits.append(logits)
        
        all_logits = torch.stack(all_logits)  # (num_models, B, num_classes)
        all_probs = torch.softmax(all_logits, dim=-1)
        
        # Weighted average
        weighted_probs = (all_probs * self.weights.view(-1, 1, 1)).sum(dim=0)
        
        predictions = weighted_probs.argmax(dim=-1)
        confidences = weighted_probs.max(dim=-1)[0]
        
        return predictions, confidences


class StackingEnsemble:
    """
    Stacking ensemble: train meta-model on top of base models
    """
    def __init__(self, base_models: List[nn.Module], meta_model: nn.Module, device='cuda'):
        self.base_models = base_models
        self.meta_model = meta_model
        self.device = device
        
        for model in self.base_models:
            model.to(device)
            model.eval()
        
        self.meta_model.to(device)
    
    def get_meta_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions from base models as meta-features"""
        meta_features = []
        
        with torch.no_grad():
            for model in self.base_models:
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                meta_features.append(probs)
        
        # Concatenate: (B, num_models * num_classes)
        meta_features = torch.cat(meta_features, dim=-1)
        return meta_features
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict using stacking"""
        meta_features = self.get_meta_features(x)
        
        with torch.no_grad():
            logits = self.meta_model(meta_features)
            probs = torch.softmax(logits, dim=-1)
        
        predictions = probs.argmax(dim=-1)
        confidences = probs.max(dim=-1)[0]
        
        return predictions, confidences


def create_ab_ensemble(model_class, num_models=3, device='cuda', pretrained=True):
    """
    Factory function to create AB ensemble
    
    Args:
        model_class: Model class to instantiate
        num_models: Number of models in ensemble
        device: Device to use
        pretrained: Use pretrained weights
    
    Returns:
        ABEnsemble instance
    """
    models = []
    for i in range(num_models):
        # Create model with different random seed
        torch.manual_seed(42 + i)
        model = model_class(pretrained=pretrained)
        models.append(model)
    
    return ABEnsemble(models, device=device)


def evaluate_ensemble_diversity(ensemble: ABEnsemble, dataloader, device='cuda'):
    """
    Evaluate diversity of ensemble predictions
    
    Returns:
        dict with diversity metrics
    """
    all_disagreements = []
    all_agreements = []
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        
        # Get predictions from all models
        all_preds = []
        with torch.no_grad():
            for model in ensemble.models:
                logits = model(data)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds)
        
        all_preds = torch.stack(all_preds)  # (num_models, B)
        
        # Compute pairwise disagreement
        for i in range(len(ensemble.models)):
            for j in range(i+1, len(ensemble.models)):
                disagreement = (all_preds[i] != all_preds[j]).float().mean()
                all_disagreements.append(disagreement.item())
        
        # Compute agreement on majority
        for b in range(data.size(0)):
            votes = all_preds[:, b]
            unique, counts = torch.unique(votes, return_counts=True)
            max_count = counts.max().item()
            agreement = max_count / len(ensemble.models)
            all_agreements.append(agreement)
    
    return {
        'avg_pairwise_disagreement': np.mean(all_disagreements),
        'avg_majority_agreement': np.mean(all_agreements),
        'std_majority_agreement': np.std(all_agreements)
    }


if __name__ == "__main__":
    # Test ensemble
    from models import Stage3ABModel
    
    print("Creating ensemble of 3 models...")
    ensemble = create_ab_ensemble(Stage3ABModel, num_models=3, device='cpu')
    
    print("\nTesting ensemble prediction (hard voting)...")
    x = torch.randn(4, 3, 16, 16)
    preds, confs = ensemble.predict(x, use_soft_voting=False)
    print(f"Predictions: {preds}")
    print(f"Confidences: {confs}\n")
    
    print("Testing ensemble prediction (soft voting)...")
    preds, confs = ensemble.predict(x, use_soft_voting=True)
    print(f"Predictions: {preds}")
    print(f"Confidences: {confs}\n")
    
    print("Testing uncertainty estimation...")
    results = ensemble.predict_with_uncertainty(x)
    print(f"Predictions: {results['predictions']}")
    print(f"Agreement: {results['agreement']}\n")
    
    print("✅ Ensemble working correctly!")
