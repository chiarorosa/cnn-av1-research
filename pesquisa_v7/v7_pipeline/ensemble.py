"""
V7 Pipeline - Ensemble Methods
Multi-Model Voting for Hierarchical Classification

Based on:
Ahad, M.T., et al. (2024). A study on Deep Convolutional Neural Networks, 
Transfer Learning and Ensemble Model for Breast Cancer Detection.

Key concepts:
1. Diversity through different architectures or initializations
2. Soft voting (average probabilities) vs Hard voting (majority class)
3. Learnable ensemble weights
4. Stage-wise ensemble (not just final prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class SoftVotingEnsemble(nn.Module):
    """
    Soft voting ensemble (Ahad et al., 2024)
    Averages predicted probabilities from multiple models
    
    Args:
        models: List of trained models (same architecture or different)
        weights: Optional fixed weights for each model
        learnable_weights: If True, learn weights during training
    """
    def __init__(self, models: List[nn.Module], weights=None, learnable_weights=False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        if weights is None:
            # Equal weights by default
            weights = torch.ones(self.num_models) / self.num_models
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights / weights.sum()  # Normalize
        
        if learnable_weights:
            self.weights = nn.Parameter(weights)
        else:
            self.register_buffer('weights', weights)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            logits: Weighted average of model predictions [B, num_classes]
        """
        # Collect predictions from all models
        all_logits = []
        for model in self.models:
            with torch.set_grad_enabled(model.training):
                logits = model(x)
                all_logits.append(logits)
        
        # Stack: [num_models, B, num_classes]
        all_logits = torch.stack(all_logits, dim=0)
        
        # Convert to probabilities
        all_probs = F.softmax(all_logits, dim=-1)
        
        # Weighted average
        weights = F.softmax(self.weights, dim=0)  # Ensure normalized
        weights = weights.view(-1, 1, 1)  # [num_models, 1, 1]
        
        ensemble_probs = (all_probs * weights).sum(dim=0)  # [B, num_classes]
        
        # Convert back to logits (for loss computation)
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        return ensemble_logits
    
    def predict(self, x):
        """Get class predictions"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class HardVotingEnsemble(nn.Module):
    """
    Hard voting ensemble
    Takes majority vote of predicted classes
    """
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
    
    def forward(self, x):
        """
        Returns:
            votes: Tensor of shape [B, num_models] with class predictions
        """
        all_preds = []
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds)
        
        # Stack: [num_models, B]
        votes = torch.stack(all_preds, dim=0)
        return votes.t()  # [B, num_models]
    
    def predict(self, x):
        """Get majority class"""
        votes = self.forward(x)  # [B, num_models]
        # Mode along model dimension
        majority, _ = torch.mode(votes, dim=1)
        return majority


class HierarchicalEnsemble(nn.Module):
    """
    Multi-stage hierarchical ensemble
    Each stage has its own ensemble of models
    
    Inspired by Ahad et al. (2024) ensemble achieving 99.94% accuracy
    
    Args:
        stage1_models: List of Stage 1 models (binary)
        stage2_models: List of Stage 2 models (3-way)
        stage3_rect_models: List of Stage 3-RECT models
        stage3_ab_models: List of Stage 3-AB models
        voting_type: 'soft' or 'hard'
    """
    def __init__(
        self, 
        stage1_models: List[nn.Module],
        stage2_models: List[nn.Module],
        stage3_rect_models: List[nn.Module],
        stage3_ab_models: List[nn.Module],
        voting_type='soft'
    ):
        super().__init__()
        
        if voting_type == 'soft':
            self.stage1_ensemble = SoftVotingEnsemble(stage1_models, learnable_weights=True)
            self.stage2_ensemble = SoftVotingEnsemble(stage2_models, learnable_weights=True)
            self.stage3_rect_ensemble = SoftVotingEnsemble(stage3_rect_models, learnable_weights=True)
            self.stage3_ab_ensemble = SoftVotingEnsemble(stage3_ab_models, learnable_weights=True)
        else:
            self.stage1_ensemble = HardVotingEnsemble(stage1_models)
            self.stage2_ensemble = HardVotingEnsemble(stage2_models)
            self.stage3_rect_ensemble = HardVotingEnsemble(stage3_rect_models)
            self.stage3_ab_ensemble = HardVotingEnsemble(stage3_ab_models)
        
        self.voting_type = voting_type
    
    def forward(self, x, return_intermediates=False):
        """
        Full hierarchical pipeline with ensemble at each stage
        
        Returns:
            final_pred: Final partition type prediction
            intermediates: Dict with stage predictions (if requested)
        """
        intermediates = {}
        
        # Stage 1: NONE vs PARTITION
        stage1_logits = self.stage1_ensemble(x)
        stage1_pred = torch.sigmoid(stage1_logits).squeeze() > 0.5
        
        intermediates['stage1_pred'] = stage1_pred
        
        # If NONE (0), return immediately
        none_mask = ~stage1_pred
        if none_mask.all():
            return torch.zeros_like(stage1_pred, dtype=torch.long), intermediates
        
        # Stage 2: SPLIT, RECT, AB (only for PARTITION blocks)
        stage2_logits = self.stage2_ensemble(x)
        stage2_pred = torch.argmax(stage2_logits, dim=1)
        
        intermediates['stage2_pred'] = stage2_pred
        
        # Map stage 2 predictions:
        # 0: SPLIT → return 3 (PARTITION_SPLIT)
        # 1: RECT → go to Stage 3-RECT
        # 2: AB → go to Stage 3-AB
        
        final_pred = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Handle NONE
        final_pred[none_mask] = 0  # PARTITION_NONE
        
        # Handle SPLIT
        split_mask = stage1_pred & (stage2_pred == 0)
        final_pred[split_mask] = 3  # PARTITION_SPLIT
        
        # Stage 3-RECT: HORZ vs VERT
        rect_mask = stage1_pred & (stage2_pred == 1)
        if rect_mask.any():
            rect_logits = self.stage3_rect_ensemble(x[rect_mask])
            rect_pred = torch.argmax(rect_logits, dim=1)
            # 0: HORZ → 1 (PARTITION_HORZ)
            # 1: VERT → 2 (PARTITION_VERT)
            final_pred[rect_mask] = rect_pred + 1
            intermediates['stage3_rect_pred'] = rect_pred
        
        # Stage 3-AB: HORZ_A, HORZ_B, VERT_A, VERT_B
        ab_mask = stage1_pred & (stage2_pred == 2)
        if ab_mask.any():
            ab_logits = self.stage3_ab_ensemble(x[ab_mask])
            ab_pred = torch.argmax(ab_logits, dim=1)
            # 0: HORZ_A → 4
            # 1: HORZ_B → 5
            # 2: VERT_A → 6
            # 3: VERT_B → 7
            final_pred[ab_mask] = ab_pred + 4
            intermediates['stage3_ab_pred'] = ab_pred
        
        if return_intermediates:
            return final_pred, intermediates
        return final_pred


class DiverseBackboneEnsemble(nn.Module):
    """
    Ensemble with diverse backbone architectures (Ahad et al., 2024)
    
    Example from paper: DenseNet121 + InceptionV3 + ResNet18
    For v7: ResNet18 + MobileNetV2 + EfficientNet
    
    Args:
        backbones: List of different backbone architectures
        head_factory: Function to create head for each backbone
        voting_type: 'soft' or 'hard'
    """
    def __init__(self, backbones: List[nn.Module], head_factory, voting_type='soft'):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.heads = nn.ModuleList([head_factory() for _ in backbones])
        self.voting_type = voting_type
        
        if voting_type == 'soft':
            num_models = len(backbones)
            self.weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, x):
        all_logits = []
        
        for backbone, head in zip(self.backbones, self.heads):
            features = backbone(x)
            logits = head(features)
            all_logits.append(logits)
        
        all_logits = torch.stack(all_logits, dim=0)  # [num_models, B, num_classes]
        
        if self.voting_type == 'soft':
            # Soft voting
            all_probs = F.softmax(all_logits, dim=-1)
            weights = F.softmax(self.weights, dim=0).view(-1, 1, 1)
            ensemble_probs = (all_probs * weights).sum(dim=0)
            return torch.log(ensemble_probs + 1e-8)
        else:
            # Hard voting
            all_preds = torch.argmax(all_logits, dim=-1)  # [num_models, B]
            majority, _ = torch.mode(all_preds, dim=0)
            return majority


# Utility functions
def create_ensemble_from_checkpoints(checkpoint_paths: List[str], model_factory, device='cuda'):
    """
    Load multiple model checkpoints and create ensemble
    
    Args:
        checkpoint_paths: List of paths to saved model checkpoints
        model_factory: Function that creates a model instance
        device: 'cuda' or 'cpu'
    
    Returns:
        SoftVotingEnsemble with loaded models
    """
    models = []
    for path in checkpoint_paths:
        model = model_factory()
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    
    return SoftVotingEnsemble(models, learnable_weights=False)


def analyze_ensemble_diversity(models: List[nn.Module], dataloader, device='cuda'):
    """
    Analyze diversity of ensemble models (for debugging)
    
    Returns:
        metrics: Dict with diversity metrics
            - agreement: Pairwise agreement between models
            - disagreement: Percentage of samples with disagreement
    """
    all_predictions = []
    
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch['block'].to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                preds.append(pred.cpu())
        preds = torch.cat(preds, dim=0)
        all_predictions.append(preds)
    
    # Stack: [num_models, num_samples]
    all_predictions = torch.stack(all_predictions, dim=0)
    
    # Pairwise agreement
    num_models = len(models)
    agreements = []
    for i in range(num_models):
        for j in range(i+1, num_models):
            agreement = (all_predictions[i] == all_predictions[j]).float().mean().item()
            agreements.append(agreement)
    
    avg_agreement = sum(agreements) / len(agreements) if agreements else 1.0
    
    # Disagreement: at least one model differs
    mode_pred, _ = torch.mode(all_predictions, dim=0)
    disagreement = (all_predictions != mode_pred.unsqueeze(0)).any(dim=0).float().mean().item()
    
    return {
        'avg_pairwise_agreement': avg_agreement,
        'disagreement_rate': disagreement,
        'num_models': num_models
    }
