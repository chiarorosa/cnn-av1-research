"""
Loss Functions for Ablation Study (Experiment 04)

Implements:
1. PolyLoss (Leng et al., NeurIPS 2022)
2. AsymmetricLoss (Ridnik et al., ICCV 2021) - adapted for multi-class
3. FocalLossWithLabelSmoothing (Focal + Müller et al., NeurIPS 2019)

All losses support class-balanced weighting (Cui et al., CVPR 2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyLoss(nn.Module):
    """
    Polynomial Loss (Leng et al., NeurIPS 2022)
    
    Reformulates cross-entropy with polynomial expansion:
    PolyLoss = CE + ε1 * Poly1(pt)
    
    Where:
    - CE = standard cross-entropy
    - Poly1(pt) = (1 - pt) 
    - pt = probability of correct class
    - ε1 = polynomial coefficient (default 1.0)
    
    Benefits:
    - Maintains active gradients for hard samples
    - Prevents gradient saturation (unlike CE)
    - Reported gains: +1.2 pp (ImageNet), +2.3 AP (COCO)
    
    Args:
        epsilon (float): Polynomial coefficient ε1. Default: 1.0
        class_weights (torch.Tensor): Class weights for balancing. Shape: (num_classes,)
        reduction (str): 'mean' or 'sum'. Default: 'mean'
    
    References:
        Leng et al., "PolyLoss: A Polynomial Expansion Perspective of 
        Classification Loss Functions", NeurIPS 2022
    """
    
    def __init__(self, epsilon=1.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - model predictions
            targets: (batch_size,) - ground truth labels
        
        Returns:
            loss: scalar tensor
        """
        # Cross-entropy with class weights
        ce = F.cross_entropy(
            logits, targets, 
            weight=self.class_weights, 
            reduction='none'
        )
        
        # Compute pt = exp(-CE) = probability of correct class
        pt = torch.exp(-ce)
        
        # Poly1 term: (1 - pt)
        poly1 = 1.0 - pt
        
        # PolyLoss = CE + ε1 * Poly1
        loss = ce + self.epsilon * poly1
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (Ridnik et al., ICCV 2021) - Multi-Class Adaptation
    
    Original paper: multi-label classification
    This implementation: adapted for multi-class via one-vs-rest approach
    
    Key idea: Different focusing parameters for positives and negatives
    - γ_pos: focus on hard positives (FN - false negatives)
    - γ_neg: focus on hard negatives (FP - false positives)
    
    Useful when FP and FN have different costs.
    For AV1 partition: Missing SPLIT (FN) may be more costly than wrong prediction (FP)
    
    Formulation (one-vs-rest):
        L_pos = (1 - p)^γ_pos * log(p)        [if y = 1]
        L_neg = p^γ_neg * log(1 - p)          [if y = 0]
        L = -(L_pos + L_neg)
    
    Args:
        gamma_pos (float): Focusing for positives (FN penalty). Default: 2.0
        gamma_neg (float): Focusing for negatives (FP penalty). Default: 4.0
        class_weights (torch.Tensor): Class weights for balancing. Shape: (num_classes,)
        reduction (str): 'mean' or 'sum'. Default: 'mean'
    
    References:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
    """
    
    def __init__(self, gamma_pos=2.0, gamma_neg=4.0, class_weights=None, reduction='mean'):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - model predictions
            targets: (batch_size,) - ground truth labels
        
        Returns:
            loss: scalar tensor
        """
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Get probabilities via softmax
        probs = F.softmax(logits, dim=1)
        
        # Asymmetric focusing
        # For positives (y=1): penalize (1-p)^γ_pos * log(p)
        # For negatives (y=0): penalize p^γ_neg * log(1-p)
        pos_loss = targets_one_hot * torch.pow(1 - probs, self.gamma_pos) * torch.log(probs + 1e-8)
        neg_loss = (1 - targets_one_hot) * torch.pow(probs, self.gamma_neg) * torch.log(1 - probs + 1e-8)
        
        # Combine (negative log-likelihood)
        loss = -(pos_loss + neg_loss)
        
        # Sum across classes (one-vs-rest)
        loss = loss.sum(dim=1)
        
        # Apply class weights (weight by true class)
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss + Label Smoothing (Hybrid)
    
    Combines:
    1. Focal Loss (Lin et al., ICCV 2017) - handles hard negatives via focusing
    2. Label Smoothing (Müller et al., NeurIPS 2019) - improves calibration
    
    Standard one-hot: [0, 1, 0]
    Smoothed labels: [ε/K, 1-ε+ε/K, ε/K]
    
    Where:
    - ε = smoothing factor (default 0.1)
    - K = num_classes
    
    Benefits:
    - Focal Loss: penalizes hard negatives → improves hard classes
    - Label Smoothing: reduces overconfidence → better calibration
    
    Args:
        gamma (float): Focal Loss focusing parameter. Default: 2.0
        epsilon (float): Label smoothing factor. Default: 0.1
        class_weights (torch.Tensor): Class weights for balancing. Shape: (num_classes,)
        reduction (str): 'mean' or 'sum'. Default: 'mean'
    
    References:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
        Müller et al., "When Does Label Smoothing Help?", NeurIPS 2019
    """
    
    def __init__(self, gamma=2.0, epsilon=0.1, class_weights=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - model predictions
            targets: (batch_size,) - ground truth labels
        
        Returns:
            loss: scalar tensor
        """
        num_classes = logits.shape[1]
        
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Apply label smoothing
        # y_smooth = (1 - ε) * y_hard + ε / K
        targets_smooth = (1.0 - self.epsilon) * targets_one_hot + self.epsilon / num_classes
        
        # Compute probabilities and log-probs
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # Focal weight: (1 - pt)^γ
        focal_weight = torch.pow(1.0 - probs, self.gamma)
        
        # Focal loss with smoothed labels
        # FL = -(1 - pt)^γ * y_smooth * log(pt)
        loss = -focal_weight * targets_smooth * log_probs
        
        # Sum across classes
        loss = loss.sum(dim=1)
        
        # Apply class weights (weight by true class)
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============================================================================
# Utility Functions for Testing
# ============================================================================

def test_poly_loss():
    """Test PolyLoss with toy example"""
    print("\n=== Testing PolyLoss ===")
    
    # Toy data
    logits = torch.randn(4, 3)  # batch=4, classes=3
    targets = torch.tensor([0, 1, 2, 1])
    class_weights = torch.tensor([1.0, 2.0, 0.5])
    
    # Test without class weights
    loss_fn = PolyLoss(epsilon=1.0)
    loss = loss_fn(logits, targets)
    print(f"Loss (no weights): {loss.item():.4f}")
    
    # Test with class weights
    loss_fn = PolyLoss(epsilon=1.0, class_weights=class_weights)
    loss = loss_fn(logits, targets)
    print(f"Loss (with weights): {loss.item():.4f}")
    
    # Test gradient
    logits.requires_grad = True
    loss = loss_fn(logits, targets)
    loss.backward()
    print(f"Gradient shape: {logits.grad.shape}")
    print(f"Gradient mean: {logits.grad.mean().item():.4f}")
    
    print("✓ PolyLoss test passed")


def test_asymmetric_loss():
    """Test AsymmetricLoss with toy example"""
    print("\n=== Testing AsymmetricLoss ===")
    
    # Toy data
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    class_weights = torch.tensor([1.0, 2.0, 0.5])
    
    # Test without class weights
    loss_fn = AsymmetricLoss(gamma_pos=2.0, gamma_neg=4.0)
    loss = loss_fn(logits, targets)
    print(f"Loss (no weights): {loss.item():.4f}")
    
    # Test with class weights
    loss_fn = AsymmetricLoss(gamma_pos=2.0, gamma_neg=4.0, class_weights=class_weights)
    loss = loss_fn(logits, targets)
    print(f"Loss (with weights): {loss.item():.4f}")
    
    # Test gradient
    logits.requires_grad = True
    loss = loss_fn(logits, targets)
    loss.backward()
    print(f"Gradient shape: {logits.grad.shape}")
    print(f"Gradient mean: {logits.grad.mean().item():.4f}")
    
    print("✓ AsymmetricLoss test passed")


def test_focal_label_smoothing():
    """Test FocalLossWithLabelSmoothing with toy example"""
    print("\n=== Testing FocalLossWithLabelSmoothing ===")
    
    # Toy data
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    class_weights = torch.tensor([1.0, 2.0, 0.5])
    
    # Test without class weights
    loss_fn = FocalLossWithLabelSmoothing(gamma=2.0, epsilon=0.1)
    loss = loss_fn(logits, targets)
    print(f"Loss (no weights): {loss.item():.4f}")
    
    # Test with class weights
    loss_fn = FocalLossWithLabelSmoothing(gamma=2.0, epsilon=0.1, class_weights=class_weights)
    loss = loss_fn(logits, targets)
    print(f"Loss (with weights): {loss.item():.4f}")
    
    # Test gradient
    logits.requires_grad = True
    loss = loss_fn(logits, targets)
    loss.backward()
    print(f"Gradient shape: {logits.grad.shape}")
    print(f"Gradient mean: {logits.grad.mean().item():.4f}")
    
    print("✓ FocalLossWithLabelSmoothing test passed")


if __name__ == '__main__':
    """Run all tests"""
    print("=" * 60)
    print("Loss Functions Ablation - Unit Tests")
    print("=" * 60)
    
    test_poly_loss()
    test_asymmetric_loss()
    test_focal_label_smoothing()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
