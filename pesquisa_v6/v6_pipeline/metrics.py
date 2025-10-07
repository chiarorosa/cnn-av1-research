"""
V6 Pipeline - Metrics and Evaluation Utilities
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   labels: List[str] = None) -> Dict:
    """
    Compute comprehensive classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class names
    
    Returns:
        Dictionary with metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro/Weighted averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_p),
        'weighted_recall': float(weighted_r),
        'weighted_f1': float(weighted_f1),
        'per_class': {}
    }
    
    # Per-class details
    num_classes = len(precision)
    for i in range(num_classes):
        class_name = labels[i] if labels else f'class_{i}'
        metrics['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_scores: np.ndarray = None) -> Dict:
    """
    Compute binary classification metrics with threshold analysis
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        y_scores: Prediction scores/probabilities (optional)
    
    Returns:
        Dictionary with binary metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'f1': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    # Add AUC if scores available
    if y_scores is not None:
        try:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_scores))
        except:
            metrics['auc_roc'] = None
    
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray, 
                          metric='f1') -> Tuple[float, Dict]:
    """
    Find optimal threshold for binary classification
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Prediction scores
        metric: Metric to optimize ('f1', 'precision', 'accuracy')
    
    Returns:
        (optimal_threshold, metrics_at_threshold)
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = 0.0
    best_metrics = {}
    
    for th in thresholds:
        y_pred = (y_scores >= th).astype(int)
        metrics = compute_binary_metrics(y_true, y_pred, y_scores)
        
        score = metrics[metric]
        if score > best_score:
            best_score = score
            best_threshold = th
            best_metrics = metrics
    
    return best_threshold, best_metrics


def compute_stage_metrics(stage_name: str, y_true: np.ndarray, 
                          y_pred: np.ndarray, labels: List[str]) -> Dict:
    """
    Compute stage-specific metrics
    
    Args:
        stage_name: 'stage1', 'stage2', 'stage3_rect', 'stage3_ab'
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class names
    
    Returns:
        Dictionary with stage metrics
    """
    if stage_name == 'stage1':
        # Binary: NONE (0) vs PARTITION (1)
        return compute_binary_metrics(y_true, y_pred)
    else:
        # Multi-class
        return compute_metrics(y_true, y_pred, labels)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], 
                          save_path: str = None, normalize=False):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix
        labels: Class labels
        save_path: Path to save figure
        normalize: Whether to normalize by row
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(precisions: List[float], recalls: List[float], 
                                thresholds: List[float], save_path: str = None):
    """Plot precision-recall vs threshold curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', marker='o')
    plt.plot(thresholds, recalls, label='Recall', marker='s')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


class MetricsTracker:
    """Track metrics during training"""
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """Update history with epoch metrics"""
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['train_acc'].append(train_metrics.get('accuracy', 0))
        self.history['val_acc'].append(val_metrics.get('accuracy', 0))
        self.history['train_f1'].append(train_metrics.get('f1', 0))
        self.history['val_f1'].append(val_metrics.get('f1', 0))
        self.history['learning_rate'].append(lr)
    
    def save(self, save_path: str):
        """Save history to JSON"""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {save_path}")
    
    def plot(self, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Val')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1
        axes[1, 0].plot(self.history['train_f1'], label='Train')
        axes[1, 0].plot(self.history['val_f1'], label='Val')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training curves to {save_path}")
        else:
            plt.show()
        
        plt.close()


def print_metrics_summary(metrics: Dict, stage_name: str = ""):
    """Pretty print metrics summary"""
    print(f"\n{'='*60}")
    print(f"  {stage_name} Metrics Summary")
    print(f"{'='*60}")
    
    if 'accuracy' in metrics:
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    
    if 'macro_f1' in metrics:
        print(f"  Macro F1:  {metrics['macro_f1']*100:.2f}%")
        print(f"  Macro Precision: {metrics['macro_precision']*100:.2f}%")
        print(f"  Macro Recall: {metrics['macro_recall']*100:.2f}%")
    
    if 'f1' in metrics:
        print(f"  F1:        {metrics['f1']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
    
    if 'per_class' in metrics:
        print(f"\n  Per-class metrics:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"    {class_name:12s}: F1={class_metrics['f1']*100:5.2f}%  "
                  f"P={class_metrics['precision']*100:5.2f}%  "
                  f"R={class_metrics['recall']*100:5.2f}%  "
                  f"(n={class_metrics['support']})")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test metrics
    print("Testing binary metrics...")
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_scores = np.array([0.2, 0.6, 0.8, 0.9, 0.4, 0.3, 0.85, 0.1])
    
    metrics = compute_binary_metrics(y_true, y_pred, y_scores)
    print_metrics_summary(metrics, "Stage 1 (Binary)")
    
    print("\nTesting multi-class metrics...")
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1])
    labels = ['SPLIT', 'RECT', 'AB']
    
    metrics = compute_metrics(y_true, y_pred, labels)
    print_metrics_summary(metrics, "Stage 2 (3-way)")
    
    print("\nTesting threshold optimization...")
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)
    
    best_th, best_metrics = find_optimal_threshold(y_true, y_scores, metric='f1')
    print(f"Optimal threshold: {best_th:.3f}")
    print(f"F1 at threshold: {best_metrics['f1']:.3f}")
    
    print("\n✅ All metrics working correctly!")
