"""
Script 006: Train Stage 3 AB Ensemble
Stage 3-AB: 4-way classification (HORZ_A, HORZ_B, VERT_A, VERT_B)
Strategy: Ensemble of 3 independent models with majority voting

Key Improvements (from PLANO_V6.md):
1. Ensemble of 3 Models - Different seeds and augmentation
2. Oversampling 5x - Minority classes (HORZ_A, VERT_B)
3. Mixup Augmentation (Zhang et al., 2018) - Between AB classes
4. Focal Loss (Lin et al., 2017) - Handle class imbalance
5. Heavy Data Augmentation (Shorten & Khoshgoftaar, 2019)

References:
- Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. ICLR.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation. Journal of Big Data.
- Cui, Y., et al. (2019). Class-Balanced Loss Based on Effective Number of Samples. CVPR.
- Goodfellow, I., et al. (2016). Deep Learning. MIT Press.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import json
from tqdm import tqdm
import random

# Add v6 pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "v6_pipeline"))
from data_hub import HierarchicalBlockDatasetV6, BlockRecord, build_hierarchical_dataset_v6
from models import Stage3ABModel
from augmentation import Stage3ABAugmentation
from losses import FocalLoss
from metrics import compute_metrics


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MixupLoss:
    """Mixup data augmentation with Focal Loss (Zhang et al., 2018)"""
    def __init__(self, focal_loss, alpha=0.4):
        self.focal_loss = focal_loss
        self.alpha = alpha
    
    def __call__(self, model, images, labels):
        """Apply mixup and compute loss"""
        if self.alpha > 0 and model.training:
            # Sample lambda from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Random permutation
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(images.device)
            
            # Mixup images
            mixed_images = lam * images + (1 - lam) * images[index]
            labels_a, labels_b = labels, labels[index]
            
            # Forward pass
            logits = model(mixed_images)
            
            # Mixup loss
            loss = lam * self.focal_loss(logits, labels_a) + (1 - lam) * self.focal_loss(logits, labels_b)
            return logits, loss
        else:
            # No mixup during validation or if alpha=0
            logits = model(images)
            loss = self.focal_loss(logits, labels)
            return logits, loss


def create_oversampling_weights(labels, oversample_factor=5.0):
    """
    Create sampling weights for oversampling minority classes
    HORZ_A (label=4) and VERT_B (label=7) are minorities (~8% each)
    HORZ_B (label=5) and VERT_A (label=6) are majorities (~42% each)
    """
    labels_np = labels if isinstance(labels, np.ndarray) else labels.numpy()
    unique, counts = np.unique(labels_np, return_counts=True)
    
    # Identify minority classes (HORZ_A=4, VERT_B=7)
    minority_classes = [4, 7]
    
    # Create weights: 5x for minorities, 1x for majorities
    weights = np.ones(len(labels_np), dtype=np.float32)
    for cls in minority_classes:
        mask = labels_np == cls
        weights[mask] = oversample_factor
    
    print(f"  Oversampling weights created:")
    for cls, cnt in zip(unique, counts):
        cls_mask = labels_np == cls
        avg_weight = weights[cls_mask].mean()
        effective_samples = cnt * avg_weight
        print(f"    Class {cls}: {cnt:6d} samples → {effective_samples:8.0f} effective (weight={avg_weight:.1f}x)")
    
    return weights


class BatchMetricsAccumulator:
    """Accumulates metrics across batches"""
    def __init__(self):
        self.total_loss = 0.0
        self.all_labels = []
        self.all_preds = []
        self.count = 0
    
    def update(self, loss, labels, preds):
        self.total_loss += loss
        self.all_labels.append(labels.cpu())
        self.all_preds.append(preds.cpu())
        self.count += 1
    
    def get_average(self):
        all_labels = torch.cat(self.all_labels)
        all_preds = torch.cat(self.all_preds)
        
        # Compute 4-way metrics
        class_names = ['HORZ_A', 'HORZ_B', 'VERT_A', 'VERT_B']
        metrics_dict = compute_metrics(all_labels.numpy(), all_preds.numpy(), labels=class_names)
        metrics_dict['loss'] = self.total_loss / self.count if self.count > 0 else 0.0
        return metrics_dict


def train_epoch(model, dataloader, mixup_criterion, optimizer, device, epoch):
    """Train for one epoch with Mixup"""
    model.train()
    # Keep backbone frozen
    model.backbone.eval()
    
    metrics = BatchMetricsAccumulator()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label_stage3_AB'].to(device)
        
        # Filter valid samples (AB only)
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            continue
        
        images = images[valid_mask]
        labels = labels[valid_mask]
        
        optimizer.zero_grad()
        
        # Mixup forward pass and loss
        logits, loss = mixup_criterion(model, images, labels)
        
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute predictions for metrics
        preds = logits.argmax(dim=1)
        metrics.update(loss.item(), labels, preds)
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return metrics.get_average()


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    metrics = BatchMetricsAccumulator()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label_stage3_AB'].to(device)
            
            # Filter valid samples
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            images = images[valid_mask]
            labels = labels[valid_mask]
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            preds = logits.argmax(dim=1)
            metrics.update(loss.item(), labels, preds)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return metrics.get_average()


def ensemble_predict(models, dataloader, device):
    """Predict using ensemble majority voting"""
    all_votes = []
    all_labels = []
    
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ensemble Voting"):
            images = batch['image'].to(device)
            labels = batch['label_stage3_AB'].to(device)
            
            # Filter valid samples
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            images = images[valid_mask]
            labels = labels[valid_mask]
            
            # Collect votes from all models
            batch_votes = []
            for model in models:
                logits = model(images)
                preds = logits.argmax(dim=1)
                batch_votes.append(preds.cpu().numpy())
            
            # Stack votes: (n_models, batch_size)
            batch_votes = np.stack(batch_votes, axis=0)
            all_votes.append(batch_votes)
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_votes = np.concatenate(all_votes, axis=1)  # (n_models, total_samples)
    all_labels = np.concatenate(all_labels)  # (total_samples,)
    
    # Majority voting
    from scipy import stats
    ensemble_preds, _ = stats.mode(all_votes, axis=0, keepdims=False)
    
    return ensemble_preds, all_labels


def train_single_model(model_id, train_loader, val_loader, device, args, output_dir):
    """Train a single model in the ensemble"""
    print(f"\n{'='*70}")
    print(f"  TRAINING MODEL {model_id}/3")
    print(f"{'='*70}")
    
    # Set unique seed for this model
    seeds = [42, 123, 456]
    set_seed(seeds[model_id - 1])
    print(f"  Random seed: {seeds[model_id - 1]}")
    
    # Create model
    model = Stage3ABModel(pretrained=True).to(device)
    
    # Load Stage 2 backbone
    if Path(args.stage2_model).exists():
        print(f"  Loading Stage 2 backbone...")
        checkpoint = torch.load(args.stage2_model, map_location=device, weights_only=False)
        stage2_state = checkpoint['model_state_dict']
        backbone_state = {k.replace('backbone.', ''): v for k, v in stage2_state.items() if k.startswith('backbone.')}
        model.backbone.load_state_dict(backbone_state, strict=False)
    
    # Freeze backbone (always frozen as per PLANO_V6.md)
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,} (head only, backbone always frozen)")
    
    # Focal Loss (Lin et al., 2017)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Mixup wrapper (Zhang et al., 2018)
    mixup_criterion = MixupLoss(focal_loss, alpha=0.4)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=3e-6
    )
    
    # Training loop
    best_macro_f1 = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_macro_f1': [],
        'val_per_class_f1': []
    }
    
    class_names = ['HORZ_A', 'HORZ_B', 'VERT_A', 'VERT_B']
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, mixup_criterion, optimizer, device, epoch)
        val_metrics = validate_epoch(model, val_loader, focal_loss, device, epoch)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Extract per-class F1 scores
        per_class_f1 = [val_metrics['per_class'][cls]['f1'] for cls in class_names]
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_per_class_f1'].append(per_class_f1)
        
        print(f"\nEpoch {epoch}/{args.epochs} - LR: {current_lr:.2e} [FROZEN]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}")
        print(f"  Val   - Macro F1: {val_metrics['macro_f1']:.2%}")
        for cls_name, f1 in zip(class_names, per_class_f1):
            print(f"          {cls_name:10s} F1: {f1:.2%}")
        
        # Save best model
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_macro_f1': best_macro_f1,
                'val_metrics': val_metrics,
            }, output_dir / f"stage3_ab_model_{model_id}_best.pt")
            print(f"  ✅ New best Macro F1: {best_macro_f1:.2%} (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⚠️  Early stopping! No improvement for {patience} epochs.")
                print(f"  Best Macro F1: {best_macro_f1:.2%} at epoch {best_epoch}")
                break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'history': history,
        'best_macro_f1': best_macro_f1,
        'best_epoch': best_epoch,
    }, output_dir / f"stage3_ab_model_{model_id}_final.pt")
    
    print(f"\n  ✅ Model {model_id} completed!")
    print(f"  Best Macro F1: {best_macro_f1:.2%} (epoch {best_epoch})")
    
    # Load best model for ensemble
    best_checkpoint = torch.load(output_dir / f"stage3_ab_model_{model_id}_best.pt", weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    return model, best_macro_f1, history


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='pesquisa_v6/v6_dataset_stage3/AB/block_16')
    parser.add_argument('--stage2_model', type=str, default='pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt')
    parser.add_argument('--output_dir', type=str, default='pesquisa_v6/logs/v6_experiments/stage3_ab')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--oversample_factor', type=float, default=5.0)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("  Training Stage 3-AB - ENSEMBLE (3 Models)")
    print("  (HORZ_A, HORZ_B, VERT_A, VERT_B)")
    print("  STRATEGY: Ensemble + Oversampling + Mixup + Focal Loss")
    print("="*70)
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Ensemble: 3 models (seeds: 42, 123, 456)")
    print(f"  Epochs: {args.epochs} per model (early stop patience=5)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Loss: Focal Loss (α=0.25, γ=2.0) + Mixup (α=0.4)")
    print(f"  Oversampling: {args.oversample_factor}x for minorities")
    
    # Load datasets
    print(f"\n[1/4] Loading datasets...")
    train_data = torch.load(Path(args.dataset_dir) / "train_v1.pt", weights_only=False)
    val_data = torch.load(Path(args.dataset_dir) / "val.pt", weights_only=False)
    
    print(f"  Train samples: {len(train_data['samples'])}")
    print(f"  Val samples: {len(val_data['samples'])}")
    
    # Create oversampling weights
    print(f"\n[2/4] Creating oversampling weights...")
    train_labels = train_data['labels'] if isinstance(train_data['labels'], np.ndarray) else train_data['labels'].numpy()
    sampling_weights = create_oversampling_weights(train_labels, args.oversample_factor)
    sampler = WeightedRandomSampler(
        weights=sampling_weights,
        num_samples=len(sampling_weights),
        replacement=True
    )
    
    # Data Augmentation
    train_aug = Stage3ABAugmentation(train=True)
    val_aug = Stage3ABAugmentation(train=False)
    
    # Convert to BlockRecord
    train_samples = train_data['samples'] if isinstance(train_data['samples'], np.ndarray) else train_data['samples'].numpy()
    train_qps = train_data['qps'] if isinstance(train_data['qps'], np.ndarray) else train_data['qps'].numpy()
    
    val_samples = val_data['samples'] if isinstance(val_data['samples'], np.ndarray) else val_data['samples'].numpy()
    val_labels = val_data['labels'] if isinstance(val_data['labels'], np.ndarray) else val_data['labels'].numpy()
    val_qps = val_data['qps'] if isinstance(val_data['qps'], np.ndarray) else val_data['qps'].numpy()
    
    train_record = BlockRecord(
        samples=train_samples,
        labels=train_labels,
        qps=train_qps.reshape(-1, 1)
    )
    val_record = BlockRecord(
        samples=val_samples,
        labels=val_labels,
        qps=val_qps.reshape(-1, 1)
    )
    
    train_dataset = build_hierarchical_dataset_v6(train_record, augmentation=train_aug, stage='stage3_ab')
    val_dataset = build_hierarchical_dataset_v6(val_record, augmentation=val_aug, stage='stage3_ab')
    
    # Create dataloaders (train with oversampling)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Train ensemble
    print(f"\n[3/4] Training Ensemble...")
    ensemble_models = []
    ensemble_results = []
    
    for model_id in range(1, 4):
        model, best_f1, history = train_single_model(
            model_id, train_loader, val_loader, device, args, output_dir
        )
        ensemble_models.append(model)
        ensemble_results.append({
            'model_id': model_id,
            'best_macro_f1': float(best_f1),
            'history': history
        })
    
    # Evaluate ensemble
    print(f"\n[4/4] Evaluating Ensemble...")
    print(f"  Computing majority voting predictions...")
    ensemble_preds, true_labels = ensemble_predict(ensemble_models, val_loader, device)
    
    # Compute ensemble metrics
    class_names = ['HORZ_A', 'HORZ_B', 'VERT_A', 'VERT_B']
    ensemble_metrics = compute_metrics(true_labels, ensemble_preds, labels=class_names)
    
    print(f"\n{'='*70}")
    print(f"  ENSEMBLE RESULTS (Majority Voting)")
    print(f"{'='*70}")
    print(f"  Validation Accuracy: {ensemble_metrics['accuracy']:.2%}")
    print(f"  Macro F1: {ensemble_metrics['macro_f1']:.2%}")
    print(f"  Per-class F1:")
    for cls_name in class_names:
        f1 = ensemble_metrics['per_class'][cls_name]['f1']
        precision = ensemble_metrics['per_class'][cls_name]['precision']
        recall = ensemble_metrics['per_class'][cls_name]['recall']
        print(f"    {cls_name:10s}: F1={f1:.2%}, Prec={precision:.2%}, Rec={recall:.2%}")
    
    # Save ensemble metrics
    ensemble_summary = {
        'ensemble_metrics': {
            'accuracy': float(ensemble_metrics['accuracy']),
            'macro_f1': float(ensemble_metrics['macro_f1']),
            'per_class': {
                cls: {k: float(v) for k, v in metrics.items()}
                for cls, metrics in ensemble_metrics['per_class'].items()
            }
        },
        'individual_models': ensemble_results,
        'config': vars(args)
    }
    
    with open(output_dir / "ensemble_metrics.json", 'w') as f:
        json.dump(ensemble_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  ✅ Ensemble training completed!")
    individual_f1s = [r['best_macro_f1'] for r in ensemble_results]
    print(f"  Individual models F1: {individual_f1s}")
    print(f"  Ensemble Macro F1: {ensemble_metrics['macro_f1']:.2%}")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
