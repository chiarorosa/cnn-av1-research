#!/usr/bin/env python3
"""
Script 004b: Train Stage 2 Flat (7-way Direct Classification)

Flatten Architecture: Stage 2 predicts 7 partition types directly
Classes: HORZ, VERT, SPLIT, HORZ_A, HORZ_B, VERT_A, VERT_B
No Stage 3 needed → eliminates cascade error (-95% degradation)

Key Design Decisions (Literature-Based):
1. **Class-Balanced Focal Loss** (Cui et al., 2019)
   - β=0.9999 for extreme imbalance (2.8:1 ratio)
   - γ=2.5 for hard negative mining
   - Handles long-tailed distribution (VERT 38k vs HORZ_A 14k samples)

2. **Balanced Sampling** (Buda et al., 2018)
   - WeightedRandomSampler for oversampling minorities
   - Ensures all classes appear ~equally per epoch
   - Prevents model collapse to majority class

3. **Strong Augmentation** (Cubuk et al., 2019 - RandAugment)
   - MixUp (α=0.5, prob=0.4) for regularization
   - CutMix (β=1.0, prob=0.5) for spatial robustness
   - Geometric: rotation ±5°, brightness ±0.1, contrast ±0.1
   - Target: increase effective training data 3-4×

4. **Label Smoothing** (Müller et al., 2019)
   - ε=0.1 to prevent overconfident predictions
   - Compatible with Focal Loss (unlike with CrossEntropy)
   - Improves calibration and generalization

5. **OneCycleLR** (Smith & Topin, 2019)
   - Fast convergence in 50 epochs
   - max_lr backbone: 5e-4, head: 2e-3 (discriminative rates)
   - Cosine annealing with warmup

6. **Progressive Dropout** (Srivastava et al., 2014)
   - ResNet layers: 0.1 → 0.2 → 0.3 → 0.4
   - Prevents overfitting to majority classes

Expected Performance:
- Standalone Stage 2 Flat: Macro F1 ≥ 40%, Accuracy ≥ 55%
- Pipeline (Stage1 × Stage2_flat): 0.73 × 0.58 = ~53% accuracy
- Success Criteria: ≥50% pipeline accuracy (vs 47.66% hierarchical)

Imbalance Analysis:
- Train dataset: 152,600 samples, 7 classes
- Imbalance: 2.8:1 (VERT 38,703 vs HORZ_A 13,915)
- Much better than documented 96:1 (HORZ_4/VERT_4 don't exist in dataset)

Training Schedule:
- Phase 1 (Epochs 1-15): Freeze backbone, train head only
- Phase 2 (Epochs 16-50): Unfreeze, fine-tune with discriminative LR
- Early stopping: on macro F1 (patience=8)

Scientific Contribution:
- Direct 7-way classification > hierarchical 3-stage pipeline
- Eliminates cascade error (Stage 2 low accuracy → Stage 3 collapse)
- Simpler architecture, better performance

References:
- Cui, Y., et al. (2019). Class-Balanced Loss Based on Effective Number of Samples. CVPR.
- Buda, M., et al. (2018). A systematic study of the class imbalance problem in CNNs. Neural Networks.
- Cubuk, E. D., et al. (2019). RandAugment: Practical automated data augmentation. CVPR.
- Müller, R., et al. (2019). When Does Label Smoothing Help? NeurIPS.
- Smith, L. N., & Topin, N. (2019). Super-convergence: Very fast training of neural networks. Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications.
- Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. JMLR.

Author: Chiaro Rosa (PhD Research - AV1 Partition Prediction)
Date: 2025-01
Version: v6-flatten
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import json
from tqdm import tqdm
from collections import Counter

# Add v6 pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "v6_pipeline"))
from data_hub import FLATTEN_ID_TO_NAME
from augmentation import Stage2Augmentation
from metrics import compute_metrics

# ---------------------------------------------------------------------------
# Dataset for Flatten Architecture (7 classes directly)
# ---------------------------------------------------------------------------

class FlattenDataset(torch.utils.data.Dataset):
    """
    Dataset for Flatten architecture: loads 7-class labels directly.
    No hierarchical structure needed.
    """
    def __init__(self, data_path: Path, augmentation=None, split='train'):
        """
        Args:
            data_path: Path to .pt file (train.pt or val.pt)
            augmentation: Optional augmentation module
            split: 'train' or 'val' (for logging)
        """
        print(f"  Loading {split} dataset from: {data_path}")
        data = torch.load(data_path)
        
        self.samples = data['samples']  # (N, C, H, W) normalized [0, 1]
        self.labels = data['labels']    # (N,) labels 0-6
        self.qps = data['qps']          # (N,) quantization parameters
        self.augmentation = augmentation
        self.split = split
        
        # Verify label range
        unique_labels = torch.unique(self.labels).tolist()
        print(f"    Samples: {len(self.labels)}")
        print(f"    Unique labels: {unique_labels}")
        print(f"    Label range: [{self.labels.min()}, {self.labels.max()}]")
        
        # Compute class distribution
        counter = Counter(self.labels.tolist())
        print(f"    Class distribution:")
        for class_id in range(7):  # 0-6
            count = counter.get(class_id, 0)
            percentage = (count / len(self.labels)) * 100
            print(f"      {class_id} ({FLATTEN_ID_TO_NAME.get(class_id, 'UNKNOWN')}): "
                  f"{count:6d} ({percentage:5.2f}%)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.samples[idx]
        label = self.labels[idx]
        qp = self.qps[idx]
        
        # Apply augmentation if training
        if self.augmentation is not None:
            image = self.augmentation(image)
        
        return {
            'image': image,
            'label': label,
            'qp': qp
        }


# ---------------------------------------------------------------------------
# Model: ResNet-18 + SE + Attention + 7-class head
# ---------------------------------------------------------------------------

class Stage2FlatModel(nn.Module):
    """
    Stage 2 Flat: 7-way direct classification
    Architecture: ResNet-18 (ImageNet pretrained) + SE blocks + Spatial Attention + 7-class head
    """
    def __init__(self, num_classes=7, pretrained=True, dropout_rates=None):
        super().__init__()
        
        # Import backbone from v6_pipeline.models
        from models import ImprovedBackbone
        
        self.backbone = ImprovedBackbone(
            pretrained=pretrained,
            dropout_rates=dropout_rates or [0.1, 0.2, 0.3, 0.4]
        )
        
        # 7-class classification head
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        print(f"  Stage2FlatModel initialized:")
        print(f"    Backbone: ResNet-18 (pretrained={pretrained})")
        print(f"    Output classes: {num_classes}")
        print(f"    Dropout rates: {dropout_rates or [0.1, 0.2, 0.3, 0.4]}")
    
    def forward(self, x):
        features = self.backbone(x)  # (B, 512)
        logits = self.head(features)  # (B, 7)
        return logits


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class BatchMetricsAccumulator:
    """Accumulate losses and predictions for metrics computation"""
    def __init__(self):
        self.losses = []
        self.all_labels = []
        self.all_preds = []
    
    def update(self, loss, labels, preds):
        self.losses.append(loss)
        self.all_labels.append(labels.cpu())
        self.all_preds.append(preds.cpu())
    
    def get_average(self):
        avg_loss = np.mean(self.losses)
        all_labels = torch.cat(self.all_labels).numpy()
        all_preds = torch.cat(self.all_preds).numpy()
        return {'loss': avg_loss, 'labels': all_labels, 'preds': all_preds}


def create_balanced_sampler(dataset):
    """
    Create WeightedRandomSampler for balanced sampling.
    Oversamples minority classes to appear ~equally per epoch.
    
    Literature: Buda et al., 2018 - "A systematic study of the class imbalance problem"
    """
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=7)
    
    print(f"\n  Creating balanced sampler:")
    print(f"    Class counts: {class_counts.tolist()}")
    
    # Inverse frequency weights
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    
    print(f"    Class weights: {class_weights.tolist()}")
    print(f"    Sample weights range: [{sample_weights.min():.6f}, {sample_weights.max():.6f}]")
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, freeze_backbone=False):
    """Train for one epoch"""
    model.train()
    
    # Freeze backbone if requested (Phase 1)
    if freeze_backbone:
        model.backbone.eval()
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        model.backbone.train()
        for param in model.backbone.parameters():
            param.requires_grad = True
    
    metrics = BatchMetricsAccumulator()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        preds = logits.argmax(dim=1)
        metrics.update(loss.item(), labels, preds)
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute epoch metrics
    result = metrics.get_average()
    epoch_metrics = compute_metrics(result['labels'], result['preds'], num_classes=7)
    epoch_metrics['loss'] = result['loss']
    
    return epoch_metrics


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    metrics = BatchMetricsAccumulator()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Val]")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Accumulate metrics
            preds = logits.argmax(dim=1)
            metrics.update(loss.item(), labels, preds)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute epoch metrics
    result = metrics.get_average()
    epoch_metrics = compute_metrics(result['labels'], result['preds'], num_classes=7)
    epoch_metrics['loss'] = result['loss']
    
    return epoch_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)


def save_history(history, path):
    """Save training history"""
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(item) for item in obj]
        return obj
    
    history_python = convert_to_python(history)
    
    with open(path, 'w') as f:
        json.dump(history_python, f, indent=2)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_stage2_flat(
    dataset_dir: Path,
    output_dir: Path,
    batch_size: int = 128,
    num_epochs: int = 50,
    freeze_epochs: int = 15,
    lr_backbone: float = 5e-4,
    lr_head: float = 2e-3,
    weight_decay: float = 1e-4,
    beta: float = 0.9999,
    gamma: float = 2.5,
    label_smoothing: float = 0.1,
    device: str = 'cuda',
    seed: int = 42
):
    """
    Train Stage 2 Flat model (7-way direct classification)
    
    Args:
        dataset_dir: Path to flatten dataset (v6_dataset_flatten/block_16)
        output_dir: Output directory for checkpoints and logs
        batch_size: Batch size (default: 128)
        num_epochs: Total epochs (default: 50)
        freeze_epochs: Epochs to freeze backbone (default: 15)
        lr_backbone: Backbone learning rate (default: 5e-4)
        lr_head: Head learning rate (default: 2e-3)
        weight_decay: L2 regularization (default: 1e-4)
        beta: CB-Focal beta (default: 0.9999)
        gamma: Focal gamma (default: 2.5)
        label_smoothing: Label smoothing epsilon (default: 0.1)
        device: Device ('cuda' or 'cpu')
        seed: Random seed
    """
    print(f"\n{'='*70}")
    print(f"  Training Stage 2 Flat (7-way Direct Classification)")
    print(f"{'='*70}\n")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = Path(dataset_dir)
    
    # -----------------------------------------------------------------------
    # 1. Load datasets
    # -----------------------------------------------------------------------
    print(f"[1/7] Loading datasets...")
    
    # Augmentation for training
    train_augmentation = Stage2Augmentation(
        geometric_prob=0.5,
        mixup_prob=0.4,
        mixup_alpha=0.5,
        cutmix_prob=0.5,
        cutmix_beta=1.0
    )
    
    train_dataset = FlattenDataset(
        dataset_dir / "train.pt",
        augmentation=train_augmentation,
        split='train'
    )
    
    val_dataset = FlattenDataset(
        dataset_dir / "val.pt",
        augmentation=None,  # No augmentation for validation
        split='val'
    )
    
    # -----------------------------------------------------------------------
    # 2. Create balanced sampler
    # -----------------------------------------------------------------------
    print(f"\n[2/7] Creating balanced sampler...")
    train_sampler = create_balanced_sampler(train_dataset)
    
    # -----------------------------------------------------------------------
    # 3. Create dataloaders
    # -----------------------------------------------------------------------
    print(f"\n[3/7] Creating dataloaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Use balanced sampler
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches: {len(val_loader)}")
    
    # -----------------------------------------------------------------------
    # 4. Create model
    # -----------------------------------------------------------------------
    print(f"\n[4/7] Creating Stage2FlatModel...")
    
    model = Stage2FlatModel(
        num_classes=7,
        pretrained=True,
        dropout_rates=[0.1, 0.2, 0.3, 0.4]
    ).to(device)
    
    # -----------------------------------------------------------------------
    # 5. Create loss and optimizer
    # -----------------------------------------------------------------------
    print(f"\n[5/7] Creating loss and optimizer...")
    
    # Class-Balanced Focal Loss
    from losses import ClassBalancedFocalLoss
    
    # Get class counts from train dataset
    class_counts = np.bincount(train_dataset.labels.numpy(), minlength=7)
    
    criterion = ClassBalancedFocalLoss(
        samples_per_class=class_counts.tolist(),
        num_classes=7,
        beta=beta,
        gamma=gamma,
        label_smoothing=label_smoothing,
        device=device
    )
    
    print(f"    Loss: CB-Focal (β={beta}, γ={gamma}, ε={label_smoothing})")
    print(f"    Class counts: {class_counts.tolist()}")
    
    # Discriminative learning rates (Howard & Ruder, 2018)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.head.parameters(), 'lr': lr_head}
    ], weight_decay=weight_decay)
    
    print(f"    Optimizer: AdamW")
    print(f"      Backbone LR: {lr_backbone}")
    print(f"      Head LR: {lr_head}")
    print(f"      Weight decay: {weight_decay}")
    
    # OneCycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr_backbone, lr_head],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30% warmup
        anneal_strategy='cos'
    )
    
    print(f"    Scheduler: OneCycleLR (max_lr=[{lr_backbone}, {lr_head}])")
    
    # -----------------------------------------------------------------------
    # 6. Training loop
    # -----------------------------------------------------------------------
    print(f"\n[6/7] Starting training...")
    print(f"    Total epochs: {num_epochs}")
    print(f"    Freeze backbone: epochs 1-{freeze_epochs}")
    print(f"    Unfreeze: epochs {freeze_epochs+1}-{num_epochs}")
    print(f"    Early stopping: patience=8 on macro F1")
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_macro_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_macro_f1': [],
        'val_per_class_f1': []
    }
    
    best_f1 = 0.0
    patience_counter = 0
    patience = 8
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"  Epoch {epoch}/{num_epochs}")
        print(f"{'='*70}")
        
        # Determine if backbone should be frozen
        freeze_backbone = (epoch <= freeze_epochs)
        
        if epoch == 1:
            print(f"  Phase 1: Freezing backbone (epochs 1-{freeze_epochs})")
        elif epoch == freeze_epochs + 1:
            print(f"  Phase 2: Unfreezing backbone (epochs {freeze_epochs+1}-{num_epochs})")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, freeze_backbone
        )
        
        print(f"\n  Train metrics:")
        print(f"    Loss: {train_metrics['loss']:.4f}")
        print(f"    Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"    Macro F1: {train_metrics['macro_f1']:.4f}")
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        print(f"\n  Val metrics:")
        print(f"    Loss: {val_metrics['loss']:.4f}")
        print(f"    Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"    Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Per-class F1:")
        for class_id, f1 in enumerate(val_metrics['per_class_f1']):
            class_name = FLATTEN_ID_TO_NAME.get(class_id, f"Class_{class_id}")
            print(f"    {class_name:20s}: {f1:.4f}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_macro_f1'].append(train_metrics['macro_f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_per_class_f1'].append(val_metrics['per_class_f1'])
        
        # Save checkpoint if best F1
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            patience_counter = 0
            
            checkpoint_path = output_dir / "stage2_flat_model_best.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_path)
            print(f"\n  ✅ New best F1: {best_f1:.4f} (saved to {checkpoint_path.name})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break
        
        # Step scheduler
        if scheduler:
            scheduler.step()
    
    # -----------------------------------------------------------------------
    # 7. Save final checkpoint and history
    # -----------------------------------------------------------------------
    print(f"\n[7/7] Saving final checkpoint and history...")
    
    final_checkpoint_path = output_dir / "stage2_flat_model_final.pt"
    save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, final_checkpoint_path)
    print(f"  Final checkpoint: {final_checkpoint_path}")
    
    history_path = output_dir / "stage2_flat_history.json"
    save_history(history, history_path)
    print(f"  History: {history_path}")
    
    # Save final metrics
    final_metrics = {
        'best_epoch': int(np.argmax(history['val_macro_f1']) + 1),
        'best_val_f1': float(best_f1),
        'final_val_f1': float(val_metrics['macro_f1']),
        'final_val_accuracy': float(val_metrics['accuracy']),
        'total_epochs': epoch
    }
    
    metrics_path = output_dir / "stage2_flat_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}")
    
    print(f"\n{'='*70}")
    print(f"  ✅ Training complete!")
    print(f"  Best val F1: {best_f1:.4f} (epoch {final_metrics['best_epoch']})")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    return final_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Stage 2 Flat (7-way Direct Classification)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults
  python3 004b_train_stage2_flat_7classes.py
  
  # Custom hyperparameters
  python3 004b_train_stage2_flat_7classes.py \\
      --batch-size 256 \\
      --num-epochs 100 \\
      --freeze-epochs 20 \\
      --lr-backbone 1e-3 \\
      --lr-head 5e-3
  
  # Train on CPU
  python3 004b_train_stage2_flat_7classes.py --device cpu

Output:
  logs/v6_experiments/stage2_flat/
    stage2_flat_model_best.pt    # Best macro F1
    stage2_flat_model_final.pt   # Last epoch
    stage2_flat_history.json     # Per-epoch metrics
    stage2_flat_metrics.json     # Summary
        """
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to flatten dataset (default: pesquisa_v6/v6_dataset_flatten/block_16)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: pesquisa_v6/logs/v6_experiments/stage2_flat)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Total epochs (default: 50)"
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=15,
        help="Epochs to freeze backbone (default: 15)"
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=5e-4,
        help="Backbone learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=2e-3,
        help="Head learning rate (default: 2e-3)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization (default: 1e-4)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9999,
        help="CB-Focal beta (default: 0.9999)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.5,
        help="Focal gamma (default: 2.5)"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing epsilon (default: 0.1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help="Device (default: cuda)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    script_dir = Path(__file__).parent.parent
    
    if args.dataset_dir is None:
        dataset_dir = script_dir / "v6_dataset_flatten" / "block_16"
    else:
        dataset_dir = Path(args.dataset_dir)
    
    if args.output_dir is None:
        output_dir = script_dir / "logs" / "v6_experiments" / "stage2_flat"
    else:
        output_dir = Path(args.output_dir)
    
    # Verify dataset exists
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Flatten dataset not found: {dataset_dir}\n"
            f"Please run 001b_prepare_flatten_dataset.py first."
        )
    
    # Run training
    train_stage2_flat(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        freeze_epochs=args.freeze_epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        beta=args.beta,
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        device=args.device,
        seed=args.seed
    )
