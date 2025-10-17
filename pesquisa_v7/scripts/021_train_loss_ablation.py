"""
Script 021: Loss Function Ablation Study (Experiment 04)

Tests 4 loss functions for Stage 2:
1. Baseline: ClassBalancedFocalLoss γ=2.0
2. Exp 4A: ClassBalancedFocalLoss γ=3.0 (harder focusing)
3. Exp 4B: PolyLoss (Leng et al., NeurIPS 2022)
4. Exp 4C: AsymmetricLoss (Ridnik et al., ICCV 2021)
5. Exp 4D: FocalLoss + Label Smoothing (hybrid)

Usage:
    # Test specific loss
    python3 script/021_train_loss_ablation.py --loss-type poly --output-dir logs/exp04b_poly_loss
    
    # Run all experiments (sequential)
    bash pesquisa_v7/scripts/run_loss_ablation.sh
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from tqdm import tqdm
import argparse

# Import v7 pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "v7_pipeline"))
from v7_pipeline.data_hub import create_balanced_sampler, STAGE2_GROUPS_V6, compute_class_distribution_v6
from v7_pipeline.backbone import create_stage2_head, ImprovedBackbone
from v7_pipeline.conv_adapter import AdapterBackbone
from v7_pipeline.losses import ClassBalancedFocalLoss
from v7_pipeline.losses_ablation import PolyLoss, AsymmetricLoss, FocalLossWithLabelSmoothing
from v7_pipeline.evaluation import MetricsCalculator


def get_loss_function(loss_type: str, samples_per_class: np.ndarray, class_weights: torch.Tensor):
    """
    Factory function to create loss function based on type
    
    Args:
        loss_type: One of ['baseline', 'focal_gamma3', 'poly', 'asymmetric', 'focal_smoothing']
        samples_per_class: Number of samples per class (for ClassBalancedFocalLoss)
        class_weights: Class weights for balancing (for other losses)
    
    Returns:
        loss_fn: nn.Module loss function
        loss_name: Human-readable name for logging
    """
    if loss_type == 'baseline':
        # ClassBalancedFocalLoss γ=2.0 (current default)
        loss_fn = ClassBalancedFocalLoss(
            samples_per_class=samples_per_class,
            beta=0.9999,
            gamma=2.0,
            reduction='mean'
        )
        loss_name = "ClassBalancedFocalLoss (γ=2.0)"
    
    elif loss_type == 'focal_gamma3':
        # Exp 4A: Increase focusing to γ=3.0
        loss_fn = ClassBalancedFocalLoss(
            samples_per_class=samples_per_class,
            beta=0.9999,
            gamma=3.0,
            reduction='mean'
        )
        loss_name = "ClassBalancedFocalLoss (γ=3.0)"
    
    elif loss_type == 'poly':
        # Exp 4B: PolyLoss (Leng et al., NeurIPS 2022)
        loss_fn = PolyLoss(
            epsilon=1.0,
            class_weights=class_weights,
            reduction='mean'
        )
        loss_name = "PolyLoss (ε=1.0)"
    
    elif loss_type == 'asymmetric':
        # Exp 4C: AsymmetricLoss (Ridnik et al., ICCV 2021)
        loss_fn = AsymmetricLoss(
            gamma_pos=2.0,
            gamma_neg=4.0,
            class_weights=class_weights,
            reduction='mean'
        )
        loss_name = "AsymmetricLoss (γ_pos=2, γ_neg=4)"
    
    elif loss_type == 'focal_smoothing':
        # Exp 4D: Focal + Label Smoothing
        loss_fn = FocalLossWithLabelSmoothing(
            gamma=2.0,
            epsilon=0.1,
            class_weights=class_weights,
            reduction='mean'
        )
        loss_name = "FocalLoss + LabelSmoothing (γ=2.0, ε=0.1)"
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: baseline, focal_gamma3, poly, asymmetric, focal_smoothing")
    
    return loss_fn, loss_name


def train_stage2_loss_ablation(
    dataset_dir: Path,
    stage1_checkpoint: Path,
    output_dir: Path,
    loss_type: str,
    adapter_reduction: int = 4,
    device: str = "cuda",
    batch_size: int = 128,
    epochs: int = 50,
    lr_adapter: float = 1e-3,
    lr_head: float = 1e-4,
    patience: int = 15,
    seed: int = 42
):
    """
    Train Stage 2 with specified loss function for ablation study
    
    All other hyperparameters kept identical to baseline (020_train_adapter_solution.py)
    Only loss function changes.
    """
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT 04: Loss Function Ablation")
    print(f"  Loss type: {loss_type}")
    print(f"{'='*80}\n")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir = output_dir / "stage2_adapter"
    stage2_dir.mkdir(exist_ok=True)

    # Load Stage 1 checkpoint
    print(f"[1/7] Loading Stage 1 checkpoint...")
    stage1_ckpt = torch.load(stage1_checkpoint, map_location='cpu', weights_only=False)
    print(f"  Stage 1 F1: {stage1_ckpt.get('val_metrics', {}).get('f1', 'N/A')}")

    # Load dataset
    print(f"\n[2/7] Loading dataset...")
    train_data = torch.load(dataset_dir / "train.pt", weights_only=False)
    val_data = torch.load(dataset_dir / "val.pt", weights_only=False)

    # Filter for Stage 2 (remove NONE samples)
    train_stage2_mask = train_data['labels_stage2'] >= 0
    val_stage2_mask = val_data['labels_stage2'] >= 0
    
    train_stage2_data = {
        'samples': train_data['samples'][train_stage2_mask],
        'labels': train_data['labels_stage2'][train_stage2_mask],
        'qps': train_data['qps'][train_stage2_mask]
    }
    
    val_stage2_data = {
        'samples': val_data['samples'][val_stage2_mask],
        'labels': val_data['labels_stage2'][val_stage2_mask],
        'qps': val_data['qps'][val_stage2_mask]
    }

    # Create datasets
    train_dataset = TensorDataset(
        train_stage2_data['samples'],
        train_stage2_data['labels'],
        train_stage2_data['qps']
    )
    
    val_dataset = TensorDataset(
        val_stage2_data['samples'],
        val_stage2_data['labels'],
        val_stage2_data['qps']
    )

    # Create balanced sampler
    stage2_labels = train_stage2_data['labels'].numpy()
    sampler = create_balanced_sampler(stage2_labels)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
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

    print(f"  Train samples: {len(train_dataset)} (balanced)")
    print(f"  Val samples: {len(val_dataset)}")

    # Show Stage 2 distribution
    train_dist = compute_class_distribution_v6(train_stage2_data['labels'].numpy())
    val_dist = compute_class_distribution_v6(val_stage2_data['labels'].numpy())

    print(f"\n  Stage 2 distribution:")
    print(f"  Train: {', '.join([f'{k}: {v*100:.1f}%' for k, v in train_dist.items()])}")
    print(f"  Val:   {', '.join([f'{k}: {v*100:.1f}%' for k, v in val_dist.items()])}")

    # Create model with adapter
    print(f"\n[3/7] Creating Stage 2 model with Conv-Adapter...")

    # Load backbone from Stage 1 checkpoint
    backbone = ImprovedBackbone(pretrained=False)
    backbone_state_dict = {k.replace('backbone.', '', 1): v 
                          for k, v in stage1_ckpt['model_state_dict'].items() 
                          if k.startswith('backbone.')}
    backbone.load_state_dict(backbone_state_dict)

    # Create adapter backbone
    adapter_config = {
        'reduction': adapter_reduction,
        'layers': ['layer3', 'layer4'],
        'variant': 'conv_parallel'
    }
    adapter_backbone = AdapterBackbone(backbone, adapter_config=adapter_config)

    # Freeze backbone
    for param in adapter_backbone.backbone.parameters():
        param.requires_grad = False

    # Create Stage 2 head
    stage2_head = create_stage2_head()

    # Wrapper to extract features
    class AdapterBackboneWrapper(nn.Module):
        def __init__(self, adapter_backbone):
            super().__init__()
            self.adapter_backbone = adapter_backbone
        
        def forward(self, x):
            features, _ = self.adapter_backbone(x)
            return features

    # Create full model
    class Stage2AdapterModel(nn.Module):
        def __init__(self, adapter_backbone, head):
            super().__init__()
            self.backbone = AdapterBackboneWrapper(adapter_backbone)
            self.head = head
        
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    model = Stage2AdapterModel(adapter_backbone, stage2_head)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_params = sum(p.numel() for p in adapter_backbone.adapters.parameters())
    head_params = sum(p.numel() for p in stage2_head.parameters())

    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  - Adapter params: {adapter_params:,}")
    print(f"  - Head params: {head_params:,}")

    # Compute class weights for loss
    print(f"\n[4/7] Computing class weights...")
    samples_per_class = np.bincount(stage2_labels, minlength=3)
    
    # Compute effective number (Cui et al., 2019)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, samples_per_class)
    class_weights = (1.0 - beta) / np.array(effective_num)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.from_numpy(class_weights).float().to(device)
    
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Effective weights: {class_weights.cpu().numpy()}")

    # Create loss function (THE ONLY THING THAT CHANGES)
    print(f"\n[5/7] Creating loss function...")
    criterion, loss_name = get_loss_function(loss_type, samples_per_class, class_weights)
    criterion = criterion.to(device)
    
    print(f"  Loss function: {loss_name}")

    # Optimizer (same as baseline)
    optimizer = optim.AdamW([
        {'params': adapter_backbone.adapters.parameters(), 'lr': lr_adapter, 'weight_decay': 0.01},
        {'params': stage2_head.parameters(), 'lr': lr_head, 'weight_decay': 0.01}
    ])

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Class names for metrics
    class_names = list(STAGE2_GROUPS_V6.keys())

    # Training loop
    print(f"\n[6/7] Training Stage 2...")
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': [],
        'val_precision': [], 'val_recall': []
    }

    for epoch in range(epochs):
        # =====================
        # Training
        # =====================
        model.train()
        adapter_backbone.backbone.eval()  # ← CRITICAL: BatchNorm fix
        
        train_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (samples, labels, _) in enumerate(pbar):
            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            logits = model(samples)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)
        train_metrics = MetricsCalculator.calculate_classification_metrics(all_targets, all_preds, num_classes=3)

        # =====================
        # Validation
        # =====================
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for samples, labels, _ in pbar:
                samples = samples.to(device)
                labels = labels.to(device)

                logits = model(samples)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss /= len(val_loader)
        val_metrics = MetricsCalculator.calculate_classification_metrics(all_targets, all_preds, num_classes=3)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1_macro'])
        history['val_precision'].append(val_metrics['precision_macro'])
        history['val_recall'].append(val_metrics['recall_macro'])

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, F1: {train_metrics['f1_macro']*100:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, F1: {val_metrics['f1_macro']*100:.2f}%")
        
        # Per-class F1
        if 'f1_per_class' in val_metrics:
            per_class_str = ', '.join([f'{class_names[i]}: {f1*100:.1f}%' for i, f1 in enumerate(val_metrics['f1_per_class'])])
            print(f"  Per-class F1: {per_class_str}")

        # Learning rate scheduling
        scheduler.step(val_metrics['f1_macro'])

        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'loss_type': loss_type,
                'loss_name': loss_name,
                'adapter_reduction': adapter_reduction,
                'param_efficiency': trainable_params / total_params * 100
            }
            torch.save(checkpoint, stage2_dir / "stage2_adapter_model_best.pt")
            print(f"  ✓ Best model saved (F1: {best_val_f1*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    # Save final model and history
    print(f"\n[7/7] Saving artifacts...")
    torch.save(model.state_dict(), stage2_dir / "stage2_adapter_model_final.pt")
    torch.save(history, stage2_dir / "stage2_adapter_history.pt")

    # Convert per-class metrics to dict format for compatibility
    per_class_f1 = {class_names[i]: float(val_metrics['f1_per_class'][i]) for i in range(len(class_names))} if 'f1_per_class' in val_metrics else {}
    
    # Save metrics JSON
    final_metrics = {
        'loss_type': loss_type,
        'loss_name': loss_name,
        'best_val_f1': float(best_val_f1),
        'best_epoch': int(history['val_f1'].index(max(history['val_f1'])) + 1),
        'total_epochs': len(history['val_f1']),
        'final_val_metrics': {
            'f1': float(val_metrics['f1_macro']),
            'precision': float(val_metrics['precision_macro']),
            'recall': float(val_metrics['recall_macro']),
            'accuracy': float(val_metrics['accuracy']),
            'per_class_f1': per_class_f1
        },
        'param_efficiency': float(trainable_params / total_params * 100),
        'trainable_params': int(trainable_params),
        'total_params': int(total_params)
    }

    with open(stage2_dir / "stage2_adapter_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  Training completed!")
    print(f"  Best Val F1: {best_val_f1*100:.2f}%")
    print(f"  Checkpoints saved to: {stage2_dir}")
    print(f"{'='*80}\n")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Loss Function Ablation Study (Experiment 04)")
    
    # Required
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="Path to v7_dataset/block_16")
    parser.add_argument("--stage1-checkpoint", type=str, required=True,
                       help="Path to Stage 1 checkpoint")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for this experiment")
    parser.add_argument("--loss-type", type=str, required=True,
                       choices=['baseline', 'focal_gamma3', 'poly', 'asymmetric', 'focal_smoothing'],
                       help="Loss function to test")
    
    # Optional
    parser.add_argument("--adapter-reduction", type=int, default=4,
                       help="Adapter reduction factor (default: 4)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size (default: 128)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Max epochs (default: 50)")
    parser.add_argument("--lr-adapter", type=float, default=1e-3,
                       help="Learning rate for adapter (default: 1e-3)")
    parser.add_argument("--lr-head", type=float, default=1e-4,
                       help="Learning rate for head (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience (default: 15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    args = parser.parse_args()

    # Convert paths
    dataset_dir = Path(args.dataset_dir)
    stage1_checkpoint = Path(args.stage1_checkpoint)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint}")

    # Run training
    metrics = train_stage2_loss_ablation(
        dataset_dir=dataset_dir,
        stage1_checkpoint=stage1_checkpoint,
        output_dir=output_dir,
        loss_type=args.loss_type,
        adapter_reduction=args.adapter_reduction,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_adapter=args.lr_adapter,
        lr_head=args.lr_head,
        patience=args.patience,
        seed=args.seed
    )

    print("\n✓ Experiment completed successfully!")
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
