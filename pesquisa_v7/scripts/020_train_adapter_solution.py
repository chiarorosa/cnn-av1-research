"""
Script 020: Train Solution 1 - Conv-Adapter
Parameter Efficient Transfer Learning for AV1 Partition Prediction

Based on: Chen et al. (CVPR 2024) - "Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets"

Workflow:
1. Train Stage 1 normally (backbone + binary head)
2. Freeze backbone after Stage 1
3. Train Stage 2 with Conv-Adapter (3.5% params) + 3-way head
4. Train Stage 3 with Conv-Adapter + specialist heads

Expected improvements:
- Stage 2 F1: 46% → 60-65% (solves negative transfer)
- Parameter efficiency: 3.5% trainable vs 100% full fine-tuning
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import json
from tqdm import tqdm
import argparse

# Import v7 pipeline (fully independent)
sys.path.insert(0, str(Path(__file__).parent.parent / "v7_pipeline"))
from data_hub import (
    load_block_records,
    train_test_split,
    build_hierarchical_dataset_v6,
    filter_for_stage2,
    filter_for_stage3,
    get_class_weights,
    create_balanced_sampler,
    compute_class_distribution_v6,
    STAGE2_GROUPS_V6
)
from backbone import create_stage1_head, create_stage2_head, create_stage3_rect_head, create_stage3_ab_head
from conv_adapter import ConvAdapter, AdapterBackbone
from losses import FocalLoss, ClassBalancedFocalLoss
from evaluation import MetricsCalculator


def train_stage1_adapter_solution(
    dataset_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 128,
    epochs: int = 100,
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    patience: int = 15,
    seed: int = 42
):
    """
    Train Stage 1 for Conv-Adapter solution
    Trains backbone + binary head normally (end-to-end)
    """
    print(f"\n{'='*80}")
    print(f"  SOLUTION 1 - Conv-Adapter: Training Stage 1")
    print(f"{'='*80}\n")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    print(f"[1/6] Loading dataset...")
    train_dataset = torch.load(dataset_dir / "train.pt")
    val_dataset = torch.load(dataset_dir / "val.pt")

    # Build hierarchical datasets
    train_hier = build_hierarchical_dataset_v6(train_dataset, stage='stage1')
    val_hier = build_hierarchical_dataset_v6(val_dataset, stage='stage1')

    # Create data loaders
    train_loader = DataLoader(
        train_hier,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_hier,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"  Train samples: {len(train_hier)}")
    print(f"  Val samples: {len(val_hier)}")

    # Create model
    print(f"\n[2/6] Creating Stage 1 model...")
    model = create_stage1_head(pretrained=True)
    model = model.to(device)

    # Loss function
    criterion = FocalLoss(gamma=2.0, alpha=0.25)

    # Optimizer with discriminative learning rates
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.head.parameters(), 'lr': lr_head}
    ])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print(f"\n[3/6] Training Stage 1...")
    best_f1 = 0.0
    patience_counter = 0

    history = {
        'train_loss': [], 'train_f1': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_acc': [],
        'lr_backbone': [], 'lr_head': []
    }

    metrics_calc = MetricsCalculator(num_classes=2, class_names=['NONE', 'PARTITION'])

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['label_stage1'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = batch['label_stage1'].to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_losses.append(loss.item())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        train_metrics = metrics_calc.compute_metrics(
            np.array(train_targets), np.array(train_preds)
        )
        val_metrics = metrics_calc.compute_metrics(
            np.array(val_targets), np.array(val_preds)
        )

        # Update history
        history['train_loss'].append(np.mean(train_losses))
        history['train_f1'].append(train_metrics['f1_macro'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(np.mean(val_losses))
        history['val_f1'].append(val_metrics['f1_macro'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr_backbone'].append(optimizer.param_groups[0]['lr'])
        history['lr_head'].append(optimizer.param_groups[1]['lr'])

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {history['train_loss'][-1]:.4f}, F1: {history['train_f1'][-1]:.4f}")
        print(f"  Val   - Loss: {history['val_loss'][-1]:.4f}, F1: {history['val_f1'][-1]:.4f}")

        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'history': history
            }
            torch.save(checkpoint, output_dir / "stage1_model_best.pt")
            print(f"  ✅ Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        scheduler.step(val_metrics['f1_macro'])

    # Save final model and history
    torch.save(history, output_dir / "stage1_history.pt")

    final_metrics = {
        'best_f1': best_f1,
        'final_train_f1': history['train_f1'][-1],
        'final_val_f1': history['val_f1'][-1],
        'epochs_trained': len(history['train_loss']),
        'early_stopped': patience_counter >= patience
    }

    with open(output_dir / "stage1_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  Stage 1 training completed!")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Model saved: {output_dir / 'stage1_model_best.pt'}")
    print(f"{'='*80}\n")

    return best_f1


def train_stage2_with_adapter(
    dataset_dir: Path,
    stage1_checkpoint: Path,
    output_dir: Path,
    adapter_reduction: int = 4,
    device: str = "cuda",
    batch_size: int = 128,
    epochs: int = 100,
    lr_adapter: float = 1e-3,
    lr_head: float = 1e-3,
    patience: int = 15,
    seed: int = 42
):
    """
    Train Stage 2 with Conv-Adapter
    Backbone frozen, only adapter (3.5% params) + head trainable
    """
    print(f"\n{'='*80}")
    print(f"  SOLUTION 1 - Conv-Adapter: Training Stage 2")
    print(f"  Adapter reduction: {adapter_reduction}")
    print(f"{'='*80}\n")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load Stage 1 checkpoint
    print(f"[1/7] Loading Stage 1 checkpoint...")
    stage1_ckpt = torch.load(stage1_checkpoint, map_location='cpu')

    # Load dataset
    print(f"[2/7] Loading dataset...")
    train_dataset = torch.load(dataset_dir / "train.pt")
    val_dataset = torch.load(dataset_dir / "val.pt")

    # Filter for Stage 2 (remove NONE samples)
    train_stage2 = filter_for_stage2(train_dataset)
    val_stage2 = filter_for_stage2(val_dataset)

    # Build datasets
    train_hier = build_hierarchical_dataset_v6(train_stage2, stage='stage2')
    val_hier = build_hierarchical_dataset_v6(val_stage2, stage='stage2')

    # Create balanced sampler for Stage 2 (3 classes)
    stage2_labels = train_hier.labels_stage2.numpy()
    sampler = create_balanced_sampler(stage2_labels)

    # Create data loaders
    train_loader = DataLoader(
        train_hier,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_hier,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"  Train samples: {len(train_hier)} (balanced)")
    print(f"  Val samples: {len(val_hier)}")

    # Show Stage 2 distribution
    stage2_names = list(STAGE2_GROUPS_V6.keys())
    train_dist = compute_class_distribution_v6(train_stage2.labels)
    val_dist = compute_class_distribution_v6(val_stage2.labels)

    print(f"\n  Stage 2 distribution:")
    print(f"  Train: {[f'{k}: {v*100:.1f}%' for k, v in train_dist.items()]}")
    print(f"  Val:   {[f'{k}: {v*100:.1f}%' for k, v in val_dist.items()]}")

    # Create model with adapter
    print(f"\n[3/7] Creating Stage 2 model with Conv-Adapter...")

    # Load backbone from Stage 1 (frozen)
    base_model = create_stage1_head(pretrained=False)
    base_model.load_state_dict(stage1_ckpt['model_state_dict'])

    # Create adapter backbone (frozen backbone + adapter)
    adapter_backbone = AdapterBackbone(
        base_model.backbone,
        adapter_reduction=adapter_reduction,
        freeze_backbone=True
    )

    # Create Stage 2 head
    stage2_head = create_stage2_head()

    # Combine
    model = nn.Sequential(adapter_backbone, stage2_head)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_efficiency = trainable_params / total_params * 100

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter efficiency: {param_efficiency:.1f}%")

    # Loss function with class balancing
    class_weights = get_class_weights(stage2_labels, beta=0.9999)
    class_weights = torch.from_numpy(class_weights).float().to(device)
    criterion = ClassBalancedFocalLoss(
        gamma=2.0,
        alpha=0.25,
        class_weights=class_weights
    )

    # Optimizer (only adapter and head)
    optimizer = optim.AdamW([
        {'params': adapter_backbone.adapter.parameters(), 'lr': lr_adapter},
        {'params': stage2_head.parameters(), 'lr': lr_head}
    ])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print(f"\n[4/7] Training Stage 2 with Conv-Adapter...")
    best_f1 = 0.0
    patience_counter = 0

    history = {
        'train_loss': [], 'train_f1': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_acc': [],
        'lr_adapter': [], 'lr_head': []
    }

    metrics_calc = MetricsCalculator(num_classes=3, class_names=stage2_names)

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['label_stage2'].to(device)

            # Skip samples not in Stage 2 (-1 labels)
            valid_mask = targets >= 0
            if valid_mask.sum() == 0:
                continue

            images = images[valid_mask]
            targets = targets[valid_mask]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = batch['label_stage2'].to(device)

                valid_mask = targets >= 0
                if valid_mask.sum() == 0:
                    continue

                images = images[valid_mask]
                targets = targets[valid_mask]

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_losses.append(loss.item())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        if train_preds and val_preds:
            train_metrics = metrics_calc.compute_metrics(
                np.array(train_targets), np.array(train_preds)
            )
            val_metrics = metrics_calc.compute_metrics(
                np.array(val_targets), np.array(val_preds)
            )

            # Update history
            history['train_loss'].append(np.mean(train_losses))
            history['train_f1'].append(train_metrics['f1_macro'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(np.mean(val_losses))
            history['val_f1'].append(val_metrics['f1_macro'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['lr_adapter'].append(optimizer.param_groups[0]['lr'])
            history['lr_head'].append(optimizer.param_groups[1]['lr'])

            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train - Loss: {history['train_loss'][-1]:.4f}, F1: {history['train_f1'][-1]:.4f}")
            print(f"  Val   - Loss: {history['val_loss'][-1]:.4f}, F1: {history['val_f1'][-1]:.4f}")

            # Save best model
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'adapter_backbone_state_dict': adapter_backbone.state_dict(),
                    'stage2_head_state_dict': stage2_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1': best_f1,
                    'adapter_reduction': adapter_reduction,
                    'param_efficiency': param_efficiency,
                    'history': history
                }
                torch.save(checkpoint, output_dir / "stage2_adapter_model_best.pt")
                print(f"  ✅ Saved best model (F1: {best_f1:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            scheduler.step(val_metrics['f1_macro'])

    # Save final model and history
    torch.save(history, output_dir / "stage2_adapter_history.pt")

    final_metrics = {
        'best_f1': best_f1,
        'final_train_f1': history['train_f1'][-1] if history['train_f1'] else 0,
        'final_val_f1': history['val_f1'][-1] if history['val_f1'] else 0,
        'epochs_trained': len(history['train_loss']),
        'early_stopped': patience_counter >= patience,
        'adapter_reduction': adapter_reduction,
        'param_efficiency': param_efficiency,
        'total_params': total_params,
        'trainable_params': trainable_params
    }

    with open(output_dir / "stage2_adapter_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  Stage 2 Conv-Adapter training completed!")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Parameter efficiency: {param_efficiency:.1f}%")
    print(f"  Model saved: {output_dir / 'stage2_adapter_model_best.pt'}")
    print(f"{'='*80}\n")

    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Train Solution 1: Conv-Adapter")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to v7 dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default=None,
        help="Path to Stage 1 checkpoint (if None, trains Stage 1 first)"
    )
    parser.add_argument(
        "--adapter-reduction",
        type=int,
        default=4,
        choices=[2, 4, 8, 16],
        help="Adapter reduction ratio (default: 4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum epochs"
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=1e-4,
        help="Learning rate for backbone (Stage 1)"
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=1e-3,
        help="Learning rate for heads"
    )
    parser.add_argument(
        "--lr-adapter",
        type=float,
        default=1e-3,
        help="Learning rate for adapter"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train Stage 1 if checkpoint not provided
    if args.stage1_checkpoint is None:
        print("No Stage 1 checkpoint provided. Training Stage 1 first...")
        stage1_dir = output_dir / "stage1"
        stage1_dir.mkdir(exist_ok=True)

        train_stage1_adapter_solution(
            dataset_dir=dataset_dir,
            output_dir=stage1_dir,
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            patience=args.patience,
            seed=args.seed
        )

        stage1_checkpoint = stage1_dir / "stage1_model_best.pt"
    else:
        stage1_checkpoint = Path(args.stage1_checkpoint)

    # Train Stage 2 with adapter
    stage2_dir = output_dir / "stage2_adapter"
    stage2_dir.mkdir(exist_ok=True)

    train_stage2_with_adapter(
        dataset_dir=dataset_dir,
        stage1_checkpoint=stage1_checkpoint,
        output_dir=stage2_dir,
        adapter_reduction=args.adapter_reduction,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_adapter=args.lr_adapter,
        lr_head=args.lr_head,
        patience=args.patience,
        seed=args.seed
    )

    print(f"\n{'='*80}")
    print(f"  SOLUTION 1 (Conv-Adapter) training completed!")
    print(f"  Results saved in: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()