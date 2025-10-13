"""
Script 005: Train Stage 3 RECT
Stage 3-RECT: Binary classification (HORZ vs VERT)

Improvements based on literature:
1. Label Smoothing (Szegedy et al., 2016) - Rethinking the Inception Architecture
2. Gradient Clipping (Pascanu et al., 2013) - On the difficulty of training RNNs
3. Early Stopping (Goodfellow et al., 2016) - Deep Learning Book
4. Balanced Accuracy (Brodersen et al., 2010) - The Balanced Accuracy and Its Posterior Distribution
5. Progressive Unfreezing (Howard & Ruder, 2018) - Universal Language Model Fine-tuning

References:
- Szegedy, C., et al. (2016). Rethinking the Inception Architecture for Computer Vision. CVPR.
- Pascanu, R., et al. (2013). On the difficulty of training recurrent neural networks. ICML.
- Goodfellow, I., et al. (2016). Deep Learning. MIT Press.
- Brodersen, K. H., et al. (2010). The Balanced Accuracy and Its Posterior Distribution. ICPR.
- Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. ACL.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm

# Add v6 pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "v6_pipeline"))
from data_hub import HierarchicalBlockDatasetV6
from models import Stage3RectModel
from losses import FocalLoss
from augmentation import Stage2Augmentation
from metrics import compute_binary_metrics, MetricsTracker


class NoisyDataset(torch.utils.data.Dataset):
    """
    Dataset with noise injection for adversarial training.
    
    Mixes clean RECT samples with noisy samples from other classes (AB, SPLIT).
    Noisy samples are assigned random labels to simulate Stage 2 misclassification.
    
    References:
    - Hendrycks et al., 2019: "Using Pre-Training Can Improve Model Robustness"
    - Natarajan et al., 2013: "Learning with Noisy Labels"
    
    Args:
        clean_dataset: Original RECT dataset (clean samples)
        noise_records: List of BlockRecord objects from noise sources (AB, SPLIT)
        noise_ratio: Fraction of dataset to be noise (0.0-1.0)
        augmentation: Augmentation to apply
        seed: Random seed for reproducibility
    """
    def __init__(self, clean_dataset, noise_records, noise_ratio=0.25, augmentation=None, seed=42):
        from data_hub import BlockRecord, build_hierarchical_dataset_v6
        
        self.clean_dataset = clean_dataset
        self.noise_ratio = noise_ratio
        self.augmentation = augmentation
        
        # Calculate sample counts
        total_samples = len(clean_dataset)
        n_clean = int(total_samples * (1 - noise_ratio))
        n_noise = total_samples - n_clean
        
        # Create clean indices
        rng = np.random.RandomState(seed)
        clean_indices = rng.choice(len(clean_dataset), n_clean, replace=False)
        self.clean_indices = sorted(clean_indices)
        
        # Create noise datasets from each source
        self.noise_datasets = []
        if noise_records:
            per_source = n_noise // len(noise_records)
            for noise_record in noise_records:
                noise_ds = build_hierarchical_dataset_v6(
                    noise_record, 
                    augmentation=augmentation, 
                    stage='stage3_rect'  # Use same stage structure
                )
                # Sample random indices from noise source
                noise_indices = rng.choice(len(noise_ds), per_source, replace=False)
                self.noise_datasets.append((noise_ds, noise_indices))
        
        self.total_len = n_clean + n_noise
        
        print(f"\n[NoisyDataset] Created with:")
        print(f"  Clean samples: {n_clean} ({(1-noise_ratio)*100:.1f}%)")
        print(f"  Noise samples: {n_noise} ({noise_ratio*100:.1f}%)")
        print(f"  Noise sources: {len(noise_records)}")
        print(f"  Total samples: {self.total_len}")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        n_clean = len(self.clean_indices)
        
        if idx < n_clean:
            # Return clean sample
            clean_idx = self.clean_indices[idx]
            return self.clean_dataset[clean_idx]
        else:
            # Return noisy sample with random label
            noise_idx = idx - n_clean
            source_idx = noise_idx % len(self.noise_datasets)
            noise_ds, noise_indices = self.noise_datasets[source_idx]
            
            sample_idx = noise_indices[noise_idx // len(self.noise_datasets)]
            batch = noise_ds[sample_idx]
            
            # Replace label with random HORZ (0) or VERT (1)
            batch['label_stage3_RECT'] = torch.tensor(np.random.randint(0, 2), dtype=torch.long)
            
            return batch


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
        metrics = compute_binary_metrics(all_labels, all_preds)
        metrics['loss'] = self.total_loss / self.count if self.count > 0 else 0.0
        return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, freeze_backbone=True):
    """Train for one epoch"""
    model.train()
    
    # Freeze/unfreeze backbone based on epoch
    if freeze_backbone:
        model.backbone.eval()
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        model.backbone.train()
        for param in model.backbone.parameters():
            param.requires_grad = True
    
    metrics = BatchMetricsAccumulator()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label_stage3_RECT'].to(device)
        
        # Filter valid samples (RECT only)
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            continue
        
        images = images[valid_mask]
        labels = labels[valid_mask]
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        
        # Gradient Clipping (Pascanu et al., 2013)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics (use argmax for predictions)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            
        metrics.update(loss.item(), labels, preds)
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return metrics.get_average()


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    metrics = BatchMetricsAccumulator()
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label_stage3_RECT'].to(device)
            
            # Filter valid samples (RECT only)
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            images = images[valid_mask]
            labels = labels[valid_mask]
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Use argmax for predictions
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            metrics.update(loss.item(), labels, preds)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute detailed metrics
    probs = torch.softmax(torch.stack([1 - all_preds.float(), all_preds.float()], dim=1), dim=1)[:, 1]
    binary_metrics = compute_binary_metrics(all_labels.numpy(), all_preds.numpy(), probs.numpy())
    avg_metrics = metrics.get_average()
    avg_metrics.update(binary_metrics)
    
    # Balanced Accuracy (Brodersen et al., 2010)
    balanced_acc = (binary_metrics['recall'] + binary_metrics['specificity']) / 2
    avg_metrics['balanced_accuracy'] = balanced_acc
    
    return avg_metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Stage 3 RECT")
    parser.add_argument("--dataset-dir", type=str, 
                       default="pesquisa_v6/v6_dataset_stage3/RECT/block_16",
                       help="RECT dataset directory")
    parser.add_argument("--stage2-model", type=str,
                       default="pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt",
                       help="Path to trained Stage 2 model (for backbone)")
    parser.add_argument("--output-dir", type=str,
                       default="pesquisa_v6/logs/v6_experiments/stage3_rect",
                       help="Output directory for logs and models")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Max epochs (may stop early with patience=5)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                       help="Learning rate (increased from 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                       help="Focal loss gamma")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                       help="Focal loss alpha")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Noise Injection arguments (Hendrycks et al., 2019)
    parser.add_argument("--noise-injection", type=float, default=0.0,
                       help="Fraction of noisy samples to inject (0.0-1.0). E.g., 0.25 = 25%% noise")
    parser.add_argument("--noise-sources", nargs='+', 
                       choices=['AB', 'SPLIT'], default=['AB'],
                       help="Sources for noisy samples: AB (Stage 3-AB), SPLIT (PARTITION_SPLIT)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup paths
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  Training Stage 3-RECT - Binary Classification (HORZ vs VERT)")
    if args.noise_injection > 0:
        print(f"  IMPROVEMENTS v3: Noise Injection + Label Smoothing + Early Stop")
    else:
        print(f"  IMPROVEMENTS v2: Label Smoothing + Gradient Clip + Early Stop")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {args.epochs} (early stop patience=5, unfreeze after epoch 5)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr} (head) / {args.lr*0.01:.2e} (backbone)")
    print(f"  Loss: CrossEntropy + Class Weights + Label Smoothing (0.1)")
    print(f"  Regularization: Gradient Clipping (norm=1.0)")
    if args.noise_injection > 0:
        print(f"  Noise Injection: {args.noise_injection*100:.1f}% from sources {args.noise_sources}")
    
    # Load datasets
    print(f"\n[1/5] Loading datasets...")
    train_data = torch.load(dataset_dir / "train.pt", weights_only=False)
    val_data = torch.load(dataset_dir / "val.pt", weights_only=False)
    
    print(f"  Train samples: {len(train_data['samples'])}")
    print(f"  Val samples: {len(val_data['samples'])}")
    
    # Create augmentation
    train_aug = Stage2Augmentation(train=True)
    val_aug = Stage2Augmentation(train=False)
    
    # Create datasets
    from data_hub import BlockRecord, build_hierarchical_dataset_v6
    
    # Convert to numpy if needed (data is already in (N, H, W, C) format)
    train_samples = train_data['samples'] if isinstance(train_data['samples'], np.ndarray) else train_data['samples'].numpy()
    train_labels = train_data['labels'] if isinstance(train_data['labels'], np.ndarray) else train_data['labels'].numpy()
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
    
    # Create clean datasets first
    train_dataset_clean = build_hierarchical_dataset_v6(train_record, augmentation=train_aug, stage='stage3_rect')
    val_dataset = build_hierarchical_dataset_v6(val_record, augmentation=val_aug, stage='stage3_rect')
    
    # Apply Noise Injection if specified (Hendrycks et al., 2019)
    if args.noise_injection > 0:
        print(f"\n[Noise Injection] Loading noise sources...")
        noise_records = []
        
        # Find project root (go up from dataset_dir until we find pesquisa_v6)
        project_root = dataset_dir
        while project_root.name != 'pesquisa_v6' and project_root.parent != project_root:
            project_root = project_root.parent
        
        v6_dataset_dir = project_root / "v6_dataset"
        v6_stage3_dir = project_root / "v6_dataset_stage3"
        
        if 'AB' in args.noise_sources:
            ab_path = v6_stage3_dir / "AB" / "block_16" / "train.pt"
            if ab_path.exists():
                print(f"  Loading AB samples from: {ab_path}")
                ab_data = torch.load(ab_path, weights_only=False)
                ab_samples = ab_data['samples'] if isinstance(ab_data['samples'], np.ndarray) else ab_data['samples'].numpy()
                ab_labels = ab_data['labels'] if isinstance(ab_data['labels'], np.ndarray) else ab_data['labels'].numpy()
                ab_qps = ab_data['qps'] if isinstance(ab_data['qps'], np.ndarray) else ab_data['qps'].numpy()
                
                ab_record = BlockRecord(
                    samples=ab_samples,
                    labels=ab_labels,
                    qps=ab_qps.reshape(-1, 1)
                )
                noise_records.append(ab_record)
            else:
                print(f"  WARNING: AB dataset not found at {ab_path}")
        
        if 'SPLIT' in args.noise_sources:
            main_path = v6_dataset_dir / "block_16" / "train.pt"
            if main_path.exists():
                print(f"  Loading SPLIT samples from: {main_path}")
                main_data = torch.load(main_path, weights_only=False)
                main_samples = main_data['samples'] if isinstance(main_data['samples'], np.ndarray) else main_data['samples'].numpy()
                main_labels = main_data['labels_stage0'] if isinstance(main_data['labels_stage0'], np.ndarray) else main_data['labels_stage0'].numpy()
                main_qps = main_data['qps'] if isinstance(main_data['qps'], np.ndarray) else main_data['qps'].numpy()
                
                # Convert from (N, C, H, W) to (N, H, W, C) format
                if main_samples.shape[1] == 1:  # Check if it's (N, C, H, W)
                    main_samples = np.transpose(main_samples, (0, 2, 3, 1))
                
                # Filter SPLIT only (label = 3)
                split_mask = main_labels == 3
                split_samples = main_samples[split_mask]
                split_labels = main_labels[split_mask]
                split_qps = main_qps[split_mask]
                
                print(f"    Found {len(split_samples)} SPLIT samples")
                
                split_record = BlockRecord(
                    samples=split_samples,
                    labels=split_labels,
                    qps=split_qps.reshape(-1, 1)
                )
                noise_records.append(split_record)
            else:
                print(f"  WARNING: Main dataset not found at {main_path}")
        
        if noise_records:
            train_dataset = NoisyDataset(
                train_dataset_clean, 
                noise_records, 
                noise_ratio=args.noise_injection,
                augmentation=train_aug,
                seed=args.seed
            )
        else:
            print("  WARNING: No noise sources found, using clean dataset")
            train_dataset = train_dataset_clean
    else:
        train_dataset = train_dataset_clean
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    
    # Create model
    print(f"\n[2/5] Creating model...")
    model = Stage3RectModel(pretrained=True).to(device)
    
    # Load Stage 2 backbone
    if Path(args.stage2_model).exists():
        print(f"  Loading Stage 2 backbone from: {args.stage2_model}")
        checkpoint = torch.load(args.stage2_model, map_location=device, weights_only=False)
        stage2_state = checkpoint['model_state_dict']
        backbone_state = {k.replace('backbone.', ''): v for k, v in stage2_state.items() if k.startswith('backbone.')}
        model.backbone.load_state_dict(backbone_state, strict=False)
        print(f"  ✅ Backbone initialized from Stage 2")
    else:
        print(f"  ⚠️  Stage 2 model not found, using random initialization")
    
    # Freeze backbone initially
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (head only, backbone frozen for first 5 epochs)")
    
    # Compute class distribution for weighted loss
    print(f"\n[3/5] Computing class weights...")
    train_labels = train_data['labels'] if isinstance(train_data['labels'], np.ndarray) else train_data['labels'].numpy()
    unique, counts = np.unique(train_labels, return_counts=True)
    total = counts.sum()
    class_weights = total / (len(unique) * counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"  Class distribution:")
    for cls, cnt, weight in zip(unique, counts, class_weights):
        class_name = "HORZ" if cls == 0 else "VERT"
        print(f"    {class_name}: {cnt:6d} samples ({cnt/total*100:5.2f}%), weight: {weight:.3f}")
    
    # Label Smoothing (Szegedy et al., 2016)
    print(f"\n[4/5] Creating loss function...")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    print(f"  Using: CrossEntropyLoss + Class Weights + Label Smoothing (0.1)")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=3e-6
    )
    
    # Training loop
    print(f"\n[5/5] Training...")
    best_f1 = 0.0
    best_epoch = 0
    # Early Stopping (Goodfellow et al., 2016)
    patience = 5
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'val_precision': [], 'val_recall': [], 'val_balanced_acc': []
    }
    
    # Progressive Unfreezing (Howard & Ruder, 2018)
    freeze_backbone_until = 5
    backbone_unfrozen = False
    
    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after freeze_backbone_until epochs
        freeze_backbone = epoch <= freeze_backbone_until
        
        if not freeze_backbone and not backbone_unfrozen:
            print(f"\n  🔓 Unfreezing backbone at epoch {epoch}")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Add backbone parameters to optimizer with lower LR
            optimizer.add_param_group({
                'params': model.backbone.parameters(),
                'lr': args.lr * 0.01  # 100x lower LR for backbone
            })
            backbone_unfrozen = True
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, freeze_backbone)
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        backbone_status = "[FROZEN]" if freeze_backbone else "[UNFROZEN]"
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
        
        print(f"\nEpoch {epoch}/{args.epochs} - LR: {current_lr:.2e} {backbone_status}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}")
        print(f"  Val   - F1: {val_metrics['f1']:.2%}, Prec: {val_metrics['precision']:.2%}, Rec: {val_metrics['recall']:.2%}")
        print(f"  Val   - Balanced Acc: {val_metrics['balanced_accuracy']:.2%}")
        
        # Save best model ONLY after unfreezing backbone (avoid saving collapsed model)
        if not freeze_backbone and val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0  # Reset patience counter
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
            }, output_dir / "stage3_rect_model_best.pt")
            print(f"  ✅ New best F1: {best_f1:.2%} (epoch {epoch})")
        elif not freeze_backbone:
            # Early Stopping (Goodfellow et al., 2016)
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⚠️  Early stopping! No improvement for {patience} epochs.")
                print(f"  Best F1: {best_f1:.2%} at epoch {best_epoch}")
                break
    
    # Save final artifacts
    print(f"\n[6/6] Saving artifacts...")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, output_dir / "stage3_rect_model_final.pt")
    
    # Save history
    torch.save(history, output_dir / "stage3_rect_history.pt")
    
    # Save metrics summary
    metrics_summary = {
        'best_f1': float(best_f1),
        'best_epoch': int(best_epoch),
        'final_val_f1': float(history['val_f1'][-1]),
        'final_val_precision': float(history['val_precision'][-1]),
        'final_val_recall': float(history['val_recall'][-1]),
        'final_val_balanced_acc': float(history['val_balanced_acc'][-1]),
        'total_epochs_trained': len(history['val_f1']),
        'early_stopped': len(history['val_f1']) < args.epochs,
        'config': vars(args),
    }
    
    with open(output_dir / "stage3_rect_metrics.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  ✅ Training completed!")
    print(f"  Best F1: {best_f1:.2%} (epoch {best_epoch})")
    print(f"  Total epochs: {len(history['val_f1'])}/{args.epochs}")
    if metrics_summary['early_stopped']:
        print(f"  Early stopped: No improvement for {patience} epochs")
    print(f"  Models saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
