"""
Script 003: Train Stage 1 Improved
Stage 1: Binary classification (NONE vs PARTITION)

Improvements based on literature:
1. Focal Loss (Lin et al., 2017) - Addresses class imbalance
2. Hard Negative Mining (Shrivastava et al., 2016) - Focuses on difficult examples
3. Balanced Sampling (Chawla et al., 2002) - Handles imbalanced datasets
4. Data Augmentation (Shorten & Khoshgoftaar, 2019) - Improves generalization

References:
- Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
- Shrivastava, A., et al. (2016). Training Region-based Object Detectors with Online Hard Example Mining. CVPR.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation. Journal of Big Data.
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
from data_hub import HierarchicalBlockDatasetV6, create_balanced_sampler
from models import Stage1Model
from losses import FocalLoss, HardNegativeMiningLoss
from augmentation import Stage1Augmentation
from metrics import compute_binary_metrics


# Simple batch metrics accumulator
class BatchMetricsAccumulator:
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
        metrics = compute_binary_metrics(all_labels, all_preds)
        metrics['loss'] = avg_loss
        return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    metrics = BatchMetricsAccumulator()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label_stage1'].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs > 0.5).long()
            
        metrics.update(loss.item(), labels, preds)
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return metrics.get_average()


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    metrics = BatchMetricsAccumulator()
    
    all_probs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label_stage1'].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs > 0.5).long()
            
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            
            metrics.update(loss.item(), labels, preds)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    # Compute detailed metrics
    all_preds = (all_probs > 0.5).long()
    binary_metrics = compute_binary_metrics(
        all_labels.numpy(), 
        all_preds.numpy(), 
        y_scores=all_probs.numpy()
    )
    avg_metrics = metrics.get_average()
    avg_metrics.update(binary_metrics)
    
    return avg_metrics, all_probs, all_labels


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Stage 1 Improved")
    parser.add_argument("--dataset-dir", type=str, 
                       default="pesquisa_v6/v6_dataset/block_16",
                       help="Dataset directory")
    parser.add_argument("--output-dir", type=str,
                       default="pesquisa_v6/logs/v6_experiments/stage1",
                       help="Output directory for logs and models")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--focal-gamma", type=float, default=2.5,
                       help="Focal loss gamma")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                       help="Focal loss alpha")
    parser.add_argument("--use-hard-mining", action="store_true",
                       help="Use hard negative mining")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
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
    print(f"  Training Stage 1 - Binary Classification (NONE vs PARTITION)")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Focal Loss - gamma: {args.focal_gamma}, alpha: {args.focal_alpha}")
    print(f"  Hard mining: {args.use_hard_mining}")
    
    # Load datasets
    print(f"\n[1/6] Loading datasets...")
    train_data = torch.load(dataset_dir / "train.pt")
    val_data = torch.load(dataset_dir / "val.pt")
    
    # Data Augmentation (Shorten & Khoshgoftaar, 2019)
    train_aug = Stage1Augmentation(train=True)
    val_aug = Stage1Augmentation(train=False)
    
    # Create datasets
    from data_hub import BlockRecord, build_hierarchical_dataset_v6
    
    train_record = BlockRecord(
        samples=train_data['samples'].numpy().transpose(0, 2, 3, 1),
        labels=train_data['labels_stage0'].numpy(),
        qps=train_data['qps'].numpy().reshape(-1, 1)
    )
    val_record = BlockRecord(
        samples=val_data['samples'].numpy().transpose(0, 2, 3, 1),
        labels=val_data['labels_stage0'].numpy(),
        qps=val_data['qps'].numpy().reshape(-1, 1)
    )
    
    train_dataset = build_hierarchical_dataset_v6(train_record, augmentation=train_aug, stage='stage1')
    val_dataset = build_hierarchical_dataset_v6(val_record, augmentation=val_aug, stage='stage1')
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Balanced Sampling (Chawla et al., 2002)
    print(f"\n[2/6] Creating balanced sampler...")
    train_labels = train_data['labels_stage1'].numpy()
    sampler = create_balanced_sampler(train_labels, oversample_factor=None)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
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
    print(f"\n[3/6] Creating model...")
    model = Stage1Model(pretrained=True).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss
    print(f"\n[4/6] Creating loss function...")
    # Focal Loss (Lin et al., 2017)
    base_criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    # Hard Negative Mining (Shrivastava et al., 2016)
    if args.use_hard_mining:
        criterion = HardNegativeMiningLoss(base_criterion, neg_pos_ratio=3.0)
        print(f"  Using Focal Loss + Hard Negative Mining (ratio 3:1)")
    else:
        criterion = base_criterion
        print(f"  Using Focal Loss only")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )
    
    # Training loop
    print(f"\n[5/6] Training...")
    best_f1 = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'val_precision': [], 'val_recall': []
    }
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics, val_probs, val_labels = validate_epoch(model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        print(f"\nEpoch {epoch}/{args.epochs} - LR: {current_lr:.2e}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}")
        print(f"  Val   - F1: {val_metrics['f1']:.2%}, Prec: {val_metrics['precision']:.2%}, Rec: {val_metrics['recall']:.2%}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
            }, output_dir / "stage1_model_best.pt")
            print(f"  ✅ New best F1: {best_f1:.2%}")
    
    # Save final artifacts
    print(f"\n[6/6] Saving artifacts...")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, output_dir / "stage1_model_final.pt")
    
    # Save history
    torch.save(history, output_dir / "stage1_history.pt")
    
    # Save metrics summary
    metrics_summary = {
        'best_f1': float(best_f1),
        'final_val_f1': float(history['val_f1'][-1]),
        'final_val_precision': float(history['val_precision'][-1]),
        'final_val_recall': float(history['val_recall'][-1]),
        'config': vars(args),
    }
    
    with open(output_dir / "stage1_metrics.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  ✅ Training completed!")
    print(f"  Best F1: {best_f1:.2%}")
    print(f"  Models saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
