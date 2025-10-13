#!/usr/bin/env python3
"""
Script 004c: Train Stage 2 Flat with Pipeline-Aware Distribution

**CRITICAL EXPERIMENT: Tests Distribution Shift Hypothesis (H2.1)**

Problem Statement:
- Stage 2 Flat trained standalone: F1=31.65% ✅ (works well)
- Stage 2 Flat in pipeline: F1≈0% ❌ (complete collapse for PARTITION classes)
- Hypothesis H2.1: Training distribution ≠ pipeline inference distribution

Root Cause Analysis:
1. **Training (script 004b)**: Balanced 7-class dataset
   - All 152,600 samples used equally (via WeightedRandomSampler)
   - Model learns features that discriminate HORZ vs VERT vs SPLIT, etc.
   - Distribution: artificially balanced by sampler

2. **Pipeline Inference (script 008b)**: Stage 1 filtered samples
   - Stage 1 (threshold 0.45) predicts 98.81% as NONE
   - Only 1.19% samples reach Stage 2 Flat
   - These samples have DIFFERENT characteristics than training set
   - Distribution: heavily filtered by Stage 1's decision boundary

3. **Covariate Shift** (Shimodaira, 2000; Ben-David et al., 2010):
   - P_train(X) ≠ P_pipeline(X|Stage1=PARTITION)
   - Model trained on one distribution fails on another
   - Common in cascaded systems (Kumar et al., 2012)

Pipeline-Aware Solution:
- **Train Stage 2 with REALISTIC pipeline distribution**
- Filter training samples through Stage 1 (threshold 0.45)
- Use only samples predicted as PARTITION
- This matches training and inference distributions

Expected Outcome:
- If H2.1 is correct: F1-macro should improve from 10.24% to >40%
- If H2.1 is wrong: F1-macro remains low, indicating other issues
  (e.g., Stage 1 features are insufficient for Stage 2 discrimination)

Scientific Contribution:
- Tests domain adaptation necessity in cascaded CNN pipelines
- Demonstrates that component-wise training can fail due to distribution shift
- Validates pipeline-aware training as solution (Zhang et al., 2021)

Key Design Decisions:
1. **Pipeline-Aware Dataset Creation**:
   - Load Stage 1 model checkpoint
   - Filter all training samples through Stage 1
   - Keep only PARTITION predictions (prob ≥ 0.45)
   - Result: ~1-2% of original dataset (realistic pipeline distribution)

2. **Same Training Protocol as 004b**:
   - Class-Balanced Focal Loss (Cui et al., 2019)
   - Balanced sampling (Buda et al., 2018)
   - Strong augmentation (MixUp, CutMix)
   - OneCycleLR scheduler
   - Two-phase training (freeze → unfreeze)
   
3. **Evaluation**:
   - Validate on Stage 1 filtered validation set (matching inference)
   - Compare with 004b baseline (F1=31.65% on full dataset)
   - Test in full pipeline (008b) to verify improvement

Success Criteria:
- ✅ F1-macro >40% on pipeline-filtered validation set
- ✅ F1-macro >40% in full pipeline evaluation (008b)
- ✅ All 7 classes have F1 >0% (no collapse)

Failure Analysis:
- If fails: Distribution shift NOT the root cause
- Alternative hypotheses:
  * Stage 1 features insufficient for fine-grained discrimination
  * Model capacity too small for complex decision boundary
  * Need end-to-end training (Option B: Multi-Task)

References:
- Shimodaira, H. (2000). Improving predictive inference under covariate shift. JMLR.
- Ben-David, S., et al. (2010). A theory of learning from different domains. Machine Learning.
- Kumar, M. P., et al. (2012). Learning graphs to match. ICCV.
- Zhang, H., et al. (2021). Understanding deep learning requires rethinking generalization. ICLR.
- Cui, Y., et al. (2019). Class-Balanced Loss Based on Effective Number of Samples. CVPR.
- Buda, M., et al. (2018). A systematic study of the class imbalance problem in CNNs. Neural Networks.

Author: Chiaro Rosa (PhD Research - AV1 Partition Prediction)
Date: 2025-10
Version: v6-flatten-pipeline-aware
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import numpy as np
import json
from tqdm import tqdm
from collections import Counter

# Add v6 pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "v6_pipeline"))
from data_hub import FLATTEN_ID_TO_NAME
from models import Stage1Model, ImprovedBackbone
from losses import ClassBalancedFocalLoss
from augmentation import Stage2Augmentation
from metrics import compute_metrics


# ---------------------------------------------------------------------------
# Stage 2 Flat Model Definition
# ---------------------------------------------------------------------------

class Stage2FlatModel(nn.Module):
    """
    Stage 2 Flat: 7-way direct classification
    Architecture: ResNet-18 (ImageNet pretrained) + SE blocks + Spatial Attention + 7-class head
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        self.backbone = ImprovedBackbone(pretrained=pretrained)
        
        # 7-class classification head
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)  # (B, 512)
        logits = self.head(features)  # (B, 7)
        return logits


# ---------------------------------------------------------------------------
# Step 1: Filter Training Data Through Stage 1
# ---------------------------------------------------------------------------

def filter_dataset_through_stage1(
    dataset_path: Path,
    stage1_model_path: Path,
    threshold: float,
    device: torch.device,
    batch_size: int = 256
):
    """
    Filter dataset through Stage 1 to create pipeline-aware training set.
    
    Args:
        dataset_path: Path to original dataset (train.pt or val.pt)
        stage1_model_path: Path to trained Stage 1 checkpoint
        threshold: Probability threshold for PARTITION prediction
        device: Device for inference
        batch_size: Batch size for filtering
    
    Returns:
        dict with filtered samples, labels, qps
    """
    print(f"\n[FILTER] Loading dataset from: {dataset_path}")
    data = torch.load(dataset_path, weights_only=False)
    
    samples = data['samples']  # (N, C, H, W)
    labels = data['labels']    # (N,) flatten labels 0-6
    qps = data['qps']          # (N,)
    
    print(f"  Original dataset size: {len(samples)}")
    
    # Load Stage 1 model
    print(f"\n[FILTER] Loading Stage 1 model from: {stage1_model_path}")
    stage1_model = Stage1Model(pretrained=False).to(device)
    checkpoint = torch.load(stage1_model_path, weights_only=False)
    stage1_model.load_state_dict(checkpoint['model_state_dict'])
    stage1_model.eval()
    
    # Filter samples through Stage 1
    print(f"\n[FILTER] Filtering samples (threshold={threshold})...")
    partition_indices = []
    partition_probs = []
    
    dataset = TensorDataset(samples, labels, qps)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for batch_idx, (batch_samples, batch_labels, batch_qps) in enumerate(tqdm(loader, desc="Filtering")):
            batch_samples = batch_samples.to(device)
            
            # Stage 1 inference
            logits = stage1_model(batch_samples)
            probs = torch.sigmoid(logits.squeeze())
            
            # Keep samples predicted as PARTITION (prob >= threshold)
            mask = probs >= threshold
            local_indices = torch.where(mask)[0]
            global_indices = batch_idx * batch_size + local_indices.cpu()
            
            partition_indices.extend(global_indices.tolist())
            partition_probs.extend(probs[mask].cpu().tolist())
    
    partition_indices = np.array(partition_indices, dtype=np.int64)
    partition_probs = np.array(partition_probs, dtype=np.float32)
    
    # Filter dataset
    filtered_samples = samples[partition_indices]
    filtered_labels = labels[partition_indices]
    filtered_qps = qps[partition_indices]
    
    print(f"\n[FILTER] Results:")
    print(f"  Original samples: {len(samples):,}")
    print(f"  Filtered samples: {len(filtered_samples):,}")
    print(f"  Retention rate: {len(filtered_samples)/len(samples)*100:.2f}%")
    print(f"  Mean PARTITION prob: {partition_probs.mean():.4f} ± {partition_probs.std():.4f}")
    
    # Class distribution after filtering
    counter = Counter(filtered_labels.tolist())
    print(f"\n[FILTER] Filtered class distribution:")
    for class_id in range(7):
        count = counter.get(class_id, 0)
        percentage = (count / len(filtered_labels)) * 100 if len(filtered_labels) > 0 else 0
        print(f"    {class_id} ({FLATTEN_ID_TO_NAME.get(class_id, 'UNKNOWN')}): "
              f"{count:6d} ({percentage:5.2f}%)")
    
    return {
        'samples': filtered_samples,
        'labels': filtered_labels,
        'qps': filtered_qps,
        'stage1_probs': partition_probs,
        'original_indices': partition_indices
    }


# ---------------------------------------------------------------------------
# Dataset for Pipeline-Aware Training
# ---------------------------------------------------------------------------

class PipelineAwareDataset(torch.utils.data.Dataset):
    """
    Dataset with samples filtered through Stage 1.
    Matches realistic pipeline distribution.
    """
    def __init__(self, filtered_data: dict, augmentation=None, split='train'):
        self.samples = filtered_data['samples']
        self.labels = filtered_data['labels']
        self.qps = filtered_data['qps']
        self.augmentation = augmentation
        self.split = split
        
        print(f"\n  PipelineAwareDataset ({split}):")
        print(f"    Samples: {len(self.labels)}")
        print(f"    Label range: [{self.labels.min()}, {self.labels.max()}]")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.samples[idx]
        label = self.labels[idx]
        qp = self.qps[idx]
        
        if self.augmentation is not None:
            image = self.augmentation(image)
        
        return {
            'image': image,
            'label': label,
            'qp': qp
        }


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------

def create_balanced_sampler_from_labels(labels: torch.Tensor, oversample_factor: float = 2.0):
    """Create WeightedRandomSampler for balanced training"""
    counter = Counter(labels.tolist())
    n_classes = len(counter)
    
    # Compute class weights
    class_counts = torch.tensor([counter[i] for i in range(n_classes)], dtype=torch.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    
    # Sample weights
    sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.float32)
    
    # Number of samples per epoch
    n_samples = int(len(labels) * oversample_factor)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_samples,
        replacement=True
    )
    
    print(f"\n  Balanced Sampler:")
    print(f"    Samples per epoch: {n_samples:,} ({oversample_factor}x dataset)")
    print(f"    Class weights: {class_weights.tolist()}")
    
    return sampler


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, freeze_backbone):
    """Train for one epoch"""
    model.train()
    
    # Freeze/unfreeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = not freeze_backbone
    
    losses = []
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:2d} [Train, {'Frozen' if freeze_backbone else 'Unfrozen'}]")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            losses.append(loss.item())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Update scheduler (OneCycleLR updates per batch)
    if scheduler is not None:
        scheduler.step()
    
    # Compute metrics
    avg_loss = np.mean(losses)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss
    
    return metrics


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    
    losses = []
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:2d} [Val]")
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            preds = torch.argmax(logits, dim=1)
            losses.append(loss.item())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute metrics
    avg_loss = np.mean(losses)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss
    
    return metrics


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Stage 2 Flat with Pipeline-Aware Distribution")
    parser.add_argument("--dataset-dir", type=str,
                       default="pesquisa_v6/v6_dataset_flatten/block_16",
                       help="Original flatten dataset directory")
    parser.add_argument("--stage1-model", type=str,
                       default="pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt",
                       help="Path to trained Stage 1 model")
    parser.add_argument("--threshold", type=float, default=0.45,
                       help="Stage 1 threshold for PARTITION prediction")
    parser.add_argument("--output-dir", type=str,
                       default="pesquisa_v6/logs/v6_experiments/stage2_pipeline_aware",
                       help="Output directory for logs and models")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--freeze-epochs", type=int, default=15,
                       help="Number of epochs to freeze backbone")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--lr-backbone", type=float, default=5e-4,
                       help="Learning rate for backbone")
    parser.add_argument("--lr-head", type=float, default=2e-3,
                       help="Learning rate for head")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--focal-beta", type=float, default=0.9999,
                       help="CB-Focal Loss beta")
    parser.add_argument("--focal-gamma", type=float, default=2.5,
                       help="CB-Focal Loss gamma")
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
    stage1_model_path = Path(args.stage1_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"  Stage 2 Flat - Pipeline-Aware Training")
    print(f"  Testing Distribution Shift Hypothesis (H2.1)")
    print(f"{'='*80}")
    print(f"  Device: {device}")
    print(f"  Original dataset: {dataset_dir}")
    print(f"  Stage 1 model: {stage1_model_path}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {args.epochs} ({args.freeze_epochs} frozen + {args.epochs - args.freeze_epochs} unfrozen)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rates: backbone={args.lr_backbone}, head={args.lr_head}")
    
    # Step 1: Filter training data through Stage 1
    print(f"\n{'='*80}")
    print(f"  [1/6] Filtering Training Data Through Stage 1")
    print(f"{'='*80}")
    
    train_filtered = filter_dataset_through_stage1(
        dataset_path=dataset_dir / "train.pt",
        stage1_model_path=stage1_model_path,
        threshold=args.threshold,
        device=device,
        batch_size=256
    )
    
    val_filtered = filter_dataset_through_stage1(
        dataset_path=dataset_dir / "val.pt",
        stage1_model_path=stage1_model_path,
        threshold=args.threshold,
        device=device,
        batch_size=256
    )
    
    # Save filtered datasets for reproducibility
    print(f"\n  Saving filtered datasets...")
    torch.save(train_filtered, output_dir / "train_filtered.pt")
    torch.save(val_filtered, output_dir / "val_filtered.pt")
    print(f"    Saved: {output_dir / 'train_filtered.pt'}")
    print(f"    Saved: {output_dir / 'val_filtered.pt'}")
    
    # Step 2: Create datasets
    print(f"\n{'='*80}")
    print(f"  [2/6] Creating Pipeline-Aware Datasets")
    print(f"{'='*80}")
    
    train_aug = Stage2Augmentation(train=True)
    val_aug = Stage2Augmentation(train=False)
    
    train_dataset = PipelineAwareDataset(train_filtered, augmentation=train_aug, split='train')
    val_dataset = PipelineAwareDataset(val_filtered, augmentation=val_aug, split='val')
    
    # Step 3: Create balanced sampler
    print(f"\n{'='*80}")
    print(f"  [3/6] Creating Balanced Sampler")
    print(f"{'='*80}")
    
    sampler = create_balanced_sampler_from_labels(train_filtered['labels'], oversample_factor=2.0)
    
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
    
    # Step 4: Create model
    print(f"\n{'='*80}")
    print(f"  [4/6] Creating Stage 2 Flat Model")
    print(f"{'='*80}")
    
    model = Stage2FlatModel(num_classes=7, pretrained=True).to(device)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Backbone parameters: {sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"  Head parameters: {sum(p.numel() for p in model.head.parameters()):,}")
    
    # Step 5: Create loss and optimizer
    print(f"\n{'='*80}")
    print(f"  [5/6] Creating Loss and Optimizer")
    print(f"{'='*80}")
    
    # Class-Balanced Focal Loss
    samples_per_class = torch.tensor([
        (train_filtered['labels'] == i).sum().item() for i in range(7)
    ], dtype=torch.float32)
    
    criterion = ClassBalancedFocalLoss(
        samples_per_class=samples_per_class,
        beta=args.focal_beta,
        gamma=args.focal_gamma
    ).to(device)
    
    print(f"  Loss: Class-Balanced Focal Loss")
    print(f"    Beta: {args.focal_beta}")
    print(f"    Gamma: {args.focal_gamma}")
    print(f"    Samples per class: {samples_per_class.tolist()}")
    
    # Optimizer with discriminative learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': model.head.parameters(), 'lr': args.lr_head}
    ], weight_decay=args.weight_decay)
    
    # OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone, args.lr_head],
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    print(f"  Optimizer: AdamW with discriminative LR")
    print(f"  Scheduler: OneCycleLR")
    print(f"    Steps per epoch: {steps_per_epoch}")
    print(f"    Total steps: {total_steps}")
    
    # Step 6: Training loop
    print(f"\n{'='*80}")
    print(f"  [6/6] Training Loop")
    print(f"{'='*80}")
    
    history = {
        'train_loss': [], 'train_f1': [], 'train_acc': [],
        'val_loss': [], 'val_f1': [], 'val_acc': []
    }
    
    best_f1 = 0.0
    patience_counter = 0
    patience = 8
    
    for epoch in range(1, args.epochs + 1):
        # Freeze/unfreeze backbone
        freeze_backbone = epoch <= args.freeze_epochs
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, freeze_backbone
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Log metrics
        print(f"\n  Epoch {epoch:2d} Summary:")
        print(f"    Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['macro_f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"    Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['macro_f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_f1'].append(train_metrics['macro_f1'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['macro_f1'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Save best model
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'args': vars(args)
            }, output_dir / "stage2_pipeline_aware_best.pt")
            
            print(f"    ✅ New best F1: {best_f1:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"    Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break
    
    # Save final model and history
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': val_metrics,
        'args': vars(args)
    }, output_dir / "stage2_pipeline_aware_final.pt")
    
    torch.save(history, output_dir / "stage2_pipeline_aware_history.pt")
    
    # Save final metrics
    with open(output_dir / "stage2_pipeline_aware_metrics.json", 'w') as f:
        json.dump({
            'best_val_f1': best_f1,
            'final_val_metrics': val_metrics,
            'args': vars(args),
            'experiment': 'pipeline_aware_training',
            'hypothesis': 'H2.1_distribution_shift'
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"  Training Complete!")
    print(f"{'='*80}")
    print(f"  Best Val F1: {best_f1:.4f}")
    print(f"  Final Val Acc: {val_metrics['accuracy']:.4f}")
    print(f"  Saved to: {output_dir}")
    print(f"\n  Next Step: Run pipeline evaluation (008b) with new checkpoint")
    print(f"  Command:")
    print(f"    python3 pesquisa_v6/scripts/008b_run_pipeline_flatten_eval.py \\")
    print(f"      --stage1-model {stage1_model_path} \\")
    print(f"      --stage2-model {output_dir / 'stage2_pipeline_aware_best.pt'} \\")
    print(f"      --threshold {args.threshold}")


if __name__ == "__main__":
    main()
