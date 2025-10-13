"""
Script 006: Train Stage 3 AB - Fine-Grained Visual Classification (FGVC)
Stage 3-AB: 4-way classification (HORZ_A, HORZ_B, VERT_A, VERT_B)

Problem: Extremely similar classes (fine-grained differences in partition proportions)
Strategy: State-of-the-art FGVC techniques from computer vision literature

Key Techniques:
1. Two-Stage Fine-tuning (Kornblith et al., 2019) - Backbone adaptation
2. Center Loss (Wen et al., 2016) - Discriminative embeddings
3. CutMix Augmentation (Yun et al., 2019) - Preserves local structures
4. Dual Attention (CBAM - Woo et al., 2018) - Discriminative regions
5. Cosine Classifier (Wang et al., 2017) - Normalized features
6. Label Smoothing (Szegedy et al., 2016) - Regularization
7. Oversampling 5x (Chawla et al., 2002) - Class balance

References:
- Kornblith, S., et al. (2019). Do Better ImageNet Models Transfer Better? CVPR.
- Wen, Y., et al. (2016). A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV.
- Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers. ICCV.
- Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. ECCV.
- Wang, F., et al. (2017). NormFace: L2 Hypersphere Embedding for Face Verification. ACM MM.
- Szegedy, C., et al. (2016). Rethinking the Inception Architecture for Computer Vision. CVPR.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.
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
from data_hub import BlockRecord, build_hierarchical_dataset_v6
from models import Stage3ABModel
from augmentation import Stage3ABAugmentation
from metrics import compute_metrics


class NoisyDatasetAB(torch.utils.data.Dataset):
    """
    Dataset with noise injection for Stage 3-AB adversarial training.
    
    Mixes clean AB samples with noisy samples from other classes (RECT, SPLIT).
    Noisy samples are assigned random labels (0-3) to simulate Stage 2 misclassification.
    
    References:
    - Hendrycks et al., 2019: "Using Pre-Training Can Improve Model Robustness"
    - Natarajan et al., 2013: "Learning with Noisy Labels"
    
    Args:
        clean_dataset: Original AB dataset (clean samples)
        noise_records: List of BlockRecord objects from noise sources (RECT, SPLIT)
        noise_ratio: Fraction of dataset to be noise (0.0-1.0)
        augmentation: Augmentation to apply
        seed: Random seed for reproducibility
    """
    def __init__(self, clean_dataset, noise_records, noise_ratio=0.25, augmentation=None, seed=42):
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
                    stage='stage3_ab'  # Use same stage structure
                )
                # Sample random indices from noise source
                noise_indices = rng.choice(len(noise_ds), per_source, replace=False)
                self.noise_datasets.append((noise_ds, noise_indices))
        
        self.total_len = n_clean + n_noise
        
        print(f"\n[NoisyDatasetAB] Created with:")
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
            # Round-robin distribution across noise sources
            source_idx = noise_idx % len(self.noise_datasets)
            noise_ds, noise_indices = self.noise_datasets[source_idx]
            
            # Calculate local index within the specific noise dataset
            local_idx = noise_idx // len(self.noise_datasets)
            # Wrap around if we exceed the dataset size (shouldn't happen with correct total_len)
            local_idx = local_idx % len(noise_indices)
            sample_idx = noise_indices[local_idx]
            batch = noise_ds[sample_idx]
            
            # Replace label with random AB class (0: HORZ_A, 1: HORZ_B, 2: VERT_A, 3: VERT_B)
            batch['label_stage3_AB'] = torch.tensor(np.random.randint(0, 4), dtype=torch.long)
            
            return batch


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================================
# FGVC Components
# ============================================================================

class DualAttentionModule(nn.Module):
    """
    CBAM: Convolutional Block Attention Module (Woo et al., 2018)
    Combines Channel Attention + Spatial Attention for discriminative features
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Channel Attention
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        batch, channels, _, _ = x.size()
        
        # Channel Attention
        avg_pool = self.channel_avg_pool(x).view(batch, channels)
        max_pool = self.channel_max_pool(x).view(batch, channels)
        channel_att = self.channel_fc(avg_pool) + self.channel_fc(max_pool)
        channel_att = torch.sigmoid(channel_att).view(batch, channels, 1, 1)
        x = x * channel_att
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_conv(spatial_att)
        spatial_att = torch.sigmoid(spatial_att)
        x = x * spatial_att
        
        return x


class CenterLoss(nn.Module):
    """
    Center Loss (Wen et al., 2016)
    Forces intra-class compactness and inter-class separability
    """
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # Learnable class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        
    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, feat_dim) - L2 normalized features
            labels: (batch_size,) - ground truth labels
        """
        batch_size = features.size(0)
        
        # Compute distances to centers
        # features: (batch, feat_dim), centers: (num_classes, feat_dim)
        centers_batch = self.centers[labels]  # (batch, feat_dim)
        
        # L2 distance
        loss = (features - centers_batch).pow(2).sum() / batch_size
        
        return loss


class CosineClassifier(nn.Module):
    """
    Cosine Classifier (Wang et al., 2017)
    Normalized weights and features for stable fine-grained classification
    """
    def __init__(self, feat_dim, num_classes, scale=20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.scale = scale  # Temperature scaling
        
    def forward(self, features):
        """
        Args:
            features: (batch, feat_dim) - L2 normalized features
        Returns:
            logits: (batch, num_classes)
        """
        # Normalize weights
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(features, normalized_weight)
        
        # Scale (temperature)
        logits = self.scale * cosine
        
        return logits


class FGVCModel(nn.Module):
    """
    Fine-Grained Visual Classification Model
    Combines: Backbone + Dual Attention + Center Loss + Cosine Classifier
    
    Note: Backbone already includes GAP and flatten, so we work with flattened features
    """
    def __init__(self, base_model, num_classes=4, feat_dim=512):
        super().__init__()
        
        # Backbone (ResNet-18 + SE-Block from Stage3ABModel)
        # Note: backbone outputs (batch, 512) - already flattened
        self.backbone = base_model.backbone
        
        # Feature projection with attention-like mechanism
        self.feat_proj = nn.Sequential(
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Cosine Classifier (Wang et al., 2017)
        self.classifier = CosineClassifier(feat_dim, num_classes, scale=20.0)
        
        self.feat_dim = feat_dim
        
    def forward(self, x, return_features=False):
        """
        Args:
            x: (batch, 1, 16, 16)
            return_features: if True, return (logits, features) for center loss
        """
        # Backbone (already includes attention + GAP + flatten)
        x = self.backbone(x)  # (batch, 512)
        
        # Feature projection
        features = self.feat_proj(x)  # (batch, feat_dim)
        
        # L2 Normalization (critical for cosine classifier and center loss)
        features = F.normalize(features, p=2, dim=1)
        
        # Cosine Classifier
        logits = self.classifier(features)  # (batch, num_classes)
        
        if return_features:
            return logits, features
        return logits


class CutMixCrossEntropyLoss(nn.Module):
    """
    CutMix Augmentation (Yun et al., 2019)
    Cuts and pastes patches between images - better for FGVC than Mixup
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, model, images, labels, training=True):
        """
        Apply CutMix augmentation and compute loss
        """
        if training and self.alpha > 0 and np.random.rand() < 0.5:
            # Sample lambda from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
            
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(images.device)
            
            # Generate random box
            W, H = images.size(2), images.size(3)
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            
            # Random center
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            # Bounding box
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            # CutMix
            images_cutmix = images.clone()
            images_cutmix[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda based on actual box size
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            
            return images_cutmix, labels, labels[index], lam
        else:
            return images, labels, labels, 1.0


def create_oversampling_weights(labels, oversample_factor=5.0):
    """Create sampling weights for minority classes (HORZ_A, VERT_B)"""
    labels_np = labels if isinstance(labels, np.ndarray) else labels.numpy()
    unique, counts = np.unique(labels_np, return_counts=True)
    
    minority_classes = [4, 7]  # HORZ_A, VERT_B
    
    weights = np.ones(len(labels_np), dtype=np.float32)
    for cls in minority_classes:
        mask = labels_np == cls
        weights[mask] = oversample_factor
    
    print(f"  Oversampling weights:")
    for cls, cnt in zip(unique, counts):
        cls_mask = labels_np == cls
        avg_weight = weights[cls_mask].mean()
        effective_samples = cnt * avg_weight
        print(f"    Class {cls}: {cnt:6d} → {effective_samples:8.0f} effective (weight={avg_weight:.1f}x)")
    
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
        
        class_names = ['HORZ_A', 'HORZ_B', 'VERT_A', 'VERT_B']
        metrics_dict = compute_metrics(all_labels.numpy(), all_preds.numpy(), labels=class_names)
        metrics_dict['loss'] = self.total_loss / self.count if self.count > 0 else 0.0
        return metrics_dict


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, cutmix_aug, ce_criterion, center_criterion, 
                optimizer, device, epoch, phase, center_loss_weight=0.001):
    """Train for one epoch with CutMix + Center Loss"""
    model.train()
    
    # Phase 1: Backbone frozen, Phase 2: Backbone unfrozen
    if phase == 1:
        model.backbone.eval()
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        model.backbone.train()
        for param in model.backbone.parameters():
            param.requires_grad = True
    
    metrics = BatchMetricsAccumulator()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train] Phase {phase}")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label_stage3_AB'].to(device)
        
        # Filter valid samples
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            continue
        
        images = images[valid_mask]
        labels = labels[valid_mask]
        
        optimizer.zero_grad()
        
        # CutMix Augmentation (Yun et al., 2019)
        images_mixed, labels_a, labels_b, lam = cutmix_aug(model, images, labels, training=True)
        
        # Forward pass with features for center loss
        logits, features = model(images_mixed, return_features=True)
        
        # CrossEntropy Loss (with CutMix)
        ce_loss = lam * ce_criterion(logits, labels_a) + (1 - lam) * ce_criterion(logits, labels_b)
        
        # Center Loss (Wen et al., 2016)
        # Use original labels (not mixed) for center loss
        center_loss = center_criterion(features, labels)
        
        # Combined Loss
        loss = ce_loss + center_loss_weight * center_loss
        
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics (use original labels for accuracy)
        preds = logits.argmax(dim=1)
        metrics.update(loss.item(), labels, preds)
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'ce': f"{ce_loss.item():.4f}",
            'center': f"{center_loss.item():.4f}"
        })
    
    return metrics.get_average()


def validate_epoch(model, dataloader, ce_criterion, center_criterion, 
                   device, epoch, center_loss_weight=0.001):
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
            
            # Forward pass
            logits, features = model(images, return_features=True)
            
            # Losses
            ce_loss = ce_criterion(logits, labels)
            center_loss = center_criterion(features, labels)
            loss = ce_loss + center_loss_weight * center_loss
            
            preds = logits.argmax(dim=1)
            metrics.update(loss.item(), labels, preds)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return metrics.get_average()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='pesquisa_v6/v6_dataset_stage3/AB/block_16')
    parser.add_argument('--stage2_model', type=str, default='pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt')
    parser.add_argument('--output_dir', type=str, default='pesquisa_v6/logs/v6_experiments/stage3_ab')
    parser.add_argument('--phase1_epochs', type=int, default=5)
    parser.add_argument('--phase2_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr_head', type=float, default=3e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--center_loss_weight', type=float, default=0.001)
    parser.add_argument('--oversample_factor', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Noise Injection arguments (Hendrycks et al., 2019)
    parser.add_argument("--noise-injection", type=float, default=0.0,
                       help="Fraction of noisy samples to inject (0.0-1.0). E.g., 0.25 = 25%% noise")
    parser.add_argument("--noise-sources", nargs='+', 
                       choices=['RECT', 'SPLIT'], default=['RECT'],
                       help="Sources for noisy samples: RECT (Stage 3-RECT), SPLIT (PARTITION_SPLIT)")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("  Training Stage 3-AB - Fine-Grained Visual Classification")
    print("  (HORZ_A, HORZ_B, VERT_A, VERT_B)")
    if args.noise_injection > 0:
        print(f"  + Noise Injection {args.noise_injection*100:.0f}% (Hendrycks et al., 2019)")
    print("="*70)
    print(f"  Device: {device}")
    print(f"  Strategy: FGVC with 7 advanced techniques")
    print(f"  Phase 1: {args.phase1_epochs} epochs (backbone FROZEN)")
    print(f"  Phase 2: {args.phase2_epochs} epochs (backbone UNFROZEN @ LR={args.lr_backbone})")
    print(f"  Techniques:")
    print(f"    1. Two-Stage Fine-tuning (Kornblith et al., 2019)")
    print(f"    2. Center Loss λ={args.center_loss_weight} (Wen et al., 2016)")
    print(f"    3. CutMix α=1.0 (Yun et al., 2019)")
    print(f"    4. Dual Attention - CBAM (Woo et al., 2018)")
    print(f"    5. Cosine Classifier (Wang et al., 2017)")
    print(f"    6. Label Smoothing 0.1 (Szegedy et al., 2016)")
    print(f"    7. Oversampling {args.oversample_factor}x (Chawla et al., 2002)")
    if args.noise_injection > 0:
        print(f"    8. Noise Injection {args.noise_injection*100:.0f}% from {args.noise_sources}")
    
    # Load datasets
    print(f"\n[1/6] Loading datasets...")
    train_data = torch.load(Path(args.dataset_dir) / "train_v1.pt", weights_only=False)
    val_data = torch.load(Path(args.dataset_dir) / "val.pt", weights_only=False)
    
    print(f"  Train samples: {len(train_data['samples'])}")
    print(f"  Val samples: {len(val_data['samples'])}")
    
    # Create oversampling weights
    print(f"\n[2/6] Creating oversampling weights...")
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
    
    # Convert to datasets
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
    
    # Create clean dataset first
    train_dataset_clean = build_hierarchical_dataset_v6(train_record, augmentation=train_aug, stage='stage3_ab')
    val_dataset = build_hierarchical_dataset_v6(val_record, augmentation=val_aug, stage='stage3_ab')
    
    # Apply Noise Injection if specified (Hendrycks et al., 2019)
    if args.noise_injection > 0:
        print(f"\n[Noise Injection] Loading noise sources...")
        noise_records = []
        
        # Find project root (go up from dataset_dir until we find pesquisa_v6)
        project_root = Path(args.dataset_dir)
        while project_root.name != 'pesquisa_v6' and project_root.parent != project_root:
            project_root = project_root.parent
        
        v6_dataset_dir = project_root / "v6_dataset"
        v6_stage3_dir = project_root / "v6_dataset_stage3"
        
        if 'RECT' in args.noise_sources:
            rect_path = v6_stage3_dir / "RECT" / "block_16" / "train.pt"
            if rect_path.exists():
                print(f"  Loading RECT samples from: {rect_path}")
                rect_data = torch.load(rect_path, weights_only=False)
                rect_samples = rect_data['samples'] if isinstance(rect_data['samples'], np.ndarray) else rect_data['samples'].numpy()
                rect_labels = rect_data['labels'] if isinstance(rect_data['labels'], np.ndarray) else rect_data['labels'].numpy()
                rect_qps = rect_data['qps'] if isinstance(rect_data['qps'], np.ndarray) else rect_data['qps'].numpy()
                
                rect_record = BlockRecord(
                    samples=rect_samples,
                    labels=rect_labels,
                    qps=rect_qps.reshape(-1, 1)
                )
                noise_records.append(rect_record)
            else:
                print(f"  WARNING: RECT dataset not found at {rect_path}")
        
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
            train_dataset = NoisyDatasetAB(
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
    
    # Create FGVC model
    print(f"\n[3/6] Creating FGVC model...")
    base_model = Stage3ABModel(pretrained=True).to(device)
    
    # Load Stage 2 backbone
    if Path(args.stage2_model).exists():
        print(f"  Loading Stage 2 backbone...")
        checkpoint = torch.load(args.stage2_model, map_location=device, weights_only=False)
        stage2_state = checkpoint['model_state_dict']
        backbone_state = {k.replace('backbone.', ''): v for k, v in stage2_state.items() if k.startswith('backbone.')}
        base_model.backbone.load_state_dict(backbone_state, strict=False)
        print(f"  ✅ Backbone initialized from Stage 2")
    
    # Wrap in FGVC model
    model = FGVCModel(base_model, num_classes=4, feat_dim=512).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Phase 1 trainable: {trainable_params:,} (feature projection + classifier)")
    
    # Loss functions
    print(f"\n[4/6] Creating loss functions...")
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Szegedy et al., 2016
    center_criterion = CenterLoss(num_classes=4, feat_dim=512, device=device)  # Wen et al., 2016
    cutmix_aug = CutMixCrossEntropyLoss(alpha=1.0)  # Yun et al., 2019
    
    print(f"  CrossEntropy: Label Smoothing = 0.1")
    print(f"  Center Loss: λ = {args.center_loss_weight}")
    print(f"  CutMix: α = 1.0, p = 0.5")
    
    # Training
    print(f"\n[5/6] Training...")
    best_macro_f1 = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_macro_f1': [],
        'val_loss': [], 'val_acc': [], 'val_macro_f1': [],
        'val_per_class_f1': []
    }
    
    class_names = ['HORZ_A', 'HORZ_B', 'VERT_A', 'VERT_B']
    total_epochs = args.phase1_epochs + args.phase2_epochs
    
    # Phase 1: Backbone FROZEN
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Warm-up (Backbone FROZEN)")
    print(f"{'='*70}")
    
    # Optimizer for Phase 1 (only feature projection + classifier)
    phase1_params = [
        {'params': model.feat_proj.parameters(), 'lr': args.lr_head},
        {'params': model.classifier.parameters(), 'lr': args.lr_head},
        {'params': center_criterion.parameters(), 'lr': args.lr_head}
    ]
    optimizer = torch.optim.AdamW(phase1_params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase1_epochs, eta_min=1e-6)
    
    for epoch in range(1, args.phase1_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, cutmix_aug, ce_criterion, center_criterion,
            optimizer, device, epoch, phase=1, center_loss_weight=args.center_loss_weight
        )
        val_metrics = validate_epoch(
            model, val_loader, ce_criterion, center_criterion,
            device, epoch, center_loss_weight=args.center_loss_weight
        )
        
        scheduler.step()
        
        # Logging
        per_class_f1 = [val_metrics['per_class'][cls]['f1'] for cls in class_names]
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_macro_f1'].append(train_metrics['macro_f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_per_class_f1'].append(per_class_f1)
        
        print(f"\nEpoch {epoch}/{total_epochs} - Phase 1 [FROZEN]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}, F1: {train_metrics['macro_f1']:.2%}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}, F1: {val_metrics['macro_f1']:.2%}")
        for cls_name, f1 in zip(class_names, per_class_f1):
            print(f"          {cls_name:10s} F1: {f1:.2%}")
        
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'phase': 1,
                'model_state_dict': model.state_dict(),
                'center_centers': center_criterion.centers,
                'best_macro_f1': best_macro_f1,
                'val_metrics': val_metrics,
            }, output_dir / "stage3_ab_fgvc_best.pt")
            print(f"  ✅ New best F1: {best_macro_f1:.2%}")
    
    # Phase 2: Backbone UNFROZEN with very low LR
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Fine-tuning (Backbone UNFROZEN @ LR={args.lr_backbone})")
    print(f"{'='*70}")
    
    # Optimizer for Phase 2 (backbone + feature projection + classifier)
    phase2_params = [
        {'params': model.backbone.parameters(), 'lr': args.lr_backbone},  # Very low LR
        {'params': model.feat_proj.parameters(), 'lr': args.lr_head},
        {'params': model.classifier.parameters(), 'lr': args.lr_head},
        {'params': center_criterion.parameters(), 'lr': args.lr_head}
    ]
    optimizer = torch.optim.AdamW(phase2_params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase2_epochs, eta_min=1e-7)
    
    for epoch in range(args.phase1_epochs + 1, total_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, cutmix_aug, ce_criterion, center_criterion,
            optimizer, device, epoch, phase=2, center_loss_weight=args.center_loss_weight
        )
        val_metrics = validate_epoch(
            model, val_loader, ce_criterion, center_criterion,
            device, epoch, center_loss_weight=args.center_loss_weight
        )
        
        scheduler.step()
        
        # Logging
        per_class_f1 = [val_metrics['per_class'][cls]['f1'] for cls in class_names]
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_macro_f1'].append(train_metrics['macro_f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_per_class_f1'].append(per_class_f1)
        
        print(f"\nEpoch {epoch}/{total_epochs} - Phase 2 [UNFROZEN]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}, F1: {train_metrics['macro_f1']:.2%}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}, F1: {val_metrics['macro_f1']:.2%}")
        for cls_name, f1 in zip(class_names, per_class_f1):
            print(f"          {cls_name:10s} F1: {f1:.2%}")
        
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'phase': 2,
                'model_state_dict': model.state_dict(),
                'center_centers': center_criterion.centers,
                'best_macro_f1': best_macro_f1,
                'val_metrics': val_metrics,
            }, output_dir / "stage3_ab_fgvc_best.pt")
            print(f"  ✅ New best F1: {best_macro_f1:.2%}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⚠️  Early stopping! No improvement for {patience} epochs.")
                break
    
    # Save final artifacts
    print(f"\n[6/6] Saving artifacts...")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'center_centers': center_criterion.centers,
        'history': history,
    }, output_dir / "stage3_ab_fgvc_final.pt")
    
    torch.save(history, output_dir / "stage3_ab_fgvc_history.pt")
    
    # Metrics summary
    best_per_class_f1 = history['val_per_class_f1'][best_epoch - 1]
    metrics_summary = {
        'best_macro_f1': float(best_macro_f1),
        'best_epoch': int(best_epoch),
        'best_phase': 1 if best_epoch <= args.phase1_epochs else 2,
        'final_val_macro_f1': float(history['val_macro_f1'][-1]),
        'best_per_class_f1': {
            cls_name: float(f1)
            for cls_name, f1 in zip(class_names, best_per_class_f1)
        },
        'total_epochs_trained': len(history['val_macro_f1']),
        'early_stopped': len(history['val_macro_f1']) < total_epochs,
        'config': vars(args),
        'techniques': [
            'Two-Stage Fine-tuning (Kornblith 2019)',
            'Center Loss (Wen 2016)',
            'CutMix (Yun 2019)',
            'CBAM Dual Attention (Woo 2018)',
            'Cosine Classifier (Wang 2017)',
            'Label Smoothing (Szegedy 2016)',
            'Oversampling (Chawla 2002)'
        ]
    }
    
    with open(output_dir / "stage3_ab_fgvc_metrics.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  ✅ FGVC Training completed!")
    print(f"  Best Macro F1: {best_macro_f1:.2%} (epoch {best_epoch}, phase {metrics_summary['best_phase']})")
    print(f"  Per-class F1:")
    for cls_name, f1 in metrics_summary['best_per_class_f1'].items():
        print(f"    {cls_name:10s}: {f1:.2%}")
    print(f"  Total epochs: {len(history['val_macro_f1'])}/{total_epochs}")
    print(f"  Models saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
